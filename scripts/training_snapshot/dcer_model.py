"""Qwen3 + region-aware Dual-Class Evidence Reader.

Training keeps the full Qwen hidden-state graph connected. The backbone is
called once and only its final token sequence is returned.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class PooledEvidence:
    tokens: torch.Tensor  # [B, regions * chunks, hidden]
    valid: torch.Tensor  # [B, regions * chunks], True for non-empty chunks


def region_aware_chunk_pool(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    region_ids: torch.Tensor,
    *,
    regions: int = 4,
    chunks_per_region: int = 4,
) -> PooledEvidence:
    """Split each sample by region and mean-pool contiguous token chunks.

    Padding and non-region marker tokens are excluded. Pooling is independent
    per batch row, and the indexed sums remain differentiable with respect to
    ``hidden``.
    """
    if hidden.ndim != 3 or attention_mask.shape != hidden.shape[:2]:
        raise ValueError("Expected hidden [B,N,D] and matching attention mask")
    if region_ids.shape != attention_mask.shape:
        raise ValueError("region_ids must match the token and attention shapes")
    if chunks_per_region < 1 or regions < 1:
        raise ValueError("regions and chunks_per_region must be positive")

    batch, _, dim = hidden.shape
    device = hidden.device
    # Pool in fp32 for stable accumulation; this cast does not detach gradients.
    hidden32 = hidden.float()
    pooled = hidden32.new_zeros((batch, regions * chunks_per_region, dim))
    valid = torch.zeros(
        (batch, regions * chunks_per_region), dtype=torch.bool, device=device
    )

    for b in range(batch):
        for region in range(regions):
            positions = torch.nonzero(
                attention_mask[b].bool() & region_ids[b].eq(region),
                as_tuple=False,
            ).flatten()
            count = int(positions.numel())
            if count == 0:
                continue
            region_hidden = hidden32[b].index_select(0, positions)
            chunk_ids = torch.div(
                torch.arange(count, device=device) * chunks_per_region,
                count,
                rounding_mode="floor",
            ).clamp_max(chunks_per_region - 1)
            sums = hidden32.new_zeros((chunks_per_region, dim)).index_add(
                0, chunk_ids, region_hidden
            )
            counts = torch.bincount(chunk_ids, minlength=chunks_per_region)
            nonempty = counts.gt(0)
            start = region * chunks_per_region
            pooled[b, start : start + chunks_per_region] = (
                sums / counts.clamp_min(1).unsqueeze(-1)
            )
            valid[b, start : start + chunks_per_region] = nonempty

    return PooledEvidence(tokens=pooled, valid=valid)


class LinearBinaryHead(nn.Module):
    """The Qwen last-valid-token + linear binary-classifier baseline."""

    def __init__(self, hidden_size: int = 1024):
        super().__init__()
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, last_hidden: torch.Tensor, attention_mask: torch.Tensor):
        lengths = attention_mask.long().sum(dim=1)
        if torch.any(lengths == 0):
            raise ValueError("Every sample needs at least one valid token")
        # The supported preprocessing right-pads sequences; deriving the final
        # index from the mask also checks that padding is excluded correctly.
        positions = torch.arange(last_hidden.shape[1], device=last_hidden.device)
        last_positions = positions.unsqueeze(0).expand_as(attention_mask).masked_fill(
            ~attention_mask.bool(), -1
        ).amax(dim=1)
        if not torch.equal(last_positions, lengths - 1):
            raise ValueError("Input must be right padded without interior holes")
        global_feature = last_hidden[
            torch.arange(last_hidden.shape[0], device=last_hidden.device), last_positions
        ].float()
        logits = self.classifier(global_feature)
        z0 = logits[:, 1] - logits[:, 0]
        return {"logits": logits, "z0": z0, "z": z0}


class DualClassEvidenceReader(nn.Module):
    """Four-region chunk pooling followed by shared dual-query cross-attention."""

    def __init__(
        self,
        hidden_size: int = 1024,
        reader_dim: int = 256,
        num_heads: int = 4,
        regions: int = 4,
        chunks_per_region: int = 4,
        alpha: float = 0.1,
        alpha_mode: Literal["learnable", "fixed"] = "learnable",
        dropout: float = 0.1,
    ):
        super().__init__()
        if reader_dim % num_heads:
            raise ValueError("reader_dim must be divisible by num_heads")
        self.regions = regions
        self.chunks_per_region = chunks_per_region
        self.projection = nn.Linear(hidden_size, reader_dim)
        self.region_embedding = nn.Embedding(regions, reader_dim)
        self.chunk_embedding = nn.Embedding(chunks_per_region, reader_dim)
        self.evidence_norm = nn.LayerNorm(reader_dim)
        self.null_evidence = nn.Parameter(torch.empty(1, 1, reader_dim))
        nn.init.normal_(self.null_evidence, mean=0.0, std=0.02)

        # A distinct, learned class query for each outcome. The same attention
        # parameters read both queries so scores inhabit a comparable space.
        self.phishing_query = nn.Parameter(torch.empty(1, 1, reader_dim))
        self.benign_query = nn.Parameter(torch.empty(1, 1, reader_dim))
        nn.init.normal_(self.phishing_query, mean=0.0, std=0.02)
        nn.init.normal_(self.benign_query, mean=0.0, std=0.02)
        self.cross_attention = nn.MultiheadAttention(
            reader_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.query_norm = nn.LayerNorm(reader_dim)
        self.scorers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(reader_dim),
                    nn.Linear(reader_dim, reader_dim // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(reader_dim // 2, 1),
                )
                for _ in range(2)
            ]
        )
        # Small but nonzero output weights keep the residual near zero while
        # allowing gradients to reach both class queries from the first step.
        for scorer in self.scorers:
            nn.init.normal_(scorer[-1].weight, mean=0.0, std=1e-3)
            nn.init.zeros_(scorer[-1].bias)

        if alpha_mode == "learnable":
            self.alpha = nn.Parameter(torch.tensor(float(alpha), dtype=torch.float32))
        elif alpha_mode == "fixed":
            self.register_buffer("alpha", torch.tensor(float(alpha), dtype=torch.float32))
        else:
            raise ValueError(f"Unknown alpha_mode: {alpha_mode}")
        self.alpha_mode = alpha_mode

    def forward(
        self,
        last_hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        region_ids: torch.Tensor,
        classifier: nn.Module,
    ) -> dict[str, torch.Tensor]:
        pooled = region_aware_chunk_pool(
            last_hidden,
            attention_mask,
            region_ids,
            regions=self.regions,
            chunks_per_region=self.chunks_per_region,
        )
        batch_size = last_hidden.shape[0]
        evidence = self.projection(pooled.tokens)
        region_ids_e = torch.arange(self.regions, device=last_hidden.device).repeat_interleave(
            self.chunks_per_region
        )
        chunk_ids_e = torch.arange(self.chunks_per_region, device=last_hidden.device).repeat(
            self.regions
        )
        evidence = self.evidence_norm(
            evidence
            + self.region_embedding(region_ids_e).unsqueeze(0)
            + self.chunk_embedding(chunk_ids_e).unsqueeze(0)
        )

        # Always keep one learned null key unmasked. This prevents undefined
        # softmax rows for pages whose structured evidence is entirely absent.
        null = self.null_evidence.expand(batch_size, -1, -1)
        keys = torch.cat((null, evidence), dim=1)
        key_padding_mask = torch.cat(
            (
                torch.zeros((batch_size, 1), dtype=torch.bool, device=last_hidden.device),
                ~pooled.valid,
            ),
            dim=1,
        )
        queries = torch.cat(
            (
                self.phishing_query.expand(batch_size, -1, -1),
                self.benign_query.expand(batch_size, -1, -1),
            ),
            dim=1,
        )
        read, attention_weights = self.cross_attention(
            query=queries,
            key=keys,
            value=keys,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        read = self.query_norm(queries + read)
        phishing_state, benign_state = read[:, 0], read[:, 1]
        s_phishing = self.scorers[0](phishing_state).squeeze(-1)
        s_benign = self.scorers[1](benign_state).squeeze(-1)

        lengths = attention_mask.long().sum(dim=1)
        last_positions = torch.arange(last_hidden.shape[1], device=last_hidden.device)
        last_positions = last_positions.unsqueeze(0).expand_as(attention_mask).masked_fill(
            ~attention_mask.bool(), -1
        ).amax(dim=1)
        if torch.any(lengths == 0) or not torch.equal(last_positions, lengths - 1):
            raise ValueError("Input must be non-empty and right padded")
        global_feature = last_hidden[
            torch.arange(batch_size, device=last_hidden.device), last_positions
        ].float()
        base_logits = classifier(global_feature)
        z0 = base_logits[:, 1] - base_logits[:, 0]
        correction = s_phishing - s_benign
        z = z0 + self.alpha * correction
        return {
            "logits": base_logits,
            "z0": z0,
            "z": z,
            "prob_phishing": torch.sigmoid(z),
            "s_phishing": s_phishing,
            "s_benign": s_benign,
            "correction": correction,
            "alpha": self.alpha,
            "evidence_tokens": evidence,
            "evidence_valid": pooled.valid,
            "phishing_state": phishing_state,
            "benign_state": benign_state,
        }


class DCERClassifier(nn.Module):
    """Base linear classifier plus a residual Dual-Class Evidence Reader."""

    def __init__(
        self,
        hidden_size: int = 1024,
        reader_dim: int = 256,
        num_heads: int = 4,
        chunks_per_region: int = 4,
        alpha: float = 0.1,
        alpha_mode: Literal["learnable", "fixed"] = "learnable",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.classifier = nn.Linear(hidden_size, 2)
        self.reader = DualClassEvidenceReader(
            hidden_size=hidden_size,
            reader_dim=reader_dim,
            num_heads=num_heads,
            chunks_per_region=chunks_per_region,
            alpha=alpha,
            alpha_mode=alpha_mode,
            dropout=dropout,
        )

    def forward(self, last_hidden, attention_mask, region_ids):
        return self.reader(last_hidden, attention_mask, region_ids, self.classifier)


class URLHTMLSemanticConsistency(nn.Module):
    """Learn phishing semantics in URL and HTML views, then exchange evidence.

    The module reads the URL region and the three HTML regions as two views.
    Two bidirectional cross-attention blocks allow either view to condition on
    the other. A shared pair of class prototypes makes both view embeddings
    answer the same benign/phishing question. The fused representation emits
    a small residual correction to the existing classifier.
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        reader_dim: int = 256,
        num_heads: int = 4,
        chunks_per_region: int = 4,
        dropout: float = 0.1,
        use_multi_layer: bool = True,
        use_reliability_fusion: bool = True,
        branch_mode: Literal["cross_conditioned", "independent"] = "cross_conditioned",
    ):
        super().__init__()
        if reader_dim % num_heads:
            raise ValueError("reader_dim must be divisible by num_heads")
        self.chunks_per_region = chunks_per_region
        self.use_multi_layer = bool(use_multi_layer)
        self.use_reliability_fusion = bool(use_reliability_fusion)
        if branch_mode not in {"cross_conditioned", "independent"}:
            raise ValueError(f"Unknown semantic branch mode: {branch_mode}")
        self.branch_mode = branch_mode
        self.layer_indices = (7, 14, 21, 28)
        self.projection = nn.Linear(hidden_size, reader_dim)
        self.layer_embedding = nn.Embedding(len(self.layer_indices), reader_dim)
        self.layer_gate = nn.Sequential(
            nn.LayerNorm(reader_dim),
            nn.Linear(reader_dim, reader_dim // 2),
            nn.GELU(),
            nn.Linear(reader_dim // 2, 1),
        )
        nn.init.zeros_(self.layer_gate[-1].weight)
        nn.init.zeros_(self.layer_gate[-1].bias)
        self.region_embedding = nn.Embedding(4, reader_dim)
        self.position_embedding = nn.Embedding(4 * chunks_per_region, reader_dim)
        self.view_embedding = nn.Embedding(2, reader_dim)
        self.input_norm = nn.LayerNorm(reader_dim)
        self.null_url = nn.Parameter(torch.empty(1, 1, reader_dim))
        self.null_html = nn.Parameter(torch.empty(1, 1, reader_dim))
        nn.init.normal_(self.null_url, mean=0.0, std=0.02)
        nn.init.normal_(self.null_html, mean=0.0, std=0.02)

        self.interaction = nn.ModuleList(
            [
                BidirectionalEvidenceInteraction(reader_dim, num_heads, dropout)
                for _ in range(2)
            ]
        )
        self.url_readout = nn.Parameter(torch.empty(1, 1, reader_dim))
        self.html_readout = nn.Parameter(torch.empty(1, 1, reader_dim))
        nn.init.normal_(self.url_readout, mean=0.0, std=0.02)
        nn.init.normal_(self.html_readout, mean=0.0, std=0.02)
        if self.branch_mode == "independent":
            self.url_branch_readout = nn.Parameter(torch.empty(1, 1, reader_dim))
            self.html_branch_readout = nn.Parameter(torch.empty(1, 1, reader_dim))
            nn.init.normal_(self.url_branch_readout, mean=0.0, std=0.02)
            nn.init.normal_(self.html_branch_readout, mean=0.0, std=0.02)
        self.readout_attention = nn.MultiheadAttention(
            reader_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.view_norm = nn.LayerNorm(reader_dim)

        # Both modalities use the same benign/phishing semantic anchors.
        self.class_prototypes = nn.Parameter(torch.empty(2, reader_dim))
        nn.init.normal_(self.class_prototypes, mean=0.0, std=0.02)
        self.logit_scale = nn.Parameter(torch.tensor(2.3025851))  # log(10)

        # Map each independent modality readout into the same task-semantic
        # space while retaining its modality-private branch representation.
        # Identity initialization preserves the previous classifier behavior
        # when loading an older M1+BMCC checkpoint.
        self.url_semantic_projection = nn.Linear(reader_dim, reader_dim)
        self.html_semantic_projection = nn.Linear(reader_dim, reader_dim)
        nn.init.eye_(self.url_semantic_projection.weight)
        nn.init.zeros_(self.url_semantic_projection.bias)
        nn.init.eye_(self.html_semantic_projection.weight)
        nn.init.zeros_(self.html_semantic_projection.bias)

        fused_dim = reader_dim * 4
        self.fusion = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, reader_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(reader_dim * 2, reader_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(reader_dim, 1),
        )
        nn.init.normal_(self.fusion[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.fusion[-1].bias)
        self.reliability_gate = nn.Sequential(
            nn.LayerNorm(fused_dim + 4),
            nn.Linear(fused_dim + 4, reader_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(reader_dim, 2),
        )
        nn.init.zeros_(self.reliability_gate[-1].weight)
        nn.init.zeros_(self.reliability_gate[-1].bias)
        self.branch_residual_scale = nn.Parameter(torch.tensor(0.01, dtype=torch.float32))
        self.alpha = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))
        if not self.use_multi_layer:
            for parameter in self.layer_embedding.parameters():
                parameter.requires_grad_(False)
            for parameter in self.layer_gate.parameters():
                parameter.requires_grad_(False)
        if not self.use_reliability_fusion:
            for parameter in self.reliability_gate.parameters():
                parameter.requires_grad_(False)
            self.branch_residual_scale.requires_grad_(False)

    @staticmethod
    def _ensure_nonempty(sequence, valid, null_token):
        """Keep attention numerically defined when one modality is absent."""
        missing = ~valid.any(dim=1)
        if missing.any():
            sequence = sequence.clone()
            valid = valid.clone()
            sequence[missing, 0] = null_token[0, 0]
            valid[missing, 0] = True
        return sequence, valid

    def _pool_view(self, sequence, valid, query):
        readout, _ = self.readout_attention(
            query=query.expand(sequence.shape[0], -1, -1),
            key=sequence,
            value=sequence,
            key_padding_mask=~valid,
            need_weights=False,
        )
        return self.view_norm(query[:, 0] + readout[:, 0])

    def forward(self, last_hidden, attention_mask, region_ids, layer_hidden_states=None):
        if not self.use_multi_layer:
            selected_hidden = (last_hidden,)
        else:
            if layer_hidden_states is None:
                raise ValueError("Multi-layer semantic mode requires Qwen hidden states")
            if max(self.layer_indices) >= len(layer_hidden_states):
                raise ValueError(
                    f"Need hidden states through layer {max(self.layer_indices)}, "
                    f"received {len(layer_hidden_states)} states"
                )
            selected_hidden = tuple(layer_hidden_states[index] for index in self.layer_indices)

        pooled_layers = [
            region_aware_chunk_pool(
                layer_hidden,
                attention_mask,
                region_ids,
                regions=4,
                chunks_per_region=self.chunks_per_region,
            )
            for layer_hidden in selected_hidden
        ]
        pooled_valid = pooled_layers[0].valid
        if any(not torch.equal(pooled_valid, pooled.valid) for pooled in pooled_layers[1:]):
            raise AssertionError("Layer-wise evidence masks must stay aligned")

        batch_size = last_hidden.shape[0]
        projected_layers = []
        for layer_index, pooled in enumerate(pooled_layers):
            projected = self.projection(pooled.tokens)
            if self.use_multi_layer:
                projected = projected + self.layer_embedding.weight[layer_index].view(1, 1, -1)
            projected_layers.append(projected)
        projected = torch.stack(projected_layers, dim=1)
        region_ids_e = torch.arange(4, device=last_hidden.device).repeat_interleave(
            self.chunks_per_region
        )
        position_ids = torch.arange(4 * self.chunks_per_region, device=last_hidden.device)
        projected = self.input_norm(
            projected
            + self.region_embedding(region_ids_e).view(1, 1, -1, projected.shape[-1])
            + self.position_embedding(position_ids).view(1, 1, -1, projected.shape[-1])
        )
        if self.use_multi_layer:
            layer_scores = self.layer_gate(projected).squeeze(-1)
            valid_for_gate = pooled_valid.clone()
            empty_rows = ~valid_for_gate.any(dim=1)
            if empty_rows.any():
                valid_for_gate[empty_rows, 0] = True
            layer_scores = layer_scores.masked_fill(
                ~valid_for_gate.unsqueeze(1), torch.finfo(layer_scores.dtype).min
            )
            layer_weights = torch.softmax(layer_scores, dim=1)
            projected = (projected * layer_weights.unsqueeze(-1)).sum(dim=1)
        else:
            layer_weights = projected.new_ones((batch_size, 1, projected.shape[2]))
            projected = projected[:, 0]
        projected = projected * pooled_valid.unsqueeze(-1)

        # region 0 is URL; regions 1..3 are Head, Interactive/Form and Body.
        url = projected[:, : self.chunks_per_region]
        url_valid = pooled_valid[:, : self.chunks_per_region]
        html = projected[:, self.chunks_per_region :]
        html_valid = pooled_valid[:, self.chunks_per_region :]
        url_present = url_valid.any(dim=1)
        html_present = html_valid.any(dim=1)
        url_coverage = url_valid.float().mean(dim=1)
        html_coverage = html_valid.float().mean(dim=1)
        url = url + self.view_embedding.weight[0].view(1, 1, -1)
        html = html + self.view_embedding.weight[1].view(1, 1, -1)
        url, url_valid = self._ensure_nonempty(url, url_valid, self.null_url)
        html, html_valid = self._ensure_nonempty(html, html_valid, self.null_html)

        if self.branch_mode == "independent":
            url_branch_state = self._pool_view(url, url_valid, self.url_branch_readout)
            html_branch_state = self._pool_view(html, html_valid, self.html_branch_readout)
        else:
            url_branch_state = None
            html_branch_state = None

        for layer in self.interaction:
            url, html = layer(url, url_valid, html, html_valid)

        url_state = self._pool_view(url, url_valid, self.url_readout)
        html_state = self._pool_view(html, html_valid, self.html_readout)
        prototypes = F.normalize(self.class_prototypes, dim=-1)
        scale = self.logit_scale.exp().clamp(max=100.0)
        if self.branch_mode == "independent":
            url_semantic_input = url_branch_state
            html_semantic_input = html_branch_state
            gate_url_state = url_branch_state
            gate_html_state = html_branch_state
        else:
            url_semantic_input = url_state
            html_semantic_input = html_state
            gate_url_state = url_state
            gate_html_state = html_state

        url_shared_state = self.url_semantic_projection(url_semantic_input)
        html_shared_state = self.html_semantic_projection(html_semantic_input)
        url_crossconditioned_state = self.url_semantic_projection(url_state)
        html_crossconditioned_state = self.html_semantic_projection(html_state)
        url_semantic_logits = scale * F.normalize(url_shared_state, dim=-1) @ prototypes.t()
        html_semantic_logits = scale * F.normalize(html_shared_state, dim=-1) @ prototypes.t()
        url_crossconditioned_logits = (
            scale * F.normalize(url_crossconditioned_state, dim=-1) @ prototypes.t()
        )
        html_crossconditioned_logits = (
            scale * F.normalize(html_crossconditioned_state, dim=-1) @ prototypes.t()
        )

        url_prob = torch.softmax(url_semantic_logits.float(), dim=-1)
        html_prob = torch.softmax(html_semantic_logits.float(), dim=-1)
        url_confidence = url_prob.max(dim=-1).values
        html_confidence = html_prob.max(dim=-1).values
        fusion_input = torch.cat(
            (
                gate_url_state,
                gate_html_state,
                torch.abs(gate_url_state - gate_html_state),
                gate_url_state * gate_html_state,
                url_confidence.unsqueeze(-1),
                html_confidence.unsqueeze(-1),
                url_coverage.unsqueeze(-1),
                html_coverage.unsqueeze(-1),
            ),
            dim=-1,
        )
        present = torch.stack((url_present, html_present), dim=-1)
        no_view = ~present.any(dim=-1)
        if no_view.any():
            present[no_view] = True
        if self.use_reliability_fusion:
            gate_logits = self.reliability_gate(fusion_input)
            gate_logits = gate_logits.masked_fill(~present, torch.finfo(gate_logits.dtype).min)
            learned_reliability_weights = torch.softmax(gate_logits, dim=-1)
        else:
            learned_reliability_weights = present.to(url_state.dtype)
            learned_reliability_weights = learned_reliability_weights / learned_reliability_weights.sum(
                dim=-1, keepdim=True
            ).clamp_min(1.0)

        equal_reliability_weights = present.to(url_state.dtype)
        equal_reliability_weights = equal_reliability_weights / equal_reliability_weights.sum(
            dim=-1, keepdim=True
        ).clamp_min(1.0)

        def fuse_views(weights):
            cross_reliability = torch.sqrt((weights[:, 0] * weights[:, 1]).clamp_min(1e-8))
            fused = torch.cat(
                (
                    weights[:, 0:1] * url_state,
                    weights[:, 1:2] * html_state,
                    cross_reliability[:, None] * torch.abs(url_state - html_state),
                    (weights[:, 0] * weights[:, 1])[:, None] * (url_state * html_state),
                ),
                dim=-1,
            )
            fusion_correction = self.fusion(fused).squeeze(-1)
            branch_margin = (
                weights[:, 0] * (url_semantic_logits[:, 1] - url_semantic_logits[:, 0])
                + weights[:, 1] * (html_semantic_logits[:, 1] - html_semantic_logits[:, 0])
            )
            branch_residual = (
                self.branch_residual_scale * branch_margin
                if self.use_reliability_fusion
                else torch.zeros_like(branch_margin)
            )
            return fusion_correction + branch_residual, fusion_correction, branch_margin, branch_residual

        correction, fusion_correction, branch_margin, branch_residual = fuse_views(
            learned_reliability_weights
        )
        if self.use_reliability_fusion and not self.training:
            correction_equal_weight, _, equal_branch_margin, equal_branch_residual = fuse_views(
                equal_reliability_weights
            )
        else:
            correction_equal_weight = correction
            equal_branch_margin = branch_margin
            equal_branch_residual = branch_residual
        return {
            "semantic_correction": correction,
            "semantic_correction_equal_weight": correction_equal_weight,
            "semantic_fusion_correction": fusion_correction,
            "semantic_branch_margin": branch_margin,
            "semantic_branch_residual": branch_residual,
            "semantic_branch_margin_equal_weight": equal_branch_margin,
            "semantic_branch_residual_equal_weight": equal_branch_residual,
            "semantic_alpha": self.alpha,
            "url_semantic_logits": url_semantic_logits,
            "html_semantic_logits": html_semantic_logits,
            "url_crossconditioned_semantic_logits": url_crossconditioned_logits,
            "html_crossconditioned_semantic_logits": html_crossconditioned_logits,
            "url_present": url_present,
            "html_present": html_present,
            "url_coverage": url_coverage,
            "html_coverage": html_coverage,
            "view_reliability_weights": learned_reliability_weights,
            "equal_view_reliability_weights": equal_reliability_weights,
            "layer_weights": layer_weights.mean(dim=-1),
            "url_state": url_state,
            "html_state": html_state,
            "url_branch_state": url_branch_state if url_branch_state is not None else url_state,
            "html_branch_state": html_branch_state if html_branch_state is not None else html_state,
            "url_shared_state": url_shared_state,
            "html_shared_state": html_shared_state,
            "url_crossconditioned_shared_state": url_crossconditioned_state,
            "html_crossconditioned_shared_state": html_crossconditioned_state,
            "url_valid": url_valid,
            "html_valid": html_valid,
            # Preserve the pre-co-attention region chunks for typed mechanism
            # modeling (URL source, page claim, and interactive/form action).
            "typed_evidence_tokens": projected,
            "typed_evidence_valid": pooled_valid,
        }


class BidirectionalEvidenceInteraction(nn.Module):
    """One parallel co-attention + feed-forward update for both views."""

    def __init__(self, dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.url_reads_html = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.html_reads_url = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.url_cross_norm = nn.LayerNorm(dim)
        self.html_cross_norm = nn.LayerNorm(dim)
        self.url_ffn_norm = nn.LayerNorm(dim)
        self.html_ffn_norm = nn.LayerNorm(dim)
        self.url_ffn = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 2, dim)
        )
        self.html_ffn = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 2, dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, url, url_valid, html, html_valid):
        # Both directions read the other view's pre-update states (parallel co-attention).
        url_read, _ = self.url_reads_html(
            url, html, html, key_padding_mask=~html_valid, need_weights=False
        )
        html_read, _ = self.html_reads_url(
            html, url, url, key_padding_mask=~url_valid, need_weights=False
        )
        url = self.url_cross_norm(url + self.dropout(url_read))
        html = self.html_cross_norm(html + self.dropout(html_read))
        url = self.url_ffn_norm(url + self.dropout(self.url_ffn(url)))
        html = self.html_ffn_norm(html + self.dropout(self.html_ffn(html)))
        return url, html


class URLIdentityHTMLClaimActionRelation(nn.Module):
    """Model source identity vs page claims and form behavior.

    The inputs are the four canonical evidence regions: URL, HTML head,
    interactive/form, and HTML body. Head and body jointly represent the
    identity/content claimed by the page; interactive evidence represents
    requested information and form/interaction destinations. Pairwise typed
    edges encode comparisons without treating any single mismatch (such as an
    off-domain form endpoint) as deterministically malicious.
    """

    def __init__(self, dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.role_queries = nn.Parameter(torch.empty(4, dim))
        self.null_roles = nn.Parameter(torch.empty(4, dim))
        nn.init.normal_(self.role_queries, mean=0.0, std=0.02)
        nn.init.normal_(self.null_roles, mean=0.0, std=0.02)
        self.key_projection = nn.Linear(dim, dim, bias=False)
        self.value_projection = nn.Linear(dim, dim, bias=False)
        self.claim_gate = nn.Sequential(
            nn.LayerNorm(dim * 4 + 2),
            nn.Linear(dim * 4 + 2, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 2),
        )

        # URL↔claim, URL↔action, and claim↔action, each in both directions.
        self.edge_types = nn.Embedding(6, dim)
        nn.init.normal_(self.edge_types.weight, mean=0.0, std=0.02)
        self.edge_encoder = nn.Sequential(
            nn.LayerNorm(dim * 5),
            nn.Linear(dim * 5, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
        )
        self.edge_score = nn.Linear(dim, 1)
        self.null_edge = nn.Parameter(torch.empty(1, 1, dim))
        self.null_edge_score = nn.Parameter(torch.zeros(1))
        nn.init.normal_(self.null_edge, mean=0.0, std=0.02)

        # Node states + relation edges + pairwise difference features + masks.
        relation_input_dim = dim * 7 + 3
        self.classifier = nn.Sequential(
            nn.LayerNorm(relation_input_dim),
            nn.Linear(relation_input_dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 1),
        )
        nn.init.normal_(self.classifier[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.classifier[-1].bias)
        self.alpha = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def _read_region(self, tokens, valid, region_index):
        start = region_index * (tokens.shape[1] // 4)
        end = start + (tokens.shape[1] // 4)
        region_tokens = tokens[:, start:end]
        region_valid = valid[:, start:end].bool()
        present = region_valid.any(dim=1)
        empty = ~present
        if empty.any():
            region_tokens = region_tokens.clone()
            region_valid = region_valid.clone()
            region_tokens[empty, 0] = self.null_roles[region_index]
            region_valid[empty, 0] = True
        keys = self.key_projection(region_tokens)
        values = self.value_projection(region_tokens)
        scores = torch.einsum("bnd,d->bn", keys, self.role_queries[region_index])
        scores = scores / (self.dim ** 0.5)
        scores = scores.masked_fill(~region_valid, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores.float(), dim=-1).to(values.dtype)
        pooled = torch.sum(weights.unsqueeze(-1) * values, dim=1)
        return pooled, present

    def forward(self, tokens, valid):
        if tokens.ndim != 3 or valid.shape != tokens.shape[:2] or tokens.shape[1] % 4:
            raise ValueError("Expected [B, 4*chunks, D] typed evidence and matching mask")

        # Canonical region ids: URL=0, Head=1, Interactive/Form=2, Body=3.
        url, url_present = self._read_region(tokens, valid, 0)
        head, head_present = self._read_region(tokens, valid, 1)
        action, action_present = self._read_region(tokens, valid, 2)
        body, body_present = self._read_region(tokens, valid, 3)

        claim_present = head_present | body_present
        claim_presence = torch.stack((head_present, body_present), dim=-1)
        claim_input = torch.cat(
            (
                head,
                body,
                torch.abs(head - body),
                head * body,
                head_present.unsqueeze(-1).to(head.dtype),
                body_present.unsqueeze(-1).to(body.dtype),
            ),
            dim=-1,
        )
        claim_scores = self.claim_gate(claim_input)
        safe_claim_presence = claim_presence.clone()
        no_claim = ~claim_present
        if no_claim.any():
            safe_claim_presence[no_claim, 0] = True
        claim_scores = claim_scores.masked_fill(
            ~safe_claim_presence, torch.finfo(claim_scores.dtype).min
        )
        claim_weights = torch.softmax(claim_scores.float(), dim=-1).to(head.dtype)
        claim = claim_weights[:, :1] * head + claim_weights[:, 1:] * body

        nodes = (url, claim, action)
        node_present = (url_present, claim_present, action_present)
        edge_specs = ((0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1))
        edge_features = []
        edge_valid = []
        for edge_index, (source_index, target_index) in enumerate(edge_specs):
            source, target = nodes[source_index], nodes[target_index]
            pair = torch.cat(
                (source, target, source - target, torch.abs(source - target), source * target),
                dim=-1,
            )
            edge = self.edge_encoder(pair) + self.edge_types.weight[edge_index]
            edge_features.append(edge)
            edge_valid.append(node_present[source_index] & node_present[target_index])
        edges = torch.stack(edge_features, dim=1)
        edge_valid = torch.stack(edge_valid, dim=1)

        batch = tokens.shape[0]
        null_edge = self.null_edge.expand(batch, -1, -1)
        all_edges = torch.cat((null_edge, edges), dim=1)
        edge_scores = self.edge_score(edges).squeeze(-1)
        edge_scores = edge_scores.masked_fill(~edge_valid, torch.finfo(edge_scores.dtype).min)
        null_score = self.null_edge_score.expand(batch, 1).to(edge_scores.dtype)
        edge_weights = torch.softmax(torch.cat((null_score, edge_scores), dim=1).float(), dim=-1)
        edge_context = torch.sum(edge_weights.unsqueeze(-1).to(all_edges.dtype) * all_edges, dim=1)

        url_claim_diff = torch.abs(url - claim)
        url_action_diff = torch.abs(url - action)
        claim_action_diff = torch.abs(claim - action)
        presence = torch.stack(node_present, dim=-1).to(url.dtype)
        classifier_input = torch.cat(
            (url, claim, action, edge_context, url_claim_diff, url_action_diff,
             claim_action_diff, presence),
            dim=-1,
        )
        relation_logit = self.classifier(classifier_input).squeeze(-1)
        return {
            "identity_action_logit": relation_logit,
            "identity_action_alpha": self.alpha,
            "identity_action_edge_weights": edge_weights,
            "identity_action_presence": presence,
            "identity_action_claim_weights": claim_weights,
        }


class SemanticDCERClassifier(DCERClassifier):
    """DCER plus a full URL–HTML semantic consistency module."""

    def __init__(
        self,
        hidden_size: int = 1024,
        reader_dim: int = 256,
        num_heads: int = 4,
        chunks_per_region: int = 4,
        alpha: float = 0.1,
        alpha_mode: Literal["learnable", "fixed"] = "learnable",
        dropout: float = 0.1,
        use_multi_layer: bool = True,
        use_reliability_fusion: bool = True,
        branch_mode: Literal["cross_conditioned", "independent"] = "cross_conditioned",
        use_identity_action_relation: bool = False,
    ):
        super().__init__(
            hidden_size=hidden_size,
            reader_dim=reader_dim,
            num_heads=num_heads,
            chunks_per_region=chunks_per_region,
            alpha=alpha,
            alpha_mode=alpha_mode,
            dropout=dropout,
        )
        self.semantic = URLHTMLSemanticConsistency(
            hidden_size=hidden_size,
            reader_dim=reader_dim,
            num_heads=num_heads,
            chunks_per_region=chunks_per_region,
            dropout=dropout,
            use_multi_layer=use_multi_layer,
            use_reliability_fusion=use_reliability_fusion,
            branch_mode=branch_mode,
        )
        self.identity_action_relation = (
            URLIdentityHTMLClaimActionRelation(reader_dim, dropout)
            if use_identity_action_relation
            else None
        )

    def forward(self, last_hidden, attention_mask, region_ids, layer_hidden_states=None):
        result = super().forward(last_hidden, attention_mask, region_ids)
        semantic = self.semantic(
            last_hidden, attention_mask, region_ids, layer_hidden_states=layer_hidden_states
        )
        z_before_semantic = result["z"]
        z = z_before_semantic + semantic["semantic_alpha"] * semantic["semantic_correction"]
        z_equal_weight = (
            z_before_semantic
            + semantic["semantic_alpha"] * semantic["semantic_correction_equal_weight"]
        )
        relation = {}
        if self.identity_action_relation is not None:
            relation = self.identity_action_relation(
                semantic["typed_evidence_tokens"], semantic["typed_evidence_valid"]
            )
            z = z + relation["identity_action_alpha"] * relation["identity_action_logit"]
        return {
            **result,
            **semantic,
            "z_before_semantic": z_before_semantic,
            "z_equal_weight": z_equal_weight,
            "z": z,
            "prob_phishing": torch.sigmoid(z),
            **relation,
        }
