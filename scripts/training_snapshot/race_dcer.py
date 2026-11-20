from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from dcer_model import region_aware_chunk_pool


class RaceDCER(nn.Module):
    """Read the four RACE regions and correct the binary logit margin."""

    def __init__(self, backbone, dim=256, heads=4, chunks=4, alpha=0.01, semantic=False):
        super().__init__()
        self.backbone = backbone
        self.chunks = chunks
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
        self.project = nn.Linear(backbone.config.hidden_size, dim)
        self.region_embedding = nn.Embedding(4, dim)
        self.chunk_embedding = nn.Embedding(chunks, dim)
        self.norm = nn.LayerNorm(dim)
        self.queries = nn.Parameter(torch.empty(1, 2, dim))
        nn.init.normal_(self.queries, std=0.02)
        self.null_evidence = nn.Parameter(torch.zeros(1, 1, dim))
        self.read = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.phishing_head = nn.Linear(dim, 1)
        self.benign_head = nn.Linear(dim, 1)
        for head in (self.phishing_head, self.benign_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        self.semantic = None
        if semantic:
            from semantic_branch import SemanticBranch

            self.semantic = SemanticBranch(backbone.config.hidden_size, dim, heads, chunks)

    def forward(self, input_ids, attention_mask, region_ids):
        model = self.backbone.get_base_model()
        hidden = model.model(input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state
        # Match the original Qwen sequence classifier's pad-token rule.
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        last = (positions * input_ids.ne(model.config.pad_token_id)).argmax(dim=-1)
        base_logits = model.score(hidden)[torch.arange(input_ids.shape[0], device=input_ids.device), last].float()

        pooled = region_aware_chunk_pool(
            hidden, attention_mask, region_ids, chunks_per_region=self.chunks
        )
        region_slots = torch.arange(4, device=hidden.device).repeat_interleave(self.chunks)
        chunk_slots = torch.arange(self.chunks, device=hidden.device).repeat(4)
        evidence = self.norm(
            self.project(pooled.tokens)
            + self.region_embedding(region_slots)
            + self.chunk_embedding(chunk_slots)
        )
        # Use the null key only when all four evidence regions are empty.
        keys = torch.cat((self.null_evidence.expand(hidden.shape[0], -1, -1), evidence), dim=1)
        key_mask = torch.cat((pooled.valid.any(dim=1, keepdim=True), ~pooled.valid), dim=1)
        states, _ = self.read(
            self.queries.expand(hidden.shape[0], -1, -1),
            keys, keys, key_padding_mask=key_mask, need_weights=False,
        )
        s_phishing = self.phishing_head(states[:, 0]).squeeze(-1)
        s_benign = self.benign_head(states[:, 1]).squeeze(-1)
        correction = self.alpha * (s_phishing - s_benign)
        semantic_correction, semantic_details = None, None
        total_correction = correction
        if self.semantic is not None:
            semantic_correction, semantic_details = self.semantic(pooled)
            total_correction = correction + semantic_correction
        logits = base_logits + torch.stack((-0.5 * total_correction, 0.5 * total_correction), dim=1)
        return SimpleNamespace(
            logits=logits, base_logits=base_logits, correction=correction,
            semantic_correction=semantic_correction, semantic_details=semantic_details,
        )

    def save_pretrained(self, directory):
        self.backbone.save_pretrained(directory)
        state = {
            key: value.detach().cpu()
            for key, value in self.state_dict().items()
            if not key.startswith("backbone.")
        }
        torch.save(state, Path(directory, "dcer.pt.tmp"))
        # Publish the head only after the checkpoint write completes.
        Path(directory, "dcer.pt.tmp").replace(Path(directory, "dcer.pt"))
