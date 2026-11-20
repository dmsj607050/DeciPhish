"""BUEI and BMCC on the shared backbone's pooled RACE evidence."""

import torch
from torch import nn
from torch.nn import functional as F

from dcer_model import BidirectionalEvidenceInteraction


VIEW_CE_WEIGHT = 0.25
VIEW_KL_WEIGHT = 0.2
GATE_KL_WEIGHT = 0.1
RELIABILITY_SIGMA = 1.0


def masked_mean(values, valid):
    return (values * valid).sum() / valid.sum().clamp_min(1)


class SemanticBranch(nn.Module):
    def __init__(self, hidden_size, dim, heads, chunks):
        super().__init__()
        self.chunks = chunks
        self.url_encoder = nn.Sequential(nn.Linear(hidden_size, dim), nn.LayerNorm(dim))
        self.html_encoder = nn.Sequential(nn.Linear(hidden_size, dim), nn.LayerNorm(dim))
        self.interaction = BidirectionalEvidenceInteraction(dim, heads, dropout=0.0)
        self.url_head = nn.Linear(dim, 2)
        self.html_head = nn.Linear(dim, 2)
        self.gate = nn.Linear(3 * dim + 4, 2)
        self.residual_head = nn.Linear(3 * dim, 2)
        for layer in (self.gate, self.residual_head):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        self.alpha = nn.Parameter(torch.tensor(0.01))

    @staticmethod
    def with_null_key(tokens, valid):
        null = tokens.new_zeros(tokens.shape[0], 1, tokens.shape[-1])
        return torch.cat((null, tokens), dim=1), torch.cat((~valid.any(dim=1, keepdim=True), valid), dim=1)

    @staticmethod
    def pool(tokens, valid):
        weights = valid.unsqueeze(-1)
        return (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)

    def forward(self, pooled):
        url_tokens, html_tokens = pooled.tokens.split(
            (self.chunks, 3 * self.chunks), dim=1
        )
        url_valid, html_valid = pooled.valid.split((self.chunks, 3 * self.chunks), dim=1)
        present = torch.stack((url_valid.any(dim=1), html_valid.any(dim=1)), dim=1)
        coverage = torch.stack((url_valid.float().mean(dim=1), html_valid.float().mean(dim=1)), dim=1)
        url, url_mask = self.with_null_key(self.url_encoder(url_tokens), url_valid)
        html, html_mask = self.with_null_key(self.html_encoder(html_tokens), html_valid)
        url, html = self.interaction(url, url_mask, html, html_mask)
        url_state, html_state = self.pool(url[:, 1:], url_valid), self.pool(html[:, 1:], html_valid)
        view_logits = torch.stack((self.url_head(url_state), self.html_head(html_state)), dim=1)
        probs = view_logits.softmax(dim=-1)[..., 1]
        difference = (url_state - html_state).abs()
        gate_input = torch.cat((url_state, html_state, difference, probs, coverage), dim=1)
        gate_logits = self.gate(gate_input)
        safe_present = present | ~present.any(dim=1, keepdim=True)
        weights = gate_logits.masked_fill(~safe_present, torch.finfo(gate_logits.dtype).min).softmax(dim=1)
        fused = torch.cat((weights[:, :1] * url_state, weights[:, 1:] * html_state, difference), dim=1)
        scores = self.residual_head(fused)
        disagreement = (probs[:, 0] - probs[:, 1]).abs() * present.all(dim=1)
        correction = self.alpha.clamp_min(0) * disagreement * (scores[:, 1] - scores[:, 0]).clamp_min(0)
        return correction, (view_logits, present, coverage, weights)


def semantic_losses(details, labels):
    view_logits, present, coverage, weights = details
    log_probs = view_logits.log_softmax(dim=-1)
    probs = log_probs.exp()
    ce = -log_probs.gather(2, labels[:, None, None].expand(-1, 2, 1)).squeeze(-1)
    aux_ce = 0.5 * (masked_mean(ce[:, 0], present[:, 0]) + masked_mean(ce[:, 1], present[:, 1]))
    with torch.no_grad():
        error = ce + 1.0 - coverage
        rho = torch.exp(-error / (2 * RELIABILITY_SIGMA ** 2)) * present
        target = rho / rho.sum(dim=1, keepdim=True).clamp_min(1e-8)
    directional_kl = (probs.detach() * (log_probs.detach() - log_probs.flip((1,)))).sum(dim=-1)
    view_kl = masked_mean((target * directional_kl).sum(dim=1), present.all(dim=1))
    gate_kl = masked_mean(
        (target * (target.clamp_min(1e-8).log() - weights.clamp_min(1e-8).log())).sum(dim=1),
        present.any(dim=1),
    )
    return aux_ce, view_kl, gate_kl
