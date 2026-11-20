"""Qwen3 sequence classification with the fixed DCER residual."""

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch import nn
from transformers import AutoTokenizer, Qwen3ForSequenceClassification

from .protocol import LORA_ALPHA, LORA_DROPOUT, LORA_RANK, LORA_TARGETS, REGION_NAMES

REGION_COUNT = len(REGION_NAMES)
CHUNKS_PER_REGION = 4
READER_DIM = 256
ATTENTION_HEADS = 4
INITIAL_ALPHA = 0.01


def region_aware_chunk_pool(hidden, attention_mask, region_ids):
    """Mean-pool contiguous valid tokens into the four fixed slots per region."""
    hidden = hidden.float()
    batch, _, width = hidden.shape
    slots = REGION_COUNT * CHUNKS_PER_REGION
    pooled = hidden.new_zeros((batch, slots, width))
    valid = torch.zeros((batch, slots), dtype=torch.bool, device=hidden.device)
    for row in range(batch):
        for region in range(REGION_COUNT):
            positions = torch.nonzero(
                attention_mask[row].bool() & region_ids[row].eq(region), as_tuple=False
            ).flatten()
            count = positions.numel()
            if not count:
                continue
            chunk_ids = torch.div(
                torch.arange(count, device=hidden.device) * CHUNKS_PER_REGION,
                count,
                rounding_mode="floor",
            ).clamp_max(CHUNKS_PER_REGION - 1)
            sums = hidden.new_zeros((CHUNKS_PER_REGION, width)).index_add(
                0, chunk_ids, hidden[row].index_select(0, positions)
            )
            counts = torch.bincount(chunk_ids, minlength=CHUNKS_PER_REGION)
            start = region * CHUNKS_PER_REGION
            stop = start + CHUNKS_PER_REGION
            pooled[row, start:stop] = sums / counts.clamp_min(1).unsqueeze(-1)
            valid[row, start:stop] = counts.gt(0)
    return pooled, valid


@dataclass
class DetectorOutput:
    logits: torch.Tensor
    base_logits: torch.Tensor
    correction: torch.Tensor


class RaceDCER(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.alpha = nn.Parameter(torch.tensor(INITIAL_ALPHA, dtype=torch.float32))
        self.project = nn.Linear(backbone.config.hidden_size, READER_DIM)
        self.region_embedding = nn.Embedding(REGION_COUNT, READER_DIM)
        self.chunk_embedding = nn.Embedding(CHUNKS_PER_REGION, READER_DIM)
        self.norm = nn.LayerNorm(READER_DIM)
        self.queries = nn.Parameter(torch.empty(1, 2, READER_DIM))
        nn.init.normal_(self.queries, std=0.02)
        self.null_evidence = nn.Parameter(torch.zeros(1, 1, READER_DIM))
        self.read = nn.MultiheadAttention(READER_DIM, ATTENTION_HEADS, batch_first=True)
        self.phishing_head = nn.Linear(READER_DIM, 1)
        self.benign_head = nn.Linear(READER_DIM, 1)
        for head in (self.phishing_head, self.benign_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, input_ids, attention_mask, region_ids):
        base = self.backbone.get_base_model()
        hidden = base.model(
            input_ids, attention_mask=attention_mask, use_cache=False
        ).last_hidden_state
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        last = (positions * input_ids.ne(base.config.pad_token_id)).argmax(dim=-1)
        rows = torch.arange(input_ids.shape[0], device=input_ids.device)
        base_logits = base.score(hidden)[rows, last].float()

        chunks, valid = region_aware_chunk_pool(hidden, attention_mask, region_ids)
        region_slots = torch.arange(
            REGION_COUNT, device=hidden.device
        ).repeat_interleave(CHUNKS_PER_REGION)
        chunk_slots = torch.arange(CHUNKS_PER_REGION, device=hidden.device).repeat(
            REGION_COUNT
        )
        evidence = self.norm(
            self.project(chunks)
            + self.region_embedding(region_slots)
            + self.chunk_embedding(chunk_slots)
        )
        keys = torch.cat(
            (self.null_evidence.expand(len(rows), -1, -1), evidence), dim=1
        )
        # An all-empty sample attends to the null key only.
        key_mask = torch.cat((valid.any(dim=1, keepdim=True), ~valid), dim=1)
        readouts, _ = self.read(
            self.queries.expand(len(rows), -1, -1),
            keys,
            keys,
            key_padding_mask=key_mask,
            need_weights=False,
        )
        phishing = self.phishing_head(readouts[:, 0]).squeeze(-1)
        benign = self.benign_head(readouts[:, 1]).squeeze(-1)
        correction = self.alpha * (phishing - benign)
        logits = base_logits + torch.stack((-0.5 * correction, 0.5 * correction), dim=1)
        return DetectorOutput(logits, base_logits, correction)

    def save_checkpoint(self, destination: Path):
        state = {
            name: value.detach().cpu()
            for name, value in self.state_dict().items()
            if not name.startswith("backbone.")
        }
        with TemporaryDirectory(dir=destination.parent) as directory:
            staged = Path(directory) / "checkpoint"
            self.backbone.save_pretrained(staged)
            torch.save(state, staged / "dcer.pt")
            staged.rename(destination)


def load_tokenizer(base_model):
    tokenizer = AutoTokenizer.from_pretrained(
        base_model, local_files_only=True, use_fast=True
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_detector(base_model, tokenizer, device, checkpoint=None):
    base = Qwen3ForSequenceClassification.from_pretrained(
        base_model, num_labels=2, torch_dtype=torch.bfloat16, local_files_only=True
    )
    base.config.pad_token_id = tokenizer.pad_token_id
    base.config.use_cache = False
    base.config.id2label = {0: "BENIGN", 1: "PHISHING"}
    base.config.label2id = {"BENIGN": 0, "PHISHING": 1}
    if checkpoint is None:
        config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=list(LORA_TARGETS),
            modules_to_save=["score"],
            bias="none",
        )
        backbone = get_peft_model(base, config).to(device)
        backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        backbone.enable_input_require_grads()
    else:
        backbone = PeftModel.from_pretrained(
            base, checkpoint, local_files_only=True
        ).to(device)
    # Preserve the backbone RNG stream across DCER initialization.
    with torch.random.fork_rng(devices=[]):
        detector = RaceDCER(backbone).to(device)
    if checkpoint is not None:
        state = torch.load(
            Path(checkpoint) / "dcer.pt", map_location="cpu", weights_only=True
        )
        expected = {
            name for name in detector.state_dict() if not name.startswith("backbone.")
        }
        if set(state) != expected:
            raise ValueError("Checkpoint does not contain the fixed DCER reader")
        detector.load_state_dict(state, strict=False)
        detector.eval()
    return detector
