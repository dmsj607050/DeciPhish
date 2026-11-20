"""One epoch of joint training over both orders of every page."""

import json
import math
import random

import torch
import torch.nn.functional as F
from transformers import get_cosine_schedule_with_warmup

from .io import atomic_file, write_json
from .model import load_detector
from .protocol import (
    BATCH_SIZE,
    DATASET_SHA256,
    LEARNING_RATE,
    MICRO_BATCH_SIZE,
    ORDERS,
    SEED,
    WARMUP_FRACTION,
    WEIGHT_DECAY,
)

SAVE_EVERY = 250
LOG_EVERY = 50


def train(
    rows, base_model, encoder, device, destination, *,
    train_sha256=DATASET_SHA256["train"], input_profile=None,
):
    """Train or resume an interrupted run in the same output directory."""
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint = destination / "checkpoint"
    if checkpoint.exists():
        raise FileExistsError(f"Completed checkpoint already exists: {checkpoint}")
    manifest = destination / "training_protocol.json"
    state_path = destination / "training_state.pt"
    protocol = {
        "method": "RACE + dual-order training + DCER",
        "base_model": str(base_model),
        "train_sha256": train_sha256,
        "seed": SEED,
        "epochs": 1,
        "batch_size": BATCH_SIZE,
        "micro_batch_size": MICRO_BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "warmup_fraction": WARMUP_FRACTION,
    }
    if input_profile is not None:
        protocol["input_profile"] = input_profile
    if manifest.exists():
        if json.loads(manifest.read_text(encoding="utf-8")) != protocol:
            raise ValueError("Run directory belongs to a different training protocol")
    else:
        write_json(manifest, protocol)

    torch.manual_seed(SEED)
    model = load_detector(base_model, encoder.tokenizer, device)
    model.train()
    parameters = {
        name: value for name, value in model.named_parameters() if value.requires_grad
    }
    optimizer = torch.optim.AdamW(
        parameters.values(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    total_steps = math.ceil(len(rows) / BATCH_SIZE)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(total_steps * WARMUP_FRACTION), total_steps
    )
    sample_order = list(range(len(rows)))
    random.Random(SEED).shuffle(sample_order)
    views = encoder.encode([row["view"] for row in rows], tuple(ORDERS))
    first_step = 0
    if state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        if set(state["parameters"]) != set(parameters):
            raise ValueError("Training state does not match the trainable parameters")
        with torch.no_grad():
            for name, value in parameters.items():
                value.copy_(state["parameters"][name])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["cpu_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        first_step = state["step"]
        if not 0 <= first_step <= total_steps:
            raise ValueError("Training state has an invalid step")
        del state

    accumulated_loss = 0.0
    logged_steps = 0
    for step in range(first_step, total_steps):
        indices = sample_order[step * BATCH_SIZE : (step + 1) * BATCH_SIZE]
        optimizer.zero_grad(set_to_none=True)
        for order in ORDERS:
            for begin in range(0, len(indices), MICRO_BATCH_SIZE):
                micro = indices[begin : begin + MICRO_BATCH_SIZE]
                batch = encoder.batch([views[order][i] for i in micro], device)
                labels = torch.tensor([rows[i]["label"] for i in micro], device=device)
                result = model(**batch)
                if step == 0 and order == "url_html" and begin == 0:
                    if not torch.equal(result.logits, result.base_logits):
                        raise RuntimeError(
                            "Zero-initialized DCER must preserve the base logits"
                        )
                loss = F.cross_entropy(result.logits, labels)
                weighted_loss = (0.5 * len(micro) / len(indices)) * loss
                weighted_loss.backward()
                accumulated_loss += weighted_loss.detach().item()
                del result, batch, loss, weighted_loss
        optimizer.step()
        scheduler.step()
        logged_steps += 1
        completed = step + 1
        if completed % LOG_EVERY == 0 or completed == total_steps:
            print(
                f"step {completed}/{total_steps} mean CE {accumulated_loss / logged_steps:.6f}",
                flush=True,
            )
            accumulated_loss = 0.0
            logged_steps = 0
        if completed % SAVE_EVERY == 0 or completed == total_steps:
            with atomic_file(state_path, binary=True) as handle:
                torch.save(
                    {
                        "step": completed,
                        "parameters": {
                            name: value.detach().cpu()
                            for name, value in parameters.items()
                        },
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "cpu_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state_all(),
                    },
                    handle,
                )

    model.save_checkpoint(checkpoint)
    state_path.unlink(missing_ok=True)
    return checkpoint
