"""Score one saved page or RACE evidence record with the archived DCER checkpoint."""
import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel

import sys

ROOT = Path(__file__).resolve().parent.parent
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent / "training_snapshot"))

import train_race_dcer as run
from race_dcer import RaceDCER

from deciphish.evidence import make_view


def raw_evidence(url, html):
    return {"view": make_view(url, html.encode("utf-8", errors="replace"), profile="phreshphish")}


def load_model(base, adapter, device):
    tokenizer = run.AutoTokenizer.from_pretrained(base, local_files_only=True, use_fast=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    backbone = run.Qwen3ForSequenceClassification.from_pretrained(
        base, num_labels=2, torch_dtype=torch.bfloat16, local_files_only=True
    )
    backbone.config.pad_token_id = tokenizer.pad_token_id
    backbone.config.use_cache = False
    backbone.config.id2label = {0: "BENIGN", 1: "PHISHING"}
    backbone.config.label2id = {"BENIGN": 0, "PHISHING": 1}
    tuned = PeftModel.from_pretrained(backbone, adapter, is_trainable=False, local_files_only=True)
    model = RaceDCER(tuned, semantic=False)
    state = torch.load(adapter / "dcer.pt", map_location="cpu", weights_only=True)
    expected = {key for key in model.state_dict() if not key.startswith("backbone.")}
    if state.keys() != expected:
        raise ValueError("DCER checkpoint keys do not match the archived model")
    model.load_state_dict(state, strict=False)
    return model.to(device).eval(), tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--evidence", type=Path, help="JSON object with one view")
    mode.add_argument("--html", type=Path, help="Saved UTF-8 HTML")
    parser.add_argument("--url", help="Page URL used as text")
    parser.add_argument("--base-model", type=Path, default=ROOT / "model/base")
    parser.add_argument("--adapter", type=Path, default=ROOT / "model/adapter")
    parser.add_argument("--device", choices=("cuda:0", "cpu"), default="cuda:0")
    args = parser.parse_args()
    if (args.html is not None) != (args.url is not None):
        parser.error("--html and --url must be used together")
    row = (json.loads(args.evidence.read_text(encoding="utf-8")) if args.evidence
           else raw_evidence(args.url, args.html.read_text(encoding="utf-8")))
    config = json.loads((ROOT / "config/inference.json").read_text())
    run.MAXLEN = config["max_length"]
    model, tokenizer = load_model(args.base_model, args.adapter, args.device)
    views, _ = run.prep([row], tokenizer, "inference")
    ids, regions = views[config["test_order"]][0]
    batch = tokenizer.pad([{"input_ids": ids}], padding=True, pad_to_multiple_of=8,
                          return_tensors="pt")
    padded = batch["input_ids"].shape[1]
    batch["region_ids"] = torch.tensor([regions + [-1] * (padded - len(regions))])
    with torch.inference_mode():
        out = model(**{name: value.to(args.device) for name, value in batch.items()})
        logits = out.logits.float()[0].cpu()
        score = (logits / config["temperature"]).softmax(0)[1].item()
    print(json.dumps({"logits": logits.tolist(), "phishing_score": score,
                      "prediction": int(score >= config["threshold"]),
                      "threshold": config["threshold"], "temperature": config["temperature"],
                      "dcer_logit_correction": out.correction.item(), "input_tokens": len(ids)},
                     allow_nan=False))


if __name__ == "__main__":
    main()
