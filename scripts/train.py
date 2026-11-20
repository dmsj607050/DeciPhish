"""Run the archived PhreshPhish training and complete-Test evaluation."""
import argparse
import os
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "training_snapshot"))

import run_phreshphish as job
import train_race_dcer as run

ROOT = Path(__file__).resolve().parent.parent
HASHES = {
    "train": "4bd2dd994d79dd740049e98aa18f2041adf29b5631adcaa077cc362d4aebce57",
    "test": "8c45b35717d869068b07b50b946a6f0ac5b55bbc480d35ee60c37417ca5fd44b",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--official-data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-inputs", action="store_true", help="Verify prepared evidence checksums and official shard presence without running training")
    parser.add_argument("--base-model", type=Path, default=ROOT / "model/base")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.output is None and not args.verify_inputs:
        parser.error("--output is required unless --verify-inputs is passed")
    job.SOURCE = args.evidence_dir.resolve()
    job.OFFICIAL = args.official_data_dir.resolve()
    for split, digest in HASHES.items():
        source = job.SOURCE / f"{split}.jsonl"
        if not source.is_file():
            parser.error(f"missing prepared evidence: {source}")
        if run.sha(source) != digest:
            parser.error(f"{source} does not match the archived split checksum")
        if not any(job.OFFICIAL.glob(f"{split}-*.parquet")):
            parser.error(f"missing official {split} parquet shards in {job.OFFICIAL}")
    if args.verify_inputs:
        print("Prepared evidence SHA256 verified for train and test; official shards found.")
        return

    import torch

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        parser.error("training requires a BF16-capable CUDA GPU")
    run.BASE = args.base_model.resolve()
    if not (run.BASE / "config.json").is_file():
        parser.error(f"missing Qwen3 model configuration: {run.BASE}")
    out = args.output.resolve()
    if (out == ROOT or ROOT in out.parents) and not out.is_relative_to(ROOT / "runs"):
        parser.error("--output must be under runs/ or outside this checkout")
    job.OUT, job.CACHE, job.WORKERS = out, out / "token_cache", args.workers
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    job.main()


if __name__ == "__main__":
    main()
