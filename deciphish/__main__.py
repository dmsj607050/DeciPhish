"""Run the fixed method with local model, data and output paths."""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(prog="python -m deciphish", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train_parser = commands.add_parser(
        "train", help="Train DeciPhish on local data"
    )
    train_parser.add_argument(
        "--data", type=Path, required=True, help="Training JSONL file"
    )
    train_parser.add_argument(
        "--output", type=Path, required=True, help="Training run directory"
    )
    train_parser.add_argument(
        "--custom-data", action="store_true",
        help="Read local JSONL records with case_id, label, url and html",
    )
    eval_parser = commands.add_parser(
        "evaluate", help="Evaluate both orders on frozen TR-OP"
    )
    eval_parser.add_argument(
        "--data", type=Path, required=True, help="Original tr_op_external.jsonl"
    )
    eval_parser.add_argument(
        "--output", type=Path, required=True, help="Evaluation output directory"
    )
    predict_parser = commands.add_parser(
        "predict", help="Classify one locally saved HTML snapshot"
    )
    predict_parser.add_argument("--url", required=True)
    predict_parser.add_argument(
        "--profile", choices=("phreshphish", "ablation"), default="phreshphish",
        help="Raw evidence profile; choose ablation for a PhishIntention-trained checkpoint",
    )
    predict_parser.add_argument(
        "--html", type=Path, required=True, help="Offline HTML snapshot"
    )
    serve_parser = commands.add_parser("serve", help="Serve browser snapshots on loopback HTTP")
    serve_parser.add_argument("--port", type=int, default=8765)
    serve_parser.add_argument("--device", default="auto", help="auto, cpu, or a CUDA device such as cuda:0")
    serve_parser.add_argument("--profile", choices=("phreshphish", "ablation"), default="phreshphish")
    for command in (train_parser, eval_parser, predict_parser, serve_parser):
        command.add_argument(
            "--base-model", type=Path, required=True, help="Local Qwen3-0.6B directory"
        )
    for command in (eval_parser, predict_parser, serve_parser):
        command.add_argument(
            "--checkpoint", type=Path, required=True, help="LoRA and DCER checkpoint"
        )
    args = parser.parse_args()

    if args.command == "serve":
        if not 1 <= args.port <= 65535:
            parser.error("--port must be between 1 and 65535")
        from .server import serve

        serve(args.base_model.resolve(strict=True), args.checkpoint.resolve(strict=True),
              port=args.port, device=args.device, profile=args.profile)
        return

    # The original training uses one CUDA device and expandable allocator segments.
    if os.name != "nt":
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch

    from .model import load_detector, load_tokenizer
    from .race import RaceEncoder

    if args.command == "train" and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        parser.error("Training requires a BF16-capable CUDA GPU")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    base_model = args.base_model.resolve(strict=True)
    encoder = RaceEncoder(load_tokenizer(base_model))
    if args.command == "train":
        from .io import read_custom_training, read_dataset
        from .training import train

        options = {}
        if args.custom_data:
            rows, digest = read_custom_training(args.data)
            options = {"train_sha256": digest, "input_profile": "phreshphish"}
        else:
            rows = read_dataset(args.data, "train")
        print(train(rows, base_model, encoder, device, args.output, **options))
        return

    model = load_detector(base_model, encoder.tokenizer, device, args.checkpoint)
    if args.command == "evaluate":
        from .evaluation import evaluate
        from .io import read_dataset

        evaluate(model, encoder, read_dataset(args.data, "test"), device, args.output)
    else:
        from .evidence import make_view
        from .evaluation import predict

        view = make_view(args.url, args.html.read_bytes(), profile=args.profile)
        print(json.dumps(predict(model, encoder, view, device), allow_nan=False))


if __name__ == "__main__":
    main()
