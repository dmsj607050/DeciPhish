# DeciPhish

DeciPhish detects phishing from a URL and a saved HTML snapshot. It uses **Qwen3-0.6B**, **RACE** evidence serialization, **dual-view training**, and **DCER** residual correction. Training uses both URL → HTML and HTML → URL. Inference uses one HTML → URL pass with a phishing threshold of 0.5.

## Setup

Use Python 3.10 or newer. Install a PyTorch build for your platform, then install the remaining dependencies:

```sh
python -m pip install -r requirements.txt
```

Training requires a BF16-capable CUDA GPU. CPU inference is supported. Place a local [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) snapshot in `model/base`, or pass its location with `--base-model`.

## Train

From the repository root, point training at a UTF-8 JSONL file of local snapshots. Each line must contain a unique `case_id`, the page `url`, the saved `html` text, and an integer `label` (`0` benign, `1` phishing). Supply examples of both classes. For example, a benign record looks like this:

```json
{"case_id":"benign-example","url":"https://example.org/","html":"<html><body>Example page</body></html>","label":0}
```

```sh
python -m deciphish train --custom-data --base-model model/base --data data/train.jsonl --output runs/my-training
```

The trainer extracts evidence locally, trains LoRA, the classification head, and DCER, and saves the model under `runs/my-training/checkpoint`. Training uses one epoch, a 4,096-token input budget, LoRA rank 8 and alpha 32. An interrupted run can resume from the most recent saved training state. Use the same inputs and output directory for a resume.

The optional `scripts/train.py` launcher provides the checksum-locked PhreshPhish training protocol. It requires the original structured-evidence files and the official Parquet splits. Run `python scripts/train.py --help` for the inputs.

## Predict

Use a checkpoint produced by training. Its directory must contain `adapter_config.json`, `adapter_model.safetensors` and `dcer.pt`.

```sh
python scripts/infer.py --base-model model/base --adapter runs/my-training/checkpoint --html examples/example.html --url https://example.org/ --device cpu
```

The command returns the phishing score and prediction (`0` benign, `1` phishing). It reads the supplied HTML file; it neither visits the URL nor executes page scripts. Use `--device cuda:0` for GPU inference.

## Layout

| Path | Purpose |
|---|---|
| `deciphish/` | Evidence extraction, RACE, DCER, the Python CLI, and training |
| `scripts/` | Standalone inference and the checksum-locked PhreshPhish training launcher |
| `config/` | Inference settings, training settings, environment, and archived checkpoint checksums |
| `examples/` | Benign HTML example for offline inference |

Model weights, datasets, predictions, and experiment logs are stored separately from Git.
