# DeciPhish Python package

See the [repository README](../README.md) for setup, training on local URL/HTML records, and inference.

- `evidence.py` and `extraction/` extract evidence from saved HTML and URL text.
- `race.py` serializes four evidence regions under the shared token budget.
- `model.py` defines the Qwen3 classifier and DCER residual.
- `training.py` optimizes both input orders and saves resumable checkpoints.

The Python CLI also retains the checksum-locked prepared-input protocol. For a checkpoint trained with that ablation protocol, pass `--profile ablation` to `python -m deciphish predict`.
