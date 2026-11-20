"""Train the selected full model on the official PhreshPhish splits."""

import hashlib
import json
import os
import signal
import time
from bisect import bisect_right
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

import train_race_dcer as run
OUT = Path(__file__).resolve().parent
CACHE = Path(os.environ.get("PHRESH_TOKEN_CACHE", str(OUT / "token_cache")))
SOURCE = OUT / "data" / "phreshphish_evidence"
OFFICIAL = OUT / "data" / "phreshphish"
SIZES = {"train": 498255, "test": 168060}
LABELS = {"benign": 0, "phish": 1}
ORDERS = ("url_html", "html_url")
SHARD_ROWS = 1024
WORKERS = 16
STOP = False
TOKENIZER = None


def request_stop(_signum, _frame):
    global STOP
    STOP = True


def should_pause():
    return STOP


def log(stage, **fields):
    print(json.dumps({"stage": stage, **fields}), flush=True)


def make_tokenizer(base_model):
    tok = run.AutoTokenizer.from_pretrained(str(base_model), local_files_only=True, use_fast=True)
    tok.padding_side = "right"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def init_worker(base_model):
    global TOKENIZER
    TOKENIZER = make_tokenizer(base_model)


def save_arrays(path, **arrays):
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def index_source(split, folder):
    source = SOURCE / f"{split}.jsonl"
    stat = source.stat()
    stamp = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "shard_rows": SHARD_ROWS}
    target = folder / "index.json"
    if target.is_file():
        saved = json.loads(target.read_text())
        if saved["stamp"] != stamp:
            raise RuntimeError(f"{source} changed since tokenization started")
        return source, saved
    digest = hashlib.sha256()
    offsets, rows, offset = [], 0, 0
    with source.open("rb") as stream:
        for line in stream:
            if rows % SHARD_ROWS == 0:
                offsets.append(offset)
            digest.update(line)
            offset += len(line)
            rows += 1
    if rows != SIZES[split]:
        raise ValueError(f"{source}: expected {SIZES[split]} rows, got {rows}")
    saved = {"stamp": stamp, "sha256": digest.hexdigest(), "rows": rows, "offsets": offsets}
    run.write(target, saved)
    return source, saved


def tokenize_shard(source, offset, count, path, split):
    with source.open("rb") as stream:
        stream.seek(offset)
        rows = [json.loads(stream.readline()) for _ in range(count)]
    ids = []
    labels = []
    for row in rows:
        case_id = row["case_id"]
        if not case_id.startswith("phresh-") or len(case_id) != 71:
            raise ValueError("Unexpected PhreshPhish case ID")
        if row["label"] not in (0, 1) or set(row["view"]) != {"coverage", "url", "html"}:
            raise ValueError(f"Unexpected evidence row for {case_id}")
        ids.append(case_id.removeprefix("phresh-").encode("ascii"))
        labels.append(row["label"])
    views, audit = run.prep(rows, TOKENIZER, split)
    arrays = {
        "case_ids": np.asarray(ids, dtype="S64"),
        "labels": np.asarray(labels, dtype=np.uint8),
        "truncated": np.asarray(audit["truncated_samples_by_allocator"]),
    }
    for order in (ORDERS if split == "train" else ("html_url",)):
        sequences = views[order]
        arrays[order + "_offsets"] = np.cumsum([0] + [len(ids) for ids, _ in sequences], dtype=np.int32)
        arrays[order + "_ids"] = np.concatenate([np.asarray(ids, dtype=np.int32) for ids, _ in sequences])
        arrays[order + "_regions"] = np.concatenate([np.asarray(regions, dtype=np.int8) for _, regions in sequences])
    save_arrays(path, **arrays)


class CachedSplit:
    def __init__(self, paths):
        self.shards = []
        self.starts = [0]
        for path in paths:
            with np.load(path) as arrays:
                shard = dict(arrays)
            self.shards.append(shard)
            self.starts.append(self.starts[-1] + len(shard["labels"]))

    def __len__(self):
        return self.starts[-1]

    def locate(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        shard = bisect_right(self.starts, index) - 1
        return self.shards[shard], index - self.starts[shard]

    def __getitem__(self, index):
        shard, i = self.locate(index)
        digest = shard["case_ids"][i].decode("ascii")
        return {"case_id": "phresh-" + digest, "label": int(shard["labels"][i])}


class CachedView:
    def __init__(self, split, order):
        self.split = split
        self.order = order

    def __getitem__(self, index):
        shard, i = self.split.locate(index)
        start, end = shard[self.order + "_offsets"][i:i + 2]
        return (
            shard[self.order + "_ids"][start:end].tolist(),
            shard[self.order + "_regions"][start:end].tolist(),
        )


def prepare(split):
    folder = CACHE / split
    folder.mkdir(parents=True, exist_ok=True)
    source, index = index_source(split, folder)
    paths = [folder / f"{i:04d}.npz" for i in range(len(index["offsets"]))]
    pending = [
        (source, offset, min(SHARD_ROWS, index["rows"] - i * SHARD_ROWS), path, split)
        for i, (offset, path) in enumerate(zip(index["offsets"], paths)) if not path.is_file()
    ]
    log("prepare", split=split, cached=len(paths) - len(pending), shards=len(paths))
    if pending:
        with ProcessPoolExecutor(
            max_workers=WORKERS, initializer=init_worker, initargs=(run.BASE,)
        ) as pool:
            tasks = [pool.submit(tokenize_shard, *args) for args in pending]
            for done, task in enumerate(as_completed(tasks), len(paths) - len(pending) + 1):
                task.result()
                if done % 10 == 0 or done == len(paths):
                    log("tokenize", split=split, shards_done=done, shards=len(paths))
                if should_pause():
                    pool.shutdown(wait=True, cancel_futures=True)
                    return None, None
    return CachedSplit(paths), index


def audit(split, cache, index):
    truth = {}
    for path in sorted(OFFICIAL.glob(f"{split}-*.parquet")):
        for batch in pq.ParquetFile(path).iter_batches(columns=["sha256", "label"]):
            ids, labels = (column.to_pylist() for column in batch.columns)
            for digest, label in zip(ids, labels):
                if digest in truth:
                    raise ValueError(f"Duplicate official ID in {split}")
                truth[digest] = LABELS[label]
    if len(truth) != len(cache) or len(cache) != SIZES[split]:
        raise ValueError(f"Official/cache size mismatch in {split}")
    counts = Counter()
    seen = set()
    for row in cache:
        digest = row["case_id"][7:]
        if digest in seen or truth.get(digest) != row["label"]:
            raise ValueError(f"Official/cache ID or label mismatch in {split}: {digest}")
        seen.add(digest)
        counts[row["label"]] += 1
    result = {
        "rows": len(cache), "class_counts": dict(counts), "source_sha256": index["sha256"],
        "truncated_samples": sum(int(shard["truncated"]) for shard in cache.shards),
    }
    return seen, result


def evaluate(model, data, view, tok, audit_result):
    directory = OUT / "test"
    shards = directory / "shards"
    shards.mkdir(parents=True, exist_ok=True)
    model.eval()
    paths = []
    with torch.inference_mode():
        for i, (start, end) in enumerate(zip(data.starts, data.starts[1:])):
            path = shards / f"{i:04d}.npz"
            paths.append(path)
            if path.is_file():
                continue
            if should_pause():
                log("paused", split="test", rows_done=start)
                return None
            logits, scores, corrections = [], [], []
            started = time.perf_counter()
            for offset in range(start, end, run.BATCH):
                indices = range(offset, min(offset + run.BATCH, end))
                output = model(**run.pad(view, indices, tok))
                logits.append(output.logits.cpu().numpy())
                scores.append((output.logits / run.TEMP).softmax(dim=1)[:, 1].cpu().numpy())
                corrections.append(output.correction.cpu().numpy())
            save_arrays(
                path, logits=np.concatenate(logits), scores=np.concatenate(scores),
                correction=np.concatenate(corrections), seconds=time.perf_counter() - started,
            )
            log("evaluate", view="html_url", done=end, total=len(data))
    scores, seconds = [], 0.0
    target = directory / "predictions.jsonl"
    tmp = target.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        for start, path in zip(data.starts, paths):
            with np.load(path) as arrays:
                seconds += float(arrays["seconds"])
                scores.extend(arrays["scores"].tolist())
                for offset, (logits, score, correction) in enumerate(
                    zip(arrays["logits"], arrays["scores"], arrays["correction"])
                ):
                    row = data[start + offset]
                    row.update(
                        logit_benign=float(logits[0]), logit_phishing=float(logits[1]),
                        phishing_score=float(score), prediction=int(score >= run.THRESH),
                        threshold=run.THRESH, temperature=run.TEMP,
                        dcer_logit_correction=float(correction),
                    )
                    stream.write(json.dumps(row) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(target)
    if len(scores) != len(data):
        raise RuntimeError("Test prediction count mismatch")
    metrics = run.metrics(data, scores)
    metrics.update(
        test_sha256=audit_result["source_sha256"], truncated_samples=audit_result["truncated_samples"],
        evaluation_batch_size=run.BATCH,
        throughput_samples_per_second=len(data) / seconds,
        single_sample_latency=run.latency(model, data, {"html_url": view}, tok, "html_url"),
    )
    run.write(directory / "metrics.json", metrics)
    return metrics


def main():
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    run.OUT = OUT
    run.SEMANTIC = False
    status = OUT / "status.json"
    if status.exists() and json.loads(status.read_text()).get("stage") == "COMPLETE":
        return
    if should_pause():
        return
    run.write(status, {"stage": "PREPARING", "updated_at_utc": run.now()})
    train, train_index = prepare("train")
    if train is None or should_pause():
        return
    test, test_index = prepare("test")
    if test is None or should_pause():
        return
    train_ids, train_audit = audit("train", train, train_index)
    test_ids, test_audit = audit("test", test, test_index)
    if train_ids & test_ids:
        raise ValueError("Train and test share official IDs")
    run.write(OUT / "data_audit.json", {"train": train_audit, "test": test_audit, "shared_ids": 0})
    run.TRAIN_SHA, run.TEST_SHA = train_index["sha256"], test_index["sha256"]
    run.write(OUT / "config.json", {
        "model": "Qwen3-0.6B + RACE + dual view + DCER",
        "initialization": str(run.BASE), "epochs": run.EPOCHS, "max_length": run.MAXLEN,
        "lora_r": run.RANK, "lora_alpha": run.ALPHA, "learning_rate": run.LR,
        "batch": run.BATCH, "micro_batch": run.MICRO, "checkpoint_steps": run.SAVE_EVERY,
        "seed": run.SEED, "loss": "mean classification cross-entropy over both training orders",
        "token_cache": str(CACHE),
        "train_orders": ORDERS, "test_order": "html_url", "threshold": run.THRESH,
        "temperature": run.TEMP, "dataset": "PhreshPhish official train and test",
    })
    if should_pause():
        return
    tok = make_tokenizer(run.BASE)
    run.write(status, {"stage": "TRAINING", "updated_at_utc": run.now()})
    views = {order: CachedView(train, order) for order in ORDERS}
    model, training = run.train("dual_view", train, views, tok, pause_requested=should_pause)
    if model is None or should_pause():
        run.write(status, {"stage": "PAUSED", "updated_at_utc": run.now()})
        return
    run.write(status, {"stage": "EVALUATING", "updated_at_utc": run.now()})
    metrics = evaluate(model, test, CachedView(test, "html_url"), tok, test_audit)
    if metrics is None:
        run.write(status, {"stage": "PAUSED", "updated_at_utc": run.now()})
        return
    run.write(OUT / "summary.json", {"training": training, "test": metrics})
    run.write(status, {"stage": "COMPLETE", "updated_at_utc": run.now()})
    log("complete", output=str(OUT))


if __name__ == "__main__":
    main()
