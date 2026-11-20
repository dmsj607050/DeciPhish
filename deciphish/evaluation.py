"""Evaluate each order independently at the fixed operating threshold."""

import json

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from .io import atomic_file, write_json
from .protocol import BATCH_SIZE, DATASET_SHA256, EVAL_VIEWS, PRIMARY_VIEW, THRESHOLD


def binary_metrics(labels, scores):
    actual = np.asarray(labels, dtype=bool)
    predicted = np.asarray(scores, dtype=np.float64) >= THRESHOLD
    tp = int(np.count_nonzero(actual & predicted))
    tn = int(np.count_nonzero(~actual & ~predicted))
    fp = int(np.count_nonzero(~actual & predicted))
    fn = int(np.count_nonzero(actual & ~predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn)
    return {
        "samples": len(labels),
        "threshold": THRESHOLD,
        "temperature": 1.0,
        "threshold_source": "fixed a priori; no test tuning",
        "test_sha256": DATASET_SHA256["test"],
        "accuracy": (tp + tn) / len(labels),
        "precision": precision,
        "phishing_recall": recall,
        "f1": 2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0,
        "fpr": fp / (fp + tn),
        "roc_auc": float(roc_auc_score(actual, scores)),
        "average_precision": float(average_precision_score(actual, scores)),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


@torch.inference_mode()
def evaluate(model, encoder, rows, device, destination):
    model.eval()
    labels = [row["label"] for row in rows]
    views = encoder.encode([row["view"] for row in rows], EVAL_VIEWS)
    results = {}
    for view in EVAL_VIEWS:
        scores = []
        path = destination / f"predictions_{view}.jsonl"
        with atomic_file(path) as output:
            for start in range(0, len(rows), BATCH_SIZE):
                stop = min(start + BATCH_SIZE, len(rows))
                batch = encoder.batch(views[view][start:stop], device)
                result = model(**batch)
                logits = result.logits.cpu()
                probabilities = logits.softmax(dim=-1)[:, 1].tolist()
                scores.extend(probabilities)
                corrections = result.correction.cpu().tolist()
                for row, pair, probability, correction in zip(
                    rows[start:stop],
                    logits.tolist(),
                    probabilities,
                    corrections,
                    strict=True,
                ):
                    prediction = {
                        "case_id": row["case_id"],
                        "label": row["label"],
                        "logit_benign": pair[0],
                        "logit_phishing": pair[1],
                        "phishing_score": probability,
                        "prediction": int(probability >= THRESHOLD),
                        "threshold": THRESHOLD,
                        "temperature": 1.0,
                        "dcer_logit_correction": correction,
                    }
                    output.write(
                        json.dumps(prediction, ensure_ascii=False, allow_nan=False)
                        + "\n"
                    )
                if stop % 1000 == 0 or stop == len(rows):
                    print(f"{view} {stop}/{len(rows)}", flush=True)
        results[view] = binary_metrics(labels, scores)
        write_json(destination / f"metrics_{view}.json", results[view])
    write_json(
        destination / "summary.json",
        {"primary_view": PRIMARY_VIEW, "test_views": results},
    )
    return results


@torch.inference_mode()
def predict(model, encoder, view, device):
    model.eval()
    sequence = encoder.encode([view], (PRIMARY_VIEW,))[PRIMARY_VIEW]
    logits = model(**encoder.batch(sequence, device)).logits.cpu()
    score = logits.softmax(dim=-1)[0, 1].item()
    return {"phishing_score": score, "prediction": int(score >= THRESHOLD)}
