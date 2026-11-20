"""Frozen experiment inputs and atomic result files."""

import hashlib
import json
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile

from .protocol import DATASET_SHA256


def read_dataset(path: Path, split: str):
    # Parse the same bytes that were checked against the experiment manifest.
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != DATASET_SHA256[split]:
        raise ValueError(f"{split} SHA-256 differs from the frozen experiment")
    rows = [json.loads(line) for line in content.splitlines() if line.strip()]
    if Counter(row["label"] for row in rows) != {0: 5000, 1: 5000}:
        raise ValueError(f"{split} must contain 5,000 rows per class")
    if len({row["case_id"] for row in rows}) != len(rows):
        raise ValueError(f"{split} contains duplicate sample IDs")
    if any(set(row["view"]) != {"coverage", "url", "html"} for row in rows):
        raise ValueError(f"{split} rows must contain the structured evidence view")
    return rows


def read_custom_training(path: Path):
    """Read local URL/HTML JSONL and bind its exact bytes to the run manifest."""
    from .evidence import make_view

    content = path.read_bytes()
    rows = []
    seen = set()
    for line_number, line in enumerate(content.splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            case_id, label = record["case_id"], record["label"]
            url, snapshot = record["url"], record["html"]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{path}:{line_number}: expected case_id, label, url and html") from error
        if not isinstance(case_id, str) or not case_id.strip() or case_id in seen:
            raise ValueError(f"{path}:{line_number}: empty or duplicate case_id")
        if type(label) is not int or label not in (0, 1):
            raise ValueError(f"{path}:{line_number}: label must be integer 0 or 1")
        if not isinstance(url, str) or not isinstance(snapshot, str):
            raise ValueError(f"{path}:{line_number}: url and html must be strings")
        seen.add(case_id)
        rows.append({
            "case_id": case_id,
            "label": label,
            "view": make_view(url, snapshot.encode("utf-8"), profile="phreshphish"),
        })
    if {row["label"] for row in rows} != {0, 1}:
        raise ValueError("Custom training data must contain examples of both classes")
    return rows, hashlib.sha256(content).hexdigest()


@contextmanager
def atomic_file(path: Path, *, binary=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "wb" if binary else "w"
    encoding = None if binary else "utf-8"
    temporary = NamedTemporaryFile(
        mode=mode,
        encoding=encoding,
        newline=None if binary else "\n",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    )
    staged = Path(temporary.name)
    try:
        with temporary as handle:
            yield handle
        staged.replace(path)
    finally:
        staged.unlink(missing_ok=True)


def write_json(path, value):
    with atomic_file(path) as output:
        json.dump(value, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write("\n")
