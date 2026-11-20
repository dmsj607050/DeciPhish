"""Compact structured evidence and serialize it with aligned region IDs."""

import json
from dataclasses import dataclass

from .protocol import MAX_TOKENS, ORDERS, REGION_HEADERS

INTERACTIVE_CLAIMS = {
    "html.anchor",
    "html.button",
    "html.form",
    "html.input",
    "html.interaction",
    "html.iframe",
    "html.script_behavior",
}
DROP_FIELDS = {"analysis", "method_source", "action_source", "action_raw", "href_raw"}
TEXT_LIMITS = {"text": 128, "label_hint": 64, "purpose_hint": 48, "visibility_hint": 48}
TRIM_ORDER = (3, 2, 1, 0)
MIN_REGION_TOKENS = 64
MAX_HTML_RECORDS = 64


@dataclass(frozen=True)
class Sequence:
    input_ids: list[int]
    region_ids: list[int]


def html_region(item):
    claim = item["claim"]
    locator = str((item.get("source") or {}).get("locator", "")).lower()
    if claim in INTERACTIVE_CLAIMS:
        return 2
    if claim in {"html.document_title", "html.meta_description"} or locator.startswith(
        "/html/head"
    ):
        return 1
    return 3


def compact_value(value, field=""):
    if field in DROP_FIELDS or value is None:
        return None
    if isinstance(value, dict):
        compacted = {}
        for key, child in value.items():
            if key in {"source", "locator"} or key.endswith("_source"):
                continue
            if isinstance(child, str) and child.strip().upper() == "UNKNOWN":
                continue
            child = compact_value(child, key)
            if child is not None:
                compacted[key] = child
        return compacted or None
    if isinstance(value, list):
        children = [compact_value(child, field) for child in value[:8]]
        return [child for child in children if child is not None] or None
    if isinstance(value, str):
        limit = TEXT_LIMITS.get(field, 192)
        return value if len(value) <= limit else value[:limit] + "…"
    return value


def compact_regions(view):
    url = [
        [item["claim"], item["value"], (item.get("source") or {}).get("locator")]
        for item in view["url"]
    ]
    regions = [url, [], [], []]
    seen = [set() for _ in regions]
    for item in view["html"]:
        region = html_region(item)
        if len(regions[region]) >= MAX_HTML_RECORDS:
            continue
        record = [item["claim"].removeprefix("html."), compact_value(item.get("value"))]
        key = json.dumps(
            record, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )
        if key not in seen[region]:
            seen[region].add(key)
            regions[region].append(record)
    return regions


def allocate_tokens(lengths, budget):
    retained = list(lengths)
    overflow = max(0, sum(retained) - budget)
    for floor in (MIN_REGION_TOKENS, 0):
        for region in TRIM_ORDER:
            cut = min(overflow, max(0, retained[region] - floor))
            retained[region] -= cut
            overflow -= cut
        if overflow == 0:
            return retained
    raise ValueError("The region headers exceed the token budget")


class RaceEncoder:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.headers = [
            tokenizer.encode(text, add_special_tokens=False) for text in REGION_HEADERS
        ]
        self.prefix = [] if tokenizer.bos_token_id is None else [tokenizer.bos_token_id]
        self.budget = MAX_TOKENS - len(self.prefix) - sum(map(len, self.headers))

    def encode(self, views, orders):
        sequences = {order: [] for order in orders}
        # Encode the four regions once. Both orders use exactly the same payloads.
        for begin in range(0, len(views), 256):
            groups = [compact_regions(view) for view in views[begin : begin + 256]]
            tokenized = []
            for region in range(len(REGION_HEADERS)):
                texts = [
                    json.dumps(group[region], ensure_ascii=False, separators=(",", ":"))
                    for group in groups
                ]
                tokenized.append(
                    self.tokenizer(texts, add_special_tokens=False, truncation=False)[
                        "input_ids"
                    ]
                )
            for group, *tokens in zip(groups, *tokenized):
                lengths = allocate_tokens([len(ids) for ids in tokens], self.budget)
                for order in orders:
                    ids = list(self.prefix)
                    region_ids = [-1] * len(ids)
                    for region in ORDERS[order]:
                        ids.extend(self.headers[region])
                        region_ids.extend([-1] * len(self.headers[region]))
                        ids.extend(tokens[region][: lengths[region]])
                        region_ids.extend(
                            [region if group[region] else -1] * lengths[region]
                        )
                    sequences[order].append(Sequence(ids, region_ids))
        return sequences

    def batch(self, sequences, device):
        batch = self.tokenizer.pad(
            [{"input_ids": sequence.input_ids} for sequence in sequences],
            padding=True,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )
        width = batch["input_ids"].shape[1]
        batch["region_ids"] = batch["input_ids"].new_tensor(
            [
                sequence.region_ids + [-1] * (width - len(sequence.region_ids))
                for sequence in sequences
            ]
        )
        return {key: value.to(device) for key, value in batch.items()}
