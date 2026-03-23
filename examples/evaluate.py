"""Evaluate M&A extraction against the golden dataset.

Computes per-field F1, hallucination rate, and miss rate.
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# Critical fields get 3x weight in overall scoring
CRITICAL_FIELDS = {
    "deal_type", "deal_status", "acquirer_name", "target_name",
    "consideration_size", "deal_announcement_date",
}

# Map golden CSV column names to our model field names
FIELD_MAP = {
    "acquirer_name": "acquirors.0.name",
    "acquirer_legal_name": "acquirors.0.legal_name",
    "acquirer_entity_type": "acquirors.0.entity_type",
    "acquirer_ticker": "acquirors.0.ticker",
    "acquirer_country": "acquirors.0.country",
    "acquirer_state_province": "acquirors.0.state_province",
    "acquirer_ultimate_parent_name": "acquiror_ultimate_parents.0.name",
    "target_name": "target.name",
    "target_legal_name": "target.legal_name",
    "target_entity_type": "target.entity_type",
    "target_ticker": "target.ticker",
    "target_country": "target.country",
    "target_state_province": "target.state_province",
    "seller_name": "sellers.0.name",
    "pe": "price_to_earnings",
}

# Fields that are direct matches (same name in both)
DIRECT_FIELDS = [
    "deal_type", "deal_status", "deal_attitude", "deal_summary", "deal_rationale",
    "undisturbed_target_share_price_date", "deal_rumor_date", "bid_date",
    "letter_of_intent_date", "agreement_signing_date", "deal_announcement_date",
    "expected_completion_timeframe", "revision_date", "cancellation_date",
    "completion_date", "consideration_size", "pct_shares_acquired",
    "pct_shares_owned_post_deal_completion", "source_of_funding_summary",
]


def normalize(val: str | None) -> str | None:
    """Normalize a value for comparison."""
    if val is None or val == "" or val == "NULL" or val == "Null" or val == "null":
        return None
    return str(val).strip().lower()


def get_nested(data: dict, path: str):
    """Get a nested value from a dict using dot notation (e.g., 'acquirors.0.name')."""
    parts = path.split(".")
    current = data
    for part in parts:
        if current is None:
            return None
        if part.isdigit():
            idx = int(part)
            if isinstance(current, list) and idx < len(current):
                current = current[idx]
            else:
                return None
        elif isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current


def compute_field_metrics(golden_vals: list, extracted_vals: list) -> dict:
    """Compute TP, FP, FN for a single field across all rows."""
    tp = fp = fn = 0
    for g, e in zip(golden_vals, extracted_vals):
        gn = normalize(g)
        en = normalize(e)
        if gn is not None and en is not None:
            if gn == en:
                tp += 1
            else:
                fp += 1  # Hallucination (wrong value)
        elif gn is not None and en is None:
            fn += 1  # Miss
        elif gn is None and en is not None:
            fp += 1  # Hallucination (value where none should be)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    total = len(golden_vals)
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "hallucination_rate": round(fp / total, 4) if total > 0 else 0.0,
        "miss_rate": round(fn / total, 4) if total > 0 else 0.0,
    }


def load_golden(path: str = "docs/golden_data_set.csv") -> list[dict]:
    """Load golden dataset."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def evaluate(extractions: dict[str, dict], golden_path: str = "docs/golden_data_set.csv"):
    """Compare extractions against golden dataset.

    Args:
        extractions: dict mapping document_id -> extracted transaction dict
        golden_path: path to golden CSV
    """
    golden = load_golden(golden_path)
    print(f"Golden dataset: {len(golden)} rows")
    print(f"Extractions: {len(extractions)} documents")

    # Collect per-field values
    field_golden = defaultdict(list)
    field_extracted = defaultdict(list)

    matched = 0
    for row in golden:
        doc_id = row.get("document_id", "")
        if doc_id not in extractions:
            continue
        matched += 1
        txn = extractions[doc_id]

        # Direct fields
        for field in DIRECT_FIELDS:
            field_golden[field].append(row.get(field))
            field_extracted[field].append(txn.get(field))

        # Mapped fields
        for csv_col, model_path in FIELD_MAP.items():
            field_golden[csv_col].append(row.get(csv_col))
            field_extracted[csv_col].append(get_nested(txn, model_path))

    print(f"Matched rows: {matched}")
    if matched == 0:
        print("No matching documents found. Ensure document_ids match.")
        return

    # Compute per-field metrics
    print(f"\n{'Field':<50} {'F1':>6} {'Prec':>6} {'Rec':>6} {'Halluc':>7} {'Miss':>6}")
    print("-" * 85)

    all_metrics = {}
    weighted_f1_sum = 0
    weight_sum = 0

    for field in sorted(set(list(field_golden.keys()))):
        metrics = compute_field_metrics(field_golden[field], field_extracted[field])
        all_metrics[field] = metrics

        weight = 3 if field in CRITICAL_FIELDS else 1
        weighted_f1_sum += metrics["f1"] * weight
        weight_sum += weight

        critical_marker = " ***" if field in CRITICAL_FIELDS else ""
        print(
            f"{field:<50} {metrics['f1']:>6.3f} {metrics['precision']:>6.3f} "
            f"{metrics['recall']:>6.3f} {metrics['hallucination_rate']:>7.3f} "
            f"{metrics['miss_rate']:>6.3f}{critical_marker}"
        )

    overall_f1 = weighted_f1_sum / weight_sum if weight_sum > 0 else 0
    print(f"\n{'Overall weighted F1:':<50} {overall_f1:.4f}")
    print("(*** = critical field, 3x weight)")

    return all_metrics


if __name__ == "__main__":
    # Usage: python examples/evaluate.py extractions.json
    if len(sys.argv) < 2:
        print("Usage: python examples/evaluate.py <extractions.json>")
        print("  extractions.json should map document_id -> transaction dict")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        extractions = json.load(f)

    evaluate(extractions)
