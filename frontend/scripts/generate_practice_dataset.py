"""
Generate the synthetic dataset that ships with the /practice sample brief.

The brief at /practice is "Restaurant Inspection Risk Scoring" — a regional health
department with 6 years of inspection records, 60+ columns, target = critical violation
within 90 days. We can't ship a real dataset, but a synthetic CSV that *looks* like
real inspection data lets the learner actually run pandas / sklearn against it while
working through the brief on paper.

Run:
    python frontend/scripts/generate_practice_dataset.py
Output:
    frontend/public/datasets/restaurant_inspections_sample.csv

The dataset is deliberately compact (~500 rows) so it fits in a static asset.
"""

from __future__ import annotations

import csv
import random
from datetime import date, timedelta
from pathlib import Path

SEED = 42
N_ROWS = 500
CRITICAL_RATE = 0.07

CUISINES = [
    ("American", 0.06),
    ("Mexican", 0.10),
    ("Asian", 0.12),
    ("Pizza", 0.04),
    ("Mediterranean", 0.05),
    ("BBQ", 0.07),
    ("Seafood", 0.09),
    ("Bakery", 0.03),
    ("Cafe", 0.04),
]
OWNERSHIP = ["independent", "chain"]
COUNTIES = [
    "Cook",
    "DuPage",
    "Kane",
    "Lake",
    "Will",
    "McHenry",
    "Kendall",
    "Winnebago",
    "Peoria",
    "Sangamon",
]
COMPLAINTS_NEG = [
    "found rodent droppings near prep area",
    "raw chicken stored above ready-to-eat foods",
    "refrigerator temperature too high; reading 52F",
    "expired dairy in walk-in cooler",
    "soap dispensers empty in employee bathroom",
    "evidence of pest activity in dry storage",
    "cross-contamination risk on cutting boards",
    "improper handwashing observed",
    "thermometer missing in reach-in cooler",
    "food held at unsafe temperatures",
]
COMPLAINTS_NEUTRAL = [
    "no complaints filed in past 90 days",
    "minor signage issue noted on previous visit",
    "routine check; nothing unusual reported",
    "manager certificate expired (renewing)",
    "",
    "",
    "",
]


def main() -> None:
    rng = random.Random(SEED)
    base_date = date(2024, 1, 1)

    # First pass — generate features + a continuous risk signal.
    pre = []
    for i in range(N_ROWS):
        restaurant_id = f"R{i+1:04d}"
        inspection_date = base_date + timedelta(days=rng.randint(0, 365))
        inspector_id = f"I{rng.randint(1, 88):03d}"

        cuisine, cuisine_critical_lift = rng.choice(CUISINES)
        seating_capacity = max(8, int(rng.gauss(60, 35)))
        ownership_type = rng.choice(OWNERSHIP)
        county = rng.choice(COUNTIES)

        days_since_last_inspection = max(60, int(rng.gauss(365, 180)))
        prior_critical_count = max(0, int(rng.expovariate(1.0)))

        if rng.random() < 0.20:
            renovation_disclosed = rng.choice(["Yes", "No"])
        else:
            renovation_disclosed = ""

        neighborhood_median_income = max(28000, int(rng.gauss(62000, 22000)))

        risk = (
            cuisine_critical_lift
            + 0.001 * days_since_last_inspection
            + 0.04 * prior_critical_count
            + (-0.02 if renovation_disclosed == "Yes" else 0.0)
            + (0.03 if renovation_disclosed == "" else 0.0)
            + (-0.01 if ownership_type == "chain" else 0.01)
            + rng.uniform(-0.05, 0.05)
        )

        pre.append(
            {
                "restaurant_id": restaurant_id,
                "inspection_date": inspection_date.isoformat(),
                "inspector_id": inspector_id,
                "cuisine_type": cuisine,
                "seating_capacity": seating_capacity,
                "ownership_type": ownership_type,
                "county": county,
                "days_since_last_inspection": days_since_last_inspection,
                "prior_critical_count": prior_critical_count,
                "renovation_disclosed": renovation_disclosed,
                "neighborhood_median_income": neighborhood_median_income,
                "_risk": risk,
            }
        )

    # Pick the top CRITICAL_RATE × N rows as critical. Everything else is split
    # into ~20% minor / ~80% pass.
    pre_sorted = sorted(pre, key=lambda r: -r["_risk"])
    n_critical = int(round(N_ROWS * CRITICAL_RATE))
    critical_ids = set(r["restaurant_id"] for r in pre_sorted[:n_critical])

    rows = []
    for r in pre:
        is_critical = r["restaurant_id"] in critical_ids
        if is_critical:
            result = "critical"
            complaint = (
                rng.choice(COMPLAINTS_NEG) if rng.random() < 0.7 else rng.choice(COMPLAINTS_NEUTRAL)
            )
        else:
            result = "minor" if rng.random() < 0.18 else "pass"
            complaint = rng.choice(COMPLAINTS_NEUTRAL)

        rows.append(
            {
                "restaurant_id": r["restaurant_id"],
                "inspection_date": r["inspection_date"],
                "result": result,
                "inspector_id": r["inspector_id"],  # KNOWN LEAKER — learners drop this in Phase 2
                "cuisine_type": r["cuisine_type"],
                "seating_capacity": r["seating_capacity"],
                "ownership_type": r["ownership_type"],
                "county": r["county"],
                "days_since_last_inspection": r["days_since_last_inspection"],
                "prior_critical_count": r["prior_critical_count"],
                "renovation_disclosed": r["renovation_disclosed"],
                "complaint_text": complaint,
                "neighborhood_median_income": r["neighborhood_median_income"],
            }
        )

    out_path = (
        Path(__file__).resolve().parent.parent
        / "public"
        / "datasets"
        / "restaurant_inspections_sample.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    n_critical = sum(1 for r in rows if r["result"] == "critical")
    n_missing_renov = sum(1 for r in rows if r["renovation_disclosed"] == "")
    print(f"Wrote {len(rows)} rows to {out_path}")
    print(
        f"  critical rate: {n_critical / len(rows):.1%}  (target ~{CRITICAL_RATE:.0%})"
    )
    print(
        f"  renovation_disclosed missing: {n_missing_renov / len(rows):.0%}"
    )


if __name__ == "__main__":
    main()
