from __future__ import annotations

import json
import sys
from pathlib import Path

from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rgel.utils import write_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT.parent.parent / "outputs"


def main() -> None:
    payload = json.loads(
        (OUTPUT_ROOT / "risk_constrained_coverage.json").read_text(
            encoding="utf-8"
        )
    )
    rows = []
    for target in payload["target_risks"]:
        selected = [
            row
            for row in payload["rows"]
            if float(row["target_risk"]) == float(target)
        ]
        pm_only = sum(
            row["pmrc_test_risk"] > target and row["msp_test_risk"] <= target
            for row in selected
        )
        msp_only = sum(
            row["msp_test_risk"] > target and row["pmrc_test_risk"] <= target
            for row in selected
        )
        discordant = pm_only + msp_only
        p_value = (
            binomtest(pm_only, discordant, 0.5).pvalue
            if discordant
            else 1.0
        )
        rows.append(
            {
                "target_risk": target,
                "n_runs": len(selected),
                "pmrc_only_violation": pm_only,
                "msp_only_violation": msp_only,
                "discordant_runs": discordant,
                "mcnemar_exact_p": float(p_value),
            }
        )
    write_json(OUTPUT_ROOT / "risk_violation_paired.json", {"rows": rows})
    print(rows)


if __name__ == "__main__":
    main()
