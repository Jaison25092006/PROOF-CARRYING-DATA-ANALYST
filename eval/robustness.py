"""Run the unchanged detector on datasets generated with other seeds.

Regenerates data/ for each seed, scores and evaluates it, then restores the default
seed 2026. Writes eval/robustness.json.

Run: python eval/robustness.py
"""

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [2026, 7, 42, 99]


def run(*args, seed):
    env = {**os.environ, "DATA_SEED": str(seed), "PYTHONIOENCODING": "utf-8"}
    subprocess.run([sys.executable, *args], cwd=ROOT, env=env, check=True,
                   stdout=subprocess.DEVNULL)


def main():
    rows = []
    for seed in SEEDS + [2026]:  # finish on 2026 so data/ and outputs/ are the default
        run("data_gen/generate.py", seed=seed)
        run("-m", "fraud.pipeline", seed=seed)
        run("eval/evaluate.py", seed=seed)
        if len(rows) == len(SEEDS):
            break
        c = json.loads((ROOT / "eval" / "scorecard.json").read_text("utf-8"))
        t = c["transactions"]["final (real-time + network)"]
        a = c["accounts"]["flagged (risk >= 0.5)"]
        rows.append({
            "seed": seed, "role": "development" if seed == 2026 else "unseen",
            "txn_precision": t["precision"], "txn_recall": t["recall"], "txn_false_alarms": t["fp"],
            "fraud_accounts_caught": f"{a['tp']}/{a['tp'] + a['fn']}",
            "innocent_accounts_flagged": a["fp"],
            "rings_exact": f"{sum(r['jaccard'] == 1 for r in c['rings'])}/{len(c['rings'])}",
            "lookalike_false_alarms": sum(v["flagged"] for v in
                                          c["false_alarms_on_benign_lookalikes"].values()),
        })
        r = rows[-1]
        print(f"seed {seed:>4} ({r['role']:11}): precision {r['txn_precision']:.3f}  recall "
              f"{r['txn_recall']:.3f}  false alarms {r['txn_false_alarms']}  accounts "
              f"{r['fraud_accounts_caught']} (+{r['innocent_accounts_flagged']} innocent)  rings "
              f"{r['rings_exact']}  look-alike FPs {r['lookalike_false_alarms']}")
    (ROOT / "eval" / "robustness.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
