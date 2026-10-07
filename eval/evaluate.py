"""Score the detector's outputs against the hidden labels.

Run after `python -m fraud.pipeline`. Prints a scorecard and writes eval/scorecard.json.
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT, EVAL = ROOT / "outputs", ROOT / "eval"
FLAG = 0.50


def prf(flagged, truth):
    tp = int((flagged & truth).sum())
    fp = int((flagged & ~truth).sum())
    fn = int((~flagged & truth).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": round(p, 3), "recall": round(r, 3),
            "f1": round(2 * p * r / (p + r), 3) if p + r else 0.0, "tp": tp, "fp": fp, "fn": fn}


def main():
    txn = pd.read_csv(OUT / "transaction_scores.csv", keep_default_na=False)
    acc = pd.read_csv(OUT / "account_scores.csv", keep_default_na=False)
    rings = json.loads((OUT / "rings.json").read_text("utf-8"))
    tl = pd.read_csv(EVAL / "transaction_labels.csv", keep_default_na=False)
    al = pd.read_csv(EVAL / "account_labels.csv", keep_default_na=False)
    t = txn.merge(tl, on="txn_id")
    a = acc.merge(al, on="account_id")
    truth = t["is_fraud"].astype(str) == "True"
    a_truth = a["is_fraud"].astype(str) == "True"

    card = {
        "transactions": {
            "final (real-time + network)": prf(t["risk"] >= FLAG, truth),
            "real-time only": prf(t["realtime_risk"] >= FLAG, truth),
            "naive baseline: flag top 1% amounts":
                prf(t["amount_inr"] >= t["amount_inr"].quantile(0.99), truth),
            "roc_auc": round(roc_auc_score(truth, t["risk"]), 4),
            "average_precision": round(average_precision_score(truth, t["risk"]), 4),
            "false_positive_rate": round(float(((t["risk"] >= FLAG) & ~truth).sum() / (~truth).sum()), 5),
        },
        "recall_by_pattern": {
            p: round(float((g["risk"] >= FLAG).mean()), 3)
            for p, g in t[truth].groupby("pattern")},
        "false_alarms_on_benign_lookalikes": {
            tag: {"transactions": len(g), "flagged": int((g["risk"] >= FLAG).sum())}
            for tag, g in t[~truth & (t["benign_tag"] != "")].groupby("benign_tag")},
        "accounts": {
            "flagged (risk >= 0.5)": prf(a["risk"] >= FLAG, a_truth),
            "roc_auc": round(roc_auc_score(a_truth, a["risk"]), 4),
            f"precision_at_{int(a_truth.sum())}": round(float(
                a.nlargest(int(a_truth.sum()), "risk")["is_fraud"].astype(str).eq("True").mean()), 3),
        },
        "rings": [],
    }
    true_rings = al[al["ring_id"] != ""].groupby("ring_id")["account_id"].apply(set).to_dict()
    for rid, members in true_rings.items():
        best = max(rings, key=lambda r: len(members & set(r["members"])) /
                   len(members | set(r["members"])), default=None)
        jac = len(members & set(best["members"])) / len(members | set(best["members"])) if best else 0
        card["rings"].append({"true_ring": rid, "size": len(members),
                              "matched": best["ring_id"] if best and jac > 0 else None,
                              "jaccard": round(jac, 3)})

    (EVAL / "scorecard.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    print(json.dumps(card, indent=2))


if __name__ == "__main__":
    main()
