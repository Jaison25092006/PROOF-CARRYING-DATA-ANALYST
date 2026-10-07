"""Print a readable summary of the detector's output.

Run: python -m fraud.pipeline && python eval/evaluate.py && python report.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
LINE = "=" * 90


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    acc = pd.read_csv(OUT / "account_scores.csv", keep_default_na=False)
    txn = pd.read_csv(OUT / "transaction_scores.csv", keep_default_na=False)
    rings = json.loads((OUT / "rings.json").read_text("utf-8"))

    print(f"{LINE}\nFRAUD RINGS DETECTED\n{LINE}")
    for r in rings:
        print(f"\n{r['ring_id']}  |  {r['size']} accounts  |  ring score {r['score']}  |  "
              f"₹{r['amount_inr']:,.0f} involved")
        print(f"  pattern  : {r.get('pattern', '')}")
        print(f"  members  : {', '.join(r['members'])}")
        print(f"  mule     : {', '.join(r['mules']) or '-'}")
        print(f"  period   : {r['first_seen']}  →  {r['last_seen']}")
        print(f"  linked by: {', '.join(f'{k} ({v})' for k, v in r['evidence_kinds'].items())}")
        for e in r["evidence"][:2]:
            print(f"    - {e}")
        print(f"  ACTION   : {r['recommended_action']}")

    print(f"\n{LINE}\nTOP 10 RISKIEST ACCOUNTS\n{LINE}")
    for a in acc.head(10).itertuples():
        print(f"{a.account_id}  risk {a.risk:.2f}  {a.action:38}  {a.ring_id or '-':7}  "
              f"{a.n_flagged} flagged txns")

    print(f"\n{LINE}\nEXAMPLE TRANSACTION EXPLANATIONS\n{LINE}")
    for title, code in [("Account takeover", "NEW_DEVICE_ABROAD"),
                        ("Card testing", "CARD_TEST_CASHOUT"),
                        ("Mule cash-out", "PASS_THROUGH")]:
        hits = txn[txn["explanation"].str.contains(code)]
        if hits.empty:
            continue
        t = hits.nlargest(1, "risk").iloc[0]
        print(f"\n[{title}] {t.txn_id}  {t.timestamp}  {t.account_id}  ₹{t.amount_inr:,.2f}  "
              f"{t.city}, {t.country}")
        print(f"  risk {t.risk:.2f}  →  {t.action}")
        for part in t.explanation.split("; ")[:4]:
            print(f"    • {part}")
    big = txn[(txn["amount_inr"] > 100000) & (txn["risk"] < 0.3)]
    if not big.empty:
        t = big.nlargest(1, "amount_inr").iloc[0]
        print(f"\n[Big but legitimate] {t.txn_id}  {t.account_id}  ₹{t.amount_inr:,.2f}  "
              f"risk {t.risk:.2f}  →  {t.action}")
        print(f"    • {t.explanation}")

    card_path = ROOT / "eval" / "scorecard.json"
    if card_path.exists():
        c = json.loads(card_path.read_text("utf-8"))
        f = c["transactions"]["final (real-time + network)"]
        a = c["accounts"]["flagged (risk >= 0.5)"]
        n = c["transactions"]["naive baseline: flag top 1% amounts"]
        legit = len(txn) - f["tp"] - f["fn"]
        lookalikes = ", ".join(f"{k} {v['flagged']}/{v['transactions']}"
                               for k, v in c["false_alarms_on_benign_lookalikes"].items())
        print(f"\n{LINE}\nSCORECARD\n{LINE}")
        print(f"Transactions : precision {f['precision']:.1%}, recall {f['recall']:.1%}, "
              f"false alarms {f['fp']} of {legit:,} legitimate")
        print(f"Accounts     : precision {a['precision']:.1%}, recall {a['recall']:.1%} "
              f"({a['tp']} of {a['tp'] + a['fn']} fraud accounts caught)")
        print(f"Rings        : " + ", ".join(f"{r['true_ring']} matched {r['matched']} "
                                             f"(overlap {r['jaccard']:.0%})" for r in c["rings"]))
        print(f"Look-alikes  : flagged {lookalikes}")
        print(f"Naive 'flag the biggest amounts': precision {n['precision']:.0%}, "
              f"recall {n['recall']:.0%}")


if __name__ == "__main__":
    main()
