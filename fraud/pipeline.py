"""End-to-end run: real-time scores → anomaly model → network rings → final risk + actions.

Reads data/, writes outputs/:
  transaction_scores.csv   risk, action and reasons for every transaction
  account_scores.csv       risk, action and reasons for every account
  rings.json               detected rings with members, evidence and timeline
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .network import build_graph, find_rings
from .stream import noisy_or, score_stream

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / "data", ROOT / "outputs"

BASELINE_DAYS = 20          # anomaly model learns "normal" from the first 20 days
ANOMALY_WEIGHT = 0.20
TXN_ACTIONS = [(0.80, "BLOCK"), (0.50, "HOLD + step-up verification"), (0.30, "REVIEW"),
               (0.0, "ALLOW")]
ACCOUNT_ACTIONS = [(0.80, "FREEZE account and investigate ring"),
                   (0.50, "RESTRICT transfers, verify identity"), (0.30, "MONITOR"),
                   (0.0, "NO ACTION")]
FLAG_THRESHOLD = 0.50       # HOLD or BLOCK counts as "flagged as fraud"


def action(score, table):
    return next(name for cut, name in table if score >= cut)


def load():
    tx = pd.read_csv(DATA / "transactions.csv", dtype=str, keep_default_na=False)
    tx["timestamp"] = pd.to_datetime(tx["timestamp"])
    tx["amount_inr"] = tx["amount_inr"].astype(float)
    accounts = pd.read_csv(DATA / "accounts.csv", dtype=str)
    merchants = pd.read_csv(DATA / "merchants.csv", dtype=str)
    return tx.sort_values("timestamp", kind="stable").reset_index(drop=True), accounts, merchants


def anomaly_reasons(scored, tx):
    """Isolation Forest trained on the baseline period; flags the most unusual 0.5%."""
    cols = [c for c in scored.columns if c.startswith("f_")]
    X = scored[cols].to_numpy(float)
    base = (tx["timestamp"] < tx["timestamp"].min() + pd.Timedelta(days=BASELINE_DAYS)).to_numpy()
    model = IsolationForest(n_estimators=200, random_state=0).fit(X[base])
    raw = -model.score_samples(X)
    cut = np.quantile(raw[base], 0.995)
    scored["anomaly_score"] = raw
    return raw > cut


def run():
    tx, accounts, merchants = load()
    scored = score_stream(tx, accounts, merchants)
    is_anomaly = anomaly_reasons(scored, tx)

    g, mules = build_graph(tx, accounts, merchants)
    rings = find_rings(g, mules)
    ring_of = {m: r for r in rings for m in r["members"]}
    evidence_txns = {x: r for r in rings for x in r["evidence_txns"]}
    # Accounts that are fraudster-controlled as a whole: mules, and synthetic identities
    # (new accounts opened on shared devices). Everything they do is suspect. Members of
    # behaviour-only rings may be recruited real customers, so only their ring evidence counts.
    controlled = {m: "mule account" for m in mules}
    for r in rings:
        if "shared_device" in r["evidence_kinds"]:
            for m in r["members"]:
                controlled.setdefault(m, f"synthetic identity in {r['ring_id']} "
                                         f"(new account on a shared device)")
    acct_of = dict(zip(tx["txn_id"], tx["account_id"]))

    # Final transaction risk = real-time reasons + anomaly + network evidence.
    # Card testing is only recognisable after a few tiny payments; flag those retroactively.
    burst_member = {x for ids in scored["burst_members"] for x in ids}
    final_reasons = []
    for i, row in scored.iterrows():
        reasons = list(row["reasons"])
        if row["txn_id"] in burst_member and not any(r["code"] == "CARD_TEST_BURST" for r in reasons):
            reasons.append({"code": "CARD_TEST_BURST", "weight": 0.7,
                            "evidence": "one of a burst of tiny test payments at the same merchant "
                                        "(recognised after the burst, flagged retroactively)"})
        if is_anomaly[i]:
            reasons.append({"code": "ANOMALY", "weight": ANOMALY_WEIGHT,
                            "evidence": "unusual combination of amount, time, device and history "
                                        "compared with normal behaviour (Isolation Forest)"})
        ring = evidence_txns.get(row["txn_id"])
        if ring:
            reasons.append({"code": "RING_EVIDENCE", "weight": 0.6 if ring["score"] >= 0.5 else 0.3,
                            "evidence": f"part of the evidence linking {ring['ring_id']} "
                                        f"({ring['size']} accounts): {ring['evidence'][0]}"})
        why = controlled.get(acct_of[row["txn_id"]])
        if why:
            reasons.append({"code": "CONTROLLED_ACCOUNT", "weight": 0.6,
                            "evidence": f"made by a fraudster-controlled account: {why}"})
        final_reasons.append(reasons)
    scored["reasons"] = final_reasons
    scored["risk"] = [round(noisy_or(r), 4) for r in final_reasons]
    scored["realtime_risk"] = scored["realtime_risk"].round(4)
    scored["action"] = scored["risk"].map(lambda s: action(s, TXN_ACTIONS))
    scored["explanation"] = [
        "; ".join(f"{r['code']}: {r['evidence']}" for r in sorted(rs, key=lambda r: -r["weight"]))
        or "no risk factors; consistent with this account's normal behaviour"
        for rs in final_reasons]
    txn = tx.merge(scored.drop(columns=[c for c in scored.columns if c.startswith("f_")]
                               + ["burst_members"]),
                   on="txn_id")

    # Account risk = top transaction risk, ring membership and mule behaviour.
    acc_rows = []
    for acct, t in txn.groupby("account_id"):
        top = t.nlargest(3, "risk")
        reasons = []
        if top["risk"].iloc[0] >= 0.3:
            r0 = top.iloc[0]
            reasons.append({"code": "HIGH_RISK_TRANSACTION", "weight": float(r0["risk"]),
                            "evidence": f"{r0['txn_id']} scored {r0['risk']:.2f} ({r0['action']})"})
        ring = ring_of.get(acct)
        if ring:
            reasons.append({"code": "RING_MEMBER", "weight": ring["score"],
                            "evidence": f"member of {ring['ring_id']} ({ring['size']} accounts, "
                                        f"linked by {', '.join(ring['evidence_kinds'])})"})
        if acct in mules:
            m = mules[acct]
            reasons.append({"code": "MULE", "weight": 0.7,
                            "evidence": f"received ₹{m['received']:,.0f} from {m['senders']} "
                                        f"accounts and passed {m['pass_through']:.0%} straight "
                                        f"to gift cards/crypto"})
        risk = noisy_or(reasons)
        acc_rows.append({
            "account_id": acct, "risk": round(risk, 4),
            "action": action(risk, ACCOUNT_ACTIONS), "ring_id": ring["ring_id"] if ring else "",
            "n_transactions": len(t), "n_flagged": int((t["risk"] >= FLAG_THRESHOLD).sum()),
            "total_inr": round(t["amount_inr"].sum(), 2),
            "explanation": "; ".join(f"{r['code']}: {r['evidence']}" for r in reasons)
                           or "no risk factors"})
    acc = pd.DataFrame(acc_rows).sort_values("risk", ascending=False)

    for r in rings:
        rt = txn[txn["txn_id"].isin(r["evidence_txns"])]
        r["first_seen"], r["last_seen"] = str(rt["timestamp"].min()), str(rt["timestamp"].max())
        r["amount_inr"] = round(float(rt["amount_inr"].sum()), 2)
        r["recommended_action"] = ("FREEZE all member accounts, block payouts to "
                                   + ", ".join(r["mules"]) if r["mules"] else
                                   "FREEZE member accounts and verify identities")

    OUT.mkdir(exist_ok=True)
    txn.drop(columns=["reasons"]).assign(timestamp=txn["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")) \
        .to_csv(OUT / "transaction_scores.csv", index=False)
    acc.to_csv(OUT / "account_scores.csv", index=False)
    (OUT / "rings.json").write_text(json.dumps(rings, indent=2, default=str), encoding="utf-8")
    return txn, acc, rings, g


if __name__ == "__main__":
    import time
    t0 = time.time()
    txn, acc, rings, _ = run()
    print(f"scored {len(txn)} transactions and {len(acc)} accounts in {time.time() - t0:.1f}s; "
          f"{(txn['risk'] >= FLAG_THRESHOLD).sum()} transactions flagged, {len(rings)} rings")
    for r in rings:
        print(f"  {r['ring_id']}: {r['size']} accounts, score {r['score']}, "
              f"{r['evidence_kinds']}, mules {r['mules']}")
