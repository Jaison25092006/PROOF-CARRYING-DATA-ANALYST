"""Network analysis: link accounts through shared evidence and extract fraud rings.

Edges between accounts (each carries the evidence that created it):
  shared_device     a device used by 3+ accounts that were new when they used it
  shared_subnet     an IP /24 range used by 4+ new accounts
  same_sequence     the same ordered trio of rarely-used merchants within 72 hours,
                    repeated by 4+ accounts (no single purchase looks bad)
  mule_payee        senders linked to an account that receives from 3+ accounts and
                    passes the money straight through to gift cards / crypto

Households (old accounts sharing a phone) and landlords (many tenants, but the money
stays put) do not create edges.
"""

from collections import defaultdict
from itertools import combinations

import networkx as nx
import pandas as pd

from .stream import RISKY_CATEGORIES, YOUNG_DAYS

EDGE_WEIGHT = {"shared_device": 0.5, "shared_subnet": 0.3, "same_sequence": 0.6,
               "mule_payee": 0.6}
RARE_MERCHANT_SHARE = 0.10  # merchant used by fewer than 10% of accounts
SEQUENCE_WINDOW_H = 72
MIN_SEQUENCE_ACCOUNTS = 4


def _add(g, a, b, kind, detail, txns):
    if a == b:
        return
    if not g.has_edge(a, b):
        g.add_edge(a, b, evidence=[])
    g[a][b]["evidence"].append({"kind": kind, "detail": detail})
    for n in (a, b):
        g.nodes[n].setdefault("evidence_txns", set()).update(txns.get(n, ()))


def build_graph(tx, accounts, merchants):
    g = nx.Graph()
    g.add_nodes_from(accounts["account_id"])
    signup = pd.to_datetime(accounts.set_index("account_id")["signup_date"])
    tx = tx.assign(age=(tx["timestamp"] - tx["account_id"].map(signup)).dt.days)
    category = merchants.set_index("merchant_id")["category"]
    name = merchants.set_index("merchant_id")["merchant_name"]

    # 1. Shared devices / subnets among accounts that were new when they used them.
    young = tx[(tx["age"] < YOUNG_DAYS) & (tx["device_id"] != "")]
    for kind, col, key, min_n in (("shared_device", "device_id", lambda v: v, 3),
                                  ("shared_subnet", "ip_address", lambda v: v.rsplit(".", 1)[0], 4)):
        groups = defaultdict(lambda: defaultdict(list))
        for t in young.itertuples(index=False):
            groups[key(getattr(t, col))][t.account_id].append(t.txn_id)
        for value, accts in groups.items():
            if len(accts) >= min_n:
                label = f"{value}.x" if kind == "shared_subnet" else value
                for a, b in combinations(sorted(accts), 2):
                    _add(g, a, b, kind, f"{kind.replace('_', ' ')} {label} "
                                        f"({len(accts)} new accounts)", accts)

    # 2. Same ordered trio of rare merchants within 72h, repeated across accounts.
    purchases = tx[tx["merchant_id"] != ""]
    share = purchases.groupby("merchant_id")["account_id"].nunique() / tx["account_id"].nunique()
    rare = set(share[share < RARE_MERCHANT_SHARE].index)
    seqs = defaultdict(dict)  # (m1, m2, m3) -> account -> txn ids
    for acct, p in purchases.groupby("account_id"):
        rows = list(p[["timestamp", "merchant_id", "txn_id"]].itertuples(index=False))
        for i, (t1, m1, x1) in enumerate(rows):
            window = [r for r in rows[i + 1:] if (r[0] - t1).total_seconds() <= SEQUENCE_WINDOW_H * 3600]
            for (t2, m2, x2), (t3, m3, x3) in combinations(window, 2):
                trio = (m1, m2, m3)
                if len(set(trio)) == 3 and sum(m in rare for m in trio) >= 2:
                    seqs[trio].setdefault(acct, [x1, x2, x3])
    for trio, accts in seqs.items():
        if len(accts) >= MIN_SEQUENCE_ACCOUNTS:
            label = " → ".join(name[m] for m in trio)
            for a, b in combinations(sorted(accts), 2):
                _add(g, a, b, "same_sequence", f"same purchase sequence {label} "
                                               f"({len(accts)} accounts)", accts)

    # 3. Mule payees: collect from 3+ senders and pass most of it to risky merchants.
    transfers = tx[tx["payee_account_id"] != ""]
    risky = purchases[purchases["merchant_id"].map(category).isin(RISKY_CATEGORIES)]
    mules = {}
    for payee, inc in transfers.groupby("payee_account_id"):
        if inc["account_id"].nunique() < 3:
            continue
        out = risky[risky["account_id"] == payee]
        passed = 0.0
        for t in inc.itertuples(index=False):
            after = out[(out["timestamp"] >= t.timestamp) &
                        (out["timestamp"] <= t.timestamp + pd.Timedelta(hours=24))]
            passed += min(after["amount_inr"].sum(), t.amount_inr)
        ratio = passed / inc["amount_inr"].sum()
        if ratio >= 0.6:
            mules[payee] = {"senders": int(inc["account_id"].nunique()),
                            "received": float(inc["amount_inr"].sum()), "pass_through": ratio}
            txns = {payee: list(out["txn_id"])}
            for sender, s in inc.groupby("account_id"):
                txns[sender] = list(s["txn_id"])
                _add(g, sender, payee, "mule_payee",
                     f"sent ₹{s['amount_inr'].sum():,.0f} to {payee}, which passed "
                     f"{ratio:.0%} of its inflows to gift cards/crypto within 24h", txns)
    return g, mules


def describe_pattern(kinds, size):
    """Name the fraud pattern from the kinds of evidence that link the ring."""
    parts = []
    if "same_sequence" in kinds:
        parts.append("Coordinated purchase ring: accounts in different places, on different "
                     "devices, repeat the same unusual purchase sequence")
    if "shared_device" in kinds or "shared_subnet" in kinds:
        parts.append("Device farm: newly opened (likely synthetic) accounts operated from the "
                     "same phones / network")
    if "mule_payee" in kinds and not parts:
        parts.append(f"Mule cash-out: {size - 1} accounts sent money to one mule account, "
                     "consistent with compromised accounts being drained")
    elif "mule_payee" in kinds:
        parts.append("money collected by a mule that passes it straight on to gift cards / crypto")
    return "; ".join(parts)


def find_rings(g, mules):
    rings = []
    linked = g.edge_subgraph([e for e in g.edges if g.edges[e]["evidence"]])
    for i, comp in enumerate(sorted(nx.connected_components(linked), key=len, reverse=True)):
        if len(comp) < 3:
            continue
        kinds = defaultdict(int)
        details = set()
        for a, b in linked.subgraph(comp).edges:
            for ev in g[a][b]["evidence"]:
                kinds[ev["kind"]] += 1
                details.add(ev["detail"])
        score = 1.0
        for kind in kinds:
            score *= 1 - EDGE_WEIGHT[kind]
        edges = [{"a": a, "b": b,
                  "kinds": sorted({ev["kind"] for ev in g[a][b]["evidence"]}),
                  "evidence": [dict(t) for t in {tuple(ev.items()) for ev in g[a][b]["evidence"]}]}
                 for a, b in linked.subgraph(comp).edges]
        rings.append({
            "ring_id": f"RING-{len(rings) + 1}", "members": sorted(comp), "size": len(comp),
            "pattern": describe_pattern(kinds, len(comp)),
            "edges": edges,
            "score": round(1 - score, 3), "evidence_kinds": dict(kinds),
            "evidence": sorted(details),
            "mules": sorted(m for m in comp if m in mules),
            "evidence_txns": sorted(set().union(*(g.nodes[n].get("evidence_txns", set())
                                                  for n in comp))),
        })
    return rings
