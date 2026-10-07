"""Fraud Intelligence demo app (HackNex 2026, HNX26PSI04).

Run: streamlit run app.py   (after `python -m fraud.pipeline`)
"""

import json
from pathlib import Path

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fraud.pipeline import TXN_ACTIONS, action, load
from fraud.stream import REASONS, score_stream

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"

st.set_page_config(page_title="Fraud Intelligence", page_icon="🛡️", layout="wide")

ACTION_COLOR = {"BLOCK": "#d62728", "HOLD + step-up verification": "#ff7f0e", "REVIEW": "#e6b800",
                "ALLOW": "#2ca02c", "FREEZE account and investigate ring": "#d62728",
                "RESTRICT transfers, verify identity": "#ff7f0e", "MONITOR": "#e6b800",
                "NO ACTION": "#2ca02c"}
EDGE_STYLE = {"same_sequence": ("Same purchase sequence", "#9467bd"),
              "shared_device": ("Shared phone", "#d62728"),
              "shared_subnet": ("Shared IP range", "#ff7f0e"),
              "mule_payee": ("Paid the mule", "#1f77b4")}
EXTRA_WEIGHTS = {"ANOMALY": 0.20, "RING_EVIDENCE": 0.60, "CONTROLLED_ACCOUNT": 0.60}


@st.cache_data
def load_outputs():
    txn = pd.read_csv(OUT / "transaction_scores.csv", keep_default_na=False)
    txn["timestamp"] = pd.to_datetime(txn["timestamp"])
    acc = pd.read_csv(OUT / "account_scores.csv", keep_default_na=False)
    rings = json.loads((OUT / "rings.json").read_text("utf-8"))
    merchants = pd.read_csv(ROOT / "data" / "merchants.csv", dtype=str)
    txn["merchant"] = txn["merchant_id"].map(merchants.set_index("merchant_id")["merchant_name"]) \
        .fillna("")
    card_path = ROOT / "eval" / "scorecard.json"
    card = json.loads(card_path.read_text("utf-8")) if card_path.exists() else None
    return txn, acc, rings, merchants, card


@st.cache_data
def load_raw():
    return load()


def badge(name):
    color = ACTION_COLOR.get(name, "#888")
    return (f"<span style='background:{color};color:white;padding:2px 10px;border-radius:12px;"
            f"font-weight:600'>{name}</span>")


def reasons_of(explanation):
    """'CODE: evidence; CODE: evidence' -> [(code, weight, evidence)]."""
    out = []
    for part in explanation.split("; "):
        code, _, evidence = part.partition(": ")
        if code in REASONS or code in EXTRA_WEIGHTS or code == "CARD_TEST_BURST":
            weight = REASONS[code][0] if code in REASONS else EXTRA_WEIGHTS.get(code, 0.7)
            out.append((code, weight, evidence))
    return out


def show_reasons(explanation):
    rs = reasons_of(explanation)
    if not rs:
        st.success("No risk factors: consistent with this account's normal behaviour.")
        return
    for code, weight, evidence in sorted(rs, key=lambda r: -r[1]):
        st.markdown(f"- **{code}** · weight {weight:.2f}: {evidence}")
    st.caption("Risk = 1 − ∏(1 − weight): each independent warning sign raises the risk; "
               "no single weak signal can block a payment on its own.")


def ring_figure(ring, acc):
    """Accounts plus the things they share (a phone, an IP range, a purchase sequence) as
    hub nodes; money sent to the mule as direct links. Avoids an unreadable all-pairs web."""
    risk = acc.set_index("account_id")["risk"]
    g = nx.Graph()
    hubs = {}  # hub node -> (kind, label)
    for e in ring["edges"]:
        for ev in e["evidence"]:
            if ev["kind"] == "mule_payee":
                g.add_edge(e["a"], e["b"], kind="mule_payee")
            else:
                label = ev["detail"].split(" (")[0].replace("same purchase sequence ", "")
                label = label.replace("shared device ", "📱 ").replace("shared subnet ", "🌐 ")
                hub = f"hub:{label}"
                hubs[hub] = (ev["kind"], label)
                g.add_edge(e["a"], hub, kind=ev["kind"])
                g.add_edge(e["b"], hub, kind=ev["kind"])
    # Pin the mule(s) on the left and the shared-evidence hubs on the right so they don't overlap.
    anchors = {m: (-0.7, 0.4 * i) for i, m in enumerate(ring["mules"]) if m in g}
    anchors.update({h: (0.7, 0.8 * (i - (len(hubs) - 1) / 2)) for i, h in enumerate(hubs)})
    pos = nx.spring_layout(g, seed=7, k=1.6 / max(len(g) ** 0.5, 1), iterations=200,
                           pos=anchors or None, fixed=list(anchors) or None)
    fig = go.Figure()
    for kind, (label, color) in EDGE_STYLE.items():
        xs, ys = [], []
        for a, b, d in g.edges(data=True):
            if d["kind"] == kind:
                xs += [pos[a][0], pos[b][0], None]
                ys += [pos[a][1], pos[b][1], None]
        if xs:
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=label, hoverinfo="skip",
                                     line=dict(color=color, width=2)))
    if hubs:
        fig.add_trace(go.Scatter(
            x=[pos[h][0] for h in hubs], y=[pos[h][1] for h in hubs], mode="markers+text",
            text=[lbl for _, lbl in hubs.values()], textposition="bottom left",
            name="Shared evidence", hoverinfo="text", hovertext=[lbl for _, lbl in hubs.values()],
            marker=dict(size=16, symbol="square", color=[EDGE_STYLE[k][1] for k, _ in hubs.values()],
                        line=dict(color="#333", width=1))))
    nodes = [n for n in g.nodes if n not in hubs]
    is_mule = [n in ring["mules"] for n in nodes]
    fig.add_trace(go.Scatter(
        x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes], mode="markers+text",
        text=nodes, textposition="top center", name="Account (◆ = mule)",
        marker=dict(size=[32 if m else 18 for m in is_mule],
                    symbol=["diamond" if m else "circle" for m in is_mule],
                    color=[risk.get(n, 0) for n in nodes], colorscale="Reds", cmin=0, cmax=1,
                    line=dict(color="#333", width=1),
                    colorbar=dict(title="Account risk", x=1.02)),
        hovertext=[f"{n}<br>risk {risk.get(n, 0):.2f}{'<br>MULE: receives and passes money on' if m else ''}"
                   for n, m in zip(nodes, is_mule)], hoverinfo="text"))
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10),
                      xaxis=dict(visible=False), yaxis=dict(visible=False),
                      legend=dict(orientation="h", y=-0.05))
    return fig


# --------------------------------------------------------------------------- #

txn, acc, rings, merchants, card = load_outputs()
flagged = txn[txn["risk"] >= 0.5]

with st.sidebar:
    st.title("🛡️ Fraud Intelligence")
    st.caption("HackNex 2026 · HNX26PSI04 · Real-Time Financial Fraud Intelligence")
    st.markdown(
        "**Three layers**\n"
        "1. **Real-time:** every payment checked against the customer's own history\n"
        "2. **Anomaly model:** Isolation Forest trained on normal behaviour\n"
        "3. **Network:** links accounts through shared phones, IPs, purchase "
        "sequences and mules to find **rings**\n\n"
        "Every score comes with its reasons and a recommended action.")
    st.caption("Synthetic, fictional data, generated by data_gen/generate.py.")

tabs = st.tabs(["Overview", "Fraud rings", "Accounts", "Transactions", "Live check"])

# ---------------- Overview ----------------
with tabs[0]:
    c = st.columns(5)
    c[0].metric("Transactions scored", f"{len(txn):,}")
    c[1].metric("Flagged (hold / block)", f"{len(flagged):,}")
    c[2].metric("Fraud rings", len(rings))
    c[3].metric("Accounts to freeze", int((acc["action"].str.startswith("FREEZE")).sum()))
    c[4].metric("Money stopped", f"₹{flagged['amount_inr'].sum() / 1e5:,.1f} lakh")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Flagged payments over time")
        daily = flagged.groupby([flagged["timestamp"].dt.date, "action"]).size().reset_index(name="n")
        fig = go.Figure([go.Bar(x=d["timestamp"], y=d["n"], name=a,
                                marker_color=ACTION_COLOR.get(a))
                         for a, d in daily.groupby("action")])
        fig.update_layout(barmode="stack", height=320, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width="stretch")
    with right:
        st.subheader("Decisions")
        counts = txn["action"].value_counts()
        st.dataframe(pd.DataFrame({"action": counts.index, "transactions": counts.values}),
                     hide_index=True, width="stretch")
        big = txn[(txn["amount_inr"] > 100000) & (txn["risk"] < 0.3)]
        st.info(f"**Big ≠ fraud:** {len(big)} payments over ₹1 lakh were allowed because they "
                f"matched their customers' normal behaviour.")

    if card:
        st.subheader("Scorecard against hidden labels")
        f = card["transactions"]["final (real-time + network)"]
        a = card["accounts"]["flagged (risk >= 0.5)"]
        n = card["transactions"]["naive baseline: flag top 1% amounts"]
        rt = card["transactions"]["real-time only"]
        st.dataframe(pd.DataFrame([
            {"method": "Our system (real-time + network)", "precision": f["precision"],
             "recall": f["recall"], "false alarms": f["fp"]},
            {"method": "Real-time layer only", "precision": rt["precision"],
             "recall": rt["recall"], "false alarms": rt["fp"]},
            {"method": "Naive: flag the biggest 1% of payments", "precision": n["precision"],
             "recall": n["recall"], "false alarms": n["fp"]},
        ]), hide_index=True, width="stretch")
        s1, s2, s3 = st.columns(3)
        s1.metric("Fraud accounts caught", f"{a['tp']} / {a['tp'] + a['fn']}")
        s2.metric("Innocent accounts flagged", a["fp"])
        s3.metric("Rings recovered exactly",
                  f"{sum(r['jaccard'] == 1 for r in card['rings'])} / {len(card['rings'])}")
        st.caption("False alarms on innocent look-alikes: " + ", ".join(
            f"{k.replace('_', ' ')} {v['flagged']}/{v['transactions']}"
            for k, v in card["false_alarms_on_benign_lookalikes"].items()))

# ---------------- Rings ----------------
with tabs[1]:
    if not rings:
        st.info("No rings detected.")
    else:
        pick = st.radio("Ring", [r["ring_id"] for r in rings], horizontal=True,
                        format_func=lambda rid: next(f"{r['ring_id']} · {r['size']} accounts"
                                                     for r in rings if r["ring_id"] == rid))
        ring = next(r for r in rings if r["ring_id"] == pick)
        m = st.columns(4)
        m[0].metric("Accounts", ring["size"])
        m[1].metric("Ring score", f"{ring['score']:.2f}")
        m[2].metric("Money involved", f"₹{ring['amount_inr']:,.0f}")
        m[3].metric("Mule", ", ".join(ring["mules"]) or "-")
        st.markdown(f"**Pattern found:** {ring.get('pattern', '')}")
        st.markdown(f"**Recommended action:** {badge('BLOCK')} {ring['recommended_action']}",
                    unsafe_allow_html=True)
        left, right = st.columns([3, 2])
        with left:
            st.plotly_chart(ring_figure(ring, acc), width="stretch")
        with right:
            st.subheader("Why these accounts are linked")
            for e in ring["evidence"][:8]:
                st.markdown(f"- {e}")
            if len(ring["evidence"]) > 8:
                st.caption(f"… and {len(ring['evidence']) - 8} more pieces of evidence")
            st.caption(f"Active {ring['first_seen'][:16]} → {ring['last_seen'][:16]}")
        st.subheader("Timeline of the ring's activity")
        ev = txn[txn["txn_id"].isin(ring["evidence_txns"])].copy()
        ev["what"] = ev.apply(lambda r: f"→ {r['payee_account_id']}" if r["payee_account_id"]
                              else r["merchant"], axis=1)
        fig = go.Figure()
        for kind, d in ev.groupby("txn_type"):
            fig.add_trace(go.Scatter(
                x=d["timestamp"], y=d["account_id"], mode="markers", name=kind,
                marker=dict(size=(d["amount_inr"] ** 0.5) / 6 + 6, opacity=0.8),
                hovertext=[f"{r.txn_id}<br>{r.what}<br>₹{r.amount_inr:,.0f}<br>risk {r.risk:.2f}"
                           for r in d.itertuples()], hoverinfo="text"))
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width="stretch")

# ---------------- Accounts ----------------
with tabs[2]:
    actions = st.multiselect("Show actions", list(acc["action"].unique()),
                             default=[a for a in acc["action"].unique() if a != "NO ACTION"])
    view = acc[acc["action"].isin(actions)]
    st.dataframe(view[["account_id", "risk", "action", "ring_id", "n_flagged", "n_transactions",
                       "total_inr"]], hide_index=True, width="stretch", height=280)
    who = st.selectbox("Inspect an account", view["account_id"] if len(view) else acc["account_id"])
    a = acc.set_index("account_id").loc[who]
    st.markdown(f"### {who} · risk {a['risk']:.2f} {badge(a['action'])}", unsafe_allow_html=True)
    for part in a["explanation"].split("; "):
        st.markdown(f"- {part}")
    t = txn[txn["account_id"] == who].sort_values("timestamp")
    fig = go.Figure(go.Scatter(x=t["timestamp"], y=t["risk"], mode="markers",
                               marker=dict(color=t["risk"], colorscale="RdYlGn_r", cmin=0, cmax=1,
                                           size=8),
                               hovertext=[f"{r.txn_id} ₹{r.amount_inr:,.0f} {r.merchant or r.payee_account_id}"
                                          for r in t.itertuples()], hoverinfo="text"))
    fig.update_layout(height=260, yaxis=dict(title="risk", range=[-0.05, 1.05]),
                      margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width="stretch")
    st.dataframe(t[["txn_id", "timestamp", "txn_type", "merchant", "payee_account_id",
                    "amount_inr", "city", "country", "device_id", "risk", "action"]]
                 .sort_values("risk", ascending=False), hide_index=True, width="stretch")

# ---------------- Transactions ----------------
with tabs[3]:
    acts = st.multiselect("Show decisions", [a for _, a in TXN_ACTIONS],
                          default=["BLOCK", "HOLD + step-up verification"])
    view = txn[txn["action"].isin(acts)].sort_values("risk", ascending=False)
    st.dataframe(view[["txn_id", "timestamp", "account_id", "txn_type", "merchant",
                       "payee_account_id", "amount_inr", "city", "country", "risk", "action"]],
                 hide_index=True, width="stretch", height=300)
    if len(view):
        tid = st.selectbox("Explain a transaction", view["txn_id"])
        r = txn.set_index("txn_id").loc[tid]
        st.markdown(f"### {tid} · ₹{r['amount_inr']:,.2f} · risk {r['risk']:.2f} "
                    f"{badge(r['action'])}", unsafe_allow_html=True)
        st.caption(f"{r['timestamp']} · {r['account_id']} · {r['merchant'] or '→ ' + r['payee_account_id']}"
                   f" · {r['city']}, {r['country']} · device {r['device_id'] or 'card in store'}")
        st.caption(f"Real-time score when it happened: {r['realtime_risk']:.2f}; final score after "
                   f"network analysis: {r['risk']:.2f}")
        show_reasons(r["explanation"])

# ---------------- Live check ----------------
with tabs[4]:
    st.markdown("Score a **new payment** in real time against the customer's full history. "
                "This runs the real-time layer exactly as it would when the payment arrives.")
    st.caption("Try it: score a normal payment first, then the same account with "
               "**NEW-DEVICE**, country **AE**, hour **3**, and a **GiftHub Digital** gift card.")
    tx, accounts, merch = load_raw()
    c1, c2, c3 = st.columns(3)
    with c1:
        ids = list(accounts["account_id"])
        who = st.selectbox("Account", ids, index=ids.index("A0002") if "A0002" in ids else 0)
        hist = tx[tx["account_id"] == who]
        devices = sorted(d for d in hist["device_id"].unique() if d)
        device = st.selectbox("Device", (devices or []) + ["NEW-DEVICE (never seen)"])
    with c2:
        kind = st.radio("Type", ["purchase", "transfer"], horizontal=True)
        if kind == "purchase":
            m = st.selectbox("Merchant", merch["merchant_name"] + " (" + merch["category"] + ")")
            merchant_id = merch["merchant_id"].iloc[list(merch["merchant_name"] + " (" +
                                                         merch["category"] + ")").index(m)]
            payee = ""
        else:
            payee = st.selectbox("Payee", [a for a in accounts["account_id"] if a != who])
            merchant_id = ""
        amount = st.number_input("Amount (₹)", min_value=1.0, value=2500.0, step=500.0)
    with c3:
        country = st.selectbox("Country", ["IN", "AE", "SG", "GB", "DE", "TH"])
        hour = st.slider("Hour of day", 0, 23, 14)
        st.caption(f"{who} usually spends a median of ₹{hist['amount_inr'].median():,.0f} "
                   f"over {len(hist)} past payments.")
    if st.button("Score this payment", type="primary"):
        when = tx["timestamp"].max().normalize() + pd.Timedelta(days=1, hours=hour)
        new = {"txn_id": "LIVE-1", "timestamp": when, "account_id": who, "txn_type": kind,
               "channel": "app", "merchant_id": merchant_id, "payee_account_id": payee,
               "amount_inr": float(amount),
               "device_id": "D-LIVE" if device.startswith("NEW") else device,
               "ip_address": "203.0.113.7" if country != "IN" else
               (hist["ip_address"][hist["ip_address"] != ""].iloc[-1]
                if (hist["ip_address"] != "").any() else "100.1.1.1"),
               "city": "", "country": country}
        with st.spinner("Scoring against the full transaction history…"):
            scored = score_stream(pd.concat([tx, pd.DataFrame([new])], ignore_index=True),
                                  accounts, merch)
        row = scored.iloc[-1]
        decision = action(row["realtime_risk"], TXN_ACTIONS)
        st.markdown(f"### Risk {row['realtime_risk']:.2f} {badge(decision)}",
                    unsafe_allow_html=True)
        show_reasons("; ".join(f"{r['code']}: {r['evidence']}" for r in row["reasons"]))
