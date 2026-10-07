"""Unit tests on small hand-built datasets (no files needed).

Run: python -m unittest discover tests
"""

import unittest

import pandas as pd

from fraud.network import build_graph, find_rings
from fraud.stream import score_stream

T0 = pd.Timestamp("2026-08-01 10:00")


def accounts(*rows):
    return pd.DataFrame([{"account_id": a, "account_type": kind, "home_city": "Chennai",
                          "signup_date": signup} for a, kind, signup in rows])


MERCHANTS = pd.DataFrame([
    {"merchant_id": "G1", "merchant_name": "Grocer", "category": "grocery", "city": "Chennai"},
    {"merchant_id": "E1", "merchant_name": "Gadgets", "category": "electronics", "city": "online"},
    {"merchant_id": "GC", "merchant_name": "GiftHub", "category": "gift_cards", "city": "online"},
    {"merchant_id": "CX", "merchant_name": "CoinSwift", "category": "crypto_exchange", "city": "online"},
    {"merchant_id": "TP", "merchant_name": "Recharge", "category": "utilities", "city": "online"},
])


class Tx:
    def __init__(self):
        self.rows = []

    def add(self, acct, when, amount, merchant="G1", payee="", device="D1", ip="100.1.1.1",
            country="IN", channel="app"):
        self.rows.append({"txn_id": f"T{len(self.rows) + 1:04d}", "timestamp": when,
                          "account_id": acct, "txn_type": "transfer" if payee else "purchase",
                          "channel": channel, "merchant_id": "" if payee else merchant,
                          "payee_account_id": payee, "amount_inr": float(amount),
                          "device_id": device, "ip_address": ip, "city": "Chennai",
                          "country": country})
        return self.rows[-1]["txn_id"]

    def history(self, acct, n=20, device="D1"):
        for i in range(n):
            self.add(acct, T0 + pd.Timedelta(days=i, hours=i % 5), 500 + 37 * i, device=device)

    def frame(self):
        return pd.DataFrame(self.rows).sort_values("timestamp", kind="stable").reset_index(drop=True)


def score(tx, accts):
    s = score_stream(tx.frame(), accts, MERCHANTS)
    return s.set_index("txn_id")


def codes(row):
    return {r["code"] for r in row["reasons"]}


class RealTimeTests(unittest.TestCase):
    accts = accounts(("A1", "personal", "2024-01-01"), ("B1", "business", "2024-01-01"),
                     ("B2", "business", "2026-06-01"), ("B3", "business", "2024-01-01"),
                     ("B4", "business", "2024-01-01"))

    def test_normal_payment_has_no_risk(self):
        tx = Tx()
        tx.history("A1")
        last = tx.add("A1", T0 + pd.Timedelta(days=25), 700)
        self.assertEqual(score(tx, self.accts).loc[last, "realtime_risk"], 0)

    def test_takeover_new_device_abroad_at_night_is_held(self):
        tx = Tx()
        tx.history("A1")
        t = tx.add("A1", pd.Timestamp("2026-08-26 03:10"), 9500, merchant="GC", device="DX",
                   ip="203.0.113.7", country="AE")
        row = score(tx, self.accts).loc[t]
        self.assertGreaterEqual(row["realtime_risk"], 0.5)
        self.assertTrue({"NEW_DEVICE_ABROAD", "NIGHT_ACTIVITY", "FIRST_RISKY_CATEGORY"} <= codes(row))

    def test_travel_on_known_device_is_not_flagged(self):
        tx = Tx()
        tx.history("A1")
        t = tx.add("A1", T0 + pd.Timedelta(days=25, hours=4), 3000, device="D1",
                   ip="203.0.113.9", country="SG")
        self.assertLess(score(tx, self.accts).loc[t, "realtime_risk"], 0.3)

    def test_card_testing_burst_then_cashout(self):
        tx = Tx()
        tx.history("A1")
        start = T0 + pd.Timedelta(days=25)
        tiny = [tx.add("A1", start + pd.Timedelta(minutes=i), 5, merchant="TP", device="DX")
                for i in range(6)]
        big = tx.add("A1", start + pd.Timedelta(minutes=15), 40000, merchant="E1", device="DX")
        s = score(tx, self.accts)
        self.assertIn("CARD_TEST_BURST", codes(s.loc[tiny[-1]]))
        self.assertIn("CARD_TEST_CASHOUT", codes(s.loc[big]))
        self.assertGreaterEqual(s.loc[big, "realtime_risk"], 0.8)
        # earlier tiny payments are reported so they can be flagged retroactively
        self.assertTrue(set(tiny[:3]) <= set(s.loc[tiny[-1], "burst_members"]))

    def test_business_paying_new_supplier_is_not_flagged(self):
        tx = Tx()
        for b in ("B1", "B3", "B4"):
            tx.history(b, device=f"D{b}")
        for i, b in enumerate(("B1", "B3", "B4")):  # young business B2 collects from three payers
            last = tx.add(b, T0 + pd.Timedelta(days=22, hours=i), 150000, payee="B2", device=f"D{b}")
        self.assertLess(score(tx, self.accts).loc[last, "realtime_risk"], 0.5)


class NetworkTests(unittest.TestCase):
    def build(self, landlord=False):
        rows = [("M1", "personal", "2026-06-20"), ("L1", "business", "2020-01-01")]
        rows += [(f"R{i}", "personal", "2026-06-01") for i in range(5)]
        rows += [(f"N{i}", "personal", "2023-01-01") for i in range(60)]
        accts = accounts(*rows)
        tx = Tx()
        for i in range(60):  # ordinary customers keep the ring's merchants rare
            tx.add(f"N{i}", T0 + pd.Timedelta(hours=i), 400, device=f"DN{i}", ip=f"100.2.{i}.1")
        payee = "L1" if landlord else "M1"
        for i in range(5):
            d, t = f"DR{i}", T0 + pd.Timedelta(days=10 + i)
            if not landlord:
                for k, m in enumerate(("GC", "E1", "CX")):
                    tx.add(f"R{i}", t + pd.Timedelta(hours=k * 3), 4000, merchant=m, device=d,
                           ip=f"100.3.{i}.1")
            tx.add(f"R{i}", t + pd.Timedelta(hours=10), 12000, payee=payee, device=d, ip=f"100.3.{i}.1")
            if not landlord:  # mule passes the money straight to crypto
                tx.add("M1", t + pd.Timedelta(hours=12), 11500, merchant="CX", device="DM",
                       ip="100.9.9.9")
        return tx.frame(), accts

    def test_sequence_ring_with_mule_is_found(self):
        tx, accts = self.build()
        g, mules = build_graph(tx, accts, MERCHANTS)
        rings = find_rings(g, mules)
        self.assertEqual(list(mules), ["M1"])
        self.assertEqual(len(rings), 1)
        self.assertEqual(set(rings[0]["members"]), {"M1", "R0", "R1", "R2", "R3", "R4"})
        self.assertIn("same_sequence", rings[0]["evidence_kinds"])
        self.assertIn("mule_payee", rings[0]["evidence_kinds"])

    def test_landlord_with_many_tenants_is_not_a_ring(self):
        tx, accts = self.build(landlord=True)
        g, mules = build_graph(tx, accts, MERCHANTS)
        self.assertEqual(mules, {})
        self.assertEqual(find_rings(g, mules), [])


if __name__ == "__main__":
    unittest.main()
