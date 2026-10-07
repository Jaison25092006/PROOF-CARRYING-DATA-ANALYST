"""Generate a synthetic payments dataset with planted fraud rings and benign look-alikes.

Everything is fictional and driven by a fixed seed. The detector only ever sees
data/; the labels live in eval/ and are used for scoring.

Fraud patterns (label `pattern`):
  sequence_ring   10 accounts in different cities, devices and IPs buy the same unusual
                  item sequence (gift card -> electronics -> crypto), then send money to
                  the same mule. No single transaction looks bad.
  device_farm     8 brand-new accounts share 2 devices and one IP subnet, make quick
                  marketplace purchases and transfer to a mule.
  account_takeover  5 established accounts: new device, foreign IP, 1-4 am, gift cards,
                  electronics, and a transfer to a new payee within two hours.
  card_testing    3 accounts: a burst of tiny payments at one merchant from a new device,
                  then one large purchase.
  mule_cashout    the 3 mule accounts forward what they receive to a crypto exchange
                  within hours.

Benign look-alikes (label `tag`), which a good detector must NOT flag:
  high_value      one-off big purchases (laptops, flights) on the customer's own device
  business        business accounts with large, frequent supplier payments
  travel          customers abroad using their usual device
  shared_device   households sharing one phone and Wi-Fi
  rent            ~60 tenants paying the same 3 property managers every month (a benign
                  "many accounts pay one place" hub)
  crypto_hobby    normal customers who occasionally buy gift cards or crypto
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

SEED = int(os.environ.get("DATA_SEED", 2026))  # change to test on an unseen dataset
ROOT = Path(__file__).resolve().parents[1]
DATA, EVAL = ROOT / "data", ROOT / "eval"
rng = np.random.default_rng(SEED)

START = pd.Timestamp("2026-08-01")
DAYS = 60

DOMESTIC = ["Chennai", "Coimbatore", "Bengaluru", "Mumbai", "Delhi", "Hyderabad", "Kochi", "Pune"]
ABROAD = [("Singapore", "SG"), ("Dubai", "AE"), ("London", "GB"), ("Frankfurt", "DE"),
          ("Bangkok", "TH")]

# category: (min INR, max INR, weight for a normal customer, in-store?)
CATEGORIES = {
    "grocery": (200, 3000, 30, True), "restaurants": (150, 2500, 20, True),
    "fuel": (500, 4000, 10, True), "pharmacy": (100, 2000, 6, True),
    "fashion": (500, 6000, 7, True), "online_marketplace": (300, 12000, 15, False),
    "utilities": (100, 5000, 8, False), "electronics": (1500, 25000, 2, False),
    "travel": (2000, 20000, 2, False), "gift_cards": (500, 5000, 0, False),
    "crypto_exchange": (2000, 20000, 0, False),
}
NAMES = {
    "grocery": ["FreshBasket", "DailyMart", "GreenLeaf Grocers", "Annapoorna Stores"],
    "restaurants": ["Spice Route", "Dosa Corner", "Urban Bites", "Tiffin Box"],
    "fuel": ["CityFuel", "HighwayPetro", "QuickFill"],
    "pharmacy": ["CarePlus Pharmacy", "MediQuick", "HealthFirst"],
    "fashion": ["Threads & Co", "StyleLane", "Kurta House"],
    "online_marketplace": ["ShopSphere", "BuyNest", "CartKart"],
    "utilities": ["PowerGrid Pay", "QuickPay Recharge", "AquaBill"],
    "electronics": ["VoltX Gadgets", "MegaByte Store", "CircuitHub"],
    "travel": ["SkyTrail Travels", "RailEase", "StayWell Hotels"],
    "gift_cards": ["GiftHub Digital", "CardVault"],
    "crypto_exchange": ["CoinSwift Exchange", "BitHarbor"],
}


def ts(day, hour, minute=None):
    minute = rng.integers(0, 60) if minute is None else minute
    return START + pd.Timedelta(days=float(day)) + pd.Timedelta(hours=float(hour),
                                                                minutes=float(minute))


def domestic_ip(city):
    return f"{100 + DOMESTIC.index(city)}.{rng.integers(1, 255)}.{rng.integers(1, 255)}.{rng.integers(1, 255)}"


def abroad_ip(i):
    return f"{200 + i}.{rng.integers(1, 255)}.{rng.integers(1, 255)}.{rng.integers(1, 255)}"


# --------------------------------------------------------------------------- #
# Entities
# --------------------------------------------------------------------------- #

def build_merchants():
    rows = []
    for cat, names in NAMES.items():
        in_store = CATEGORIES[cat][3]
        for name in names:
            rows.append({"merchant_id": f"M{len(rows) + 1:03d}", "merchant_name": name,
                         "category": cat,
                         "city": str(rng.choice(DOMESTIC)) if in_store else "online"})
    return pd.DataFrame(rows)


class World:
    def __init__(self):
        self.merchants = build_merchants()
        self.by_name = dict(zip(self.merchants["merchant_name"], self.merchants["merchant_id"]))
        self.by_cat = self.merchants.groupby("category")["merchant_id"].apply(list).to_dict()
        self.category = dict(zip(self.merchants["merchant_id"], self.merchants["category"]))
        self.accounts, self.txns = [], []
        self.n_devices = 0

    def device(self):
        self.n_devices += 1
        return f"D{self.n_devices:04d}"

    def account(self, kind="personal", role="normal", ring="", signup_day=None, city=None):
        city = city or str(rng.choice(DOMESTIC))
        signup = signup_day if signup_day is not None else -int(rng.integers(60, 1500))
        a = {"account_id": f"A{len(self.accounts) + 1:04d}", "account_type": kind,
             "home_city": city, "signup_date": (START + pd.Timedelta(days=signup)).date(),
             "_signup_day": signup, "_role": role, "_ring": ring, "_tags": set(),
             "devices": [self.device() for _ in range(1 if rng.random() < 0.8 else 2)],
             "ips": [domestic_ip(city) for _ in range(1 if rng.random() < 0.7 else 2)],
             "rate": float(rng.uniform(0.4, 1.6)) if kind == "personal" else float(rng.uniform(3, 6)),
             "spend": float(rng.lognormal(0, 0.35)), "peak": float(rng.uniform(10, 20))}
        weights = np.array([v[2] for v in CATEGORIES.values()], float)
        weights *= rng.uniform(0.5, 1.5, len(weights))
        a["cat_weights"] = weights / weights.sum()
        self.accounts.append(a)
        return a

    def add(self, a, when, kind, amount, merchant=None, payee=None, channel=None, device=None,
            ip=None, city=None, country="IN", fraud=False, pattern="", tag=""):
        in_store = merchant is not None and CATEGORIES[self.category[merchant]][3]
        channel = channel or ("pos" if in_store else "app")
        self.txns.append({
            "timestamp": when, "account_id": a["account_id"], "txn_type": kind,
            "channel": channel, "merchant_id": merchant or "", "payee_account_id": payee or "",
            "amount_inr": round(float(amount), 2),
            "device_id": "" if channel == "pos" else (device or str(rng.choice(a["devices"]))),
            "ip_address": "" if channel == "pos" else (ip or str(rng.choice(a["ips"]))),
            "city": city or a["home_city"], "country": country,
            "_fraud": fraud, "_pattern": pattern, "_ring": a["_ring"] if fraud else "", "_tag": tag})

    # ------------------------------------------------------------------- #
    def normal_activity(self, a, day_from=0, day_to=DAYS, rate_mult=1.0):
        cats = list(CATEGORIES)
        start = max(day_from, a["_signup_day"])
        for day in range(start, day_to):
            for _ in range(rng.poisson(a["rate"] * rate_mult)):
                cat = str(rng.choice(cats, p=a["cat_weights"]))
                lo, hi = CATEGORIES[cat][:2]
                amount = min(hi, lo * np.exp(rng.uniform(0, np.log(hi / lo))) * a["spend"])
                hour = float(np.clip(rng.normal(a["peak"], 3), 6, 23))
                self.add(a, ts(day, hour), "purchase", amount, merchant=str(rng.choice(self.by_cat[cat])))


# --------------------------------------------------------------------------- #

def build():
    w = World()
    personal = [w.account() for _ in range(520)]
    business = [w.account(kind="business") for _ in range(20)]
    landlords = [w.account(kind="business") for _ in range(3)]
    for x in landlords:
        x["_tags"].add("rent_hub")
    mules = [w.account(role="mule", ring=r, signup_day=-int(rng.integers(20, 90)))
             for r in ("R1", "R2", "ATO")]
    ring1 = [w.account(role="ring_member", ring="R1", signup_day=-int(rng.integers(15, 45)),
                       city=DOMESTIC[i % len(DOMESTIC)]) for i in range(10)]
    farm_devices = [w.device(), w.device()]
    farm_subnet = "45.77.12."
    ring2 = []
    for i in range(8):
        a = w.account(role="ring_member", ring="R2", signup_day=int(rng.integers(18, 25)),
                      city=str(rng.choice(DOMESTIC)))
        a["devices"] = [farm_devices[i % 2]]
        a["ips"] = [farm_subnet + str(rng.integers(2, 250))]
        ring2.append(a)

    # Normal background for everyone (mules are quiet).
    for a in personal + business + landlords + ring1:
        w.normal_activity(a)
    for a in ring2:  # fake accounts: a little cover activity
        w.normal_activity(a, rate_mult=0.3)
    for m in mules:
        w.normal_activity(m, rate_mult=0.15)

    # ---------------- benign look-alikes ----------------
    for b in business:
        b["_tags"].add("business")
        others = [x for x in business if x is not b]
        for day in range(DAYS):
            for _ in range(rng.poisson(0.8)):
                w.add(b, ts(day, rng.uniform(9, 18)), "transfer", rng.uniform(20000, 200000),
                      payee=str(rng.choice(others)["account_id"]), tag="business")
            if rng.random() < 0.15:
                w.add(b, ts(day, rng.uniform(9, 18)), "purchase", rng.uniform(30000, 150000),
                      merchant=str(rng.choice(w.by_cat["electronics"])), tag="business")

    idx = rng.permutation(len(personal))
    pick = lambda n, off: [personal[i] for i in idx[off:off + n]]  # noqa: E731
    high_value, travellers = pick(40, 0), pick(30, 40)
    households, renters, hobby = pick(36, 70), pick(60, 106), pick(25, 166)
    ato_victims, ct_victims = pick(5, 191), pick(3, 196)

    for a in high_value:
        a["_tags"].add("high_value")
        cat = str(rng.choice(["electronics", "travel"]))
        w.add(a, ts(rng.integers(5, DAYS), rng.uniform(10, 21)), "purchase",
              rng.uniform(40000, 140000), merchant=str(rng.choice(w.by_cat[cat])),
              device=a["devices"][0], tag="high_value")

    for a in travellers:
        a["_tags"].add("travel")
        city, cc = ABROAD[int(rng.integers(len(ABROAD)))]
        trip_ip = abroad_ip(ABROAD.index((city, cc)))
        d0 = int(rng.integers(5, DAYS - 7))
        for day in range(d0, d0 + int(rng.integers(3, 7))):
            for _ in range(rng.poisson(2)):
                if rng.random() < 0.6:
                    w.add(a, ts(day, rng.uniform(8, 22)), "purchase", rng.uniform(800, 9000),
                          merchant=str(rng.choice(w.by_cat["restaurants"] + w.by_cat["fashion"])),
                          channel="pos", city=city, country=cc, tag="travel")
                else:
                    w.add(a, ts(day, rng.uniform(8, 22)), "purchase", rng.uniform(500, 6000),
                          merchant=str(rng.choice(w.by_cat["online_marketplace"])),
                          device=a["devices"][0], ip=trip_ip, city=city, country=cc, tag="travel")

    for h in range(12):  # households of 3 sharing one phone and Wi-Fi
        members = households[h * 3:(h + 1) * 3]
        dev, ip = w.device(), domestic_ip(members[0]["home_city"])
        for a in members:
            a["_tags"].add("shared_device")
            a["devices"], a["ips"] = [dev], [ip]
    # Household members' existing app transactions should use the shared device.
    shared = {a["account_id"]: a for a in households}
    for t in w.txns:
        a = shared.get(t["account_id"])
        if a and t["channel"] != "pos" and t["country"] == "IN":
            t["device_id"], t["ip_address"], t["_tag"] = a["devices"][0], a["ips"][0], "shared_device"

    for i, a in enumerate(renters):
        a["_tags"].add("rent")
        landlord = landlords[i % 3]
        rent = float(rng.uniform(8000, 25000))
        for month_start in (0, 31):
            w.add(a, ts(month_start + rng.integers(0, 5), rng.uniform(8, 21)), "transfer", rent,
                  payee=landlord["account_id"], tag="rent")

    for a in hobby:
        a["_tags"].add("crypto_hobby")
        for _ in range(int(rng.integers(1, 4))):
            cat = str(rng.choice(["gift_cards", "crypto_exchange"]))
            lo, hi = CATEGORIES[cat][:2]
            w.add(a, ts(rng.integers(0, DAYS), rng.uniform(9, 22)), "purchase",
                  rng.uniform(lo, hi), merchant=str(rng.choice(w.by_cat[cat])), tag="crypto_hobby")

    # ---------------- fraud ----------------
    gift, volt, coin = w.by_name["GiftHub Digital"], w.by_name["VoltX Gadgets"], w.by_name["CoinSwift Exchange"]
    mule_in = {m["_ring"]: [] for m in mules}

    for i, a in enumerate(ring1):  # same unusual sequence, then the same mule
        d = 34 + i * 1.3 + rng.uniform(0, 0.8)
        t0 = ts(d, rng.uniform(10, 20))
        steps = [(gift, rng.uniform(3000, 6000)), (volt, rng.uniform(8000, 15000)),
                 (coin, rng.uniform(5000, 9000))]
        t = t0
        for merchant, amount in steps:
            w.add(a, t, "purchase", amount, merchant=merchant, fraud=True, pattern="sequence_ring")
            t += pd.Timedelta(hours=float(rng.uniform(1, 18)))
        amount = rng.uniform(10000, 20000)
        w.add(a, t, "transfer", amount, payee=mules[0]["account_id"], fraud=True,
              pattern="sequence_ring")
        mule_in["R1"].append((t, amount))

    for a in ring2:  # device farm
        for _ in range(int(rng.integers(3, 6))):
            w.add(a, ts(rng.uniform(25, 40), rng.uniform(0, 24)), "purchase",
                  rng.uniform(2000, 7000), merchant=str(rng.choice(w.by_cat["online_marketplace"])),
                  fraud=True, pattern="device_farm")
        for _ in range(int(rng.integers(1, 3))):
            t, amount = ts(rng.uniform(28, 42), rng.uniform(0, 24)), rng.uniform(5000, 15000)
            w.add(a, t, "transfer", amount, payee=mules[1]["account_id"], fraud=True,
                  pattern="device_farm")
            mule_in["R2"].append((t, amount))

    for i, a in enumerate(ato_victims):  # account takeover
        a["_role"], a["_ring"] = "compromised", "ATO"
        city, cc = ABROAD[i % len(ABROAD)]
        dev, ip = w.device(), abroad_ip(i)
        t = ts(rng.integers(30, 56), rng.uniform(1, 4))
        for merchant, amount in [(gift, rng.uniform(8000, 10000)), (gift, rng.uniform(8000, 10000)),
                                 (str(rng.choice(w.by_cat["electronics"])), rng.uniform(30000, 60000))]:
            w.add(a, t, "purchase", amount, merchant=merchant, device=dev, ip=ip, city=city,
                  country=cc, fraud=True, pattern="account_takeover")
            t += pd.Timedelta(minutes=float(rng.uniform(5, 30)))
        amount = rng.uniform(20000, 50000)
        w.add(a, t, "transfer", amount, payee=mules[2]["account_id"], device=dev, ip=ip,
              city=city, country=cc, fraud=True, pattern="account_takeover")
        mule_in["ATO"].append((t, amount))

    recharge = w.by_name["QuickPay Recharge"]
    for a in ct_victims:  # card testing
        a["_role"] = "compromised"  # independent victims, not a ring
        dev, ip = w.device(), domestic_ip(str(rng.choice(DOMESTIC)))
        t = ts(rng.integers(20, 55), rng.uniform(0, 24))
        for _ in range(int(rng.integers(6, 12))):
            w.add(a, t, "purchase", rng.uniform(1, 10), merchant=recharge, channel="online",
                  device=dev, ip=ip, fraud=True, pattern="card_testing")
            t += pd.Timedelta(seconds=float(rng.uniform(20, 120)))
        w.add(a, t + pd.Timedelta(minutes=10), "purchase", rng.uniform(25000, 45000),
              merchant=w.by_name["MegaByte Store"], channel="online", device=dev, ip=ip,
              fraud=True, pattern="card_testing")

    for m in mules:  # cash out each incoming transfer within hours
        for t, amount in mule_in[m["_ring"]]:
            w.add(m, t + pd.Timedelta(hours=float(rng.uniform(0.5, 6))), "purchase",
                  amount * rng.uniform(0.9, 0.98), merchant=coin, fraud=True,
                  pattern="mule_cashout")

    return w


def main():
    for d in (DATA, EVAL):
        d.mkdir(parents=True, exist_ok=True)
    w = build()

    # Device-farm and mule accounts are fake accounts run by fraudsters, so all their
    # activity is fraud. Sequence-ring members may be recruited real customers: only
    # their ring transactions are fraud.
    fake = {a["account_id"]: a for a in w.accounts
            if a["_role"] == "mule" or (a["_role"] == "ring_member" and a["_ring"] == "R2")}
    for t in w.txns:
        a = fake.get(t["account_id"])
        if a and not t["_fraud"]:
            t["_fraud"], t["_ring"] = True, a["_ring"]
            t["_pattern"] = "mule_activity" if a["_role"] == "mule" else "device_farm"

    tx = pd.DataFrame(w.txns).sort_values("timestamp", kind="stable").reset_index(drop=True)
    tx = tx[tx["timestamp"] < START + pd.Timedelta(days=DAYS)]
    tx.insert(0, "txn_id", [f"T{i + 1:06d}" for i in range(len(tx))])
    tx["timestamp"] = tx["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

    public = [c for c in tx.columns if not c.startswith("_")]
    tx[public].to_csv(DATA / "transactions.csv", index=False)
    labels = tx[["txn_id", "_fraud", "_pattern", "_ring", "_tag"]]
    labels.columns = ["txn_id", "is_fraud", "pattern", "ring_id", "benign_tag"]
    labels.to_csv(EVAL / "transaction_labels.csv", index=False)

    acc = pd.DataFrame(w.accounts)
    acc[["account_id", "account_type", "home_city", "signup_date"]].to_csv(
        DATA / "accounts.csv", index=False)
    acc_labels = pd.DataFrame({
        "account_id": acc["account_id"],
        "is_fraud": acc["_role"].isin(["ring_member", "mule", "compromised"]),
        "role": acc["_role"], "ring_id": acc["_ring"],
        "benign_tags": acc["_tags"].map(lambda s: ";".join(sorted(s)))})
    acc_labels.to_csv(EVAL / "account_labels.csv", index=False)
    w.merchants.to_csv(DATA / "merchants.csv", index=False)

    summary = {
        "transactions": len(tx), "accounts": len(acc), "merchants": len(w.merchants),
        "devices": w.n_devices, "fraud_transactions": int(labels["is_fraud"].sum()),
        "fraud_rate_pct": round(100 * labels["is_fraud"].mean(), 3),
        "fraud_by_pattern": labels[labels["is_fraud"]]["pattern"].value_counts().to_dict(),
        "fraud_accounts_by_role": acc_labels[acc_labels["is_fraud"]]["role"].value_counts().to_dict(),
        "benign_lookalike_transactions": labels[labels["benign_tag"] != ""]["benign_tag"]
        .value_counts().to_dict(),
    }
    (EVAL / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
