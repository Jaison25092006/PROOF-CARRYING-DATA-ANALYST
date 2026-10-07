"""Real-time scoring: each transaction is scored only from what happened before it.

Every risk factor that fires becomes a reason (code, weight, human-readable evidence).
Transaction risk combines reasons with a noisy-OR:  risk = 1 - prod(1 - weight).
"""

import math
from collections import defaultdict, deque

import pandas as pd

RISKY_CATEGORIES = {"gift_cards", "crypto_exchange"}
YOUNG_DAYS = 45  # an account younger than this is "new"

# code: (weight, description template)
REASONS = {
    "NEW_DEVICE": (0.15, "device {device} never used by this account before"),
    "NEW_DEVICE_ABROAD": (0.45, "new device {device} used from {country}, outside the customer's history"),
    "NIGHT_ACTIVITY": (0.20, "transaction at {hour:02d}:00; this account almost never transacts at night"),
    "AMOUNT_SPIKE": (0.25, "₹{amount:,.0f} is {z:.1f} standard deviations above this account's usual spend"),
    "AMOUNT_SPIKE_NEW_DEVICE": (0.35, "unusually large amount on a device the customer has never used"),
    "FIRST_RISKY_CATEGORY": (0.15, "first ever {category} purchase on this account"),
    "VELOCITY": (0.25, "{n_1h} transactions in the past hour"),
    "CARD_TEST_BURST": (0.70, "{tiny} payments under ₹20 at the same merchant within 30 minutes"),
    "CARD_TEST_CASHOUT": (0.85, "large purchase minutes after a burst of tiny test payments on the same device"),
    "SHARED_DEVICE_NEW_ACCOUNTS": (0.45, "device {device} is also used by {n} other new accounts"),
    "SHARED_SUBNET_NEW_ACCOUNTS": (0.25, "IP range {subnet}.x is shared with {n} other new accounts"),
    "NEW_ACCOUNT": (0.10, "account opened {age} days ago"),
    "NEW_PAYEE_LARGE": (0.30, "first transfer to {payee}, and the amount is unusually large"),
    "NEW_PAYEE": (0.05, "first transfer to {payee}"),
    "PAYEE_FAN_IN": (0.35, "payee {payee} is a {payee_age}-day-old account that received money from {senders} different accounts in 14 days"),
    "PASS_THROUGH": (0.70, "spends {share:.0%} of the ₹{inflow:,.0f} received in the last 24h on {category} (money passing straight through)"),
}


def reason(code, **ctx):
    weight, template = REASONS[code]
    return {"code": code, "weight": weight, "evidence": template.format(**ctx)}


def noisy_or(reasons):
    p = 1.0
    for r in reasons:
        p *= 1 - r["weight"]
    return 1 - p


class AccountState:
    def __init__(self):
        self.n = 0
        self.log_sum = self.log_sq = 0.0
        self.night = 0
        self.digital = 0
        self.device_first_seen = {}
        self.countries, self.categories, self.payees = set(), set(), set()
        self.recent = deque()       # (time, amount, merchant, device) within 24h
        self.inflows = deque()      # (time, amount) received within 24h
        self.tiny_burst_until = {}  # device -> time a card-testing burst was seen

    def amount_z(self, log_amount):
        if self.n < 5:
            return 0.0
        mean = self.log_sum / self.n
        std = math.sqrt(max(self.log_sq / self.n - mean * mean, 0))
        return (log_amount - mean) / max(std, 0.35)


def score_stream(tx, accounts, merchants):
    """Return one row per transaction: real-time risk, reasons, and model features."""
    signup = pd.to_datetime(accounts.set_index("account_id")["signup_date"]).to_dict()
    kind = accounts.set_index("account_id")["account_type"].to_dict()
    category = merchants.set_index("merchant_id")["category"].to_dict()
    states = defaultdict(AccountState)
    device_accounts = defaultdict(set)
    subnet_accounts = defaultdict(set)
    payee_senders = defaultdict(deque)  # payee -> (time, sender)
    out = []

    for t in tx.itertuples(index=False):
        now, acct = t.timestamp, t.account_id
        s = states[acct]
        age = (now - signup[acct]).days
        cat = category.get(t.merchant_id, "transfer")
        digital = t.channel != "pos"
        amount, log_amount = t.amount_inr, math.log1p(t.amount_inr)
        z = s.amount_z(log_amount)
        hour = now.hour
        reasons = []

        while s.recent and (now - s.recent[0][0]).total_seconds() > 86400:
            s.recent.popleft()
        while s.inflows and (now - s.inflows[0][0]).total_seconds() > 86400:
            s.inflows.popleft()
        n_1h = sum(1 for r in s.recent if (now - r[0]).total_seconds() <= 3600)

        # Device: new to this account, counting a device first seen in the last 6 hours.
        new_device = False
        if digital and t.device_id:
            first = s.device_first_seen.get(t.device_id)
            established = s.digital >= 3
            new_device = established and (first is None or (now - first).total_seconds() < 6 * 3600)
            if first is None:
                s.device_first_seen[t.device_id] = now
        if new_device:
            if t.country != "IN" and t.country not in s.countries:
                reasons.append(reason("NEW_DEVICE_ABROAD", device=t.device_id, country=t.country))
            else:
                reasons.append(reason("NEW_DEVICE", device=t.device_id))
            if z > 2:
                reasons.append(reason("AMOUNT_SPIKE_NEW_DEVICE"))
        if z > 3 and not new_device:
            reasons.append(reason("AMOUNT_SPIKE", amount=amount, z=z))
        if hour < 5 and s.n >= 10 and s.night / s.n < 0.05:
            reasons.append(reason("NIGHT_ACTIVITY", hour=hour))
        if cat in RISKY_CATEGORIES and cat not in s.categories and s.n >= 5:
            reasons.append(reason("FIRST_RISKY_CATEGORY", category=cat.replace("_", " ")))
        if n_1h >= 4:
            reasons.append(reason("VELOCITY", n_1h=n_1h + 1))

        # Card testing: burst of tiny payments at one merchant, then a large purchase.
        tiny = 0
        if amount < 20:
            tiny = 1 + sum(1 for r in s.recent if r[1] < 20 and r[2] == t.merchant_id
                           and (now - r[0]).total_seconds() <= 1800)
            if tiny >= 4:
                reasons.append(reason("CARD_TEST_BURST", tiny=tiny))
                burst_ids = [r[4] for r in s.recent if r[1] < 20 and r[2] == t.merchant_id
                             and (now - r[0]).total_seconds() <= 1800]
                s.tiny_burst_until[t.device_id] = now
        elif amount > 5000 and t.device_id in s.tiny_burst_until and \
                (now - s.tiny_burst_until[t.device_id]).total_seconds() <= 7200:
            reasons.append(reason("CARD_TEST_CASHOUT"))

        # Shared infrastructure between new accounts (households are old accounts).
        if digital and t.device_id:
            device_accounts[t.device_id].add(acct)
            young = [a for a in device_accounts[t.device_id]
                     if a != acct and (now - signup[a]).days < YOUNG_DAYS]
            if age < YOUNG_DAYS and len(young) >= 2:
                reasons.append(reason("SHARED_DEVICE_NEW_ACCOUNTS", device=t.device_id, n=len(young)))
        if digital and t.ip_address:
            subnet = t.ip_address.rsplit(".", 1)[0]
            subnet_accounts[subnet].add(acct)
            young = [a for a in subnet_accounts[subnet]
                     if a != acct and (now - signup[a]).days < YOUNG_DAYS]
            if age < YOUNG_DAYS and len(young) >= 3:
                reasons.append(reason("SHARED_SUBNET_NEW_ACCOUNTS", subnet=subnet, n=len(young)))
        if age < YOUNG_DAYS and reasons:
            reasons.append(reason("NEW_ACCOUNT", age=age))

        # Transfers: new payees, and payees that collect from many senders.
        payee = t.payee_account_id
        if payee:
            if payee not in s.payees:
                # Businesses add new suppliers routinely; a large first payment is normal there.
                large = z > 2 and kind[acct] == "personal"
                reasons.append(reason("NEW_PAYEE_LARGE" if large else "NEW_PAYEE", payee=payee))
            q = payee_senders[payee]
            q.append((now, acct))
            while q and (now - q[0][0]).days > 14:
                q.popleft()
            senders = len({a for _, a in q})
            payee_age = (now - signup[payee]).days
            # Collecting from many payers is a business's job; for a new personal account
            # it is the classic mule pattern.
            if senders >= 3 and payee_age < 120 and kind[payee] == "personal":
                reasons.append(reason("PAYEE_FAN_IN", payee=payee, payee_age=payee_age,
                                      senders=senders))
            states[payee].inflows.append((now, amount))

        # Mule behaviour: spending most of what just came in on gift cards / crypto.
        inflow = sum(a for _, a in s.inflows)
        if inflow > 0 and cat in RISKY_CATEGORIES and amount >= 0.6 * inflow:
            reasons.append(reason("PASS_THROUGH", share=min(amount / inflow, 1), inflow=inflow,
                                  category=cat.replace("_", " ")))

        out.append({
            "txn_id": t.txn_id, "realtime_risk": noisy_or(reasons), "reasons": reasons,
            # earlier tiny payments that turned out to be part of this burst
            "burst_members": burst_ids if tiny >= 4 else [],
            # numeric features for the anomaly model
            "f_log_amount": log_amount, "f_amount_z": z, "f_new_device": int(new_device),
            "f_hour": hour, "f_night": int(hour < 5), "f_velocity_1h": n_1h,
            "f_account_age": age, "f_foreign": int(t.country != "IN"),
            "f_risky_category": int(cat in RISKY_CATEGORIES), "f_transfer": int(bool(payee)),
            "f_device_accounts": len(device_accounts.get(t.device_id, ())) if t.device_id else 0,
        })

        # Update history after scoring (so the score only used the past).
        s.n += 1
        s.log_sum += log_amount
        s.log_sq += log_amount ** 2
        s.night += hour < 5
        s.digital += digital
        if t.country != "IN" and not new_device:
            s.countries.add(t.country)
        s.categories.add(cat)
        if payee:
            s.payees.add(payee)
        s.recent.append((now, amount, t.merchant_id, t.device_id, t.txn_id))

    return pd.DataFrame(out)
