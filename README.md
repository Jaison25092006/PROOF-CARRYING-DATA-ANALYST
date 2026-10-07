# Fraud Intelligence — HackNex 2026 (HNX26PSI04)

Real-time financial fraud detection that finds **fraud rings**: groups of accounts acting
together where no single transaction looks suspicious. Every transaction and account gets
a risk score with **written reasons** and a **recommended action**, and the system stays quiet
on legitimate high-value activity.

No external API, LLM, or GPU is needed. Everything runs locally in about 15 seconds.

## How it works

```
transactions (time order)
   │
   ├─► 1. Real-time layer      each payment vs. that customer's own history, past data only
   │      new device / new country / 3 am / amount spike / velocity / card-testing burst /
   │      shared phone among new accounts / new payee / payee collecting from many / pass-through
   │
   ├─► 2. Anomaly model        Isolation Forest trained on the first 20 days ("normal")
   │
   └─► 3. Network layer        accounts linked by evidence → connected components = rings
          shared phone (new accounts) · shared IP range · same rare purchase sequence ·
          paying a mule (receives from 3+ accounts, passes ≥60% on to gift cards/crypto)
                │
                ▼
   risk = 1 − ∏(1 − weight of each reason)     ← every score is the sum of named reasons
   actions: ALLOW · REVIEW · HOLD + step-up verification · BLOCK
            account: MONITOR · RESTRICT · FREEZE and investigate ring
```

Benign look-alikes are handled deliberately:
- **Households sharing a phone:** old accounts, so no shared-device link.
- **Landlords:** collect from many tenants but don't pass the money on, so not a mule.
- **Businesses:** collecting payments and adding new suppliers is normal for them.
- **Travellers:** they use their usual phone abroad.

## Results

Scored against hidden labels (`eval/evaluate.py`). Flagged means risk ≥ 0.5 (HOLD or BLOCK).

| Dataset | Transaction precision | Transaction recall | False alarms | Fraud accounts caught | Innocent accounts flagged | Rings recovered exactly |
|---|---|---|---|---|---|---|
| Seed 2026 (development) | 100% | 99.3% | 0 | 29 / 29 | 0 | 3 / 3 |
| Seed 7 (**unseen**, no retuning) | 98.1% | 100% | 6 | 29 / 29 | 2 | 3 / 3 |
| Seed 42 (**unseen**, no retuning) | 99.4% | 100% | 2 | 29 / 29 | 2 | 3 / 3 |
| Seed 99 (**unseen**, no retuning) | 95.5% | 100% | 13 | 29 / 29 | 4 | 2 / 3 (3rd at 92% overlap) |

Reproduce with `python eval/robustness.py` (writes `eval/robustness.json`). On seed 99 an
innocent customer who occasionally buys gift cards and crypto happened to repeat the ring's
purchase sequence within 72 hours. It was pulled into the ring, and its payments became 8 of
the 13 false alarms. All 10 real members were still found. We report this rather than tune it
away (see Limitations).

- **False alarms on look-alikes (seed 2026):** 0 out of 2,221. That covers 1,174 business payments,
  560 shared-phone payments, 279 travel payments, 120 rent payments, 40 big one-off purchases, and
  48 gift-card/crypto hobby purchases.
- **Naive baseline** ("flag the biggest 1% of payments"): **0% precision, 0% recall**. Big ≠ fraud.
- **Real-time layer alone:** 100% precision, 44% recall. The network layer is what catches the rings.

## Data

`data_gen/generate.py` creates a fictional dataset: 564 accounts, about 40,000 transactions
over 60 days, in INR. The detector only reads `data/`; the labels are kept in `eval/`.

| Planted fraud | What happens |
|---|---|
| Sequence ring (10 accounts) | Different cities, phones and IPs, but the same gift card → electronics → crypto sequence, then all pay one mule |
| Device farm (8 accounts) | New accounts sharing 2 phones and one IP range, quick purchases, then pay a mule |
| Account takeover (5) | New phone, foreign IP, 1–4 am, gift cards, electronics, transfer to a mule |
| Card testing (3) | Burst of ₹1–10 payments at one merchant, then one large purchase |
| Mules (3) | Receive from ring members and pass the money to crypto within hours |

## Usage

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt      # Windows
python data_gen/generate.py          # dataset (DATA_SEED=7 for an unseen variant)
python -m fraud.pipeline             # score everything → outputs/
python eval/evaluate.py              # scorecard vs hidden labels → eval/scorecard.json
python report.py                     # readable summary in the terminal
streamlit run app.py                 # demo web app
python -m unittest discover tests    # unit tests
```

### Demo app tabs
- **Overview:** totals, flagged payments over time, and the scorecard against the naive baseline.
- **Fraud rings:** connection diagram (accounts, shared evidence, mule), why they're linked,
  a timeline, and the recommended action.
- **Accounts:** ranked accounts with explanations and each account's risk history.
- **Transactions:** every held or blocked payment, with its reasons and weights.
- **Live check:** score a new payment in real time, for example a new phone in Dubai at 3 am.

## Outputs (`outputs/`)

| File | Content |
|---|---|
| `transaction_scores.csv` | risk (real-time and final), action, explanation for every transaction |
| `account_scores.csv` | risk, action, ring, explanation for every account |
| `rings.json` | members, mules, links and evidence, timeline, amount, recommended action |

## Repository layout

```
fraud/stream.py        real-time scoring with reason codes
fraud/network.py       account graph and ring extraction
fraud/pipeline.py      real-time + anomaly model + network → final risk and actions
data_gen/generate.py   synthetic dataset with fraud patterns and benign look-alikes
eval/evaluate.py       metrics against the hidden labels
report.py              terminal report
app.py                 Streamlit demo
tests/test_fraud.py    unit tests on small hand-built datasets
```

## Scope note

**Minimum viable (done):** synthetic data, real-time scoring with explanations, detection of
at least one ring, transaction and account risk, recommended actions.

**Stretch (done):** three ring types (behavioural sequence, device farm, mule-linked takeover),
an Isolation Forest anomaly model, retroactive card-testing flags, a connection diagram, a live
scoring page, and evaluation on unseen seeds.

**Limitations:**
- Reason weights and thresholds are hand-set from known fraud patterns, not learned from data.
- Tested on synthetic data only, which we designed.
- The network layer runs as a batch, not on every event.
- A single coincidental match on the "same purchase sequence" evidence can pull an innocent
  account into a ring (seed 99). The planned fix is to require two independent kinds of
  evidence per ring member, for example same sequence **and** paying the mule.
- Fraud patterns very different from these may need new rules (the anomaly model is the
  fallback for those).

## Resources

- Python 3.12, pandas, numpy, scikit-learn (Isolation Forest), networkx, Streamlit, Plotly
- All data is synthetic and fictional.
- AI coding assistance (Claude Code) was used during development.
