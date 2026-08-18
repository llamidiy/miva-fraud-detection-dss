# Business Scenarios

This catalog documents all 20 business validation scenarios used to test the fraud detection decision support system, and the specific dataset/model evidence that motivated each one. See `validation_report.md` for how the system actually responded to each scenario.

## Fraud Scenarios

### F1: Unusually High-Value Transfer With Origin Fully Drained

**Expected outcome:** `fraud_high_confidence` | **Engineered features exercised:** `balanceDeltaOrig`, `newbalanceOrig`

**Scenario Description**

| Field | Value |
| --- | --- |
| `step` | 14 |
| `type` | TRANSFER |
| `amount` | 5,200,000.00 |
| `oldbalanceOrg` | 5,200,000.00 |
| `newbalanceOrig` | 0.00 |
| `oldbalanceDest` | 850,000.00 |
| `newbalanceDest` | 6,050,000.00 |
| `isFlaggedFraud` | 0 |

**Business Context**

A customer's account is used to transfer its entire $5.2M balance out in a single transaction, with the destination account correctly credited.

**Dataset/Feature Rationale**

*Source evidence:* 98.05% of fraudulent transactions in the engineered dataset leave newbalanceOrig == 0; fraud amounts average $1,467,967 (median $441,423), and fraud never occurs outside TRANSFER/CASH_OUT.

*Expected reasoning:* newbalanceOrig (native importance rank #2 (0.3068); mean |SHAP| rank #3 (2.6173)) and balanceDeltaOrig (native importance rank #1 (0.3576); mean |SHAP| rank #2 (3.1724)) should dominate: the full-balance drain on a very large TRANSFER mirrors the single strongest fraud signature in the training data.

### F2: Cash-Out With Zero Origin Balance After Transaction

**Expected outcome:** `fraud_high_confidence` | **Engineered features exercised:** `isOrigZeroBalance`, `type_CASH_OUT`

**Scenario Description**

| Field | Value |
| --- | --- |
| `step` | 10 |
| `type` | CASH_OUT |
| `amount` | 650,000.00 |
| `oldbalanceOrg` | 650,000.00 |
| `newbalanceOrig` | 0.00 |
| `oldbalanceDest` | 120,000.00 |
| `newbalanceDest` | 770,000.00 |
| `isFlaggedFraud` | 0 |

**Business Context**

A cash withdrawal empties an account of $650,000 (near the fraud-population median amount), with the receiving agent account properly credited.

**Dataset/Feature Rationale**

*Source evidence:* Fraud amount median is $441,423 in the engineered dataset; this scenario sits close to that median while reproducing the 98.05% origin-drain pattern for CASH_OUT.

*Expected reasoning:* isOrigZeroBalance (native importance rank #6 (0.0120); mean |SHAP| rank #5 (2.0478)) and type_CASH_OUT (native importance rank #7 (0.0041); mean |SHAP| rank #12 (0.8464)) should contribute strongly, reflecting the origin-drain signature on a fraud-eligible type.

### F3: Abnormal Balance Discrepancy (Ledger Inconsistency)

**Expected outcome:** `fraud_high_confidence` | **Engineered features exercised:** `errorBalanceOrig`

**Scenario Description**

| Field | Value |
| --- | --- |
| `step` | 16 |
| `type` | TRANSFER |
| `amount` | 500,000.00 |
| `oldbalanceOrg` | 500,000.00 |
| `newbalanceOrig` | 50,000.00 |
| `oldbalanceDest` | 200,000.00 |
| `newbalanceDest` | 700,000.00 |
| `isFlaggedFraud` | 0 |

**Business Context**

A transfer's reported ending origin balance ($50,000) does not reconcile with the starting balance minus the transferred amount, suggesting a manipulated or corrupted account record.

**Dataset/Feature Rationale**

*Source evidence:* errorBalanceOrig is the single strongest engineered feature by mean |SHAP| value in Sprint 5's analysis and ranks #3 in native XGBoost importance.

*Expected reasoning:* errorBalanceOrig (native importance rank #3 (0.2685); mean |SHAP| rank #1 (4.0605)) should be the dominant factor: this scenario deliberately produces a $50,000 ledger discrepancy on the origin side (destination side reconciles cleanly).

### F4: Destination Account Never Credited (Funds Vanish)

**Expected outcome:** `fraud_high_confidence` | **Engineered features exercised:** `errorBalanceDest`, `isDestZeroBalance`

**Scenario Description**

| Field | Value |
| --- | --- |
| `step` | 9 |
| `type` | CASH_OUT |
| `amount` | 300,000.00 |
| `oldbalanceOrg` | 300,000.00 |
| `newbalanceOrig` | 0.00 |
| `oldbalanceDest` | 0.00 |
| `newbalanceDest` | 0.00 |
| `isFlaggedFraud` | 0 |

**Business Context**

A cash-out drains the origin account, but the receiving account balance never changes — the destination side of the ledger shows zero before and after.

**Dataset/Feature Rationale**

*Source evidence:* 49.63% of fraudulent transactions in the engineered dataset leave oldbalanceDest == newbalanceDest == 0 — the single most common destination-side fraud pattern observed.

*Expected reasoning:* errorBalanceDest (native importance rank #9 (0.0023); mean |SHAP| rank #14 (0.7926)) and isDestZeroBalance (native importance rank #8 (0.0031); mean |SHAP| rank #18 (0.1391)) should be prominent, reflecting funds that are debited from the origin but never land in the destination account in this snapshot.

### F5: Large Transfer at an Off-Peak Hour

**Expected outcome:** `fraud_high_confidence` | **Engineered features exercised:** `transactionHour`, `balanceDeltaOrig`

**Scenario Description**

| Field | Value |
| --- | --- |
| `step` | 3 |
| `type` | TRANSFER |
| `amount` | 2,200,000.00 |
| `oldbalanceOrg` | 2,200,000.00 |
| `newbalanceOrig` | 0.00 |
| `oldbalanceDest` | 500,000.00 |
| `newbalanceDest` | 2,700,000.00 |
| `isFlaggedFraud` | 0 |

**Business Context**

A $2.2M transfer, fully draining the origin account, occurs at 3am — a time window with very little legitimate customer activity.

**Dataset/Feature Rationale**

*Source evidence:* Legitimate activity concentrates around hours 12-13 and 18-20 (this sprint's re-analysis of the engineered dataset), while fraud volume is flat across hours (274-375 cases/hour). This scenario tests whether the model's decision holds up on the balance evidence alone at an hour with little legitimate precedent.

*Expected reasoning:* The origin-drain/amount signals (balanceDeltaOrig, newbalanceOrig) should drive the prediction rather than transactionHour itself, since this sprint's re-analysis found fraud distributed near-uniformly across all 24 hours.

### F6a: Structuring Pattern — Part 1 of 2

**Expected outcome:** `fraud_medium_confidence` | **Engineered features exercised:** `balanceDeltaOrig`

**Scenario Description**

| Field | Value |
| --- | --- |
| `step` | 101 |
| `type` | TRANSFER |
| `amount` | 480,000.00 |
| `oldbalanceOrg` | 960,000.00 |
| `newbalanceOrig` | 480,000.00 |
| `oldbalanceDest` | 0.00 |
| `newbalanceDest` | 480,000.00 |
| `isFlaggedFraud` | 0 |

**Business Context**

The first of two same-day transfers that together drain a $960,000 account in two roughly equal steps rather than one large transaction — a common structuring tactic to avoid drawing attention to a single large transfer.

**Dataset/Feature Rationale**

*Source evidence:* PaySim/the champion model score transactions independently, so a real structuring pattern is represented here as two related, sequential scenarios (see F6b) rather than a single multi-transaction feature, an explicit limitation of a single-transaction classifier worth documenting.

*Expected reasoning:* A partial (50%) balance drain on a fraud-eligible TRANSFER type; individually more ambiguous than a full drain, testing the model on a less extreme variant of the dominant fraud pattern.

### F6b: Structuring Pattern — Part 2 of 2

**Expected outcome:** `fraud_high_confidence` | **Engineered features exercised:** `newbalanceOrig`

**Scenario Description**

| Field | Value |
| --- | --- |
| `step` | 102 |
| `type` | TRANSFER |
| `amount` | 480,000.00 |
| `oldbalanceOrg` | 480,000.00 |
| `newbalanceOrig` | 0.00 |
| `oldbalanceDest` | 480,000.00 |
| `newbalanceDest` | 960,000.00 |
| `isFlaggedFraud` | 0 |

**Business Context**

The second transfer in the same structuring sequence as F6a: the remaining $480,000 is moved out shortly after, fully draining the account.

**Dataset/Feature Rationale**

*Source evidence:* Companion scenario to F6a; together they represent the 98.05% origin-drain pattern reached via two steps instead of one.

*Expected reasoning:* newbalanceOrig (native importance rank #2 (0.3068); mean |SHAP| rank #3 (2.6173)) should dominate, reflecting the completed full drain by the end of the two-step sequence.

## Legitimate Scenarios

### L1: Monthly Salary Deposit

**Expected outcome:** `legitimate` | **Engineered features exercised:** `type_CASH_IN`

**Scenario Description**

| Field | Value |
| --- | --- |
| `step` | 33 |
| `type` | CASH_IN |
| `amount` | 4,800.00 |
| `oldbalanceOrg` | 45,200.00 |
| `newbalanceOrig` | 50,000.00 |
| `oldbalanceDest` | 800,000.00 |
| `newbalanceDest` | 804,800.00 |
| `isFlaggedFraud` | 0 |

**Business Context**

An employer deposits a $4,800 salary payment into a customer's account.

**Dataset/Feature Rationale**

*Source evidence:* Zero fraud cases in the engineered dataset are CASH_IN; a real CASH_IN sample row shows the same clean oldbalanceOrg -> newbalanceOrig increment pattern used here.

*Expected reasoning:* type_CASH_OUT-style flags should be absent; CASH_IN is never associated with fraud in the training data, and the balances reconcile cleanly on the origin side once CASH_IN's crediting direction is accounted for.

### L2: Routine Household Bill Payment

**Expected outcome:** `legitimate` | **Engineered features exercised:** `isOrigZeroBalance`, `isDestZeroBalance`, `type_PAYMENT`

**Scenario Description**

| Field | Value |
| --- | --- |
| `step` | 19 |
| `type` | PAYMENT |
| `amount` | 125.50 |
| `oldbalanceOrg` | 0.00 |
| `newbalanceOrig` | 0.00 |
| `oldbalanceDest` | 0.00 |
| `newbalanceDest` | 0.00 |
| `isFlaggedFraud` | 0 |

**Business Context**

A customer pays a $125.50 utility bill through the app.

**Dataset/Feature Rationale**

*Source evidence:* A real PAYMENT sample row in the engineered dataset shows all four balance fields at exactly 0.0 (PaySim does not track running balances for PAYMENT merchant accounts); this scenario reproduces that exact pattern at a typical bill-payment amount (10th-90th percentile: $1,743-$28,811).

*Expected reasoning:* isOrigZeroBalance and isDestZeroBalance will both be True here — identical to many fraud cases — so this scenario specifically tests whether the model relies on type/amount rather than the zero-balance flags in isolation, since PAYMENT is never fraudulent in the training data.

### L3: Small Peer-to-Peer Transfer

**Expected outcome:** `legitimate` | **Engineered features exercised:** `isOrigZeroBalance`, `amount`, `type_TRANSFER`

**Scenario Description**

| Field | Value |
| --- | --- |
| `step` | 42 |
| `type` | TRANSFER |
| `amount` | 250.00 |
| `oldbalanceOrg` | 0.00 |
| `newbalanceOrig` | 0.00 |
| `oldbalanceDest` | 3,200.00 |
| `newbalanceDest` | 3,450.00 |
| `isFlaggedFraud` | 0 |

**Business Context**

A customer sends a friend $250 via a person-to-person transfer.

**Dataset/Feature Rationale**

*Source evidence:* A real small legitimate TRANSFER in the engineered dataset (amount=$306.80) shows oldbalanceOrg == newbalanceOrig == 0 — proving isOrigZeroBalance is common in legitimate low-value transfers too, not an unambiguous fraud tell on its own.

*Expected reasoning:* This is the sharpest discriminating test in the set: same TRANSFER type and the same isOrigZeroBalance=True flag as most fraud, but at a tiny amount with the destination properly credited. amount/logAmount should keep this scored as legitimate despite the shared flag.

### L4: Regular Merchant Payment

**Expected outcome:** `legitimate` | **Engineered features exercised:** `type_PAYMENT`

**Scenario Description**

| Field | Value |
| --- | --- |
| `step` | 61 |
| `type` | PAYMENT |
| `amount` | 890.00 |
| `oldbalanceOrg` | 0.00 |
| `newbalanceOrig` | 0.00 |
| `oldbalanceDest` | 0.00 |
| `newbalanceDest` | 0.00 |
| `isFlaggedFraud` | 0 |

**Business Context**

A customer pays an $890 merchant invoice.

**Dataset/Feature Rationale**

*Source evidence:* PAYMENT amounts in the engineered dataset range from a 10th percentile of $1,743 up; $890 represents a smaller but still typical merchant payment.

*Expected reasoning:* Same PAYMENT zero-balance pattern as L2, at a larger everyday amount.

### L5: Typical Debit Transaction

**Expected outcome:** `legitimate` | **Engineered features exercised:** `type_DEBIT`

**Scenario Description**

| Field | Value |
| --- | --- |
| `step` | 65 |
| `type` | DEBIT |
| `amount` | 3,200.00 |
| `oldbalanceOrg` | 15,000.00 |
| `newbalanceOrig` | 11,800.00 |
| `oldbalanceDest` | 200,000.00 |
| `newbalanceDest` | 203,200.00 |
| `isFlaggedFraud` | 0 |

**Business Context**

A customer makes a routine $3,200 debit against their account.

**Dataset/Feature Rationale**

*Source evidence:* Amount sits near the DEBIT-type median ($3,048.99) in the engineered dataset; zero fraud cases are DEBIT.

*Expected reasoning:* DEBIT is never associated with fraud in training data; balances reconcile cleanly on both sides.

### L6: Large but Legitimate Business Transfer

**Expected outcome:** `legitimate` | **Engineered features exercised:** `balanceDeltaOrig`, `amount`

**Scenario Description**

| Field | Value |
| --- | --- |
| `step` | 83 |
| `type` | TRANSFER |
| `amount` | 1,750,000.00 |
| `oldbalanceOrg` | 6,000,000.00 |
| `newbalanceOrig` | 4,250,000.00 |
| `oldbalanceDest` | 2,000,000.00 |
| `newbalanceDest` | 3,750,000.00 |
| `isFlaggedFraud` | 0 |

**Business Context**

A business moves $1.75M between its own accounts, leaving a substantial remaining balance rather than draining the account.

**Dataset/Feature Rationale**

*Source evidence:* Legitimate TRANSFERs in the engineered dataset reach a 90th percentile of $1.77M — large legitimate transfers are common, so this scenario checks the model does not simply flag every high-value transfer.

*Expected reasoning:* balanceDeltaOrig (native importance rank #1 (0.3576); mean |SHAP| rank #2 (3.1724)) should indicate a partial, non-draining balance change despite the large absolute amount — this scenario deliberately tests that the model does not use amount as a naive threshold.

### L7: Routine Cash Withdrawal

**Expected outcome:** `legitimate` | **Engineered features exercised:** `type_CASH_OUT`, `isOrigZeroBalance`

**Scenario Description**

| Field | Value |
| --- | --- |
| `step` | 92 |
| `type` | CASH_OUT |
| `amount` | 18,000.00 |
| `oldbalanceOrg` | 95,000.00 |
| `newbalanceOrig` | 77,000.00 |
| `oldbalanceDest` | 40,000.00 |
| `newbalanceDest` | 58,000.00 |
| `isFlaggedFraud` | 0 |

**Business Context**

A customer withdraws $18,000 in cash, leaving most of their balance intact.

**Dataset/Feature Rationale**

*Source evidence:* CASH_OUT is a fraud-eligible type, so this scenario specifically probes whether type alone triggers a fraud prediction absent the drain/discrepancy patterns that separate the 4,116 real CASH_OUT fraud cases from the 2,233,384 legitimate ones.

*Expected reasoning:* Same fraud-eligible type as F2/F4, but without the origin-drain or destination-untouched signatures.

## Edge Cases

### E1: Zero-Amount Transaction

**Expected outcome:** `uncertain` | **Engineered features exercised:** `logAmount`, `amount`

**Scenario Description**

| Field | Value |
| --- | --- |
| `step` | 12 |
| `type` | PAYMENT |
| `amount` | 0.00 |
| `oldbalanceOrg` | 5,000.00 |
| `newbalanceOrig` | 5,000.00 |
| `oldbalanceDest` | 0.00 |
| `newbalanceDest` | 0.00 |
| `isFlaggedFraud` | 0 |

**Business Context**

A payment is submitted with a transaction amount of exactly $0.00.

**Dataset/Feature Rationale**

*Source evidence:* The engineered dataset's amount column has a real minimum of $0.00 (legitimate transactions), so this boundary is genuinely reachable in production, not purely synthetic.

*Expected reasoning:* logAmount collapses to log1p(0) = 0; this tests whether the model and pipeline handle the degenerate zero-amount boundary without error.

### E2: Extremely Large Amount Beyond the Training Range

**Expected outcome:** `fraud_high_confidence` | **Engineered features exercised:** `amount`, `balanceDeltaOrig`, `errorBalanceDest`

**Scenario Description**

| Field | Value |
| --- | --- |
| `step` | 2 |
| `type` | TRANSFER |
| `amount` | 150,000,000.00 |
| `oldbalanceOrg` | 150,000,000.00 |
| `newbalanceOrig` | 0.00 |
| `oldbalanceDest` | 0.00 |
| `newbalanceDest` | 0.00 |
| `isFlaggedFraud` | 0 |

**Business Context**

A transfer for $150,000,000 fully drains its origin account, with the destination account never credited — testing extrapolation far beyond any amount seen during training.

**Dataset/Feature Rationale**

*Source evidence:* The engineered dataset's fraud amounts cap at exactly $10,000,000, and even legitimate TRANSFER amounts max out near $92.4M; $150M exceeds both, directly probing out-of-distribution behavior.

*Expected reasoning:* The origin-drain and destination-untouched signatures should still dominate even at a scale roughly 15x the largest fraud amount seen during training ($10,000,000), testing whether the model extrapolates sensibly.

### E3: Perfectly Reconciled Large Transfer

**Expected outcome:** `legitimate` | **Engineered features exercised:** `errorBalanceOrig`, `errorBalanceDest`

**Scenario Description**

| Field | Value |
| --- | --- |
| `step` | 15 |
| `type` | TRANSFER |
| `amount` | 900,000.00 |
| `oldbalanceOrg` | 1,500,000.00 |
| `newbalanceOrig` | 600,000.00 |
| `oldbalanceDest` | 300,000.00 |
| `newbalanceDest` | 1,200,000.00 |
| `isFlaggedFraud` | 0 |

**Business Context**

A $900,000 transfer where the origin and destination balances reconcile exactly to the cent — zero ledger error on either side.

**Dataset/Feature Rationale**

*Source evidence:* errorBalanceOrig/errorBalanceDest are the top two engineered features by mean |SHAP| value; this scenario isolates their effect at exactly zero while varying nothing else suspicious.

*Expected reasoning:* With errorBalanceOrig and errorBalanceDest both exactly 0, and a substantial remaining origin balance ($600,000), this tests whether perfect ledger consistency is enough to be scored as legitimate at a large-but-not-extreme amount.

### E4: Minimal Balances Throughout (Dormant Account)

**Expected outcome:** `uncertain` | **Engineered features exercised:** `isOrigZeroBalance`, `amount`

**Scenario Description**

| Field | Value |
| --- | --- |
| `step` | 6 |
| `type` | TRANSFER |
| `amount` | 10.00 |
| `oldbalanceOrg` | 10.00 |
| `newbalanceOrig` | 0.00 |
| `oldbalanceDest` | 0.00 |
| `newbalanceDest` | 10.00 |
| `isFlaggedFraud` | 0 |

**Business Context**

A near-empty, likely dormant account transfers its entire $10 balance.

**Dataset/Feature Rationale**

*Source evidence:* Both oldbalanceOrg and oldbalanceDest have a real minimum of $0.00 in the engineered dataset; this scenario is a plausible low-activity account rather than a fabricated extreme.

*Expected reasoning:* Every balance field is at or near zero; this exercises the low end of the feature space where signal is weakest and the outcome is genuinely uncertain.

### E5: Amount Exceeds Stated Origin Balance

**Expected outcome:** `fraud_high_confidence` | **Engineered features exercised:** `errorBalanceOrig`

**Scenario Description**

| Field | Value |
| --- | --- |
| `step` | 1 |
| `type` | CASH_OUT |
| `amount` | 5,000.00 |
| `oldbalanceOrg` | 1,000.00 |
| `newbalanceOrig` | 0.00 |
| `oldbalanceDest` | 0.00 |
| `newbalanceDest` | 0.00 |
| `isFlaggedFraud` | 0 |

**Business Context**

A $5,000 cash-out is recorded against an account that shows only $1,000 available beforehand — an internally inconsistent ledger record.

**Dataset/Feature Rationale**

*Source evidence:* A large positive errorBalanceOrig is the strongest fraud signal by mean |SHAP| value in Sprint 5's analysis; this scenario constructs that condition directly via an overdraft-like ledger anomaly, a realistic account-takeover artifact.

*Expected reasoning:* errorBalanceOrig (native importance rank #3 (0.2685); mean |SHAP| rank #1 (4.0605)) should spike to $4,000 (0 + 5,000 - 1,000), a clear ledger-inconsistency signal on top of an origin-drain and destination-untouched pattern.

### E6: Day-Boundary Transaction

**Expected outcome:** `legitimate` | **Engineered features exercised:** `transactionDay`, `transactionHour`

**Scenario Description**

| Field | Value |
| --- | --- |
| `step` | 48 |
| `type` | PAYMENT |
| `amount` | 45.00 |
| `oldbalanceOrg` | 0.00 |
| `newbalanceOrig` | 0.00 |
| `oldbalanceDest` | 0.00 |
| `newbalanceDest` | 0.00 |
| `isFlaggedFraud` | 0 |

**Business Context**

A small routine payment occurs at the exact simulated-day rollover (step 48, hour 0).

**Dataset/Feature Rationale**

*Source evidence:* step=48 is an exact multiple of 24, the precise boundary where transactionHour wraps to 0.

*Expected reasoning:* transactionDay/transactionHour should compute cleanly to (2, 0); this tests the day-rollover arithmetic (`step // 24`, `step % 24`) at an exact boundary.
