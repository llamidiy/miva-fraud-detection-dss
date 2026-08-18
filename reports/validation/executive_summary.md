# Executive Validation Summary

This document summarizes the Sprint 5.5 business-scenario validation of the fraud detection decision support system, for dissertation Chapter 5, supervisor review, and project defense. It is generated entirely from the existing validation outputs (`scenario_results.csv`, `validation_report.md`, `business_scenarios.md`); no scenarios were re-run and no model or SHAP computation occurred while producing this summary.

## 1. Validation Overview

- **Validation date:** 2026-08-07 13:57 UTC
- **Number of scenarios:** 20
- **Fraud scenarios:** 7
- **Legitimate scenarios:** 7
- **Edge cases:** 6

## 2. Overall Results

| Metric | Count |
| --- | --- |
| Total scenarios executed | 20 |
| Successful executions (PASS) | 20 |
| Validation failures (FAIL) | 0 |
| Expected vs. actual matches | 15 |
| Expected vs. actual mismatches | 3 |
| Exploratory / N/A scenarios | 2 |

## 3. Key Findings

- High-confidence fraud detection: 5/7 fraud scenarios were predicted as Fraud at or above the 90% confidence threshold (5/7 predicted Fraud in total).
- Legitimate transaction handling: 7/7 legitimate scenarios were correctly predicted as Not Fraud.
- Edge-case behaviour: 6 boundary-condition scenarios produced 2 Fraud and 4 Not Fraud predictions, with 0 runtime failure(s).
- Confidence distribution: 100.0%-100.0% across all scenarios (mean 100.0%, median 100.0%).
- Explainability: SHAP-based explanations were successfully generated for 20/20 scenarios.

## 4. Business Insights

- Fraud behaviours consistently detected at high confidence: F1 (Unusually High-Value Transfer With Origin Fully Drained), F2 (Cash-Out With Zero Origin Balance After Transaction), F4 (Destination Account Never Credited (Funds Vanish)), F5 (Large Transfer at an Off-Peak Hour), F6b (Structuring Pattern — Part 2 of 2). These scenarios share the dataset's dominant fraud signature — a fully or substantially drained origin account on a TRANSFER/CASH_OUT transaction.
- Behaviours requiring analyst review: scenarios F3, F6a, E5 did not match the expected decision (see Section 5) and represent cases better suited to human review than fully automated action.
- Explanations are particularly valuable in the mismatched scenarios (F3, F6a, E5): the SHAP factors reported for each make it possible to see exactly which balance signal drove the model's decision, rather than leaving the discrepancy from expectation unexplained.
- Scenarios E1, E4 were deliberately constructed without a strong prior expectation (genuine boundary conditions); explanations there support interpretation of the result rather than confirming or refuting a hypothesis.
- The recommendation engine produced a valid, deterministic action for all 20 scenarios across 2 distinct tiers, giving every prediction — including the mismatched and exploratory ones — a clear, auditable next step rather than a bare probability score.

## 5. Notable Exceptions

Scenarios where the expected decision did not match the actual recommendation. These are reported in full rather than omitted, as they are the most valuable signal for where the system's behaviour should be scrutinized further.

| Scenario ID | Scenario Name | Expected Decision | Actual Decision | Confidence | Short Explanation |
| --- | --- | --- | --- | --- | --- |
| F3 | Abnormal Balance Discrepancy (Ledger Inconsistency) | High-confidence fraud (flag / suspend / escalate) | Legitimate | 100.0% | A transfer's reported ending origin balance ($50,000) does not reconcile with the starting balance minus the transferred amount, suggesting a manipulated or corrupted account record. Model predicted Not Fraud at 100.0% confidence, versus the expected high-confidence fraud (flag / suspend / escalate). |
| F6a | Structuring Pattern — Part 1 of 2 | Medium-confidence fraud (verify / monitor) | Legitimate | 100.0% | The first of two same-day transfers that together drain a $960,000 account in two roughly equal steps rather than one large transaction — a common structuring tactic to avoid drawing attention to a single large transfer. Model predicted Not Fraud at 100.0% confidence, versus the expected medium-confidence fraud (verify / monitor). |
| E5 | Amount Exceeds Stated Origin Balance | High-confidence fraud (flag / suspend / escalate) | Legitimate | 100.0% | A $5,000 cash-out is recorded against an account that shows only $1,000 available beforehand — an internally inconsistent ledger record. Model predicted Not Fraud at 100.0% confidence, versus the expected high-confidence fraud (flag / suspend / escalate). |

## 6. Decision Support Assessment

**Prediction capability.** The champion model produced a valid prediction for 20/20 scenarios spanning known fraud patterns, legitimate transaction types across 5 PaySim transaction types (CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER), and deliberate boundary conditions, with no runtime failures observed.

**Explainability.** SHAP-based, feature-level explanations were generated for 20/20 scenarios, each traceable to specific, auditable transaction fields (balance changes, ledger discrepancies) rather than an opaque score.

**Recommendation quality.** Recommendations follow the same fixed, transparent rules used throughout Sprint 5.5:

Recommendations are generated deterministically from the prediction class and confidence score alone — no additional model logic is involved:

- **Fraud, confidence >= 90%:** Flag immediately, Suspend pending investigation, Escalate to fraud analyst
- **Fraud, confidence < 90%:** Request additional verification, Monitor subsequent activity
- **Not Fraud (any confidence):** Approve transaction, Continue routine monitoring

**Suitability for analyst support.** These results support using the system as a triage and explanation aid for fraud analysts — surfacing high-confidence cases for expedited action and routing lower-confidence or mismatched cases for review — rather than as a fully autonomous decision-maker. The system assists human judgement; it does not replace it, and the mismatches documented in Section 5 illustrate concretely why analyst oversight remains necessary.

## 7. Limitations

Observed directly from this validation run:

- **Uncertainty around small/moderate-value ledger anomalies.** Scenarios F3, E5 constructed a balance discrepancy pattern similar to known fraud signatures but at a smaller dollar magnitude than typical training-set fraud, and were not flagged as expected — suggesting sensitivity to anomaly patterns may scale with transaction size.
- **Dependence on historical PaySim patterns.** Every scenario's rationale (see `business_scenarios.md`) is grounded in statistics from the PaySim simulation dataset; behaviour on real banking data with different fraud patterns is untested.
- **Deterministic recommendation rules.** The recommendation engine maps prediction class and confidence to a fixed action list via a single confidence threshold (90%); it does not learn or adapt from outcomes.
- **No ground truth for genuine boundary conditions.** Scenarios E1, E4 were constructed with no strong prior expectation; their validity can be assessed for robustness (did the pipeline run without error) but not for correctness.
- **Single-transaction scoring.** The classifier scores each transaction independently; multi-step patterns such as structuring (see scenarios F6a/F6b in `business_scenarios.md`) are only partially represented as separate, related scenarios.

## 8. Recommendations

Realistic next steps suggested by this validation run:

1. **Review confidence-threshold calibration** in light of the 3 mismatched scenario(s) (F3, F6a, E5), particularly around moderate-magnitude balance discrepancies that were not flagged as expected.
2. **Add targeted business rules for structuring/sequential patterns**, since the current single-transaction classifier cannot see related transactions across steps.
3. **Test against real banking transaction data** to confirm the PaySim-derived patterns generalize beyond the simulation.
4. **Establish a continuous retraining/monitoring cadence** so the model and its SHAP-based explanations stay aligned with evolving fraud patterns over time.
