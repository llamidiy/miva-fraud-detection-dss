# Validation Report

This report records how the fraud detection decision support system responded to 20 realistic business scenarios (see `business_scenarios.md` for why each scenario was constructed), and compares the actual recommendation against the expected outcome anticipated when the scenario was designed.

## Summary

- **Scenarios executed:** 20
- **Validation status:** 20/20 PASS
- **Expected vs. actual outcome:** 15 match, 3 mismatch, 2 without a strong prior expectation (mismatches: F3, F6a, E5)
- **Execution time:** 29.16s for 20 scenarios (1.46s/scenario)

## Recommendation Rules

Recommendations are generated deterministically from the prediction class and confidence score alone — no additional model logic is involved:

- **Fraud, confidence >= 90%:** Flag immediately, Suspend pending investigation, Escalate to fraud analyst
- **Fraud, confidence < 90%:** Request additional verification, Monitor subsequent activity
- **Not Fraud (any confidence):** Approve transaction, Continue routine monitoring

## Fraud Scenarios

### F1: Unusually High-Value Transfer With Origin Fully Drained

**Prediction:** Fraud

**Confidence:** 100.0%

**Top SHAP Factors**

- Low change in origin account balance
- Low balance discrepancy on the origin account
- Low transaction time step

**Recommendation:** Flag immediately, Suspend pending investigation, Escalate to fraud analyst

**Validation Result:** PASS (prediction=True, explanation=True, recommendation=True, confidence=True) — expected vs. actual: Match

**Notes:** Actual recommendation tier (`fraud_high_confidence`) matched the expected outcome.

### F2: Cash-Out With Zero Origin Balance After Transaction

**Prediction:** Fraud

**Confidence:** 100.0%

**Top SHAP Factors**

- Low change in origin account balance
- Low balance discrepancy on the origin account
- Low transaction time step

**Recommendation:** Flag immediately, Suspend pending investigation, Escalate to fraud analyst

**Validation Result:** PASS (prediction=True, explanation=True, recommendation=True, confidence=True) — expected vs. actual: Match

**Notes:** Actual recommendation tier (`fraud_high_confidence`) matched the expected outcome.

### F3: Abnormal Balance Discrepancy (Ledger Inconsistency)

**Prediction:** Not Fraud

**Confidence:** 100.0%

**Top SHAP Factors**

- Low balance discrepancy on the origin account
- High origin account's ending balance
- Low transaction time step

**Recommendation:** Approve transaction, Continue routine monitoring

**Validation Result:** PASS (prediction=True, explanation=True, recommendation=True, confidence=True) — expected vs. actual: Mismatch

**Notes:** Actual recommendation tier (`legitimate`) did **not** match the expected outcome (`fraud_high_confidence`) — worth further review.

### F4: Destination Account Never Credited (Funds Vanish)

**Prediction:** Fraud

**Confidence:** 100.0%

**Top SHAP Factors**

- Low balance discrepancy on the origin account
- Low change in origin account balance
- Low transaction time step

**Recommendation:** Flag immediately, Suspend pending investigation, Escalate to fraud analyst

**Validation Result:** PASS (prediction=True, explanation=True, recommendation=True, confidence=True) — expected vs. actual: Match

**Notes:** Actual recommendation tier (`fraud_high_confidence`) matched the expected outcome.

### F5: Large Transfer at an Off-Peak Hour

**Prediction:** Fraud

**Confidence:** 100.0%

**Top SHAP Factors**

- Low change in origin account balance
- Low balance discrepancy on the origin account
- Low transaction time step

**Recommendation:** Flag immediately, Suspend pending investigation, Escalate to fraud analyst

**Validation Result:** PASS (prediction=True, explanation=True, recommendation=True, confidence=True) — expected vs. actual: Match

**Notes:** Actual recommendation tier (`fraud_high_confidence`) matched the expected outcome.

### F6a: Structuring Pattern — Part 1 of 2

**Prediction:** Not Fraud

**Confidence:** 100.0%

**Top SHAP Factors**

- High origin account's ending balance
- Low change in origin account balance
- Low balance discrepancy on the origin account

**Recommendation:** Approve transaction, Continue routine monitoring

**Validation Result:** PASS (prediction=True, explanation=True, recommendation=True, confidence=True) — expected vs. actual: Mismatch

**Notes:** Actual recommendation tier (`legitimate`) did **not** match the expected outcome (`fraud_medium_confidence`) — worth further review.

### F6b: Structuring Pattern — Part 2 of 2

**Prediction:** Fraud

**Confidence:** 100.0%

**Top SHAP Factors**

- Low balance discrepancy on the origin account
- Low change in origin account balance
- High origin account's ending balance

**Recommendation:** Flag immediately, Suspend pending investigation, Escalate to fraud analyst

**Validation Result:** PASS (prediction=True, explanation=True, recommendation=True, confidence=True) — expected vs. actual: Match

**Notes:** Actual recommendation tier (`fraud_high_confidence`) matched the expected outcome.

## Legitimate Scenarios

### L1: Monthly Salary Deposit

**Prediction:** Not Fraud

**Confidence:** 100.0%

**Top SHAP Factors**

- Low transaction amount
- Low transaction amount (log-scaled)
- High origin account's ending balance

**Recommendation:** Approve transaction, Continue routine monitoring

**Validation Result:** PASS (prediction=True, explanation=True, recommendation=True, confidence=True) — expected vs. actual: Match

**Notes:** Actual recommendation tier (`legitimate`) matched the expected outcome.

### L2: Routine Household Bill Payment

**Prediction:** Not Fraud

**Confidence:** 100.0%

**Top SHAP Factors**

- High change in origin account balance
- Low transaction amount
- Low balance discrepancy on the origin account

**Recommendation:** Approve transaction, Continue routine monitoring

**Validation Result:** PASS (prediction=True, explanation=True, recommendation=True, confidence=True) — expected vs. actual: Match

**Notes:** Actual recommendation tier (`legitimate`) matched the expected outcome.

### L3: Small Peer-to-Peer Transfer

**Prediction:** Not Fraud

**Confidence:** 100.0%

**Top SHAP Factors**

- High change in origin account balance
- Low balance discrepancy on the origin account
- Low transaction amount

**Recommendation:** Approve transaction, Continue routine monitoring

**Validation Result:** PASS (prediction=True, explanation=True, recommendation=True, confidence=True) — expected vs. actual: Match

**Notes:** Actual recommendation tier (`legitimate`) matched the expected outcome.

### L4: Regular Merchant Payment

**Prediction:** Not Fraud

**Confidence:** 100.0%

**Top SHAP Factors**

- High change in origin account balance
- Low transaction amount
- Low balance discrepancy on the origin account

**Recommendation:** Approve transaction, Continue routine monitoring

**Validation Result:** PASS (prediction=True, explanation=True, recommendation=True, confidence=True) — expected vs. actual: Match

**Notes:** Actual recommendation tier (`legitimate`) matched the expected outcome.

### L5: Typical Debit Transaction

**Prediction:** Not Fraud

**Confidence:** 100.0%

**Top SHAP Factors**

- High origin account's ending balance
- Low transaction amount
- Low transaction amount (log-scaled)

**Recommendation:** Approve transaction, Continue routine monitoring

**Validation Result:** PASS (prediction=True, explanation=True, recommendation=True, confidence=True) — expected vs. actual: Match

**Notes:** Actual recommendation tier (`legitimate`) matched the expected outcome.

### L6: Large but Legitimate Business Transfer

**Prediction:** Not Fraud

**Confidence:** 100.0%

**Top SHAP Factors**

- High origin account's ending balance
- Low change in origin account balance
- High destination account's ending balance

**Recommendation:** Approve transaction, Continue routine monitoring

**Validation Result:** PASS (prediction=True, explanation=True, recommendation=True, confidence=True) — expected vs. actual: Match

**Notes:** Actual recommendation tier (`legitimate`) matched the expected outcome.

### L7: Routine Cash Withdrawal

**Prediction:** Not Fraud

**Confidence:** 100.0%

**Top SHAP Factors**

- High origin account's ending balance
- Low transaction amount
- Low balance discrepancy on the origin account

**Recommendation:** Approve transaction, Continue routine monitoring

**Validation Result:** PASS (prediction=True, explanation=True, recommendation=True, confidence=True) — expected vs. actual: Match

**Notes:** Actual recommendation tier (`legitimate`) matched the expected outcome.

## Edge Cases

### E1: Zero-Amount Transaction

**Prediction:** Not Fraud

**Confidence:** 100.0%

**Top SHAP Factors**

- High origin account's ending balance
- Low transaction amount
- High change in origin account balance

**Recommendation:** Approve transaction, Continue routine monitoring

**Validation Result:** PASS (prediction=True, explanation=True, recommendation=True, confidence=True) — expected vs. actual: N/A (no strong prior expectation)

**Notes:** No strong prior expectation was set for this scenario (genuine edge case).

### E2: Extremely Large Amount Beyond the Training Range

**Prediction:** Fraud

**Confidence:** 100.0%

**Top SHAP Factors**

- Low change in origin account balance
- Low balance discrepancy on the origin account
- Low transaction time step

**Recommendation:** Flag immediately, Suspend pending investigation, Escalate to fraud analyst

**Validation Result:** PASS (prediction=True, explanation=True, recommendation=True, confidence=True) — expected vs. actual: Match

**Notes:** Actual recommendation tier (`fraud_high_confidence`) matched the expected outcome.

### E3: Perfectly Reconciled Large Transfer

**Prediction:** Not Fraud

**Confidence:** 100.0%

**Top SHAP Factors**

- High origin account's ending balance
- Low change in origin account balance
- Low transaction time step

**Recommendation:** Approve transaction, Continue routine monitoring

**Validation Result:** PASS (prediction=True, explanation=True, recommendation=True, confidence=True) — expected vs. actual: Match

**Notes:** Actual recommendation tier (`legitimate`) matched the expected outcome.

### E4: Minimal Balances Throughout (Dormant Account)

**Prediction:** Fraud

**Confidence:** 100.0%

**Top SHAP Factors**

- Low balance discrepancy on the origin account
- High origin account's ending balance
- Low transaction amount (log-scaled)

**Recommendation:** Flag immediately, Suspend pending investigation, Escalate to fraud analyst

**Validation Result:** PASS (prediction=True, explanation=True, recommendation=True, confidence=True) — expected vs. actual: N/A (no strong prior expectation)

**Notes:** No strong prior expectation was set for this scenario (genuine edge case).

### E5: Amount Exceeds Stated Origin Balance

**Prediction:** Not Fraud

**Confidence:** 100.0%

**Top SHAP Factors**

- Low balance discrepancy on the origin account
- Not zero origin balance
- Low transaction amount

**Recommendation:** Approve transaction, Continue routine monitoring

**Validation Result:** PASS (prediction=True, explanation=True, recommendation=True, confidence=True) — expected vs. actual: Mismatch

**Notes:** Actual recommendation tier (`legitimate`) did **not** match the expected outcome (`fraud_high_confidence`) — worth further review.

### E6: Day-Boundary Transaction

**Prediction:** Not Fraud

**Confidence:** 100.0%

**Top SHAP Factors**

- High change in origin account balance
- Low transaction amount
- Low balance discrepancy on the origin account

**Recommendation:** Approve transaction, Continue routine monitoring

**Validation Result:** PASS (prediction=True, explanation=True, recommendation=True, confidence=True) — expected vs. actual: Match

**Notes:** Actual recommendation tier (`legitimate`) matched the expected outcome.
