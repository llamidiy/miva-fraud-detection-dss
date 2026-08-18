"""Boundary-condition scenarios for system robustness testing.

Complements `scenario_generator.py`'s fraud/legitimate scenarios with
transactions at the edges of the feature space — zero/extreme amounts,
perfectly reconciled ledgers, and internally inconsistent records — to
confirm the system produces a valid prediction, explanation, and
recommendation everywhere, not just on "typical" transactions.
"""

from src.validation.scenario_generator import Scenario, cite_feature_evidence


def build_edge_case_scenarios() -> list[Scenario]:
    """Construct the edge-case scenario set.

    Returns:
        The list of boundary-condition scenarios.
    """
    return [
        Scenario(
            scenario_id="E1",
            title="Zero-Amount Transaction",
            category="edge_case",
            business_description="A payment is submitted with a transaction amount of exactly $0.00.",
            transaction={
                "step": 12,
                "type": "PAYMENT",
                "amount": 0.0,
                "oldbalanceOrg": 5_000.0,
                "newbalanceOrig": 5_000.0,
                "oldbalanceDest": 0.0,
                "newbalanceDest": 0.0,
                "isFlaggedFraud": 0,
            },
            expected_reasoning=(
                "logAmount collapses to log1p(0) = 0; this tests whether the model and pipeline "
                "handle the degenerate zero-amount boundary without error."
            ),
            source_rationale=(
                "The engineered dataset's amount column has a real minimum of $0.00 (legitimate "
                "transactions), so this boundary is genuinely reachable in production, not purely "
                "synthetic."
            ),
            expected_outcome="uncertain",
            tags=["logAmount", "amount"],
        ),
        Scenario(
            scenario_id="E2",
            title="Extremely Large Amount Beyond the Training Range",
            category="edge_case",
            business_description=(
                "A transfer for $150,000,000 fully drains its origin account, with the "
                "destination account never credited — testing extrapolation far beyond any "
                "amount seen during training."
            ),
            transaction={
                "step": 2,
                "type": "TRANSFER",
                "amount": 150_000_000.0,
                "oldbalanceOrg": 150_000_000.0,
                "newbalanceOrig": 0.0,
                "oldbalanceDest": 0.0,
                "newbalanceDest": 0.0,
                "isFlaggedFraud": 0,
            },
            expected_reasoning=(
                "The origin-drain and destination-untouched signatures should still dominate "
                "even at a scale roughly 15x the largest fraud amount seen during training "
                "($10,000,000), testing whether the model extrapolates sensibly."
            ),
            source_rationale=(
                "The engineered dataset's fraud amounts cap at exactly $10,000,000, and even "
                "legitimate TRANSFER amounts max out near $92.4M; $150M exceeds both, directly "
                "probing out-of-distribution behavior."
            ),
            expected_outcome="fraud_high_confidence",
            tags=["amount", "balanceDeltaOrig", "errorBalanceDest"],
        ),
        Scenario(
            scenario_id="E3",
            title="Perfectly Reconciled Large Transfer",
            category="edge_case",
            business_description=(
                "A $900,000 transfer where the origin and destination balances reconcile "
                "exactly to the cent — zero ledger error on either side."
            ),
            transaction={
                "step": 15,
                "type": "TRANSFER",
                "amount": 900_000.0,
                "oldbalanceOrg": 1_500_000.0,
                "newbalanceOrig": 600_000.0,
                "oldbalanceDest": 300_000.0,
                "newbalanceDest": 1_200_000.0,
                "isFlaggedFraud": 0,
            },
            expected_reasoning=(
                "With errorBalanceOrig and errorBalanceDest both exactly 0, and a substantial "
                "remaining origin balance ($600,000), this tests whether perfect ledger "
                "consistency is enough to be scored as legitimate at a large-but-not-extreme amount."
            ),
            source_rationale=(
                "errorBalanceOrig/errorBalanceDest are the top two engineered features by mean "
                "|SHAP| value; this scenario isolates their effect at exactly zero while varying "
                "nothing else suspicious."
            ),
            expected_outcome="legitimate",
            tags=["errorBalanceOrig", "errorBalanceDest"],
        ),
        Scenario(
            scenario_id="E4",
            title="Minimal Balances Throughout (Dormant Account)",
            category="edge_case",
            business_description="A near-empty, likely dormant account transfers its entire $10 balance.",
            transaction={
                "step": 6,
                "type": "TRANSFER",
                "amount": 10.0,
                "oldbalanceOrg": 10.0,
                "newbalanceOrig": 0.0,
                "oldbalanceDest": 0.0,
                "newbalanceDest": 10.0,
                "isFlaggedFraud": 0,
            },
            expected_reasoning=(
                "Every balance field is at or near zero; this exercises the low end of the "
                "feature space where signal is weakest and the outcome is genuinely uncertain."
            ),
            source_rationale=(
                "Both oldbalanceOrg and oldbalanceDest have a real minimum of $0.00 in the "
                "engineered dataset; this scenario is a plausible low-activity account rather "
                "than a fabricated extreme."
            ),
            expected_outcome="uncertain",
            tags=["isOrigZeroBalance", "amount"],
        ),
        Scenario(
            scenario_id="E5",
            title="Amount Exceeds Stated Origin Balance",
            category="edge_case",
            business_description=(
                "A $5,000 cash-out is recorded against an account that shows only $1,000 "
                "available beforehand — an internally inconsistent ledger record."
            ),
            transaction={
                "step": 1,
                "type": "CASH_OUT",
                "amount": 5_000.0,
                "oldbalanceOrg": 1_000.0,
                "newbalanceOrig": 0.0,
                "oldbalanceDest": 0.0,
                "newbalanceDest": 0.0,
                "isFlaggedFraud": 0,
            },
            expected_reasoning=(
                f"errorBalanceOrig ({cite_feature_evidence('errorBalanceOrig')}) should spike to "
                "$4,000 (0 + 5,000 - 1,000), a clear ledger-inconsistency signal on top of an "
                "origin-drain and destination-untouched pattern."
            ),
            source_rationale=(
                "A large positive errorBalanceOrig is the strongest fraud signal by mean |SHAP| "
                "value in Sprint 5's analysis; this scenario constructs that condition directly "
                "via an overdraft-like ledger anomaly, a realistic account-takeover artifact."
            ),
            expected_outcome="fraud_high_confidence",
            tags=["errorBalanceOrig"],
        ),
        Scenario(
            scenario_id="E6",
            title="Day-Boundary Transaction",
            category="edge_case",
            business_description="A small routine payment occurs at the exact simulated-day rollover (step 48, hour 0).",
            transaction={
                "step": 48,
                "type": "PAYMENT",
                "amount": 45.0,
                "oldbalanceOrg": 0.0,
                "newbalanceOrig": 0.0,
                "oldbalanceDest": 0.0,
                "newbalanceDest": 0.0,
                "isFlaggedFraud": 0,
            },
            expected_reasoning=(
                "transactionDay/transactionHour should compute cleanly to (2, 0); this tests the "
                "day-rollover arithmetic (`step // 24`, `step % 24`) at an exact boundary."
            ),
            source_rationale="step=48 is an exact multiple of 24, the precise boundary where transactionHour wraps to 0.",
            expected_outcome="legitimate",
            tags=["transactionDay", "transactionHour"],
        ),
    ]
