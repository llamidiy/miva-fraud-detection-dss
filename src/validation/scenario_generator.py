"""Business scenario definitions for system validation.

Scenarios are not arbitrary — every transaction pattern below is
grounded in a statistic pulled from the raw PaySim dataset, the
engineered dataset, or Sprint 5's frozen `feature_importance.csv` and
`shap_values.joblib`. See each scenario's `source_rationale` for the
specific evidence behind it.

Key findings from that analysis (engineered dataset, 6,362,620 rows):

- Fraud occurs *only* in `TRANSFER` (4,097 cases) and `CASH_OUT` (4,116
  cases); zero fraud in `CASH_IN`, `PAYMENT`, or `DEBIT`.
- 98.05% of fraudulent transactions leave `newbalanceOrig == 0` (the
  origin account fully drained).
- 49.63% of fraudulent transactions leave the destination account
  balance completely unchanged (`oldbalanceDest == newbalanceDest == 0`
  — funds never actually arrive anywhere in this snapshot).
- Fraud amounts are large (mean $1,467,967, median $441,423) versus
  legitimate transactions (mean $178,197, median $74,685) — but
  legitimate `TRANSFER`s can also be large (90th percentile $1.77M), so
  amount alone does not separate the classes.
- `isFlaggedFraud` (the existing rule-based control) is 1 for only
  0.19% of actual fraud cases, consistent with Sprint 2/3's finding
  that it is an unreliable signal — every scenario below leaves it at 0.
- Fraud is distributed near-uniformly across all 24 hours (~340
  cases/hour), while legitimate activity concentrates around hours
  12-13 and 18-20. `transactionHour` is therefore *not* a strong global
  driver on its own (it does not appear in the top-10 native importance
  or top-5 mean |SHAP| features), so hour-based scenarios here test
  whether the model still relies on balance signals rather than
  time-of-day cues, matching how real fraud-monitoring rules often flag
  high-value activity during low-legitimate-volume windows regardless
  of what the ML model itself weights.
"""

import functools
import logging
from dataclasses import dataclass, field
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import FEATURE_IMPORTANCE_PATH, SHAP_VALUES_PATH

logger = logging.getLogger(__name__)

#: Valid values for `Scenario.category`.
CATEGORIES = ("fraud", "legitimate", "edge_case")

#: Valid values for `Scenario.expected_outcome` — the recommendation-engine
#: tier a domain-informed reviewer would expect, or "uncertain" where the
#: scenario deliberately has no strong prior.
EXPECTED_OUTCOMES = ("fraud_high_confidence", "fraud_medium_confidence", "legitimate", "uncertain")

#: Raw PaySim-style fields every scenario's `transaction` dict must supply.
#: Engineered columns (balanceDeltaOrig, errorBalanceOrig, logAmount, ...)
#: are derived automatically by the frozen `engineer_features` function —
#: see `src.validation.validator.build_transaction_dataframe`.
RAW_TRANSACTION_FIELDS = (
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "isFlaggedFraud",
)


@dataclass
class Scenario:
    """A single business validation scenario.

    Attributes:
        scenario_id: Short, stable identifier (e.g. ``"F1"``).
        title: Descriptive title.
        category: One of `CATEGORIES`.
        business_description: The real-world situation this transaction
            represents.
        transaction: Raw PaySim-style fields (see
            `RAW_TRANSACTION_FIELDS`) describing the transaction.
        expected_reasoning: What SHAP-visible signal we expect to drive
            the model's decision.
        source_rationale: The specific dataset/model evidence that
            motivated this scenario.
        expected_outcome: One of `EXPECTED_OUTCOMES` — the anticipated
            recommendation-engine tier, used later to check the actual
            recommendation against expectation.
    """

    scenario_id: str
    title: str
    category: str
    business_description: str
    transaction: dict[str, Any]
    expected_reasoning: str
    source_rationale: str
    expected_outcome: str
    tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.category not in CATEGORIES:
            raise ValueError(f"Invalid category '{self.category}'. Must be one of {CATEGORIES}.")
        if self.expected_outcome not in EXPECTED_OUTCOMES:
            raise ValueError(
                f"Invalid expected_outcome '{self.expected_outcome}'. Must be one of {EXPECTED_OUTCOMES}."
            )
        missing = [f for f in RAW_TRANSACTION_FIELDS if f not in self.transaction]
        if missing:
            raise ValueError(f"Scenario '{self.scenario_id}' transaction is missing fields: {missing}")


@functools.lru_cache(maxsize=1)
def _load_feature_importance() -> pd.DataFrame:
    """Load Sprint 5's frozen native feature importance table (read-only reuse)."""
    return pd.read_csv(FEATURE_IMPORTANCE_PATH)


@functools.lru_cache(maxsize=1)
def _load_shap_mean_abs() -> pd.Series:
    """Load Sprint 5's frozen cached SHAP values and rank features by mean |SHAP| (read-only reuse)."""
    cached = joblib.load(SHAP_VALUES_PATH)
    explanation = cached["explanation"]
    mean_abs = np.abs(explanation.values).mean(axis=0)
    return pd.Series(mean_abs, index=explanation.feature_names).sort_values(ascending=False)


def cite_feature_evidence(feature: str) -> str:
    """Build a short, data-grounded citation for why `feature` matters.

    Reads directly from Sprint 5's frozen `feature_importance.csv` and
    `shap_values.joblib`, so the citation reflects the actual computed
    rankings rather than a hardcoded claim.

    Args:
        feature: An encoded feature column name (e.g. ``"errorBalanceOrig"``).

    Returns:
        A short citation string, e.g. ``"native importance rank #3
        (0.2685); mean |SHAP| rank #1 (4.0605)"``.
    """
    parts = []

    importance_df = _load_feature_importance().reset_index(drop=True)
    match = importance_df.index[importance_df["feature"] == feature]
    if len(match):
        rank = int(match[0]) + 1
        value = importance_df.loc[match[0], "importance"]
        parts.append(f"native importance rank #{rank} ({value:.4f})")

    shap_ranking = _load_shap_mean_abs()
    if feature in shap_ranking.index:
        rank = int(shap_ranking.index.get_loc(feature)) + 1
        parts.append(f"mean |SHAP| rank #{rank} ({shap_ranking[feature]:.4f})")

    return "; ".join(parts) if parts else "not in the top-ranked feature set"


def _build_fraud_scenarios() -> list[Scenario]:
    """Construct the fraud scenario set, grounded in real fraud patterns."""
    return [
        Scenario(
            scenario_id="F1",
            title="Unusually High-Value Transfer With Origin Fully Drained",
            category="fraud",
            business_description=(
                "A customer's account is used to transfer its entire $5.2M balance out "
                "in a single transaction, with the destination account correctly credited."
            ),
            transaction={
                "step": 14,
                "type": "TRANSFER",
                "amount": 5_200_000.0,
                "oldbalanceOrg": 5_200_000.0,
                "newbalanceOrig": 0.0,
                "oldbalanceDest": 850_000.0,
                "newbalanceDest": 6_050_000.0,
                "isFlaggedFraud": 0,
            },
            expected_reasoning=(
                f"newbalanceOrig ({cite_feature_evidence('newbalanceOrig')}) and "
                f"balanceDeltaOrig ({cite_feature_evidence('balanceDeltaOrig')}) should dominate: "
                "the full-balance drain on a very large TRANSFER mirrors the single strongest "
                "fraud signature in the training data."
            ),
            source_rationale=(
                "98.05% of fraudulent transactions in the engineered dataset leave "
                "newbalanceOrig == 0; fraud amounts average $1,467,967 (median $441,423), and "
                "fraud never occurs outside TRANSFER/CASH_OUT."
            ),
            expected_outcome="fraud_high_confidence",
            tags=["balanceDeltaOrig", "newbalanceOrig"],
        ),
        Scenario(
            scenario_id="F2",
            title="Cash-Out With Zero Origin Balance After Transaction",
            category="fraud",
            business_description=(
                "A cash withdrawal empties an account of $650,000 (near the fraud-population "
                "median amount), with the receiving agent account properly credited."
            ),
            transaction={
                "step": 10,
                "type": "CASH_OUT",
                "amount": 650_000.0,
                "oldbalanceOrg": 650_000.0,
                "newbalanceOrig": 0.0,
                "oldbalanceDest": 120_000.0,
                "newbalanceDest": 770_000.0,
                "isFlaggedFraud": 0,
            },
            expected_reasoning=(
                f"isOrigZeroBalance ({cite_feature_evidence('isOrigZeroBalance')}) and "
                f"type_CASH_OUT ({cite_feature_evidence('type_CASH_OUT')}) should contribute "
                "strongly, reflecting the origin-drain signature on a fraud-eligible type."
            ),
            source_rationale=(
                "Fraud amount median is $441,423 in the engineered dataset; this scenario sits "
                "close to that median while reproducing the 98.05% origin-drain pattern for CASH_OUT."
            ),
            expected_outcome="fraud_high_confidence",
            tags=["isOrigZeroBalance", "type_CASH_OUT"],
        ),
        Scenario(
            scenario_id="F3",
            title="Abnormal Balance Discrepancy (Ledger Inconsistency)",
            category="fraud",
            business_description=(
                "A transfer's reported ending origin balance ($50,000) does not reconcile with "
                "the starting balance minus the transferred amount, suggesting a manipulated or "
                "corrupted account record."
            ),
            transaction={
                "step": 16,
                "type": "TRANSFER",
                "amount": 500_000.0,
                "oldbalanceOrg": 500_000.0,
                "newbalanceOrig": 50_000.0,
                "oldbalanceDest": 200_000.0,
                "newbalanceDest": 700_000.0,
                "isFlaggedFraud": 0,
            },
            expected_reasoning=(
                f"errorBalanceOrig ({cite_feature_evidence('errorBalanceOrig')}) should be the "
                "dominant factor: this scenario deliberately produces a $50,000 ledger "
                "discrepancy on the origin side (destination side reconciles cleanly)."
            ),
            source_rationale=(
                "errorBalanceOrig is the single strongest engineered feature by mean |SHAP| "
                "value in Sprint 5's analysis and ranks #3 in native XGBoost importance."
            ),
            expected_outcome="fraud_high_confidence",
            tags=["errorBalanceOrig"],
        ),
        Scenario(
            scenario_id="F4",
            title="Destination Account Never Credited (Funds Vanish)",
            category="fraud",
            business_description=(
                "A cash-out drains the origin account, but the receiving account balance never "
                "changes — the destination side of the ledger shows zero before and after."
            ),
            transaction={
                "step": 9,
                "type": "CASH_OUT",
                "amount": 300_000.0,
                "oldbalanceOrg": 300_000.0,
                "newbalanceOrig": 0.0,
                "oldbalanceDest": 0.0,
                "newbalanceDest": 0.0,
                "isFlaggedFraud": 0,
            },
            expected_reasoning=(
                f"errorBalanceDest ({cite_feature_evidence('errorBalanceDest')}) and "
                f"isDestZeroBalance ({cite_feature_evidence('isDestZeroBalance')}) should be "
                "prominent, reflecting funds that are debited from the origin but never land "
                "in the destination account in this snapshot."
            ),
            source_rationale=(
                "49.63% of fraudulent transactions in the engineered dataset leave "
                "oldbalanceDest == newbalanceDest == 0 — the single most common destination-side "
                "fraud pattern observed."
            ),
            expected_outcome="fraud_high_confidence",
            tags=["errorBalanceDest", "isDestZeroBalance"],
        ),
        Scenario(
            scenario_id="F5",
            title="Large Transfer at an Off-Peak Hour",
            category="fraud",
            business_description=(
                "A $2.2M transfer, fully draining the origin account, occurs at 3am — a time "
                "window with very little legitimate customer activity."
            ),
            transaction={
                "step": 3,
                "type": "TRANSFER",
                "amount": 2_200_000.0,
                "oldbalanceOrg": 2_200_000.0,
                "newbalanceOrig": 0.0,
                "oldbalanceDest": 500_000.0,
                "newbalanceDest": 2_700_000.0,
                "isFlaggedFraud": 0,
            },
            expected_reasoning=(
                "The origin-drain/amount signals (balanceDeltaOrig, newbalanceOrig) should "
                "drive the prediction rather than transactionHour itself, since this sprint's "
                "re-analysis found fraud distributed near-uniformly across all 24 hours."
            ),
            source_rationale=(
                "Legitimate activity concentrates around hours 12-13 and 18-20 (this sprint's "
                "re-analysis of the engineered dataset), while fraud volume is flat across hours "
                "(274-375 cases/hour). This scenario tests whether the model's decision holds up "
                "on the balance evidence alone at an hour with little legitimate precedent."
            ),
            expected_outcome="fraud_high_confidence",
            tags=["transactionHour", "balanceDeltaOrig"],
        ),
        Scenario(
            scenario_id="F6a",
            title="Structuring Pattern — Part 1 of 2",
            category="fraud",
            business_description=(
                "The first of two same-day transfers that together drain a $960,000 account in "
                "two roughly equal steps rather than one large transaction — a common structuring "
                "tactic to avoid drawing attention to a single large transfer."
            ),
            transaction={
                "step": 101,
                "type": "TRANSFER",
                "amount": 480_000.0,
                "oldbalanceOrg": 960_000.0,
                "newbalanceOrig": 480_000.0,
                "oldbalanceDest": 0.0,
                "newbalanceDest": 480_000.0,
                "isFlaggedFraud": 0,
            },
            expected_reasoning=(
                "A partial (50%) balance drain on a fraud-eligible TRANSFER type; individually "
                "more ambiguous than a full drain, testing the model on a less extreme variant "
                "of the dominant fraud pattern."
            ),
            source_rationale=(
                "PaySim/the champion model score transactions independently, so a real "
                "structuring pattern is represented here as two related, sequential scenarios "
                "(see F6b) rather than a single multi-transaction feature, an explicit "
                "limitation of a single-transaction classifier worth documenting."
            ),
            expected_outcome="fraud_medium_confidence",
            tags=["balanceDeltaOrig"],
        ),
        Scenario(
            scenario_id="F6b",
            title="Structuring Pattern — Part 2 of 2",
            category="fraud",
            business_description=(
                "The second transfer in the same structuring sequence as F6a: the remaining "
                "$480,000 is moved out shortly after, fully draining the account."
            ),
            transaction={
                "step": 102,
                "type": "TRANSFER",
                "amount": 480_000.0,
                "oldbalanceOrg": 480_000.0,
                "newbalanceOrig": 0.0,
                "oldbalanceDest": 480_000.0,
                "newbalanceDest": 960_000.0,
                "isFlaggedFraud": 0,
            },
            expected_reasoning=(
                f"newbalanceOrig ({cite_feature_evidence('newbalanceOrig')}) should dominate, "
                "reflecting the completed full drain by the end of the two-step sequence."
            ),
            source_rationale=(
                "Companion scenario to F6a; together they represent the 98.05% origin-drain "
                "pattern reached via two steps instead of one."
            ),
            expected_outcome="fraud_high_confidence",
            tags=["newbalanceOrig"],
        ),
    ]


def _build_legitimate_scenarios() -> list[Scenario]:
    """Construct the legitimate scenario set, grounded in real non-fraud patterns."""
    return [
        Scenario(
            scenario_id="L1",
            title="Monthly Salary Deposit",
            category="legitimate",
            business_description="An employer deposits a $4,800 salary payment into a customer's account.",
            transaction={
                "step": 33,
                "type": "CASH_IN",
                "amount": 4_800.0,
                "oldbalanceOrg": 45_200.0,
                "newbalanceOrig": 50_000.0,
                "oldbalanceDest": 800_000.0,
                "newbalanceDest": 804_800.0,
                "isFlaggedFraud": 0,
            },
            expected_reasoning=(
                "type_CASH_OUT-style flags should be absent; CASH_IN is never associated with "
                "fraud in the training data, and the balances reconcile cleanly on the origin "
                "side once CASH_IN's crediting direction is accounted for."
            ),
            source_rationale=(
                "Zero fraud cases in the engineered dataset are CASH_IN; a real CASH_IN sample "
                "row shows the same clean oldbalanceOrg -> newbalanceOrig increment pattern used here."
            ),
            expected_outcome="legitimate",
            tags=["type_CASH_IN"],
        ),
        Scenario(
            scenario_id="L2",
            title="Routine Household Bill Payment",
            category="legitimate",
            business_description="A customer pays a $125.50 utility bill through the app.",
            transaction={
                "step": 19,
                "type": "PAYMENT",
                "amount": 125.50,
                "oldbalanceOrg": 0.0,
                "newbalanceOrig": 0.0,
                "oldbalanceDest": 0.0,
                "newbalanceDest": 0.0,
                "isFlaggedFraud": 0,
            },
            expected_reasoning=(
                "isOrigZeroBalance and isDestZeroBalance will both be True here — identical to "
                "many fraud cases — so this scenario specifically tests whether the model relies "
                "on type/amount rather than the zero-balance flags in isolation, since PAYMENT is "
                "never fraudulent in the training data."
            ),
            source_rationale=(
                "A real PAYMENT sample row in the engineered dataset shows all four balance "
                "fields at exactly 0.0 (PaySim does not track running balances for PAYMENT "
                "merchant accounts); this scenario reproduces that exact pattern at a typical "
                "bill-payment amount (10th-90th percentile: $1,743-$28,811)."
            ),
            expected_outcome="legitimate",
            tags=["isOrigZeroBalance", "isDestZeroBalance", "type_PAYMENT"],
        ),
        Scenario(
            scenario_id="L3",
            title="Small Peer-to-Peer Transfer",
            category="legitimate",
            business_description="A customer sends a friend $250 via a person-to-person transfer.",
            transaction={
                "step": 42,
                "type": "TRANSFER",
                "amount": 250.0,
                "oldbalanceOrg": 0.0,
                "newbalanceOrig": 0.0,
                "oldbalanceDest": 3_200.0,
                "newbalanceDest": 3_450.0,
                "isFlaggedFraud": 0,
            },
            expected_reasoning=(
                "This is the sharpest discriminating test in the set: same TRANSFER type and "
                "the same isOrigZeroBalance=True flag as most fraud, but at a tiny amount with "
                "the destination properly credited. amount/logAmount should keep this scored as "
                "legitimate despite the shared flag."
            ),
            source_rationale=(
                "A real small legitimate TRANSFER in the engineered dataset (amount=$306.80) "
                "shows oldbalanceOrg == newbalanceOrig == 0 — proving isOrigZeroBalance is common "
                "in legitimate low-value transfers too, not an unambiguous fraud tell on its own."
            ),
            expected_outcome="legitimate",
            tags=["isOrigZeroBalance", "amount", "type_TRANSFER"],
        ),
        Scenario(
            scenario_id="L4",
            title="Regular Merchant Payment",
            category="legitimate",
            business_description="A customer pays an $890 merchant invoice.",
            transaction={
                "step": 61,
                "type": "PAYMENT",
                "amount": 890.0,
                "oldbalanceOrg": 0.0,
                "newbalanceOrig": 0.0,
                "oldbalanceDest": 0.0,
                "newbalanceDest": 0.0,
                "isFlaggedFraud": 0,
            },
            expected_reasoning="Same PAYMENT zero-balance pattern as L2, at a larger everyday amount.",
            source_rationale=(
                "PAYMENT amounts in the engineered dataset range from a 10th percentile of "
                "$1,743 up; $890 represents a smaller but still typical merchant payment."
            ),
            expected_outcome="legitimate",
            tags=["type_PAYMENT"],
        ),
        Scenario(
            scenario_id="L5",
            title="Typical Debit Transaction",
            category="legitimate",
            business_description="A customer makes a routine $3,200 debit against their account.",
            transaction={
                "step": 65,
                "type": "DEBIT",
                "amount": 3_200.0,
                "oldbalanceOrg": 15_000.0,
                "newbalanceOrig": 11_800.0,
                "oldbalanceDest": 200_000.0,
                "newbalanceDest": 203_200.0,
                "isFlaggedFraud": 0,
            },
            expected_reasoning="DEBIT is never associated with fraud in training data; balances reconcile cleanly on both sides.",
            source_rationale="Amount sits near the DEBIT-type median ($3,048.99) in the engineered dataset; zero fraud cases are DEBIT.",
            expected_outcome="legitimate",
            tags=["type_DEBIT"],
        ),
        Scenario(
            scenario_id="L6",
            title="Large but Legitimate Business Transfer",
            category="legitimate",
            business_description=(
                "A business moves $1.75M between its own accounts, leaving a substantial "
                "remaining balance rather than draining the account."
            ),
            transaction={
                "step": 83,
                "type": "TRANSFER",
                "amount": 1_750_000.0,
                "oldbalanceOrg": 6_000_000.0,
                "newbalanceOrig": 4_250_000.0,
                "oldbalanceDest": 2_000_000.0,
                "newbalanceDest": 3_750_000.0,
                "isFlaggedFraud": 0,
            },
            expected_reasoning=(
                f"balanceDeltaOrig ({cite_feature_evidence('balanceDeltaOrig')}) should indicate "
                "a partial, non-draining balance change despite the large absolute amount — this "
                "scenario deliberately tests that the model does not use amount as a naive threshold."
            ),
            source_rationale=(
                "Legitimate TRANSFERs in the engineered dataset reach a 90th percentile of "
                "$1.77M — large legitimate transfers are common, so this scenario checks the "
                "model does not simply flag every high-value transfer."
            ),
            expected_outcome="legitimate",
            tags=["balanceDeltaOrig", "amount"],
        ),
        Scenario(
            scenario_id="L7",
            title="Routine Cash Withdrawal",
            category="legitimate",
            business_description="A customer withdraws $18,000 in cash, leaving most of their balance intact.",
            transaction={
                "step": 92,
                "type": "CASH_OUT",
                "amount": 18_000.0,
                "oldbalanceOrg": 95_000.0,
                "newbalanceOrig": 77_000.0,
                "oldbalanceDest": 40_000.0,
                "newbalanceDest": 58_000.0,
                "isFlaggedFraud": 0,
            },
            expected_reasoning="Same fraud-eligible type as F2/F4, but without the origin-drain or destination-untouched signatures.",
            source_rationale=(
                "CASH_OUT is a fraud-eligible type, so this scenario specifically probes whether "
                "type alone triggers a fraud prediction absent the drain/discrepancy patterns "
                "that separate the 4,116 real CASH_OUT fraud cases from the 2,233,384 legitimate ones."
            ),
            expected_outcome="legitimate",
            tags=["type_CASH_OUT", "isOrigZeroBalance"],
        ),
    ]


def generate_scenarios() -> list[Scenario]:
    """Generate the full set of business validation scenarios.

    Combines fraud scenarios, legitimate scenarios (this module), and
    edge-case scenarios (`src.validation.edge_cases`) into a single
    ordered list of approximately 15-20 scenarios.

    Returns:
        The complete list of scenarios.
    """
    from src.validation.edge_cases import build_edge_case_scenarios

    scenarios = [*_build_fraud_scenarios(), *_build_legitimate_scenarios(), *build_edge_case_scenarios()]
    logger.info(
        "Generated %d business scenarios (%d fraud, %d legitimate, %d edge case).",
        len(scenarios),
        sum(1 for s in scenarios if s.category == "fraud"),
        sum(1 for s in scenarios if s.category == "legitimate"),
        sum(1 for s in scenarios if s.category == "edge_case"),
    )
    return scenarios
