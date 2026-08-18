"""Feature engineering transformations for the PaySim dataset.

Each function takes a DataFrame and returns a new DataFrame with one or
more additional columns, so transformations can be composed, tested, and
reused independently. No categorical encoding happens here — see
:mod:`src.preprocessing.encoding` for that step.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: PaySim's `step` field advances by 1 per simulated hour.
STEPS_PER_DAY = 24


def add_balance_delta_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the observed change in balance for the origin and destination accounts.

    Args:
        df: DataFrame containing ``oldbalanceOrg``, ``newbalanceOrig``,
            ``oldbalanceDest``, and ``newbalanceDest``.

    Returns:
        A copy of ``df`` with ``balanceDeltaOrig`` and ``balanceDeltaDest``
        columns added.
    """
    df = df.copy()
    df["balanceDeltaOrig"] = df["newbalanceOrig"] - df["oldbalanceOrg"]
    df["balanceDeltaDest"] = df["newbalanceDest"] - df["oldbalanceDest"]
    return df


def add_balance_error_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add discrepancy features between the expected and actual balance change.

    For a legitimate transaction, the origin balance should drop by exactly
    ``amount`` and the destination balance should rise by exactly
    ``amount``. Non-zero values in the resulting columns flag transactions
    where the ledger does not balance as expected — a known indicator of
    fraud in the PaySim dataset.

    Args:
        df: DataFrame containing ``amount``, ``oldbalanceOrg``,
            ``newbalanceOrig``, ``oldbalanceDest``, and ``newbalanceDest``.

    Returns:
        A copy of ``df`` with ``errorBalanceOrig`` and ``errorBalanceDest``
        columns added.
    """
    df = df.copy()
    df["errorBalanceOrig"] = df["newbalanceOrig"] + df["amount"] - df["oldbalanceOrg"]
    df["errorBalanceDest"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
    return df


def add_zero_balance_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add boolean flags for accounts with zero balance before and after.

    Args:
        df: DataFrame containing ``oldbalanceOrg``, ``newbalanceOrig``,
            ``oldbalanceDest``, and ``newbalanceDest``.

    Returns:
        A copy of ``df`` with ``isOrigZeroBalance`` and ``isDestZeroBalance``
        boolean columns added.
    """
    df = df.copy()
    df["isOrigZeroBalance"] = (df["oldbalanceOrg"] == 0) & (df["newbalanceOrig"] == 0)
    df["isDestZeroBalance"] = (df["oldbalanceDest"] == 0) & (df["newbalanceDest"] == 0)
    return df


def add_log_amount_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Add a log1p-transformed transaction amount to reduce right skew.

    Args:
        df: DataFrame containing ``amount``.

    Returns:
        A copy of ``df`` with a ``logAmount`` column added.
    """
    df = df.copy()
    df["logAmount"] = np.log1p(df["amount"])
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add day-of-simulation and hour-of-day features derived from `step`.

    In PaySim, `step` counts simulated hours starting at 1, so hour-of-day
    and day-of-simulation can be recovered via modular arithmetic.

    Args:
        df: DataFrame containing ``step``.

    Returns:
        A copy of ``df`` with ``transactionDay`` and ``transactionHour``
        columns added.
    """
    df = df.copy()
    df["transactionDay"] = df["step"] // STEPS_PER_DAY
    df["transactionHour"] = df["step"] % STEPS_PER_DAY
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the full set of feature engineering transformations.

    Runs each modular transformation in sequence: balance deltas, balance
    error flags, zero-balance flags, log-transformed amount, and
    time-derived features.

    Args:
        df: The cleaned, validated input DataFrame.

    Returns:
        A new DataFrame with all engineered feature columns added.
    """
    logger.info("Starting feature engineering on %d rows.", len(df))

    df = add_balance_delta_features(df)
    df = add_balance_error_features(df)
    df = add_zero_balance_flags(df)
    df = add_log_amount_feature(df)
    df = add_time_features(df)

    logger.info("Feature engineering complete. Resulting shape: %s", df.shape)
    return df
