"""Categorical encoding utilities.

Encoding is kept separate from feature engineering so that a fitted
encoder can be persisted and reused at inference time without re-running
the rest of the pipeline.
"""

import logging

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

logger = logging.getLogger(__name__)

#: Low-cardinality categorical columns suitable for one-hot encoding.
CATEGORICAL_COLUMNS: list[str] = ["type"]

#: High-cardinality identifier columns that are not encoded as features.
ID_COLUMNS: list[str] = ["nameOrig", "nameDest"]


def encode_categorical_columns(
    df: pd.DataFrame, columns: list[str] = CATEGORICAL_COLUMNS
) -> tuple[pd.DataFrame, OneHotEncoder]:
    """One-hot encode categorical columns and return the fitted encoder.

    Args:
        df: DataFrame containing the columns to encode.
        columns: Names of the categorical columns to one-hot encode.
            Defaults to :data:`CATEGORICAL_COLUMNS`.

    Returns:
        A tuple of ``(encoded_df, encoder)`` where ``encoded_df`` has the
        original categorical columns replaced by one-hot encoded columns,
        and ``encoder`` is the fitted :class:`~sklearn.preprocessing.OneHotEncoder`
        (retained so it can be reused on new data, e.g. at inference time).

    Raises:
        ValueError: If any requested column is missing from ``df``.
    """
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Cannot encode missing columns: {missing}")

    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoded_array = encoder.fit_transform(df[columns])
    encoded_columns = encoder.get_feature_names_out(columns)

    encoded_df = pd.DataFrame(encoded_array, columns=encoded_columns, index=df.index)

    result = pd.concat([df.drop(columns=columns), encoded_df], axis=1)

    logger.info(
        "One-hot encoded columns %s into %d new columns.", columns, len(encoded_columns)
    )
    return result, encoder


def encode_with_fitted_encoder(
    df: pd.DataFrame, encoder: OneHotEncoder, columns: list[str] = CATEGORICAL_COLUMNS
) -> pd.DataFrame:
    """Apply a previously fitted encoder to new data.

    Intended for inference-time use, where the encoder must have been
    fitted during training via :func:`encode_categorical_columns` and
    persisted alongside the model.

    Args:
        df: DataFrame containing the columns to encode.
        encoder: A fitted :class:`~sklearn.preprocessing.OneHotEncoder`.
        columns: Names of the categorical columns the encoder expects.
            Defaults to :data:`CATEGORICAL_COLUMNS`.

    Returns:
        A copy of ``df`` with the categorical columns replaced by one-hot
        encoded columns produced by ``encoder``.

    Raises:
        ValueError: If any requested column is missing from ``df``.
    """
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Cannot encode missing columns: {missing}")

    encoded_array = encoder.transform(df[columns])
    encoded_columns = encoder.get_feature_names_out(columns)

    encoded_df = pd.DataFrame(encoded_array, columns=encoded_columns, index=df.index)

    return pd.concat([df.drop(columns=columns), encoded_df], axis=1)
