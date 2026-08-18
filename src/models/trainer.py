"""Shared model training orchestration.

Every registered model is trained through the exact same workflow so
results are directly comparable:

    1. Load the engineered dataset.
    2. Train/validation/test split using `TEST_SIZE`/`VALIDATION_SIZE`
       from :mod:`src.config`.
    3. Fit the categorical encoder on the training set only.
    4. Transform the validation/test sets with that fitted encoder.
    5. Apply SMOTE to the training data only (skipped for models that
       declare ``uses_resampling = False``, e.g. Isolation Forest).
    6. Train the selected model.
    7. Evaluate the model on the held-out test set.
    8. Save a self-contained artifact bundle to `models/<name>/`: the
       trained model, its fitted encoder, a metadata record, and a
       feature schema for validating future inference input (see
       :mod:`src.models.artifacts`).

Intended to be run as a script (``python -m src.models.trainer``) or
imported and called via :func:`run_training`.
"""

import logging
import time
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

from src.config import (
    MODEL_TRAINING_LOG_PATH,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
    VALIDATION_SIZE,
)
from src.models.artifacts import save_encoder, save_feature_schema, save_metadata
from src.models.base_model import BaseModel
from src.models.evaluator import evaluate_model, save_comparison_table
from src.models.model_registry import (
    get_encoder_path,
    get_feature_schema_path,
    get_metadata_path,
    get_model,
    get_model_dir,
    get_model_path,
    list_available_models,
)
from src.preprocessing.encoding import ID_COLUMNS, encode_categorical_columns, encode_with_fitted_encoder
from src.preprocessing.loader import load_engineered_dataset
from src.utils.logger import configure_logging

logger = logging.getLogger(__name__)


def split_dataset(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    test_size: float = TEST_SIZE,
    validation_size: float = VALIDATION_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified three-way split into train/validation/test sets.

    `test_size` and `validation_size` are both interpreted as fractions
    of the *full* dataset (e.g. the defaults of 0.20 and 0.10 yield a
    70/10/20 train/validation/test split), matching the constants defined
    in :mod:`src.config`. Splits are stratified on `target_column` to
    preserve the (heavily imbalanced) fraud rate in every split.

    Args:
        df: The full engineered dataset.
        target_column: Name of the binary target column to stratify on.
        test_size: Fraction of the full dataset held out as the test set.
        validation_size: Fraction of the full dataset held out as the
            validation set.
        random_state: Seed for reproducibility.

    Returns:
        A ``(train_df, val_df, test_df)`` tuple.
    """
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df[target_column], random_state=random_state
    )

    relative_val_size = validation_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_size,
        stratify=train_val_df[target_column],
        random_state=random_state,
    )

    logger.info(
        "Split dataset into train=%d, validation=%d, test=%d rows.",
        len(train_df),
        len(val_df),
        len(test_df),
    )
    return train_df, val_df, test_df


def prepare_features(
    df: pd.DataFrame, target_column: str = TARGET_COLUMN
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a DataFrame into a model-ready feature matrix and target vector.

    Drops high-cardinality identifier columns (`nameOrig`, `nameDest`)
    which are not predictive features, along with the target column.

    Args:
        df: A DataFrame produced by :func:`split_dataset`.
        target_column: Name of the binary target column.

    Returns:
        An ``(X, y)`` tuple.
    """
    X = df.drop(columns=[*ID_COLUMNS, target_column])
    y = df[target_column]
    return X, y


def apply_smote(
    X_train: pd.DataFrame, y_train: pd.Series, random_state: int = RANDOM_STATE
) -> tuple[pd.DataFrame, pd.Series]:
    """Oversample the minority (fraud) class in the training data via SMOTE.

    Must only ever be called on the training split — applying SMOTE to
    validation/test data would leak synthetic, non-real examples into
    evaluation.

    Args:
        X_train: Training feature matrix (already encoded).
        y_train: Training target vector.
        random_state: Seed for reproducibility.

    Returns:
        A resampled ``(X_train, y_train)`` tuple with a balanced class
        distribution.
    """
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    logger.info(
        "Applied SMOTE to training data: %d -> %d rows.", len(X_train), len(X_resampled)
    )
    return X_resampled, y_resampled


def _build_model_metadata(
    model: BaseModel,
    target_column: str,
    feature_names: list[str],
    evaluation_result: dict[str, Any],
    training_timestamp: str,
    random_state: int,
) -> dict[str, Any]:
    """Assemble the per-model metadata record.

    Args:
        model: The fitted model wrapper.
        target_column: Name of the binary target column.
        feature_names: The encoded feature columns the model was fit on,
            in order.
        evaluation_result: The result dictionary returned by
            :func:`~src.models.evaluator.evaluate_model`.
        training_timestamp: ISO-8601 UTC timestamp captured at the start
            of training for this model.
        random_state: Seed used for the split/resampling/model.

    Returns:
        A JSON-serializable metadata dictionary.
    """
    metric_keys = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "confusion_matrix",
        "classification_report",
    ]
    return {
        "model_name": model.name,
        "algorithm": type(model.model).__name__,
        "training_timestamp": training_timestamp,
        "target_column": target_column,
        "feature_names": feature_names,
        "evaluation_metrics": {key: evaluation_result[key] for key in metric_keys},
        "training_time_seconds": evaluation_result["training_time_seconds"],
        "prediction_time_seconds": evaluation_result["prediction_time_seconds"],
        "random_state": random_state,
    }


def _build_feature_schema(
    target_column: str, raw_feature_names: list[str], encoded_feature_order: list[str]
) -> dict[str, Any]:
    """Assemble the feature schema used to validate future inference input.

    Args:
        target_column: Name of the binary target column.
        raw_feature_names: The pre-encoding input columns a caller must
            supply (e.g. a Streamlit form), in order.
        encoded_feature_order: The post-encoding columns the fitted model
            expects, in the exact order it was trained on.

    Returns:
        A JSON-serializable feature schema dictionary.
    """
    return {
        "target_column": target_column,
        "raw_feature_names": raw_feature_names,
        "encoded_feature_order": encoded_feature_order,
    }


def train_single_model(
    model_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    **model_kwargs: Any,
) -> dict[str, Any]:
    """Run the full shared training workflow for a single model.

    Args:
        model_name: Registered model name (see
            :func:`~src.models.model_registry.list_available_models`).
        train_df: Training split, as returned by :func:`split_dataset`.
        val_df: Validation split, as returned by :func:`split_dataset`.
            Held out for future hyperparameter tuning; not used for the
            headline evaluation metrics in this sprint.
        test_df: Test split, as returned by :func:`split_dataset`. Used
            for the final evaluation metrics.
        **model_kwargs: Hyperparameters forwarded to the model wrapper's
            constructor.

    Returns:
        The evaluation result dictionary for this model (see
        :func:`~src.models.evaluator.evaluate_model`), plus
        ``model_save_path`` and ``model_dir`` keys.
    """
    logger.info("=== Training model: %s ===", model_name)
    training_timestamp = datetime.now(timezone.utc).isoformat()
    model = get_model(model_name, **model_kwargs)

    X_train, y_train = prepare_features(train_df)
    X_val, y_val = prepare_features(val_df)
    X_test, y_test = prepare_features(test_df)
    del X_val, y_val  # reserved for future hyperparameter tuning

    raw_feature_names = X_train.columns.tolist()

    logger.info("Fitting categorical encoder on the training set only.")
    X_train_encoded, encoder = encode_categorical_columns(X_train)
    X_test_encoded = encode_with_fitted_encoder(X_test, encoder)
    encoded_feature_order = X_train_encoded.columns.tolist()

    if model.uses_resampling:
        X_train_final, y_train_final = apply_smote(X_train_encoded, y_train)
    else:
        logger.info("Skipping SMOTE for '%s' (uses_resampling=False).", model_name)
        X_train_final, y_train_final = X_train_encoded, y_train

    logger.info("Training start: %s", model_name)
    start = time.perf_counter()
    model.fit(X_train_final, y_train_final)
    training_time = time.perf_counter() - start
    logger.info("Training finish: %s (%.2fs)", model_name, training_time)

    result = evaluate_model(model, X_test_encoded, y_test, training_time=training_time)

    model_dir = get_model_dir(model_name)

    model_path = get_model_path(model_name)
    model.save(model_path)

    encoder_path = get_encoder_path(model_name)
    save_encoder(encoder, encoder_path)

    metadata = _build_model_metadata(
        model=model,
        target_column=TARGET_COLUMN,
        feature_names=encoded_feature_order,
        evaluation_result=result,
        training_timestamp=training_timestamp,
        random_state=RANDOM_STATE,
    )
    save_metadata(metadata, get_metadata_path(model_name))

    feature_schema = _build_feature_schema(
        target_column=TARGET_COLUMN,
        raw_feature_names=raw_feature_names,
        encoded_feature_order=encoded_feature_order,
    )
    save_feature_schema(feature_schema, get_feature_schema_path(model_name))

    logger.info("Completed artifact bundle for '%s' in %s.", model_name, model_dir)

    result["model_save_path"] = str(model_path)
    result["model_dir"] = str(model_dir)
    return result


def run_training(model_names: Optional[list[str]] = None) -> pd.DataFrame:
    """Train and evaluate every requested model, then save a comparison table.

    Args:
        model_names: Registered model names to train. Defaults to every
            model in the registry (see
            :func:`~src.models.model_registry.list_available_models`).

    Returns:
        The model comparison table (one row per successfully trained
        model), as saved to `reports/model_results.csv`.

    Raises:
        FileNotFoundError: If the engineered dataset cannot be found.
        ValueError: If no model trained successfully.
    """
    model_names = model_names or list_available_models()
    logger.info("=== Model training started for: %s ===", model_names)

    df = load_engineered_dataset()
    train_df, val_df, test_df = split_dataset(df)

    results: list[dict[str, Any]] = []
    for name in model_names:
        try:
            results.append(train_single_model(name, train_df, val_df, test_df))
        except Exception:
            logger.exception("Training failed for model '%s'; skipping.", name)

    comparison_df = save_comparison_table(results)
    logger.info("=== Model training finished. Trained %d/%d models. ===", len(results), len(model_names))
    return comparison_df


def main() -> None:
    """CLI entry point: configure logging and run training for all models."""
    configure_logging(MODEL_TRAINING_LOG_PATH)
    run_training()


if __name__ == "__main__":
    main()
