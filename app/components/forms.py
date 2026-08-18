"""Reusable form components.

Phase 1 renders input widgets only — no submission is sent anywhere.
Phase 2 will pass a submitted form's values into
`src.models.predictor`/`src.explainability` using the same field names.
"""

from typing import Optional, TypedDict

import streamlit as st


class TransactionFormValues(TypedDict):
    """Raw transaction fields collected by `transaction_input_form`."""

    step: int
    transaction_type: str
    amount: float
    old_balance_org: float
    new_balance_orig: float
    old_balance_dest: float
    new_balance_dest: float


def transaction_input_form(key_prefix: str = "single_txn") -> Optional[TransactionFormValues]:
    """Render a transaction input form and return its values once submitted.

    Args:
        key_prefix: Prefix for Streamlit widget keys, so the form can be
            embedded more than once across the app without key collisions.

    Returns:
        The submitted form values, or ``None`` if the form has not been
        submitted on this run.
    """
    with st.form(key=f"{key_prefix}_form"):
        col1, col2 = st.columns(2)

        with col1:
            transaction_type = st.selectbox(
                "Transaction Type",
                options=["TRANSFER", "CASH_OUT", "CASH_IN", "PAYMENT", "DEBIT"],
                key=f"{key_prefix}_type",
            )
            amount = st.number_input(
                "Amount ($)", min_value=0.0, value=1000.0, step=100.0, key=f"{key_prefix}_amount"
            )
            step = st.number_input(
                "Time Step (hour index)", min_value=0, value=1, step=1, key=f"{key_prefix}_step"
            )

        with col2:
            old_balance_org = st.number_input(
                "Origin Balance — Before ($)", min_value=0.0, value=5000.0, step=100.0, key=f"{key_prefix}_ob_org"
            )
            new_balance_orig = st.number_input(
                "Origin Balance — After ($)", min_value=0.0, value=4000.0, step=100.0, key=f"{key_prefix}_nb_orig"
            )
            old_balance_dest = st.number_input(
                "Destination Balance — Before ($)", min_value=0.0, value=0.0, step=100.0, key=f"{key_prefix}_ob_dest"
            )
            new_balance_dest = st.number_input(
                "Destination Balance — After ($)", min_value=0.0, value=1000.0, step=100.0, key=f"{key_prefix}_nb_dest"
            )

        submitted = st.form_submit_button("Assess Transaction", use_container_width=True)

    if not submitted:
        return None

    return TransactionFormValues(
        step=int(step),
        transaction_type=transaction_type,
        amount=float(amount),
        old_balance_org=float(old_balance_org),
        new_balance_orig=float(new_balance_orig),
        old_balance_dest=float(old_balance_dest),
        new_balance_dest=float(new_balance_dest),
    )
