# Fraud Detection Decision Support System — Deployment Package

This is a **staging copy**, prepared by Sprint 6.7.3, containing only the files the deployed Streamlit application genuinely needs at runtime — traced through actual imports and file reads, not copied wholesale from the research repository.

See **`DEPLOYMENT_MANIFEST.md`** for the complete breakdown (entrypoint, dependencies, runtime files, exclusions, size, Linux compatibility, smoke-test results, and next steps), and **`DEPLOYMENT_CHECKSUMS.txt`** for SHA-256 verification of every model artifact.

## Quick Facts

- **Entrypoint:** `app/app.py`
- **Total size:** ~10 MB
- **Model:** XGBoost only (`models/xgboost/model.joblib` + `encoder.joblib`)
- **Excluded:** the 471 MB raw dataset and 946 MB engineered dataset — neither is used at runtime

## Running Locally

```
pip install -r requirements.txt
streamlit run app/app.py
```

This directory is not yet a git repository and has not been pushed anywhere — see `DEPLOYMENT_MANIFEST.md` section K for the exact next steps.
