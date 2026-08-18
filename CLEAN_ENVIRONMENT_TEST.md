# Clean Environment Test

**Sprint 6.7.4.** Verifies that `deployment/requirements.txt` is sufficient for the staged application to install and run **independently of the research repository's `.venv`**. A genuinely fresh virtual environment was created outside the project (`/tmp/fraud-dss-deployment-test-venv`), and every test below ran exclusively through that environment's Python interpreter — confirmed programmatically at the start of each test (`assert '/tmp/fraud-dss-deployment-test-venv' in sys.executable`).

## 1. Python Version

**Python 3.12.13** — created explicitly with `/usr/local/bin/python3.12` (the system default `python3` resolved to 3.14.6, which was rejected as it doesn't match the intended Streamlit Community Cloud 3.12.x runtime). This is the exact same patch version used throughout the research repository's own `.venv`.

## 2. Installation Result

**PASS.** `pip install -r requirements.txt` completed in **83 seconds**, exit code 0. `pip check` afterward reported "No broken requirements found." No manual package installation was performed outside what `requirements.txt` specifies.

## 3. Installed Dependency Versions

All 10 pinned packages installed at their exact specified versions, plus their transitive dependencies resolved cleanly with no conflicts:

```
streamlit==1.60.0        pandas==3.0.5           numpy==2.3.5
scikit-learn==1.9.0      xgboost==3.3.0          shap==0.52.0
matplotlib==3.11.1       joblib==1.5.3           pillow==12.3.0
imbalanced-learn==0.14.2
```

No dependency conflicts. No warnings that could affect deployment (the only pip notice was a routine "new pip version available" message, irrelevant to the application).

## 4. Import Test Result

**PASS.** All 10 core packages imported successfully. The application's own import surface was tested in two parts, matching how the app is actually invoked (as a Streamlit-run script with `app/` added to `sys.path`, not as an installable `app` package):

- `import app.app` (as a namespace package, with the deployment root on `sys.path`) — succeeded. Verified this exact pattern behaves identically in the original research repository's `.venv` (also succeeds there), confirming it is the correct test, not a deployment-specific quirk.
- `services.prediction_service`, `services.explainability_service`, `services.metrics_service`, `services.validation_service`, `services.report_service`, and `state` (with `app/` added to `sys.path`, matching `app.py`'s own runtime behavior) — all imported successfully.

Zero packages resolved from the original project's `.venv` — confirmed by asserting on `sys.executable` at the start of every test.

## 5. Model Artifact Test Result

**PASS.** All four artifacts loaded successfully using only the clean environment:

- `model.joblib` → `XGBoostModel` (name=`xgboost`)
- `encoder.joblib` → `OneHotEncoder`
- `metadata.json` → model_name=`xgboost`, algorithm=`XGBClassifier`
- `feature_schema.json` → 17 raw features, 21 encoded features

SHA-256 checksums re-verified against `DEPLOYMENT_CHECKSUMS.txt` after loading: all 4 model artifacts **OK, byte-identical**. No retraining occurred; no artifact was modified (loading via `joblib.load`/`json.load` is read-only).

## 6. Prediction Test Result

**PASS.** Ran a real prediction through `services.prediction_service.predict_transaction()` using validation scenario **L1** (Monthly Salary Deposit — the same scenario used in the participant evaluation package):

| Field | Result |
|---|---|
| error | `None` |
| prediction | `Not Fraud` |
| confidence | `100.0` |
| recommendation | `Approve transaction, Continue routine monitoring` |

No missing-file errors, no import errors. Result matches the expected outcome established in every prior sprint.

## 7. SHAP Test Result

**PASS — all 6 steps.** Using scenario **F1** (Unusually High-Value Transfer With Origin Fully Drained):

1. XGBoost champion model loaded via `get_champion_explainer()` — ✅
2. `FraudExplainer` created (model name confirmed: `xgboost`) — ✅
3. Local explanation generated via `get_local_explanation()`: prediction=`Fraud`, confidence=`100.0`, 3 top factors returned, `error=None` — ✅
4-5. Waterfall visualization generated and confirmed written to disk (`reports/ui/single_transaction_waterfall.png`, 88,000 bytes) — ✅
6. SHAP values produced correctly (3 ranked contributing factors, non-empty) — ✅

*(Note: an initial attempt called `get_local_explanation()` directly without the `isFlaggedFraud` field, which `prediction_service.predict_transaction()` normally defaults to `0` before calling it — this raised a `KeyError`. This was a test-script error, not an application defect; corrected by supplying the same field the real wrapper always supplies, after which all 6 steps passed.)*

## 8. Streamlit Startup Result

**PASS.** `streamlit run app/app.py --server.port 8511` launched from `deployment/` using the clean venv's own `streamlit` binary. Server started with **zero errors** in the log. Confirmed via `lsof` that the process was genuinely running on port 8511.

## 9. Seven-Page Startup Result

**PASS — all 7 pages.** All page routes returned HTTP 200. Because Streamlit's multipage routing is client-side (a plain HTTP GET does not itself trigger page-script execution), a real browser session was used to navigate to each page, and the server log was checked for genuine per-page execution evidence:

```
Loading Dashboard page...
Loading Single Transaction page...
Loading Explainability page...
Loading Validation page...
Loading Model Performance page...
Loading About page...
Batch Assessment page loaded in 0.041s.   <- confirms execution; this page has no "Loading..." start-log line in its source (pre-existing, not a deployment defect)
```

Zero `ERROR`, `Traceback`, or `Exception` entries anywhere in the server log across all 7 pages. The Dashboard was also confirmed via `get_page_text` to render real data (XGBoost champion, ROC-AUC 0.9998, 20 validation scenarios, "Operational" status) — genuine script execution, not a cached/static shell.

## 10. Batch Prediction Result

**PASS.** Used a temporary copy of `reports/user_evaluation/participant_package/batch_assessment_sample.csv` (copied to `/tmp/batch_test_copy.csv`; the original file was never modified — confirmed by MD5 before and after). Ran through `services.prediction_service.predict_batch()`:

- All 5 rows scored (`n_processed == 5`, `errors == []`).
- Every row returned a prediction, confidence, and recommendation:
  - Rows 0-2 (L1, L2, L3 — legitimate scenarios): `Not Fraud`, 100.0% confidence, approve/monitor.
  - Rows 3-4 (F1, F2 — fraud scenarios): `Fraud`, 100.0% confidence, flag/escalate.
- No dataset (`data/raw/` or `data/processed/`) was required.

## 11. Large-Dataset Access Check

**PASS — confirmed absent, not merely unused.** `deployment/data/raw/` and `deployment/data/processed/` do not exist in the staging directory at all (only `deployment/data/explainability/shap_sample.csv` is present). Every test in sections 6-10 above succeeded despite this — which is itself direct proof neither large dataset is required, since any attempted access would have raised `FileNotFoundError`. The full Streamlit server log was additionally searched for any reference to `paysim.csv` or `paysim_engineered.csv`: **none found**.

## 12. Warnings

- Benign `streamlit.runtime.caching.cache_data_api: No runtime found, using MemoryCacheStorageManager` warnings appear when service-layer functions decorated with `@st.cache_data`/`@st.cache_resource` are called outside a live Streamlit server context (i.e., in the standalone Python test scripts in sections 4-7 and 10). These do not appear when the app is actually served by Streamlit (section 9's server log is clean) and have no effect on correctness.
- `requirements.txt` was not tested via `pip install` into an environment that already has conflicting package versions present — a genuinely empty venv was used, which is the more relevant test for a fresh Streamlit Community Cloud build.

## 13. Blockers

**None found in this test.** The one item that remains genuinely unverifiable without an actual Streamlit Community Cloud deployment is XGBoost's OpenMP runtime behavior on that specific Linux container image — this local test used macOS, and while `xgboost==3.3.0`'s manylinux wheels are expected to work on Linux out of the box, this specific test cannot prove that. This was already flagged as a known risk in `DEPLOYMENT_MANIFEST.md` §J and remains the one item to verify immediately after the real deployment.

## 14. Final Deployment Readiness Classification

# READY

Every critical test passed: clean install, full import chain, model loading, real prediction, live SHAP explanation and waterfall generation, Streamlit startup, all 7 pages, real batch prediction, and confirmed independence from both the original `.venv` and both large datasets. The only open item (XGBoost-on-Linux) is a genuinely unverifiable-without-real-deployment risk, not a failure of anything tested here, and does not downgrade this classification below READY — it is explicitly called out as the first thing to check immediately after the actual Streamlit Community Cloud deployment.
