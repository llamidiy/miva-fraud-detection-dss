# Deployment Manifest

**Sprint 6.7.3 — Deployment Preparation.** This staging directory (`deployment/`) is a clean, independently verified copy of only the files the deployed Streamlit application requires — traced through actual imports and file reads, not assumed from folder structure. The original research repository is completely untouched; see section I for the integrity proof.

---

## A. Deployment Entrypoint

**`app/app.py`** — confirmed by inspection: configures `st.set_page_config`, loads `app/assets/styles.css`, and wires up `st.navigation` across all seven pages. Confirmed functionally by smoke test (section I).

## B. Python Version

The staging copy was built and smoke-tested with the same local environment as the research repository: **Python 3.12.13**. No `runtime.txt` is included in this package — selecting a Python 3.12.x runtime on the deployment platform is a step for whoever performs the actual deployment (see section K), not something baked into this staging directory.

## C. Runtime Dependencies

`requirements.txt` (10 packages, all version-pinned to the exact locally-installed versions that were smoke-tested):

```
streamlit==1.60.0
pandas==3.0.5
numpy==2.3.5
scikit-learn==1.9.0
xgboost==3.3.0
shap==0.52.0
matplotlib==3.11.1
joblib==1.5.3
pillow==12.3.0
imbalanced-learn==0.14.2
```

Two corrections from the Sprint 6.7.2 audit, established this sprint through direct tracing and an empirical import test (not re-assumed):

1. **`shap` and `matplotlib` are hard runtime dependencies, not offline-only.** `prediction_service.predict_transaction()` calls `explainability_service.get_local_explanation()` → `local_explanations.explain_single_transaction()` → live SHAP `TreeExplainer` computation and a live matplotlib waterfall save, on **every** Single Transaction assessment. Confirmed by the smoke test: `deployment/reports/ui/single_transaction_waterfall.png` was freshly generated during testing.
2. **`imbalanced-learn` is a genuine transitive runtime dependency.** `src/models/__init__.py` unconditionally imports `from src.models.trainer import run_training`, and `trainer.py` imports `from imblearn.over_sampling import SMOTE` at module load time. Since `src/models/predictor.py` (used live by the explainability service) is a submodule of `src.models`, importing it first executes `src/models/__init__.py` in full — meaning `imbalanced-learn` must be installed even though SMOTE itself is never *called* at runtime. Verified empirically: importing the staged `services.prediction_service` module was confirmed to load `imblearn` into `sys.modules`.

`appnope==0.1.4` (macOS-only, present in the research repo's `requirements.txt`) is **not included** here.

## D. Runtime Model Artifacts

Only the XGBoost champion bundle — confirmed as the sole model loaded at runtime via `CHAMPION_MODEL_NAME = "xgboost"`:

| File | Size | SHA-256 matches original |
|---|---|---|
| `models/xgboost/model.joblib` | 804 KB | ✅ Verified identical |
| `models/xgboost/encoder.joblib` | 4 KB | ✅ Verified identical |
| `models/xgboost/metadata.json` | 4 KB | ✅ Verified identical |
| `models/xgboost/feature_schema.json` | 4 KB | ✅ Verified identical |

No model was retrained, refit, or modified to produce this package — all four files are byte-for-byte copies of the research repository's originals (see `DEPLOYMENT_CHECKSUMS.txt`). `models/random_forest/`, `models/logistic_regression/`, and `models/isolation_forest/` are **not included** — confirmed via `model_registry.py`/`metrics_service.py` tracing that the deployed app never loads their `.joblib` files; the Model Performance page's four-model comparison comes entirely from the already-included `reports/model_results.csv`.

## E. Runtime Reports/Assets

| File(s) | Used by | Notes |
|---|---|---|
| `reports/model_results.csv` | Dashboard, Model Performance | 4-model comparison table |
| `reports/feature_importance.csv` | Explainability | Native XGBoost importance |
| `reports/figures/*.png` (6 files) | Explainability | Pre-generated SHAP/importance figures |
| `reports/shap/shap_values.joblib` | Explainability | **Corrected finding this sprint** — read live by `get_shap_summary()` for the "Top Features by Mean \|SHAP\| Value" list; Sprint 6.7.2 incorrectly called this offline-only |
| `reports/validation/scenario_results.csv` | Validation, Dashboard, Explainability | 20 scenario outcomes |
| `reports/validation/validation_report.md`, `business_scenarios.md`, `executive_summary.md` | Validation, About, Dashboard | Full validation documents |
| `reports/validation/screenshots/waterfall_*.png` (20 files) | Validation, Explainability | Per-scenario SHAP waterfalls |
| `data/explainability/shap_sample.csv` | Single Transaction (live) | **Corrected finding this sprint** — `get_local_explanation()` loads this as the SHAP background reference sample on every live assessment; Sprint 6.7.2 incorrectly called this offline-only |
| `app/assets/styles.css`, `logo.png`, `favicon.png` | Every page | |
| `app/assets/diagrams/system_architecture.png`, `app_architecture.png` | About | |

None of these is required merely for app *startup* — each is loaded lazily by the specific page that needs it (Streamlit's page-based execution model), except `styles.css` and the sidebar branding assets, which load on every page via `app.py`.

## F. Files Intentionally Excluded

| File(s) | Why excluded |
|---|---|
| `data/raw/paysim.csv` (471 MB) | Never loaded by the deployed app; already committed to the research repo's git history — a size-based GitHub blocker, not a runtime need |
| `data/processed/paysim_engineered.csv` (946 MB) | Never loaded by the deployed app (only by offline training/sampling scripts) |
| `models/random_forest/`, `models/logistic_regression/`, `models/isolation_forest/` `.joblib` files | Not loaded at runtime — see section D |
| `reports/model_interpretation.md`, `reports/preprocessing_metadata.json`, `reports/preprocessing_report.md` | Not read by any app code |
| `notebooks/`, `database/` (empty) | Not referenced by the deployed app |
| `reports/user_evaluation/`, `reports/deployment/`, `reports/evaluation/` (Sprint 6.5/6.6/6.6.1/6.7 evidence) | Dissertation evidence, not application runtime files |
| `~21 Jupyter/IDE packages` from the research repo's `requirements.txt` | Never imported by the deployed app; would only slow the Streamlit Cloud build |

## G. Total Deployment Size

**~10 MB total** — comfortably small for Streamlit Community Cloud (no meaningful platform size constraint at this scale).

| Directory | Size |
|---|---|
| `app/` | 516 KB |
| `data/` | 1.5 MB |
| `models/` | 816 KB |
| `reports/` | 6.9 MB |
| `src/` | 584 KB |

Largest individual files: `reports/shap/shap_values.joblib` (4.5 MB), `data/explainability/shap_sample.csv` (1.5 MB), `models/xgboost/model.joblib` (804 KB) — everything else is under 250 KB.

## H. Linux Compatibility Findings

- **No hardcoded `/Users/` or `/System/` paths** anywhere in the staged `.py` files.
- **No platform-specific shell commands** (`open`, `pbcopy`, `osascript`, etc.).
- **No `os.getcwd()` dependence** — every path resolves via `Path(__file__).resolve()`, confirmed in `src/config.py`, `app/app.py`, and `app/utils/paths.py`.
- **`appnope` correctly excluded** from `requirements.txt` (see section C).
- **XGBoost/OpenMP:** not independently testable without an actual Linux deployment, but `xgboost==3.3.0`'s official manylinux wheels have bundled the OpenMP runtime since v1.6+, and the smoke test (section I) confirms XGBoost imports and predicts correctly in this exact Python 3.12 environment.

No application logic was modified to work around any hypothetical problem — nothing genuine was found that would require it.

## I. Smoke-Test Results

Performed by actually launching `deployment/app/app.py` with Streamlit (an isolated process, `sys.path` pointed only at `deployment/`, port 8510) and driving it through a real browser — not static analysis.

| # | Check | Result |
|---|---|---|
| 1 | App imports successfully | ✅ Pass |
| 2 | Streamlit starts | ✅ Pass, zero server errors in logs |
| 3 | Dashboard loads | ✅ Pass — real metrics (XGBoost, ROC-AUC 0.9998, 20 scenarios, "Operational") |
| 4 | Single Transaction loads | ✅ Pass |
| 5 | Batch Assessment loads | ✅ Pass |
| 6 | Explainability loads | ✅ Pass |
| 7 | Validation loads | ✅ Pass |
| 8 | Model Performance loads | ✅ Pass |
| 9 | About loads | ✅ Pass |
| 10 | XGBoost model loads successfully | ✅ Pass (confirmed via successful live prediction) |
| 11 | Encoder loads successfully | ✅ Pass (confirmed via successful live prediction) |
| 12 | A real Single Transaction prediction works | ✅ Pass — "Not Fraud", 100.0% confidence, correct recommendation |
| 13 | A real Single Transaction explanation/waterfall works | ✅ Pass — live SHAP factors rendered; `reports/ui/single_transaction_waterfall.png` freshly written to disk during the test (84 KB) |
| 14 | A real batch assessment works | ✅ Pass — 2-row CSV uploaded, previewed, assessed (1 fraud, 1 not fraud), traceable results table rendered |
| 15 | Static figures/diagrams load | ✅ Pass — all 6 SHAP/importance figures, both architecture diagrams, and the F1 waterfall all confirmed rendering (no "not found" warnings on any page) |

No dataset from `data/raw/` or `data/processed/` was used or needed for this test. All test transactions used ad-hoc or already-established sample values (the existing default form values, and a 2-row synthetic CSV for batch).

## J. Known Risks

1. **XGBoost/OpenMP on Linux** — expected to work (see section H) but genuinely unverifiable without a real Streamlit Community Cloud deployment. Recommend this be the first thing checked after the actual deploy.
2. **This staging copy has not been tested with a completely fresh `pip install` from `requirements.txt` into an empty virtual environment** — the smoke test reused the research repository's existing `.venv`, which already had every package installed at the pinned versions. A `pip install -r requirements.txt` into a clean environment before the real deployment would close this gap and is a reasonable, low-cost next check.
3. **`reports/model_interpretation.md` is intentionally excluded**, so if any future page change starts reading it, that would silently break in deployment — not a current risk, but worth remembering if `app/pages/7_About.py` or similar is ever modified again.

## K. Exact Next Steps for GitHub + Streamlit Community Cloud

This sprint deliberately stops here. The next steps (not performed, and requiring explicit separate authorization) are:

1. Decide how to handle the research repository's git-history size problem (the already-committed 471 MB `data/raw/paysim.csv`) — this staging directory sidesteps the problem entirely by not being a git repository yet and not containing that file, but the *research* repository's history issue is unrelated to this package and still needs its own resolution if that repository is ever pushed.
2. Initialize a **new** git repository inside (or from) `deployment/` — e.g. `cd deployment && git init`.
3. Review `git status` after `git add`, confirm nothing unexpected is staged, and make the first commit.
4. Create a new, separate GitHub repository for the deployment (or reuse an existing empty one) and add it as the remote.
5. Push this staging copy's history to that repository.
6. Follow the previously-audited deployment plan (`reports/deployment/deployment_plan.md`, steps 4-11) using this repository: connect Streamlit Community Cloud, select this repo/branch, set the entrypoint to `app/app.py`, select a Python 3.12.x runtime, deploy, and re-run the same smoke-test checklist (section I above) against the live Linux deployment before sharing the URL with evaluation participants.
