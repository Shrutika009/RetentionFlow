# RetentionFlow

RetentionFlow is a Streamlit dashboard for customer churn analysis and retention planning. It combines model scoring, risk segmentation, explainability charts, ROI prioritization, and manual customer prediction workflows for the ChurnZero sample dataset.

## Features

- Churn risk dashboard with KPIs, segment filters, and customer-level scoring tables
- Model comparison, calibration, confusion matrix, feature importance, and SHAP chart views
- ROI calculator and customer cloning outputs for retention prioritization
- Manual prediction form for estimating churn probability from customer attributes
- Export tabs for training scoring, test scoring, predictions, ROI output, and customer cloning
- Bundled sample raw, processed, chart, and scoring outputs used by the app

## Project Structure

```text
.
|-- app.py                         # Streamlit application
|-- Retentionflow.ipynb            # Notebook used for analysis/model output generation
|-- notebook_model_artifact.py     # Helper for loading notebook-created model artifacts
|-- requirements.txt               # Python dependencies
|-- runtime.txt                    # Python runtime target
|-- assets/                        # Dashboard image assets
|-- Churnzero_data/                # Raw and processed CSV data
`-- retention_flow_outputs/        # Generated scoring files, charts, and model artifacts
```

## Setup

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS or Linux, activate the environment with:

```bash
source .venv/bin/activate
```

## Run the App

```bash
streamlit run app.py
```

The app expects the included data and output folders to keep their current relative paths. The dashboard includes overview, EDA, features, model, risk, revenue, ROI, recommendations, cloning, survival, prediction, and export tabs.

Model artifacts are loaded from:

```text
retention_flow_outputs/artifacts/
```

Required local artifact files:

```text
retentionflow_best_model.pkl
retentionflow_encoders.pkl
retentionflow_scaler.pkl
retentionflow_config.json
retentionflow_features.json
shap_feature_importance.csv
model_comparison_results.csv
```

If those artifact files are missing locally, rerun the notebook workflow to regenerate them before launching the Streamlit app. The `retention_flow_outputs/artifacts/` directory is intentionally ignored by Git because it can contain generated model binaries.

## Streamlit Community Cloud Deployment

This app must be deployed with Python 3.11. Streamlit Community Cloud may ignore
`runtime.txt` and default to a newer Python release, which can make pinned ML
packages such as scikit-learn and CatBoost fail during dependency installation.

For a new deployment:

1. Open the app in Streamlit Community Cloud.
2. Go to app settings / advanced settings.
3. Set the Python version to 3.11.
4. Deploy the app.

If the app is already deployed with Python 3.14, delete the Streamlit Cloud app
and redeploy it with Python 3.11 selected in Advanced settings. A normal reboot
can keep the same Python runtime.

## Data and Outputs

- Raw data: `Churnzero_data/CZ_raw/`
- Processed data: `Churnzero_data/CZ_processed/`
- Scoring exports: `retention_flow_outputs/*.csv`
- Charts: `retention_flow_outputs/charts/`
- Dashboard assets: `assets/`

## Development Notes

- Python target: `python-3.11`
- Main entry point: `app.py`
- Notebook entry point for rebuilding outputs: `Retentionflow.ipynb`
- Local caches, model binaries, virtual environments, and generated artifact folders are ignored by Git.
