# NIFTY 50 Correction-Risk Dashboard — Deployment Starter

This folder contains a Streamlit dashboard starter for the NIFTY 50 correction-risk project.

## Included features

- Overview and historical dataset exploration
- Recomputed chronological holdout comparison for Logistic Regression, Random Forest, and an India VIX threshold baseline
- Confusion matrices and Random Forest feature importance
- Experimental latest-daily-data scoring using Yahoo Finance tickers:
  - `^NSEI` — NIFTY 50
  - `^INDIAVIX` — India VIX
  - `USDINR=X` — USD/INR
- CSV downloads for historical data, metrics, test predictions, and feature importance

## Important research and data limitations

1. The included historical dataset ends on **31 December 2020**.
2. The supplied `correction_within_20d` target is used as-is. Its exact construction and correctness have not been independently validated.
3. The dashboard retrains its own deployable feature set. It intentionally excludes NIFTY P/E and USD/INR reference rate because the selected free quote feed does not reliably provide these values. Therefore, **the dashboard's metrics are not the same as the earlier reported metrics** and should be treated as a fresh experiment.
4. Yahoo Finance data availability, timestamps, completeness, and delays are not guaranteed. The live page is an experimental daily-data demonstration, not an exchange-grade real-time feed.
5. Missing live inputs are imputed with training medians and explicitly displayed. If important inputs are missing, do not treat the score as meaningful.
6. This is an educational research prototype, not investment advice or a trading system.

## Run locally

From this `dashboard` directory:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Push the entire `dashboard/` folder to your GitHub repository, preserving its structure.
2. Go to https://share.streamlit.io/ and sign in.
3. Choose **Create app**.
4. Select your GitHub repository and branch.
5. Set the main file path to `dashboard/app.py`.
6. Deploy. Streamlit installs dependencies from `dashboard/requirements.txt` when the dependency file is in the app directory.
7. Test every page after deployment.

### Team-only access

For a private team dashboard, deploy from a **private GitHub repository** and set app visibility to **Only specific people can view this app** in Streamlit app settings. Invite club members by email as viewers. Viewers sign in using Google or a one-time emailed link. Do not give team members GitHub write access unless they should be able to modify the code.

Streamlit Community Cloud currently documents a limit of one private app per account/workspace. Confirm current plan/platform limits before relying on this for a long-term club service.

## Suggested repository layout

```text
repository/
├── README.md
├── dashboard/
│   ├── app.py
│   ├── requirements.txt
│   ├── .gitignore
│   ├── .streamlit/
│   │   └── config.toml
│   ├── data/
│   │   └── historical_nifty50.csv
│   └── src/
│       └── modeling.py
├── notebooks/
├── reports/
├── presentation/
└── outputs/
```

## Before presenting live predictions

- Validate the supplied target labels and document the exact peak/look-forward logic.
- Rebuild and verify the model with a strictly chronological, leakage-audited evaluation.
- Confirm that all live inputs are available and timestamp-aligned.
- Replace Yahoo Finance with a licensed exchange/vendor API if reliable operational data is required.
- Monitor data failures, model drift, and prediction quality.
