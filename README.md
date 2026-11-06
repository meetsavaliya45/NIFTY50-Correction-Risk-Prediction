# NIFTY 50 Correction-Risk Prediction Using Machine Learning

**A machine learning project exploring whether historical market indicators can help identify the risk of a NIFTY 50 market correction within the next 20 observations.**

---

## 1. Project Overview

Market corrections are an important aspect of financial markets. Identifying conditions associated with a potential correction can help investors and analysts better understand market risk.

This project investigates the relationship between NIFTY 50 market behaviour and correction risk using historical financial indicators and machine learning techniques.

The project combines:

- NIFTY 50 price and valuation data
- India VIX (market volatility indicator)
- USD/INR exchange-rate data
- Historical returns and rolling volatility features

Three approaches are evaluated:

1. Logistic Regression
2. Random Forest Classifier
3. India VIX threshold-based benchmark

The objective is to compare their ability to identify observations labelled as potential market corrections.

> **Note:** This is a historical machine learning experiment for academic and research purposes. It is not a live trading system or investment recommendation.

---

## 2. Project Objectives

- Study the relationship between market indicators and NIFTY 50 correction risk.
- Investigate the usefulness of valuation, volatility and currency indicators.
- Develop machine learning models for binary classification.
- Compare Logistic Regression and Random Forest performance.
- Evaluate the models using precision, recall, F1-score, PR-AUC and ROC-AUC.
- Identify the most influential features using feature-importance analysis.
- Establish a baseline using an India VIX threshold rule.

---

## 3. Market Correction Criteria

For this project, the working definition of a market correction is approximately a **10% decline from a reference market peak**.

The prediction horizon is the next **20 observations**, intended to represent trading observations.

The problem is formulated as a binary classification task:

| Class | Meaning |
|---|---|
| 1 | Correction within the next 20 observations |
| 0 | No correction within the next 20 observations |

### Important clarification

The models use the supplied dataset column `correction_within_20d` as the target variable.

The exact reference-peak lookback and target-generation formula were not fully documented in the supplied dataset. The labels were retained as provided and were not independently verified.

Therefore, the reported results measure agreement with the supplied labels, not independently confirmed real-world market corrections.

---

## 4. Dataset Description

**Historical coverage:** December 2010 – December 2020

The dataset contains approximately 2,495 observations before excluding rows with missing target values.

### Indicators used

| Category | Variables |
|---|---|
| NIFTY 50 | Open, High, Low, Close |
| Valuation | NIFTY 50 P/E |
| Market volatility | India VIX |
| Currency | USD/INR |
| Derived features | Returns, rolling volatility and other historical indicators |

### Feature engineering

The following types of features were constructed:

- NIFTY 50 returns over 1, 5, 20 and 60 observations.
- NIFTY 50 rolling volatility over 20 and 60 observations.
- Historical drawdown measures.
- India VIX changes and rolling statistics.
- USD/INR returns and rolling volatility.

Features were constructed from raw market columns rather than relying on pre-existing helper columns that could contain target-related or future information.

**Prediction timing assumption:** Current-day OHLC and closing indicators are included. Therefore, the model assumes that predictions are generated after the relevant trading day's close.

---

## 5. Methodology

The project follows a chronological machine learning workflow.

### Step 1: Data preparation

- Load the historical dataset.
- Parse dates and arrange observations chronologically.
- Inspect missing values and target distribution.
- Exclude observations with missing target values.

### Step 2: Feature engineering

Construct historical returns, volatility and other market-indicator features.

### Step 3: Train-validation-test split

The data is divided chronologically using a nominal:

- 70% Training set
- 15% Validation set
- 15% Testing set

A 20-observation purge gap is used at the split boundaries to reduce overlap associated with the forward prediction horizon.

No random shuffling is performed.

### Step 4: Missing-value treatment

- Median imputation is performed through machine learning pipelines.
- Imputation statistics are learned from training data.
- Logistic Regression uses standard scaling.
- Features with excessive missingness are assessed using training data.

### Step 5: Model training

The following models are trained and evaluated:

**Logistic Regression**

An interpretable linear classification baseline using median imputation, feature scaling and class weighting.

**Random Forest**

An ensemble of decision trees designed to capture nonlinear relationships and interactions between market indicators.

**India VIX threshold benchmark**

A simple rule-based classifier using an India VIX threshold selected through validation performance.

### Step 6: Threshold selection

Classification thresholds are selected using validation data rather than the final test set.

### Step 7: Final evaluation

The selected models are evaluated on the held-out chronological test set.

---

## 6. Models and Evaluation Metrics

The following metrics are used:

| Metric | Purpose |
|---|---|
| Accuracy | Overall proportion of correct predictions |
| Precision | Proportion of predicted corrections that are positive labels |
| Recall | Proportion of positive labels correctly identified |
| F1-score | Balance between precision and recall |
| PR-AUC | Performance across precision-recall thresholds |
| ROC-AUC | Ability to distinguish between the two classes |
| Confusion Matrix | Breakdown of true and incorrect predictions |

Particular attention is given to:

- **False alarms:** Observations predicted as corrections when the supplied label is negative.
- **Missed corrections:** Positive-labelled observations that the model fails to identify.

---

## 7. Model Performance

The following are the previously reported test-evaluation results for the experiment.

**Final test set:** 372 observations, including 190 positive labels.

| Metric | Logistic Regression | Random Forest | India VIX Rule |
|---|---:|---:|---:|
| Precision | 90.2% | 90.0% | 49.0% |
| Recall | 87.4% | 75.8% | 92.1% |
| F1-score | 0.888 | 0.823 | 0.640 |
| Accuracy | 88.7% | 83.3% | 47.0% |
| PR-AUC | 0.947 | 0.867 | 0.492 |
| ROC-AUC | 0.950 | 0.906 | 0.461 |
| False alarms | 18 | 16 | 182 |
| Missed positive labels | 24 | 46 | 15 |

*These are previously reported experiment results and should be checked against a fresh execution of the final notebook before being treated as independently reproduced results.*

### Confusion Matrix Results

| Model | True Negatives | False Positives | False Negatives | True Positives |
|---|---:|---:|---:|---:|
| Logistic Regression | 164 | 18 | 24 | 166 |
| Random Forest | 166 | 16 | 46 | 144 |
| India VIX Rule | 0 | 182 | 15 | 175 |

### Key observation

Logistic Regression achieved the strongest overall balance between precision and recall in the reported evaluation.

Random Forest produced slightly fewer false alarms but missed more positive labels.

The India VIX threshold rule achieved high recall but generated a substantially larger number of false alarms.

---

## 8. Feature Importance

Random Forest impurity-based feature importance was used to identify influential input variables.

The leading features included:

1. `nifty50_close`
2. `nifty50_low`
3. `nifty50_high`
4. `nifty50_open`
5. `nifty_return_5d_recalc`
6. `nifty50_pe`
7. `nifty_volatility_20d_recalc`
8. `usd_inr_reference_rate`
9. `india_vix_close`
10. `nifty_return_60d_recalc`
11. `usd_inr_close`
12. `nifty_return_20d_recalc`

The NIFTY 50 price-related variables dominated the top positions in this model's importance ranking.

**Interpretation:** Feature importance indicates how variables contributed to this fitted model's predictions. It does not establish causation or prove that these variables independently drive market corrections.

---

## 9. Key Project Outcomes

- Developed a historical machine learning workflow for correction-risk classification.
- Compared two machine learning approaches with a single-indicator benchmark.
- Evaluated model performance using multiple classification metrics.
- Identified the trade-off between missed corrections and false alarms.
- Analysed the relative importance of market indicators.
- Found that Logistic Regression achieved the best overall reported test performance.

---

## 10. Limitations

The following limitations must be considered when interpreting the results:

1. The dataset contains date/calendar irregularities and suspicious price movements that require further investigation.
2. The supplied correction labels showed inconsistencies against at least one independently tested interpretation of the 20-observation correction rule.
3. The exact target-generation methodology requires verification.
4. The historical dataset ends in December 2020.
5. The model has not been validated using live market data.
6. The results do not account for transaction costs, execution constraints or portfolio-level risk.
7. Current-day closing indicators mean the prediction is assumed to occur after market close.

Consequently, the reported performance should be interpreted as an experimental result rather than evidence of a reliable live-market forecasting system.

---

## 11. Future Scope

Potential improvements include:

- Reconstructing and independently validating the correction target.
- Improving historical data quality and trading-calendar consistency.
- Extending the dataset beyond 2020.
- Implementing walk-forward validation across different market regimes.
- Exploring probability calibration and alternative classification thresholds.
- Incorporating additional financial indicators where reliable historical data is available.
- Developing an interactive dashboard for historical analysis and model predictions.
- Integrating updated market data after validating feature compatibility.

---

## 12. Project Deliverables

The repository contains or is intended to contain:

- Historical market-indicator dataset.
- Jupyter Notebook containing data preparation, feature engineering, training and evaluation.
- Python model implementation.
- Written project report.
- Club presentation.
- Model comparison results.
- Confusion matrices.
- Precision-recall curve.
- Random Forest feature-importance analysis.
- Logistic Regression coefficients.

---

## 13. Conclusion

This project explores the use of machine learning and historical financial indicators to study NIFTY 50 correction risk.

By comparing Logistic Regression, Random Forest and an India VIX threshold benchmark, the experiment investigates whether combining market information can provide a more informative classification approach than relying on a single indicator.

The reported results favour Logistic Regression on the supplied test labels. However, target verification, data-quality improvements and validation on more recent market conditions are essential before extending the work into a live prediction application.

---

