# Homework 4 — Evaluation

Homework for the model evaluation module of the DataTalksClub Machine Learning Zoomcamp 2026.

## Topics

- Data preprocessing and train/validation/test splitting
- One-hot encoding with DictVectorizer
- Logistic regression with Scikit-Learn
- ROC AUC and feature importance
- Precision and recall
- Classification thresholds
- F1 score optimization
- K-fold cross-validation
- Hyperparameter tuning and regularization

## Dataset

`course_lead_scoring_2026.csv`

The model predicts `converted` (whether a potential customer signs up for the platform) using lead characteristics such as industry, location, lead source, interaction count, and number of courses viewed.

The focus is on evaluating classification performance and selecting optimal thresholds and regularization parameters.

## Running

From the repository root:

```bash
uv run python hw4/hw4.py
```
