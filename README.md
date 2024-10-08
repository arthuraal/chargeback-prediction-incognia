# Chargeback Prediction with LightGBM

## Overview

This project is focused on predicting chargebacks in transaction data. Chargebacks result in a total loss, while successful transactions generate 15% profit. The goal is to create a model that helps prevent chargebacks and maximize profitability.

## Steps

### 1. Data Preprocessing

- **Missing Data**: Handled missing values through imputations and transformations.
- **Feature Engineering**: New features related to fraud detection and transaction behavior were created, such as:
  - `transaction_velocity_device`
  - `fraud_installations_ratio`
  - `rolling_avg_7d`
  - Temporal features like `day_of_week` and `hour_of_day`

### 2. Model Training

- We use LightGBM to handle the imbalance in the chargeback data.

### 3. Model Evaluation

The model's performance is evaluated using the following metrics:

- **Precision**: How well the model correctly predicts chargebacks.
- **Recall**: How well the model captures all chargebacks.
- **F1-Score**: Combination of precision and recall.
- **Financial Metric**: Focus on maximizing profit by preventing chargebacks.

The goal is to achieve high recall for chargebacks while keeping overall performance high.

### 4. Threshold Tuning

Threshold tuning was applied to adjust the decision threshold for chargebacks to balance Precision and Recall. A plot was used to visualize the best threshold based on Precision, Recall, F1-Score, and ROC-AUC.

## Installation

Create a virtual environment and activate it:

```bash
python -m venv env
source env/bin/activate
```

After that, you can install the necessary Python libraries by running the following command:

```bash
pip install -r requirements.txt
```
