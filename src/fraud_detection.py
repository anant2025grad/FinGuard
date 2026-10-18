"""
FinGuard - Fraud Detection Engine

Uses Isolation Forest to detect anomalous financial transactions.

Pipeline:

transactions.csv
        ↓
Feature Selection
        ↓
Feature Scaling
        ↓
Isolation Forest
        ↓
Anomaly Score
        ↓
FinGuard Risk Score (0-100)
        ↓
Fraud Classification
        ↓
Model Evaluation

Author: Anant Palve
Project: FinGuard
"""

import os
import pickle

import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

DATA_PATH = "data/synthetic/transactions.csv"

MODEL_DIRECTORY = "models"

MODEL_PATH = os.path.join(
    MODEL_DIRECTORY,
    "isolation_forest.pkl"
)

SCALER_PATH = os.path.join(
    MODEL_DIRECTORY,
    "scaler.pkl"
)

OUTPUT_PATH = (
    "data/synthetic/"
    "transactions_scored.csv"
)

RANDOM_SEED = 42

# Expected percentage of anomalies.
# This should roughly correspond to the fraud rate
# generated in data_generator.py.
CONTAMINATION = 0.05

FEATURES = [
    "amount",
    "is_new_device",
    "failed_logins",
    "transactions_last_24h",
    "distance_from_last_transaction",
    "account_age_days",
    "is_after_hours",
    "hour",
    "day_of_week",
    "is_weekend",
    "log_amount",
    "login_velocity_score",
    "geographic_risk",
    "amount_velocity_score",
]

def load_data():
    """
    Load the synthetic transaction dataset.
    """

    print("\nLoading transaction data...")

    df = pd.read_csv(
        DATA_PATH
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    print(
        f"Loaded {len(df):,} transactions."
    )

    return df

def validate_features(df):
    """
    Make sure every feature required by the model
    exists in the dataset.
    """

    missing_features = [
        feature
        for feature in FEATURES
        if feature not in df.columns
    ]

    if missing_features:

        raise ValueError(
            "Missing required features: "
            + ", ".join(missing_features)
        )

    print(
        f"All {len(FEATURES)} ML features are present."
    )

def prepare_features(df):
    """
    Extract and clean the features used by the
    Isolation Forest model.
    """

    X = df[
        FEATURES
    ].copy()

    # Replace infinite values
    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Fill missing numerical values
    X = X.fillna(
        X.median()
    )

    return X

def scale_features(X):
    """
    Standardize features so that large numerical
    ranges do not dominate the model.
    """

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X
    )

    return X_scaled, scaler

def train_model(X_scaled):
    """
    Train the Isolation Forest anomaly detector.
    """

    print(
        "\nTraining Isolation Forest..."
    )

    model = IsolationForest(
        n_estimators=300,
        contamination=CONTAMINATION,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )

    model.fit(
        X_scaled
    )

    print(
        "Model training complete."
    )

    return model

def generate_anomaly_scores(
    model,
    X_scaled
):
    """
    Generate raw anomaly scores.

    Isolation Forest returns lower scores for
    more anomalous observations.

    We invert the score so that:
        higher = more suspicious
    """

    raw_scores = model.decision_function(
        X_scaled
    )

    anomaly_scores = -raw_scores

    return anomaly_scores

def calculate_risk_score(
    anomaly_scores
):
    """
    Convert anomaly scores into a normalized
    0-100 FinGuard Risk Score.

    0   = very low anomaly
    100 = extremely anomalous
    """

    minimum = anomaly_scores.min()
    maximum = anomaly_scores.max()

    normalized = (
        (anomaly_scores - minimum)
        / (maximum - minimum)
    )

    risk_scores = (
        normalized * 100
    )

    return np.round(
        risk_scores,
        2
    )

def classify_transactions(
    risk_scores
):
    """
    Convert the continuous risk score into
    operational risk categories.
    """

    risk_category = np.select(

        [
            risk_scores < 20,
            risk_scores < 40,
            risk_scores < 60,
            risk_scores < 80,
            risk_scores >= 80,
        ],

        [
            "Low",
            "Moderate",
            "Elevated",
            "High",
            "Critical",
        ],

        default="Low"
    )

    return risk_category

def evaluate_model(
    df,
    risk_scores,
    model,
    X_scaled
):
    """
    Compare model predictions against the known
    synthetic fraud labels.

    Important:
    The model itself is unsupervised.

    The is_fraud column is ONLY used here for
    evaluation, not training.
    """

    actual = df[
        "is_fraud"
    ].astype(int)

    # Isolation Forest:
    #   1  = normal
    #  -1  = anomaly
    predictions = model.predict(
        X_scaled
    )

    # Convert:
    #   -1 → 1 (fraud/anomaly)
    #    1 → 0 (normal)
    predicted_fraud = (
        predictions == -1
    ).astype(int)

    print(
        "\n" + "=" * 60
    )

    print(
        "MODEL PERFORMANCE"
    )

    print(
        "=" * 60
    )

    print(
        "\nClassification Report:\n"
    )

    print(
        classification_report(
            actual,
            predicted_fraud,
            target_names=[
                "Legitimate",
                "Fraud"
            ],
            zero_division=0
        )
    )

    precision = precision_score(
        actual,
        predicted_fraud,
        zero_division=0
    )

    recall = recall_score(
        actual,
        predicted_fraud,
        zero_division=0
    )

    f1 = f1_score(
        actual,
        predicted_fraud,
        zero_division=0
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall:    {recall:.4f}"
    )

    print(
        f"F1 Score:  {f1:.4f}"
    )

    print(
        "\nConfusion Matrix:\n"
    )

    matrix = confusion_matrix(
        actual,
        predicted_fraud
    )

    print(
        matrix
    )

    # ROC-AUC using continuous risk scores
    try:

        auc = roc_auc_score(
            actual,
            risk_scores
        )

        print(
            f"\nROC-AUC:   {auc:.4f}"
        )

    except ValueError:

        print(
            "\nROC-AUC could not be calculated."
        )

def save_model(
    model,
    scaler
):
    """
    Save the trained model and scaler.
    """

    os.makedirs(
        MODEL_DIRECTORY,
        exist_ok=True
    )

    with open(
        MODEL_PATH,
        "wb"
    ) as file:

        pickle.dump(
            model,
            file
        )

    with open(
        SCALER_PATH,
        "wb"
    ) as file:

        pickle.dump(
            scaler,
            file
        )

    print(
        f"\nModel saved to: {MODEL_PATH}"
    )

    print(
        f"Scaler saved to: {SCALER_PATH}"
    )

def save_scored_data(
    df
):
    """
    Save transactions with FinGuard risk scores.
    """

    os.makedirs(
        os.path.dirname(
            OUTPUT_PATH
        ),
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        f"\nScored dataset saved to:"
        f"\n{OUTPUT_PATH}"
    )

def display_risk_summary(
    df
):
    """
    Display useful summary information
    about the generated risk scores.
    """

    print(
        "\n" + "=" * 60
    )

    print(
        "FINGUARD RISK SUMMARY"
    )

    print(
        "=" * 60
    )

    print(
        "\nRisk Category Distribution:"
    )

    print(
        df[
            "risk_category"
        ].value_counts()
    )

    print(
        "\nAverage Risk Score:"
    )

    print(
        f"{df['risk_score'].mean():.2f}"
    )

    print(
        "\nAverage Fraud Risk Score:"
    )

    fraud_average = df[
        df["is_fraud"] == 1
    ]["risk_score"].mean()

    print(
        f"{fraud_average:.2f}"
    )

    print(
        "\nAverage Legitimate Risk Score:"
    )

    legitimate_average = df[
        df["is_fraud"] == 0
    ]["risk_score"].mean()

    print(
        f"{legitimate_average:.2f}"
    )

    print(
        "\nTop 10 Highest-Risk Transactions:"
    )

    columns = [
        "transaction_id",
        "account_id",
        "amount",
        "risk_score",
        "risk_category",
        "is_fraud",
        "fraud_type"
    ]

    top_transactions = (
        df
        .sort_values(
            "risk_score",
            ascending=False
        )
        .head(10)
    )

    print(
        top_transactions[
            columns
        ].to_string(
            index=False
        )
    )

def main():

    print(
        "=" * 60
    )

    print(
        "FINGUARD FRAUD DETECTION ENGINE"
    )

    print(
        "=" * 60
    )

    df = load_data()

    validate_features(
        df
    )

    print(
        "\nPreparing ML features..."
    )

    X = prepare_features(
        df
    )

    print(
        "Scaling features..."
    )

    X_scaled, scaler = scale_features(
        X
    )

    model = train_model(
        X_scaled
    )

    print(
        "\nGenerating anomaly scores..."
    )

    anomaly_scores = generate_anomaly_scores(
        model,
        X_scaled
    )

    risk_scores = calculate_risk_score(
        anomaly_scores
    )

    df["anomaly_score"] = np.round(
        anomaly_scores,
        6
    )

    df["risk_score"] = risk_scores

    df["risk_category"] = classify_transactions(
        risk_scores
    )

    evaluate_model(
        df,
        risk_scores,
        model,
        X_scaled
    )

    save_model(
        model,
        scaler
    )

    save_scored_data(
        df
    )

    display_risk_summary(
        df
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "FINGUARD FRAUD DETECTION COMPLETE"
    )

    print(
        "=" * 60
    )


if __name__ == "__main__":
    main()
