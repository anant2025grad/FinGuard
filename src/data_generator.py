"""
FinGuard - Synthetic Financial Transaction Data Generator

Generates realistic synthetic banking transaction data for:
- Fraud detection
- Behavioral anomaly detection
- Financial risk scoring
- Backtesting experiments

The dataset intentionally contains several different fraud patterns:
    1. Account takeover
    2. Transaction velocity fraud
    3. Unusual transaction amount
    4. Impossible travel
    5. New device fraud
    6. Repeated failed-login activity

Author: Anant Palve
Project: FinGuard
"""

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

RANDOM_SEED = 42

NUM_ACCOUNTS = 5000
NUM_TRANSACTIONS = 100000

START_DATE = datetime(2025, 1, 1)
END_DATE = datetime(2025, 6, 30)

FRAUD_RATE = 0.05

OUTPUT_PATH = "data/synthetic/transactions.csv"


# Reproducibility
np.random.seed(RANDOM_SEED)
rng = np.random.default_rng(RANDOM_SEED)

MERCHANT_CATEGORIES = [
    "Grocery",
    "Restaurant",
    "Gas Station",
    "Retail",
    "Travel",
    "Entertainment",
    "Healthcare",
    "Utilities",
    "Online Shopping",
    "Electronics",
    "Financial Services",
]

LOCATIONS = [
    "Pittsburgh",
    "Philadelphia",
    "New York",
    "Chicago",
    "Boston",
    "Washington DC",
    "Cleveland",
    "Columbus",
    "Detroit",
    "Baltimore",
]

DEVICE_TYPES = [
    "iPhone",
    "Android",
    "Windows",
    "Mac",
    "Tablet",
]

FRAUD_TYPES = [
    "Account Takeover",
    "Velocity Fraud",
    "Unusual Amount",
    "Impossible Travel",
    "New Device",
    "Failed Login Attack",
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def generate_account_ids(num_accounts):

    return [
        f"ACC{str(i).zfill(6)}"
        for i in range(1, num_accounts + 1)
    ]


def generate_device_ids(num_devices):

    return [
        f"DEV{str(i).zfill(7)}"
        for i in range(1, num_devices + 1)
    ]


def random_dates(start_date, end_date, n):

    total_seconds = int((end_date - start_date).total_seconds())

    random_seconds = rng.integers(
        0,
        total_seconds,
        size=n
    )

    return [
        start_date + timedelta(seconds=int(seconds))
        for seconds in random_seconds
    ]


def generate_transaction_amounts(n):

    amounts = rng.lognormal(
        mean=3.5,
        sigma=1.0,
        size=n
    )

    # Keep normal transactions within a reasonable range
    amounts = np.clip(amounts, 2, 2500)

    return np.round(amounts, 2)


def generate_account_age(n):

    return rng.integers(
        low=30,
        high=3650,
        size=n
    )


def generate_failed_logins(n):

    return rng.poisson(
        lam=0.4,
        size=n
    )


def generate_transaction_velocity(n):

    velocity = rng.poisson(
        lam=2.5,
        size=n
    )

    return np.clip(
        velocity,
        0,
        20
    )

def generate_accounts():
    """
    Create a synthetic account table.

    Each account receives:
    - Account ID
    - Home location
    - Primary device
    - Account age
    - Typical transaction amount
    """

    account_ids = generate_account_ids(NUM_ACCOUNTS)

    accounts = pd.DataFrame({
        "account_id": account_ids,

        "home_location": rng.choice(
            LOCATIONS,
            size=NUM_ACCOUNTS
        ),

        "primary_device": rng.choice(
            DEVICE_TYPES,
            size=NUM_ACCOUNTS
        ),

        "account_age_days": generate_account_age(
            NUM_ACCOUNTS
        ),

        "typical_transaction_amount": np.round(
            rng.lognormal(
                mean=3.4,
                sigma=0.55,
                size=NUM_ACCOUNTS
            ),
            2
        ),
    })

    return accounts

def generate_transactions(accounts):

    account_indices = rng.integers(
        0,
        len(accounts),
        size=NUM_TRANSACTIONS
    )

    selected_accounts = accounts.iloc[
        account_indices
    ].reset_index(drop=True)

    timestamps = random_dates(
        START_DATE,
        END_DATE,
        NUM_TRANSACTIONS
    )

    merchant_categories = rng.choice(
        MERCHANT_CATEGORIES,
        size=NUM_TRANSACTIONS
    )

    locations = selected_accounts[
        "home_location"
    ].values.copy()

    devices = selected_accounts[
        "primary_device"
    ].values.copy()

    typical_amounts = selected_accounts[
        "typical_transaction_amount"
    ].values

    # Generate amounts around each account's typical spending
    amounts = (
        typical_amounts
        *rng.lognormal(
            mean=0,
            sigma=0.45,
            size=NUM_TRANSACTIONS
        )
    )

    amounts = np.clip(
        amounts,
        2,
        2500
    )

    amounts = np.round(
        amounts,
        2
    )

    # Login behavior
    failed_logins = generate_failed_logins(
        NUM_TRANSACTIONS
    )

    # Transaction velocity
    transaction_velocity = generate_transaction_velocity(
        NUM_TRANSACTIONS
    )

    # Most normal transactions happen during reasonable hours.
    hours = np.array([
        timestamp.hour
        for timestamp in timestamps
    ])

    # Small probability of nighttime transactions
    is_after_hours = (
        (hours < 6) |
        (hours >= 23)
    ).astype(int)

    # Build dataframe
    transactions = pd.DataFrame({

        "transaction_id": [
            f"TXN{str(i).zfill(8)}"
            for i in range(1, NUM_TRANSACTIONS + 1)
        ],

        "account_id": selected_accounts[
            "account_id"
        ].values,

        "timestamp": timestamps,

        "amount": amounts,

        "merchant_category": merchant_categories,

        "location": locations,

        "device_id": devices,

        "is_new_device": np.zeros(
            NUM_TRANSACTIONS,
            dtype=int
        ),

        "failed_logins": failed_logins,

        "transactions_last_24h": transaction_velocity,

        "distance_from_last_transaction": np.zeros(
            NUM_TRANSACTIONS
        ),

        "account_age_days": selected_accounts[
            "account_age_days"
        ].values,

        "is_after_hours": is_after_hours,

        "is_fraud": np.zeros(
            NUM_TRANSACTIONS,
            dtype=int
        ),

        "fraud_type": np.full(
            NUM_TRANSACTIONS,
            "None",
            dtype=object
        ),
    })

    return transactions

def inject_impossible_travel(df, fraud_indices):

    for index in fraud_indices:

        current_location = df.at[
            index,
            "location"
        ]

        alternative_locations = [
            location
            for location in LOCATIONS
            if location != current_location
        ]

        new_location = rng.choice(
            alternative_locations
        )

        df.at[
            index,
            "location"
        ] = new_location

        # Simulate large geographic distance
        df.at[
            index,
            "distance_from_last_transaction"
        ] = rng.uniform(
            500,
            3000
        )

        df.at[
            index,
            "is_fraud"
        ] = 1

        df.at[
            index,
            "fraud_type"
        ] = "Impossible Travel"

def inject_new_device_fraud(df, fraud_indices):

    for index in fraud_indices:

        df.at[
            index,
            "device_id"
        ] = f"UNKNOWN_DEV_{index}"

        df.at[
            index,
            "is_new_device"
        ] = 1

        df.at[
            index,
            "is_fraud"
        ] = 1

        df.at[
            index,
            "fraud_type"
        ] = "New Device"

def inject_failed_login_fraud(df, fraud_indices):

    for index in fraud_indices:

        df.at[
            index,
            "failed_logins"
        ] = rng.integers(
            5,
            20
        )

        df.at[
            index,
            "is_after_hours"
        ] = rng.choice(
            [0, 1],
            p=[0.3, 0.7]
        )

        df.at[
            index,
            "is_fraud"
        ] = 1

        df.at[
            index,
            "fraud_type"
        ] = "Failed Login Attack"

def inject_velocity_fraud(df, fraud_indices):

    for index in fraud_indices:

        df.at[
            index,
            "transactions_last_24h"
        ] = rng.integers(
            15,
            50
        )

        df.at[
            index,
            "is_fraud"
        ] = 1

        df.at[
            index,
            "fraud_type"
        ] = "Velocity Fraud"

def inject_unusual_amount_fraud(df, fraud_indices):

    for index in fraud_indices:

        current_amount = df.at[
            index,
            "amount"
        ]

        multiplier = rng.uniform(
            5,
            20
        )

        df.at[
            index,
            "amount"
        ] = round(
            min(
                current_amount * multiplier,
                25000
            ),
            2
        )

        df.at[
            index,
            "is_fraud"
        ] = 1

        df.at[
            index,
            "fraud_type"
        ] = "Unusual Amount"

def inject_account_takeover(df, fraud_indices):
    """
    Simulate a coordinated account takeover.

    Characteristics:
    - New device
    - High transaction amount
    - Multiple failed logins
    - High transaction velocity
    - After-hours activity
    - Geographic anomaly
    """

    for index in fraud_indices:

        df.at[
            index,
            "device_id"
        ] = f"ATTACKER_DEV_{index}"

        df.at[
            index,
            "is_new_device"
        ] = 1

        df.at[
            index,
            "failed_logins"
        ] = rng.integers(
            5,
            15
        )

        df.at[
            index,
            "transactions_last_24h"
        ] = rng.integers(
            10,
            30
        )

        df.at[
            index,
            "amount"
        ] = round(
            rng.uniform(
                1000,
                15000
            ),
            2
        )

        df.at[
            index,
            "distance_from_last_transaction"
        ] = rng.uniform(
            500,
            3000
        )

        df.at[
            index,
            "is_after_hours"
        ] = rng.choice(
            [0, 1],
            p=[0.2, 0.8]
        )

        df.at[
            index,
            "is_fraud"
        ] = 1

        df.at[
            index,
            "fraud_type"
        ] = "Account Takeover"

def inject_fraud(df):

    num_fraud = int(
        NUM_TRANSACTIONS * FRAUD_RATE
    )

    fraud_indices = rng.choice(
        df.index,
        size=num_fraud,
        replace=False
    )

    # Divide fraud cases among several attack types
    shuffled_indices = rng.permutation(
        fraud_indices
    )

    chunks = np.array_split(
        shuffled_indices,
        len(FRAUD_TYPES)
    )

    inject_account_takeover(
        df,
        chunks[0]
    )

    inject_velocity_fraud(
        df,
        chunks[1]
    )

    inject_unusual_amount_fraud(
        df,
        chunks[2]
    )

    inject_impossible_travel(
        df,
        chunks[3]
    )

    inject_new_device_fraud(
        df,
        chunks[4]
    )

    inject_failed_login_fraud(
        df,
        chunks[5]
    )

    return df

def engineer_features(df):

    df["hour"] = df[
        "timestamp"
    ].dt.hour

    df["day_of_week"] = df[
        "timestamp"
    ].dt.dayofweek

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    # Log-transform transaction amount.
    # Useful for ML models because transaction amounts
    # can have a heavily skewed distribution.
    df["log_amount"] = np.log1p(
        df["amount"]
    )

    # Interaction features
    df["login_velocity_score"] = (
        df["failed_logins"]
        * df["transactions_last_24h"]
    )

    df["geographic_risk"] = (
        df[
            "distance_from_last_transaction"
        ] / 3000
    )

    df["amount_velocity_score"] = (
        df["amount"]
        * (1 + df["transactions_last_24h"])
    )

    # Cap extreme values
    df["geographic_risk"] = np.clip(
        df["geographic_risk"],
        0,
        1
    )

    return df

def validate_dataset(df):

    print("\n" + "=" * 60)
    print("DATASET VALIDATION")
    print("=" * 60)

    print(
        f"Total transactions: {len(df):,}"
    )

    print(
        f"Unique accounts: "
        f"{df['account_id'].nunique():,}"
    )

    fraud_count = df[
        "is_fraud"
    ].sum()

    fraud_percentage = (
        fraud_count / len(df)
    ) * 100

    print(
        f"Fraudulent transactions: "
        f"{fraud_count:,}"
    )

    print(
        f"Fraud rate: "
        f"{fraud_percentage:.2f}%"
    )

    print("\nFraud distribution:")

    fraud_distribution = (
        df[df["is_fraud"] == 1]
        ["fraud_type"]
        .value_counts()
    )

    print(
        fraud_distribution
    )

    print("\nTransaction statistics:")

    print(
        df["amount"].describe()
    )

    print("\nMissing values:")

    print(
        df.isnull().sum()
    )

    print("=" * 60)

def save_dataset(df):

    output_directory = os.path.dirname(
        OUTPUT_PATH
    )

    os.makedirs(
        output_directory,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        f"\nDataset saved to: {OUTPUT_PATH}"
    )

def main():

    print("=" * 60)
    print("FINGUARD DATA GENERATION")
    print("=" * 60)

    print(
        f"\nGenerating "
        f"{NUM_TRANSACTIONS:,} transactions "
        f"across "
        f"{NUM_ACCOUNTS:,} accounts..."
    )

    # Step 1: Generate accounts
    print("\n[1/5] Generating accounts...")

    accounts = generate_accounts()

    # Step 2: Generate normal transactions
    print(
        "[2/5] Generating normal transactions..."
    )

    transactions = generate_transactions(
        accounts
    )

    # Convert timestamp column
    transactions["timestamp"] = pd.to_datetime(
        transactions["timestamp"]
    )

    # Step 3: Inject fraud
    print(
        "[3/5] Injecting fraud scenarios..."
    )

    transactions = inject_fraud(
        transactions
    )

    # Step 4: Engineer additional features
    print(
        "[4/5] Engineering behavioral features..."
    )

    transactions = engineer_features(
        transactions
    )

    # Sort chronologically
    transactions = transactions.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    # Step 5: Validate and save
    print(
        "[5/5] Validating and saving..."
    )

    validate_dataset(
        transactions
    )

    save_dataset(
        transactions
    )

    print(
        "\nFinGuard data generation complete!"
    )

if __name__ == "__main__":
    main()