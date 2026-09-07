# inspect_zero_amount.py

import pandas as pd


DATA_PATH = "data/processed/transactions_cleaned.parquet"


df = pd.read_parquet(DATA_PATH)

zero_transactions = df[df["amount"] == 0]

print("\nZERO AMOUNT TRANSACTIONS")
print("=" * 80)

print(zero_transactions.to_string(index=False))

print("\nTRANSACTION TYPES:")
print(zero_transactions["type"].value_counts())

print("\nFRAUD DISTRIBUTION:")
print(zero_transactions["isFraud"].value_counts())