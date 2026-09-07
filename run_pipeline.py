# run_pipeline.py

import pandas as pd

from src.config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
    QUALITY_REPORT_PATH
)


from src.data.profiling import profile_data
from src.data.cleaning import clean_transactions

from src.data.business_validation import run_business_validation

from src.data.quality_report import (
    generate_data_quality_report,
    save_data_quality_report
)


RAW_DATA_PATH = "data/raw/transactions.csv"

PROCESSED_DATA_PATH = (
    "data/processed/transactions_cleaned.parquet"
)

QUALITY_REPORT_PATH = (
    "reports/data_quality_report.csv"
)

def main():

    print("Loading raw dataset...")

    df = pd.read_csv(RAW_DATA_PATH)

    print("Dataset loaded successfully!")

    # Initial data profiling
    profile_data(df)

    # Cleaning
    cleaned_df = clean_transactions(df)

    run_business_validation(cleaned_df)

    # Generate Data Quality Report

    quality_report = generate_data_quality_report(
    cleaned_df
)

    save_data_quality_report(quality_report , QUALITY_REPORT_PATH)

    # Save cleaned dataset
    print("\nSaving cleaned dataset...")

    cleaned_df.to_parquet(
        PROCESSED_DATA_PATH,
        index=False
    )

    print("✓ Cleaned dataset saved successfully!")


if __name__ == "__main__":
    main()
