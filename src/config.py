from pathlib import Path


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# Data directories
DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

PROCESSED_DATA_DIR = DATA_DIR / "processed"


# Report directory
REPORTS_DIR = PROJECT_ROOT / "reports"


# Dataset paths
RAW_DATA_PATH = (
    RAW_DATA_DIR /
    "transactions.csv"
)

PROCESSED_DATA_PATH = (
    PROCESSED_DATA_DIR /
    "transactions_cleaned.parquet"
)


# Report paths
QUALITY_REPORT_PATH = (
    REPORTS_DIR /
    "data_quality_report.csv"
)