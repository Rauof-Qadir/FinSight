import pandas as pd

from api.database import SessionLocal
from api.models import FraudPrediction


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/modeling/hybrid_predictions.parquet"

# Start with 50,000 transactions
SAMPLE_SIZE = 50000


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("FINSIGHT DATABASE DATA LOADER")
print("=" * 70)

print("\nLoading hybrid predictions...")

df = pd.read_parquet(DATA_PATH)

print(f"✓ Total rows available: {len(df):,}")

print("\nDataset columns:")
print(df.columns.tolist())


# ============================================================
# SAMPLE DATA
# ============================================================

if len(df) > SAMPLE_SIZE:

    df = df.sample(
        n=SAMPLE_SIZE,
        random_state=42
    )

print(f"\n✓ Selected transactions: {len(df):,}")


# ============================================================
# DATABASE CONNECTION
# ============================================================

db = SessionLocal()

try:

    print("\nClearing old predictions...")

    db.query(FraudPrediction).delete()

    db.commit()

    print("✓ Old data cleared")


    # ========================================================
    # INSERT DATA
    # ========================================================

    print("\nSaving predictions to PostgreSQL...")

    records = []

    for _, row in df.iterrows():

        record = FraudPrediction(

            step=int(row["step"]),

            type=str(row["type"]),

            amount=float(row["amount"]),

            oldbalance_org=float(
                row.get("oldbalanceOrg", 0)
            ),

            oldbalance_dest=float(
                row.get("oldbalanceDest", 0)
            ),

            fraud_probability=float(
                row["fraud_probability"]
            ),

            risk_level=str(
                row["risk_level"]
            ),

            decision=str(
                row["decision"]
            ),

            decision_reason=str(
                row.get(
                    "decision_reason",
                    "Hybrid ML risk decision"
                )
            ),
        )

        records.append(record)


    # ========================================================
    # BATCH INSERT
    # ========================================================

    BATCH_SIZE = 5000

    for i in range(0, len(records), BATCH_SIZE):

        batch = records[i:i + BATCH_SIZE]

        db.bulk_save_objects(batch)

        db.commit()

        print(
            f"✓ Inserted "
            f"{min(i + BATCH_SIZE, len(records)):,} "
            f"transactions"
        )


    print("\n" + "=" * 70)
    print("DATABASE LOADING COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print(f"\nTotal transactions loaded: {len(records):,}")


finally:

    db.close()

    print("\n✓ Database connection closed")