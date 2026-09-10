from api.predictor import predict_transaction


transaction = {
    "step": 378,
    "type": "CASH_OUT",
    "amount": 60422.12,
    "oldbalanceOrg": 100000,
    "oldbalanceDest": 50000,
    "is_high_value": 0,
    "is_zero_amount": 0,
    "is_transfer": 0,
    "is_cash_out": 1,
    "is_payment": 0,
    "is_cash_in": 0,
    "is_debit": 0,
    "hour_of_day": 18,
    "day": 15,
}


result = predict_transaction(
    transaction
)


print("\n" + "=" * 70)
print("FINSIGHT REAL-TIME ML PREDICTION")
print("=" * 70)

print(
    f"Fraud probability: "
    f"{result['fraud_probability']:.6f}"
)

print(
    f"Risk level: "
    f"{result['risk_level']}"
)

print(
    f"Decision: "
    f"{result['decision']}"
)

print(
    f"Reason: "
    f"{result['decision_reason']}"
)

print("=" * 70)