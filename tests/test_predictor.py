from api.predictor import predict_transaction


def test_predict_legitimate_transaction():

    transaction = {
        "step": 378,
        "type": "PAYMENT",
        "amount": 1000.0,
        "oldbalanceOrg": 50000.0,
        "oldbalanceDest": 100000.0,
    }

    result = predict_transaction(transaction)

    assert "fraud_probability" in result
    assert "risk_level" in result
    assert "decision" in result

    assert result["risk_level"] in [
        "LOW",
        "MEDIUM",
        "HIGH",
    ]

    assert result["decision"] in [
        "ALLOW",
        "REVIEW",
        "BLOCK",
    ]