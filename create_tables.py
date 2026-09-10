from api.database import engine, Base
from api.models import FraudPrediction


Base.metadata.create_all(bind=engine)

print("Database tables created successfully.")


