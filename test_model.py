import pandas as pd
import joblib

# Load model
model = joblib.load(
    r"C:\ProgFiles\IntelligentFraudDetection\Outputs\best_rf_model2.pkl"
)

# Load one sample
df = pd.read_csv(
    r"C:\ProgFiles\IntelligentFraudDetection\test_data.csv"
)

X = df.drop("Class", axis=1)

sample = X.iloc[[0]]

prob = model.predict_proba(sample)[0][1]
pred = model.predict(sample)[0]

print("Fraud Probability:", prob)
print("Prediction:", pred)
print("Risk Score:", round(prob * 100, 2))