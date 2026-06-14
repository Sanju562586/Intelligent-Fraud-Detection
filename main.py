import pandas as pd
import numpy as np
import joblib
import time
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve

# Load data and prepare test set as in the notebook
df = pd.read_csv("creditcard.csv")
X = df.drop("Class", axis=1)
y = df["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load the saved model
model = joblib.load("Outputs/best_rf_model2.pkl")

# Calculate optimal threshold based on notebook logic
y_probs = model.predict_proba(X_test)[:,1]
precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
f1_scores = (2 * precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1])
best_idx = np.argmax(f1_scores)
BEST_THRESHOLD = thresholds[best_idx]

print("Best Threshold:", BEST_THRESHOLD)

# Setup dataframes for simulation
test_df = X_test.copy()
test_df["Class"] = y_test
normal_df = test_df[test_df["Class"] == 0]
fraud_df = test_df[test_df["Class"] == 1]

transactions_per_second = 20
counter = 0
output_file = "predictions_output.csv"

# Write header for output file
result_header = pd.DataFrame(columns=["timestamp", "actual_class", "fraud_probability", "risk_score", "prediction"])
result_header.to_csv(output_file, index=False)

print("Starting simulation...")
while True:
    start_time = time.time()
    for _ in range(transactions_per_second):
        # 10 normal + 1 fraud
        if counter % 11 == 10:
            row = fraud_df.sample(1)
        else:
            row = normal_df.sample(1)
            
        counter += 1
        actual_class = int(row["Class"].values[0])
        X_transaction = row.drop("Class", axis=1)
        
        fraud_prob = model.predict_proba(X_transaction)[0][1]
        prediction = int(fraud_prob >= BEST_THRESHOLD)
        risk_score = round(fraud_prob * 100, 2)
        
        print(
            f"Actual={actual_class} | "
            f"Probability={fraud_prob:.4f} | "
            f"Risk={risk_score:.2f}% | "
            f"Prediction={prediction}"
        )
        
        result = pd.DataFrame({
            "timestamp": [datetime.now()],
            "actual_class": [actual_class],
            "fraud_probability": [fraud_prob],
            "risk_score": [risk_score],
            "prediction": [prediction]
        })
        
        result.to_csv(
            output_file,
            mode="a",
            header=False,
            index=False
        )
        
    elapsed = time.time() - start_time
    if elapsed < 1:
        time.sleep(1 - elapsed)
