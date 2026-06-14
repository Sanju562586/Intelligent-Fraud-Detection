import time
import pandas as pd
from datetime import datetime
import os
from data import load_data, get_simulation_data
from model import load_trained_model, get_optimal_threshold

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "creditcard.csv")
    model_path = os.path.join(base_dir, "Outputs", "best_rf_model2.pkl")
    
    # Load data
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_data(data_path)
    
    # Load model and threshold
    print("Loading model and calculating threshold...")
    model = load_trained_model(model_path)
    best_threshold = get_optimal_threshold(model, X_test, y_test)
    print(f"Optimal Threshold: {best_threshold}")
    
    # Get simulation data
    normal_df, fraud_df = get_simulation_data(X_test, y_test)
    
    transactions_per_second = 20
    counter = 0
    output_file = os.path.join(base_dir, "predictions_output.csv")
    
    # Write header for output file
    result_header = pd.DataFrame(columns=["timestamp", "actual_class", "fraud_probability", "risk_score", "prediction"])
    result_header.to_csv(output_file, index=False)
    
    print("Starting simulation...")
    while True:
        start_time = time.time()
        for _ in range(transactions_per_second):
            if counter % 11 == 10:
                row = fraud_df.sample(1)
            else:
                row = normal_df.sample(1)
                
            counter += 1
            actual_class = int(row["Class"].values[0])
            X_transaction = row.drop("Class", axis=1)
            
            fraud_prob = model.predict_proba(X_transaction)[0][1]
            prediction = int(fraud_prob >= best_threshold)
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

if __name__ == "__main__":
    main()
