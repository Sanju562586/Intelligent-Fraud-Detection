import pandas as pd
import joblib
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from data import load_data

def train_model():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "creditcard.csv")
    model_dir = os.path.join(base_dir, "Outputs")
    os.makedirs(model_dir, exist_ok=True)
    
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_data(data_path)

    print("Scaling and resampling data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

    best_rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )

    print("Training Random Forest model...")
    best_rf.fit(X_train, y_train)

    model_path = os.path.join(model_dir, "best_rf_model2.pkl")
    joblib.dump(best_rf, model_path)
    print(f"Model saved to {model_path}")

    y_pred = best_rf.predict(X_test)

    print("=== Best RandomForest Classifier ===")
    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print("Precision: ", precision_score(y_test, y_pred))
    print("Recall: ", recall_score(y_test, y_pred))
    print("F-1 score: ", f1_score(y_test, y_pred))

if __name__ == "__main__":
    train_model()
