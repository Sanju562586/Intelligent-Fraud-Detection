import joblib
import numpy as np
from sklearn.metrics import precision_recall_curve

def load_trained_model(model_path):
    return joblib.load(model_path)

def get_optimal_threshold(model, X_test, y_test):
    y_probs = model.predict_proba(X_test)[:,1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
    f1_scores = (2 * precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1])
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx]
