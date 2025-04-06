"""
    This script is used to train and evaluate three different machine learning models 
    (Decision Tree, Random Forest, and Support Vector Machine) on an accident dataset.
"""
import pandas as pd  
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

# Configure log file (UTF-8 to avoid encoding issues)
logging.basicConfig(filename="log.txt", level=logging.INFO, encoding="utf-8", format="%(asctime)s - %(message)s")

# Read Excel data
file_path = "new-gpt.xlsx"  
df = pd.read_excel(file_path)

# Handling accident factors: Split the `factor` column into individual feature columns
factors = [f"X{i}" for i in range(1, 46)]  # X1 to X45
def process_factors(row):
    present_factors = row.split(", ")
    return {factor: 1 if factor in present_factors else 0 for factor in factors}

factor_matrix = df["factor"].apply(process_factors).apply(pd.Series)

# Combine the processed feature matrix with the label
data = pd.concat([factor_matrix, df["label"]], axis=1)

# Split into training and testing sets
X = data[factors]  # Features
y = data["label"]  # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42, stratify=y)

# Calculate metrics: overall calculation based on TP/TN/FP/FN
def calc_metrics_from_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    TN, FP = cm[0]
    FN, TP = cm[1]

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return accuracy, precision, recall, f1, TP, TN, FP, FN

# Model training and evaluation (only report overall metrics)
def train_and_evaluate(model, model_name, threshold=0.5):
    model.fit(X_train, y_train)

    # Get prediction probabilities
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = model.decision_function(X_test)
        y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min())

    y_pred = (y_proba >= threshold).astype(int)

    # Calculate metrics
    accuracy, precision, recall, f1, TP, TN, FP, FN = calc_metrics_from_confusion(y_test, y_pred)

    # Log output
    log_info = f"""
        Model: {model_name} (Threshold: {threshold})
        ---------------------------------
        Confusion Matrix:
        - TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}

        Statistical Metrics (based on overall sample):
        - Accuracy: {accuracy:.4f}
        - Precision: {precision:.4f}
        - Recall: {recall:.4f}
        - F1 Score: {f1:.4f}
        ---------------------------------
        """
    print(log_info)
    logging.info(log_info)

# Run three classification models
train_and_evaluate(
    DecisionTreeClassifier(random_state=42, max_depth=10, min_samples_leaf=5, min_samples_split=5),
    "Decision Tree", threshold=0.5)

train_and_evaluate(
    RandomForestClassifier(class_weight="balanced", n_estimators=500, max_depth=10, random_state=42),
    "Random Forest", threshold=0.5)

train_and_evaluate(
    SVC(class_weight="balanced", probability=True, random_state=42),
    "Support Vector Machine", threshold=0.5)
