"""
    This script is used to train and evaluate three different machine learning models
    (Decision Tree, Random Forest, and Support Vector Machine) on an accident dataset.
    It now includes functionality to run the training and evaluation multiple times
    with random subsampling of the data and reports the average metrics.
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
file_path = "new-ds.xlsx"
df = pd.read_excel(file_path)

# Handling accident factors: Split the `factor` column into individual feature columns
factors = [f"X{i}" for i in range(1, 46)]  # X1 to X45
def process_factors(row):
    present_factors = row.split(", ")
    return {factor: 1 if factor in present_factors else 0 for factor in factors}

factor_matrix = df["factor"].apply(process_factors).apply(pd.Series)

# Combine the processed feature matrix with the label
data = pd.concat([factor_matrix, df["label"]], axis=1)

# Define the number of runs and the proportion of data to sample
num_runs = 10  # Number of times to run the experiment
sample_proportion = 0.5  # Proportion of the data to randomly sample in each run

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

# Model training and evaluation with random sampling
def train_and_evaluate_with_sampling(model, model_name, n_runs, sample_size_proportion, threshold=0.5):
    all_accuracy = []
    all_precision = []
    all_recall = []
    all_f1 = []
    all_tp = []
    all_tn = []
    all_fp = []
    all_fn = []

    for i in range(n_runs):
        print(f"\n--- Running {model_name} - Run {i+1} ---")
        logging.info(f"--- Running {model_name} - Run {i+1} ---")

        # Randomly sample the data
        sampled_data = data.sample(frac=sample_size_proportion, random_state=39 + i)

        # Split into training and testing sets
        X_sampled = sampled_data[factors]  # Features
        y_sampled = sampled_data["label"]  # Target variable
        X_train_sampled, X_test_sampled, y_train_sampled, y_test_sampled = train_test_split(
            X_sampled, y_sampled, test_size=0.4, random_state=39 + i, stratify=y_sampled
        )

        model.fit(X_train_sampled, y_train_sampled)

        # Get prediction probabilities
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test_sampled)[:, 1]
        else:
            y_proba = model.decision_function(X_test_sampled)
            y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min())

        y_pred = (y_proba >= threshold).astype(int)

        # Calculate metrics for this run
        accuracy, precision, recall, f1, TP, TN, FP, FN = calc_metrics_from_confusion(y_test_sampled, y_pred)

        # Append metrics for averaging
        all_accuracy.append(accuracy)
        all_precision.append(precision)
        all_recall.append(recall)
        all_f1.append(f1)
        all_tp.append(TP)
        all_tn.append(TN)
        all_fp.append(FP)
        all_fn.append(FN)

        # Log output for this run
        log_info = f"""
            Model: {model_name} (Run {i+1}, Threshold: {threshold})
            ---------------------------------
            Confusion Matrix:
            - TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}

            Statistical Metrics (based on sampled test set):
            - Accuracy: {accuracy:.4f}
            - Precision: {precision:.4f}
            - Recall: {recall:.4f}
            - F1 Score: {f1:.4f}
            ---------------------------------
            """
        print(log_info)
        logging.info(log_info)

    # Calculate and report average metrics
    avg_accuracy = np.mean(all_accuracy)
    avg_precision = np.mean(all_precision)
    avg_recall = np.mean(all_recall)
    avg_f1 = np.mean(all_f1)
    avg_tp = np.mean(all_tp)
    avg_tn = np.mean(all_tn)
    avg_fp = np.mean(all_fp)
    avg_fn = np.mean(all_fn)

    avg_log_info = f"""
    --- {model_name} - Average Metrics over {n_runs} runs (Sample Proportion: {sample_size_proportion:.2f}) ---
    ----------------------------------------------------------------------------------------------------
    Average Confusion Matrix:
    - TP: {avg_tp:.2f}, TN: {avg_tn:.2f}, FP: {avg_fp:.2f}, FN: {avg_fn:.2f}

    Average Statistical Metrics:
    - Accuracy: {avg_accuracy:.4f}
    - Precision: {avg_precision:.4f}
    - Recall: {avg_recall:.4f}
    - F1 Score: {avg_f1:.4f}
    ----------------------------------------------------------------------------------------------------
    """
    print(avg_log_info)
    logging.info(avg_log_info)

# Run three classification models with multiple runs and random sampling
train_and_evaluate_with_sampling(
    DecisionTreeClassifier(random_state=39, max_depth=10, min_samples_leaf=5, min_samples_split=5),
    "Decision Tree", num_runs, sample_proportion, threshold=0.5)

train_and_evaluate_with_sampling(
    RandomForestClassifier(class_weight="balanced", n_estimators=500, max_depth=10, random_state=39),
    "Random Forest", num_runs, sample_proportion, threshold=0.5)

train_and_evaluate_with_sampling(
    SVC(class_weight="balanced", probability=True, random_state=39),
    "Support Vector Machine", num_runs, sample_proportion, threshold=0.5)