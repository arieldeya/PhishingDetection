import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

print("=" * 90)
print("PHISHING URL DETECTION - RANDOM FOREST THRESHOLD ANALYSIS")
print("=" * 90)

TRAIN_FILE = "train_features.csv"
TEST_FILE = "test_features.csv"

MODEL_FILE = "random_forest_core_url_model.pkl"

TARGET = "label"

CORE_FEATURES = [
    "URLLength",
    "DomainLength",
    "TLDLength",
    "NoOfSubDomain",
    "NoOfLettersInURL",
    "LetterRatioInURL",
    "NoOfDegitsInURL",
    "DegitRatioInURL",
    "NoOfOtherSpecialCharsInURL",
    "SpacialCharRatioInURL",
    "CharContinuationRate",
    "URLCharProb"
]

print("\nLoading test dataset...")

test_df = pd.read_csv(TEST_FILE)

X_test = test_df[CORE_FEATURES].copy()
y_test = test_df[TARGET]

print(f"Testing dataset: {test_df.shape}")
print(f"Features used: {len(CORE_FEATURES)}")

print("\nLoading Random Forest model...")

model = joblib.load(MODEL_FILE)

print("Model loaded successfully.")

print("\nGenerating probability predictions...")

probabilities = model.predict_proba(X_test)[:, 1]

print("Probability predictions generated.")

thresholds = [
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90
]

results = []

print("\n" + "=" * 90)
print("THRESHOLD RESULTS")
print("=" * 90)

print(
    f"{'Threshold':<12}"
    f"{'Accuracy':<12}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'F1':<12}"
    f"{'FP':<10}"
    f"{'FN':<10}"
)

print("-" * 90)

for threshold in thresholds:

    y_pred = (
        probabilities >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    tn, fp, fn, tp = cm.ravel()

    results.append({
        "Threshold": threshold,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "True_Negative": tn,
        "False_Positive": fp,
        "False_Negative": fn,
        "True_Positive": tp
    })

    print(
        f"{threshold:<12.2f}"
        f"{accuracy:<12.6f}"
        f"{precision:<12.6f}"
        f"{recall:<12.6f}"
        f"{f1:<12.6f}"
        f"{fp:<10,}"
        f"{fn:<10,}"
    )

results_df = pd.DataFrame(results)

results_df.to_csv(
    "rf_threshold_results.csv",
    index=False
)

print("\nCreated:")
print("rf_threshold_results.csv")

print("\n" + "=" * 90)
print("BEST F1 THRESHOLD")
print("=" * 90)

best_f1 = results_df.loc[
    results_df["F1"].idxmax()
]

print(
    best_f1.to_string()
)

print("\n" + "=" * 90)
print("HIGHEST RECALL THRESHOLD")
print("=" * 90)

highest_recall = results_df.loc[
    results_df["Recall"].idxmax()
]

print(
    highest_recall.to_string()
)

print("\n" + "=" * 90)
print("HIGHEST PRECISION THRESHOLD")
print("=" * 90)

highest_precision = results_df.loc[
    results_df["Precision"].idxmax()
]

print(
    highest_precision.to_string()
)

print("\n" + "=" * 90)
print("ANALYSIS COMPLETE")
print("=" * 90)

print("\nThe threshold controls the trade-off between:")
print("- False positives")
print("- False negatives")
print("- Precision")
print("- Recall")

print("\nThe default threshold is 0.50.")
print("No final threshold is selected automatically.")