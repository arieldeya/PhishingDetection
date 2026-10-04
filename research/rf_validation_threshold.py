import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

print("=" * 70)
print("RANDOM FOREST VALIDATION THRESHOLD ANALYSIS")
print("=" * 70)

# ============================================================
# 1. Load datasets
# ============================================================

print("\nLoading datasets...")

train_df = pd.read_csv("train_model_domain.csv")
validation_df = pd.read_csv("validation_domain.csv")

print(f"Training shape:    {train_df.shape}")
print(f"Validation shape:  {validation_df.shape}")

# ============================================================
# 2. Define the 12 core URL features
# ============================================================

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

print("\nCore features:")
for i, feature in enumerate(CORE_FEATURES, start=1):
    print(f"{i:2}. {feature}")

# ============================================================
# 3. Check that all features exist
# ============================================================

for feature in CORE_FEATURES:
    if feature not in train_df.columns:
        raise ValueError(
            f"Feature '{feature}' is missing from training data."
        )

    if feature not in validation_df.columns:
        raise ValueError(
            f"Feature '{feature}' is missing from validation data."
        )

# ============================================================
# 4. Prepare X and y
# ============================================================

X_train = train_df[CORE_FEATURES]
y_train = train_df["label"]

X_validation = validation_df[CORE_FEATURES]
y_validation = validation_df["label"]

print("\nFeature matrices:")
print(f"X_train:       {X_train.shape}")
print(f"y_train:       {y_train.shape}")
print(f"X_validation:  {X_validation.shape}")
print(f"y_validation:  {y_validation.shape}")

# ============================================================
# 5. Check missing/infinite values
# ============================================================

if X_train.isnull().sum().sum() > 0:
    raise ValueError("Missing values detected in training features.")

if X_validation.isnull().sum().sum() > 0:
    raise ValueError("Missing values detected in validation features.")

if np.isinf(X_train.values).sum() > 0:
    raise ValueError("Infinite values detected in training features.")

if np.isinf(X_validation.values).sum() > 0:
    raise ValueError("Infinite values detected in validation features.")

print("\nData quality checks: PASS")

# ============================================================
# 6. Train Random Forest
# ============================================================

print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Random Forest training completed.")

# ============================================================
# 7. Get validation probabilities
# ============================================================

print("\nGenerating validation probabilities...")

validation_probabilities = model.predict_proba(
    X_validation
)[:, 1]

# ============================================================
# 8. Evaluate thresholds
# ============================================================

thresholds = [
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90
]

results = []

print("\n" + "=" * 70)
print("VALIDATION THRESHOLD RESULTS")
print("=" * 70)

print(
    f"{'Threshold':>10} "
    f"{'Accuracy':>10} "
    f"{'Precision':>10} "
    f"{'Recall':>10} "
    f"{'F1':>10} "
    f"{'FP':>8} "
    f"{'FN':>8}"
)

print("-" * 70)

for threshold in thresholds:

    predictions = (
        validation_probabilities >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_validation,
        predictions
    )

    precision = precision_score(
        y_validation,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_validation,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_validation,
        predictions,
        zero_division=0
    )

    cm = confusion_matrix(
        y_validation,
        predictions
    )

    tn, fp, fn, tp = cm.ravel()

    results.append({
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp
    })

    print(
        f"{threshold:10.2f} "
        f"{accuracy:10.6f} "
        f"{precision:10.6f} "
        f"{recall:10.6f} "
        f"{f1:10.6f} "
        f"{fp:8d} "
        f"{fn:8d}"
    )

# ============================================================
# 9. Save results
# ============================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    "rf_validation_threshold_results.csv",
    index=False
)

# ============================================================
# 10. Find best threshold by F1
# ============================================================

best_row = results_df.loc[
    results_df["f1"].idxmax()
]

print("\n" + "=" * 70)
print("BEST VALIDATION THRESHOLD BY F1")
print("=" * 70)

print(
    f"\nThreshold:  {best_row['threshold']:.2f}"
)

print(
    f"Accuracy:   {best_row['accuracy']:.6f}"
)

print(
    f"Precision:  {best_row['precision']:.6f}"
)

print(
    f"Recall:     {best_row['recall']:.6f}"
)

print(
    f"F1 Score:   {best_row['f1']:.6f}"
)

print(
    f"False Positives: {int(best_row['fp'])}"
)

print(
    f"False Negatives: {int(best_row['fn'])}"
)

print(
    f"True Negatives:  {int(best_row['tn'])}"
)

print(
    f"True Positives:  {int(best_row['tp'])}"
)

# ============================================================
# 11. Find threshold with highest recall
# ============================================================

highest_recall = results_df.loc[
    results_df["recall"].idxmax()
]

print("\n" + "=" * 70)
print("HIGHEST RECALL THRESHOLD")
print("=" * 70)

print(
    f"\nThreshold:  {highest_recall['threshold']:.2f}"
)

print(
    f"Recall:     {highest_recall['recall']:.6f}"
)

print(
    f"Precision:  {highest_recall['precision']:.6f}"
)

print(
    f"F1 Score:   {highest_recall['f1']:.6f}"
)

print(
    f"False Positives: {int(highest_recall['fp'])}"
)

print(
    f"False Negatives: {int(highest_recall['fn'])}"
)

# ============================================================
# 12. Find threshold with highest precision
# ============================================================

highest_precision = results_df.loc[
    results_df["precision"].idxmax()
]

print("\n" + "=" * 70)
print("HIGHEST PRECISION THRESHOLD")
print("=" * 70)

print(
    f"\nThreshold:  {highest_precision['threshold']:.2f}"
)

print(
    f"Precision:  {highest_precision['precision']:.6f}"
)

print(
    f"Recall:     {highest_precision['recall']:.6f}"
)

print(
    f"F1 Score:   {highest_precision['f1']:.6f}"
)

print(
    f"False Positives: {int(highest_precision['fp'])}"
)

print(
    f"False Negatives: {int(highest_precision['fn'])}"
)

# ============================================================
# 13. ROC-AUC
# ============================================================

roc_auc = roc_auc_score(
    y_validation,
    validation_probabilities
)

print("\n" + "=" * 70)
print("VALIDATION ROC-AUC")
print("=" * 70)

print(f"\nROC-AUC: {roc_auc:.6f}")

# ============================================================
# 14. Save validation model
# ============================================================

import joblib

joblib.dump(
    model,
    "random_forest_validation_model.pkl"
)

print("\nFiles created:")
print(" - rf_validation_threshold_results.csv")
print(" - random_forest_validation_model.pkl")

print("\nValidation threshold analysis completed.")