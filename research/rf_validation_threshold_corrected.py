import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

print("=" * 75)
print("CORRECTED RANDOM FOREST PHISHING THRESHOLD ANALYSIS")
print("=" * 75)

# ============================================================
# CONFIGURATION
# ============================================================

PHISHING_LABEL = 0
LEGITIMATE_LABEL = 1

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

THRESHOLDS = np.arange(0.05, 0.96, 0.01)

# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading training and validation data...")

train_df = pd.read_csv("train_model_domain.csv")
validation_df = pd.read_csv("validation_domain.csv")

print(f"Training data:   {train_df.shape}")
print(f"Validation data: {validation_df.shape}")

# ============================================================
# PREPARE FEATURES
# ============================================================

X_train = train_df[CORE_FEATURES]
y_train = train_df["label"]

X_validation = validation_df[CORE_FEATURES]
y_validation = validation_df["label"]

# ============================================================
# DATA QUALITY
# ============================================================

if X_train.isnull().sum().sum() > 0:
    raise ValueError("Missing values in training data.")

if X_validation.isnull().sum().sum() > 0:
    raise ValueError("Missing values in validation data.")

if np.isinf(X_train.values).sum() > 0:
    raise ValueError("Infinite values in training data.")

if np.isinf(X_validation.values).sum() > 0:
    raise ValueError("Infinite values in validation data.")

print("Data quality checks: PASS")

# ============================================================
# TRAIN MODEL
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

print("Training completed.")

# ============================================================
# VERIFY LABEL MAPPING
# ============================================================

print("\nModel classes:", model.classes_)

if not np.array_equal(
    model.classes_,
    np.array([PHISHING_LABEL, LEGITIMATE_LABEL])
):
    raise ValueError(
        f"Unexpected model classes: {model.classes_}"
    )

print("PASS: Class mapping verified.")
print(f"{PHISHING_LABEL} = PHISHING")
print(f"{LEGITIMATE_LABEL} = LEGITIMATE")

# ============================================================
# GENERATE PROBABILITIES
# ============================================================

print("\nGenerating validation probabilities...")

probabilities = model.predict_proba(X_validation)

phishing_index = np.where(
    model.classes_ == PHISHING_LABEL
)[0][0]

phishing_probabilities = probabilities[:, phishing_index]

print(
    f"Phishing probability column: {phishing_index}"
)

# ============================================================
# ROC-AUC
# ============================================================

roc_auc = roc_auc_score(
    (y_validation == PHISHING_LABEL).astype(int),
    phishing_probabilities
)

print(f"Validation ROC-AUC: {roc_auc:.6f}")

# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

results = []

print("\n" + "=" * 95)
print("CORRECTED VALIDATION THRESHOLD RESULTS")
print("=" * 95)

print(
    f"{'Threshold':>10} "
    f"{'Accuracy':>10} "
    f"{'Precision':>10} "
    f"{'Recall':>10} "
    f"{'F1':>10} "
    f"{'FP':>8} "
    f"{'FN':>8}"
)

print("-" * 95)

for threshold in THRESHOLDS:

    predictions = np.where(
        phishing_probabilities >= threshold,
        PHISHING_LABEL,
        LEGITIMATE_LABEL
    )

    accuracy = accuracy_score(
        y_validation,
        predictions
    )

    precision = precision_score(
        y_validation,
        predictions,
        pos_label=PHISHING_LABEL,
        zero_division=0
    )

    recall = recall_score(
        y_validation,
        predictions,
        pos_label=PHISHING_LABEL,
        zero_division=0
    )

    f1 = f1_score(
        y_validation,
        predictions,
        pos_label=PHISHING_LABEL,
        zero_division=0
    )

    cm = confusion_matrix(
        y_validation,
        predictions,
        labels=[LEGITIMATE_LABEL, PHISHING_LABEL]
    )

    tn = cm[0, 0]
    fp = cm[0, 1]
    fn = cm[1, 0]
    tp = cm[1, 1]

    results.append({
        "threshold": round(float(threshold), 2),
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
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    "rf_corrected_validation_threshold_results.csv",
    index=False
)

# ============================================================
# BEST F1
# ============================================================

best_f1 = results_df.loc[
    results_df["f1"].idxmax()
]

print("\n" + "=" * 75)
print("BEST THRESHOLD BY F1")
print("=" * 75)

print(f"Threshold:       {best_f1['threshold']:.2f}")
print(f"Accuracy:        {best_f1['accuracy']:.6f}")
print(f"Precision:       {best_f1['precision']:.6f}")
print(f"Recall:          {best_f1['recall']:.6f}")
print(f"F1:              {best_f1['f1']:.6f}")
print(f"False Positives: {int(best_f1['fp'])}")
print(f"False Negatives: {int(best_f1['fn'])}")
print(f"True Negatives:  {int(best_f1['tn'])}")
print(f"True Positives:  {int(best_f1['tp'])}")

# ============================================================
# BEST RECALL
# ============================================================

best_recall = results_df.loc[
    results_df["recall"].idxmax()
]

print("\n" + "=" * 75)
print("BEST THRESHOLD BY RECALL")
print("=" * 75)

print(f"Threshold:       {best_recall['threshold']:.2f}")
print(f"Precision:       {best_recall['precision']:.6f}")
print(f"Recall:          {best_recall['recall']:.6f}")
print(f"F1:              {best_recall['f1']:.6f}")
print(f"False Positives: {int(best_recall['fp'])}")
print(f"False Negatives: {int(best_recall['fn'])}")

# ============================================================
# BEST PRECISION
# ============================================================

best_precision = results_df.loc[
    results_df["precision"].idxmax()
]

print("\n" + "=" * 75)
print("BEST THRESHOLD BY PRECISION")
print("=" * 75)

print(f"Threshold:       {best_precision['threshold']:.2f}")
print(f"Precision:       {best_precision['precision']:.6f}")
print(f"Recall:          {best_precision['recall']:.6f}")
print(f"F1:              {best_precision['f1']:.6f}")
print(f"False Positives: {int(best_precision['fp'])}")
print(f"False Negatives: {int(best_precision['fn'])}")

print("\n" + "=" * 75)
print("ANALYSIS COMPLETE")
print("=" * 75)

print("\nCreated:")
print(" - rf_corrected_validation_threshold_results.csv")