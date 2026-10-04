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
    confusion_matrix,
    classification_report
)

print("=" * 70)
print("FINAL PHISHING URL MODEL EVALUATION")
print("=" * 70)

# ============================================================
# 1. LABEL DEFINITIONS
# ============================================================

# Official PhiUSIIL convention:
# 0 = PHISHING
# 1 = LEGITIMATE

PHISHING_LABEL = 0
LEGITIMATE_LABEL = 1

# Locked phishing probability threshold
THRESHOLD = 0.58

print("\nLabel mapping:")
print("0 = PHISHING")
print("1 = LEGITIMATE")

print(f"\nLocked phishing probability threshold: {THRESHOLD}")

# ============================================================
# 2. CORE FEATURES
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

print("\nUsing 12 core URL features:")

for i, feature in enumerate(CORE_FEATURES):
    print(f"{i}: {feature}")

# ============================================================
# 3. LOAD ORIGINAL TRAINING DATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING TRAINING DATA")
print("=" * 70)

train_df = pd.read_csv("train_domain.csv")

print(f"Training data: {train_df.shape}")

# ============================================================
# 4. LOAD UNTOUCHED TEST DATA
# ============================================================

print("\nLoading final test data...")

test_df = pd.read_csv("test_domain.csv")

print(f"Test data: {test_df.shape}")

# ============================================================
# 5. VERIFY REQUIRED COLUMNS
# ============================================================

print("\nChecking required columns...")

required_columns = CORE_FEATURES + ["label", "Domain"]

missing_train = [
    col for col in required_columns
    if col not in train_df.columns
]

missing_test = [
    col for col in required_columns
    if col not in test_df.columns
]

if missing_train:
    raise ValueError(
        f"Missing columns in training data: {missing_train}"
    )

if missing_test:
    raise ValueError(
        f"Missing columns in test data: {missing_test}"
    )

print("PASS: All required columns are present.")

# ============================================================
# 6. VERIFY LABEL VALUES
# ============================================================

print("\nChecking labels...")

train_labels = sorted(train_df["label"].unique())
test_labels = sorted(test_df["label"].unique())

print(f"Training labels: {train_labels}")
print(f"Test labels: {test_labels}")

if train_labels != [0, 1]:
    raise ValueError(
        f"Unexpected training labels: {train_labels}"
    )

if test_labels != [0, 1]:
    raise ValueError(
        f"Unexpected test labels: {test_labels}"
    )

print("PASS: Binary labels confirmed.")

# ============================================================
# 7. VERIFY DOMAIN SEPARATION
# ============================================================

print("\n" + "=" * 70)
print("DOMAIN SEPARATION CHECK")
print("=" * 70)

train_domains = set(train_df["Domain"].astype(str))
test_domains = set(test_df["Domain"].astype(str))

domain_overlap = train_domains.intersection(test_domains)

print(f"Training domains: {len(train_domains)}")
print(f"Test domains: {len(test_domains)}")
print(f"Overlapping domains: {len(domain_overlap)}")

if len(domain_overlap) != 0:
    print("\nWARNING: Training and test domains overlap!")

    print("\nExample overlapping domains:")
    for domain in list(domain_overlap)[:10]:
        print(domain)

    raise ValueError(
        "ERROR: Training and test domains overlap!"
    )

print("PASS: Training and test domains are completely separate.")

# ============================================================
# 8. PREPARE FEATURES
# ============================================================

print("\n" + "=" * 70)
print("PREPARING FEATURES")
print("=" * 70)

X_train = train_df[CORE_FEATURES].copy()
y_train = train_df["label"].copy()

X_test = test_df[CORE_FEATURES].copy()
y_test = test_df["label"].copy()

print(f"X_train: {X_train.shape}")
print(f"y_train: {y_train.shape}")
print(f"X_test:  {X_test.shape}")
print(f"y_test:  {y_test.shape}")

# ============================================================
# 9. DATA QUALITY CHECKS
# ============================================================

print("\nRunning data quality checks...")

train_missing = X_train.isna().sum().sum()
test_missing = X_test.isna().sum().sum()

train_infinite = np.isinf(X_train.to_numpy()).sum()
test_infinite = np.isinf(X_test.to_numpy()).sum()

print(f"Training missing values: {train_missing}")
print(f"Test missing values: {test_missing}")
print(f"Training infinite values: {train_infinite}")
print(f"Test infinite values: {test_infinite}")

if train_missing > 0 or test_missing > 0:
    raise ValueError(
        "ERROR: Missing values detected."
    )

if train_infinite > 0 or test_infinite > 0:
    raise ValueError(
        "ERROR: Infinite values detected."
    )

print("PASS: No missing or infinite values.")

# ============================================================
# 10. CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)

print("\nTraining:")
print(
    y_train.value_counts()
    .sort_index()
    .rename(index={
        0: "PHISHING",
        1: "LEGITIMATE"
    })
)

print("\nTest:")
print(
    y_test.value_counts()
    .sort_index()
    .rename(index={
        0: "PHISHING",
        1: "LEGITIMATE"
    })
)

# ============================================================
# 11. TRAIN FINAL RANDOM FOREST
# ============================================================

print("\n" + "=" * 70)
print("TRAINING FINAL RANDOM FOREST")
print("=" * 70)

print(f"\nTraining on all {len(X_train):,} training rows...")

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Final model training completed.")

# ============================================================
# 12. VERIFY MODEL FEATURES
# ============================================================

print("\nVerifying trained model features...")

model_features = list(model.feature_names_in_)

if model_features != CORE_FEATURES:
    raise ValueError(
        "\nERROR: Model feature order does not match CORE_FEATURES!\n"
        f"Model features: {model_features}\n"
        f"Expected:       {CORE_FEATURES}"
    )

print("PASS: Model feature names and order are correct.")

# ============================================================
# 13. GENERATE PROBABILITIES
# ============================================================

print("\nGenerating predictions on untouched test set...")

probabilities = model.predict_proba(X_test)

print(f"Model classes: {list(model.classes_)}")

# Because:
# class 0 = PHISHING
# class 1 = LEGITIMATE
#
# Find the correct probability column dynamically.

phishing_index = list(model.classes_).index(PHISHING_LABEL)
legitimate_index = list(model.classes_).index(LEGITIMATE_LABEL)

phishing_probability = probabilities[:, phishing_index]
legitimate_probability = probabilities[:, legitimate_index]

# ============================================================
# 14. APPLY PHISHING THRESHOLD
# ============================================================

# Predict PHISHING when phishing probability >= 0.58
#
# Otherwise predict LEGITIMATE.

y_pred = np.where(
    phishing_probability >= THRESHOLD,
    PHISHING_LABEL,
    LEGITIMATE_LABEL
)

# ============================================================
# 15. CALCULATE METRICS
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

# IMPORTANT:
# phishing is class 0
precision = precision_score(
    y_test,
    y_pred,
    pos_label=PHISHING_LABEL,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    pos_label=PHISHING_LABEL,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    pos_label=PHISHING_LABEL,
    zero_division=0
)

# ROC-AUC must use:
# true phishing indicator + phishing probability

y_test_phishing = (
    y_test == PHISHING_LABEL
).astype(int)

roc_auc = roc_auc_score(
    y_test_phishing,
    phishing_probability
)

# ============================================================
# 16. FINAL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("FINAL TEST RESULTS")
print("=" * 70)

print(f"\nThreshold:  {THRESHOLD}")

print(f"\nAccuracy:   {accuracy:.6f}")
print(f"Precision:  {precision:.6f}")
print(f"Recall:     {recall:.6f}")
print(f"F1 Score:   {f1:.6f}")
print(f"ROC-AUC:    {roc_auc:.6f}")

print("\nPercentage form:")

print(f"Accuracy:   {accuracy * 100:.4f}%")
print(f"Precision:  {precision * 100:.4f}%")
print(f"Recall:     {recall * 100:.4f}%")
print(f"F1 Score:   {f1 * 100:.4f}%")
print(f"ROC-AUC:    {roc_auc * 100:.4f}%")

# ============================================================
# 17. CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

# We explicitly order classes as:
# Legitimate (1), Phishing (0)

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[
        LEGITIMATE_LABEL,
        PHISHING_LABEL
    ]
)

tn = cm[0, 0]
fp = cm[0, 1]
fn = cm[1, 0]
tp = cm[1, 1]

print("\n                Predicted")
print("              Legit  Phishing")
print(
    f"Actual Legit  {tn:6d}  {fp:8d}"
)
print(
    f"Actual Phish  {fn:6d}  {tp:8d}"
)

print("\nRaw confusion matrix:")
print(cm)

print("\nTrue Negatives (Legitimate correctly identified):", tn)
print("False Positives (Legitimate classified as phishing):", fp)
print("False Negatives (Phishing classified as legitimate):", fn)
print("True Positives (Phishing correctly identified):", tp)

# ============================================================
# 18. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

report = classification_report(
    y_test,
    y_pred,
    labels=[
        LEGITIMATE_LABEL,
        PHISHING_LABEL
    ],
    target_names=[
        "Legitimate",
        "Phishing"
    ],
    zero_division=0
)

print(report)

# ============================================================
# 19. SAVE FINAL MODEL
# ============================================================

model_filename = "final_random_forest_phishing_model.pkl"

joblib.dump(model, model_filename)

# ============================================================
# 20. SAVE RESULTS
# ============================================================

results_df = pd.DataFrame({
    "URL": test_df["URL"],
    "ActualLabel": y_test,
    "PhishingProbability": phishing_probability,
    "LegitimateProbability": legitimate_probability,
    "PredictedLabel": y_pred
})

results_df["ActualClass"] = results_df["ActualLabel"].map({
    0: "Phishing",
    1: "Legitimate"
})

results_df["PredictedClass"] = results_df["PredictedLabel"].map({
    0: "Phishing",
    1: "Legitimate"
})

results_df["Correct"] = (
    results_df["ActualLabel"]
    == results_df["PredictedLabel"]
)

results_filename = "final_random_forest_results.csv"

results_df.to_csv(
    results_filename,
    index=False
)

# ============================================================
# 21. SAVE METRICS SUMMARY
# ============================================================

metrics_df = pd.DataFrame({
    "Metric": [
        "Threshold",
        "Accuracy",
        "Precision_Phaling",
        "Recall_Phaling",
        "F1_Phaling",
        "ROC_AUC"
    ],
    "Value": [
        THRESHOLD,
        accuracy,
        precision,
        recall,
        f1,
        roc_auc
    ]
})

metrics_df.to_csv(
    "final_random_forest_metrics.csv",
    index=False
)

# ============================================================
# 22. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FILES CREATED")
print("=" * 70)

print(f" - {model_filename}")
print(f" - {results_filename}")
print(" - final_random_forest_metrics.csv")

print("\n" + "=" * 70)
print("FINAL EVALUATION COMPLETED")
print("=" * 70)