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

print("=" * 80)
print("PHISHING URL DETECTION - RANDOM FOREST")
print("=" * 80)


# ============================================================
# 1. FILES
# ============================================================

TRAIN_FILE = "train_features.csv"
TEST_FILE = "test_features.csv"
TARGET = "label"


# ============================================================
# 2. URL-ONLY FEATURES
# ============================================================

URL_FEATURES = [
    "URLLength",
    "DomainLength",
    "IsDomainIP",
    "TLDLegitimateProb",
    "TLDLength",
    "NoOfSubDomain",
    "HasObfuscation",
    "NoOfObfuscatedChar",
    "ObfuscationRatio",
    "NoOfLettersInURL",
    "LetterRatioInURL",
    "NoOfDegitsInURL",
    "DegitRatioInURL",
    "NoOfEqualsInURL",
    "NoOfQMarkInURL",
    "NoOfAmpersandInURL",
    "NoOfOtherSpecialCharsInURL",
    "SpacialCharRatioInURL",
    "CharContinuationRate",
    "URLCharProb"
]


# ============================================================
# 3. LOAD DATA
# ============================================================

print("\nLoading datasets...")

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

print(f"Training dataset: {train_df.shape}")
print(f"Testing dataset:  {test_df.shape}")


# ============================================================
# 4. SEPARATE FEATURES AND TARGET
# ============================================================

y_train = train_df[TARGET]
y_test = test_df[TARGET]

X_train = train_df[URL_FEATURES].copy()
X_test = test_df[URL_FEATURES].copy()


# ============================================================
# 5. VALIDATE FEATURES
# ============================================================

print("\n" + "=" * 80)
print("FEATURE VALIDATION")
print("=" * 80)

missing_train = [
    feature
    for feature in URL_FEATURES
    if feature not in train_df.columns
]

missing_test = [
    feature
    for feature in URL_FEATURES
    if feature not in test_df.columns
]

if missing_train:
    raise ValueError(
        f"Missing training features: {missing_train}"
    )

if missing_test:
    raise ValueError(
        f"Missing testing features: {missing_test}"
    )

print(f"Features used: {len(URL_FEATURES)}")

print("\nTraining feature matrix:")
print(X_train.shape)

print("\nTesting feature matrix:")
print(X_test.shape)


# ============================================================
# 6. CHECK MISSING VALUES
# ============================================================

print("\n" + "=" * 80)
print("DATA VALIDATION")
print("=" * 80)

train_missing = X_train.isna().sum().sum()
test_missing = X_test.isna().sum().sum()

print(f"Training missing values: {train_missing}")
print(f"Testing missing values:  {test_missing}")

if train_missing > 0 or test_missing > 0:
    raise ValueError(
        "Missing values detected."
    )


# ============================================================
# 7. CHECK INFINITE VALUES
# ============================================================

train_inf = np.isinf(
    X_train.to_numpy()
).sum()

test_inf = np.isinf(
    X_test.to_numpy()
).sum()

print(f"Training infinite values: {train_inf}")
print(f"Testing infinite values:  {test_inf}")

if train_inf > 0 or test_inf > 0:
    raise ValueError(
        "Infinite values detected."
    )


# ============================================================
# 8. RANDOM FOREST
# ============================================================

print("\n" + "=" * 80)
print("MODEL CONFIGURATION")
print("=" * 80)

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

print("Algorithm: Random Forest")
print("Number of trees: 300")
print("Max depth: None")
print("Max features: sqrt")
print("Class weight: balanced")
print("Random state: 42")
print("CPU workers: all available")


# ============================================================
# 9. TRAIN
# ============================================================

print("\n" + "=" * 80)
print("MODEL TRAINING")
print("=" * 80)

print("Training Random Forest...")

model.fit(
    X_train,
    y_train
)

print("Training complete.")


# ============================================================
# 10. PREDICTIONS
# ============================================================

print("\n" + "=" * 80)
print("GENERATING PREDICTIONS")
print("=" * 80)

y_pred = model.predict(
    X_test
)

y_probability = model.predict_proba(
    X_test
)[:, 1]

print("Predictions generated.")


# ============================================================
# 11. METRICS
# ============================================================

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

roc_auc = roc_auc_score(
    y_test,
    y_probability
)

cm = confusion_matrix(
    y_test,
    y_pred
)


# ============================================================
# 12. PERFORMANCE
# ============================================================

print("\n" + "=" * 80)
print("RANDOM FOREST PERFORMANCE")
print("=" * 80)

print(f"Accuracy:  {accuracy:.6f}")
print(f"Precision: {precision:.6f}")
print(f"Recall:    {recall:.6f}")
print(f"F1-score:  {f1:.6f}")
print(f"ROC-AUC:   {roc_auc:.6f}")


# ============================================================
# 13. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 80)
print("CLASSIFICATION REPORT")
print("=" * 80)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Legitimate",
            "Phishing"
        ],
        digits=6,
        zero_division=0
    )
)


# ============================================================
# 14. CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 80)
print("CONFUSION MATRIX")
print("=" * 80)

print(cm)

print("\nMatrix format:")
print("[[True Negative, False Positive]")
print(" [False Negative, True Positive]]")

print(
    f"\nTrue Negatives : {cm[0, 0]:,}"
)

print(
    f"False Positives: {cm[0, 1]:,}"
)

print(
    f"False Negatives: {cm[1, 0]:,}"
)

print(
    f"True Positives : {cm[1, 1]:,}"
)


# ============================================================
# 15. FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 80)
print("RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 80)

importance_df = pd.DataFrame({
    "feature": URL_FEATURES,
    "importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    "importance",
    ascending=False
)

print(
    importance_df.to_string(
        index=False
    )
)


# ============================================================
# 16. SAVE FEATURE IMPORTANCE
# ============================================================

importance_df.to_csv(
    "random_forest_feature_importance.csv",
    index=False
)


# ============================================================
# 17. SAVE MODEL
# ============================================================

model_file = "random_forest_url_model.pkl"

joblib.dump(
    model,
    model_file
)

print("\n" + "=" * 80)
print("MODEL SAVED")
print("=" * 80)

print(f"Created: {model_file}")
print("Created: random_forest_feature_importance.csv")


# ============================================================
# 18. SAVE RESULTS
# ============================================================

results = pd.DataFrame([
    {
        "Model": "Random Forest URL-Only",
        "Features": len(URL_FEATURES),
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC-AUC": roc_auc,
        "True Negative": cm[0, 0],
        "False Positive": cm[0, 1],
        "False Negative": cm[1, 0],
        "True Positive": cm[1, 1]
    }
])

results.to_csv(
    "random_forest_results.csv",
    index=False
)

print("Created: random_forest_results.csv")


# ============================================================
# 19. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("RANDOM FOREST COMPLETE")
print("=" * 80)

print(f"Features:  {len(URL_FEATURES)}")
print(f"Accuracy:  {accuracy:.6f}")
print(f"Precision: {precision:.6f}")
print(f"Recall:    {recall:.6f}")
print(f"F1-score:  {f1:.6f}")
print(f"ROC-AUC:   {roc_auc:.6f}")

print("\nThe model was evaluated on the same")
print("domain-aware test set used by Logistic Regression.")

print("\nNext step:")
print("Compare Random Forest against URL-only Logistic Regression.")