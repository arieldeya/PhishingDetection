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
print("PHISHING URL DETECTION - RANDOM FOREST WITHOUT TLD PROBABILITY")
print("=" * 80)

TRAIN_FILE = "train_features.csv"
TEST_FILE = "test_features.csv"
TARGET = "label"

URL_FEATURES = [
    "URLLength",
    "DomainLength",
    "IsDomainIP",
    # "TLDLegitimateProb",  # intentionally removed
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

print("\nLoading datasets...")

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

print(f"Training dataset: {train_df.shape}")
print(f"Testing dataset:  {test_df.shape}")

y_train = train_df[TARGET]
y_test = test_df[TARGET]

X_train = train_df[URL_FEATURES].copy()
X_test = test_df[URL_FEATURES].copy()

print("\n" + "=" * 80)
print("FEATURE VALIDATION")
print("=" * 80)

print(f"Features used: {len(URL_FEATURES)}")

print("\nFeatures:")
for i, feature in enumerate(URL_FEATURES, start=1):
    print(f"{i:2}. {feature}")

print("\nTraining feature matrix:")
print(X_train.shape)

print("\nTesting feature matrix:")
print(X_test.shape)

print("\n" + "=" * 80)
print("DATA VALIDATION")
print("=" * 80)

train_missing = X_train.isna().sum().sum()
test_missing = X_test.isna().sum().sum()

print(f"Training missing values: {train_missing}")
print(f"Testing missing values:  {test_missing}")

if train_missing > 0 or test_missing > 0:
    raise ValueError("Missing values detected.")

train_inf = np.isinf(X_train.to_numpy()).sum()
test_inf = np.isinf(X_test.to_numpy()).sum()

print(f"Training infinite values: {train_inf}")
print(f"Testing infinite values:  {test_inf}")

if train_inf > 0 or test_inf > 0:
    raise ValueError("Infinite values detected.")

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

print("\n" + "=" * 80)
print("MODEL TRAINING")
print("=" * 80)

print("Training Random Forest...")

model.fit(X_train, y_train)

print("Training complete.")

print("\n" + "=" * 80)
print("GENERATING PREDICTIONS")
print("=" * 80)

y_pred = model.predict(X_test)
y_probability = model.predict_proba(X_test)[:, 1]

print("Predictions generated.")

accuracy = accuracy_score(y_test, y_pred)
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

print("\n" + "=" * 80)
print("RANDOM FOREST WITHOUT TLD PROBABILITY")
print("=" * 80)

print(f"Accuracy:  {accuracy:.6f}")
print(f"Precision: {precision:.6f}")
print(f"Recall:    {recall:.6f}")
print(f"F1-score:  {f1:.6f}")
print(f"ROC-AUC:   {roc_auc:.6f}")

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

print("\n" + "=" * 80)
print("CONFUSION MATRIX")
print("=" * 80)

print(cm)

print("\nMatrix format:")
print("[[True Negative, False Positive]")
print(" [False Negative, True Positive]]")

print(f"\nTrue Negatives : {cm[0, 0]:,}")
print(f"False Positives: {cm[0, 1]:,}")
print(f"False Negatives: {cm[1, 0]:,}")
print(f"True Positives : {cm[1, 1]:,}")

print("\n" + "=" * 80)
print("FEATURE IMPORTANCE")
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
    importance_df.to_string(index=False)
)

importance_df.to_csv(
    "rf_without_tld_feature_importance.csv",
    index=False
)

print("\nCreated:")
print("rf_without_tld_feature_importance.csv")

results = pd.DataFrame([
    {
        "Model": "Random Forest URL-Only Without TLD Probability",
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
    "rf_without_tld_results.csv",
    index=False
)

print("Created:")
print("rf_without_tld_results.csv")

model_file = "random_forest_url_without_tld_model.pkl"

joblib.dump(
    model,
    model_file
)

print(f"Created: {model_file}")

print("\n" + "=" * 80)
print("EXPERIMENT COMPLETE")
print("=" * 80)

print(f"Features:  {len(URL_FEATURES)}")
print(f"Accuracy:  {accuracy:.6f}")
print(f"Precision: {precision:.6f}")
print(f"Recall:    {recall:.6f}")
print(f"F1-score:  {f1:.6f}")
print(f"ROC-AUC:   {roc_auc:.6f}")

print("\nThis model uses the same")
print("domain-aware test set as the previous models.")

print("\nNext step:")
print("Compare this result with the previous")
print("Random Forest and Logistic Regression models.")