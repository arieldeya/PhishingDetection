import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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
print("PHISHING URL DETECTION - URL-ONLY LOGISTIC REGRESSION")
print("=" * 80)


# ============================================================
# 1. FILES
# ============================================================

TRAIN_FILE = "train_features.csv"
TEST_FILE = "test_features.csv"
TARGET = "label"


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\nLoading prepared datasets...")

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

print(f"Training dataset: {train_df.shape}")
print(f"Testing dataset:  {test_df.shape}")


# ============================================================
# 3. SEPARATE TARGET
# ============================================================

X_train_full = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]

X_test_full = test_df.drop(columns=[TARGET])
y_test = test_df[TARGET]


# ============================================================
# 4. DEFINE URL-ONLY FEATURES
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
# 5. VERIFY FEATURES
# ============================================================

print("\n" + "=" * 80)
print("URL FEATURE VALIDATION")
print("=" * 80)

missing_train = [
    feature
    for feature in URL_FEATURES
    if feature not in X_train_full.columns
]

missing_test = [
    feature
    for feature in URL_FEATURES
    if feature not in X_test_full.columns
]

if missing_train:
    raise ValueError(
        f"ERROR: Missing URL features in training data: {missing_train}"
    )

if missing_test:
    raise ValueError(
        f"ERROR: Missing URL features in testing data: {missing_test}"
    )

print(f"URL features selected: {len(URL_FEATURES)}")

print("\nSelected features:")

for feature in URL_FEATURES:
    print(f"- {feature}")


# ============================================================
# 6. CREATE URL-ONLY MATRICES
# ============================================================

X_train = X_train_full[URL_FEATURES].copy()
X_test = X_test_full[URL_FEATURES].copy()

print("\nURL-only training shape:")
print(X_train.shape)

print("\nURL-only testing shape:")
print(X_test.shape)


# ============================================================
# 7. DATA VALIDATION
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
        "ERROR: Missing values detected."
    )


train_inf = np.isinf(
    X_train.select_dtypes(include=np.number).to_numpy()
).sum()

test_inf = np.isinf(
    X_test.select_dtypes(include=np.number).to_numpy()
).sum()

print(f"Training infinite values: {train_inf}")
print(f"Testing infinite values:  {test_inf}")

if train_inf > 0 or test_inf > 0:
    raise ValueError(
        "ERROR: Infinite values detected."
    )


# ============================================================
# 8. SCALE
# ============================================================

print("\n" + "=" * 80)
print("FEATURE SCALING")
print("=" * 80)

scaler = StandardScaler()

print("Fitting scaler on training data...")

X_train_scaled = scaler.fit_transform(X_train)

print("Transforming testing data...")

X_test_scaled = scaler.transform(X_test)

print("Scaling complete.")


# ============================================================
# 9. TRAIN LOGISTIC REGRESSION
# ============================================================

print("\n" + "=" * 80)
print("MODEL TRAINING")
print("=" * 80)

model = LogisticRegression(
    max_iter=5000,
    solver="liblinear",
    random_state=42
)

print("Training Logistic Regression...")
print("Maximum iterations: 5000")

model.fit(
    X_train_scaled,
    y_train
)

print("Model training complete.")


# ============================================================
# 10. PREDICTIONS
# ============================================================

print("\n" + "=" * 80)
print("GENERATING PREDICTIONS")
print("=" * 80)

y_pred = model.predict(X_test_scaled)

y_probability = model.predict_proba(
    X_test_scaled
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
print("URL-ONLY MODEL PERFORMANCE")
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
# 15. MODEL COEFFICIENTS
# ============================================================

print("\n" + "=" * 80)
print("URL FEATURE COEFFICIENTS")
print("=" * 80)

coefficients = pd.DataFrame({
    "feature": URL_FEATURES,
    "coefficient": model.coef_[0]
})

coefficients["absolute_coefficient"] = (
    coefficients["coefficient"].abs()
)

coefficients = coefficients.sort_values(
    "absolute_coefficient",
    ascending=False
)

print(
    coefficients.to_string(
        index=False
    )
)


# ============================================================
# 16. SAVE RESULTS
# ============================================================

results = pd.DataFrame([
    {
        "Model": "URL-Only Logistic Regression",
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

results_file = "url_only_results.csv"

results.to_csv(
    results_file,
    index=False
)

print("\n" + "=" * 80)
print("RESULTS SAVED")
print("=" * 80)

print(f"Created: {results_file}")


# ============================================================
# 17. SAVE URL-ONLY MODEL
# ============================================================

import joblib

model_file = "url_only_logistic_model.pkl"
scaler_file = "url_only_scaler.pkl"

joblib.dump(
    model,
    model_file
)

joblib.dump(
    scaler,
    scaler_file
)

print(f"Created: {model_file}")
print(f"Created: {scaler_file}")


# ============================================================
# 18. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("URL-ONLY EXPERIMENT COMPLETE")
print("=" * 80)

print(f"Features used: {len(URL_FEATURES)}")
print(f"Accuracy:      {accuracy:.6f}")
print(f"Precision:     {precision:.6f}")
print(f"Recall:        {recall:.6f}")
print(f"F1-score:      {f1:.6f}")
print(f"ROC-AUC:       {roc_auc:.6f}")

print("\nImportant:")
print(
    "This model uses URL-derived structural features only "
    "and does not use webpage-content features."
)

print("\nNext step:")
print(
    "Compare these results with the full-feature model "
    "before moving to Random Forest."
)

print("\n" + "=" * 80)