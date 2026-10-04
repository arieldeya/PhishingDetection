import pandas as pd
import numpy as np
import joblib

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
print("PHISHING URL DETECTION - LOGISTIC REGRESSION CORE FEATURES")
print("=" * 80)

TRAIN_FILE = "train_features.csv"
TEST_FILE = "test_features.csv"
TARGET = "label"

# EXACTLY the same 12 features used by the Core Random Forest
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

print("\nLoading datasets...")

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

print(f"Training dataset: {train_df.shape}")
print(f"Testing dataset:  {test_df.shape}")

y_train = train_df[TARGET]
y_test = test_df[TARGET]

X_train = train_df[CORE_FEATURES].copy()
X_test = test_df[CORE_FEATURES].copy()

print("\n" + "=" * 80)
print("FEATURE VALIDATION")
print("=" * 80)

print(f"Number of features: {len(CORE_FEATURES)}")

for i, feature in enumerate(CORE_FEATURES, start=1):
    print(f"{i:2}. {feature}")

print("\nTraining matrix:")
print(X_train.shape)

print("\nTesting matrix:")
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
print("FEATURE SCALING")
print("=" * 80)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("StandardScaler fitted on training data.")
print("Training features scaled.")
print("Testing features scaled.")

print("\n" + "=" * 80)
print("MODEL CONFIGURATION")
print("=" * 80)

model = LogisticRegression(
    max_iter=5000,
    solver="liblinear",
    random_state=42
)

print("Algorithm: Logistic Regression")
print("Solver: liblinear")
print("Maximum iterations: 5000")
print("Random state: 42")

print("\n" + "=" * 80)
print("MODEL TRAINING")
print("=" * 80)

print("Training Logistic Regression...")

model.fit(
    X_train_scaled,
    y_train
)

print("Training complete.")

print("\n" + "=" * 80)
print("GENERATING PREDICTIONS")
print("=" * 80)

y_pred = model.predict(X_test_scaled)

y_probability = model.predict_proba(
    X_test_scaled
)[:, 1]

print("Predictions generated.")

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

print("\n" + "=" * 80)
print("LOGISTIC REGRESSION CORE FEATURE PERFORMANCE")
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
print("MODEL COEFFICIENTS")
print("=" * 80)

coefficient_df = pd.DataFrame({
    "feature": CORE_FEATURES,
    "coefficient": model.coef_[0]
})

coefficient_df["absolute_coefficient"] = (
    coefficient_df["coefficient"].abs()
)

coefficient_df = coefficient_df.sort_values(
    "absolute_coefficient",
    ascending=False
)

print(
    coefficient_df.to_string(index=False)
)

coefficient_df.to_csv(
    "logistic_core_coefficients.csv",
    index=False
)

print("\nCreated: logistic_core_coefficients.csv")

print("\n" + "=" * 80)
print("SAVING MODEL")
print("=" * 80)

model_file = "logistic_core_url_model.pkl"
scaler_file = "logistic_core_url_scaler.pkl"

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

results = pd.DataFrame([
    {
        "Model": "Logistic Regression Core URL Features",
        "Features": len(CORE_FEATURES),
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
    "logistic_core_results.csv",
    index=False
)

print("Created: logistic_core_results.csv")

print("\n" + "=" * 80)
print("EXPERIMENT COMPLETE")
print("=" * 80)

print(f"Features:  {len(CORE_FEATURES)}")
print(f"Accuracy:  {accuracy:.6f}")
print(f"Precision: {precision:.6f}")
print(f"Recall:    {recall:.6f}")
print(f"F1-score:  {f1:.6f}")
print(f"ROC-AUC:   {roc_auc:.6f}")

print("\nEvaluation uses the same")
print("domain-aware test set as the Core Random Forest.")

print("\nNext step:")
print("Compare Logistic Regression and Random Forest")
print("using exactly the same 12 core URL features.")