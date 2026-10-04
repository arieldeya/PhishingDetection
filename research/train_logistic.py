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

print("=" * 70)
print("PHISHING URL DETECTION - LOGISTIC REGRESSION")
print("=" * 70)

# ------------------------------------------------------------
# 1. Load prepared datasets
# ------------------------------------------------------------

TRAIN_FILE = "train_features.csv"
TEST_FILE = "test_features.csv"

print("\nLoading datasets...")

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

print(f"Training dataset: {train_df.shape}")
print(f"Testing dataset:  {test_df.shape}")

# ------------------------------------------------------------
# 2. Separate features and target
# ------------------------------------------------------------

TARGET = "label"

X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]

X_test = test_df.drop(columns=[TARGET])
y_test = test_df[TARGET]

print("\nFeature matrix:")
print(f"X_train: {X_train.shape}")
print(f"X_test:  {X_test.shape}")

print("\nTarget distribution:")
print("Training:")
print(y_train.value_counts())

print("\nTesting:")
print(y_test.value_counts())

# ------------------------------------------------------------
# 3. Verify feature alignment
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FEATURE ALIGNMENT CHECK")
print("=" * 70)

if list(X_train.columns) != list(X_test.columns):
    raise ValueError(
        "ERROR: Training and testing feature columns do not match."
    )

print("Training and testing feature columns match.")

# ------------------------------------------------------------
# 4. Check missing values
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DATA VALIDATION")
print("=" * 70)

train_missing = X_train.isna().sum().sum()
test_missing = X_test.isna().sum().sum()

print(f"Training missing values: {train_missing}")
print(f"Testing missing values:  {test_missing}")

# ------------------------------------------------------------
# 5. Standardize numerical features
# ------------------------------------------------------------
#
# IMPORTANT:
# The scaler is fitted ONLY on training data.
# The testing data is transformed using the training scaler.
#
# This prevents data leakage.
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FEATURE SCALING")
print("=" * 70)

scaler = StandardScaler()

print("Fitting StandardScaler on training data...")

X_train_scaled = scaler.fit_transform(X_train)

print("Transforming testing data...")

X_test_scaled = scaler.transform(X_test)

print("Scaling complete.")

# ------------------------------------------------------------
# 6. Train Logistic Regression
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("MODEL TRAINING")
print("=" * 70)

model = LogisticRegression(
    max_iter=2000,
    solver="liblinear",
    random_state=42
)

print("Training Logistic Regression...")

model.fit(
    X_train_scaled,
    y_train
)

print("Model training complete.")

# ------------------------------------------------------------
# 7. Predictions
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("GENERATING PREDICTIONS")
print("=" * 70)

y_pred = model.predict(X_test_scaled)

y_probability = model.predict_proba(
    X_test_scaled
)[:, 1]

print("Predictions generated.")

# ------------------------------------------------------------
# 8. Calculate metrics
# ------------------------------------------------------------

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

# ------------------------------------------------------------
# 9. Display results
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")

# ------------------------------------------------------------
# 10. Classification report
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Class 0",
            "Class 1"
        ],
        digits=4,
        zero_division=0
    )
)

# ------------------------------------------------------------
# 11. Confusion matrix
# ------------------------------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(cm)

print("\nMatrix format:")
print("[[True Negative, False Positive]")
print(" [False Negative, True Positive]]")

# ------------------------------------------------------------
# 12. Feature coefficients
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("TOP MODEL FEATURES")
print("=" * 70)

coefficients = pd.DataFrame({
    "feature": X_train.columns,
    "coefficient": model.coef_[0]
})

coefficients["absolute_coefficient"] = (
    coefficients["coefficient"].abs()
)

top_features = coefficients.sort_values(
    "absolute_coefficient",
    ascending=False
).head(20)

print(top_features.to_string(index=False))

# ------------------------------------------------------------
# 13. Save model
# ------------------------------------------------------------

import joblib

print("\n" + "=" * 70)
print("SAVING MODEL")
print("=" * 70)

joblib.dump(
    model,
    "logistic_phishing_model.pkl"
)

joblib.dump(
    scaler,
    "logistic_scaler.pkl"
)

print("Model saved:")
print("logistic_phishing_model.pkl")

print("Scaler saved:")
print("logistic_scaler.pkl")

# ------------------------------------------------------------
# 14. Final summary
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("LOGISTIC REGRESSION COMPLETE")
print("=" * 70)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")

print("\nModel training and evaluation completed successfully.")