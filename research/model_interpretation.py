import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    f1_score
)

print("=" * 70)
print("PHISHING DETECTION MODEL INTERPRETATION")
print("=" * 70)

# ============================================================
# 1. Configuration
# ============================================================

MODEL_FILE = "final_random_forest_phishing_model.pkl"
TEST_FILE = "test_domain.csv"

THRESHOLD = 0.35

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

OUTPUT_DIR = "model_analysis"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 2. Load model
# ============================================================

print("\nLoading final model...")

model = joblib.load(MODEL_FILE)

print("Model loaded successfully.")

# ============================================================
# 3. Load test data
# ============================================================

print("\nLoading test dataset...")

test_df = pd.read_csv(TEST_FILE)

print(f"Test dataset shape: {test_df.shape}")

# ============================================================
# 4. Prepare test features
# ============================================================

X_test = test_df[CORE_FEATURES]
y_test = test_df["label"]

print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

# ============================================================
# 5. Generate probabilities
# ============================================================

print("\nGenerating prediction probabilities...")

probabilities = model.predict_proba(X_test)[:, 1]

predictions = (
    probabilities >= THRESHOLD
).astype(int)

print("Predictions generated.")

# ============================================================
# 6. FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

importance = model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": CORE_FEATURES,
    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

feature_importance["Rank"] = range(
    1,
    len(feature_importance) + 1
)

feature_importance = feature_importance[
    ["Rank", "Feature", "Importance"]
]

print("\n")
print(feature_importance.to_string(index=False))

# Save feature importance

feature_importance.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "final_feature_importance.csv"
    ),
    index=False
)

# ============================================================
# 7. FEATURE IMPORTANCE CHART
# ============================================================

print("\nCreating feature importance chart...")

plot_df = feature_importance.sort_values(
    by="Importance",
    ascending=True
)

plt.figure(figsize=(10, 7))

plt.barh(
    plot_df["Feature"],
    plot_df["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Random Forest Feature Importance")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "feature_importance.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved: model_analysis/feature_importance.png")

# ============================================================
# 8. CONFUSION MATRIX
# ============================================================

print("\nCreating confusion matrix...")

cm = confusion_matrix(
    y_test,
    predictions
)

print("\nConfusion matrix:")
print(cm)

fig, ax = plt.subplots(figsize=(7, 6))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Legitimate",
        "Phishing"
    ]
)

disp.plot(
    ax=ax,
    values_format="d"
)

plt.title(
    f"Confusion Matrix - Threshold {THRESHOLD}"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "confusion_matrix.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved: model_analysis/confusion_matrix.png")

# ============================================================
# 9. ROC CURVE
# ============================================================

print("\nCreating ROC curve...")

fpr, tpr, roc_thresholds = roc_curve(
    y_test,
    probabilities
)

roc_auc = auc(
    fpr,
    tpr
)

plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    label=f"Random Forest (AUC = {roc_auc:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")

plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "roc_curve.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved: model_analysis/roc_curve.png")

# ============================================================
# 10. PRECISION-RECALL CURVE
# ============================================================

print("\nCreating Precision-Recall curve...")

precision, recall, pr_thresholds = precision_recall_curve(
    y_test,
    probabilities
)

average_precision = average_precision_score(
    y_test,
    probabilities
)

plt.figure(figsize=(8, 6))

plt.plot(
    recall,
    precision,
    label=f"Random Forest (AP = {average_precision:.4f})"
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")

plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "precision_recall_curve.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Saved: model_analysis/precision_recall_curve.png"
)

# ============================================================
# 11. THRESHOLD ANALYSIS
# ============================================================

print("\nCreating threshold/F1 analysis...")

threshold_values = np.arange(
    0.05,
    0.96,
    0.01
)

threshold_results = []

for threshold in threshold_values:

    threshold_predictions = (
        probabilities >= threshold
    ).astype(int)

    f1 = f1_score(
        y_test,
        threshold_predictions,
        zero_division=0
    )

    threshold_results.append({
        "threshold": threshold,
        "f1": f1
    })

threshold_df = pd.DataFrame(
    threshold_results
)

best_row = threshold_df.loc[
    threshold_df["f1"].idxmax()
]

print(
    f"\nHighest F1 on test set across this diagnostic curve: "
    f"{best_row['f1']:.6f}"
)

print(
    f"Corresponding threshold: "
    f"{best_row['threshold']:.2f}"
)

threshold_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "test_threshold_diagnostic.csv"
    ),
    index=False
)

plt.figure(figsize=(9, 6))

plt.plot(
    threshold_df["threshold"],
    threshold_df["f1"]
)

plt.axvline(
    THRESHOLD,
    linestyle="--",
    label=f"Locked threshold = {THRESHOLD}"
)

plt.xlabel("Classification Threshold")
plt.ylabel("F1 Score")
plt.title("F1 Score Across Classification Thresholds")

plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "threshold_f1_curve.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved: model_analysis/threshold_f1_curve.png")

# ============================================================
# 12. Save prediction results
# ============================================================

print("\nSaving test predictions...")

prediction_output = pd.DataFrame({
    "URL": test_df["URL"],
    "Actual_Label": y_test,
    "Phishing_Probability": probabilities,
    "Predicted_Label": predictions
})

prediction_output["Prediction"] = prediction_output[
    "Predicted_Label"
].map({
    0: "Legitimate",
    1: "Phishing"
})

prediction_output.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "test_predictions.csv"
    ),
    index=False
)

print(
    "Saved: model_analysis/test_predictions.csv"
)

# ============================================================
# 13. Summary
# ============================================================

print("\n" + "=" * 70)
print("MODEL INTERPRETATION COMPLETED")
print("=" * 70)

print("\nGenerated files:")

for filename in sorted(
    os.listdir(OUTPUT_DIR)
):

    print(
        f" - {OUTPUT_DIR}/{filename}"
    )

print("\nDone.")