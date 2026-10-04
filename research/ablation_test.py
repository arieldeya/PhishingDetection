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
    confusion_matrix
)

print("=" * 80)
print("PHISHING URL DETECTION - FEATURE ABLATION EXPERIMENT")
print("=" * 80)

TRAIN_FILE = "train_features.csv"
TEST_FILE = "test_features.csv"
TARGET = "label"


# ============================================================
# 1. LOAD DATA
# ============================================================

print("\nLoading prepared datasets...")

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

print(f"Training dataset: {train_df.shape}")
print(f"Testing dataset:  {test_df.shape}")


# ============================================================
# 2. SEPARATE FEATURES AND TARGET
# ============================================================

X_train_full = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]

X_test_full = test_df.drop(columns=[TARGET])
y_test = test_df[TARGET]


# ============================================================
# 3. VERIFY FEATURE ALIGNMENT
# ============================================================

print("\n" + "=" * 80)
print("FEATURE ALIGNMENT")
print("=" * 80)

if list(X_train_full.columns) != list(X_test_full.columns):
    raise ValueError(
        "ERROR: Training and testing feature columns do not match."
    )

print("Training and testing feature columns match.")


# ============================================================
# 4. DEFINE EXPERIMENTS
# ============================================================

experiments = {

    "Baseline - All Prepared Features": [],

    "Without IsHTTPS": [
        "IsHTTPS"
    ],

    "Without Suspicious Binary Features": [
        "IsHTTPS",
        "IsDomainIP",
        "HasObfuscation"
    ],

    "URL Structure Focused": [
        "IsHTTPS",
        "IsDomainIP",
        "HasObfuscation",

        "HasTitle",
        "Title",
        "HasFavicon",
        "Robots",
        "IsResponsive",
        "LineOfCode",
        "LargestLineLength",
        "HasDescription",
        "NoOfPopup",
        "NoOfiFrame",
        "HasExternalFormSubmit",
        "HasSocialNet",
        "HasSubmitButton",
        "HasHiddenFields",
        "HasPasswordField",
        "Bank",
        "Pay",
        "Crypto",
        "HasCopyrightInfo",
        "NoOfImage",
        "NoOfCSS",
        "NoOfJS",
        "NoOfSelfRef",
        "NoOfEmptyRef",
        "NoOfExternalRef"
    ]
}


# ============================================================
# 5. FUNCTION TO RUN EXPERIMENT
# ============================================================

def run_experiment(name, columns_to_remove):

    print("\n")
    print("=" * 80)
    print(f"EXPERIMENT: {name}")
    print("=" * 80)

    # Check that requested columns exist
    missing_columns = [
        column
        for column in columns_to_remove
        if column not in X_train_full.columns
    ]

    if missing_columns:
        print("\nWARNING:")
        print("These columns were not found:")
        for column in missing_columns:
            print(f"- {column}")

    columns_to_remove_existing = [
        column
        for column in columns_to_remove
        if column in X_train_full.columns
    ]

    # Remove selected features
    X_train = X_train_full.drop(
        columns=columns_to_remove_existing
    ).copy()

    X_test = X_test_full.drop(
        columns=columns_to_remove_existing
    ).copy()

    print("\nRemoved features:")

    if columns_to_remove_existing:
        for column in columns_to_remove_existing:
            print(f"- {column}")
    else:
        print("- None")

    print(f"\nNumber of features: {X_train.shape[1]}")

    # ========================================================
    # CHECK DATA
    # ========================================================

    if X_train.isna().sum().sum() > 0:
        raise ValueError(
            "ERROR: Missing values found in training data."
        )

    if X_test.isna().sum().sum() > 0:
        raise ValueError(
            "ERROR: Missing values found in testing data."
        )

    train_inf = np.isinf(
        X_train.select_dtypes(include=np.number).to_numpy()
    ).sum()

    test_inf = np.isinf(
        X_test.select_dtypes(include=np.number).to_numpy()
    ).sum()

    if train_inf > 0 or test_inf > 0:
        raise ValueError(
            "ERROR: Infinite values found."
        )

    # ========================================================
    # SCALE FEATURES
    # ========================================================

    print("\nScaling features...")

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    # ========================================================
    # TRAIN MODEL
    # ========================================================

    print("Training Logistic Regression...")

    model = LogisticRegression(
        max_iter=2000,
        solver="liblinear",
        random_state=42
    )

    model.fit(
        X_train_scaled,
        y_train
    )

    # ========================================================
    # PREDICTIONS
    # ========================================================

    print("Generating predictions...")

    y_pred = model.predict(X_test_scaled)

    y_probability = model.predict_proba(
        X_test_scaled
    )[:, 1]

    # ========================================================
    # METRICS
    # ========================================================

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

    print("\n" + "-" * 80)
    print("RESULTS")
    print("-" * 80)

    print(f"Accuracy:  {accuracy:.6f}")
    print(f"Precision: {precision:.6f}")
    print(f"Recall:    {recall:.6f}")
    print(f"F1-score:  {f1:.6f}")
    print(f"ROC-AUC:   {roc_auc:.6f}")

    print("\nConfusion Matrix:")
    print(cm)

    return {
        "Experiment": name,
        "Features": X_train.shape[1],
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


# ============================================================
# 6. RUN ALL EXPERIMENTS
# ============================================================

results = []

for experiment_name, columns in experiments.items():

    result = run_experiment(
        experiment_name,
        columns
    )

    results.append(result)


# ============================================================
# 7. CREATE RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(results)


# ============================================================
# 8. DISPLAY COMPARISON
# ============================================================

print("\n\n")
print("=" * 80)
print("ABLATION EXPERIMENT RESULTS")
print("=" * 80)

display_columns = [
    "Experiment",
    "Features",
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "ROC-AUC"
]

print(
    results_df[display_columns].to_string(
        index=False
    )
)


# ============================================================
# 9. DISPLAY CONFUSION MATRICES
# ============================================================

print("\n")
print("=" * 80)
print("CONFUSION MATRIX COMPARISON")
print("=" * 80)

for _, row in results_df.iterrows():

    print(f"\n{row['Experiment']}")

    print(
        "[[TN, FP],"
    )

    print(
        f" [{int(row['False Negative'])}, "
        f"{int(row['True Positive'])}]]"
    )

    print(
        f"TN={int(row['True Negative'])}, "
        f"FP={int(row['False Positive'])}, "
        f"FN={int(row['False Negative'])}, "
        f"TP={int(row['True Positive'])}"
    )


# ============================================================
# 10. SAVE RESULTS
# ============================================================

output_file = "ablation_results.csv"

results_df.to_csv(
    output_file,
    index=False
)

print("\n")
print("=" * 80)
print("RESULTS SAVED")
print("=" * 80)

print(f"Created: {output_file}")


# ============================================================
# 11. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 80)
print("ABLATION EXPERIMENT COMPLETE")
print("=" * 80)

print(
    "\nThe experiments use the same domain-aware "
    "training and testing datasets."
)

print(
    "\nCompare the results to determine how strongly "
    "the model depends on suspicious or website-content features."
)

print("\nNext step:")
print("Review ablation_results.csv before training additional models.")