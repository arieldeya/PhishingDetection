import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from sklearn.model_selection import GroupShuffleSplit

from feature_extractor_v2 import extract_features, FEATURE_NAMES


DATASET = "PhiUSIIL_Phishing_URL_Dataset.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.20


def extract_feature_matrix(urls):
    """
    Extract the 10 deployment-consistent V2 features
    from every URL.
    """

    return pd.DataFrame(
        [
            extract_features(url)
            for url in urls
        ],
        columns=FEATURE_NAMES
    )


print("=" * 100)
print("PHISHGUARD V2 - DOMAIN-GROUPED VALIDATION")
print("=" * 100)
print()


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("1. LOADING DATASET")
print("-" * 100)

df = pd.read_csv(DATASET)

print(f"Dataset rows: {len(df):,}")
print()


# ============================================================
# 2. PREPARE FEATURES
# ============================================================

print("2. EXTRACTING V2 FEATURES")
print("-" * 100)

X = extract_feature_matrix(df["URL"])

y = df["label"]


print(f"Feature matrix shape: {X.shape}")
print(f"Number of features:   {len(FEATURE_NAMES)}")
print()


# ============================================================
# 3. CREATE DOMAIN GROUPS
# ============================================================

print("3. PREPARING DOMAIN GROUPS")
print("-" * 100)

groups = df["Domain"].astype(str).str.lower().str.strip()

print(f"Unique domains: {groups.nunique():,}")
print()


# ============================================================
# 4. DOMAIN-GROUPED SPLIT
# ============================================================

print("4. CREATING DOMAIN-GROUPED TRAIN/TEST SPLIT")
print("-" * 100)

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

train_indices, test_indices = next(
    splitter.split(
        X,
        y,
        groups=groups
    )
)

X_train = X.iloc[train_indices]
X_test = X.iloc[test_indices]

y_train = y.iloc[train_indices]
y_test = y.iloc[test_indices]

train_domains = set(
    groups.iloc[train_indices]
)

test_domains = set(
    groups.iloc[test_indices]
)

overlap = train_domains.intersection(
    test_domains
)


print(f"Training records:     {len(X_train):,}")
print(f"Testing records:      {len(X_test):,}")
print(f"Training domains:     {len(train_domains):,}")
print(f"Testing domains:      {len(test_domains):,}")
print(f"Domain overlap:       {len(overlap)}")
print()


if len(overlap) == 0:
    print("PASS: No domain appears in both training and testing sets.")
else:
    print("WARNING: Domain overlap detected.")

print()


# ============================================================
# 5. LABEL DISTRIBUTION
# ============================================================

print("5. LABEL DISTRIBUTION")
print("-" * 100)

print("Training labels:")
print(y_train.value_counts())
print()

print("Testing labels:")
print(y_test.value_counts())
print()


# ============================================================
# 6. TRAIN RANDOM FOREST
# ============================================================

print("6. TRAINING RANDOM FOREST")
print("-" * 100)

model = RandomForestClassifier(
    n_estimators=300,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)

print("Model training complete.")
print()


# ============================================================
# 7. PREDICTION
# ============================================================

print("7. TESTING MODEL")
print("-" * 100)

y_pred = model.predict(X_test)


# ============================================================
# 8. METRICS
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


print(f"Accuracy:   {accuracy:.6f}")
print(f"Precision:  {precision:.6f}")
print(f"Recall:     {recall:.6f}")
print(f"F1 Score:   {f1:.6f}")
print()


# ============================================================
# 9. PERCENTAGE METRICS
# ============================================================

print("8. PERCENTAGE METRICS")
print("-" * 100)

print(f"Accuracy:   {accuracy * 100:.2f}%")
print(f"Precision:  {precision * 100:.2f}%")
print(f"Recall:     {recall * 100:.2f}%")
print(f"F1 Score:   {f1 * 100:.2f}%")
print()


# ============================================================
# 10. CONFUSION MATRIX
# ============================================================

print("9. CONFUSION MATRIX")
print("-" * 100)

cm = confusion_matrix(
    y_test,
    y_pred
)

print(cm)
print()


# ============================================================
# 11. CLASSIFICATION REPORT
# ============================================================

print("10. CLASSIFICATION REPORT")
print("-" * 100)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Phishing (0)",
            "Legitimate (1)"
        ],
        zero_division=0
    )
)


# ============================================================
# 12. FEATURE IMPORTANCE
# ============================================================

print("11. FEATURE IMPORTANCE")
print("-" * 100)

importance = pd.DataFrame({
    "Feature": FEATURE_NAMES,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

for _, row in importance.iterrows():

    print(
        f"{row['Feature']:35s}"
        f"{row['Importance']:.6f}"
    )

print()


# ============================================================
# 13. COMPARISON WITH RANDOM SPLIT
# ============================================================

print("=" * 100)
print("VALIDATION INTERPRETATION")
print("=" * 100)
print()

print(
    "Previous random-split V2 accuracy: 98.86%"
)

print(
    f"Domain-grouped accuracy: "
    f"{accuracy * 100:.2f}%"
)

print()

difference = 98.8613 - (accuracy * 100)

print(
    f"Accuracy difference: "
    f"{difference:.2f} percentage points"
)

print()

if accuracy >= 0.95:

    print(
        "RESULT: Strong performance on unseen domains."
    )

elif accuracy >= 0.85:

    print(
        "RESULT: Good performance, but domain generalization "
        "is weaker than the random split."
    )

elif accuracy >= 0.70:

    print(
        "RESULT: Moderate domain generalization. "
        "Further model improvement may be required."
    )

else:

    print(
        "RESULT: Weak domain generalization. "
        "The random-split accuracy may substantially overestimate "
        "real-world performance."
    )

print()

print("=" * 100)
print("DOMAIN-GROUPED VALIDATION COMPLETE")
print("=" * 100)