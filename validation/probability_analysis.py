import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score

from feature_extractor_v2 import extract_features, FEATURE_NAMES


DATASET = "PhiUSIIL_Phishing_URL_Dataset.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.20


print("=" * 100)
print("PHISHGUARD V2 - PROBABILITY AND RISK ANALYSIS")
print("=" * 100)
print()


# ============================================================
# 1. LOAD DATA
# ============================================================

print("1. LOADING DATASET")
print("-" * 100)

df = pd.read_csv(DATASET)

print(f"Dataset rows: {len(df):,}")
print()


# ============================================================
# 2. EXTRACT FEATURES
# ============================================================

print("2. EXTRACTING FEATURES")
print("-" * 100)

X = pd.DataFrame(
    [
        extract_features(url)
        for url in df["URL"]
    ],
    columns=FEATURE_NAMES
)

y = df["label"]

print(f"Feature matrix: {X.shape}")
print()


# ============================================================
# 3. DOMAIN-GROUPED SPLIT
# ============================================================

print("3. DOMAIN-GROUPED SPLIT")
print("-" * 100)

groups = (
    df["Domain"]
    .astype(str)
    .str.lower()
    .str.strip()
)

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

print(f"Training records: {len(X_train):,}")
print(f"Testing records:  {len(X_test):,}")
print()


# ============================================================
# 4. TRAIN TEMPORARY MODEL
# ============================================================

print("4. TRAINING TEMPORARY RANDOM FOREST")
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

print("Training complete.")
print()


# ============================================================
# 5. GET PHISHING PROBABILITIES
# ============================================================

print("5. CALCULATING PROBABILITIES")
print("-" * 100)

probabilities = model.predict_proba(X_test)

classes = list(model.classes_)

phishing_index = classes.index(0)
legitimate_index = classes.index(1)

phishing_probability = probabilities[:, phishing_index]

legitimate_probability = probabilities[:, legitimate_index]

print("Probability extraction complete.")
print()


# ============================================================
# 6. ROC-AUC
# ============================================================

print("6. ROC-AUC")
print("-" * 100)

# Convert labels so:
# phishing = 1
# legitimate = 0

phishing_target = (
    y_test == 0
).astype(int)

auc = roc_auc_score(
    phishing_target,
    phishing_probability
)

print(f"ROC-AUC: {auc:.6f}")
print(f"ROC-AUC: {auc * 100:.2f}%")
print()


# ============================================================
# 7. PROBABILITY DISTRIBUTION
# ============================================================

print("7. PROBABILITY DISTRIBUTION")
print("-" * 100)

phishing_probs = phishing_probability[
    y_test.to_numpy() == 0
]

legitimate_probs = phishing_probability[
    y_test.to_numpy() == 1
]


print("ACTUAL PHISHING URLs")
print("-" * 50)

print(
    f"Minimum phishing probability: "
    f"{phishing_probs.min() * 100:.2f}%"
)

print(
    f"25th percentile:              "
    f"{pd.Series(phishing_probs).quantile(0.25) * 100:.2f}%"
)

print(
    f"Median:                       "
    f"{pd.Series(phishing_probs).median() * 100:.2f}%"
)

print(
    f"75th percentile:              "
    f"{pd.Series(phishing_probs).quantile(0.75) * 100:.2f}%"
)

print(
    f"Maximum:                      "
    f"{phishing_probs.max() * 100:.2f}%"
)

print()


print("ACTUAL LEGITIMATE URLs")
print("-" * 50)

print(
    f"Minimum phishing probability: "
    f"{legitimate_probs.min() * 100:.2f}%"
)

print(
    f"25th percentile:              "
    f"{pd.Series(legitimate_probs).quantile(0.25) * 100:.2f}%"
)

print(
    f"Median:                       "
    f"{pd.Series(legitimate_probs).median() * 100:.2f}%"
)

print(
    f"75th percentile:              "
    f"{pd.Series(legitimate_probs).quantile(0.75) * 100:.2f}%"
)

print(
    f"Maximum:                      "
    f"{legitimate_probs.max() * 100:.2f}%"
)

print()


# ============================================================
# 8. RISK THRESHOLD ANALYSIS
# ============================================================

print("8. RISK THRESHOLD ANALYSIS")
print("-" * 100)

thresholds = [
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90
]


print(
    f"{'Threshold':<15}"
    f"{'Phishing Detected':<20}"
    f"{'Legitimate Flagged':<20}"
)

print("-" * 55)


for threshold in thresholds:

    phishing_detected = (
        phishing_probs >= threshold
    ).sum()

    legitimate_flagged = (
        legitimate_probs >= threshold
    ).sum()

    phishing_total = len(phishing_probs)
    legitimate_total = len(legitimate_probs)

    phishing_rate = (
        phishing_detected /
        phishing_total *
        100
    )

    legitimate_rate = (
        legitimate_flagged /
        legitimate_total *
        100
    )

    print(
        f"{threshold * 100:>6.0f}%          "
        f"{phishing_rate:>7.2f}%"
        f" ({phishing_detected:,})"
        f"        "
        f"{legitimate_rate:>7.2f}%"
        f" ({legitimate_flagged:,})"
    )

print()


# ============================================================
# 9. CURRENT RISK LEVELS
# ============================================================

print("9. CURRENT PHISHGUARD RISK LEVELS")
print("-" * 100)

high_risk = (
    phishing_probability >= 0.70
)

medium_risk = (
    (phishing_probability >= 0.30) &
    (phishing_probability < 0.70)
)

low_risk = (
    phishing_probability < 0.30
)

print(
    f"HIGH RISK:   {high_risk.sum():,} "
    f"({high_risk.mean() * 100:.2f}%)"
)

print(
    f"MEDIUM RISK: {medium_risk.sum():,} "
    f"({medium_risk.mean() * 100:.2f}%)"
)

print(
    f"LOW RISK:    {low_risk.sum():,} "
    f"({low_risk.mean() * 100:.2f}%)"
)

print()


# ============================================================
# 10. HIGH-RISK PHISHING DETECTION
# ============================================================

print("10. HIGH-RISK DETECTION")
print("-" * 100)

high_risk_phishing = (
    phishing_probs >= 0.70
).sum()

total_phishing = len(phishing_probs)

high_risk_detection_rate = (
    high_risk_phishing /
    total_phishing *
    100
)

print(
    f"Actual phishing URLs:       "
    f"{total_phishing:,}"
)

print(
    f"Detected as HIGH RISK:      "
    f"{high_risk_phishing:,}"
)

print(
    f"HIGH-RISK detection rate:   "
    f"{high_risk_detection_rate:.2f}%"
)

print()


# ============================================================
# 11. LEGITIMATE FALSE HIGH-RISK
# ============================================================

print("11. LEGITIMATE URLs FLAGGED HIGH RISK")
print("-" * 100)

false_high_risk = (
    legitimate_probs >= 0.70
).sum()

total_legitimate = len(legitimate_probs)

false_high_risk_rate = (
    false_high_risk /
    total_legitimate *
    100
)

print(
    f"Actual legitimate URLs:     "
    f"{total_legitimate:,}"
)

print(
    f"Flagged HIGH RISK:          "
    f"{false_high_risk:,}"
)

print(
    f"False HIGH-RISK rate:       "
    f"{false_high_risk_rate:.2f}%"
)

print()


# ============================================================
# 12. FINAL INTERPRETATION
# ============================================================

print("=" * 100)
print("PROBABILITY ANALYSIS COMPLETE")
print("=" * 100)
print()

print(
    "IMPORTANT:"
)

print(
    "This analysis evaluates probability separation and "
    "risk thresholds on unseen domains."
)

print(
    "It does not modify the production V2 model."
)

print(
    "The current production model remains unchanged."
)

print()

print("=" * 100)