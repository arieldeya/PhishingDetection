import pandas as pd
import numpy as np

print("=" * 80)
print("PHISHING URL DETECTION - URL FEATURE DISTRIBUTION AUDIT")
print("=" * 80)

TRAIN_FILE = "train_features.csv"
TEST_FILE = "test_features.csv"

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

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

print("\nTraining dataset:")
print(train_df.shape)

print("\nTesting dataset:")
print(test_df.shape)


# ============================================================
# 1. FEATURE RANGE ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("FEATURE RANGE ANALYSIS")
print("=" * 80)

stats = train_df[URL_FEATURES].describe().T

stats["missing"] = train_df[URL_FEATURES].isna().sum()

stats["infinite"] = np.isinf(
    train_df[URL_FEATURES].to_numpy()
).sum(axis=0)

print(
    stats[
        [
            "count",
            "mean",
            "std",
            "min",
            "25%",
            "50%",
            "75%",
            "max",
            "missing",
            "infinite"
        ]
    ].to_string()
)


# ============================================================
# 2. EXTREME VALUES
# ============================================================

print("\n" + "=" * 80)
print("EXTREME VALUE CHECK")
print("=" * 80)

for feature in URL_FEATURES:

    series = train_df[feature]

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = (
        (series < lower) |
        (series > upper)
    ).sum()

    print(
        f"{feature:30s} "
        f"outliers={outliers:,} "
        f"min={series.min():.4f} "
        f"max={series.max():.4f}"
    )


# ============================================================
# 3. TARGET-SPECIFIC DISTRIBUTIONS
# ============================================================

print("\n" + "=" * 80)
print("FEATURE DISTRIBUTIONS BY TARGET")
print("=" * 80)

for feature in URL_FEATURES:

    print(f"\n{feature}")

    grouped = train_df.groupby("label")[feature].agg(
        [
            "count",
            "mean",
            "std",
            "min",
            "median",
            "max"
        ]
    )

    print(grouped.to_string())


# ============================================================
# 4. TLD LEGITIMATE PROBABILITY
# ============================================================

print("\n" + "=" * 80)
print("TLD LEGITIMATE PROBABILITY AUDIT")
print("=" * 80)

print("\nOverall statistics:")

print(
    train_df["TLDLegitimateProb"].describe()
)


print("\nAverage value by target:")

print(
    train_df.groupby("label")[
        "TLDLegitimateProb"
    ].agg(
        [
            "count",
            "mean",
            "std",
            "min",
            "median",
            "max"
        ]
    ).to_string()
)


# ============================================================
# 5. CORRELATION WITH TARGET
# ============================================================

print("\n" + "=" * 80)
print("CORRELATION WITH TARGET")
print("=" * 80)

correlations = (
    train_df[URL_FEATURES + ["label"]]
    .corr(numeric_only=True)["label"]
    .drop("label")
    .sort_values(
        key=lambda x: x.abs(),
        ascending=False
    )
)

print(correlations.to_string())


# ============================================================
# 6. UNIQUE VALUES
# ============================================================

print("\n" + "=" * 80)
print("UNIQUE VALUE ANALYSIS")
print("=" * 80)

for feature in URL_FEATURES:

    unique_count = train_df[feature].nunique()

    print(
        f"{feature:30s} "
        f"unique values = {unique_count:,}"
    )


# ============================================================
# 7. SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("AUDIT COMPLETE")
print("=" * 80)

print("""
Review the following carefully:

1. Extremely large maximum values.
2. Large numbers of statistical outliers.
3. Features whose distributions differ dramatically by label.
4. TLDLegitimateProb construction and relationship with label.
5. Features with unusually high correlation with the target.
""")