import pandas as pd
import numpy as np

print("=" * 70)
print("PHISHING DATASET FEATURE DEPENDENCY AUDIT")
print("=" * 70)

# ==========================================================
# LOAD DATA
# ==========================================================

DATASET_PATH = "PhiUSIIL_Phishing_URL_Dataset.csv"

df = pd.read_csv(DATASET_PATH)

TARGET = "label"

# ==========================================================
# FEATURES WE WANT TO INVESTIGATE
# ==========================================================

suspicious_features = [
    "URLSimilarityIndex",
    "HasSocialNet",
    "HasCopyrightInfo",
    "HasDescription",
    "IsHTTPS",
    "DomainTitleMatchScore",
    "URLTitleMatchScore",
    "HasTitle",
    "IsDomainIP",
    "HasObfuscation",
    "HasFavicon",
    "Robots",
    "IsResponsive",
    "HasSubmitButton",
    "HasHiddenFields",
    "HasExternalFormSubmit"
]

# ==========================================================
# 1. FEATURE CORRELATION MATRIX
# ==========================================================

print("\n" + "=" * 70)
print("CORRELATION BETWEEN SUSPICIOUS FEATURES")
print("=" * 70)

available = [
    feature
    for feature in suspicious_features
    if feature in df.columns
]

correlation_matrix = df[available].corr()

print(
    correlation_matrix.round(3).to_string()
)

# ==========================================================
# 2. HIGH FEATURE-TO-FEATURE CORRELATIONS
# ==========================================================

print("\n" + "=" * 70)
print("HIGH FEATURE-TO-FEATURE CORRELATIONS")
print("=" * 70)

pairs = []

for i in range(len(available)):

    for j in range(i + 1, len(available)):

        feature_a = available[i]
        feature_b = available[j]

        correlation = correlation_matrix.loc[
            feature_a,
            feature_b
        ]

        if abs(correlation) >= 0.70:

            pairs.append(
                (
                    feature_a,
                    feature_b,
                    correlation
                )
            )

if pairs:

    pairs.sort(
        key=lambda x: abs(x[2]),
        reverse=True
    )

    for feature_a, feature_b, correlation in pairs:

        print(
            f"{feature_a:30} "
            f"{feature_b:30} "
            f"{correlation:.4f}"
        )

else:

    print(
        "No feature pairs have correlation >= 0.70."
    )

# ==========================================================
# 3. SUSPICIOUS BINARY FEATURES
# ==========================================================

print("\n" + "=" * 70)
print("BINARY FEATURE COUNTS")
print("=" * 70)

binary_features = [
    feature
    for feature in available
    if df[feature].nunique() == 2
]

for feature in binary_features:

    print("\n--------------------------------------")
    print(feature)
    print("--------------------------------------")

    print(
        df[feature]
        .value_counts()
        .sort_index()
    )

# ==========================================================
# 4. LABEL DISTRIBUTION FOR SUSPICIOUS FEATURES
# ==========================================================

print("\n" + "=" * 70)
print("LABEL DISTRIBUTION BY SUSPICIOUS FEATURES")
print("=" * 70)

for feature in binary_features:

    print("\n--------------------------------------")
    print(feature)
    print("--------------------------------------")

    table = pd.crosstab(
        df[feature],
        df[TARGET],
        normalize="index"
    ) * 100

    print(
        table.round(2)
    )

# ==========================================================
# 5. URLSimilarityIndex ANALYSIS
# ==========================================================

print("\n" + "=" * 70)
print("URLSimilarityIndex ANALYSIS")
print("=" * 70)

print(
    df.groupby("label")["URLSimilarityIndex"]
    .agg(
        [
            "count",
            "mean",
            "std",
            "min",
            "median",
            "max"
        ]
    )
    .round(4)
)

# ==========================================================
# 6. URLSimilarityIndex BINS
# ==========================================================

print("\n" + "=" * 70)
print("URLSimilarityIndex BINS")
print("=" * 70)

bins = [
    0,
    10,
    20,
    30,
    40,
    50,
    60,
    70,
    80,
    90,
    100
]

df["URLSimilarityBin"] = pd.cut(
    df["URLSimilarityIndex"],
    bins=bins,
    include_lowest=True
)

similarity_stats = (
    df.groupby("URLSimilarityBin", observed=True)
    .agg(
        samples=("label", "size"),
        phishing_rate=("label", "mean")
    )
)

similarity_stats["phishing_rate"] *= 100

print(
    similarity_stats.round(2)
)

# ==========================================================
# 7. EXACT TARGET SEPARATION
# ==========================================================

print("\n" + "=" * 70)
print("FEATURES THAT PERFECTLY SEPARATE THE TARGET")
print("=" * 70)

for feature in binary_features:

    groups = (
        df.groupby(feature)[TARGET]
        .agg(["min", "max", "count"])
    )

    perfect = groups[
        (groups["min"] == groups["max"])
    ]

    if len(perfect) > 0:

        print(f"\n{feature}")

        print(perfect)

# ==========================================================
# 8. FINAL
# ==========================================================

print("\n" + "=" * 70)
print("FEATURE DEPENDENCY AUDIT COMPLETE")
print("=" * 70)