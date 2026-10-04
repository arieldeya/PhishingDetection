import pandas as pd
import numpy as np

print("=" * 70)
print("PHISHING URL FEATURE LEAKAGE ANALYSIS")
print("=" * 70)

# ==========================================================
# 1. LOAD DATA
# ==========================================================

DATASET_PATH = "PhiUSIIL_Phishing_URL_Dataset.csv"

df = pd.read_csv(DATASET_PATH)

TARGET = "label"

print("\nDataset:")
print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")

# ==========================================================
# 2. NUMERIC CORRELATION WITH TARGET
# ==========================================================

print("\n" + "=" * 70)
print("NUMERIC FEATURE CORRELATION WITH TARGET")
print("=" * 70)

numeric_columns = df.select_dtypes(
    include=["int64", "float64"]
).columns

correlations = (
    df[numeric_columns]
    .corr()[TARGET]
    .drop(TARGET)
    .sort_values(key=abs, ascending=False)
)

print("\nFeatures ranked by absolute correlation:\n")

for feature, correlation in correlations.items():
    print(f"{feature:35} {correlation: .6f}")

# ==========================================================
# 3. VERY HIGH CORRELATIONS
# ==========================================================

print("\n" + "=" * 70)
print("VERY HIGH CORRELATIONS")
print("=" * 70)

high_corr = correlations[
    correlations.abs() >= 0.90
]

if len(high_corr) == 0:
    print("No features have correlation >= 0.90 with the target.")
else:
    print(high_corr)

# ==========================================================
# 4. UNIQUE VALUES BY TARGET
# ==========================================================

print("\n" + "=" * 70)
print("FEATURES WITH EXTREME TARGET SEPARATION")
print("=" * 70)

binary_columns = []

for column in df.columns:

    if column == TARGET:
        continue

    unique_values = df[column].dropna().unique()

    if len(unique_values) == 2:
        binary_columns.append(column)

print("\nBinary features:")

for column in binary_columns:

    table = pd.crosstab(
        df[column],
        df[TARGET],
        normalize="index"
    )

    print("\n--------------------------------------")
    print(column)
    print("--------------------------------------")

    print(table)

# ==========================================================
# 5. TARGET MEAN FOR NUMERIC FEATURES
# ==========================================================

print("\n" + "=" * 70)
print("TARGET MEAN BY FEATURE VALUE")
print("=" * 70)

for column in numeric_columns:

    if column == TARGET:
        continue

    grouped = (
        df.groupby(column)[TARGET]
        .mean()
    )

    if len(grouped) <= 20:

        print("\n--------------------------------------")
        print(column)
        print("--------------------------------------")

        print(grouped)

# ==========================================================
# 6. STRING FEATURES
# ==========================================================

print("\n" + "=" * 70)
print("STRING FEATURES")
print("=" * 70)

string_columns = df.select_dtypes(
    include=["object", "string"]
).columns

for column in string_columns:

    print(
        f"{column:20} "
        f"unique values = {df[column].nunique():,}"
    )

# ==========================================================
# 7. LABEL BY TLD
# ==========================================================

print("\n" + "=" * 70)
print("TOP TLDs BY PHISHING RATE")
print("=" * 70)

tld_stats = (
    df.groupby("TLD")
    .agg(
        samples=("label", "size"),
        phishing_rate=("label", "mean")
    )
    .sort_values("samples", ascending=False)
)

tld_stats["phishing_rate"] = (
    tld_stats["phishing_rate"] * 100
)

print(
    tld_stats.head(30).round(2)
)

# ==========================================================
# 8. DOMAIN REUSE
# ==========================================================

print("\n" + "=" * 70)
print("DOMAIN REUSE ANALYSIS")
print("=" * 70)

domain_counts = df["Domain"].value_counts()

print(
    f"Domains appearing once: "
    f"{(domain_counts == 1).sum():,}"
)

print(
    f"Domains appearing more than once: "
    f"{(domain_counts > 1).sum():,}"
)

print(
    f"Most repeated domain count: "
    f"{domain_counts.max()}"
)

# ==========================================================
# 9. URL REUSE
# ==========================================================

print("\n" + "=" * 70)
print("URL REUSE ANALYSIS")
print("=" * 70)

url_counts = df["URL"].value_counts()

print(
    f"URLs appearing once: "
    f"{(url_counts == 1).sum():,}"
)

print(
    f"URLs appearing more than once: "
    f"{(url_counts > 1).sum():,}"
)

print(
    f"Maximum occurrences of one URL: "
    f"{url_counts.max()}"
)

# ==========================================================
# 10. FINAL MESSAGE
# ==========================================================

print("\n" + "=" * 70)
print("LEAKAGE ANALYSIS COMPLETE")
print("=" * 70)

print("""
IMPORTANT:

High correlation does NOT automatically mean data leakage.

A feature may legitimately be strongly associated with phishing.

We need to inspect suspicious features before removing anything.
""")