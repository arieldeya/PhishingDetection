import pandas as pd
from sklearn.model_selection import train_test_split

print("=" * 70)
print("PHISHING DATASET - DOMAIN-AWARE SPLIT")
print("=" * 70)

DATASET_PATH = "PhiUSIIL_Phishing_URL_Dataset.csv"

# ------------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------------

df = pd.read_csv(DATASET_PATH)

print("\nOriginal dataset:")
print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns):,}")

# ------------------------------------------------------------
# 2. Remove duplicate URLs
# ------------------------------------------------------------

df = df.drop_duplicates(subset=["URL"]).copy()

print(f"\nRows after URL deduplication: {len(df):,}")

# ------------------------------------------------------------
# 3. Create domain-level summary
# ------------------------------------------------------------

domain_summary = (
    df.groupby("Domain")
    .agg(
        samples=("label", "size"),
        positive_rate=("label", "mean")
    )
    .reset_index()
)

print("\n" + "=" * 70)
print("DOMAIN SUMMARY")
print("=" * 70)

print(f"Unique domains: {len(domain_summary):,}")

# ------------------------------------------------------------
# 4. Create domain-level stratification groups
# ------------------------------------------------------------
#
# We classify domains according to their dominant label.
#
# Pure domains:
#   positive_rate = 0 → label 0
#   positive_rate = 1 → label 1
#
# Mixed domains:
#   0 < positive_rate < 1
#
# Mixed domains are few, so we keep them together in training.
# ------------------------------------------------------------

pure_zero_domains = domain_summary[
    domain_summary["positive_rate"] == 0
]["Domain"]

pure_one_domains = domain_summary[
    domain_summary["positive_rate"] == 1
]["Domain"]

mixed_domains = domain_summary[
    (domain_summary["positive_rate"] > 0) &
    (domain_summary["positive_rate"] < 1)
]["Domain"]

print(f"Pure label-0 domains: {len(pure_zero_domains):,}")
print(f"Pure label-1 domains: {len(pure_one_domains):,}")
print(f"Mixed-label domains:   {len(mixed_domains):,}")

# ------------------------------------------------------------
# 5. Split pure label-0 domains
# ------------------------------------------------------------

zero_train, zero_test = train_test_split(
    pure_zero_domains,
    test_size=0.20,
    random_state=42
)

# ------------------------------------------------------------
# 6. Split pure label-1 domains
# ------------------------------------------------------------

one_train, one_test = train_test_split(
    pure_one_domains,
    test_size=0.20,
    random_state=42
)

# ------------------------------------------------------------
# 7. Keep mixed domains in training
# ------------------------------------------------------------
#
# There are only 54 mixed-label domains.
# Keeping them together avoids splitting their URLs across
# training and testing.
# ------------------------------------------------------------

train_domains = set(zero_train) | set(one_train) | set(mixed_domains)

test_domains = set(zero_test) | set(one_test)

# Safety check
overlap = train_domains.intersection(test_domains)

print("\n" + "=" * 70)
print("DOMAIN SPLIT CHECK")
print("=" * 70)

print(f"Training domains: {len(train_domains):,}")
print(f"Testing domains:  {len(test_domains):,}")
print(f"Domain overlap:   {len(overlap):,}")

if len(overlap) != 0:
    raise ValueError(
        "ERROR: Some domains appear in both training and testing!"
    )

print("SUCCESS: No domain appears in both sets.")

# ------------------------------------------------------------
# 8. Create row-level datasets
# ------------------------------------------------------------

train_df = df[df["Domain"].isin(train_domains)].copy()
test_df = df[df["Domain"].isin(test_domains)].copy()

# ------------------------------------------------------------
# 9. Display statistics
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL DOMAIN-AWARE SPLIT")
print("=" * 70)

print(f"Training rows: {len(train_df):,}")
print(f"Testing rows:  {len(test_df):,}")

print("\nTraining target distribution:")
print(train_df["label"].value_counts())
print(
    (train_df["label"].value_counts(normalize=True) * 100)
    .round(2)
)

print("\nTesting target distribution:")
print(test_df["label"].value_counts())
print(
    (test_df["label"].value_counts(normalize=True) * 100)
    .round(2)
)

# ------------------------------------------------------------
# 10. Check that domains really don't overlap
# ------------------------------------------------------------

actual_train_domains = set(train_df["Domain"])
actual_test_domains = set(test_df["Domain"])

actual_overlap = actual_train_domains.intersection(
    actual_test_domains
)

print("\n" + "=" * 70)
print("FINAL VALIDATION")
print("=" * 70)

print(f"Actual training domains: {len(actual_train_domains):,}")
print(f"Actual testing domains:  {len(actual_test_domains):,}")
print(f"Actual overlapping domains: {len(actual_overlap):,}")

# ------------------------------------------------------------
# 11. Save files
# ------------------------------------------------------------

train_df.to_csv("train_domain.csv", index=False)
test_df.to_csv("test_domain.csv", index=False)

print("\n" + "=" * 70)
print("FILES CREATED")
print("=" * 70)

print("Training file: train_domain.csv")
print("Testing file:  test_domain.csv")

print("\nDomain-aware split complete.")