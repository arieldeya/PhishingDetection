import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

print("=" * 70)
print("DOMAIN-AWARE VALIDATION SPLIT")
print("=" * 70)

# ---------------------------------------------------------
# 1. Load the existing domain-aware training dataset
# ---------------------------------------------------------
df = pd.read_csv("train_domain.csv")

print(f"\nOriginal training dataset shape: {df.shape}")

# ---------------------------------------------------------
# 2. Check required columns
# ---------------------------------------------------------
required_columns = ["Domain", "label"]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Required column '{column}' was not found.")

# ---------------------------------------------------------
# 3. Show original class distribution
# ---------------------------------------------------------
print("\nOriginal label distribution:")
print(df["label"].value_counts())

print("\nOriginal label percentages:")
print(
    (df["label"].value_counts(normalize=True) * 100)
    .round(2)
)

print(f"\nUnique domains: {df['Domain'].nunique():,}")

# ---------------------------------------------------------
# 4. Domain-aware 80/20 split
# ---------------------------------------------------------
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, validation_idx = next(
    splitter.split(
        df,
        y=df["label"],
        groups=df["Domain"]
    )
)

train_model = df.iloc[train_idx].copy()
validation = df.iloc[validation_idx].copy()

# ---------------------------------------------------------
# 5. Reset indexes
# ---------------------------------------------------------
train_model.reset_index(drop=True, inplace=True)
validation.reset_index(drop=True, inplace=True)

# ---------------------------------------------------------
# 6. Save the datasets
# ---------------------------------------------------------
train_model.to_csv(
    "train_model_domain.csv",
    index=False
)

validation.to_csv(
    "validation_domain.csv",
    index=False
)

# ---------------------------------------------------------
# 7. Check domain overlap
# ---------------------------------------------------------
train_domains = set(train_model["Domain"])
validation_domains = set(validation["Domain"])

overlap = train_domains.intersection(validation_domains)

# ---------------------------------------------------------
# 8. Display results
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("SPLIT RESULTS")
print("=" * 70)

print(f"\nTraining subset:")
print(f"Rows: {len(train_model):,}")
print(f"Unique domains: {train_model['Domain'].nunique():,}")

print("\nValidation subset:")
print(f"Rows: {len(validation):,}")
print(f"Unique domains: {validation['Domain'].nunique():,}")

print("\nDomain overlap:")
print(f"Overlapping domains: {len(overlap)}")

# ---------------------------------------------------------
# 9. Training class distribution
# ---------------------------------------------------------
print("\nTraining label distribution:")
print(train_model["label"].value_counts())

print("\nTraining label percentages:")
print(
    (train_model["label"].value_counts(normalize=True) * 100)
    .round(2)
)

# ---------------------------------------------------------
# 10. Validation class distribution
# ---------------------------------------------------------
print("\nValidation label distribution:")
print(validation["label"].value_counts())

print("\nValidation label percentages:")
print(
    (validation["label"].value_counts(normalize=True) * 100)
    .round(2)
)

# ---------------------------------------------------------
# 11. Final checks
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("VALIDATION CHECKS")
print("=" * 70)

if len(overlap) == 0:
    print("PASS: No domain overlap between training and validation.")
else:
    print("WARNING: Domain overlap detected!")

if len(train_model) + len(validation) == len(df):
    print("PASS: All original rows are preserved.")
else:
    print("WARNING: Row count mismatch!")

if (
    set(train_model.index).intersection(set(validation.index))
    == set()
):
    print("PASS: Training and validation rows are separate.")

print("\nFiles created:")
print(" - train_model_domain.csv")
print(" - validation_domain.csv")

print("\nDone.")