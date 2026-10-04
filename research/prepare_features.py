import pandas as pd
import numpy as np

print("=" * 70)
print("PHISHING DETECTION - FEATURE PREPARATION")
print("=" * 70)

# ------------------------------------------------------------
# Input files
# ------------------------------------------------------------

TRAIN_FILE = "train_domain.csv"
TEST_FILE = "test_domain.csv"

# ------------------------------------------------------------
# Load datasets
# ------------------------------------------------------------

print("\nLoading datasets...")

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

print("\nLoaded datasets:")
print(f"Training rows: {len(train_df):,}")
print(f"Testing rows:  {len(test_df):,}")
print(f"Training columns: {len(train_df.columns):,}")
print(f"Testing columns:  {len(test_df.columns):,}")

# ------------------------------------------------------------
# Target
# ------------------------------------------------------------

TARGET = "label"

if TARGET not in train_df.columns:
    raise ValueError("ERROR: 'label' column is missing from training data.")

if TARGET not in test_df.columns:
    raise ValueError("ERROR: 'label' column is missing from testing data.")

y_train = train_df[TARGET].copy()
y_test = test_df[TARGET].copy()

# ------------------------------------------------------------
# Columns to remove
# ------------------------------------------------------------

DROP_COLUMNS = [
    # Raw identifiers / text
    "FILENAME",
    "URL",
    "Domain",
    "Title",

    # Suspicious / potentially dataset-construction-dependent
    "URLSimilarityIndex",

    # Highly redundant with DomainTitleMatchScore
    "URLTitleMatchScore",

    # Target
    "label"
]

print("\n" + "=" * 70)
print("REMOVED FEATURES")
print("=" * 70)

for column in DROP_COLUMNS:
    print(f"- {column}")

# ------------------------------------------------------------
# Check that all columns exist before removing them
# ------------------------------------------------------------

missing_drop_columns = [
    column
    for column in DROP_COLUMNS
    if column not in train_df.columns
]

if missing_drop_columns:
    raise ValueError(
        f"ERROR: These expected columns are missing: "
        f"{missing_drop_columns}"
    )

# ------------------------------------------------------------
# Create feature matrices
# ------------------------------------------------------------

X_train = train_df.drop(columns=DROP_COLUMNS).copy()
X_test = test_df.drop(columns=DROP_COLUMNS).copy()

# ------------------------------------------------------------
# Identify categorical features
# ------------------------------------------------------------

categorical_features = [
    "TLD"
]

if "TLD" not in X_train.columns:
    raise ValueError("ERROR: TLD column is missing.")

# ------------------------------------------------------------
# Identify numerical features
# ------------------------------------------------------------

numerical_features = [
    column
    for column in X_train.columns
    if column not in categorical_features
]

print("\n" + "=" * 70)
print("FEATURE TYPES")
print("=" * 70)

print(f"Numerical features:   {len(numerical_features)}")
print(f"Categorical features: {len(categorical_features)}")

print("\nCategorical features:")

for column in categorical_features:
    print(f"- {column}")

# ------------------------------------------------------------
# Verify numerical columns are actually numeric
# ------------------------------------------------------------

non_numeric_columns = X_train[
    numerical_features
].select_dtypes(exclude=np.number).columns.tolist()

if non_numeric_columns:
    raise ValueError(
        "ERROR: These supposed numerical features are not numeric: "
        f"{non_numeric_columns}"
    )

# ------------------------------------------------------------
# Encode TLD
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("ENCODING TLD")
print("=" * 70)

print("Creating one-hot encoded TLD features...")

# Convert TLD to string to avoid unexpected mixed-type issues
X_train["TLD"] = X_train["TLD"].astype(str)
X_test["TLD"] = X_test["TLD"].astype(str)

# One-hot encoding
train_tld = pd.get_dummies(
    X_train["TLD"],
    prefix="TLD",
    dtype=int
)

test_tld = pd.get_dummies(
    X_test["TLD"],
    prefix="TLD",
    dtype=int
)

print(f"Unique TLDs in training: {len(train_tld.columns):,}")
print(f"Unique TLDs in testing:  {len(test_tld.columns):,}")

# ------------------------------------------------------------
# Align test TLD columns with training TLD columns
# ------------------------------------------------------------

test_tld = test_tld.reindex(
    columns=train_tld.columns,
    fill_value=0
)

print("TLD columns aligned successfully.")

# ------------------------------------------------------------
# Remove original TLD
# ------------------------------------------------------------

X_train_numeric = X_train.drop(columns=["TLD"])
X_test_numeric = X_test.drop(columns=["TLD"])

# ------------------------------------------------------------
# Combine numerical features + encoded TLD
# ------------------------------------------------------------

X_train_final = pd.concat(
    [
        X_train_numeric.reset_index(drop=True),
        train_tld.reset_index(drop=True)
    ],
    axis=1
)

X_test_final = pd.concat(
    [
        X_test_numeric.reset_index(drop=True),
        test_tld.reset_index(drop=True)
    ],
    axis=1
)

# ------------------------------------------------------------
# Ensure training and testing columns are identical
# ------------------------------------------------------------

X_test_final = X_test_final.reindex(
    columns=X_train_final.columns,
    fill_value=0
)

# ------------------------------------------------------------
# Check for duplicate feature columns
# ------------------------------------------------------------

duplicate_feature_columns = X_train_final.columns[
    X_train_final.columns.duplicated()
].tolist()

if duplicate_feature_columns:
    raise ValueError(
        "ERROR: Duplicate feature columns detected: "
        f"{duplicate_feature_columns}"
    )

# ------------------------------------------------------------
# Check missing values
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("MISSING VALUE CHECK")
print("=" * 70)

train_missing = X_train_final.isna().sum().sum()
test_missing = X_test_final.isna().sum().sum()

print(f"Training missing values: {train_missing}")
print(f"Testing missing values:  {test_missing}")

if train_missing > 0 or test_missing > 0:
    raise ValueError(
        "ERROR: Missing values detected in feature matrices."
    )

# ------------------------------------------------------------
# Check infinite values
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("INFINITE VALUE CHECK")
print("=" * 70)

train_numeric = X_train_final.select_dtypes(
    include=np.number
)

test_numeric = X_test_final.select_dtypes(
    include=np.number
)

train_inf = np.isinf(train_numeric.to_numpy()).sum()
test_inf = np.isinf(test_numeric.to_numpy()).sum()

print(f"Training infinite values: {train_inf}")
print(f"Testing infinite values:  {test_inf}")

if train_inf > 0 or test_inf > 0:
    raise ValueError(
        "ERROR: Infinite values detected in feature matrices."
    )

# ------------------------------------------------------------
# Check data types
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DATA TYPE CHECK")
print("=" * 70)

non_numeric_final_train = X_train_final.select_dtypes(
    exclude=np.number
).columns.tolist()

non_numeric_final_test = X_test_final.select_dtypes(
    exclude=np.number
).columns.tolist()

print(
    f"Non-numeric training columns: "
    f"{len(non_numeric_final_train)}"
)

print(
    f"Non-numeric testing columns:  "
    f"{len(non_numeric_final_test)}"
)

if non_numeric_final_train:
    print("\nNon-numeric training columns:")
    for column in non_numeric_final_train:
        print(f"- {column}")

if non_numeric_final_test:
    print("\nNon-numeric testing columns:")
    for column in non_numeric_final_test:
        print(f"- {column}")

# ------------------------------------------------------------
# Final feature matrix
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL FEATURE MATRIX")
print("=" * 70)

print(f"Training shape: {X_train_final.shape}")
print(f"Testing shape:  {X_test_final.shape}")

print(
    f"\nFinal number of features: "
    f"{X_train_final.shape[1]:,}"
)

print("\nFirst 20 features:")

for feature in X_train_final.columns[:20]:
    print(f"- {feature}")

# ------------------------------------------------------------
# Check target values
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("TARGET VALIDATION")
print("=" * 70)

print("Training labels:")
print(y_train.value_counts().sort_index())

print("\nTesting labels:")
print(y_test.value_counts().sort_index())

valid_labels = {0, 1}

if not set(y_train.unique()).issubset(valid_labels):
    raise ValueError("ERROR: Unexpected values found in training labels.")

if not set(y_test.unique()).issubset(valid_labels):
    raise ValueError("ERROR: Unexpected values found in testing labels.")

print("\nTarget validation successful.")

# ------------------------------------------------------------
# Check feature/target row alignment
# ------------------------------------------------------------

if len(X_train_final) != len(y_train):
    raise ValueError(
        "ERROR: Training features and target have different row counts."
    )

if len(X_test_final) != len(y_test):
    raise ValueError(
        "ERROR: Testing features and target have different row counts."
    )

# ------------------------------------------------------------
# Save prepared datasets
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("SAVING PREPARED DATASETS")
print("=" * 70)

train_prepared = X_train_final.copy()
train_prepared[TARGET] = y_train.reset_index(drop=True)

test_prepared = X_test_final.copy()
test_prepared[TARGET] = y_test.reset_index(drop=True)

train_output = "train_features.csv"
test_output = "test_features.csv"

train_prepared.to_csv(
    train_output,
    index=False
)

test_prepared.to_csv(
    test_output,
    index=False
)

print(f"Created: {train_output}")
print(f"Created: {test_output}")

# ------------------------------------------------------------
# Verify saved files
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("OUTPUT FILE VERIFICATION")
print("=" * 70)

print(f"{train_output}: {len(train_prepared):,} rows")
print(f"{test_output}:  {len(test_prepared):,} rows")

print(
    f"{train_output} columns: "
    f"{len(train_prepared.columns):,}"
)

print(
    f"{test_output} columns:  "
    f"{len(test_prepared.columns):,}"
)

# ------------------------------------------------------------
# Final summary
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FEATURE PREPARATION SUMMARY")
print("=" * 70)

print(f"Training rows:       {len(X_train_final):,}")
print(f"Testing rows:        {len(X_test_final):,}")
print(f"Final features:      {X_train_final.shape[1]:,}")
print(f"Training missing:    {train_missing}")
print(f"Testing missing:     {test_missing}")
print(f"Training infinity:   {train_inf}")
print(f"Testing infinity:    {test_inf}")

print("\nRemoved:")
print("- FILENAME")
print("- URL")
print("- Domain")
print("- Title")
print("- URLSimilarityIndex")
print("- URLTitleMatchScore")

print("\nEncoded:")
print("- TLD → One-Hot Encoding")

print("\n" + "=" * 70)
print("FEATURE PREPARATION COMPLETE")
print("=" * 70)