import pandas as pd
from sklearn.model_selection import train_test_split

print("=" * 70)
print("PHISHING DATASET - TRAIN/TEST SPLIT PREPARATION")
print("=" * 70)

DATASET_PATH = "PhiUSIIL_Phishing_URL_Dataset.csv"

df = pd.read_csv(DATASET_PATH)

print("\nOriginal dataset:")
print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns):,}")

# ------------------------------------------------------------
# 1. Remove exact duplicate URLs
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("URL DUPLICATE CHECK")
print("=" * 70)

duplicate_urls = df["URL"].duplicated(keep=False).sum()

print(f"Rows belonging to repeated URLs: {duplicate_urls:,}")

# Keep one copy of each URL.
# There are no conflicting URL labels according to our audit.
df_unique = df.drop_duplicates(subset=["URL"]).copy()

print(f"Rows after URL deduplication: {len(df_unique):,}")
print(f"Rows removed: {len(df) - len(df_unique):,}")

# ------------------------------------------------------------
# 2. Separate features and target
# ------------------------------------------------------------

X = df_unique.drop(columns=["label"])
y = df_unique["label"]

print("\n" + "=" * 70)
print("TARGET DISTRIBUTION")
print("=" * 70)

print(y.value_counts())
print("\nPercentages:")
print((y.value_counts(normalize=True) * 100).round(2))

# ------------------------------------------------------------
# 3. Random stratified split
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("RANDOM STRATIFIED SPLIT")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Training rows: {len(X_train):,}")
print(f"Testing rows:  {len(X_test):,}")

print("\nTraining target distribution:")
print(y_train.value_counts(normalize=True).round(4))

print("\nTesting target distribution:")
print(y_test.value_counts(normalize=True).round(4))

# ------------------------------------------------------------
# 4. Check domain overlap
# ------------------------------------------------------------

train_domains = set(X_train["Domain"])
test_domains = set(X_test["Domain"])

overlap = train_domains.intersection(test_domains)

print("\n" + "=" * 70)
print("DOMAIN OVERLAP CHECK")
print("=" * 70)

print(f"Training domains: {len(train_domains):,}")
print(f"Testing domains:  {len(test_domains):,}")
print(f"Domains appearing in BOTH: {len(overlap):,}")

if len(overlap) > 0:
    print("\nWARNING:")
    print("The random split contains domains appearing in both")
    print("training and testing datasets.")

# ------------------------------------------------------------
# 5. Save split datasets
# ------------------------------------------------------------

train_output = "train_random.csv"
test_output = "test_random.csv"

train_data = X_train.copy()
train_data["label"] = y_train

test_data = X_test.copy()
test_data["label"] = y_test

train_data.to_csv(train_output, index=False)
test_data.to_csv(test_output, index=False)

print("\n" + "=" * 70)
print("FILES CREATED")
print("=" * 70)

print(f"Training file: {train_output}")
print(f"Testing file:  {test_output}")

print("\nSplit preparation complete.")