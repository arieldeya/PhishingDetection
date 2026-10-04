import pandas as pd

print("=" * 70)
print("PHISHING URL DATASET AUDIT")
print("=" * 70)

# ---------------------------------------------------------
# 1. LOAD DATASET
# ---------------------------------------------------------

DATASET_PATH = "PhiUSIIL_Phishing_URL_Dataset.csv"

df = pd.read_csv(DATASET_PATH)

print("\nDataset shape:")
print(df.shape)

# ---------------------------------------------------------
# 2. TARGET DISTRIBUTION
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("TARGET DISTRIBUTION")
print("=" * 70)

print(df["label"].value_counts())

print("\nTarget percentages:")
print(
    df["label"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

# ---------------------------------------------------------
# 3. DUPLICATE URL CHECK
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("DUPLICATE URL AUDIT")
print("=" * 70)

duplicate_urls = df["URL"].duplicated().sum()

print(f"Duplicate URL values: {duplicate_urls}")

if duplicate_urls > 0:
    print("\nMost repeated URLs:")

    url_counts = (
        df["URL"]
        .value_counts()
        .head(20)
    )

    print(url_counts)

# ---------------------------------------------------------
# 4. SAME URL WITH DIFFERENT LABELS
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("URL / LABEL CONSISTENCY CHECK")
print("=" * 70)

url_label_counts = (
    df.groupby("URL")["label"]
    .nunique()
)

conflicting_urls = (
    url_label_counts[url_label_counts > 1]
)

print(
    f"URLs appearing with different labels: "
    f"{len(conflicting_urls)}"
)

if len(conflicting_urls) > 0:
    print("\nExamples:")
    print(conflicting_urls.head(20))

# ---------------------------------------------------------
# 5. DOMAIN DUPLICATES
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("DOMAIN ANALYSIS")
print("=" * 70)

print(
    f"Unique domains: "
    f"{df['Domain'].nunique():,}"
)

print(
    f"Total records: "
    f"{len(df):,}"
)

# ---------------------------------------------------------
# 6. DOMAIN / LABEL CONSISTENCY
# ---------------------------------------------------------

domain_label_counts = (
    df.groupby("Domain")["label"]
    .nunique()
)

conflicting_domains = (
    domain_label_counts[domain_label_counts > 1]
)

print(
    f"\nDomains appearing with both labels: "
    f"{len(conflicting_domains):,}"
)

# ---------------------------------------------------------
# 7. FILENAME CHECK
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("FILENAME CHECK")
print("=" * 70)

print(
    f"Unique filenames: "
    f"{df['FILENAME'].nunique():,}"
)

print(
    f"Duplicate filenames: "
    f"{df['FILENAME'].duplicated().sum():,}"
)

# ---------------------------------------------------------
# 8. TARGET VALUES
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("TARGET VALUE VALIDATION")
print("=" * 70)

print("Unique labels:")
print(sorted(df["label"].unique()))

invalid_labels = df[
    ~df["label"].isin([0, 1])
]

print(
    f"Invalid label values: "
    f"{len(invalid_labels)}"
)

# ---------------------------------------------------------
# 9. CONSTANT FEATURES
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("CONSTANT FEATURE CHECK")
print("=" * 70)

constant_columns = []

for column in df.columns:
    if df[column].nunique() <= 1:
        constant_columns.append(column)

if constant_columns:
    print("Constant columns:")
    for column in constant_columns:
        print(f"- {column}")
else:
    print("No constant columns found.")

# ---------------------------------------------------------
# 10. HIGH-CARDINALITY CATEGORICAL FEATURES
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("CATEGORICAL FEATURE CARDINALITY")
print("=" * 70)

categorical_columns = df.select_dtypes(
    include=["object", "string"]
).columns

for column in categorical_columns:
    print(
        f"{column}: "
        f"{df[column].nunique():,} unique values"
    )

# ---------------------------------------------------------
# 11. EXTREME NUMERIC VALUES
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("NUMERIC FEATURE RANGE CHECK")
print("=" * 70)

numeric_columns = df.select_dtypes(
    include=["int64", "float64"]
).columns

for column in numeric_columns:
    print(
        f"{column:30} "
        f"min={df[column].min():12} "
        f"max={df[column].max():12}"
    )

# ---------------------------------------------------------
# 12. FINAL SUMMARY
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)

print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")
print(f"Missing values: {df.isnull().sum().sum():,}")
print(f"Duplicate rows: {df.duplicated().sum():,}")
print(f"Duplicate URLs: {duplicate_urls:,}")
print(f"Conflicting URLs: {len(conflicting_urls):,}")
print(f"Conflicting domains: {len(conflicting_domains):,}")

print("\nNext step:")
print("Perform feature leakage analysis before model training.")