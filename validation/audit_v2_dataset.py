import pandas as pd
from urllib.parse import urlparse


DATASET = "PhiUSIIL_Phishing_URL_Dataset.csv"


def normalize_url(url):
    """Normalize a URL for exact/near duplicate analysis."""
    if pd.isna(url):
        return ""

    url = str(url).strip().lower()

    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    return url.rstrip("/")


def extract_domain(url):
    """Extract hostname from a URL."""
    try:
        parsed = urlparse(url)

        if parsed.hostname:
            return parsed.hostname.lower()

    except Exception:
        pass

    return ""


print("=" * 100)
print("PHISHGUARD V2 - DATASET LEAKAGE & DUPLICATION AUDIT")
print("=" * 100)
print()


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("1. LOADING DATASET")
print("-" * 100)

try:
    df = pd.read_csv(DATASET)
except Exception as error:
    print(f"ERROR loading dataset: {error}")
    raise SystemExit

print(f"Dataset file: {DATASET}")
print(f"Rows:          {len(df):,}")
print(f"Columns:       {len(df.columns)}")
print()


# ============================================================
# 2. BASIC DATASET INFORMATION
# ============================================================

print("2. DATASET STRUCTURE")
print("-" * 100)

print("Columns:")
for number, column in enumerate(df.columns):
    print(f"{number:2d}. {column}")

print()


# ============================================================
# 3. TARGET DISTRIBUTION
# ============================================================

print("3. TARGET LABEL DISTRIBUTION")
print("-" * 100)

if "label" not in df.columns:
    print("ERROR: 'label' column not found.")
    raise SystemExit

label_counts = df["label"].value_counts(dropna=False)

print(label_counts)
print()

for label, count in label_counts.items():
    percentage = count / len(df) * 100
    print(f"Label {label}: {count:,} ({percentage:.2f}%)")

print()


# ============================================================
# 4. MISSING VALUES
# ============================================================

print("4. MISSING VALUE AUDIT")
print("-" * 100)

missing_total = df.isna().sum().sum()

print(f"Total missing values: {missing_total:,}")

if missing_total > 0:
    print()
    print("Columns containing missing values:")

    missing = df.isna().sum()

    for column, count in missing[missing > 0].items():
        print(f"  {column}: {count:,}")

else:
    print("No missing values detected.")

print()


# ============================================================
# 5. EXACT DUPLICATE ROWS
# ============================================================

print("5. EXACT DUPLICATE ROW AUDIT")
print("-" * 100)

duplicate_rows = df.duplicated().sum()

print(f"Exact duplicate rows: {duplicate_rows:,}")

if duplicate_rows == 0:
    print("PASS: No exact duplicate rows detected.")
else:
    print("WARNING: Duplicate rows detected.")

print()


# ============================================================
# 6. EXACT DUPLICATE URLS
# ============================================================

print("6. EXACT DUPLICATE URL AUDIT")
print("-" * 100)

if "URL" not in df.columns:
    print("ERROR: 'URL' column not found.")
    raise SystemExit

df["NormalizedURL"] = df["URL"].apply(normalize_url)

duplicate_urls = df["NormalizedURL"].duplicated().sum()

unique_urls = df["NormalizedURL"].nunique()

print(f"Unique URLs:             {unique_urls:,}")
print(f"Duplicate URL records:  {duplicate_urls:,}")

if duplicate_urls == 0:
    print("PASS: No duplicate URLs detected.")
else:
    print("WARNING: Duplicate URLs detected.")

print()


# ============================================================
# 7. DUPLICATE URL LABEL CONFLICTS
# ============================================================

print("7. DUPLICATE URL LABEL-CONFLICT AUDIT")
print("-" * 100)

url_label_counts = (
    df.groupby("NormalizedURL")["label"]
    .nunique()
)

conflicting_urls = url_label_counts[
    url_label_counts > 1
]

print(
    f"URLs appearing with multiple labels: "
    f"{len(conflicting_urls):,}"
)

if len(conflicting_urls) == 0:
    print("PASS: No URL has conflicting labels.")
else:
    print("WARNING: Some URLs have conflicting labels.")
    print()
    print("Examples:")

    for url in conflicting_urls.index[:10]:
        print(f"  {url}")

print()


# ============================================================
# 8. DOMAIN EXTRACTION
# ============================================================

print("8. DOMAIN ANALYSIS")
print("-" * 100)

df["Domain_Audit"] = df["NormalizedURL"].apply(extract_domain)

empty_domains = (df["Domain_Audit"] == "").sum()

unique_domains = df["Domain_Audit"].nunique()

print(f"Unique domains:          {unique_domains:,}")
print(f"URLs with empty domain:  {empty_domains:,}")

print()


# ============================================================
# 9. DOMAIN FREQUENCY
# ============================================================

print("9. DOMAIN FREQUENCY ANALYSIS")
print("-" * 100)

domain_counts = df["Domain_Audit"].value_counts()

print("Top 20 most frequent domains:")
print()

for domain, count in domain_counts.head(20).items():
    print(f"{count:8,}  {domain}")

print()


# ============================================================
# 10. DOMAINS WITH BOTH LABELS
# ============================================================

print("10. DOMAIN LABEL CONSISTENCY AUDIT")
print("-" * 100)

domain_label_counts = (
    df.groupby("Domain_Audit")["label"]
    .nunique()
)

mixed_domains = domain_label_counts[
    domain_label_counts > 1
]

print(
    f"Domains appearing with BOTH labels: "
    f"{len(mixed_domains):,}"
)

if len(mixed_domains) == 0:
    print("PASS: No domain appears with conflicting labels.")
else:
    print(
        "NOTE: Some domains contain both legitimate and "
        "phishing URLs."
    )

    print()
    print("Examples:")

    for domain in mixed_domains.index[:20]:
        labels = sorted(
            df.loc[
                df["Domain_Audit"] == domain,
                "label"
            ].unique()
        )

        count = (
            df["Domain_Audit"] == domain
        ).sum()

        print(
            f"  {domain} "
            f"(records={count}, labels={labels})"
        )

print()


# ============================================================
# 11. DOMAINS WITH MANY URL RECORDS
# ============================================================

print("11. HIGH-FREQUENCY DOMAIN AUDIT")
print("-" * 100)

print(
    "Domains containing more than 100 dataset records:"
)

high_frequency_domains = domain_counts[
    domain_counts > 100
]

print(
    f"Number of high-frequency domains: "
    f"{len(high_frequency_domains):,}"
)

if len(high_frequency_domains) > 0:
    print()

    for domain, count in high_frequency_domains.head(20).items():
        print(f"{count:8,}  {domain}")

print()


# ============================================================
# 12. URL DUPLICATE GROUPS
# ============================================================

print("12. DUPLICATE URL GROUP ANALYSIS")
print("-" * 100)

url_frequency = (
    df["NormalizedURL"]
    .value_counts()
)

repeated_urls = url_frequency[
    url_frequency > 1
]

print(
    f"Unique URLs appearing more than once: "
    f"{len(repeated_urls):,}"
)

if len(repeated_urls) > 0:
    print()
    print("Top repeated URLs:")

    for url, count in repeated_urls.head(20).items():
        print(f"{count:8,}  {url}")

print()


# ============================================================
# 13. DOMAIN CONCENTRATION
# ============================================================

print("13. DOMAIN CONCENTRATION")
print("-" * 100)

top_10_domain_records = domain_counts.head(10).sum()

top_10_percentage = (
    top_10_domain_records /
    len(df) *
    100
)

print(
    f"Records belonging to top 10 domains: "
    f"{top_10_domain_records:,}"
)

print(
    f"Percentage of entire dataset: "
    f"{top_10_percentage:.2f}%"
)

print()


# ============================================================
# 14. LABEL DISTRIBUTION BY DOMAIN
# ============================================================

print("14. DOMAIN-LEVEL LABEL ANALYSIS")
print("-" * 100)

domain_stats = (
    df.groupby("Domain_Audit")["label"]
    .agg(
        records="count",
        unique_labels="nunique"
    )
)

print(
    "Domains containing 10 or more records:"
)

large_domains = domain_stats[
    domain_stats["records"] >= 10
].sort_values(
    "records",
    ascending=False
)

print(
    f"Number of such domains: "
    f"{len(large_domains):,}"
)

print()


# ============================================================
# 15. SUMMARY
# ============================================================

print("=" * 100)
print("AUDIT SUMMARY")
print("=" * 100)

print()

print(f"Dataset rows:                    {len(df):,}")
print(f"Unique URLs:                     {unique_urls:,}")
print(f"Unique domains:                  {unique_domains:,}")
print(f"Exact duplicate rows:            {duplicate_rows:,}")
print(f"Duplicate URL records:           {duplicate_urls:,}")
print(f"Conflicting URL labels:          {len(conflicting_urls):,}")
print(f"Domains with both labels:        {len(mixed_domains):,}")
print(f"High-frequency domains (>100):   {len(high_frequency_domains):,}")
print()

print("=" * 100)
print("AUDIT COMPLETE")
print("=" * 100)


# ============================================================
# CLEANUP
# ============================================================

# These columns were created only for this audit.
# They are not written back to the original CSV.
