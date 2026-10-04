import pandas as pd

print("=" * 70)
print("PHISHING DATASET DOMAIN AUDIT")
print("=" * 70)

DATASET_PATH = "PhiUSIIL_Phishing_URL_Dataset.csv"

df = pd.read_csv(DATASET_PATH)

# ==========================================================
# 1. DOMAIN LABEL DISTRIBUTION
# ==========================================================

print("\n" + "=" * 70)
print("DOMAIN LABEL DISTRIBUTION")
print("=" * 70)

domain_stats = (
    df.groupby("Domain")
    .agg(
        samples=("label", "size"),
        label_count=("label", "nunique"),
        phishing_rate=("label", "mean")
    )
)

print(
    domain_stats.describe().round(4)
)

# ==========================================================
# 2. DOMAINS WITH BOTH LABELS
# ==========================================================

print("\n" + "=" * 70)
print("DOMAINS WITH BOTH LABELS")
print("=" * 70)

mixed_domains = domain_stats[
    domain_stats["label_count"] > 1
].sort_values(
    "samples",
    ascending=False
)

print(
    f"Number of mixed-label domains: "
    f"{len(mixed_domains):,}"
)

print("\nLargest mixed-label domains:")

print(
    mixed_domains.head(30)
)

# ==========================================================
# 3. DOMAIN SAMPLE COUNTS
# ==========================================================

print("\n" + "=" * 70)
print("DOMAIN SAMPLE COUNTS")
print("=" * 70)

print(
    domain_stats["samples"]
    .value_counts()
    .sort_index()
    .head(30)
)

# ==========================================================
# 4. DOMAINS WITH MANY RECORDS
# ==========================================================

print("\n" + "=" * 70)
print("TOP 30 MOST FREQUENT DOMAINS")
print("=" * 70)

top_domains = (
    domain_stats
    .sort_values("samples", ascending=False)
    .head(30)
)

print(top_domains)

# ==========================================================
# 5. PURE DOMAINS
# ==========================================================

print("\n" + "=" * 70)
print("PURE DOMAIN DISTRIBUTION")
print("=" * 70)

pure_legitimate = domain_stats[
    (domain_stats["label_count"] == 1) &
    (domain_stats["phishing_rate"] == 0)
]

pure_phishing = domain_stats[
    (domain_stats["label_count"] == 1) &
    (domain_stats["phishing_rate"] == 1)
]

print(
    f"Domains appearing only with label 0: "
    f"{len(pure_legitimate):,}"
)

print(
    f"Domains appearing only with label 1: "
    f"{len(pure_phishing):,}"
)

# ==========================================================
# 6. SUMMARY
# ==========================================================

print("\n" + "=" * 70)
print("DOMAIN AUDIT SUMMARY")
print("=" * 70)

print(
    f"Total unique domains: "
    f"{df['Domain'].nunique():,}"
)

print(
    f"Mixed-label domains: "
    f"{len(mixed_domains):,}"
)

print(
    f"Pure label-0 domains: "
    f"{len(pure_legitimate):,}"
)

print(
    f"Pure label-1 domains: "
    f"{len(pure_phishing):,}"
)

print("\nDomain audit complete.")