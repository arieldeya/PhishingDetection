import pandas as pd
import numpy as np

DATASET = "PhiUSIIL_Phishing_URL_Dataset.csv"

df = pd.read_csv(DATASET)

raw_length = df["URL"].astype(str).str.len()

difference = raw_length - df["URLLength"]

df["RAW_LENGTH"] = raw_length
df["LENGTH_DIFFERENCE"] = difference

group_exact = df[difference == 0]
group_minus_one = df[difference == 1]

print("=" * 100)
print("COMPARISON OF URL LENGTH GROUPS")
print("=" * 100)

print("\nGroup 0:")
print("RAW = URLLength")
print(f"Rows: {len(group_exact):,}")

print("\nGroup 1:")
print("RAW = URLLength + 1")
print(f"Rows: {len(group_minus_one):,}")

print("\n" + "=" * 100)
print("GROUP COMPARISON")
print("=" * 100)

features = [
    "TLDLength",
    "NoOfLettersInURL",
    "NoOfDegitsInURL",
    "NoOfOtherSpecialCharsInURL",
    "NoOfEqualsInURL",
    "NoOfQMarkInURL",
    "NoOfAmpersandInURL",
    "NoOfOtherSpecialCharsInURL",
    "IsHTTPS",
    "IsDomainIP",
]

for feature in features:

    if feature not in df.columns:
        continue

    print("\n" + "-" * 80)
    print(feature)

    print(
        "RAW = URLLength:",
        group_exact[feature].mean()
    )

    print(
        "RAW = URLLength + 1:",
        group_minus_one[feature].mean()
    )

print("\n" + "=" * 100)
print("URL TERMINATION PATTERNS")
print("=" * 100)

for name, group in [
    ("RAW = URLLength", group_exact),
    ("RAW = URLLength + 1", group_minus_one)
]:

    urls = group["URL"].astype(str)

    print(f"\n{name}")

    print("Trailing slash:")
    print(urls.str.endswith("/").value_counts())

    print("\nStarts HTTP:")
    print(urls.str.startswith("http://").value_counts())

    print("\nStarts HTTPS:")
    print(urls.str.startswith("https://").value_counts())

    print("\nContains www:")
    print(urls.str.lower().str.contains("www.", regex=False).value_counts())

    print("\nContains query:")
    print(urls.str.contains("?", regex=False).value_counts())

print("\n" + "=" * 100)