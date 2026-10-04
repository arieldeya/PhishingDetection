import pandas as pd
from collections import Counter

DATASET = "PhiUSIIL_Phishing_URL_Dataset.csv"

df = pd.read_csv(DATASET)

urls = df["URL"].astype(str)
raw_length = urls.str.len()

difference = raw_length - df["URLLength"]

# Ignore the two corrupted/outlier records
df["RAW_LENGTH"] = raw_length
df["DIFFERENCE"] = difference

group0 = df[difference == 0]
group1 = df[difference == 1]

print("=" * 100)
print("INVESTIGATING THE +1 URL LENGTH DIFFERENCE")
print("=" * 100)

print(f"\nGroup 0: RAW = URLLength")
print(f"Rows: {len(group0):,}")

print(f"\nGroup 1: RAW = URLLength + 1")
print(f"Rows: {len(group1):,}")

# ---------------------------------------------------------
# CHARACTER FREQUENCY COMPARISON
# ---------------------------------------------------------

print("\n" + "=" * 100)
print("CHARACTER FREQUENCY COMPARISON")
print("=" * 100)

chars0 = Counter("".join(group0["URL"].astype(str)))
chars1 = Counter("".join(group1["URL"].astype(str)))

total0 = sum(chars0.values())
total1 = sum(chars1.values())

all_chars = set(chars0) | set(chars1)

results = []

for char in all_chars:

    freq0 = chars0[char] / total0
    freq1 = chars1[char] / total1

    results.append(
        (
            char,
            chars0[char],
            chars1[char],
            freq0,
            freq1,
            freq0 - freq1
        )
    )

results.sort(key=lambda x: abs(x[5]), reverse=True)

print(
    f"{'CHAR':<15}"
    f"{'GROUP0':>12}"
    f"{'GROUP1':>12}"
    f"{'G0 FREQ':>15}"
    f"{'G1 FREQ':>15}"
    f"{'DIFFERENCE':>15}"
)

for char, c0, c1, f0, f1, diff in results[:50]:

    display_char = repr(char)

    print(
        f"{display_char:<15}"
        f"{c0:>12,}"
        f"{c1:>12,}"
        f"{f0:>15.8f}"
        f"{f1:>15.8f}"
        f"{diff:>15.8f}"
    )

# ---------------------------------------------------------
# COMMON URL ENDINGS
# ---------------------------------------------------------

print("\n" + "=" * 100)
print("URL ENDING ANALYSIS")
print("=" * 100)

for name, group in [
    ("GROUP 0: RAW = URLLength", group0),
    ("GROUP 1: RAW = URLLength + 1", group1)
]:

    print("\n" + "-" * 80)
    print(name)

    urls_group = group["URL"].astype(str)

    endings = Counter()

    for url in urls_group:

        if len(url) >= 5:
            endings[url[-5:]] += 1

    print("\nMost common last 5 characters:")

    for ending, count in endings.most_common(20):
        print(repr(ending), count)

# ---------------------------------------------------------
# SAMPLE URL COMPARISON
# ---------------------------------------------------------

print("\n" + "=" * 100)
print("SAMPLE URL COMPARISON")
print("=" * 100)

print("\nGROUP 0 EXAMPLES:")

for _, row in group0.head(20).iterrows():

    print(
        f"\nURLLength={row['URLLength']} "
        f"Raw={row['RAW_LENGTH']} "
        f"Diff={row['DIFFERENCE']}"
    )

    print(row["URL"])


print("\n" + "=" * 100)
print("GROUP 1 EXAMPLES")
print("=" * 100)

for _, row in group1.head(20).iterrows():

    print(
        f"\nURLLength={row['URLLength']} "
        f"Raw={row['RAW_LENGTH']} "
        f"Diff={row['DIFFERENCE']}"
    )

    print(row["URL"])

print("\n" + "=" * 100)
print("INVESTIGATION COMPLETE")
print("=" * 100)