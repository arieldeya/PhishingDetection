import pandas as pd
from collections import Counter

DATASET = "PhiUSIIL_Phishing_URL_Dataset.csv"

df = pd.read_csv(DATASET)

print("=" * 100)
print("URL LENGTH DIFFERENCE DISTRIBUTION")
print("=" * 100)

differences = []

for _, row in df.head(1000).iterrows():

    url = str(row["URL"])
    dataset_length = int(row["URLLength"])
    calculated_length = len(url)

    difference = calculated_length - dataset_length

    differences.append(difference)

counter = Counter(differences)

print("\nDifference distribution:")
print("-" * 50)

for difference, count in sorted(counter.items()):
    percentage = count / len(differences) * 100

    print(
        f"Difference {difference:+4d} : "
        f"{count:4d} URLs "
        f"({percentage:6.2f}%)"
    )

print("\n" + "=" * 100)

print(f"Total URLs tested: {len(differences)}")
print(f"Exact matches: {differences.count(0)}")
print(f"Difference +1: {differences.count(1)}")
print(f"Difference -1: {differences.count(-1)}")

print("=" * 100)