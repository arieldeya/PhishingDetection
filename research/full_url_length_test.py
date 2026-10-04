import pandas as pd

DATASET = "PhiUSIIL_Phishing_URL_Dataset.csv"

df = pd.read_csv(DATASET)

print("=" * 100)
print("FULL DATASET URL LENGTH VALIDATION")
print("=" * 100)

raw = df["URL"].astype(str).str.len()

dataset = df["URLLength"]

difference = raw - dataset

print(f"\nTotal rows: {len(df):,}")

print("\nDifference distribution:")
print("-" * 60)

print(difference.value_counts().sort_index())

print("\nExact matches:")
print((difference == 0).sum())

print("\nDifference +1:")
print((difference == 1).sum())

print("\nDifference -1:")
print((difference == -1).sum())

print("\nOther differences:")
print(
    ((difference != 0) & (difference != 1) & (difference != -1)).sum()
)

print("\nMean absolute error:")
print(difference.abs().mean())

print("\nMaximum absolute error:")
print(difference.abs().max())

print("\nPercentage exact:")
print((difference == 0).mean() * 100)

print("\nPercentage where RAW = DATASET + 1:")
print((difference == 1).mean() * 100)

print("=" * 100)