import pandas as pd

DATASET = "PhiUSIIL_Phishing_URL_Dataset.csv"

df = pd.read_csv(DATASET)

print("=" * 100)
print("PHIUSIIL DATASET FEATURE INSPECTION")
print("=" * 100)

print("\nDataset shape:")
print(df.shape)

print("\n" + "=" * 100)
print("ALL COLUMNS")
print("=" * 100)

for i, column in enumerate(df.columns):
    print(f"{i:2d}. {column}")

print("\n" + "=" * 100)
print("FEATURE DATA TYPES")
print("=" * 100)

print(df.dtypes)

print("\n" + "=" * 100)
print("UNIQUE VALUES")
print("=" * 100)

for column in df.columns:

    unique_count = df[column].nunique()

    print(
        f"{column:<40} "
        f"{unique_count:>10,}"
    )

print("\n" + "=" * 100)
print("MISSING VALUES")
print("=" * 100)

missing = df.isnull().sum()

print(
    missing[
        missing > 0
    ].sort_values(ascending=False)
)

print("\n" + "=" * 100)
print("TARGET CANDIDATES")
print("=" * 100)

for column in df.columns:

    if df[column].nunique() <= 10:

        print(
            f"\n{column}"
        )

        print(
            df[column].value_counts(dropna=False)
        )

print("\n" + "=" * 100)
print("FEATURE INSPECTION COMPLETE")
print("=" * 100)