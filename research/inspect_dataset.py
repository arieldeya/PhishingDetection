import pandas as pd

DATASET_PATH = "PhiUSIIL_Phishing_URL_Dataset.csv"

print("=" * 60)
print("PHISHING URL DATASET INSPECTION")
print("=" * 60)

df = pd.read_csv(DATASET_PATH)

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
for column in df.columns:
    print("-", column)

print("\nFirst 5 rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nDataset information:")
df.info()

print("\nBasic statistics:")
print(df.describe(include="all").T)

print("\nTARGET LABEL DISTRIBUTION")
print(df["label"].value_counts())

print("\nTARGET LABEL PERCENTAGES")
print(df["label"].value_counts(normalize=True) * 100)