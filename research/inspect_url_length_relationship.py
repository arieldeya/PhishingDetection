import pandas as pd
import numpy as np

DATASET = "PhiUSIIL_Phishing_URL_Dataset.csv"

df = pd.read_csv(DATASET)

print("=" * 100)
print("PHIUSIIL URL LENGTH RELATIONSHIP ANALYSIS")
print("=" * 100)

print("\nDataset shape:")
print(df.shape)

print("\nURLLength statistics:")
print(df["URLLength"].describe())

print("\n" + "=" * 100)
print("TESTING COMMON LENGTH RELATIONSHIPS")
print("=" * 100)

tests = []

for _, row in df.head(1000).iterrows():

    url = str(row["URL"])
    dataset_length = int(row["URLLength"])

    raw = len(url)

    # Different possible calculations
    without_http = (
        url[7:] if url.startswith("http://")
        else url[8:] if url.startswith("https://")
        else url
    )

    without_www = url.replace("www.", "", 1)

    without_http_www = without_http.replace("www.", "", 1)

    tests.append({
        "dataset": dataset_length,
        "raw": raw,
        "raw_minus_1": raw - 1,
        "raw_minus_2": raw - 2,
        "raw_minus_3": raw - 3,
        "without_http": len(without_http),
        "without_www": len(without_www),
        "without_http_www": len(without_http_www),
    })

test_df = pd.DataFrame(tests)

for column in test_df.columns:

    if column == "dataset":
        continue

    difference = test_df[column] - test_df["dataset"]

    exact = (difference == 0).sum()
    mae = difference.abs().mean()

    print(
        f"{column:25} "
        f"Exact: {exact:4d}/1000   "
        f"MAE: {mae:.4f}"
    )

print("\n" + "=" * 100)
print("DIFFERENCE BETWEEN RAW LENGTH AND DATASET LENGTH")
print("=" * 100)

difference = test_df["raw"] - test_df["dataset"]

print("\nCounts:")
print(difference.value_counts().sort_index())

print("\nStatistics:")
print(difference.describe())

print("\n" + "=" * 100)
print("CHECKING OTHER DATASET FEATURES")
print("=" * 100)

print("\nColumns containing 'Length':")

for column in df.columns:

    if "length" in column.lower():
        print(
            f"{column:30} "
            f"min={df[column].min()} "
            f"max={df[column].max()} "
            f"mean={df[column].mean():.3f}"
        )

print("\n" + "=" * 100)