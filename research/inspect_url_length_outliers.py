import pandas as pd

DATASET = "PhiUSIIL_Phishing_URL_Dataset.csv"

df = pd.read_csv(DATASET)

raw_length = df["URL"].astype(str).str.len()
difference = raw_length - df["URLLength"]

print("=" * 100)
print("URL LENGTH OUTLIERS")
print("=" * 100)

outliers = df[(difference != 0) & (difference != 1)].copy()

outliers["RAW_LENGTH"] = outliers["URL"].astype(str).str.len()
outliers["DIFFERENCE"] = (
    outliers["RAW_LENGTH"] - outliers["URLLength"]
)

print(f"\nOutliers found: {len(outliers)}")

for index, row in outliers.iterrows():

    print("\n" + "-" * 100)

    print("DATASET INDEX:", index)

    print("URL:")
    print(row["URL"])

    print("\nURLLength:", row["URLLength"])

    print("Raw length:", row["RAW_LENGTH"])

    print("Difference:", row["DIFFERENCE"])

    print("\nOther relevant features:")

    for column in [
        "TLD",
        "TLDLength",
        "NoOfLettersInURL",
        "NoOfDegitsInURL",
        "NoOfOtherSpecialCharsInURL",
        "IsHTTPS",
        "IsDomainIP",
    ]:

        if column in df.columns:
            print(f"{column}: {row[column]}")

print("\n" + "=" * 100)