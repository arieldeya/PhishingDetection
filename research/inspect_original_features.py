import pandas as pd

DATASET = "PhiUSIIL_Phishing_URL_Dataset.csv"

FEATURES = [
    "URLLength",
    "DomainLength",
    "TLDLength",
    "NoOfSubDomain",
    "NoOfLettersInURL",
    "LetterRatioInURL",
    "NoOfDegitsInURL",
    "DegitRatioInURL",
    "NoOfOtherSpecialCharsInURL",
    "SpacialCharRatioInURL",
    "CharContinuationRate",
    "URLCharProb"
]

print("=" * 80)
print("PHIUSIIL ORIGINAL FEATURE INSPECTION")
print("=" * 80)

df = pd.read_csv(DATASET)

print()
print("Dataset shape:", df.shape)

print()
print("Columns:")
print(df.columns.tolist())

print()
print("=" * 80)
print("SAMPLE URLS AND ORIGINAL FEATURES")
print("=" * 80)

sample = df.sample(10, random_state=42)

for index, row in sample.iterrows():

    print()
    print("-" * 80)

    print("ROW:", index)

    print("URL:")
    print(row["URL"])

    print()
    print("Domain:")
    print(row["Domain"])

    print()
    print("TLD:")
    print(row["TLD"])

    print()
    print("LABEL:", row["label"])

    print()
    print("ORIGINAL FEATURES:")

    for feature in FEATURES:
        print(
            f"{feature:35} = {row[feature]}"
        )

print()
print("=" * 80)
print("INSPECTION COMPLETE")
print("=" * 80)