import pandas as pd

DATASET = "PhiUSIIL_Phishing_URL_Dataset.csv"

df = pd.read_csv(DATASET)

urls = df["URL"].astype(str)

dataset_length = df["URLLength"]

raw_length = urls.str.len()

# Candidate transformations
no_www = (
    urls.str.replace("www.", "", regex=False)
    .str.len()
)

no_www_minus_1 = no_www - 1

# Also test removing "www." only when it occurs after protocol
no_www_protocol = urls.str.replace(
    "://www.",
    "://",
    regex=False
).str.len()

no_www_protocol_minus_1 = no_www_protocol - 1

print("=" * 100)
print("TESTING URLLength AND WWW TRANSFORMATIONS")
print("=" * 100)

rules = {
    "RAW": raw_length,
    "RAW_MINUS_1": raw_length - 1,
    "NO_WWW": no_www,
    "NO_WWW_MINUS_1": no_www_minus_1,
    "NO_WWW_PROTOCOL": no_www_protocol,
    "NO_WWW_PROTOCOL_MINUS_1": no_www_protocol_minus_1,
}

print(
    f"\n{'RULE':<30}"
    f"{'EXACT':>15}"
    f"{'MAE':>15}"
)

for name, calculated in rules.items():

    exact = (calculated == dataset_length).sum()

    mae = (
        calculated.astype(float)
        .sub(dataset_length.astype(float))
        .abs()
        .mean()
    )

    print(
        f"{name:<30}"
        f"{exact:>15,}"
        f"{mae:>15.6f}"
    )

# ---------------------------------------------------------
# GROUP-SPECIFIC TEST
# ---------------------------------------------------------

print("\n" + "=" * 100)
print("GROUP-SPECIFIC ANALYSIS")
print("=" * 100)

difference = raw_length - dataset_length

group0 = difference == 0
group1 = difference == 1

print("\nGROUP 0: RAW = DATASET")
print("Rows:", group0.sum())

print("\nCandidate rules for Group 0:")

for name, calculated in rules.items():

    exact = (
        calculated[group0] == dataset_length[group0]
    ).sum()

    print(
        f"{name:<30}"
        f"{exact:>10,} / {group0.sum():,}"
    )

print("\nGROUP 1: RAW = DATASET + 1")
print("Rows:", group1.sum())

print("\nCandidate rules for Group 1:")

for name, calculated in rules.items():

    exact = (
        calculated[group1] == dataset_length[group1]
    ).sum()

    print(
        f"{name:<30}"
        f"{exact:>10,} / {group1.sum():,}"
    )

# ---------------------------------------------------------
# SHOW EXAMPLES
# ---------------------------------------------------------

print("\n" + "=" * 100)
print("EXAMPLES WHERE NO_WWW MATCHES DATASET")
print("=" * 100)

matches = no_www == dataset_length

shown = 0

for index in df.index[matches]:

    print("\nIndex:", index)
    print("URL:", df.loc[index, "URL"])
    print("Dataset URLLength:", dataset_length.loc[index])
    print("Raw length:", raw_length.loc[index])
    print("No-www length:", no_www.loc[index])

    shown += 1

    if shown >= 20:
        break

print("\n" + "=" * 100)
print("TEST COMPLETE")
print("=" * 100)