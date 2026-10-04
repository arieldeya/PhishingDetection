import pandas as pd

DATASET = "PhiUSIIL_Phishing_URL_Dataset.csv"

df = pd.read_csv(DATASET)

print("=" * 100)
print("URL LENGTH RULE TEST")
print("=" * 100)

rules = {
    "RAW": lambda u: len(u),

    "REMOVE_TRAILING_SLASH": lambda u:
        len(u[:-1]) if u.endswith("/") else len(u),

    "REMOVE_PROTOCOL": lambda u:
        len(
            u[8:] if u.startswith("https://")
            else u[7:] if u.startswith("http://")
            else u
        ),

    "REMOVE_PROTOCOL_TRAILING_SLASH": lambda u:
        len(
            (
                u[8:] if u.startswith("https://")
                else u[7:] if u.startswith("http://")
                else u
            ).rstrip("/")
        ),

    "RAW_MINUS_ONE": lambda u:
        len(u) - 1,
}

results = {}

for rule_name, rule in rules.items():

    exact = 0
    errors = []

    for _, row in df.head(1000).iterrows():

        url = str(row["URL"])
        expected = int(row["URLLength"])

        predicted = rule(url)

        if predicted == expected:
            exact += 1

        errors.append(abs(predicted - expected))

    results[rule_name] = {
        "exact": exact,
        "mae": sum(errors) / len(errors)
    }

print("\n")
print(
    f"{'RULE':35}"
    f"{'EXACT MATCHES':20}"
    f"{'MAE':15}"
)

print("-" * 70)

for rule_name, result in results.items():

    print(
        f"{rule_name:35}"
        f"{result['exact']:20}"
        f"{result['mae']:15.4f}"
    )

print("\n" + "=" * 100)