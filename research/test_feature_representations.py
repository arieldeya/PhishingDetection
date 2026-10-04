import pandas as pd
import re
from urllib.parse import urlparse

DATASET = "PhiUSIIL_Phishing_URL_Dataset.csv"

df = pd.read_csv(DATASET)

FEATURES = [
    "URLLength",
    "NoOfLettersInURL",
    "NoOfDegitsInURL",
    "NoOfOtherSpecialCharsInURL",
]


def representations(url):

    url = str(url)

    parsed = urlparse(url)

    hostname = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""

    no_protocol = re.sub(
        r"^https?://",
        "",
        url,
        flags=re.IGNORECASE
    )

    no_www = no_protocol

    if no_www.lower().startswith("www."):
        no_www = no_www[4:]

    # Remove final TLD
    host_parts = hostname.split(".")

    if len(host_parts) >= 2:
        without_tld = ".".join(host_parts[:-1])
    else:
        without_tld = hostname

    # Remove protocol and final TLD
    no_protocol_no_tld = no_protocol

    if "." in no_protocol_no_tld:
        # Preserve path/query while removing the final hostname TLD
        host_part = no_protocol_no_tld.split("/", 1)[0]
        remainder = ""

        if "/" in no_protocol_no_tld:
            remainder = "/" + no_protocol_no_tld.split("/", 1)[1]

        host_parts2 = host_part.split(".")

        if len(host_parts2) >= 2:
            host_without_tld = ".".join(host_parts2[:-1])
            no_protocol_no_tld = host_without_tld + remainder

    return {
        "RAW": url,
        "NO_PROTOCOL": no_protocol,
        "NO_WWW": no_www,
        "HOSTNAME": hostname,
        "DOMAIN_WITHOUT_TLD": without_tld,
        "NO_PROTOCOL_NO_TLD": no_protocol_no_tld,
        "PATH": path,
        "QUERY": query,
    }


def counts(value):

    value = str(value)

    letters = sum(c.isalpha() for c in value)
    digits = sum(c.isdigit() for c in value)

    # Count all non-alphanumeric characters
    special = sum(not c.isalnum() for c in value)

    return {
        "length": len(value),
        "letters": letters,
        "digits": digits,
        "special": special,
    }


sample = df.sample(
    1000,
    random_state=42
).copy()


print("=" * 100)
print("PHIUSIIL FEATURE REPRESENTATION TEST")
print("=" * 100)

print()
print("Dataset rows:", len(df))
print("Testing rows:", len(sample))

representations_to_test = [
    "RAW",
    "NO_PROTOCOL",
    "NO_WWW",
    "HOSTNAME",
    "DOMAIN_WITHOUT_TLD",
    "NO_PROTOCOL_NO_TLD",
    "PATH",
    "QUERY",
]


for feature in FEATURES:

    print()
    print("=" * 100)
    print("FEATURE:", feature)
    print("=" * 100)

    target_column = feature

    if feature == "URLLength":

        target_key = "length"

    elif feature == "NoOfLettersInURL":

        target_key = "letters"

    elif feature == "NoOfDegitsInURL":

        target_key = "digits"

    elif feature == "NoOfOtherSpecialCharsInURL":

        target_key = "special"

    results = []

    for representation in representations_to_test:

        exact_matches = 0
        absolute_errors = []

        for _, row in sample.iterrows():

            url = row["URL"]

            reps = representations(url)

            calculated = counts(
                reps[representation]
            )[target_key]

            actual = row[target_column]

            error = abs(
                float(actual) -
                float(calculated)
            )

            absolute_errors.append(error)

            if error == 0:
                exact_matches += 1

        mean_error = sum(absolute_errors) / len(
            absolute_errors
        )

        results.append(
            (
                representation,
                exact_matches,
                mean_error
            )
        )

    results.sort(
        key=lambda x: (
            -x[1],
            x[2]
        )
    )

    print()
    print(
        f"{'REPRESENTATION':30}"
        f"{'EXACT MATCHES':20}"
        f"{'MEAN ABS ERROR':20}"
    )

    print("-" * 70)

    for representation, exact, error in results:

        print(
            f"{representation:30}"
            f"{exact:20}"
            f"{error:20.6f}"
        )


print()
print("=" * 100)
print("TEST COMPLETE")
print("=" * 100)