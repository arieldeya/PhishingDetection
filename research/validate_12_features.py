import pandas as pd
import numpy as np
from urllib.parse import urlparse

DATASET = "PhiUSIIL_Phishing_URL_Dataset.csv"

df = pd.read_csv(DATASET)

print("=" * 100)
print("VALIDATING THE 12 RANDOM FOREST FEATURES")
print("=" * 100)

# ---------------------------------------------------------
# BASIC URL PARSING
# ---------------------------------------------------------

urls = df["URL"].astype(str)

parsed = urls.apply(urlparse)

domains = parsed.apply(
    lambda x: x.netloc
)

# Remove username/password if present
domains = domains.str.split("@").str[-1]

# Remove port
domains = domains.str.split(":").str[0]

# ---------------------------------------------------------
# URL LENGTH
# ---------------------------------------------------------

url_length = urls.str.len()

# ---------------------------------------------------------
# DOMAIN LENGTH
# ---------------------------------------------------------

domain_length = domains.str.len()

# ---------------------------------------------------------
# TLD LENGTH
# ---------------------------------------------------------

tld = domains.str.split(".").str[-1]

tld_length = tld.str.len()

# ---------------------------------------------------------
# SUBDOMAIN
# ---------------------------------------------------------

def count_subdomains(domain):

    parts = domain.split(".")

    if len(parts) <= 2:
        return 0

    return len(parts) - 2


subdomain_count = domains.apply(count_subdomains)

# ---------------------------------------------------------
# LETTER COUNT
# ---------------------------------------------------------

letters = urls.str.count(r"[A-Za-z]")

# ---------------------------------------------------------
# DIGIT COUNT
# ---------------------------------------------------------

digits = urls.str.count(r"\d")

# ---------------------------------------------------------
# OTHER SPECIAL CHARACTERS
# ---------------------------------------------------------

special_chars = urls.str.count(
    r"[^A-Za-z0-9]"
)

equals = urls.str.count("=")
question = urls.str.count(r"\?")
ampersand = urls.str.count("&")

other_special = (
    special_chars
    - equals
    - question
    - ampersand
)

# ---------------------------------------------------------
# RATIOS
# ---------------------------------------------------------

letter_ratio = letters / url_length

digit_ratio = digits / url_length

special_ratio = special_chars / url_length

# ---------------------------------------------------------
# CHARACTER CONTINUATION RATE
# ---------------------------------------------------------

def calculate_char_continuation(url):

    if len(url) <= 1:
        return 0

    same = 0

    for i in range(len(url) - 1):

        if url[i] == url[i + 1]:

            same += 1

    return same / (len(url) - 1)


char_continuation = urls.apply(
    calculate_char_continuation
)

# ---------------------------------------------------------
# COMPARISON
# ---------------------------------------------------------

calculated = {
    "URLLength": url_length,
    "DomainLength": domain_length,
    "TLDLength": tld_length,
    "NoOfSubDomain": subdomain_count,
    "NoOfLettersInURL": letters,
    "LetterRatioInURL": letter_ratio,
    "NoOfDegitsInURL": digits,
    "DegitRatioInURL": digit_ratio,
    "NoOfOtherSpecialCharsInURL": other_special,
    "SpacialCharRatioInURL": special_ratio,
    "CharContinuationRate": char_continuation,
}

features = list(calculated.keys())

print("\n" + "=" * 100)
print("FEATURE VALIDATION")
print("=" * 100)

for feature in features:

    dataset_values = df[feature]
    calculated_values = calculated[feature]

    difference = (
        calculated_values.astype(float)
        - dataset_values.astype(float)
    )

    exact = (
        difference.abs() < 1e-9
    ).sum()

    mae = difference.abs().mean()

    print("\n" + "-" * 80)

    print(f"FEATURE: {feature}")

    print(
        f"Exact matches: {exact:,} / {len(df):,} "
        f"({exact / len(df) * 100:.4f}%)"
    )

    print(
        f"Mean absolute error: {mae:.8f}"
    )

    print(
        f"Maximum absolute error: "
        f"{difference.abs().max():.8f}"
    )

print("\n" + "=" * 100)
print("FEATURE VALIDATION COMPLETE")
print("=" * 100)