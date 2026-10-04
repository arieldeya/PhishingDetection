import pandas as pd
import re
import string

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


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def calculate_basic_features(url):

    url = str(url)

    url_length = len(url)

    # Letters
    letters = sum(char.isalpha() for char in url)

    # Digits
    digits = sum(char.isdigit() for char in url)

    # Special characters
    special = url_length - letters - digits

    # Ratios
    letter_ratio = letters / url_length if url_length else 0
    digit_ratio = digits / url_length if url_length else 0
    special_ratio = special / url_length if url_length else 0

    return {
        "URLLength": url_length,
        "NoOfLettersInURL": letters,
        "LetterRatioInURL": letter_ratio,
        "NoOfDegitsInURL": digits,
        "DegitRatioInURL": digit_ratio,
        "NoOfOtherSpecialCharsInURL": special,
        "SpacialCharRatioInURL": special_ratio
    }


# ============================================================
# DOMAIN / TLD FEATURES
# ============================================================

def calculate_domain_features(url):

    url = str(url)

    # Remove protocol
    domain_part = re.sub(
        r"^https?://",
        "",
        url,
        flags=re.IGNORECASE
    )

    # Remove path
    domain_part = domain_part.split("/")[0]

    # Remove port
    domain_part = domain_part.split(":")[0]

    domain = domain_part

    # Remove www.
    domain_without_www = domain

    if domain_without_www.lower().startswith("www."):
        domain_without_www = domain_without_www[4:]

    # TLD
    parts = domain_without_www.split(".")

    if len(parts) >= 2:
        tld = parts[-1]
    else:
        tld = ""

    # Number of subdomains
    #
    # This is deliberately a candidate formula.
    # We will validate it against the dataset.
    if len(parts) >= 2:
        subdomains = max(len(parts) - 2, 0)
    else:
        subdomains = 0

    return {
        "DomainLength": len(domain),
        "TLDLength": len(tld),
        "NoOfSubDomain": subdomains
    }


# ============================================================
# CHARACTER CONTINUATION
# ============================================================

def calculate_char_continuation(url):

    url = str(url)

    if not url:
        return 0.0

    max_letters = 0
    max_digits = 0
    max_special = 0

    current_letters = 0
    current_digits = 0
    current_special = 0

    for char in url:

        if char.isalpha():

            current_letters += 1
            current_digits = 0
            current_special = 0

            max_letters = max(
                max_letters,
                current_letters
            )

        elif char.isdigit():

            current_digits += 1
            current_letters = 0
            current_special = 0

            max_digits = max(
                max_digits,
                current_digits
            )

        else:

            current_special += 1
            current_letters = 0
            current_digits = 0

            max_special = max(
                max_special,
                current_special
            )

    return (
        max_letters +
        max_digits +
        max_special
    ) / len(url)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("PHIUSIIL FEATURE FORMULA TEST")
print("=" * 80)

df = pd.read_csv(DATASET)

print()
print("Dataset:", df.shape)

sample = df.sample(1000, random_state=42).copy()


# ============================================================
# CALCULATE FEATURES
# ============================================================

for index, row in sample.iterrows():

    url = row["URL"]

    basic = calculate_basic_features(url)

    domain = calculate_domain_features(url)

    continuation = calculate_char_continuation(url)

    calculated = {}

    calculated.update(basic)
    calculated.update(domain)
    calculated["CharContinuationRate"] = continuation

    for feature in calculated:

        original = row[feature]
        predicted = calculated[feature]

        if feature in [
            "LetterRatioInURL",
            "DegitRatioInURL",
            "SpacialCharRatioInURL",
            "CharContinuationRate"
        ]:

            if abs(float(original) - float(predicted)) > 0.001:

                print()
                print("=" * 80)
                print("MISMATCH")
                print("=" * 80)

                print("ROW:", index)
                print("URL:", url)
                print()
                print("Feature:", feature)
                print("Original:", original)
                print("Calculated:", predicted)

                print()
                print("Domain:", row["Domain"])
                print("TLD:", row["TLD"])

                print("=" * 80)

                break

        else:

            if int(original) != int(predicted):

                print()
                print("=" * 80)
                print("MISMATCH")
                print("=" * 80)

                print("ROW:", index)
                print("URL:", url)
                print()
                print("Feature:", feature)
                print("Original:", original)
                print("Calculated:", predicted)

                print()
                print("Domain:", row["Domain"])
                print("TLD:", row["TLD"])

                print("=" * 80)

                break


print()
print("=" * 80)
print("TEST COMPLETE")
print("=" * 80)