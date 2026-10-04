
import pandas as pd
import numpy as np
import re
from urllib.parse import urlparse

DATASET = "PhiUSIIL_Phishing_URL_Dataset.csv"

print("=" * 100)
print("PHIUSIIL CHARACTER FEATURE REVERSE ENGINEERING")
print("=" * 100)

df = pd.read_csv(DATASET)

urls = df["URL"].astype(str)

# ------------------------------------------------------------
# REPRESENTATIONS
# ------------------------------------------------------------

def remove_protocol(url):
    return re.sub(r"^https?://", "", url, flags=re.IGNORECASE)


def remove_www(url):
    return re.sub(r"^www\.", "", url, flags=re.IGNORECASE)


def remove_protocol_and_www(url):
    x = remove_protocol(url)
    x = remove_www(x)
    return x


def hostname(url):
    try:
        return urlparse(url).hostname or ""
    except:
        return ""


def domain_from_dataset(url):
    """
    Approximation using URL parsing.
    """
    try:
        host = urlparse(url).hostname or ""
        return host
    except:
        return ""


representations = {
    "RAW": urls,
    "NO_PROTOCOL": urls.map(remove_protocol),
    "NO_WWW": urls.map(remove_www),
    "NO_PROTOCOL_NO_WWW": urls.map(remove_protocol_and_www),
    "HOSTNAME": urls.map(hostname),
}


# ------------------------------------------------------------
# CHARACTER COUNT FUNCTIONS
# ------------------------------------------------------------

def ascii_letters(s):
    return sum(("a" <= c.lower() <= "z") for c in s)


def unicode_letters(s):
    return sum(c.isalpha() for c in s)


def ascii_digits(s):
    return sum("0" <= c <= "9" for c in s)


def unicode_digits(s):
    return sum(c.isdigit() for c in s)


def non_alphanumeric(s):
    return sum(not c.isalnum() for c in s)


def special_excluding_query_symbols(s):
    return sum(
        (not c.isalnum()) and c not in "=?&"
        for c in s
    )


def special_excluding_url_structure(s):
    return sum(
        (not c.isalnum()) and c not in ":/.?=&"
        for c in s
    )


def special_excluding_common_structure(s):
    return sum(
        (not c.isalnum()) and c not in ":/.?=&-_"
        for c in s
    )


# ------------------------------------------------------------
# DATASET TARGETS
# ------------------------------------------------------------

targets = {
    "NoOfLettersInURL": df["NoOfLettersInURL"].astype(float).values,
    "NoOfDegitsInURL": df["NoOfDegitsInURL"].astype(float).values,
    "NoOfOtherSpecialCharsInURL":
        df["NoOfOtherSpecialCharsInURL"].astype(float).values,
    "CharContinuationRate":
        df["CharContinuationRate"].astype(float).values,
}


# ------------------------------------------------------------
# COMPARISON FUNCTION
# ------------------------------------------------------------

def compare(name, calculated, target):

    calculated = np.asarray(calculated, dtype=float)
    target = np.asarray(target, dtype=float)

    diff = calculated - target

    exact = np.sum(np.isclose(calculated, target, atol=1e-12))
    mae = np.mean(np.abs(diff))
    max_error = np.max(np.abs(diff))

    print(
        f"{name:<38} "
        f"Exact: {exact:>8}/{len(target)} "
        f"({exact / len(target) * 100:>7.3f}%) "
        f"MAE: {mae:.6f} "
        f"MAX: {max_error:.3f}"
    )


# ============================================================
# LETTER COUNTS
# ============================================================

print("\n" + "=" * 100)
print("1. NoOfLettersInURL")
print("=" * 100)

for rep_name, rep in representations.items():

    compare(
        f"{rep_name} - ASCII letters",
        rep.map(ascii_letters),
        targets["NoOfLettersInURL"]
    )

    compare(
        f"{rep_name} - Unicode letters",
        rep.map(unicode_letters),
        targets["NoOfLettersInURL"]
    )


# ============================================================
# DIGIT COUNTS
# ============================================================

print("\n" + "=" * 100)
print("2. NoOfDegitsInURL")
print("=" * 100)

for rep_name, rep in representations.items():

    compare(
        f"{rep_name} - ASCII digits",
        rep.map(ascii_digits),
        targets["NoOfDegitsInURL"]
    )

    compare(
        f"{rep_name} - Unicode digits",
        rep.map(unicode_digits),
        targets["NoOfDegitsInURL"]
    )


# ============================================================
# SPECIAL CHARACTERS
# ============================================================

print("\n" + "=" * 100)
print("3. NoOfOtherSpecialCharsInURL")
print("=" * 100)

special_methods = {
    "NON_ALPHANUMERIC":
        lambda s: non_alphanumeric(s),

    "EXCLUDE_=?&":
        lambda s: special_excluding_query_symbols(s),

    "EXCLUDE_:/.?=&":
        lambda s: special_excluding_url_structure(s),

    "EXCLUDE_:/.?=&-_":
        lambda s: special_excluding_common_structure(s),
}

for rep_name, rep in representations.items():

    for method_name, func in special_methods.items():

        compare(
            f"{rep_name} - {method_name}",
            rep.map(func),
            targets["NoOfOtherSpecialCharsInURL"]
        )


# ============================================================
# SHOW EXAMPLES WHERE BEST CANDIDATES DIFFER
# ============================================================

print("\n" + "=" * 100)
print("4. SAMPLE ROWS")
print("=" * 100)

for i in range(min(20, len(df))):

    url = urls.iloc[i]

    print("\n" + "-" * 100)
    print("INDEX:", i)
    print("URL:", repr(url))

    print(
        "DATASET:",
        "Letters =", df["NoOfLettersInURL"].iloc[i],
        "| Digits =", df["NoOfDegitsInURL"].iloc[i],
        "| Special =", df["NoOfOtherSpecialCharsInURL"].iloc[i],
        "| Continuation =", df["CharContinuationRate"].iloc[i]
    )

    print(
        "RAW:",
        "ASCII letters =", ascii_letters(url),
        "| Unicode letters =", unicode_letters(url),
        "| Digits =", ascii_digits(url),
        "| Special =", non_alphanumeric(url)
    )

    no_www = remove_www(url)

    print(
        "NO_WWW:",
        repr(no_www),
        "| ASCII letters =", ascii_letters(no_www),
        "| Unicode letters =", unicode_letters(no_www),
        "| Special =", non_alphanumeric(no_www)
    )

    print(
        "HOSTNAME:",
        repr(hostname(url)),
        "| ASCII letters =", ascii_letters(hostname(url)),
        "| Unicode letters =", unicode_letters(hostname(url))
    )


# ============================================================
# CHARACTER CONTINUATION EXPERIMENTS
# ============================================================

print("\n" + "=" * 100)
print("5. CHAR CONTINUATION RATE CANDIDATES")
print("=" * 100)


def continuation_adjacent(url):
    if len(url) <= 1:
        return 0.0

    matches = sum(
        url[i] == url[i - 1]
        for i in range(1, len(url))
    )

    return matches / (len(url) - 1)


def continuation_run_excess(url):
    """
    For each run of identical characters,
    count only characters beyond the first.
    """
    if not url:
        return 0.0

    excess = 0
    i = 0

    while i < len(url):

        j = i + 1

        while j < len(url) and url[j] == url[i]:
            j += 1

        run_length = j - i

        if run_length > 1:
            excess += run_length - 1

        i = j

    return excess / len(url)


def continuation_longest_run(url):
    """
    Longest identical-character run divided by URL length.
    """
    if not url:
        return 0.0

    longest = 1
    current = 1

    for i in range(1, len(url)):

        if url[i] == url[i - 1]:
            current += 1
            longest = max(longest, current)
        else:
            current = 1

    return longest / len(url)


def continuation_same_class(url):
    """
    Measures continuation of the same character class:
    letters, digits, or special characters.
    """

    if not url:
        return 0.0

    def cls(c):
        if c.isalpha():
            return "L"
        if c.isdigit():
            return "D"
        return "S"

    matches = sum(
        cls(url[i]) == cls(url[i - 1])
        for i in range(1, len(url))
    )

    return matches / (len(url) - 1) if len(url) > 1 else 0.0


continuation_methods = {
    "ADJACENT_EQUAL": continuation_adjacent,
    "RUN_EXCESS": continuation_run_excess,
    "LONGEST_RUN": continuation_longest_run,
    "SAME_CLASS": continuation_same_class,
}


for rep_name, rep in representations.items():

    for method_name, func in continuation_methods.items():

        compare(
            f"{rep_name} - {method_name}",
            rep.map(func),
            targets["CharContinuationRate"]
        )


print("\n" + "=" * 100)
print("DIAGNOSTIC COMPLETE")
print("=" * 100)
