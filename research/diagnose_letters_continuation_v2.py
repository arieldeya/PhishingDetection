
import pandas as pd
import numpy as np
import re
from urllib.parse import urlparse

DATASET = "PhiUSIIL_Phishing_URL_Dataset.csv"

print("=" * 100)
print("PHIUSIIL LETTER + CHAR CONTINUATION V2")
print("=" * 100)

df = pd.read_csv(DATASET)

urls = df["URL"].astype(str)


# ============================================================
# URL REPRESENTATIONS
# ============================================================

def remove_protocol(url):
    return re.sub(r"^https?://", "", url, flags=re.IGNORECASE)


def remove_www(url):
    return re.sub(r"^www\.", "", url, flags=re.IGNORECASE)


def remove_protocol_then_www(url):
    x = remove_protocol(url)
    return remove_www(x)


def remove_www_then_protocol(url):
    x = remove_www(url)
    return remove_protocol(x)


def remove_protocol_www_exact(url):
    return re.sub(
        r"^https?://www\.",
        "",
        url,
        flags=re.IGNORECASE
    )


representations = {
    "RAW": urls,

    "NO_PROTOCOL":
        urls.map(remove_protocol),

    "NO_PROTOCOL_NO_WWW":
        urls.map(remove_protocol_then_www),

    "PROTOCOL_WWW_EXACT":
        urls.map(remove_protocol_www_exact),
}


# ============================================================
# LETTER COUNTERS
# ============================================================

def ascii_letters(s):
    return sum(("a" <= c.lower() <= "z") for c in s)


def regex_letters(s):
    return len(re.findall(r"[A-Za-z]", s))


def unicode_letters(s):
    return sum(c.isalpha() for c in s)


# ============================================================
# TARGET
# ============================================================

target_letters = (
    df["NoOfLettersInURL"]
    .astype(float)
    .values
)


# ============================================================
# COMPARISON
# ============================================================

def compare(name, values, target):

    values = np.asarray(values, dtype=float)
    target = np.asarray(target, dtype=float)

    diff = values - target

    exact = np.sum(np.isclose(values, target, atol=1e-12))

    mae = np.mean(np.abs(diff))

    max_error = np.max(np.abs(diff))

    print(
        f"{name:<45}"
        f"Exact: {exact:>8}/{len(target)} "
        f"({exact / len(target) * 100:>7.3f}%) "
        f"MAE: {mae:.6f} "
        f"MAX: {max_error:.3f}"
    )


# ============================================================
# LETTER TESTS
# ============================================================

print("\n" + "=" * 100)
print("LETTER COUNT TESTS")
print("=" * 100)

for rep_name, rep in representations.items():

    compare(
        rep_name + " - ASCII",
        rep.map(ascii_letters),
        target_letters
    )

    compare(
        rep_name + " - REGEX",
        rep.map(regex_letters),
        target_letters
    )

    compare(
        rep_name + " - UNICODE",
        rep.map(unicode_letters),
        target_letters
    )


# ============================================================
# DOMAIN-BASED TESTS
# ============================================================

print("\n" + "=" * 100)
print("DOMAIN-BASED LETTER TESTS")
print("=" * 100)


def get_hostname(url):

    try:
        return urlparse(url).hostname or ""
    except:
        return ""


def get_domain_without_www(url):

    host = get_hostname(url)

    if host.lower().startswith("www."):
        host = host[4:]

    return host


def get_domain_without_tld(url):

    host = get_domain_without_www(url)

    parts = host.split(".")

    if len(parts) >= 2:
        return ".".join(parts[:-1])

    return host


hostname_series = urls.map(get_hostname)

domain_no_www_series = urls.map(get_domain_without_www)

domain_without_tld_series = urls.map(get_domain_without_tld)


compare(
    "HOSTNAME ASCII",
    hostname_series.map(ascii_letters),
    target_letters
)

compare(
    "DOMAIN_NO_WWW ASCII",
    domain_no_www_series.map(ascii_letters),
    target_letters
)

compare(
    "DOMAIN_WITHOUT_TLD ASCII",
    domain_without_tld_series.map(ascii_letters),
    target_letters
)


# ============================================================
# DIFFERENCE ANALYSIS
# ============================================================

print("\n" + "=" * 100)
print("DIFFERENCE ANALYSIS")
print("=" * 100)


candidate = representations["NO_PROTOCOL_NO_WWW"].map(
    ascii_letters
).astype(float).values

difference = candidate - target_letters

unique, counts = np.unique(
    difference.astype(int),
    return_counts=True
)

print("\nNO_PROTOCOL_NO_WWW - DATASET LETTER COUNT")

for d, c in zip(unique, counts):

    print(
        f"Difference {d:>4}: "
        f"{c:>8} rows "
        f"({c / len(df) * 100:.4f}%)"
    )


# ============================================================
# SHOW MISMATCH EXAMPLES
# ============================================================

print("\n" + "=" * 100)
print("MISMATCH EXAMPLES")
print("=" * 100)

mismatch_indices = np.where(difference != 0)[0]

print(
    "\nTotal mismatches:",
    len(mismatch_indices)
)

for i in mismatch_indices[:50]:

    url = urls.iloc[i]

    print("\n" + "-" * 100)

    print("INDEX:", i)

    print("URL:", repr(url))

    print(
        "DATASET LETTERS:",
        df["NoOfLettersInURL"].iloc[i]
    )

    print(
        "NO_PROTOCOL_NO_WWW:",
        repr(representations["NO_PROTOCOL_NO_WWW"].iloc[i])
    )

    print(
        "CALCULATED:",
        int(candidate[i])
    )

    print(
        "DIFFERENCE:",
        int(difference[i])
    )

    print(
        "HOSTNAME:",
        repr(hostname_series.iloc[i])
    )

    print(
        "DOMAIN NO WWW:",
        repr(domain_no_www_series.iloc[i])
    )

    print(
        "DOMAIN WITHOUT TLD:",
        repr(domain_without_tld_series.iloc[i])
    )


# ============================================================
# CHAR CONTINUATION RATE
# ============================================================

print("\n" + "=" * 100)
print("CHAR CONTINUATION RATE")
print("=" * 100)


def char_class(c):

    if c.isalpha():
        return "LETTER"

    if c.isdigit():
        return "DIGIT"

    return "SPECIAL"


def char_continuation_rate(url):

    if len(url) == 0:
        return 0.0

    if len(url) == 1:
        return 1.0

    total = 0

    current_class = char_class(url[0])
    current_length = 1

    for c in url[1:]:

        new_class = char_class(c)

        if new_class == current_class:

            current_length += 1

        else:

            total += current_length

            current_class = new_class
            current_length = 1

    total += current_length

    return total / len(url)


target_continuation = (
    df["CharContinuationRate"]
    .astype(float)
    .values
)


for rep_name, rep in representations.items():

    compare(
        rep_name + " - CLASS RUNS",
        rep.map(char_continuation_rate),
        target_continuation
    )


# ============================================================
# SPECIAL CHARACTER TEST
# ============================================================

print("\n" + "=" * 100)
print("SPECIAL CHARACTER TEST")
print("=" * 100)


target_special = (
    df["NoOfOtherSpecialCharsInURL"]
    .astype(float)
    .values
)


def special_excluding_equals_q_amp(s):

    return sum(
        (not c.isalnum()) and c not in "=?&"
        for c in s
    )


candidate_special = representations[
    "NO_PROTOCOL_NO_WWW"
].map(
    special_excluding_equals_q_amp
)


compare(
    "NO_PROTOCOL_NO_WWW - EXCLUDE =?&",
    candidate_special,
    target_special
)


print("\n" + "=" * 100)
print("V2 DIAGNOSTIC COMPLETE")
print("=" * 100)
