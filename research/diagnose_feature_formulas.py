import pandas as pd
import re
from urllib.parse import urlparse

DATASET = "PhiUSIIL_Phishing_URL_Dataset.csv"

df = pd.read_csv(DATASET)

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


def analyze(url):

    print()
    print("=" * 100)
    print("URL")
    print(url)
    print("=" * 100)

    parsed = urlparse(url)

    # --------------------------------------------------------
    # Different possible URL representations
    # --------------------------------------------------------

    raw = url

    no_protocol = re.sub(
        r"^https?://",
        "",
        url,
        flags=re.IGNORECASE
    )

    hostname = parsed.hostname or ""

    domain = hostname

    no_www = domain

    if no_www.lower().startswith("www."):
        no_www = no_www[4:]

    path = parsed.path or ""

    query = parsed.query or ""

    # --------------------------------------------------------
    # Display lengths
    # --------------------------------------------------------

    print()
    print("RAW URL:")
    print(repr(raw))
    print("Length:", len(raw))

    print()
    print("WITHOUT PROTOCOL:")
    print(repr(no_protocol))
    print("Length:", len(no_protocol))

    print()
    print("HOSTNAME:")
    print(repr(hostname))
    print("Length:", len(hostname))

    print()
    print("DOMAIN WITHOUT WWW:")
    print(repr(no_www))
    print("Length:", len(no_www))

    print()
    print("PATH:")
    print(repr(path))
    print("Length:", len(path))

    print()
    print("QUERY:")
    print(repr(query))
    print("Length:", len(query))

    # --------------------------------------------------------
    # Character counts
    # --------------------------------------------------------

    candidates = {
        "RAW": raw,
        "NO_PROTOCOL": no_protocol,
        "HOSTNAME": hostname,
        "NO_WWW": no_www,
    }

    print()
    print("CHARACTER COUNTS")
    print("-" * 100)

    for name, value in candidates.items():

        letters = sum(c.isalpha() for c in value)
        digits = sum(c.isdigit() for c in value)
        special = sum(not c.isalnum() for c in value)

        print()
        print(name)
        print("  length   :", len(value))
        print("  letters  :", letters)
        print("  digits   :", digits)
        print("  special  :", special)

    # --------------------------------------------------------
    # Dataset row
    # --------------------------------------------------------

    match = df[
        df["URL"].astype(str).str.strip() == url.strip()
    ]

    if match.empty:

        print()
        print("URL NOT FOUND IN DATASET")

        return

    row = match.iloc[0]

    print()
    print("PHIUSIIL DATASET VALUES")
    print("-" * 100)

    for feature in FEATURES:

        print(
            f"{feature:35} = {row[feature]}"
        )


# ============================================================
# SELECT REPRESENTATIVE URLs
# ============================================================

urls = [
    "https://www.texascooking.com",
    "https://www.sakuramobile.jp",
    "http://www.bibox365.us",
    "https://digitale-drehtuer.de/red",
    "http://www.4up4.com",
    "https://www.northcm.ac.th",
    "https://www.woolworthsrewards.com.au",
    "https://www.kubiena-kochblog.com",
    "https://couchjumper.com/covantage/index.html",
    "https://s1madyjv.web.app/"
]


for url in urls:

    analyze(url)


print()
print("=" * 100)
print("DIAGNOSTIC COMPLETE")
print("=" * 100)