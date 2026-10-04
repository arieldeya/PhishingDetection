from urllib.parse import urlparse


FEATURE_NAMES = [
    "URLLength",
    "DomainLength",
    "TLDLength",
    "NoOfSubDomain",
    "NoOfLettersInURL",
    "LetterRatioInURL",
    "NoOfDegitsInURL",
    "DegitRatioInURL",
    "NoOfOtherSpecialCharsInURL",
    "SpacialCharRatioInURL"
]


def get_domain(url):
    """
    Extract hostname/domain from a URL.
    """
    url = str(url).strip()

    if not url.startswith(("http://", "https://")):
        url_for_parse = "http://" + url
    else:
        url_for_parse = url

    try:
        parsed = urlparse(url_for_parse)
        return parsed.hostname or ""
    except Exception:
        return ""


def get_tld(domain):
    """
    Extract the final domain component.

    Example:
        example.com -> com
        example.co.ke -> ke
    """
    domain = str(domain).strip().lower()

    if not domain:
        return ""

    parts = domain.split(".")

    if len(parts) < 2:
        return ""

    return parts[-1]


def get_subdomain_count(domain):
    """
    Count subdomain components.

    Examples:
        example.com
            -> 0

        www.example.com
            -> 1

        mail.shop.example.com
            -> 2
    """
    domain = str(domain).strip().lower()

    if not domain:
        return 0

    parts = domain.split(".")

    if len(parts) <= 2:
        return 0

    return len(parts) - 2


def count_letters(url):
    """
    Count alphabetic characters.
    """
    return sum(c.isalpha() for c in str(url))


def count_digits(url):
    """
    Count numeric characters.
    """
    return sum(c.isdigit() for c in str(url))


def count_other_special_chars(url):
    """
    Count special characters while excluding:
        =
        ?
        &

    This is the feature representation used during V2 training.
    """
    url = str(url)

    return sum(
        not c.isalnum() and c not in "=?&"
        for c in url
    )


def extract_features(url):
    """
    Extract the exact 10 features used by Random Forest V2.

    Returns:
        list of 10 numeric values in the correct model order.
    """

    url = str(url).strip()

    # --------------------------------------------------------
    # Basic URL information
    # --------------------------------------------------------

    url_length = len(url)

    domain = get_domain(url)

    domain_length = len(domain)

    tld = get_tld(domain)

    tld_length = len(tld)

    subdomain_count = get_subdomain_count(domain)

    # --------------------------------------------------------
    # Character counts
    # --------------------------------------------------------

    letters = count_letters(url)

    digits = count_digits(url)

    special_chars = count_other_special_chars(url)

    # --------------------------------------------------------
    # Ratios
    # --------------------------------------------------------

    if url_length > 0:

        letter_ratio = letters / url_length

        digit_ratio = digits / url_length

        special_ratio = special_chars / url_length

    else:

        letter_ratio = 0.0

        digit_ratio = 0.0

        special_ratio = 0.0

    # --------------------------------------------------------
    # Return features in EXACT model order
    # --------------------------------------------------------

    return [
        url_length,
        domain_length,
        tld_length,
        subdomain_count,
        letters,
        letter_ratio,
        digits,
        digit_ratio,
        special_chars,
        special_ratio
    ]


def extract_features_dict(url):
    """
    Return features as a dictionary.

    Useful for displaying/debugging predictions.
    """

    values = extract_features(url)

    return dict(
        zip(FEATURE_NAMES, values)
    )


if __name__ == "__main__":

    print("=" * 80)
    print("FEATURE EXTRACTOR V2 TEST")
    print("=" * 80)

    test_urls = [
        "https://www.google.com",
        "https://www.example.com/login",
        "http://192.168.1.1/login",
        "https://paypal-security-login.example.com",
        "https://example.com/user/login?id=12345",
        "https://mail.shop.example.co.ke"
    ]

    for url in test_urls:

        print("\n" + "-" * 80)

        print("URL:")
        print(url)

        print("\nFeatures:")

        features = extract_features_dict(url)

        for name, value in features.items():

            print(
                f"{name:35} {value}"
            )

    print("\n" + "=" * 80)
    print("FEATURE EXTRACTOR TEST COMPLETE")
    print("=" * 80)