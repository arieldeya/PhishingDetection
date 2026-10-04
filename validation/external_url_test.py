from phishing_predictor_v2 import PhishingPredictor


print("=" * 100)
print("PHISHGUARD V2 - EXTERNAL URL TEST")
print("=" * 100)
print()

predictor = PhishingPredictor()


# ============================================================
# TEST URLS
# ============================================================

test_urls = [

    # --------------------------------------------------------
    # Known legitimate websites
    # --------------------------------------------------------

    ("LEGITIMATE", "https://www.google.com"),
    ("LEGITIMATE", "https://www.microsoft.com"),
    ("LEGITIMATE", "https://www.github.com"),
    ("LEGITIMATE", "https://www.wikipedia.org"),
    ("LEGITIMATE", "https://www.amazon.com"),
    ("LEGITIMATE", "https://www.apple.com"),
    ("LEGITIMATE", "https://www.mozilla.org"),
    ("LEGITIMATE", "https://www.python.org"),
    ("LEGITIMATE", "https://www.oracle.com"),
    ("LEGITIMATE", "https://www.ibm.com"),

    # --------------------------------------------------------
    # Synthetic phishing-style URLs
    # --------------------------------------------------------
    # These are intentionally constructed examples.
    # They are NOT claimed to be real phishing websites.
    # --------------------------------------------------------

    ("PHISHING_STYLE",
     "https://secure-login-account-verify.example.com/login"),

    ("PHISHING_STYLE",
     "https://paypal-security-check.example.com/verify"),

    ("PHISHING_STYLE",
     "https://bank-account-confirmation.example.com/login"),

    ("PHISHING_STYLE",
     "https://update-payment-information.example.com/secure"),

    ("PHISHING_STYLE",
     "https://account-verification-security.example.com/password"),

    ("PHISHING_STYLE",
     "https://confirm-your-billing-information.example.com/login"),

    ("PHISHING_STYLE",
     "https://secure-wallet-login.example.com/verify/account"),

    ("PHISHING_STYLE",
     "https://user-account-security-check.example.com/login"),

    ("PHISHING_STYLE",
     "https://signin-confirm-password.example.com/account"),

    ("PHISHING_STYLE",
     "https://verify-bank-account-information.example.com/security"),
]


# ============================================================
# RUN TESTS
# ============================================================

results = []

for number, (expected_type, url) in enumerate(test_urls, start=1):

    print("=" * 100)
    print(f"TEST {number}/{len(test_urls)}")
    print("=" * 100)

    print(f"Expected type: {expected_type}")
    print(f"URL: {url}")
    print()

    try:

        result = predictor.predict(url)

        prediction = result["prediction"]
        classification = result["result"]
        risk = result["risk_level"]

        phishing_probability = result["phishing_probability"]
        legitimate_probability = result["legitimate_probability"]

        phishing_percentage = result["phishing_percentage"]
        legitimate_percentage = result["legitimate_percentage"]

        print(f"Prediction:              {prediction}")
        print(f"Classification:          {classification}")
        print(f"Risk Level:              {risk}")
        print(
            f"Phishing Probability:    "
            f"{phishing_percentage:.2f}%"
        )
        print(
            f"Legitimate Probability:  "
            f"{legitimate_percentage:.2f}%"
        )

        probability_total = (
            phishing_probability +
            legitimate_probability
        )

        print(
            f"Probability Total:       "
            f"{probability_total:.6f}"
        )

        probability_pass = abs(
            probability_total - 1.0
        ) < 0.000001

        if probability_pass:
            print("Probability Check:       PASS")
        else:
            print("Probability Check:       FAIL")

        results.append({
            "expected_type": expected_type,
            "url": url,
            "classification": classification,
            "risk": risk,
            "phishing_percentage": phishing_percentage,
            "legitimate_percentage": legitimate_percentage,
            "probability_check": probability_pass
        })

    except Exception as e:

        print("ERROR")
        print(str(e))

        results.append({
            "expected_type": expected_type,
            "url": url,
            "classification": "ERROR",
            "risk": "ERROR",
            "phishing_percentage": 0,
            "legitimate_percentage": 0,
            "probability_check": False
        })

    print()


# ============================================================
# SUMMARY
# ============================================================

print("=" * 100)
print("EXTERNAL TEST SUMMARY")
print("=" * 100)
print()

total = len(results)

successful = sum(
    1
    for r in results
    if r["classification"] != "ERROR"
)

probability_passes = sum(
    1
    for r in results
    if r["probability_check"]
)

legitimate_expected = [
    r for r in results
    if r["expected_type"] == "LEGITIMATE"
]

phishing_expected = [
    r for r in results
    if r["expected_type"] == "PHISHING_STYLE"
]

legitimate_correct = sum(
    1
    for r in legitimate_expected
    if r["classification"] == "Legitimate"
)

phishing_correct = sum(
    1
    for r in phishing_expected
    if r["classification"] == "Phishing"
)


print(f"Total URLs tested:              {total}")
print(f"Successful predictions:         {successful}")
print(f"Probability checks passed:      {probability_passes}")
print()

print(
    f"Legitimate URLs correctly classified: "
    f"{legitimate_correct}/{len(legitimate_expected)}"
)

print(
    f"Phishing-style URLs classified:        "
    f"{phishing_correct}/{len(phishing_expected)}"
)

print()

if successful == total:
    print("FUNCTIONAL TEST: PASS")
else:
    print("FUNCTIONAL TEST: FAIL")

if probability_passes == total:
    print("PROBABILITY CONSISTENCY: PASS")
else:
    print("PROBABILITY CONSISTENCY: FAIL")

print()
print("=" * 100)
print("EXTERNAL URL TEST COMPLETE")
print("=" * 100)