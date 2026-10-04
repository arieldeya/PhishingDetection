import csv
from phishing_predictor_v2 import PhishingPredictor


TEST_URLS = [
    # Legitimate websites
    "https://www.google.com",
    "https://www.microsoft.com",
    "https://www.github.com",
    "https://www.wikipedia.org",
    "https://www.amazon.com",

    # Synthetic phishing-style URLs
    "https://paypal-login-verify-account.example.com/login",
    "https://secure-login123-example.com/account",
    "https://bank-account-verify-security.example.com/login",
    "https://update-your-account-security.example.com/verify",
    "https://confirm-payment-information.example.com/login",

    # Complex URLs
    "https://example.com/account/security/verify/login/password/reset/confirmation",
    "https://example.com/user/account/profile/settings/security",
]


def main():

    print("=" * 100)
    print("PHISHGUARD V2 - AUTOMATED MODEL VALIDATION")
    print("=" * 100)
    print()

    predictor = PhishingPredictor()

    results = []

    for number, url in enumerate(TEST_URLS, start=1):

        print("-" * 100)
        print(f"TEST {number}/{len(TEST_URLS)}")
        print(f"URL: {url}")
        print("-" * 100)

        try:
            result = predictor.predict(url)

            prediction = result["prediction"]
            classification = result["result"]
            risk_level = result["risk_level"]

            phishing_percentage = result["phishing_percentage"]
            legitimate_percentage = result["legitimate_percentage"]

            print(f"Prediction:              {prediction}")
            print(f"Classification:          {classification}")
            print(f"Phishing Probability:    {phishing_percentage:.2f}%")
            print(f"Legitimate Probability:  {legitimate_percentage:.2f}%")
            print(f"Risk Level:              {risk_level}")
            print()

            # Verify that the two probabilities approximately add to 100%.
            probability_total = (
                phishing_percentage + legitimate_percentage
            )

            if abs(probability_total - 100.0) < 0.01:
                consistency = "PASS"
            else:
                consistency = "CHECK"

            print(
                f"Probability Consistency: {consistency} "
                f"({probability_total:.2f}%)"
            )
            print()

            results.append({
                "URL": url,
                "Prediction": prediction,
                "Classification": classification,
                "Phishing Probability": round(
                    phishing_percentage, 2
                ),
                "Legitimate Probability": round(
                    legitimate_percentage, 2
                ),
                "Risk Level": risk_level,
                "Probability Consistency": consistency,
                "Status": "SUCCESS"
            })

        except Exception as error:

            print(f"ERROR: {error}")
            print()

            results.append({
                "URL": url,
                "Prediction": "",
                "Classification": "",
                "Phishing Probability": "",
                "Legitimate Probability": "",
                "Risk Level": "",
                "Probability Consistency": "",
                "Status": f"ERROR: {error}"
            })

    print("=" * 100)
    print("VALIDATION SUMMARY")
    print("=" * 100)

    successful = sum(
        1
        for result in results
        if result["Status"] == "SUCCESS"
    )

    failed = len(results) - successful

    phishing_count = sum(
        1
        for result in results
        if result["Classification"] == "Phishing"
    )

    legitimate_count = sum(
        1
        for result in results
        if result["Classification"] == "Legitimate"
    )

    consistency_pass = sum(
        1
        for result in results
        if result["Probability Consistency"] == "PASS"
    )

    print(f"Total URLs tested:       {len(results)}")
    print(f"Successful predictions:  {successful}")
    print(f"Failed predictions:      {failed}")
    print(f"Classified phishing:     {phishing_count}")
    print(f"Classified legitimate:   {legitimate_count}")
    print(f"Probability checks PASS: {consistency_pass}")
    print()

    output_file = "phishing_model_validation_results.csv"

    fieldnames = [
        "URL",
        "Prediction",
        "Classification",
        "Phishing Probability",
        "Legitimate Probability",
        "Risk Level",
        "Probability Consistency",
        "Status"
    ]

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(results)

    print(f"Results saved to: {output_file}")
    print("=" * 100)
    print("VALIDATION COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()