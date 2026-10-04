import joblib
import numpy as np

from feature_extractor_v2 import (
    extract_features,
    extract_features_dict,
    FEATURE_NAMES
)


MODEL_FILE = "final_random_forest_phishing_model_v2.pkl"


print("=" * 80)
print("PHISHING URL PREDICTOR V2")
print("=" * 80)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading model...")

package = joblib.load(MODEL_FILE)

model = package["model"]

saved_features = package["feature_names"]


print("Model loaded successfully.")

print("\nModel features:")

for i, feature in enumerate(saved_features):

    print(
        f"{i}: {feature}"
    )


# ============================================================
# VERIFY FEATURE ORDER
# ============================================================

if saved_features != FEATURE_NAMES:

    raise ValueError(
        "FEATURE ORDER MISMATCH!\n"
        f"Model: {saved_features}\n"
        f"Extractor: {FEATURE_NAMES}"
    )


print("\nFeature order verified successfully.")


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_url(url):

    features = extract_features(url)

    X = np.array(
        features,
        dtype=float
    ).reshape(1, -1)

    prediction = model.predict(X)[0]

    probabilities = model.predict_proba(X)[0]

    phishing_probability = probabilities[0]

    legitimate_probability = probabilities[1]

    return (
        prediction,
        phishing_probability,
        legitimate_probability,
        features
    )


# ============================================================
# TEST URLS
# ============================================================

test_urls = [

    "https://www.google.com",

    "https://www.microsoft.com",

    "https://www.github.com",

    "https://www.wikipedia.org",

    "https://www.example.com",

    "https://paypal-security-login.example.com",

    "http://192.168.1.1/login",

    "https://secure-login-account-verify.example.com/login",

    "https://example.com/login?id=12345",

    "https://mail.shop.example.co.ke"
]


# ============================================================
# RUN TESTS
# ============================================================

for url in test_urls:

    print("\n")
    print("=" * 80)

    print("URL:")
    print(url)

    print("=" * 80)

    (
        prediction,
        phishing_probability,
        legitimate_probability,
        features
    ) = predict_url(url)


    print("\nFEATURES:")

    for name, value in zip(
        FEATURE_NAMES,
        features
    ):

        print(
            f"{name:35} {value}"
        )


    print("\nPREDICTION:")

    if prediction == 0:

        print("⚠️ PHISHING")

    else:

        print("✓ LEGITIMATE")


    print("\nPROBABILITY:")

    print(
        f"Phishing:   {phishing_probability * 100:.2f}%"
    )

    print(
        f"Legitimate: {legitimate_probability * 100:.2f}%"
    )


print("\n")
print("=" * 80)
print("PREDICTION TEST COMPLETE")
print("=" * 80)