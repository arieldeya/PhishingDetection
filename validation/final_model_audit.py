import os
import joblib

from phishing_predictor_v2 import PhishingPredictor
from feature_extractor_v2 import FEATURE_NAMES, extract_features


MODEL_FILE = "final_random_forest_phishing_model_v2.pkl"
TEST_URL = "https://www.google.com"


print("=" * 100)
print("PHISHGUARD V2 - FINAL PRODUCTION MODEL AUDIT")
print("=" * 100)
print()


# ============================================================
# 1. MODEL FILE
# ============================================================

print("1. MODEL FILE")
print("-" * 100)

if os.path.exists(MODEL_FILE):
    print("Model file: FOUND")
else:
    print("Model file: NOT FOUND")
    raise FileNotFoundError(MODEL_FILE)

file_size = os.path.getsize(MODEL_FILE)

print(f"Model file size: {file_size:,} bytes")
print()


# ============================================================
# 2. LOAD MODEL
# ============================================================

print("2. LOADING MODEL")
print("-" * 100)

package = joblib.load(MODEL_FILE)

print(f"Package type: {type(package)}")

if not isinstance(package, dict):
    raise TypeError(
        "Expected the saved model package to be a dictionary."
    )

print(
    "Package keys:",
    list(package.keys())
)

print()


# ============================================================
# 3. MODEL OBJECT
# ============================================================

print("3. MODEL OBJECT")
print("-" * 100)

model = package["model"]

print(f"Model type: {type(model).__name__}")

print(
    f"Number of trees: "
    f"{model.n_estimators}"
)

print(
    f"Model classes: "
    f"{model.classes_}"
)

print(
    f"Model feature count: "
    f"{model.n_features_in_}"
)

print()


# ============================================================
# 4. FEATURE NAMES
# ============================================================

print("4. FEATURE NAMES")
print("-" * 100)

saved_features = package["feature_names"]

print("Saved feature names:")

for index, feature in enumerate(saved_features):

    print(
        f"{index:2d}. {feature}"
    )

print()

if saved_features == FEATURE_NAMES:
    print("Feature names: PASS")
else:
    print("Feature names: FAIL")
    print("Saved:")
    print(saved_features)
    print()
    print("Extractor:")
    print(FEATURE_NAMES)

print()


# ============================================================
# 5. FEATURE COUNT
# ============================================================

print("5. FEATURE COUNT")
print("-" * 100)

if len(saved_features) == 10:
    print("Feature count: PASS")
else:
    print("Feature count: FAIL")

if model.n_features_in_ == len(FEATURE_NAMES):
    print("Model/extractor feature count: PASS")
else:
    print("Model/extractor feature count: FAIL")

print()


# ============================================================
# 6. LABEL MAPPING
# ============================================================

print("6. LABEL MAPPING")
print("-" * 100)

label_mapping = package["label_mapping"]

print("Saved label mapping:")

for label, meaning in label_mapping.items():

    print(
        f"{label} = {meaning}"
    )

print()

if (
    label_mapping.get(0) == "Phishing"
    and
    label_mapping.get(1) == "Legitimate"
):

    print("Label mapping: PASS")

else:

    print("Label mapping: FAIL")

print()


# ============================================================
# 7. FEATURE EXTRACTION TEST
# ============================================================

print("7. FEATURE EXTRACTION")
print("-" * 100)

features = extract_features(TEST_URL)

print(f"Test URL: {TEST_URL}")
print()

for name, value in zip(
    FEATURE_NAMES,
    features
):

    print(
        f"{name:35s}: {value}"
    )

print()

if len(features) == 10:
    print("Feature extraction: PASS")
else:
    print("Feature extraction: FAIL")

print()


# ============================================================
# 8. DIRECT MODEL PREDICTION
# ============================================================

print("8. DIRECT MODEL PREDICTION")
print("-" * 100)

prediction = model.predict(
    [features]
)[0]

probabilities = model.predict_proba(
    [features]
)[0]

print(
    f"Prediction: {prediction}"
)

print(
    f"Classes: {model.classes_}"
)

print(
    f"Probabilities: {probabilities}"
)

print()


# ============================================================
# 9. PROBABILITY CONSISTENCY
# ============================================================

print("9. PROBABILITY CONSISTENCY")
print("-" * 100)

probability_total = probabilities.sum()

print(
    f"Probability total: "
    f"{probability_total:.12f}"
)

if abs(probability_total - 1.0) < 0.000001:
    print("Probability consistency: PASS")
else:
    print("Probability consistency: FAIL")

print()


# ============================================================
# 10. PRODUCTION PREDICTOR
# ============================================================

print("10. PRODUCTION PREDICTOR")
print("-" * 100)

predictor = PhishingPredictor()

result = predictor.predict(TEST_URL)

print(
    f"Classification: "
    f"{result['result']}"
)

print(
    f"Prediction: "
    f"{result['prediction']}"
)

print(
    f"Risk level: "
    f"{result['risk_level']}"
)

print(
    f"Phishing probability: "
    f"{result['phishing_percentage']:.4f}%"
)

print(
    f"Legitimate probability: "
    f"{result['legitimate_percentage']:.4f}%"
)

print()


# ============================================================
# 11. CONSISTENCY CHECK
# ============================================================

print("11. PRODUCTION CONSISTENCY")
print("-" * 100)

predictor_prediction = result["prediction"]

if predictor_prediction == prediction:

    print(
        "Predictor vs direct model prediction: PASS"
    )

else:

    print(
        "Predictor vs direct model prediction: FAIL"
    )

print()


# ============================================================
# 12. FINAL SUMMARY
# ============================================================

print("=" * 100)
print("FINAL MODEL AUDIT SUMMARY")
print("=" * 100)
print()

checks = {
    "Model file exists": os.path.exists(MODEL_FILE),
    "Package is dictionary": isinstance(package, dict),
    "Feature names match": saved_features == FEATURE_NAMES,
    "10 features": len(saved_features) == 10,
    "Model has 10 features": model.n_features_in_ == 10,
    "Correct label mapping": (
        label_mapping.get(0) == "Phishing"
        and
        label_mapping.get(1) == "Legitimate"
    ),
    "Feature extraction works": len(features) == 10,
    "Probability sums to 1": (
        abs(probability_total - 1.0) < 0.000001
    ),
    "Predictor matches model": (
        predictor_prediction == prediction
    )
}

passed = 0

for name, status in checks.items():

    print(
        f"{name:35s}: "
        f"{'PASS' if status else 'FAIL'}"
    )

    if status:
        passed += 1

print()

print(
    f"Checks passed: "
    f"{passed}/{len(checks)}"
)

print()

if passed == len(checks):

    print(
        "FINAL MODEL AUDIT: PASS"
    )

else:

    print(
        "FINAL MODEL AUDIT: REVIEW REQUIRED"
    )

print()

print("=" * 100)
print("FINAL MODEL AUDIT COMPLETE")
print("=" * 100)