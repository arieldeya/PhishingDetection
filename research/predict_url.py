import sys
import joblib
import pandas as pd
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================

DATASET = "PhiUSIIL_Phishing_URL_Dataset.csv"
MODEL = "final_random_forest_phishing_model.pkl"

THRESHOLD = 0.58

PHISHING_LABEL = 0
LEGITIMATE_LABEL = 1

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
# LOAD MODEL
# ============================================================

model = joblib.load(MODEL)

print("=" * 60)
print("PHISHING URL DETECTOR")
print("=" * 60)


# ============================================================
# GET URL
# ============================================================

if len(sys.argv) < 2:
    print()
    print("Usage:")
    print('python predict_url.py "https://example.com"')
    print()
    sys.exit(1)

url = sys.argv[1].strip()

print()
print(f"URL: {url}")


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(DATASET)

# Look for exact URL
match = df[df["URL"].astype(str).str.strip() == url]


# ============================================================
# CHECK WHETHER URL EXISTS
# ============================================================

if match.empty:

    print()
    print("STATUS: URL NOT FOUND IN PHIUSIIL DATASET")
    print()
    print("This first version only predicts URLs that already")
    print("exist in the original PhiUSIIL dataset.")
    print()
    print("We will add true unseen-URL feature extraction next.")
    print("=" * 60)

    sys.exit(0)


# ============================================================
# GET FEATURES
# ============================================================

row = match.iloc[0]

X = pd.DataFrame(
    [[row[feature] for feature in FEATURES]],
    columns=FEATURES
)


# ============================================================
# PREDICT PROBABILITIES
# ============================================================

probabilities = model.predict_proba(X)[0]

classes = model.classes_

# Find correct probability columns dynamically
phishing_index = np.where(classes == PHISHING_LABEL)[0][0]
legitimate_index = np.where(classes == LEGITIMATE_LABEL)[0][0]

phishing_probability = probabilities[phishing_index]
legitimate_probability = probabilities[legitimate_index]


# ============================================================
# APPLY THRESHOLD
# ============================================================

if phishing_probability >= THRESHOLD:

    prediction = "PHISHING"

else:

    prediction = "LEGITIMATE"


# ============================================================
# DISPLAY RESULT
# ============================================================

print()
print("-" * 60)

print(f"Prediction: {prediction}")

print(
    f"Phishing probability: "
    f"{phishing_probability * 100:.2f}%"
)

print(
    f"Legitimate probability: "
    f"{legitimate_probability * 100:.2f}%"
)

print(
    f"Threshold: "
    f"{THRESHOLD * 100:.2f}%"
)

print("-" * 60)

print()
print("Model classes:", classes)
print("Label mapping:")
print("0 = PHISHING")
print("1 = LEGITIMATE")

print()
print("=" * 60)