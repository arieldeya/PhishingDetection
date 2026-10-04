import pandas as pd
import numpy as np
import re
import joblib

from urllib.parse import urlparse

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# CONFIGURATION
# ============================================================

DATASET = "PhiUSIIL_Phishing_URL_Dataset.csv"
MODEL_FILE = "final_random_forest_phishing_model_v2.pkl"

RANDOM_STATE = 42


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 80)
print("PHISHING DETECTION MODEL V2")
print("=" * 80)

print("\nLoading dataset...")

df = pd.read_csv(DATASET)

print(f"Dataset shape: {df.shape}")
print(f"Total rows: {len(df):,}")


# ============================================================
# CHECK LABELS
# ============================================================

print("\nLabel distribution:")
print(df["label"].value_counts())

print("\nLabel meaning:")
print("0 = Phishing")
print("1 = Legitimate")


# ============================================================
# FEATURE FUNCTIONS
# ============================================================

def get_domain(url):
    """
    Extract hostname/domain from URL.
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
    Extract TLD from domain.
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
    Estimate number of subdomains.

    Example:
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
    Count special characters excluding:
        =
        ?
        &

    This corresponds to the feature we previously found
    to be the closest reproducible representation.
    """
    url = str(url)

    return sum(
        not c.isalnum() and c not in "=?&"
        for c in url
    )


def extract_features(url):
    """
    Extract the 10 deployment-safe URL features.
    """

    url = str(url).strip()

    # --------------------------------------------------------
    # Basic values
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


# ============================================================
# FEATURE NAMES
# ============================================================

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


# ============================================================
# TEST FEATURE EXTRACTION
# ============================================================

print("\nTesting feature extraction...")

test_urls = [
    "https://www.google.com",
    "https://www.example.com/login",
    "http://192.168.1.1/login",
    "https://paypal-security-login.example.com"
]

for test_url in test_urls:

    features = extract_features(test_url)

    print("\nURL:")
    print(test_url)

    for name, value in zip(FEATURE_NAMES, features):
        print(f"{name:30} {value}")


# ============================================================
# EXTRACT FEATURES FROM ENTIRE DATASET
# ============================================================

print("\n" + "=" * 80)
print("EXTRACTING FEATURES FROM DATASET")
print("=" * 80)

X = np.array(
    [extract_features(url) for url in df["URL"]],
    dtype=float
)

y = df["label"].astype(int).values


print(f"\nFeature matrix shape: {X.shape}")
print(f"Target shape: {y.shape}")


# ============================================================
# CREATE DATAFRAME FOR INSPECTION
# ============================================================

X_df = pd.DataFrame(
    X,
    columns=FEATURE_NAMES
)

print("\nExtracted feature sample:")
print(X_df.head(10).to_string())


# ============================================================
# CHECK FOR INVALID VALUES
# ============================================================

print("\nChecking feature matrix...")

print(
    "NaN values:",
    int(np.isnan(X).sum())
)

print(
    "Infinite values:",
    int(np.isinf(X).sum())
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

print("\n" + "=" * 80)
print("CREATING TRAIN / TEST SPLIT")
print("=" * 80)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"\nTraining samples: {len(X_train):,}")
print(f"Testing samples:  {len(X_test):,}")


# ============================================================
# TRAIN RANDOM FOREST
# ============================================================

print("\n" + "=" * 80)
print("TRAINING RANDOM FOREST")
print("=" * 80)

model = RandomForestClassifier(
    n_estimators=300,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    class_weight=None
)

print("\nTraining model...")

model.fit(X_train, y_train)

print("MODEL TRAINED SUCCESSFULLY!")


# ============================================================
# PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

y_pred = model.predict(X_test)

y_probability = model.predict_proba(X_test)


# ============================================================
# EVALUATION
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    pos_label=0
)

recall = recall_score(
    y_test,
    y_pred,
    pos_label=0
)

f1 = f1_score(
    y_test,
    y_pred,
    pos_label=0
)


print("\n" + "=" * 80)
print("MODEL PERFORMANCE")
print("=" * 80)

print(f"\nAccuracy:  {accuracy:.6f}")
print(f"Precision: {precision:.6f}")
print(f"Recall:    {recall:.6f}")
print(f"F1 Score:  {f1:.6f}")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 80)
print("CLASSIFICATION REPORT")
print("=" * 80)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Phishing",
            "Legitimate"
        ]
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, y_pred)

print("\n" + "=" * 80)
print("CONFUSION MATRIX")
print("=" * 80)

print("\nRows = Actual")
print("Columns = Predicted")

print("\n              Predicted")
print("             Phishing  Legitimate")
print(
    f"Phishing     {cm[0,0]:8d} {cm[0,1]:11d}"
)
print(
    f"Legitimate   {cm[1,0]:8d} {cm[1,1]:11d}"
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 80)
print("FEATURE IMPORTANCE")
print("=" * 80)

importance = pd.DataFrame({
    "Feature": FEATURE_NAMES,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

for _, row in importance.iterrows():
    print(
        f"{row['Feature']:35} "
        f"{row['Importance']:.6f}"
    )


# ============================================================
# SAVE MODEL
# ============================================================

print("\n" + "=" * 80)
print("SAVING MODEL")
print("=" * 80)

model_package = {
    "model": model,
    "feature_names": FEATURE_NAMES,
    "feature_count": len(FEATURE_NAMES),
    "label_mapping": {
        0: "Phishing",
        1: "Legitimate"
    }
}

joblib.dump(
    model_package,
    MODEL_FILE
)

print(
    f"\nMODEL SAVED SUCCESSFULLY:"
)

print(MODEL_FILE)


# ============================================================
# VERIFY SAVED MODEL
# ============================================================

print("\nVerifying saved model...")

loaded_package = joblib.load(MODEL_FILE)

loaded_model = loaded_package["model"]

print(
    "Loaded model type:",
    type(loaded_model)
)

print(
    "Number of features:",
    loaded_package["feature_count"]
)

print(
    "Feature names:"
)

for i, feature in enumerate(
    loaded_package["feature_names"]
):
    print(f"{i}: {feature}")


# ============================================================
# FINAL TEST
# ============================================================

print("\n" + "=" * 80)
print("LIVE-STYLE PREDICTION TEST")
print("=" * 80)

sample_url = "https://www.google.com"

sample_features = np.array(
    extract_features(sample_url)
).reshape(1, -1)

prediction = loaded_model.predict(
    sample_features
)[0]

probabilities = loaded_model.predict_proba(
    sample_features
)[0]

print("\nTest URL:")
print(sample_url)

print("\nPrediction:")

if prediction == 0:
    print("PHISHING")
else:
    print("LEGITIMATE")

print("\nProbabilities:")

print(
    f"Phishing:   {probabilities[0]:.6f}"
)

print(
    f"Legitimate: {probabilities[1]:.6f}"
)


print("\n" + "=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)