import joblib
import numpy as np

from feature_extractor_v2 import extract_features, FEATURE_NAMES


MODEL_FILE = "final_random_forest_phishing_model_v2.pkl"


class PhishingPredictor:

    def __init__(self, model_file=MODEL_FILE):
        self.model_file = model_file

        package = joblib.load(model_file)

        self.model = package["model"]
        self.feature_names = package["feature_names"]
        self.feature_count = package["feature_count"]
        self.label_mapping = package["label_mapping"]

        if self.feature_names != FEATURE_NAMES:
            raise ValueError(
                "Feature mismatch between model and extractor."
            )

        if self.feature_count != len(FEATURE_NAMES):
            raise ValueError(
                "Feature count mismatch."
            )

        self.classes = list(self.model.classes_)

        if 0 not in self.classes or 1 not in self.classes:
            raise ValueError(
                f"Unexpected model classes: {self.classes}"
            )


    def predict(self, url):

        features = extract_features(url)

        X = np.array(
            features,
            dtype=float
        ).reshape(1, -1)

        probabilities = self.model.predict_proba(X)[0]

        phishing_index = self.classes.index(0)
        legitimate_index = self.classes.index(1)

        phishing_probability = float(
            probabilities[phishing_index]
        )

        legitimate_probability = float(
            probabilities[legitimate_index]
        )

        if phishing_probability >= legitimate_probability:
            prediction = 0
            result = "Phishing"
        else:
            prediction = 1
            result = "Legitimate"

        if phishing_probability >= 0.70:
            risk_level = "HIGH RISK"
        elif phishing_probability >= 0.30:
            risk_level = "MEDIUM RISK"
        else:
            risk_level = "LOW RISK"

        return {
            "url": url,
            "prediction": prediction,
            "result": result,
            "risk_level": risk_level,

            "phishing_probability": phishing_probability,
            "legitimate_probability": legitimate_probability,

            "phishing_percentage": phishing_probability * 100,
            "legitimate_percentage": legitimate_probability * 100,

            "features": dict(
                zip(FEATURE_NAMES, features)
            )
        }


if __name__ == "__main__":

    print("=" * 70)
    print("PHISHING PREDICTOR V2 TEST")
    print("=" * 70)

    predictor = PhishingPredictor()

    test_urls = [
        "https://www.google.com",
        "https://secure-login-account-verify.example.com/login"
    ]

    for url in test_urls:

        print()
        print("-" * 70)
        print("URL:", url)
        print("-" * 70)

        result = predictor.predict(url)

        print("Prediction:", result["prediction"])
        print("Result:", result["result"])
        print("Risk:", result["risk_level"])

        print(
            f"Phishing Probability: "
            f"{result['phishing_percentage']:.2f}%"
        )

        print(
            f"Legitimate Probability: "
            f"{result['legitimate_percentage']:.2f}%"
        )

        print()
        print("Features:")

        for name, value in result["features"].items():
            print(f"  {name}: {value}")

    print()
    print("=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)