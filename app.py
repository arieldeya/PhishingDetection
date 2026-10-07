from flask import Flask, render_template, request
from urllib.parse import urlparse

from phishing_predictor_v2 import PhishingPredictor

app = Flask(__name__)
predictor = PhishingPredictor()


def is_valid_url(url):
    url = str(url).strip()

    if not url:
        return False

    if not url.startswith(("http://", "https://")):
        url_to_parse = "http://" + url
    else:
        url_to_parse = url

    try:
        parsed = urlparse(url_to_parse)
    except Exception:
        return False

    hostname = parsed.hostname

    if not hostname:
        return False

    if any(character.isspace() for character in url):
        return False

    if "." not in hostname:
        parts = hostname.split(".")

        is_ipv4 = (
            len(parts) == 4
            and all(part.isdigit() for part in parts)
        )

        if not is_ipv4:
            return False

    return True


def get_risk_level(phishing_probability):
    if phishing_probability < 30:
        return "LOW RISK"

    elif phishing_probability < 70:
        return "MEDIUM RISK"

    else:
        return "HIGH RISK"


@app.route("/", methods=["GET", "POST"])
def index():

    result = None
    error = None
    url = ""

    if request.method == "POST":

        url = request.form.get("url", "").strip()

        if not url:
            error = "Please enter a URL."

        elif not is_valid_url(url):
            error = "Please enter a valid URL. Example: https://example.com"

        else:

            try:

                result = predictor.predict(url)

                # Convert model prediction to readable label
                if result["prediction"] == 0:
                    result["prediction"] = "Phishing"

                elif result["prediction"] == 1:
                    result["prediction"] = "Legitimate"

                # Handle the case where predictor already returns text
                elif str(result["prediction"]).lower() == "phishing":
                    result["prediction"] = "Phishing"

                else:
                    result["prediction"] = "Legitimate"

                # Get phishing probability
                phishing_probability = result["phishing_percentage"]

                # Calculate risk level
                result["risk_level"] = get_risk_level(
                    phishing_probability
                )

            except Exception as e:

                error = f"Prediction error: {str(e)}"

    return render_template(
        "index.html",
        result=result,
        error=error,
        url=url
    )


@app.route("/health")
def health():

    return {
        "status": "OK",
        "model": "Random Forest V2",
        "features": 10
    }


if __name__ == "__main__":

    print("=" * 70)
    print("PHISHING DETECTION WEB APPLICATION")
    print("=" * 70)
    print("Model: Random Forest V2")
    print("Features: 10")
    print("Risk classification: Enabled")
    print("URL: http://127.0.0.1:5001")
    print("=" * 70)

    app.run(
        host="127.0.0.1",
        port=5001,
        debug=True
    )