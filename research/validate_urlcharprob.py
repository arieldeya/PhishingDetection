import pandas as pd
from collections import Counter
import string

DATASET = "PhiUSIIL_Phishing_URL_Dataset.csv"

print("=" * 70)
print("URLCharProb REPRODUCTION TEST")
print("=" * 70)

# ---------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------
df = pd.read_csv(DATASET)

print(f"Dataset rows: {len(df):,}")

# 0 = phishing
# 1 = legitimate
legitimate = df[df["label"] == 1].copy()

print(f"Legitimate URLs: {len(legitimate):,}")

# ---------------------------------------------------------
# 2. Calculate character frequencies from legitimate URLs
# ---------------------------------------------------------
allowed_chars = set(string.ascii_lowercase + string.digits)

char_counts = Counter()
total_chars = 0

for url in legitimate["URL"].astype(str):

    for char in url.lower():

        if char in allowed_chars:
            char_counts[char] += 1
            total_chars += 1

# Probability of each character
char_prob = {
    char: char_counts[char] / total_chars
    for char in allowed_chars
}

print("\nCharacter probabilities:")
for char in sorted(char_prob):
    print(f"{char}: {char_prob[char]:.8f}")

print(f"\nTotal alphabet/digit characters: {total_chars:,}")

# ---------------------------------------------------------
# 3. Reproduce URLCharProb
# ---------------------------------------------------------
def calculate_url_char_prob(url):

    chars = [
        char.lower()
        for char in str(url)
        if char.lower() in allowed_chars
    ]

    if not chars:
        return 0.0

    probability_sum = sum(char_prob[char] for char in chars)

    return probability_sum / len(chars)


# ---------------------------------------------------------
# 4. Compare calculated vs original dataset value
# ---------------------------------------------------------
sample = df.sample(1000, random_state=42).copy()

sample["Calculated_URLCharProb"] = sample["URL"].apply(
    calculate_url_char_prob
)

sample["Absolute_Error"] = (
    sample["URLCharProb"] -
    sample["Calculated_URLCharProb"]
).abs()

print("\n" + "=" * 70)
print("COMPARISON")
print("=" * 70)

print(
    sample[
        [
            "URL",
            "URLCharProb",
            "Calculated_URLCharProb",
            "Absolute_Error"
        ]
    ].head(20).to_string(index=False)
)

print("\nError statistics:")

print(
    sample["Absolute_Error"].describe().to_string()
)

print(
    "\nMean Absolute Error:",
    sample["Absolute_Error"].mean()
)

print(
    "Maximum Absolute Error:",
    sample["Absolute_Error"].max()
)

# Correlation
correlation = sample[
    ["URLCharProb", "Calculated_URLCharProb"]
].corr().iloc[0, 1]

print(
    "Correlation:",
    correlation
)

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)