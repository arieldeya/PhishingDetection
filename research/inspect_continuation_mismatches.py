import pandas as pd
import numpy as np
import re

DATASET = "PhiUSIIL_Phishing_URL_Dataset.csv"

df = pd.read_csv(DATASET)

urls = df["URL"].astype(str)


def char_class(c):

    if c.isalpha():
        return "L"

    if c.isdigit():
        return "D"

    return "S"


def continuation_rate(url):

    if len(url) == 0:
        return 0.0

    if len(url) == 1:
        return 1.0

    runs = 0
    current = char_class(url[0])

    for c in url[1:]:

        new_class = char_class(c)

        if new_class != current:
            runs += 1
            current = new_class

    runs += 1

    return (len(url) - runs + 1) / len(url)


target = df["CharContinuationRate"].astype(float).values

calculated = urls.map(continuation_rate).values

difference = calculated - target

mismatch = np.where(
    ~np.isclose(calculated, target, atol=1e-12)
)[0]

print("=" * 100)
print("CHAR CONTINUATION MISMATCH ANALYSIS")
print("=" * 100)

print()
print("Total rows:", len(df))
print("Exact matches:", len(df) - len(mismatch))
print("Mismatches:", len(mismatch))

print()
print("=" * 100)
print("FIRST 50 MISMATCHES")
print("=" * 100)


def show_runs(url):

    if not url:
        return []

    result = []

    current = char_class(url[0])
    chars = url[0]

    for c in url[1:]:

        cls = char_class(c)

        if cls == current:
            chars += c
        else:
            result.append((current, chars))
            current = cls
            chars = c

    result.append((current, chars))

    return result


for i in mismatch[:50]:

    url = urls.iloc[i]

    print()
    print("-" * 100)

    print("INDEX:", i)
    print("URL:", repr(url))

    print(
        "URL LENGTH:",
        len(url)
    )

    print(
        "DATASET:",
        target[i]
    )

    print(
        "CALCULATED:",
        calculated[i]
    )

    print(
        "DIFFERENCE:",
        difference[i]
    )

    print()
    print("CLASS RUNS:")

    runs = show_runs(url)

    for cls, chars in runs:

        print(
            f"{cls:<2} "
            f"length={len(chars):>4} "
            f"value={repr(chars)}"
        )


print()
print("=" * 100)
print("MOST COMMON DIFFERENCES")
print("=" * 100)

rounded_difference = np.round(
    difference,
    6
)

values, counts = np.unique(
    rounded_difference,
    return_counts=True
)

order = np.argsort(counts)[::-1]

for idx in order[:30]:

    print(
        f"{values[idx]:>12} : "
        f"{counts[idx]:>8}"
    )


print()
print("=" * 100)
print("COMPLETE")
print("=" * 100)