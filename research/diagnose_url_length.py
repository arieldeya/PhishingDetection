import pandas as pd
from urllib.parse import urlparse

DATASET = "PhiUSIIL_Phishing_URL_Dataset.csv"

df = pd.read_csv(DATASET)

print("=" * 100)
print("URL LENGTH DIFFERENCE DIAGNOSIS")
print("=" * 100)

def raw_length(url):
    return len(str(url))

def no_protocol(url):
    url = str(url)
    if url.startswith("https://"):
        return url[8:]
    if url.startswith("http://"):
        return url[7:]
    return url

def no_www(url):
    url = str(url)
    if url.startswith("https://www."):
        return "https://" + url[12:]
    if url.startswith("http://www."):
        return "http://" + url[11:]
    if url.startswith("www."):
        return url[4:]
    return url

rows = []

for _, row in df.head(1000).iterrows():

    url = str(row["URL"])
    dataset_value = row["URLLength"]

    calculated = raw_length(url)

    if calculated != dataset_value:
        rows.append({
            "URL": url,
            "DATASET": dataset_value,
            "RAW": calculated,
            "DIFF": calculated - dataset_value,
            "NO_PROTOCOL": len(no_protocol(url)),
            "NO_WWW": len(no_www(url)),
        })

print(f"\nDifferences found: {len(rows)}")

print("\nFirst 50 differences:")
print("-" * 100)

for item in rows[:50]:
    print(f"\nURL: {item['URL']}")
    print(f"Dataset URLLength : {item['DATASET']}")
    print(f"Raw calculated    : {item['RAW']}")
    print(f"Difference        : {item['DIFF']}")
    print(f"No protocol      : {item['NO_PROTOCOL']}")
    print(f"No www           : {item['NO_WWW']}")

print("\n" + "=" * 100)