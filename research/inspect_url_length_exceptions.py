import pandas as pd

DATASET = "PhiUSIIL_Phishing_URL_Dataset.csv"

df = pd.read_csv(DATASET)

print("=" * 100)
print("URL LENGTH EXCEPTION ANALYSIS")
print("=" * 100)

exceptions = []

for _, row in df.head(1000).iterrows():

    url = str(row["URL"])
    dataset_length = int(row["URLLength"])
    raw_length = len(url)

    # Our discovered rule
    predicted = raw_length - 1

    if predicted != dataset_length:

        exceptions.append({
            "URL": url,
            "DATASET": dataset_length,
            "RAW": raw_length,
            "RAW_MINUS_ONE": predicted,
            "DIFFERENCE": predicted - dataset_length,
            "HAS_TRAILING_SLASH": url.endswith("/"),
            "HAS_QUERY": "?" in url,
            "HAS_FRAGMENT": "#" in url,
            "HAS_WWW": "www." in url.lower(),
            "HAS_HTTP": url.lower().startswith("http://"),
            "HAS_HTTPS": url.lower().startswith("https://"),
        })


print(f"\nExceptions found: {len(exceptions)}")

print("\n" + "=" * 100)
print("EXCEPTION DETAILS")
print("=" * 100)

for i, item in enumerate(exceptions, start=1):

    print(f"\n{i}. URL:")
    print(item["URL"])

    print(f"   Dataset URLLength : {item['DATASET']}")
    print(f"   Raw length       : {item['RAW']}")
    print(f"   Raw - 1          : {item['RAW_MINUS_ONE']}")
    print(f"   Difference       : {item['DIFFERENCE']}")

    print(f"   Trailing slash   : {item['HAS_TRAILING_SLASH']}")
    print(f"   Query            : {item['HAS_QUERY']}")
    print(f"   Fragment         : {item['HAS_FRAGMENT']}")
    print(f"   www              : {item['HAS_WWW']}")
    print(f"   HTTP             : {item['HAS_HTTP']}")
    print(f"   HTTPS            : {item['HAS_HTTPS']}")

    if i >= 50:
        print("\n... showing first 50 exceptions only ...")
        break


print("\n" + "=" * 100)
print("EXCEPTION PATTERN SUMMARY")
print("=" * 100)

exception_df = pd.DataFrame(exceptions)

if len(exception_df) > 0:

    print("\nTrailing slash:")
    print(exception_df["HAS_TRAILING_SLASH"].value_counts())

    print("\nQuery:")
    print(exception_df["HAS_QUERY"].value_counts())

    print("\nFragment:")
    print(exception_df["HAS_FRAGMENT"].value_counts())

    print("\nWWW:")
    print(exception_df["HAS_WWW"].value_counts())

    print("\nHTTP:")
    print(exception_df["HAS_HTTP"].value_counts())

    print("\nHTTPS:")
    print(exception_df["HAS_HTTPS"].value_counts())


print("\n" + "=" * 100)