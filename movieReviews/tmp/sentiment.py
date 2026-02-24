# Average length by sentiment
if "sentiment_proxy" in df.columns and "word_count" in df.columns:
    print("\nAverage word count by sentiment:")
    print(df.groupby("sentiment_proxy")["word_count"].mean().round(1))

# Top keywords within positive vs negative
if "sentiment_proxy" in df.columns and "keywords" in df.columns:
    from collections import Counter
    for label in ["positive", "negative"]:
        freqs = Counter([k for ks in df.loc[df["sentiment_proxy"] == label, "keywords"] for k in ks])
        print(f"\nTop keywords for {label}:")
        print(freqs.most_common(10))