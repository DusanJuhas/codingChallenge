# ================================
# TASK B.6 — FEATURE ENGINEERING (Tasks 1–4)
# ================================
print("\n=== TASK B.6: FEATURE ENGINEERING ===")

# ---- Helper: choose a unified text column (review/article) ----
# We'll pick the first available among common text fields.
text_field_candidates = ["review_text", "content", "article_text", "body", "text"]
text_col = next((c for c in text_field_candidates if c in df.columns), None)

if text_col is None:
    print("No text field found among candidates:", text_field_candidates)
else:
    print(f"Using text column '{text_col}' for text-based features.")

    # Ensure text column is string and normalized (should be already normalized above)
    df[text_col] = df[text_col].astype(str)

    # -----------------------------
    # 1) WORD COUNT PER REVIEW/ARTICLE
    # -----------------------------
    # Simple tokenization by whitespace; ignore empty tokens
    df["word_count"] = (
        df[text_col]
        .str.split()                                # split on whitespace
        .apply(lambda toks: len([t for t in toks if t.strip() != ""]))
    )
    print("Created 'word_count' column.")

    # -----------------------------
    # 2) TOP KEYWORDS (simple string ops)
    # -----------------------------
    # Minimalistic stopword list to keep it dependency-free
    STOPWORDS = {
        "a", "an", "the", "and", "or", "but", "if", "while", "of", "at", "by", "for",
        "with", "about", "against", "between", "into", "through", "during", "before",
        "after", "above", "below", "to", "from", "up", "down", "in", "out", "on",
        "off", "over", "under", "again", "further", "then", "once", "here", "there",
        "when", "where", "why", "how", "all", "any", "both", "each", "few", "more",
        "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same",
        "so", "than", "too", "very", "can", "will", "just", "don", "should", "now",
        # domain-ish fillers
        "movie", "film", "review", "reviews", "article", "news"
    }

    # Clean text: keep alphanumerics and whitespace, lowercase (you already lowercased in B.4)
    def _simple_tokens(s: str):
        # replace non-alphanumeric with space, split, filter very short tokens
        import re
        toks = re.sub(r"[^a-z0-9\s]", " ", s.lower()).split()
        return [t for t in toks if len(t) >= 3 and t not in STOPWORDS]

    # Build a simple corpus frequency (global)
    from collections import Counter
    corpus_counter = Counter()
    df["_tokens"] = df[text_col].apply(_simple_tokens)
    for toks in df["_tokens"]:
        corpus_counter.update(toks)

    # Top-N keywords across the whole dataset
    TOP_N_GLOBAL = 20
    top_keywords_global = [w for w, _ in corpus_counter.most_common(TOP_N_GLOBAL)]
    print(f"Global top {TOP_N_GLOBAL} keywords:", top_keywords_global)

    # Per-row "keywords": intersection with top global, keeping row order
    def _row_keywords(tokens):
        seen = set()
        result = []
        for t in tokens:
            if t in top_keywords_global and t not in seen:
                seen.add(t)
                result.append(t)
        return result

    df["keywords"] = df["_tokens"].apply(_row_keywords)

    # (Optional) Concise string version for quick viewing
    df["keywords_str"] = df["keywords"].apply(lambda ks: ", ".join(ks))
    print("Created 'keywords' (list) and 'keywords_str' (string) columns.")

    # -----------------------------
    # 3) DETECT EXTREMELY LONG / SHORT REVIEWS/ARTICLES
    # -----------------------------
    # Rule of thumb thresholds:
    #   - "short" if < 10 words
    #   - "long"  if > 90th percentile of word_count
    SHORT_THRESHOLD = 10
    long_threshold = df["word_count"].quantile(0.90) if "word_count" in df.columns else None

    df["is_short_text"] = df["word_count"] < SHORT_THRESHOLD
    if long_threshold is not None:
        df["is_long_text"] = df["word_count"] > long_threshold
        print(f"Flagged 'is_short_text' (< {SHORT_THRESHOLD} words) and 'is_long_text' (> {int(long_threshold)} words).")
    else:
        df["is_long_text"] = False
        print("Could not compute long threshold — 'word_count' missing. Defaulting 'is_long_text' to False.")

    # -----------------------------
    # 4) SENTIMENT PROXY FROM RATINGS
    # -----------------------------
    # Map ratings to 'positive' / 'neutral' / 'negative'
    # Customize thresholds as needed:
    POS_THRESH = 8   # rating >= 8 → positive
    NEG_THRESH = 4   # rating <= 4 → negative

    if "rating" in df.columns:
        def _sentiment_from_rating(r):
            try:
                if pd.isna(r):
                    return "unknown"
                if r >= POS_THRESH:
                    return "positive"
                if r <= NEG_THRESH:
                    return "negative"
                return "neutral"
            except Exception:
                return "unknown"

        df["sentiment_proxy"] = df["rating"].apply(_sentiment_from_rating)
        print(f"Created 'sentiment_proxy' using thresholds: positive≥{POS_THRESH}, negative≤{NEG_THRESH}.")
    else:
        df["sentiment_proxy"] = "unknown"
        print("No 'rating' column found — set 'sentiment_proxy' to 'unknown'.")

    # Clean up helper column
    df.drop(columns=["_tokens"], inplace=True, errors="ignore")