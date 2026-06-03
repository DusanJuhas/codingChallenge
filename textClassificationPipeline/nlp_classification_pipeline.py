"""
NLP Text Classification Pipeline
================================
This script performs:
1. Data loading and exploration
2. Text preprocessing (tokenization, stemming, lemmatization)
3. Feature extraction (TF-IDF)
4. Model training (Naive Bayes, Logistic Regression)
5. Evaluation and comparison

Dataset: news_dataset.csv
"""

# ================================
# 📦 Imports
# ================================
import string
import subprocess
import sys
import pandas as pd

# NLP libraries
import nltk
import spacy

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ML libraries
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


# ================================
# 🔽 Download required resources
# ================================
nltk.download('punkt')
nltk.download('stopwords')

# Load spaCy model

def load_spacy_model():
    """Load the spaCy English model, installing it if missing.

    Returns:
        spacy.language.Language: Loaded spaCy language model.
    """
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        print("⚠️ spaCy model not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        return spacy.load("en_core_web_sm")

nlp = load_spacy_model()


# Initialize tools
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()


# ================================
# 📂 Load Dataset
# ================================
def load_data(filepath: str) -> pd.DataFrame:
    """Load CSV dataset into a pandas DataFrame."""
    df = pd.read_csv(filepath)
    print("\n✅ Dataset loaded successfully")
    print(df.head())
    return df


# ================================
# 🔍 Explore Data
# ================================
def explore_data(df: pd.DataFrame):
    """Print basic dataset statistics."""
    print("\n📊 Dataset Info")
    print(df.info())

    print("\n📊 Class Distribution")
    print(df['category'].value_counts())

    print("\n📏 Sample Text Lengths")
    print(df['text'].apply(lambda x: len(str(x).split())).describe())


# ================================
# 🧠 Preprocessing Functions
# ================================

def tokenize_text(text: str) -> str:
    """
    Tokenization:
    - Lowercase
    - Remove punctuation and stopwords
    """
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t not in string.punctuation]
    tokens = [t for t in tokens if t not in stop_words]
    return " ".join(tokens)


def stem_text(text: str) -> str:
    """
    Stemming:
    - Apply Porter Stemmer
    - Remove non-alphabetic tokens
    """
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalpha()]
    tokens = [stemmer.stem(t) for t in tokens if t not in stop_words]
    return " ".join(tokens)


def lemmatize_text(text: str) -> str:
    """
    Lemmatization:
    - Using spaCy
    - Remove stopwords and punctuation
    """
    doc = nlp(text)
    return " ".join([
        token.lemma_
        for token in doc
        if not token.is_stop and not token.is_punct
    ])


# ================================
# 🔄 Apply Preprocessing
# ================================
def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all preprocessing techniques."""
    print("\n⚙️ Preprocessing texts...")

    df['text_tokenized'] = df['text'].apply(tokenize_text)
    df['text_stemmed'] = df['text'].apply(stem_text)
    df['text_lemmatized'] = df['text'].apply(lemmatize_text)

    print("✅ Preprocessing completed")
    return df


# ================================
# 🧮 Feature Extraction
# ================================
def vectorize_text(text_series):
    """Convert text into TF-IDF features."""
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(text_series)
    return X, vectorizer


# ================================
# 🤖 Model Training & Evaluation
# ================================
def evaluate_models(X, y, label: str):
    """
    Train and evaluate:
    - Naive Bayes
    - Logistic Regression
    """
    print(f"\n🚀 Evaluating for: {label}")

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = {}

    # --------------------------
    # ✅ Naive Bayes
    # --------------------------
    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    y_pred_nb = nb.predict(X_test)

    print("\n🧾 Naive Bayes Report:")
    print(classification_report(y_test, y_pred_nb))

    results['Naive Bayes'] = accuracy_score(y_test, y_pred_nb)

    # --------------------------
    # ✅ Logistic Regression
    # --------------------------
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    print("\n🧾 Logistic Regression Report:")
    print(classification_report(y_test, y_pred_lr))

    results['Logistic Regression'] = accuracy_score(y_test, y_pred_lr)

    return results


# ================================
# 📊 Comparison Summary
# ================================
def print_summary(results_dict):
    """Display final comparison table."""
    print("\n📊 FINAL COMPARISON RESULTS")
    print("-" * 40)

    for prep_method, models in results_dict.items():
        for model_name, score in models.items():
            print(f"{prep_method:15} | {model_name:20} | Accuracy: {score:.2f}")


# ================================
# 🏁 Main Execution
# ================================
def main():
    # Load dataset
    df = load_data("large_news_dataset.csv")

    # Explore dataset
    explore_data(df)

    # Preprocess text
    df = preprocess_dataset(df)

    y = df['category']

    # Store results
    all_results = {}

    # --------------------------
    # Tokenized
    # --------------------------
    X_tok, _ = vectorize_text(df['text_tokenized'])
    all_results['Tokenized'] = evaluate_models(X_tok, y, "Tokenized")

    # --------------------------
    # Stemmed
    # --------------------------
    X_stem, _ = vectorize_text(df['text_stemmed'])
    all_results['Stemmed'] = evaluate_models(X_stem, y, "Stemmed")

    # --------------------------
    # Lemmatized
    # --------------------------
    X_lemma, _ = vectorize_text(df['text_lemmatized'])
    all_results['Lemmatized'] = evaluate_models(X_lemma, y, "Lemmatized")

    # Print comparison
    print_summary(all_results)


# ================================
# ▶️ Run Script
# ================================
if __name__ == "__main__":
    main()