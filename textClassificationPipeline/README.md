# NLP Text Classification Dataset

## Overview
This project provides a synthetic Natural Language Processing (NLP) dataset designed for text classification tasks. The dataset contains 500 records with labeled text data across multiple categories.

## Dataset Structure
The dataset includes the following columns:

- **id**: Unique identifier for each record
- **text**: A short sentence representing the data sample
- **category**: The label assigned to the text

## Categories
The dataset contains five categories:

- sports
- technology
- health
- finance
- entertainment

## Example Record
```
id: 1
text: "The team won the championship in a thrilling match"
category: sports
```

## Use Cases
This dataset can be used for:

- Text classification
- NLP preprocessing practice (tokenization, stemming, lemmatization)
- Machine learning model training (Naive Bayes, Logistic Regression)
- Feature extraction (Bag-of-Words, TF-IDF)

## Getting Started

### 1. Load Dataset
```python
import pandas as pd

df = pd.read_csv("large_news_dataset.csv")
print(df.head())
```

### 2. Prepare Data
```python
from sklearn.model_selection import train_test_split

X = df['text']
y = df['category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 3. Train Model (Example: Logistic Regression)
```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression())
])

pipeline.fit(X_train, y_train)
```

### 4. Evaluate Model
```python
from sklearn.metrics import accuracy_score

predictions = pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
```

## Requirements
- Python 3.x
- see requirements.txt

## Notes
- This dataset is synthetically generated for educational purposes.
- It is suitable for learning and experimentation but not for production use.

## License
This dataset is free to use for educational and research purposes.
