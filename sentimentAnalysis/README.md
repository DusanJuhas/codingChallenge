
# Sentiment Analysis Pipeline (Movie Reviews)

This project demonstrates a simple NLP pipeline for sentiment analysis using:
- Text loading (CSV input)
- Text cleaning and preprocessing
- BERT tokenization (HuggingFace)

## Requirements

Install all required Python libraries:

```bash
pip install -r requirements.txt
```

## How to Run

1. Ensure the input file `movie_reviews.csv` is in the project directory.
2. Run the main script:

```bash
python analysis.py
```

This will:
- Load the CSV file
- Clean the text
- Tokenize the sentences using BERT
- Print previews of each step

## Files
- `analysis.py` — main sentiment-analysis pipeline script (Step 1–3)
- `movie_reviews.csv` — input data (one review per row)
- `requirements.txt` — dependency list

## Notes
- The script currently performs loading, cleaning, and tokenization only.
- Further steps (encoding, model inference, classification) can be added later.
