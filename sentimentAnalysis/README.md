
# Sentiment Analysis Pipeline (Movie Reviews)

This project demonstrates a simple NLP pipeline for sentiment analysis using:
- Loading CSV input (`movie_reviews.csv`)
- Text cleaning and preprocessing
- BERT tokenization
- Token-to-ID conversion
- Saving results to `result.csv`

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
- Convert tokens to numerical IDs
- Save the complete processed dataset into `result.csv`

## Output

The script generates:
- `result.csv` — containing:
  - `review`
  - `clean_review`
  - `tokens`
  - `token_ids`

## Files
- `analysis.py` — main sentiment-analysis pipeline script
- `movie_reviews.csv` — input data
- `result.csv` — processed output
- `requirements.txt` — dependency list

## Notes
- More steps can be added later (padding, attention masks, model inference).
