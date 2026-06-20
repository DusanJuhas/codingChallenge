
# Sentiment Analysis Pipeline (Movie Reviews)

This project implements a practical BERT-based pipeline for sentiment analysis on short movie reviews.

## Features
- Loading input data from `movie_reviews.csv`
- Text cleaning & preprocessing
- BERT tokenization (WordPiece)
- Token → ID conversion
- Fixed-length padding & attention masks
- Real pretrained IMDB BERT classifier
- Final sentiment predictions with confidence scores
- Outputs two CSVs:
  - `predictions.csv` — clean final results
  - `result.csv` — full debug information

## ⚠ Important Notes on Library and Model Size
Using this project requires downloading two large components:

### 1. PyTorch Library (`torch`)
- The `torch` package is **large** and may require **hundreds of megabytes** of disk space.
- Installation time may be noticeable on slower systems.
- On Windows, PyTorch wheels include CPU kernels that increase size.

### 2. Pretrained BERT Model Files
- The model `textattack/bert-base-uncased-imdb` is also large.
- Expect an additional download of **~400–500 MB** during the first run.
- HuggingFace caches models in your user directory, which may grow over time.

### ✔ Recommendations
- Ensure at least **1 GB of free disk space** before running the script.
- Use a stable internet connection for the initial download.
- Clean HuggingFace model cache when needed:

```bash
huggingface-cli delete-cache
```

Or manually delete:
```
~/.cache/huggingface
```

## Requirements
Install dependencies with:

```bash
pip install -r requirements.txt
```

Dependencies include:
- `pandas`
- `transformers`
- `torch`

## Project Structure
The pipeline is organized as an object-oriented package, with each
stage of the process implemented as its own class in its own file:

```
main.py
sentiment_pipeline/
├── __init__.py          # exposes all classes for import
├── data_loader.py        # DataLoader        -> Step 1: load CSV
├── text_cleaner.py       # TextCleaner        -> Step 2: clean/preprocess text
├── bert_encoder.py       # BertTextEncoder     -> Steps 3-5: tokenize, IDs, padding/masks
├── classifier.py         # SentimentClassifier -> Step 6: BERT IMDB inference
├── result_formatter.py   # ResultFormatter     -> Step 7: format & save results
└── pipeline.py            # SentimentPipeline   -> orchestrates all steps
```

`SentimentPipeline` wires the stages together and exposes a single
`run()` method that executes the full pipeline end-to-end, identical
in behavior to the original procedural script.

You can also import and reuse individual classes elsewhere, e.g.:

```python
from sentiment_pipeline import TextCleaner

cleaner = TextCleaner()
clean_text = cleaner.clean("Some <b>raw</b> review text!")
```

## How to Run
1. Place your input file `movie_reviews.csv` in the project directory (next to `main.py`).
2. Run the entry point script:

```bash
python main.py
```

This instantiates `SentimentPipeline` and calls `.run()`, which executes
all seven steps in sequence and saves the output files.

## Output Files
### `predictions.csv`
Human‑readable output containing:
- Original review
- Final sentiment label with confidence (e.g., `positive (97.4%)`)

### `result.csv`
Full detailed processing output:
- cleaned text
- tokens
- token IDs
- input IDs
- attention masks
- probabilities
- final sentiment

## Notes
- The model used is `textattack/bert-base-uncased-imdb`, trained specifically for movie review sentiment analysis.
- This project can be extended for fine‑tuning or additional NLP tasks.
