# Improvement Suggestions for `analysis.py`

## 1. Missing early exit after file check

Current code:

```python
if not os.path.isfile('movie_reviews.csv'):
    print("Error: 'movie_reviews.csv' not found in the current directory.")
```

The script continues running afterward and will still fail on:

```python
df = pd.read_csv('movie_reviews.csv')
```

### Recommendation

Exit explicitly:

```python
import sys

if not os.path.isfile('movie_reviews.csv'):
    print("Error: 'movie_reviews.csv' not found.")
    sys.exit(1)
```

---

## 2. `tokenizer` may be undefined

Potential issue:

If `TOKENIZER_MODE != 'bert_local'`, then `tokenizer` is never created.

But later:

```python
input_ids = [tokenizer.cls_token_id]
```

This will crash in fallback mode.

### Recommendation

Handle both modes separately.

Example:

```python
if tokenizer_mode == 'bert_local':
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id
else:
    cls_id, sep_id, pad_id = 101, 102, 0
```

Or create dedicated padding functions for each mode.

---

## 3. Regex for HTML removal is simplistic

Current:

```python
r'<.*?>'
```

Works for basic cases but may fail on malformed HTML.

### Better approach

Use BeautifulSoup:

```python
from bs4 import BeautifulSoup

BeautifulSoup(text, "html.parser").get_text()
```

More reliable for production NLP.

---

## 4. Text cleaning removes useful symbols

Current regex:

```python
r'[^a-zA-Z0-9.,!? ]'
```

This removes:

- apostrophes
- contractions
- emojis
- accented characters

Examples:

- `"don't"` → `"dont"`
- `"it's"` → `"its"`

This may reduce sentiment quality.

### Suggested regex

```python
r"[^\w\s.,!?']"
```

Or preserve Unicode characters.

---

## 5. Large-memory Pandas apply chains

Several operations use:

```python
df['column'].apply(...)
```

This is okay for small/medium datasets, but large datasets may become slow.

Potential optimization:

- batch tokenization
- vectorized preprocessing
- HuggingFace `datasets`

---

## 6. Vocabulary generation is inefficient

This line:

```python
set(token for tokens in df['tokens'] for token in tokens)
```

Creates a full in-memory token set.

Fine for small corpora, but expensive on large datasets.

Consider:

- `Counter`
- streaming vocabulary building
- `Tokenizer` APIs

---

## 7. Hardcoded paths and constants

Current:

```python
'movie_reviews.csv'
max_length = 64
```

Better:

```python
INPUT_FILE = "movie_reviews.csv"
MAX_LENGTH = 64
```

Centralizing constants improves maintainability.

---

## 8. Missing model inference section

The script comments mention:

```python
#Step 6 — Run a Pretrained BERT Sentiment Classifier
```

But the actual inference implementation appears incomplete or missing in the visible code.

If not implemented yet, add:

- model loading
- softmax
- prediction labels
- batching
- `torch.no_grad()`

## 9. Run Python lint plugin to avoid warnings and notices

- String statement has no effect
- Reimport 'BertTokenizer' (imported line 36)
- Missing module docstring
- standard import "os" should be placed before third party import "pandas"
- standard import "pathlib.Path" should be placed before third party import "pandas"
- standard import "re" should be placed before third party import "pandas"
- Import "from transformers import BertTokenizer" should be placed at the top of the module
- Missing function or method docstring
- Line too long (130/100)
- Line too long (107/100)
- Line too long (111/100)
- Constant name "max_length" doesn't conform to UPPER_CASE naming style
- Missing function or method docstring
- Import "from transformers import BertForSequenceClassification, BertTokenizer" should be placed at the top of the module
- Line too long (113/100)
- Import "import torch" should be placed at the top of the module
- Missing function or method docstring
- Line too long (107/100)
- Line too long (140/100)
- Missing function or method docstring
- Line too long (120/100)
- Final newline missing