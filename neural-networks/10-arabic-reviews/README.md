# Unhappy Guest Detection — Arabic Hotel Reviews

Flags hotel reviews written by unhappy guests so the customer-care team can contact them.

- Input: one Arabic review (free text, as submitted on the booking site).
- Output: a probability that the guest is unhappy, and a flag (`unhappy` / `happy`) using a fixed threshold.
- Definition used: **unhappy = rating 1, 2 or 3 stars** (client decision: 3/5 is a bad stay).

## Results (held-out test set, 42,024 reviews from hotels never seen in training)

| Metric | Value |
|---|---|
| Unhappy guests caught (recall) | **91.7%** (13,537 of 14,758) |
| Unhappy guests missed | 1,221 |
| Flags that are truly unhappy (precision) | 67.1% |
| False alarms (happy guest flagged) | 6,643 |
| Total reviews flagged | 20,180 (48% of all reviews) |
| Unhappy F1 | 0.775 |

Roughly **1 in 3 flags is a happy guest**. This is the price of missing fewer than 1 in 10 unhappy guests.
The threshold can be changed to trade these two (see `EVALUATION.md`, threshold table).

## Model

- Fine-tuned **AraBERT** (`aubmindlab/bert-base-arabertv02`), 3 outputs (negative / neutral / positive).
- `p_unhappy = p(negative) + p(neutral)`; flag if `p_unhappy >= 0.4`.
- Reviews are cut to the first 128 tokens (covers ~98.5% of reviews fully).
- Weights: `models/arabert_seed0.pt`.

## Usage

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import reviews_pipeline as rp

BERT = "aubmindlab/bert-base-arabertv02"
THRESHOLD = 0.4

tok = AutoTokenizer.from_pretrained(BERT)
model = AutoModelForSequenceClassification.from_pretrained(BERT, num_labels=3)
model.load_state_dict(torch.load("models/arabert_seed0.pt"))
model.to("cuda")

texts = ["...review 1...", "...review 2..."]
p = rp.bert_unhappy_probs(model, tok, texts, max_len=128)
flags = p >= THRESHOLD
```

Requirements: Python 3.13, PyTorch with CUDA, `transformers`, `pandas`, `pyarrow`, `numpy`, `scikit-learn`, `sentencepiece`.
A GPU is recommended (developed on an RTX 4050, 6 GB).

## Input requirements

- The text must be the guest's own words. Remove the rating word the site adds at the start of the review
  (e.g. "ممتاز.") before scoring — the model was trained without it (see `strip_site_labels`).
- Decode HTML entities (`&amp;`, `&lt;`, `&gt;`, `&nbsp;`) before scoring (`html.unescape`).

## Known limitations

- About 1 in 3 flags is a false alarm; plan team capacity for ~48% of reviews being flagged at the default threshold.
- Labels come from star ratings, which are noisy: a manual audit found that most remaining "errors" are reviews
  whose text does not match their stars. Higher scores are limited by the data, not only the model.
- Text beyond 128 tokens is ignored (~1.5% of reviews). Long reviews are more often negative.
- Trained on reviews from one booking site (HARD dataset); performance on other sources is not measured.

## Project files

```
reviews_pipeline.py        data loading, cleaning, splitting, models, training, prediction
arabic-reviews.ipynb       all experiments, in order
data/hard/                 raw data, cleaned cache (reviews_clean.parquet), site_labels.json, audit/
models/                    trained weights and training histories
EVALUATION.md              full evaluation report
DELIVERY.md                project delivery notes: problems, fixes, lessons
```
