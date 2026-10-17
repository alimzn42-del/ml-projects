# Arabic Hotel Reviews — Unhappy-Guest Detection

Flags guests whose Arabic review indicates a bad stay (1–3★), so the hotel can call them before
they are lost. Input: the review text. Output: a probability that the guest is unhappy.

**Performance on 42,024 reviews from hotels the model never saw:**
**91.7% of unhappy guests caught** · precision 0.671 · unhappy F1 0.775 (threshold 0.4).

## How to score new reviews

Environment: Python 3.13 (`.venv313`), `torch` + `transformers` + `sentencepiece`, CUDA GPU recommended.

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import reviews_pipeline as rp

bert_name = "aubmindlab/bert-base-arabertv02"
btok = AutoTokenizer.from_pretrained(bert_name)
model = AutoModelForSequenceClassification.from_pretrained(bert_name, num_labels=3)
model.load_state_dict(torch.load("models/arabert_seed0.pt"))
model.to("cuda").eval()

probs = rp.bert_unhappy_probs(model, btok, ["النظافة سيئة والخدمة بطيئة", "فندق رائع"])
unhappy = probs >= 0.4        # the agreed threshold
```

### Input requirements (important)

Production input must get the same cleaning the training data got, **before** scoring:

1. **Remove the site's rating word** if present at the start of the review ("ممتاز."، "جيد."، …) —
   use `rp.strip_site_labels` with `data/hard/site_labels.json`. Scoring raw text with the rating
   word still in it inflates confidence and leaks the site's own rating into the decision.
2. **Decode HTML entities** (`&amp;` `&lt;` …) — a small number of raw reviews contain them.

### The threshold is a business dial, not a model constant

0.4 was chosen because below it each extra caught guest costs more than 4 false alarms.
At 0.4 the hotel calls ~48% of reviewers and reaches ~92% of the unhappy ones. Raising the
threshold means fewer calls and more missed guests — the full sweep is in
[`EVALUATION.md`](EVALUATION.md); pick the point that matches the cost of a call vs a lost guest.

## Files

| File | Purpose |
|---|---|
| `models/arabert_seed0.pt` | shipped model weights (state_dict for AraBERT v02, 3 labels) |
| `reviews_pipeline.py` | cleaning, splitting, tokenizer, batching, `bert_unhappy_probs` |
| `arabic-reviews.ipynb` | all experiments, in order, with outputs |
| [`DELIVERY.md`](DELIVERY.md) | delivery note: problems, decisions, lessons, open items |
| [`EVALUATION.md`](EVALUATION.md) | full numbers |
| [`data/README.md`](data/README.md) | dataset source and derived files |
