# Project Delivery — Arabic Hotel Reviews, Unhappy-Guest Detection

**Owner:** Mohammed Ali · **Delivered:** 17 Oct 2026
**Task:** flag unhappy guests (a 1–3★ stay) from the text of their Arabic review, so the hotel can call before the guest is lost.

## What was delivered

| File | What it is |
|---|---|
| [`README.md`](README.md) | client-facing: what the system does and how to score new reviews |
| [`EVALUATION.md`](EVALUATION.md) | the full numbers: every model, the threshold sweep, the final test |
| [`reviews_pipeline.py`](reviews_pipeline.py) | cleaning, hotel-based split, tokenizer, batching, all models, training and prediction functions |
| [`arabic-reviews.ipynb`](arabic-reviews.ipynb) | every experiment in order, with outputs |
| `models/arabert_seed0.pt` | the shipped model (AraBERT fine-tuned, 3-class read as binary) · threshold **0.4** |
| `data/README.md` | dataset source (HARD) and the derived files |

**Final test result** (42,024 reviews from hotels never seen in training, opened once):
**recall 0.917 · precision 0.671 · unhappy F1 0.775** — 13,537 of 14,758 unhappy guests caught.

## Layout

```
arabic-reviews.ipynb      the experiments, in order
reviews_pipeline.py       everything reusable
DELIVERY.md               this file
EVALUATION.md             full numbers
README.md                 how to use the shipped model
data/hard/                HARD dataset + derived files (not committed — see data/README.md)
models/                   all checkpoints incl. the shipped arabert_seed0.pt (not committed: 540 MB)
```

## Problems faced and how they were solved

| # | Problem | How it was found | Fix |
|---|---|---|---|
| 1 | The site adds a rating word at the start of reviews ("ممتاز.") — the label leaked into the text | Reading raw reviews: the first words looked added by the site | Words that always come with one rating (≥ 500 times) are removed: `find_site_labels`, `strip_site_labels` |
| 2 | Same hotel in train and test would inflate scores | Design decision before training | Split by hotel, not by review: `split_by_hotel` — overlap verified 0/0/0 |
| 3 | Duplicates and empty reviews | Data audit | `clean`: drop duplicates on all meta columns, drop reviews with no real text |
| 4 | Cells failed after kernel restarts (`NameError`) | Repeated errors | Cleaned data cached to parquet; cells made self-contained |
| 5 | After installing pyarrow, regex with `\u` escapes failed (`ArrowInvalid`) | Error on load | `EMPTY_CHARS` rewritten without raw-string escapes; counts checked identical |
| 6 | AraBERT 3.5× slower than expected (~13 h estimated) | Batch time and GPU power far below normal | Windows was spilling GPU memory to RAM: turned off "CUDA Sysmem Fallback", max length 128 → ~24 min/epoch |
| 7 | "Run All" re-ran every experiment | Long unexpected run | Interrupted, restarted, ran only needed cells; training cells made resumable (skip if saved) |
| 8 | Comparing models on different metrics (macro F1 vs unhappy F1) | Supervisor review | All candidates re-scored on the client metric before choosing |
| 9 | Scores stopped improving | Plateau across models | Blind audit of 120 reviews (90 of them errors): 58 were label noise vs 21 model mistakes — further tuning stopped |

## Decisions and why

- **Task: unhappy = 1–3 stars.** Client decision: a 3/5 stay is a bad stay and worth a call.
- **Model: AraBERT, 3-class, read as binary.** Best on the client metric with non-overlapping
  seed ranges (unhappy F1 0.765–0.768 vs 0.751–0.753 for the GRU trained binary), and highest recall.
- **Threshold 0.4.** Below it, each extra unhappy guest caught costs more than 4 false alarms
  (full sweep in `EVALUATION.md`).
- **Stop improving here.** The blind audit showed most remaining errors are in the labels themselves;
  the expected gain of further work (e.g. splitting reviews into sections) is about the size of the seed noise.

## Lessons

1. Look at the raw text before trusting any score. The biggest problem in the project (the leak) was found by
   reading reviews, not by metrics.
2. A feature is a leak if it comes from the label or will not exist at prediction time — not because it is
   correlated with the label (guest titles are fine, site rating words are not).
3. Compare models only on the same metric, and choose the metric from the client's goal.
4. Use several seeds; a difference counts only when the ranges do not overlap.
5. Before retraining for a new goal, check if the existing model's predictions can be re-read for it
   (AraBERT 3-class read as binary beat a GRU trained binary, with no new training).
6. The threshold does not improve the model; it chooses which error you pay for. Choose it where the marginal cost
   starts to jump, and let the client set the final value.
7. Fix everything on val, then open the test set once. Any change after seeing it makes the number optimistic.
8. Know when to stop: when the data ceiling is near, measure the expected gain before spending hours on it.
9. When a GPU job is far slower than expected, check memory and power first — it is usually the environment, not
   the model.

## Open items

- Product input step must remove the site rating word and decode HTML entities before scoring
  (the training data was cleaned; raw production input will not be).
- No standalone inference script yet; usage is through `reviews_pipeline.bert_unhappy_probs` (see README).
- Inference speed on client hardware not measured.
- Shipped model is a single seed (0); averaging the three seeds' probabilities would cut variance at 3× inference cost.
