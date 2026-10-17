# Evaluation — Arabic Hotel Reviews

All numbers come from `arabic-reviews.ipynb`. Model selection and the threshold were decided on the
**validation set**; the **test set was opened once**, at the end, with everything frozen.

## Data and split

- Source: HARD unbalanced hotel reviews (Booking.com, Arabic). After cleaning: **409,312 reviews**.
- Labels from the star rating: `neg` (1–2★) · `neu` (3★) · `pos` (4–5★). The client target
  **unhappy = neg + neu** (1–3★).
- **Split by hotel, not by review** — train 327,264 · val 40,024 · test 42,024.
  Hotel overlap verified: train–val 0, train–test 0, val–test 0.
- The site's rating words (e.g. "ممتاز.") are stripped before any model sees the text (label leak).

## Model ladder — validation, 3-class macro F1

| Model | val macro F1 | Notes |
|---|---|---|
| TF-IDF + LogisticRegression (C=0.3, balanced) | 0.650 | the baseline every network had to beat |
| GRU + attention, emb-dropout 0.3, len 128 (3 seeds) | 0.670 / 0.674 / 0.670 | best of the small-network family; mean 0.672 |
| AraBERT fine-tuned, 3-class (seed 0, best epoch) | **0.702** | pretrained wins by ~3 points over networks trained from scratch |

The intermediate small-network experiments (mean-pool, attention, conv + attention, dropout
placements, self-attention) plateaued between 0.60 and 0.67 — the full epoch logs are in the notebook.

## Binary unhappy task — validation (threshold 0.5), 3 seeds each

| Model | precision | recall | unhappy F1 |
|---|---|---|---|
| GRU trained binary | 0.715 | 0.793 | 0.751 / 0.753 / 0.752 |
| **AraBERT 3-class read as binary** | 0.682 | 0.877 | **0.767 / 0.765 / 0.768** |

The seed ranges do not overlap → AraBERT wins on the client metric **without any new training**,
and with clearly higher recall (the error the client cares about).

## Threshold sweep — validation, AraBERT seed 0

| threshold | caught | missed | false alarms | flagged | recall | precision |
|---|---|---|---|---|---|---|
| 0.1 | 11,776 | 225 | 13,716 | 25,492 | 0.981 | 0.462 |
| 0.2 | 11,518 | 483 | 9,798 | 21,316 | 0.960 | 0.540 |
| 0.3 | 11,219 | 782 | 7,630 | 18,849 | 0.935 | 0.595 |
| **0.4** | **10,911** | **1,090** | **6,208** | **17,119** | **0.909** | **0.637** |
| 0.5 | 10,578 | 1,423 | 5,156 | 15,734 | 0.881 | 0.672 |
| 0.6 | 10,168 | 1,833 | 4,163 | 14,331 | 0.847 | 0.710 |

**0.4 chosen:** below it each extra caught guest costs more than 4 false alarms; the final value
belongs to the client (it is a cost decision, not a modelling one).

## Final test — opened once (threshold 0.4, `arabert_seed0.pt`)

42,024 reviews, hotels never seen in training:

| | |
|---|---|
| unhappy guests caught | **13,537 / 14,758** |
| missed | 1,221 |
| false alarms | 6,643 |
| flagged for a call | 20,180 (48% of volume) |
| **recall** | **0.917** |
| **precision** | **0.671** |
| **unhappy F1** | **0.775** |
| 3-class macro F1 | 0.706 |

Test came out slightly better than validation — consistent with no late selection on the test set.

## Why improvement stopped — the blind audit

120 misclassified validation reviews were re-labelled blind (without seeing model prediction or label):

| verdict | count |
|---|---|
| the **label** was wrong | 58 |
| the **model** was wrong | 21 |
| neither clearly | 11 |

About half of the remaining "errors" are noise in the star ratings themselves (mixed reviews rated
3★, positive text with low stars, etc.). The expected gain of further modelling is about the size of
the seed-to-seed spread, so work stopped here.
