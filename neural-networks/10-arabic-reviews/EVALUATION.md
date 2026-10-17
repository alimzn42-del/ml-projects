# Evaluation Report — Unhappy Guest Detection

## 1. Data

- Source: HARD (Hotel Arabic Reviews Dataset), `unbalanced-reviews.txt`.
- After cleaning: **409,312 reviews**.
- Cleaning:
  - Removed the rating word added by the site at the start of reviews (a word that always appears with one single
    rating and at least 500 times). Keeping it would leak the label.
  - Removed exact duplicates (same hotel, rating, user type, room type, nights and text).
  - Removed reviews with no real text (only spaces or punctuation).
- Labels: 1–2 stars = negative, 3 = neutral, 4–5 = positive. Final task: unhappy = 1–3 stars.

### Split (by hotel, seed 42)

| Split | Reviews | Unhappy share |
|---|---|---|
| train | 327,264 | — |
| val | 40,024 | 30.0% |
| test | 42,024 | 35.1% |

Each hotel appears in one split only, so the model is always tested on hotels it never saw.
The test set was used **once**, after the model, task and threshold were fixed on val.

## 2. Model selection (val, 3 seeds each, macro F1 on 3 classes)

| Model | val macro F1 (mean, range) |
|---|---|
| TF-IDF + Logistic Regression (C = 0.3) | 0.650 |
| Embedding + mean pooling | 0.609 (0.607–0.611) |
| Embedding + attention pooling | 0.619 |
| CNN + attention | 0.650 |
| CNN + attention + embedding dropout 0.3 | 0.658 (0.655–0.661) |
| CNN + self-attention | 0.660 |
| Bidirectional GRU + attention + embedding dropout | 0.670 (0.667–0.673) |
| Same GRU, max length 128 | 0.672 (0.670–0.674) |
| **AraBERT fine-tuned, max length 128** | **0.699 (0.696–0.702)** |

A difference was accepted as real only when the seed ranges did not overlap.

### On the client task (val, unhappy F1, argmax)

| Model | Precision | Recall | Unhappy F1 |
|---|---|---|---|
| GRU, 3 classes read as binary | 0.669 | 0.841 | 0.745 |
| GRU trained binary | 0.715 | 0.793 | 0.752 (0.751–0.753) |
| **AraBERT, 3 classes read as binary** | 0.682 | **0.877** | **0.767 (0.765–0.768)** |

AraBERT was chosen: highest unhappy F1 with non-overlapping ranges, and the highest recall,
which matters most to the client (a missed unhappy guest costs more than a call to a happy one).
Seed 0 was shipped: seeds differ by less than 0.003, and seed 0 is the one the error audit was done on.

Cost: AraBERT has ~135M parameters versus ~1.6M for the GRU and needs a GPU to be fast.

### AraBERT training settings

AdamW, lr 2e-5, weight decay 0.01, 10% linear warmup then linear decay, batch 32, 3 epochs,
mixed precision (fp16), class-weighted cross-entropy (weights from train only), length-bucketed batches,
best epoch kept by val macro F1 (best epochs: 2, 3, 3). About 24 minutes per epoch on an RTX 4050.

## 3. Threshold (val, seed 0)

`p_unhappy = p(neg) + p(neu)`, flag if `p_unhappy >= threshold`.

| Threshold | Caught | Missed | False alarms | Flagged | Recall | Precision |
|---|---|---|---|---|---|---|
| 0.1 | 11,776 | 225 | 13,716 | 25,492 | 0.981 | 0.462 |
| 0.2 | 11,518 | 483 | 9,798 | 21,316 | 0.960 | 0.540 |
| 0.3 | 11,219 | 782 | 7,630 | 18,849 | 0.935 | 0.595 |
| **0.4** | 10,911 | 1,090 | 6,208 | 17,119 | 0.909 | 0.637 |
| 0.5 | 10,578 | 1,423 | 5,156 | 15,734 | 0.881 | 0.672 |
| 0.6 | 10,168 | 1,833 | 4,163 | 14,331 | 0.847 | 0.710 |

Cost of each extra unhappy guest caught:

| Step | Extra caught | Extra false alarms | False alarms per extra catch |
|---|---|---|---|
| 0.5 → 0.4 | 333 | 1,052 | 3.2 |
| 0.4 → 0.3 | 308 | 1,422 | 4.6 |
| 0.3 → 0.2 | 299 | 2,168 | 7.3 |

**0.4 chosen**: below it, the price of each extra catch rises quickly.
The final value is a business decision and depends on how many calls the care team can make.

## 4. Final test results (one run, seed 0, threshold 0.4)

| Metric | Value |
|---|---|
| Reviews | 42,024 |
| Caught (TP) | 13,537 |
| Missed (FN) | 1,221 |
| False alarms (FP) | 6,643 |
| Flagged | 20,180 |
| Recall | 0.917 |
| Precision | 0.671 |
| Unhappy F1 | 0.775 |
| Binary accuracy | 0.813 |

Reference: predicting "happy" for everyone gives 0.649 accuracy and 0 recall.

3-class results on test: macro F1 **0.706**, accuracy 0.764.

```
              pred neg  pred neu  pred pos
true neg         4354      1448       223
true neu         1613      5613      1507
true pos          342      4803     22121
```

Test and val agree (macro 0.706 vs 0.702 on val for seed 0; recall 0.917 vs 0.909 at threshold 0.4).
The test set has more unhappy guests (35.1% vs 30.0%), so the precision is not lower even with more flags.

## 5. Error analysis

- **Label noise.** A blind audit of 120 val reviews (90 errors, 30 correct, mixed and unlabeled):
  of the 90 errors, the star label was wrong in 58, the model was wrong in 21, neither was clearly right in 11.
  On correct predictions, the auditor agreed with the star label only 70% of the time.
  The model is close to the ceiling the labels allow.
- **Neutral is the hard class.** Most errors are neutral ↔ positive and neutral ↔ negative.
  Titles such as "جيد" appear with 3 stars 37% of the time and with 4 stars 54% of the time:
  guests themselves use the same words for different ratings.
- **Truncation.** 610 val and 644 test reviews are longer than 128 tokens. Long reviews are more often negative
  (29.3% vs 12.6%). The same GRU at 128 and 256 tokens scored the same, so the cut does not change the result.
- **HTML entities** (`&amp;`, `&lt;`, `&gt;`, `&nbsp;`) appear in a very small share of reviews (about 0.02% for
  `&lt;`). Not retrained for; decoded in the product input step.
- **Title check.** Review titles written by guests are linked to the rating but never fully tied to one rating
  (highest: "ممتاز جدا", 89% 5 stars). They are the guest's own words, available at prediction time: not a leak.

## 6. Future work

- **Review sections.** Reviews are built from site fields: title, "what I liked", "what I didn't like", joined by
  ". ". 86.8% of reviews split cleanly into 2–3 parts; 6.8% have no separator and 6.4% have extra full stops
  inside the text. Marking each section explicitly could help the model tell praise from complaints
  (e.g. "النظافة" in "liked" vs "didn't like"). Expected gain is small given the label-noise ceiling. Not tested.
- Train AraBERT directly on the binary task and compare at equal recall.
- Measure inference speed and memory on the client's hardware.
