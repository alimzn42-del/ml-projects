# Jena Climate — Temperature Forecasting with Recurrent Networks

**Delivered:** 4 Oct 2026 · **Task:** predict the temperature 24 hours ahead from the last 96 hours
of one weather station's readings (Jena, 2009–2016).

*(الأصل العربي الكامل لوثيقة التسليم محفوظ في [DELIVERY.ar.md](DELIVERY.ar.md))*

## Result

Final model: **single-layer GRU, 32 units, dropout 0.3 on the last summary before the output head —
4,353 weights**, restored to the best-validation epoch. Test = one full year no decision was made on.

| | test MAE (°C) | vs baseline |
|---|---|---|
| Baseline: "same hour yesterday" | 2.473 | — |
| **GRU-32** | **2.238** | **−9.5%** |

One-line conclusion: the network beats "same as yesterday" by ~10%, and the remaining error is
bounded by the information one station carries — a weather front arriving from far away is not in
its past readings.

## Data preparation (the part that mattered most)

- Full audit exposed **large hidden gaps** (the first check had only printed `head()`); data was cut
  into segments at every gap > 30 min, and filling happens **within a segment only** — no
  interpolation across a gap.
- Hourly means within each segment; **wind converted to a vector (wx, wy) before averaging**,
  because averaging angles lies.
- Cyclic (sin/cos) encoding of hour and season; no raw "year" column — the test year is one the
  network never saw.
- **Split by calendar years** (train ≤ 2014, one year val, one year test), scaling fit on train only.
- Window: 96 h in → T at +24 h, chosen from the correlation table (pressure change over the last
  48–96 h correlates most with ΔT at +24 h).
- ~52k windows, but consecutive windows share 95 of 96 hours → independent information ≈ number of
  days (~2,200). That number, not 52k, is what governs overfitting.

## The diagnostic ladder

Each step answered the previous step's diagnosis — the full story is in `DELIVERY.ar.md` §3.

| Step | Model (weights) | Best val MAE | Diagnosis |
|---|---|---|---|
| 1 | Dense 69,761 | — | memorised by epoch 2 → need weight sharing across hours |
| 2 | SimpleRNN-32 (1,473) | 2.545 | stuck — gradient at hour 96 is 5×10⁻⁷ of hour 1: **vanishing gradient, blind to most of the window** |
| 3 | SimpleRNN-68 (5,577) | 2.517 | same collapse → size is not the problem |
| 4 | LSTM-32 (5,793) | 2.554 | gradient at hour 96 now 3×10⁻² — decay solved, but it **memorised** (train 1.93, val rose) |
| 5 | LSTM-32 + dropout 0.3 on h_96 (3 seeds) | 2.538 ± 0.008 | dropout *between* hours rejected — it corrupts the memory path the gates fixed |
| 6 | **GRU-32 + dropout (3 seeds)** | **2.484 ± 0.013** | best in all seeds; GRU-16/64 overlap or re-memorise → 32, the simplest that wins |

Baseline on train years = 2.595 vs val year = 2.764: the val year is simply harder (×1.065) —
measured with a model-free tool, so year difficulty is separated from network weakness.

## Lessons (condensed from the delivery doc)

1. **Diagnose before treating** — the gradient-per-hour plot, not the loss number, identified
   vanishing gradients as the real problem.
2. **Change one thing at a time** — LSTM added gates *and* weights; the bigger SimpleRNN isolated
   the two causes and proved the gates were the difference.
3. An anti-memorisation tool **stabilises** a network; it does not raise its ceiling.
4. Growing weight norms alone are not evidence of memorisation — gates need large weights to
   saturate; the validation curve decides.
5. **When very different networks stop at the same number, the ceiling is in the information**, not
   the model — same rule as projects 02/05.
6. One seed = one data point. A difference counts when it clearly exceeds the spread.
7. The honest number is **relative to a dumb baseline, on data no decision touched** — raw MAE is
   polluted by year difficulty.

## Limits and proposed next steps

Single station is the binding limit (biggest expected gain: data from neighbouring stations).
Also untried: seed-averaged predictions, error analysis by season/hour, an LSTM sized exactly like
the GRU to test the "built-in constraints" hypothesis directly, forecasting all 24 hours at once.

## Files and usage

| File | Contents |
|---|---|
| `Jena Climate.ipynb` | all experiments in order: audit, baseline, networks, measurements |
| `jena_pipeline.py` | loading, cleaning, hourly aggregation, split, scaling, window building |
| `models/gru32_drop03.pt` | final model: best-epoch weights + config + train-set scaler numbers |

```python
ckpt = torch.load("models/gru32_drop03.pt", weights_only=False)
cfg = ckpt["config"]           # cell="gru", hidden=32, p_drop=0.3, window=96, horizon=24
model = RecurrentRegressor(cfg["cell"], hidden=cfg["hidden"], p_drop=cfg["p_drop"])
model.load_state_dict(ckpt["state_dict"]); model.eval()
# input: 96 consecutive gap-free hours × 11 features, scaled with ckpt["scaler"] (train numbers!)
# output is scaled: T = y_hat * T_std + T_mean
```
