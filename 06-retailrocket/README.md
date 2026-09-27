# 06 — Retailrocket (work in progress)

**Type:** implicit-feedback e-commerce behaviour data (view / addtocart / transaction events)
**Status:** exploration underway in `notebook.ipynb` — not yet written up.

The dataset logs 4.5 months of visitor events from a real e-commerce site: 2.76M events
in `events.csv`, plus item properties and a category tree. Unlike project 04 (invoices
with amounts), there is no monetary value here — the signal is behavioural, which is the
point of picking it.

## Layout

```
notebook.ipynb      the working notebook (reads data/events.csv)
data/               Retailrocket.zip + extracted files — see data/README.md
```

This README becomes the full write-up when the project is finished, following the
pattern of projects 01–04.
