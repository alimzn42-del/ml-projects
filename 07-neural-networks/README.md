# 07 — Neural networks (work in progress)

**Type:** first contact with PyTorch — image classification on CIFAR.
**Status:** working notebooks, not yet written up.

## Layout

```
first.ipynb          the very first PyTorch steps
test.ipynb           experiments
CIFAR-10.ipynb       the main notebook — CNN on CIFAR-10, grid search, inference on my_cat.jpg
cifar_best.pt        best trained model checkpoint (3.3 MB)
grid_results.json    grid-search experiment log
my_cat.jpg           out-of-distribution test image
data/                CIFAR-10/100 (auto-downloaded by torchvision) + queued Jena Climate data
```

`CIFAR-10.ipynb` downloads its data itself via `torchvision.datasets.CIFAR10(root="./data",
download=True)` — nothing needs fetching by hand.

**Queued next:** Jena Climate time-series forecasting — the dataset already sits in
`data/data/jena_climate_2009_2016.csv` and an empty placeholder notebook
(`data/Jena Climate.ipynb`) marks the intent.

This README becomes the full write-up when the project is finished, following the
pattern of projects 01–04.
