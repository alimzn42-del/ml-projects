# 07 — Neural network foundations (PyTorch / CIFAR)

**Type:** first contact with PyTorch — tensors, training loops, CNNs on CIFAR.
**Status:** working notebooks; the foundations that fed projects 09 and 10.

## Layout

```
first.ipynb          the very first PyTorch steps
test.ipynb           experiments
CIFAR-10.ipynb       the main notebook — CNN on CIFAR-10, grid search, inference on my_cat.jpg
cifar_best.pt        best trained model checkpoint (3.3 MB)
grid_results.json    grid-search experiment log
my_cat.jpg           out-of-distribution test image
data/                CIFAR-10/100 (auto-downloaded by torchvision; not committed)
```

`CIFAR-10.ipynb` downloads its data itself via `torchvision.datasets.CIFAR10(root="./data",
download=True)` — nothing needs fetching by hand.

The two projects that grew out of this folder now live on their own:
- [09-jena-climate](../09-jena-climate/) — temperature forecasting with recurrent networks
- [10-arabic-reviews](../10-arabic-reviews/) — unhappy-guest detection with AraBERT
