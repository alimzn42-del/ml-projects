import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sentencepiece as spm
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup

LABEL_MAP = {1: "neg", 2: "neg", 3: "neu", 4: "pos", 5: "pos"}
LABEL_IDS = {"neg": 0, "neu": 1, "pos": 2}
META = ["Hotel name", "rating", "user type", "room type", "nights", "review"]
EMPTY_CHARS = "[\\s\\.\u060C,!\u061F?:;\\-\"\u201C\u201D'()]"


def load_raw(path):
    head = Path(path).read_bytes()[:4]
    enc = "utf-16" if head[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8-sig"
    return pd.read_csv(path, sep="\t", encoding=enc, quoting=3)


def _split_first(text):
    unq = text.str.lstrip('"').str.strip()
    parts = unq.str.split(".", n=1)
    first = parts.str[0].str.strip()
    rest = parts.str[1].fillna("").str.strip().str.rstrip('"').str.strip()
    return first, rest


def find_site_labels(df, min_count=500):
    first, _ = _split_first(df["review"].astype(str))
    stats = df.assign(first=first).groupby("first")["rating"].agg(["count", "nunique"])
    keep = (stats["nunique"] == 1) & (stats["count"] >= min_count)
    return sorted(stats[keep].index)


def strip_site_labels(df, site_labels):
    out = df.copy()
    raw = out["review"].astype(str)
    first, rest = _split_first(raw)
    out["had_site_label"] = first.isin(site_labels)
    out["review_clean"] = raw.where(~out["had_site_label"], rest)
    return out


def clean(df):
    out = df.drop_duplicates(subset=META, keep="first")
    core = out["review_clean"].str.replace(EMPTY_CHARS, "", regex=True)
    out = out[core.str.len() > 0].copy()
    out["label"] = out["rating"].map(LABEL_MAP)
    return out


def split_by_hotel(df, seed=42, train=0.8, val=0.1):
    rng = np.random.default_rng(seed)
    hotels = np.array(df["Hotel name"].unique(), dtype=object)
    rng.shuffle(hotels)
    sizes = df["Hotel name"].value_counts().reindex(hotels)
    cum = sizes.cumsum() / sizes.sum()
    split = np.where(cum <= train, "train", np.where(cum <= train + val, "val", "test"))
    out = df.copy()
    out["split"] = out["Hotel name"].map(pd.Series(split, index=hotels))
    return out


def load_or_build(raw_path, cache_path, site_labels_path):
    cache_path = Path(cache_path)
    if cache_path.exists():
        return pd.read_parquet(cache_path)
    raw = load_raw(raw_path)
    site_labels = json.loads(Path(site_labels_path).read_text(encoding="utf-8"))
    data = split_by_hotel(clean(strip_site_labels(raw, site_labels)))
    data.to_parquet(cache_path, index=False)
    return data


def train_tokenizer(texts, prefix, vocab_size=16000):
    prefix = Path(prefix)
    model_file = prefix.parent / (prefix.name + ".model")
    if not model_file.exists():
        corpus = prefix.parent / (prefix.name + "_corpus.txt")
        corpus.write_text("\n".join(texts), encoding="utf-8")
        spm.SentencePieceTrainer.train(
            input=str(corpus), model_prefix=str(prefix), vocab_size=vocab_size,
            model_type="bpe", character_coverage=0.9999,
        )
    return spm.SentencePieceProcessor(model_file=str(model_file))


def add_tokens(df, sp):
    out = df.copy()
    out["tokens"] = [" ".join(sp.encode(t, out_type=str)) for t in out["review_clean"]]
    return out


def encode_ids(df, sp, max_len=256):
    return [sp.encode(t)[:max_len] for t in df["review_clean"]]


def make_batch(seqs, pad_id):
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s)
    mask = ids != pad_id
    return ids, mask


def length_batches(lengths, batch_size, seed, chunk=100):
    rng = np.random.default_rng(seed)
    lengths = np.asarray(lengths)
    idx = rng.permutation(len(lengths))
    step = batch_size * chunk
    batches = []
    for start in range(0, len(idx), step):
        part = idx[start:start + step]
        part = part[np.argsort(lengths[part], kind="stable")]
        batches += [part[i:i + batch_size] for i in range(0, len(part), batch_size)]
    rng.shuffle(batches)
    return batches


class MeanPoolClassifier(nn.Module):
    def __init__(self, vocab_size, pad_id, dim=100, n_classes=3):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim, padding_idx=pad_id)
        self.out = nn.Linear(dim, n_classes)

    def forward(self, ids, mask):
        e = self.emb(ids)
        m = mask.unsqueeze(-1).float()
        pooled = (e * m).sum(1) / m.sum(1)
        return self.out(pooled)


class AttentionClassifier(nn.Module):
    def __init__(self, vocab_size, pad_id, dim=100, n_classes=3):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim, padding_idx=pad_id)
        self.score = nn.Linear(dim, 1)
        self.out = nn.Linear(dim, n_classes)

    def forward(self, ids, mask, return_weights=False):
        e = self.emb(ids)
        s = self.score(e).squeeze(-1)
        s = s.masked_fill(~mask, float("-inf"))
        a = torch.softmax(s, dim=1)
        pooled = (a.unsqueeze(-1) * e).sum(1)
        logits = self.out(pooled)
        return (logits, a) if return_weights else logits


class ConvAttentionClassifier(nn.Module):
    def __init__(self, vocab_size, pad_id, dim=100, n_classes=3, width=3,
                 dropout=0.0, emb_dropout=0.0):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim, padding_idx=pad_id)
        self.emb_drop = nn.Dropout(emb_dropout)
        self.conv = nn.Conv1d(dim, dim, kernel_size=width, padding="same")
        self.drop = nn.Dropout(dropout)
        self.score = nn.Linear(dim, 1)
        self.out = nn.Linear(dim, n_classes)

    def forward(self, ids, mask, return_weights=False):
        e = self.emb_drop(self.emb(ids))
        h = torch.relu(self.conv(e.transpose(1, 2))).transpose(1, 2)
        h = self.drop(h)
        s = self.score(h).squeeze(-1)
        s = s.masked_fill(~mask, float("-inf"))
        a = torch.softmax(s, dim=1)
        pooled = (a.unsqueeze(-1) * h).sum(1)
        logits = self.out(pooled)
        return (logits, a) if return_weights else logits


class GRUAttentionClassifier(nn.Module):
    def __init__(self, vocab_size, pad_id, dim=100, n_classes=3, emb_dropout=0.0):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim, padding_idx=pad_id)
        self.emb_drop = nn.Dropout(emb_dropout)
        self.gru = nn.GRU(dim, dim // 2, batch_first=True, bidirectional=True)
        self.score = nn.Linear(dim, 1)
        self.out = nn.Linear(dim, n_classes)

    def forward(self, ids, mask, return_weights=False):
        e = self.emb_drop(self.emb(ids))
        lengths = mask.sum(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(e, lengths, batch_first=True,
                                                   enforce_sorted=False)
        out, _ = self.gru(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True,
                                                total_length=ids.size(1))
        s = self.score(h).squeeze(-1)
        s = s.masked_fill(~mask, float("-inf"))
        a = torch.softmax(s, dim=1)
        pooled = (a.unsqueeze(-1) * h).sum(1)
        logits = self.out(pooled)
        return (logits, a) if return_weights else logits


class ConvSelfAttentionClassifier(nn.Module):
    def __init__(self, vocab_size, pad_id, dim=100, n_classes=3, width=3, emb_dropout=0.0):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim, padding_idx=pad_id)
        self.emb_drop = nn.Dropout(emb_dropout)
        self.conv = nn.Conv1d(dim, dim, kernel_size=width, padding="same")
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.score = nn.Linear(dim, 1)
        self.out = nn.Linear(dim, n_classes)

    def forward(self, ids, mask, return_weights=False):
        e = self.emb_drop(self.emb(ids))
        c = torch.relu(self.conv(e.transpose(1, 2))).transpose(1, 2)
        q, k, v = self.q(c), self.k(c), self.v(c)
        s = q @ k.transpose(1, 2) / (c.size(-1) ** 0.5)
        s = s.masked_fill(~mask.unsqueeze(1), float("-inf"))
        h = self.norm(c + torch.softmax(s, dim=-1) @ v)
        t = self.score(h).squeeze(-1)
        t = t.masked_fill(~mask, float("-inf"))
        a = torch.softmax(t, dim=1)
        pooled = (a.unsqueeze(-1) * h).sum(1)
        logits = self.out(pooled)
        return (logits, a) if return_weights else logits


def class_weights(labels, n_classes=3):
    counts = np.bincount(labels, minlength=n_classes)
    return torch.tensor(len(labels) / (n_classes * counts), dtype=torch.float)


def predict(model, seqs, pad_id, batch_size=1024):
    model.eval()
    device = next(model.parameters()).device
    order = np.argsort([len(s) for s in seqs])
    preds = np.empty(len(seqs), dtype=int)
    with torch.no_grad():
        for i in range(0, len(order), batch_size):
            b = order[i:i + batch_size]
            ids, mask = make_batch([seqs[j] for j in b], pad_id)
            preds[b] = model(ids.to(device), mask.to(device)).argmax(1).cpu().numpy()
    return preds

def train_model(model, tr_seqs, tr_y, va_seqs, va_y, pad_id, epochs=10, lr=1e-3,
                batch_size=256, patience=2, seed=0, n_classes=3, score_fn=None,
                score_name="val macro"):
    if score_fn is None:
        score_fn = lambda y, p: f1_score(y, p, average="macro")
    device = next(model.parameters()).device
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(tr_y, n_classes).to(device))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lens = [len(s) for s in tr_seqs]
    best, best_state, wait, history = -1.0, None, 0, []
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for b in length_batches(lens, batch_size, seed + epoch):
            ids, mask = make_batch([tr_seqs[i] for i in b], pad_id)
            y = torch.tensor(tr_y[b], device=device)
            loss = loss_fn(model(ids.to(device), mask.to(device)), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        va_score = score_fn(va_y, predict(model, va_seqs, pad_id))
        history.append({"epoch": epoch + 1, "train loss": total / len(tr_y), score_name: va_score})
        print(history[-1])
        if va_score > best:
            best, wait = va_score, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return pd.DataFrame(history)


def run_seeds(make_model, name, tr_seqs, tr_y, va_seqs, va_y, pad_id, models_dir,
              seeds=(0, 1, 2), device="cpu", **train_kw):
    score_name = train_kw.get("score_name", "val macro")
    rows = []
    for seed in seeds:
        torch.manual_seed(seed)
        model = make_model().to(device)
        hist = train_model(model, tr_seqs, tr_y, va_seqs, va_y, pad_id, seed=seed, **train_kw)
        rows.append({"model": name, "seed": seed, score_name: hist[score_name].max(),
                     "best epoch": int(hist[score_name].idxmax()) + 1})
        torch.save(model.state_dict(), Path(models_dir) / f"{name}_seed{seed}.pt")
    return pd.DataFrame(rows)


def bert_batch(tok, texts, max_len, device):
    return tok(texts, truncation=True, max_length=max_len, padding=True,
               return_tensors="pt").to(device)


def predict_bert(model, tok, texts, max_len=256, batch_size=128):
    model.eval()
    device = next(model.parameters()).device
    order = np.argsort([len(t) for t in texts])
    preds = np.empty(len(texts), dtype=int)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for i in range(0, len(order), batch_size):
            b = order[i:i + batch_size]
            enc = bert_batch(tok, [texts[j] for j in b], max_len, device)
            preds[b] = model(**enc).logits.argmax(1).cpu().numpy()
    return preds


def train_bert(name, tok, tr_texts, tr_y, tr_lens, va_texts, va_y, device, seed=0,
               epochs=3, lr=2e-5, batch_size=32, max_len=256, warmup=0.1, log_every=2000):
    torch.manual_seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=3).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(tr_y).to(device))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * len(length_batches(tr_lens, batch_size, seed))
    sched = get_linear_schedule_with_warmup(opt, int(warmup * steps), steps)
    scaler = torch.amp.GradScaler("cuda")
    best, best_state, history, t0 = -1.0, None, [], time.time()
    for epoch in range(epochs):
        model.train()
        total = 0.0
        batches = length_batches(tr_lens, batch_size, seed + epoch)
        for k, b in enumerate(batches, 1):
            enc = bert_batch(tok, [tr_texts[i] for i in b], max_len, device)
            y = torch.tensor(tr_y[b], device=device)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = loss_fn(model(**enc).logits.float(), y)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item() * len(b)
            if k % log_every == 0:
                print(f"epoch {epoch + 1} batch {k}/{len(batches)} {(time.time() - t0) / 60:.1f} min")
        va_f1 = f1_score(va_y, predict_bert(model, tok, va_texts, max_len), average="macro")
        history.append({"epoch": epoch + 1, "train loss": total / len(tr_y),
                        "val macro": va_f1, "min": (time.time() - t0) / 60})
        print(history[-1])
        if va_f1 > best:
            best = va_f1
            best_state = {n: v.detach().cpu().clone() for n, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, pd.DataFrame(history)

def bert_unhappy_probs(model, tok, texts, max_len=128, batch_size=128):
    model.eval()
    device = next(model.parameters()).device
    order = np.argsort([len(t) for t in texts])
    probs = np.empty(len(texts))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for i in range(0, len(order), batch_size):
            b = order[i:i + batch_size]
            enc = bert_batch(tok, [texts[j] for j in b], max_len, device)
            p = torch.softmax(model(**enc).logits.float(), dim=1)
            probs[b] = (p[:, 0] + p[:, 1]).cpu().numpy()
    return probs