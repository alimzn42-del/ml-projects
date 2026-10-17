import numpy as np
import pandas as pd

FEATURES = ["T (degC)", "p (mbar)", "rh (%)", "Tdew (degC)", "wv (m/s)", "wx", "wy",
            "hour_sin", "hour_cos", "year_sin", "year_cos"]
TARGET = "T (degC)"
WINDOW, HORIZON = 96, 24
BIG_GAP = pd.Timedelta(minutes=30)
YEAR_SEC = 365.2425 * 24 * 3600


def load_raw(path):
    df = pd.read_csv(path)
    df["Date Time"] = pd.to_datetime(df["Date Time"], format="%d.%m.%Y %H:%M:%S")
    return df


def clean(df):
    df = df.drop_duplicates().sort_values("Date Time").reset_index(drop=True)
    df["wv (m/s)"] = df["wv (m/s)"].replace(-9999.0, np.nan)
    df["segment"] = (df["Date Time"].diff() > BIG_GAP).cumsum()
    parts = []
    for s, g in df.groupby("segment"):
        g = g.set_index("Date Time")
        g = g.reindex(pd.date_range(g.index[0], g.index[-1], freq="10min"))
        g = g.interpolate(method="time")
        g["segment"] = s
        parts.append(g)
    return pd.concat(parts).rename_axis("Date Time")


def to_hourly(df):
    df = df.copy()
    rad = np.deg2rad(df["wd (deg)"])
    df["wx"] = df["wv (m/s)"] * np.cos(rad)
    df["wy"] = df["wv (m/s)"] * np.sin(rad)
    keep = ["T (degC)", "p (mbar)", "rh (%)", "Tdew (degC)", "wv (m/s)", "wx", "wy"]
    parts = []
    for s, g in df.groupby("segment"):
        h = g[keep].resample("1h").mean()
        h["segment"] = s
        parts.append(h)
    h = pd.concat(parts)
    hour = h.index.hour + h.index.minute / 60
    h["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    h["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    sec = h.index.astype("int64") / 1e9
    h["year_sin"] = np.sin(2 * np.pi * sec / YEAR_SEC)
    h["year_cos"] = np.cos(2 * np.pi * sec / YEAR_SEC)
    return h


def split_by_year(h):
    y = h.index.year
    return h[y <= 2014], h[y == 2015], h[y == 2016]


def fit_scaler(train):
    return train[FEATURES].mean(), train[FEATURES].std()


def make_windows(h, mean, std):
    Xs, ys, base = [], [], []
    for _, g in h.groupby("segment"):
        v = ((g[FEATURES] - mean) / std).to_numpy(np.float32)
        t = g[TARGET].to_numpy(np.float32)
        for end in range(WINDOW, len(g) - HORIZON + 1):
            Xs.append(v[end - WINDOW:end])
            ys.append(t[end - 1 + HORIZON])
            base.append(t[end - 1])
    return np.stack(Xs), np.array(ys), np.array(base)