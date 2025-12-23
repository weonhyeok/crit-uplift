"""
Basic CVR (conversion) logistic regression for Criteo Uplift-style data.

Assumptions:
- df is already loaded (e.g., from parquet) and contains:
  f0..f11, treatment, exposure, visit, conversion
- All features are numeric (or can be cast to numeric).

Install (if needed):
  pip install pandas pyarrow scikit-learn

What it does:
1) (Optional) sample rows for speed
2) Select features + target
3) Basic cleaning (numeric casting, missing handling)
4) Train/test split (stratified)
5) Fit scaled logistic regression (stable on Windows)
6) Report AUC, LogLoss, baseline CVR
7) Print top coefficients
"""

from __future__ import annotations

import time
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss


# =========================
# 0) LOAD DATA (choose one)
# =========================
# Option A: If you already have df in memory, comment this out.
# Option B: Load from parquet here by setting PARQUET_PATH correctly.

PARQUET_PATH = r"D:\crit\crit-uplift\Data\criteo-uplift-v2.1.parquet"  # <-- adjust if needed

t_load0 = time.perf_counter()
df = pd.read_parquet(PARQUET_PATH)
t_load1 = time.perf_counter()
mem_gb = df.memory_usage(deep=True).sum() / (1024**3)
print(f"[Load] time={t_load1 - t_load0:.2f}s | mem={mem_gb:.2f}GB | shape={df.shape}")


# =========================
# 1) CONFIG
# =========================
RANDOM_STATE = 0
SAMPLE_N = None          # set to e.g. 2_000_000 for faster iteration
TEST_SIZE = 0.2

TARGET = "conversion"

FEATURES = [
    "f0","f1","f2","f3","f4","f5","f6",
    "f7","f8","f9","f10","f11",
    "treatment","exposure","visit",
]


# =========================
# 2) OPTIONAL SAMPLING
# =========================
if SAMPLE_N is not None and SAMPLE_N < len(df):
    df = df.sample(n=SAMPLE_N, random_state=RANDOM_STATE)
    print(f"[Sample] using n={len(df):,} rows")


# =========================
# 3) SELECT + CLEAN
# =========================
needed = FEATURES + [TARGET]
missing = [c for c in needed if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df_reg = df[needed].copy()

# Ensure target is 0/1 int
df_reg[TARGET] = df_reg[TARGET].astype(int)

# Cast features to numeric (robust if some are strings)
for c in FEATURES:
    df_reg[c] = pd.to_numeric(df_reg[c], errors="coerce")

# Fill missing values: median for numeric
for c in FEATURES:
    if df_reg[c].isna().any():
        df_reg[c] = df_reg[c].fillna(df_reg[c].median())

X = df_reg[FEATURES].astype(np.float32)   # float32 reduces memory + speeds up a bit
y = df_reg[TARGET].values

print(f"[Prep] X shape={X.shape} | y mean(CVR)={y.mean():.4%}")


# =========================
# 4) SPLIT
# =========================
t_split0 = time.perf_counter()

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

t_split1 = time.perf_counter()
print(f"[Split] time={t_split1 - t_split0:.2f}s | train={X_train.shape} | test={X_test.shape}")


# =========================
# 5) FIT LOGISTIC REGRESSION
# =========================
# Scaled logistic regression is a solid baseline.
# Using n_jobs=1 is often more stable on Windows (avoid thread oversubscription).

t_fit0 = time.perf_counter()

pipe = make_pipeline(
    StandardScaler(),  # dense numeric features -> good
    LogisticRegression(
        solver="lbfgs",
        penalty="l2",
        max_iter=200,
        n_jobs=1
    )
)

pipe.fit(X_train, y_train)

t_fit1 = time.perf_counter()
print(f"[Fit] time={t_fit1 - t_fit0:.2f}s")


# =========================
# 6) EVALUATE
# =========================
t_eval0 = time.perf_counter()

proba = pipe.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, proba)
ll = log_loss(y_test, proba)

t_eval1 = time.perf_counter()

print(f"[Eval] AUC={auc:.4f} | LogLoss={ll:.4f} | baseline CVR={y.mean():.4%} | time={t_eval1 - t_eval0:.2f}s")


# =========================
# 7) COEFFICIENTS (interpretation)
# =========================
# Retrieve coefficients from the logistic regression step.
logit = pipe.named_steps["logisticregression"]
coef = pd.Series(logit.coef_[0], index=FEATURES).sort_values(key=lambda s: s.abs(), ascending=False)

print("\n[Top coefficients by |value|]")
print(coef.head(15))

print("\n[All coefficients]")
print(coef)


# =========================
# 8) TOTAL TIME
# =========================
t_end = time.perf_counter()
print(f"\n[Total runtime] {t_end - t_load0:.2f}s")
