"""
Econometric-style full-sample logistic regression (no train/test) for CVR.

Goal (economist style):
- Estimate coefficients + robust standard errors
- Optional: cluster-robust SEs
- Optional: odds ratios + marginal effects

Requires:
  pip install pandas pyarrow statsmodels numpy

NOTE:
- Full-sample Logit on very large N can take time.
- If it runs too slowly, set SAMPLE_N to e.g. 3_000_000.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


# =========================
# 0) PATH + CONFIG
# =========================
PARQUET_PATH = Path(r"D:\crit\crit-uplift\Data\criteo-uplift-v2.1.parquet")

SAMPLE_N = None          # e.g. 3_000_000 if you want faster iteration
RANDOM_STATE = 0

TARGET = "conversion"

X_VARS = [
    "f0","f1","f2","f3","f4","f5","f6",
    "f7","f8","f9","f10","f11",
    "treatment","exposure","visit"
]

# Optional cluster variable (only if you have it in df)
CLUSTER_VAR = None       # e.g. "uid"  (set to None to use HC1 robust SE)


# =========================
# 1) LOAD
# =========================
t0 = time.perf_counter()
df = pd.read_parquet(PARQUET_PATH)
t1 = time.perf_counter()

mem_gb = df.memory_usage(deep=True).sum() / (1024**3)
print(f"[Load] time={t1 - t0:.2f}s | mem={mem_gb:.2f}GB | shape={df.shape}")

# Optional sampling (economists also do this for speed checks)
if SAMPLE_N is not None and SAMPLE_N < len(df):
    df = df.sample(n=SAMPLE_N, random_state=RANDOM_STATE)
    print(f"[Sample] using n={len(df):,} rows")


# =========================
# 2) SELECT + CLEAN
# =========================
needed = [TARGET] + X_VARS + ([CLUSTER_VAR] if CLUSTER_VAR else [])
missing = [c for c in needed if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df_reg = df[needed].copy()

# Ensure y is 0/1 integer
df_reg[TARGET] = pd.to_numeric(df_reg[TARGET], errors="coerce").astype("Int64")

# Cast X to numeric, handle missing
for c in X_VARS:
    df_reg[c] = pd.to_numeric(df_reg[c], errors="coerce")

# Drop rows with missing outcome
df_reg = df_reg.dropna(subset=[TARGET])
df_reg[TARGET] = df_reg[TARGET].astype(int)

# Fill missing X with median (simple, robust default)
for c in X_VARS:
    if df_reg[c].isna().any():
        df_reg[c] = df_reg[c].fillna(df_reg[c].median())

print(f"[Prep] n={len(df_reg):,} | mean(CVR)={df_reg[TARGET].mean():.4%}")

# Build design matrix with constant (economist standard)
X = sm.add_constant(df_reg[X_VARS], has_constant="add")
y = df_reg[TARGET]


# =========================
# 3) ESTIMATE LOGIT (FULL SAMPLE)
# =========================
t2 = time.perf_counter()

# Logit MLE; disp=True prints iteration output
logit_model = sm.Logit(y, X)
res = logit_model.fit(disp=True, maxiter=100)

t3 = time.perf_counter()
print(f"[Fit] time={t3 - t2:.2f}s")


# =========================
# 4) ROBUST / CLUSTERED SE (ECON STANDARD)
# =========================
t4 = time.perf_counter()

if CLUSTER_VAR:
    groups = df_reg[CLUSTER_VAR]
    res_rob = res.get_robustcov_results(cov_type="cluster", groups=groups)
    se_label = f"Cluster-robust (cluster={CLUSTER_VAR})"
else:
    res_rob = res.get_robustcov_results(cov_type="HC1")
    se_label = "Robust (HC1)"

t5 = time.perf_counter()
print(f"[SE] {se_label} | time={t5 - t4:.2f}s")

print("\n" + "=" * 90)
print(f"Logit results with {se_label} standard errors")
print("=" * 90)
print(res_rob.summary())


# =========================
# 5) ODDS RATIOS (INTERPRETATION)
# =========================
params = pd.Series(res_rob.params, index=res_rob.model.exog_names)
odds_ratios = np.exp(params)

print("\n" + "=" * 90)
print("Odds Ratios = exp(coef)")
print("=" * 90)
print(odds_ratios.sort_values(ascending=False))


# =========================
# 6) MARGINAL EFFECTS (OPTIONAL, ECON FRIENDLY)
# =========================
# For large N, this can be slower. It's still often useful for reporting.
try:
    mfx = res_rob.get_margeff(at="mean")
    print("\n" + "=" * 90)
    print("Average marginal effects at the mean")
    print("=" * 90)
    print(mfx.summary())
except Exception as e:
    print("\n[Marginal effects] skipped due to:", repr(e))


# =========================
# 7) SAVE A CLEAN COEF TABLE (OPTIONAL)
# =========================
coef_table = pd.DataFrame({
    "coef": res_rob.params,
    "se": res_rob.bse,
    "z": res_rob.tvalues,          # z-stats for logit
    "pvalue": res_rob.pvalues,
    "odds_ratio": np.exp(res_rob.params),
})
coef_table.index.name = "variable"

out_csv = Path("logit_cvr_results.csv")
coef_table.to_csv(out_csv)
print(f"\n[Saved] coefficient table -> {out_csv.resolve()}")

t_end = time.perf_counter()
print(f"\n[Total runtime] {t_end - t0:.2f}s")
