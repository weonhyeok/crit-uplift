
'''
pip install lightgbm scikit-uplift

'''

# =========================================================
# 0. Imports & Config
# =========================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import train_test_split

# =========================
# CONFIG
# =========================
PARQUET_PATH = Path(r"D:\crit\crit-uplift\Data\criteo-uplift-v2.1.parquet")

SAMPLE_N = None        # e.g. 3_000_000 for debugging
RANDOM_STATE = 0

TARGET = "conversion"
TREATMENT = "treatment"

FEATURES = [
    "f0","f1","f2","f3","f4","f5",
    "f6","f7","f8","f9","f10","f11"
]

# =========================================================
# 1. Load Data
# =========================================================
print("[Load]")
df = pd.read_parquet(PARQUET_PATH)

if SAMPLE_N is not None:
    df = df.sample(SAMPLE_N, random_state=RANDOM_STATE)

X = df[FEATURES]
y = df[TARGET].astype(int).values
t = df[TREATMENT].astype(int).values

print(f"Shape: {df.shape}")
print(f"CVR: {y.mean():.4%}")
print(f"Treatment ratio: {t.mean():.3f}")

# =========================================================
# 2. Train / Test Split (stratified by treatment)
# =========================================================
X_tr, X_te, y_tr, y_te, t_tr, t_te = train_test_split(
    X, y, t,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=t
)

# =========================================================
# 3. Outcome Models (μ1, μ0)
# =========================================================
print("[Fit outcome models]")

lgb_params = dict(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

mu1_model = LGBMClassifier(**lgb_params)
mu0_model = LGBMClassifier(**lgb_params)

mu1_model.fit(X_tr[t_tr == 1], y_tr[t_tr == 1])
mu0_model.fit(X_tr[t_tr == 0], y_tr[t_tr == 0])

mu1_hat = mu1_model.predict_proba(X_te)[:, 1]
mu0_hat = mu0_model.predict_proba(X_te)[:, 1]

# =========================================================
# 4. X-Learner Pseudo Outcomes
# =========================================================
print("[X-Learner]")

D = np.zeros_like(y_te, dtype=float)

# treatment group
D[t_te == 1] = y_te[t_te == 1] - mu0_hat[t_te == 1]

# control group
D[t_te == 0] = mu1_hat[t_te == 0] - y_te[t_te == 0]

# =========================================================
# 5. Uplift Model (Regression on D)
# =========================================================
uplift_model = LGBMRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

uplift_model.fit(X_te, D)
uplift_score = uplift_model.predict(X_te)

# =========================================================
# 6. Qini Curve & AUC (Custom)
# =========================================================
def qini_curve(y, t, uplift):
    order = np.argsort(-uplift)
    y, t = y[order], t[order]

    cum_t = np.cumsum(t)
    cum_c = np.cumsum(1 - t)

    cum_y_t = np.cumsum(y * t)
    cum_y_c = np.cumsum(y * (1 - t))

    eps = 1e-9
    qini = cum_y_t - cum_y_c * (cum_t / (cum_c + eps))
    return qini


def qini_auc(y, t, uplift):
    q = qini_curve(y, t, uplift)
    return np.trapz(q, dx=1.0 / len(q))


qini = qini_curve(y_te, t_te, uplift_score)
qini_score = qini_auc(y_te, t_te, uplift_score)

print(f"Qini AUC: {qini_score:.6f}")

# =========================================================
# 7. Plot
# =========================================================
plt.figure(figsize=(8, 5))
plt.plot(qini, label="X-Learner (LGBM)")
plt.plot(np.zeros_like(qini), "--", label="Random")
plt.title("Qini Curve")
plt.xlabel("Population fraction")
plt.ylabel("Incremental conversions")
plt.legend()
plt.tight_layout()
plt.show()

