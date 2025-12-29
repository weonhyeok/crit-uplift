# =========================================================
# 0. Imports & Config
# =========================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import train_test_split

# =========================================================
# CONFIG
# =========================================================
PARQUET_PATH = Path(r"D:\crit\crit-uplift\Data\criteo-uplift-v2.1.parquet")

SAMPLE_N = None        # 예: 3_000_000
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
# 2. Train / Test Split
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
lgb_clf_params = dict(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

mu1_model = LGBMClassifier(**lgb_clf_params)
mu0_model = LGBMClassifier(**lgb_clf_params)

mu1_model.fit(X_tr[t_tr == 1], y_tr[t_tr == 1])
mu0_model.fit(X_tr[t_tr == 0], y_tr[t_tr == 0])

# =========================================================
# 4. X-Learner pseudo outcome
# =========================================================
mu1_tr = mu1_model.predict_proba(X_tr)[:, 1]
mu0_tr = mu0_model.predict_proba(X_tr)[:, 1]

D_tr = np.zeros_like(y_tr, dtype=float)
D_tr[t_tr == 1] = y_tr[t_tr == 1] - mu0_tr[t_tr == 1]
D_tr[t_tr == 0] = mu1_tr[t_tr == 0] - y_tr[t_tr == 0]

# =========================================================
# 5. Uplift model
# =========================================================
lgb_reg_params = dict(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

uplift_model = LGBMRegressor(**lgb_reg_params)
uplift_model.fit(X_tr, D_tr)

uplift_score = uplift_model.predict(X_te)

# =========================================================
# 6. Qini Curve
# =========================================================
def qini_curve(y, t, uplift):
    order = np.argsort(-uplift)
    y, t = y[order], t[order]

    cum_t = np.cumsum(t)
    cum_c = np.cumsum(1 - t)

    cum_y_t = np.cumsum(y * t)
    cum_y_c = np.cumsum(y * (1 - t))

    qini = cum_y_t - cum_y_c * (cum_t / (cum_c + 1e-9))
    return qini

qini = qini_curve(y_te, t_te, uplift_score)

plt.figure(figsize=(8, 5))
plt.plot(np.arange(len(qini)) / len(qini), qini, label="X-Learner")
plt.axhline(0, linestyle="--")
plt.legend()
plt.title("Qini Curve")
plt.show()

# =========================================================
# 7. Feature Importance (GAIN)  ✅ FIXED
# =========================================================
booster = uplift_model.booster_
gain_importance = booster.feature_importance(importance_type="gain")

imp_df = (
    pd.DataFrame({
        "feature": FEATURES,
        "gain": gain_importance
    })
    .sort_values("gain", ascending=False)
)

print("\n[Uplift Feature Importance - GAIN]")
print(imp_df)

# =========================================================
# 8. SHAP (uplift 방향 해석)
# =========================================================
import shap

X_sample = X_te.sample(50_000, random_state=RANDOM_STATE)

explainer = shap.TreeExplainer(uplift_model)
shap_values = explainer.shap_values(X_sample)

shap.summary_plot(
    shap_values,
    X_sample,
    feature_names=FEATURES,
    show=True
)
