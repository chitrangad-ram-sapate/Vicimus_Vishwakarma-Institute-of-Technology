"""ML layer of the hybrid twin: LightGBM forecasters with conformalised quantile intervals and
event classifiers with per-prediction SHAP explanations (LightGBM native TreeSHAP)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb

from . import config as C

REG_PARAMS = dict(n_estimators=500, learning_rate=0.04, num_leaves=63, min_child_samples=100,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)
CLF_PARAMS = dict(n_estimators=400, learning_rate=0.04, num_leaves=31, min_child_samples=100,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)


class GlucoseForecaster:
    """Predicts glucose at each horizon as g0 + delta, with a split-conformal (CQR) interval."""

    def __init__(self, features: list[str], horizons=C.HORIZONS_MIN, quantiles: bool = True,
                 n_estimators: int | None = None):
        self.features = features
        self.horizons = tuple(horizons)
        self.quantiles = quantiles
        self.models, self.q_lo, self.q_hi, self.conformal = {}, {}, {}, {}
        self.n_estimators = n_estimators
        alpha = 1 - C.INTERVAL_COVERAGE
        self.alpha = alpha

    def _params(self, base: dict, **kw) -> dict:
        p = dict(base, **kw)
        if self.n_estimators:
            p["n_estimators"] = self.n_estimators
        return p

    def fit(self, df: pd.DataFrame) -> "GlucoseForecaster":
        for h in self.horizons:
            d = df[df[f"y_{h}"].notna() & df["g0"].notna()]
            X, y = d[self.features], d[f"y_{h}"] - d["g0"]
            self.models[h] = lgb.LGBMRegressor(**self._params(REG_PARAMS)).fit(X, y)
            if self.quantiles:
                qp = dict(n_estimators=300, learning_rate=0.05, num_leaves=31)
                self.q_lo[h] = lgb.LGBMRegressor(**self._params(REG_PARAMS, objective="quantile",
                                                                 alpha=self.alpha / 2, **qp)).fit(X, y)
                self.q_hi[h] = lgb.LGBMRegressor(**self._params(REG_PARAMS, objective="quantile",
                                                                 alpha=1 - self.alpha / 2, **qp)).fit(X, y)
        return self

    def calibrate_intervals(self, df: pd.DataFrame) -> dict:
        """Conformalized Quantile Regression (Romano et al., 2019) on held-out subjects."""
        for h in self.horizons:
            d = df[df[f"y_{h}"].notna() & df["g0"].notna()]
            lo, hi = self._raw_interval(d, h)
            y = d[f"y_{h}"].to_numpy()
            scores = np.maximum(lo - y, y - hi)
            n = len(scores)
            q = np.quantile(scores, min(1.0, (1 - self.alpha) * (n + 1) / n))
            self.conformal[h] = float(q)
        return self.conformal

    def _raw_interval(self, df: pd.DataFrame, h: int):
        X, g0 = df[self.features], df["g0"].to_numpy()
        return g0 + self.q_lo[h].predict(X), g0 + self.q_hi[h].predict(X)

    def predict(self, df: pd.DataFrame) -> dict[int, dict[str, np.ndarray]]:
        X, g0 = df[self.features], df["g0"].to_numpy()
        out = {}
        for h in self.horizons:
            point = np.clip(g0 + self.models[h].predict(X), 40, 400)
            res = {"point": point}
            if self.quantiles and h in self.q_lo:
                lo, hi = self._raw_interval(df, h)
                q = self.conformal.get(h, 0.0)
                res["lo"] = np.clip(np.minimum(lo - q, point), 40, 400)
                res["hi"] = np.clip(np.maximum(hi + q, point), 40, 400)
            out[h] = res
        return out


class EventClassifier:
    """Probability of an adverse event (new hyperglycaemic excursion or hypoglycaemia) in the next 2 h."""

    def __init__(self, features: list[str], label: str, n_estimators: int | None = None):
        self.features, self.label = features, label
        p = dict(CLF_PARAMS)
        if n_estimators:
            p["n_estimators"] = n_estimators
        self.params = p
        self.model = None
        self.threshold = 0.5

    def fit(self, df: pd.DataFrame) -> "EventClassifier":
        d = df[df[self.label].notna()]
        pos = d[self.label].mean()
        self.model = lgb.LGBMClassifier(**self.params, scale_pos_weight=float(np.clip((1 - pos) / max(pos, 1e-6), 1, 20)) ** 0.5)
        self.model.fit(d[self.features], d[self.label].astype(int))
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(df[self.features])[:, 1]

    def contributions(self, df: pd.DataFrame) -> pd.DataFrame:
        """TreeSHAP contributions (log-odds) per feature; last column is the bias."""
        c = self.model.predict(df[self.features], pred_contrib=True)
        return pd.DataFrame(c, columns=self.features + ["_bias"], index=df.index)
