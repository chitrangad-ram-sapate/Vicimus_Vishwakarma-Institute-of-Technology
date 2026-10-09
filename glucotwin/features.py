"""Fuse the static (EHR) and dynamic (CGM + wearable + meal log) streams with the twin's
physiological state into one model-ready table, one row per patient per 5 minutes.

Every feature at time t uses only information available at t: past samples, meals already
logged, the last *completed* night of sleep, and a twin calibrated on earlier days.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .meals import MEALS
from .twin import Twin

H_STEPS = {h: h // C.SAMPLE_MIN for h in C.HORIZONS_MIN}

EHR_FEATURES = ["age", "sex_m", "bmi", "hba1c", "fpg", "diabetes_duration_y", "status_t2d", "metformin",
                "sulfonylurea", "basal_insulin", "tcf7l2_rs7903146", "prs_z", "egfr", "hypertension", "family_history"]


def _ehr_vector(row: pd.Series) -> dict:
    return {
        "age": row["age"], "sex_m": int(row["sex"] == "M"), "bmi": row["bmi"], "hba1c": row["hba1c"],
        "fpg": row["fpg"], "diabetes_duration_y": row["diabetes_duration_y"], "status_t2d": int(row["status"] == "T2D"),
        "metformin": int(row["metformin"]), "sulfonylurea": int(row["sulfonylurea"]),
        "basal_insulin": int(row["basal_insulin"]), "tcf7l2_rs7903146": row["tcf7l2_rs7903146"],
        "prs_z": row["prs_z"], "egfr": row["egfr"], "hypertension": int(row["hypertension"]),
        "family_history": int(row["family_history"]),
    }


def patient_features(stream: pd.DataFrame, meals: pd.DataFrame, sleep: pd.DataFrame, ehr_row: pd.Series,
                     twin: Twin | None) -> pd.DataFrame:
    df = stream.sort_values("ts").reset_index(drop=True).copy()
    n = len(df)
    start = pd.Timestamp(C.START_DATE)
    df["day"] = ((df["ts"] - start).dt.total_seconds() // 86400).astype(int)
    g = df["cgm"].interpolate(limit=6, limit_area="inside").ffill(limit=3)
    df["cgm_missing"] = df["cgm"].isna().astype(int)
    out = pd.DataFrame({"patient_id": df["patient_id"], "ts": df["ts"], "day": df["day"], "cgm": df["cgm"]})

    # --- CGM history
    out["g0"] = g
    for k in (1, 3, 6, 12):
        out[f"d_{k * 5}"] = g - g.shift(k)
    out["accel_15"] = out["d_15"] - out["d_15"].shift(3)
    for w, name in ((12, "1h"), (36, "3h")):
        out[f"g_mean_{name}"] = g.rolling(w, min_periods=w // 2).mean()
        out[f"g_std_{name}"] = g.rolling(w, min_periods=w // 2).std()
    out["g_max_3h"] = g.rolling(36, min_periods=12).max()
    out["g_min_3h"] = g.rolling(36, min_periods=12).min()
    out["g_mean_24h"] = g.rolling(288, min_periods=72).mean()
    tod = (df["ts"].dt.hour * 60 + df["ts"].dt.minute).to_numpy()
    out["tod_sin"] = np.sin(2 * np.pi * tod / 1440)
    out["tod_cos"] = np.cos(2 * np.pi * tod / 1440)
    out["cgm_missing"] = df["cgm_missing"]

    # --- Wearables
    steps = df["steps"].astype(float)
    for w in (3, 6, 12, 24):
        out[f"steps_{w * 5}m"] = steps.rolling(w, min_periods=1).sum()
    hr, hrv = df["hr"].astype(float), df["hrv_rmssd"].astype(float)
    out["hr"] = hr
    out["hr_30m"] = hr.rolling(6, min_periods=1).mean()
    out["hr_vs_24h"] = out["hr_30m"] - hr.rolling(288, min_periods=36).median()
    out["hrv_1h"] = hrv.rolling(12, min_periods=3).mean()
    out["hrv_vs_3d"] = out["hrv_1h"] / hrv.rolling(864, min_periods=144).median()   # stress / recovery proxy
    out["hrv_day"] = hrv.rolling(144, min_periods=24).mean()
    out["asleep"] = (df["sleep_stage"] > 0).astype(int)

    # last completed night of sleep
    sl = sleep.sort_values("wake")[["wake", "sleep_h", "efficiency", "deep_frac", "rem_frac"]].rename(
        columns={"sleep_h": "sleep_last_h", "efficiency": "sleep_eff", "deep_frac": "sleep_deep", "rem_frac": "sleep_rem"})
    merged = pd.merge_asof(df[["ts"]], sl, left_on="ts", right_on="wake", direction="backward")
    for c in ("sleep_last_h", "sleep_eff", "sleep_deep", "sleep_rem"):
        out[c] = merged[c].to_numpy()
    out["sleep_debt"] = np.clip(7.0 - out["sleep_last_h"], 0, None)

    # --- Meal log (patient app, only what was logged, at the time it was logged)
    lg = meals[meals["logged"]].sort_values("logged_ts")
    grid = df["ts"].to_numpy()
    carbs_series = np.zeros(n)
    gl_series = np.zeros(n)
    pos = np.searchsorted(grid, lg["logged_ts"].to_numpy(), side="left")
    for p_, c_, k_ in zip(pos, lg["logged_carbs_g"].to_numpy(), lg["meal_key"]):
        if p_ < n:
            carbs_series[p_] += c_
            gl_series[p_] += c_ * MEALS[k_].gi / 100
    cs = pd.Series(carbs_series)
    for w in (6, 12, 24, 36):
        out[f"carbs_{w * 5}m"] = cs.rolling(w, min_periods=1).sum()
    out["gl_120m"] = pd.Series(gl_series).rolling(24, min_periods=1).sum()
    last_idx = pd.Series(np.where(carbs_series > 0, np.arange(n), np.nan)).ffill()
    out["min_since_meal"] = np.clip((np.arange(n) - last_idx.to_numpy()) * C.SAMPLE_MIN, 0, 720)
    out["min_since_meal"] = out["min_since_meal"].fillna(720)
    last_carbs = pd.Series(np.where(carbs_series > 0, carbs_series, np.nan)).ffill().fillna(0)
    out["last_meal_carbs"] = last_carbs.to_numpy()
    out["carbs_24h"] = cs.rolling(288, min_periods=1).sum()

    # --- Static EHR
    for k, v in _ehr_vector(ehr_row).items():
        out[k] = v

    # --- Twin physiology
    if twin is not None:
        idx = np.arange(n)
        fc = twin.forecast(idx, max(H_STEPS.values()))
        for h, s in H_STEPS.items():
            out[f"twin_g{h}"] = fc[:, s - 1]
            out[f"twin_d{h}"] = fc[:, s - 1] - out["g0"]
        out["twin_peak_2h"] = fc.max(1) - out["g0"]
        out["twin_x"] = twin.X
        out["twin_cob"] = twin.carbs_on_board(idx)
        for k, v in twin.p.to_dict().items():
            out[f"twin_p_{k}"] = v

    # --- Targets
    for h, s in H_STEPS.items():
        out[f"y_{h}"] = df["cgm"].shift(-s)
    fut = pd.concat([df["cgm"].shift(-k) for k in range(1, C.EVENT_WINDOW_MIN // C.SAMPLE_MIN + 1)], axis=1)
    fut_max, fut_min = fut.max(axis=1), fut.min(axis=1)
    valid = fut.notna().sum(axis=1) >= 18
    out["y_spike"] = np.where(valid, (fut_max >= C.HYPER_THRESHOLD) & (out["g0"] < C.HYPER_THRESHOLD), np.nan)
    out["y_hypo"] = np.where(valid, (fut_min < C.HYPO_THRESHOLD) & (out["g0"] >= C.HYPO_THRESHOLD), np.nan)
    out["fut_max_2h"] = fut_max
    return out


def feature_groups(columns) -> dict[str, list[str]]:
    cols = list(columns)
    cgm = ["g0", "d_5", "d_15", "d_30", "d_60", "accel_15", "g_mean_1h", "g_std_1h", "g_mean_3h", "g_std_3h",
           "g_max_3h", "g_min_3h", "g_mean_24h", "tod_sin", "tod_cos", "cgm_missing"]
    wear = ["steps_15m", "steps_30m", "steps_60m", "steps_120m", "hr", "hr_30m", "hr_vs_24h", "hrv_1h", "hrv_vs_3d",
            "hrv_day", "asleep", "sleep_last_h", "sleep_eff", "sleep_deep", "sleep_rem", "sleep_debt"]
    meal = ["carbs_30m", "carbs_60m", "carbs_120m", "carbs_180m", "gl_120m", "min_since_meal", "last_meal_carbs", "carbs_24h"]
    twin = [c for c in cols if c.startswith("twin_")]
    return {"CGM": cgm, "Wearables": wear, "Meal log": meal, "EHR": EHR_FEATURES, "Twin": twin}


# Human-readable labels for explanations in the dashboard
FEATURE_LABELS = {
    "g0": "current glucose", "d_15": "15-min trend", "d_30": "30-min trend", "d_60": "1-h trend", "d_5": "5-min change",
    "accel_15": "trend acceleration", "g_mean_24h": "24-h average glucose", "g_std_3h": "recent variability",
    "tod_sin": "time of day", "tod_cos": "time of day", "steps_30m": "steps (last 30 min)",
    "steps_60m": "steps (last hour)", "steps_120m": "steps (last 2 h)", "hrv_vs_3d": "HRV vs personal baseline",
    "hrv_1h": "heart-rate variability", "hrv_day": "daytime HRV", "hr_vs_24h": "heart rate vs baseline",
    "sleep_last_h": "last night's sleep", "sleep_debt": "sleep debt", "sleep_eff": "sleep efficiency",
    "sleep_deep": "deep-sleep share", "carbs_30m": "carbs eaten (30 min)", "carbs_60m": "carbs eaten (1 h)",
    "carbs_120m": "carbs eaten (2 h)", "gl_120m": "glycaemic load (2 h)", "min_since_meal": "time since last meal",
    "last_meal_carbs": "last meal size", "carbs_24h": "carbs (24 h)", "hba1c": "HbA1c", "fpg": "fasting glucose",
    "bmi": "BMI", "sulfonylurea": "sulfonylurea therapy", "basal_insulin": "basal insulin", "metformin": "metformin",
    "prs_z": "polygenic risk", "tcf7l2_rs7903146": "TCF7L2 genotype", "diabetes_duration_y": "diabetes duration",
    "twin_peak_2h": "twin-simulated 2-h peak", "twin_cob": "carbs still absorbing (twin)",
    "twin_x": "insulin action (twin)", "twin_d120": "twin 2-h projection", "twin_d60": "twin 1-h projection",
    "twin_d30": "twin 30-min projection",
}
