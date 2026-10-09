"""Clinical service layer behind the API: everything a doctor's screen needs about a virtual patient
at a given moment in (simulated) time. Keeps FastAPI handlers thin and makes the logic testable."""

from __future__ import annotations

import json
from functools import cached_property

import joblib
import numpy as np
import pandas as pd

from . import config as C
from .features import FEATURE_LABELS
from .meals import MEALS, by_slot
from .metrics import excursion_onsets
from .pipeline import load_data, load_twin

STEP = pd.Timedelta(minutes=C.SAMPLE_MIN)


def _clean(v):
    if isinstance(v, (np.floating, float)):
        return None if not np.isfinite(v) else round(float(v), 3)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def _series(x) -> list:
    return [_clean(v) for v in np.asarray(x, dtype=object)]


def _reason_label(f: str) -> str:
    if f.startswith("tod_"):
        return "time of day"
    if f.startswith("twin_p_"):
        return "personal physiology (twin)"
    if f in ("twin_g30", "twin_g60", "twin_g120", "twin_d30", "twin_d60", "twin_d120"):
        return "twin-simulated trajectory"
    if f in ("asleep", "sleep_last_h", "sleep_debt", "sleep_eff", "sleep_deep", "sleep_rem"):
        return {"asleep": "currently asleep"}.get(f, FEATURE_LABELS.get(f, f))
    return FEATURE_LABELS.get(f, f.replace("_", " "))


def _ord(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def glycaemic_metrics(g: np.ndarray) -> dict:
    g = g[np.isfinite(g)]
    if len(g) < 12:
        return {}
    mean = g.mean()
    return {
        "mean": round(float(mean), 0), "gmi": round(float(3.31 + 0.02392 * mean), 1),
        "cv_pct": round(float(g.std() / mean * 100), 1),
        "tir_pct": round(float(np.mean((g >= 70) & (g <= 180)) * 100), 0),
        "tar_pct": round(float(np.mean(g > 180) * 100), 0), "tar250_pct": round(float(np.mean(g > 250) * 100), 0),
        "tbr_pct": round(float(np.mean(g < 70) * 100), 1),
    }


class ClinicalService:
    def __init__(self):
        self.data = load_data()
        self.twin_params = pd.read_parquet(C.ARTIFACT_DIR / "twin_params.parquet")
        self.models = joblib.load(C.ARTIFACT_DIR / "models.joblib")
        feat = pd.read_parquet(C.ARTIFACT_DIR / "features.parquet")
        self.split = self.models["split"]
        self.ward_ids = list(self.split["test"])
        self.feat = {pid: d.reset_index(drop=True) for pid, d in feat[feat.patient_id.isin(self.ward_ids)].groupby("patient_id")}
        self.ehr = self.data["ehr"].set_index("patient_id")
        self.start = pd.Timestamp(C.START_DATE)
        self.n_steps = C.N_DAYS * 1440 // C.SAMPLE_MIN
        self.end = self.start + STEP * (self.n_steps - 1)
        self._twins = {}
        groups = self.models["groups"]
        self.stream_of = {f: g for g, fs in groups.items() for f in fs}
        self.metrics = json.loads((C.REPORT_DIR / "metrics.json").read_text())

    # ------------------------------------------------------------------ helpers
    def twin(self, pid: str):
        if pid not in self._twins:
            self._twins[pid] = load_twin(pid, self.data, self.twin_params)
        return self._twins[pid]

    def index_at(self, at: pd.Timestamp | None) -> int:
        if at is None:
            at = self.default_now
        i = int((pd.Timestamp(at) - self.start) / STEP)
        return int(np.clip(i, 0, self.n_steps - 1))

    @cached_property
    def default_now(self) -> pd.Timestamp:
        return self.start + pd.Timedelta(days=C.CALIB_DAYS, hours=12, minutes=40)

    def _rows(self, pid: str, lo: int, hi: int) -> pd.DataFrame:
        return self.feat[pid].iloc[lo: hi + 1]

    def _predict(self, rows: pd.DataFrame) -> dict:
        fc = self.models["forecaster"].predict(rows)
        p_spike = self.models["spike"].predict_proba(rows)
        hypo = self.models.get("hypo")
        p_hypo = hypo.predict_proba(rows) if hypo is not None else np.zeros(len(rows))
        return {"fc": fc, "spike": p_spike, "hypo": p_hypo}

    def _tier(self, p_spike: float, p_hypo: float, g0: float) -> str:
        hypo_thr = self.models["hypo"].threshold if self.models.get("hypo") else 1.0
        if p_hypo >= hypo_thr or g0 < 70 or g0 > 300 or p_spike >= 0.8:
            return "high"
        if p_spike >= self.models["spike"].threshold or g0 > 180:
            return "moderate"
        return "low"

    # ------------------------------------------------------------------ ward
    def ward(self, at=None) -> dict:
        i = self.index_at(at)
        out = []
        for pid in self.ward_ids:
            rows = self._rows(pid, i, i)
            pr = self._predict(rows)
            r = rows.iloc[0]
            hist = self.feat[pid]["cgm"].iloc[max(0, i - 288): i + 1].to_numpy()
            e = self.ehr.loc[pid]
            ps, ph = float(pr["spike"][0]), float(pr["hypo"][0])
            out.append({
                "patient_id": pid, "name": e["name"], "age": int(e["age"]), "sex": e["sex"], "status": e["status"],
                "glucose": _clean(r["g0"]), "trend_15": _clean(r["d_15"]), "p_spike": round(ps, 3), "p_hypo": round(ph, 3),
                "pred_60": _clean(pr["fc"][60]["point"][0]), "tier": self._tier(ps, ph, r["g0"]),
                "tir_24h": glycaemic_metrics(hist).get("tir_pct"),
            })
        order = {"high": 0, "moderate": 1, "low": 2}
        out.sort(key=lambda d: (order[d["tier"]], -d["p_spike"]))
        return {"at": str(self.start + STEP * i), "patients": out}

    # ------------------------------------------------------------------ patient
    def patient(self, pid: str, at=None, window_h: float = 12) -> dict:
        i = self.index_at(at)
        now = self.start + STEP * i
        lo = max(0, i - int(window_h * 60 / C.SAMPLE_MIN))
        rows = self._rows(pid, lo, i)
        pr = self._predict(rows)
        cur = rows.iloc[[-1]]
        g0 = float(cur["g0"].iloc[0])
        tw = self.twin(pid)

        stream = self.data["streams"]
        s = stream[(stream.patient_id == pid)].iloc[lo: i + 1]
        meals = self.data["meals"]
        pm = meals[(meals.patient_id == pid) & meals.logged & (meals.logged_ts <= now)]
        acts = self.data["activity"]
        pa = acts[(acts.patient_id == pid) & (acts.ts <= now) & (acts.ts >= now - pd.Timedelta(hours=window_h))]

        # forecasts
        fc = {h: {k: _clean(v[-1]) for k, v in pr["fc"][h].items()} for h in C.HORIZONS_MIN}
        twin_curve = tw.forecast(np.array([i]), 24)[0]
        ps, ph = float(pr["spike"][-1]), float(pr["hypo"][-1])

        # explanation of the current spike risk
        contrib = self.models["spike"].contributions(cur).iloc[0].drop("_bias")
        # merge features that mean the same thing to a clinician (e.g. sin/cos of time, twin parameters)
        label_of = {f: _reason_label(f) for f in contrib.index}
        merged = contrib.groupby(label_of).sum()
        rep = {}
        for f in contrib.abs().sort_values(ascending=False).index:
            rep.setdefault(label_of[f], f)
        top = merged.reindex(merged.abs().sort_values(ascending=False).index).head(7)
        reasons = []
        for label, v in top.items():
            f = rep[label]
            show_value = not f.startswith(("tod_", "twin_p_")) and f != "asleep"
            reasons.append({"feature": f, "label": label, "stream": self.stream_of.get(f, "Twin"),
                            "value": _clean(cur[f].iloc[0]) if show_value else None, "impact": round(float(v), 3)})
        by_stream = contrib.groupby(lambda f: self.stream_of.get(f, "Twin")).sum().round(3).to_dict()

        return {
            "at": str(now), "patient": self.profile(pid),
            "current": {"glucose": _clean(g0), "trend_15": _clean(cur["d_15"].iloc[0]),
                        "hr": _clean(cur["hr"].iloc[0]), "hrv": _clean(cur["hrv_1h"].iloc[0]),
                        "steps_today": int(stream[(stream.patient_id == pid)].iloc[(i // 288) * 288: i + 1]["steps"].sum()),
                        "cob_g": _clean(tw.carbs_on_board(np.array([i]))[0]),
                        "sleep_last_h": _clean(cur["sleep_last_h"].iloc[0])},
            "risk": {"p_spike": round(ps, 3), "p_hypo": round(ph, 3), "tier": self._tier(ps, ph, g0),
                     "spike_threshold": self.models["spike"].threshold,
                     "hypo_threshold": self.models["hypo"].threshold if self.models.get("hypo") else None,
                     "reasons": reasons, "by_stream": by_stream},
            "forecast": {"horizons": fc,
                         "twin_ts": [str(now + STEP * (k + 1)) for k in range(24)],
                         "twin": _series(np.round(twin_curve, 1))},
            "history": {
                "ts": [str(t) for t in s["ts"]], "cgm": _series(s["cgm"]), "hr": _series(s["hr"]),
                "hrv": _series(s["hrv_rmssd"]), "steps": _series(s["steps"]), "sleep_stage": _series(s["sleep_stage"]),
                "p_spike": _series(np.round(pr["spike"], 3)),
            },
            "meals": [{"ts": str(r.logged_ts), "meal_key": r.meal_key, "name": MEALS[r.meal_key].name,
                       "carbs_g": _clean(r.logged_carbs_g)} for r in pm[pm.logged_ts >= now - pd.Timedelta(hours=window_h)].itertuples()],
            "activity": [{"ts": str(r.ts), "duration_min": int(r.duration_min), "type": r.type} for r in pa.itertuples()],
            "insights": self.insights(pid, i),
        }

    def profile(self, pid: str) -> dict:
        e = self.ehr.loc[pid]
        hist = self.data["hba1c_history"]
        tp = self.twin_params.set_index("patient_id")
        meds = [m for m, flag in (("Metformin 500 mg BD", e.metformin), ("Glimepiride 2 mg OD", e.sulfonylurea),
                                  ("Insulin glargine 14 U HS", e.basal_insulin)) if flag]
        conds = [e.status if e.status == "Prediabetes" else "Type 2 diabetes"]
        conds += ["Hypertension"] if e.hypertension else []
        conds += ["Dyslipidaemia"] if e.dyslipidemia else []
        pct = lambda col, v: round(float((tp[col] < v).mean() * 100))
        return {
            "patient_id": pid, "name": e["name"], "age": int(e.age), "sex": e.sex, "city": e.city,
            "abha_masked": "XX-XXXX-XXXX-" + e.abha_id[-4:], "status": e.status, "conditions": conds, "medications": meds,
            "duration_y": _clean(e.diabetes_duration_y), "bmi": _clean(e.bmi), "weight_kg": _clean(e.weight_kg),
            "labs": {"hba1c": _clean(e.hba1c), "fpg": _clean(e.fpg), "ldl": int(e.ldl), "egfr": int(e.egfr),
                     "bp": f"{int(e.sbp)}/{int(e.dbp)}"},
            "hba1c_history": hist[hist.patient_id == pid][["date", "hba1c"]].to_dict("records"),
            "genetics": {"tcf7l2_risk_alleles": int(e.tcf7l2_rs7903146), "prs_z": _clean(e.prs_z),
                         "family_history": bool(e.family_history)},
            "twin": {
                "basal_glucose": round(float(tp.loc[pid, "gb"]), 0),
                "insulin_action_pct": pct("kx", tp.loc[pid, "kx"]),
                "glucose_effectiveness_pct": pct("sg", tp.loc[pid, "sg"]),
                "absorption_speed_pct": 100 - pct("gut", tp.loc[pid, "gut"]),
                "exercise_response_pct": pct("k_ex", tp.loc[pid, "k_ex"]),
                "calibration_gain_pct": round(float(100 * (1 - tp.loc[pid, "loss"] / tp.loc[pid, "prior_loss"]))),
            },
        }

    # ------------------------------------------------------------------ insights & summary
    def insights(self, pid: str, i: int) -> dict:
        stream = self.data["streams"]
        s = stream[stream.patient_id == pid].reset_index(drop=True)
        now = self.start + STEP * i
        g7 = s["cgm"].iloc[max(0, i - 7 * 288): i + 1].to_numpy()
        m7 = glycaemic_metrics(g7)

        # personal food response: rise from meal start to 2-h peak, per logged meal type
        meals = self.data["meals"]
        pm = meals[(meals.patient_id == pid) & meals.logged & (meals.logged_ts <= now - pd.Timedelta(hours=2))]
        resp = []
        for r in pm.itertuples():
            k = int((r.logged_ts - self.start) / STEP)
            win = s["cgm"].iloc[k: k + 25].to_numpy()
            if np.isfinite(win[0]) and np.isfinite(win).sum() > 15:
                resp.append({"meal_key": r.meal_key, "rise": float(np.nanmax(win) - win[0]), "peak": float(np.nanmax(win))})
        food = []
        if resp:
            fr = pd.DataFrame(resp).groupby("meal_key").agg(n=("rise", "size"), rise=("rise", "mean"), peak=("peak", "mean"))
            fr = fr.sort_values("rise", ascending=False)
            food = [{"meal_key": k, "name": MEALS[k].name, "n": int(v.n), "rise": round(float(v.rise)), "peak": round(float(v.peak))}
                    for k, v in fr.iterrows()]

        # sleep vs next-day glucose
        sl = self.data["sleep"]
        sl = sl[(sl.patient_id == pid) & (sl.wake <= now)].tail(7)
        nights = []
        for r in sl.itertuples():
            k0 = int((r.wake - self.start) / STEP)
            day_g = s["cgm"].iloc[k0: min(k0 + 16 * 12, i + 1)]
            nights.append({"date": str(r.wake_date), "sleep_h": round(float(r.sleep_h), 1),
                           "efficiency": round(float(r.efficiency) * 100), "next_day_mean": _clean(day_g.mean())})
        short = [n["next_day_mean"] for n in nights if n["sleep_h"] < 6 and n["next_day_mean"]]
        normal = [n["next_day_mean"] for n in nights if n["sleep_h"] >= 6 and n["next_day_mean"]]
        sleep_effect = round(float(np.mean(short) - np.mean(normal))) if short and normal else None

        excursions = len(excursion_onsets(g7))
        out = {"metrics_7d": m7, "food_response": food, "nights": nights, "sleep_effect_mgdl": sleep_effect,
               "excursions_7d": excursions}
        out["summary"] = self._summary(pid, out)
        return out

    def _summary(self, pid: str, ins: dict) -> dict:
        p = self.profile(pid)
        m = ins["metrics_7d"]
        lines, actions = [], []
        meds = ", ".join(p["medications"]) or "no glucose-lowering drugs"
        lines.append(f"{p['name']}, {p['age']}{p['sex']}, {p['status'] if p['status'] == 'Prediabetes' else 'T2D for ' + str(p['duration_y']) + ' y'}, "
                     f"on {meds}. HbA1c {p['labs']['hba1c']}%, BMI {p['bmi']}.")
        if m:
            ctl = "at target" if m["tir_pct"] >= 70 else "below target (>70%)"
            lines.append(f"Last 7 days: time in range {m['tir_pct']:.0f}% ({ctl}), mean {m['mean']:.0f} mg/dL, "
                         f"GMI {m['gmi']}%, CV {m['cv_pct']}% ({'stable' if m['cv_pct'] < 36 else 'labile'}), "
                         f"{ins['excursions_7d']} excursions above 180 mg/dL.")
            if m["tbr_pct"] >= 1:
                actions.append(f"Time below range {m['tbr_pct']}%: review hypoglycaemia risk"
                               + (" and sulfonylurea dose" if "Glimepiride 2 mg OD" in p["medications"] else "") + ".")
            if m["tir_pct"] < 50:
                actions.append("Sustained hyperglycaemia: consider therapy intensification per RSSDI/ADA guidance.")
        fr = ins["food_response"]
        if len(fr) >= 2:
            worst, best = fr[0], fr[-1]
            lines.append(f"Personal food response: {worst['name']} raises glucose by ~{worst['rise']} mg/dL on average, "
                         f"vs ~{best['rise']} mg/dL for {best['name']}.")
            if worst["rise"] > 60:
                actions.append(f"Counsel on swapping or reducing the portion of {worst['name'].lower()}; try the meal simulator for alternatives.")
        if ins["sleep_effect_mgdl"] is not None and ins["sleep_effect_mgdl"] > 5:
            lines.append(f"After nights with under 6 h sleep, next-day mean glucose was {ins['sleep_effect_mgdl']} mg/dL higher.")
            actions.append("Address sleep hygiene; short sleep is measurably worsening daytime control.")
        t = p["twin"]
        lines.append(f"Twin profile: insulin action at the {_ord(t['insulin_action_pct'])} percentile of the cohort, "
                     f"exercise response at the {_ord(t['exercise_response_pct'])} percentile.")
        if t["exercise_response_pct"] >= 50:
            actions.append("Strong exercise responder: a 15-min walk after lunch and dinner is likely to blunt peaks.")
        return {"text": " ".join(lines), "actions": actions,
                "disclaimer": "Decision support from a proof-of-concept on synthetic data. Clinician judgement required."}

    # ------------------------------------------------------------------ what-if
    def whatif(self, pid: str, at, scenarios: list[dict], horizon_min: int = 300) -> dict:
        i = self.index_at(at)
        now = self.start + STEP * i
        tw = self.twin(pid)
        H = horizon_min // C.SAMPLE_MIN
        ts = [str(now + STEP * (k + 1)) for k in range(H)]
        out = []
        for sc in scenarios:
            g = tw.simulate_scenario(i, horizon_min, sc.get("meals"), sc.get("walks"))
            out.append({"label": sc.get("label", "scenario"), "glucose": _series(np.round(g, 1)),
                        "peak": round(float(g.max())), "time_above_180_min": int((g > 180).sum() * C.SAMPLE_MIN),
                        "mean": round(float(g.mean()))})
        return {"at": str(now), "ts": ts, "start_glucose": _clean(tw.G[i]), "scenarios": out}

    def meal_ranking(self, pid: str, at, slot: str, offset_min: int = 0, walk_min: int = 0) -> list[dict]:
        i = self.index_at(at)
        tw = self.twin(pid)
        region = self.ehr.loc[pid, "region"]
        res = []
        walks = [{"offset_min": offset_min + 15, "duration_min": walk_min, "cadence": 100}] if walk_min else None
        for m in by_slot(slot, region):
            g = tw.simulate_scenario(i, 300, [{"meal_key": m.key, "offset_min": offset_min}], walks)
            res.append({"meal_key": m.key, "name": m.name, "carbs_g": m.carbs_g, "gi": m.gi, "peak": round(float(g.max())),
                        "time_above_180_min": int((g > 180).sum() * C.SAMPLE_MIN)})
        return sorted(res, key=lambda d: (d["time_above_180_min"], d["peak"]))

    def fhir(self, pid: str) -> dict:
        return json.loads((C.FHIR_DIR / f"{pid}.json").read_text())

    def meta(self) -> dict:
        f = self.metrics["final"]
        return {
            "start": str(self.start), "end": str(self.end), "default_now": str(self.default_now),
            "playback_start": str(self.start + pd.Timedelta(days=C.CALIB_DAYS)),
            "horizons": list(C.HORIZONS_MIN), "thresholds": {"hyper": C.HYPER_THRESHOLD, "hypo": C.HYPO_THRESHOLD},
            "meals": [m.to_dict() for m in MEALS.values()],
            "model_card": {
                "rmse_30": round(f["30"]["rmse"], 1), "rmse_60": round(f["60"]["rmse"], 1),
                "rmse_120": round(f["120"]["rmse"], 1), "clarke_ab_60": round(f["60"]["clarke_AB_pct"], 1),
                "spike_auroc": round(f["spike"]["auroc"], 3), "median_lead_min": f["events"]["median_lead_min"],
                "coverage_60": round(f["60"]["coverage_pct"], 1),
            },
        }
