"""Daily-life generator: sleep architecture, meals (Indian diet), step cadence, stress, drug
profiles and the wearable signals they produce (heart rate, HRV/RMSSD, steps, sleep stages).

Output signals mimic what Apple Health / Google Fit / Fitbit export at 5-minute resolution.
"""

from __future__ import annotations

import numpy as np

from .cohort import Lifestyle, PatientRecord
from .meals import MEALS, by_slot

AWAKE, LIGHT, DEEP, REM = 0, 1, 2, 3


def assign_meal_preferences(life: Lifestyle, rng: np.random.Generator) -> None:
    """Each patient has habitual meals; repeated exposures make food-response profiling possible."""
    prefs = {}
    for slot in ("breakfast", "lunch", "dinner", "snack"):
        options = by_slot(slot, life.region)
        keys = [m.key for m in options if m.key != "gulab_jamun"]
        w = rng.dirichlet(np.full(len(keys), 0.7))
        prefs[slot] = (keys, w.tolist())
    life.meal_prefs = prefs


def _sleep_night(onset: int, dur_min: int, rng: np.random.Generator, bad: bool) -> list[tuple[int, int, int]]:
    """Return (start, end, stage) segments for one night in absolute minutes."""
    segs = []
    t = onset
    latency = int(rng.uniform(8, 35 if bad else 20))
    segs.append((t, t + latency, AWAKE))
    t += latency
    end = onset + dur_min
    j = 0
    while t < end:
        cyc = int(rng.normal(92, 10))
        deep_f = max(0.36 - 0.11 * j, 0.03) * (0.7 if bad else 1.0)
        rem_f = min(0.10 + 0.07 * j, 0.42)
        light_f = 1 - deep_f - rem_f
        parts = [(LIGHT, light_f * 0.55), (DEEP, deep_f), (LIGHT, light_f * 0.45), (REM, rem_f)]
        for stage, frac in parts:
            d = max(int(cyc * frac), 1)
            segs.append((t, min(t + d, end), stage))
            t += d
            if t >= end:
                break
        if t < end and rng.random() < (0.75 if bad else 0.4):
            d = int(rng.uniform(1, 9 if bad else 5))
            segs.append((t, min(t + d, end), AWAKE))
            t += d
        j += 1
    return segs


def generate_life(rec: PatientRecord, life: Lifestyle, n_days: int, rng: np.random.Generator) -> dict:
    total = n_days * 1440
    minute = np.arange(total)
    tod = minute % 1440

    stage = np.full(total, AWAKE, dtype=np.int8)
    in_bed = np.zeros(total, dtype=bool)
    sleep_debt = np.zeros(total)
    nights, wakes = [], []

    # Night k precedes day k (k = 0..n_days); the last night runs past the horizon and is clipped.
    onsets = []
    for k in range(n_days + 1):
        bad = rng.random() < life.bad_night_p
        dur_h = rng.normal(life.sleep_mean_h, life.sleep_sd_h) - (rng.uniform(1.2, 2.5) if bad else 0.0)
        dur = int(np.clip(dur_h, 3.5, 9.5) * 60)
        # anchored on a habitual wake time: short nights mean going to bed late (screens, work)
        onset = int(k * 1440 + life.wake_mean_min + rng.normal(0, 30)) - dur
        segs = _sleep_night(onset, dur, rng, bad)
        asleep_min = 0
        deep_min = rem_min = 0
        for s, e, st in segs:
            asleep_min += (e - s) if st != AWAKE else 0
            deep_min += (e - s) if st == DEEP else 0
            rem_min += (e - s) if st == REM else 0
            s2, e2 = max(s, 0), min(e, total)
            if e2 > s2:
                stage[s2:e2] = st
                in_bed[s2:e2] = True
        onsets.append(onset)
        wakes.append(onset + dur)
        if k < n_days:
            nights.append({
                "day": k, "onset_min": onset, "wake_min": onset + dur,
                "time_in_bed_h": dur / 60, "sleep_h": asleep_min / 60,
                "efficiency": asleep_min / dur, "deep_frac": deep_min / max(asleep_min, 1),
                "rem_frac": rem_min / max(asleep_min, 1), "bad_night": bad,
            })

    for k in range(n_days):
        debt = float(np.clip((7.0 - nights[k]["sleep_h"]) / 3.0, 0, 1))
        s, e = max(wakes[k], 0), min(onsets[k + 1] + 360, total)   # effect persists into the next night
        sleep_debt[s:e] = np.maximum(sleep_debt[s:e], debt)

    # Stress days (work deadlines, illness): raised basal glucose, lower HRV
    stress = np.zeros(total)
    stress_days = rng.random(n_days) < life.stress_day_p
    for k in np.where(stress_days)[0]:
        stress[k * 1440 + 540: k * 1440 + 1320] = 1.0

    # Meals
    meals = []
    for k in range(n_days):
        day0 = k * 1440
        def pick(slot):
            keys, w = life.meal_prefs[slot]
            return str(rng.choice(keys, p=w))
        slots = []
        if rng.random() > life.skip_breakfast_p:
            slots.append(("breakfast", max(wakes[k] + rng.uniform(35, 100), day0 + 330)))
        slots.append(("lunch", day0 + (13.0 if life.region == "south" else 13.5) * 60 + rng.normal(0, 35)))
        if rng.random() < life.snack_p:
            slots.append(("snack", day0 + 17.25 * 60 + rng.normal(0, 40)))
        slots.append(("dinner", day0 + (20.5 if life.region == "south" else 21.0) * 60 + rng.normal(0, 40)))
        for slot, t in slots:
            key = pick(slot)
            m = MEALS[key]
            portion = float(np.clip(rng.normal(1.0, 0.15), 0.6, 1.5))
            meals.append({"minute": int(t), "key": key, "slot": slot, "portion": portion,
                          "carbs_g": m.carbs_g * portion, "f": m.bioavailability, "tau": m.absorption_tau})
        if rng.random() < life.sweet_p:
            m = MEALS["gulab_jamun"]
            t = slots[-1][1] + rng.uniform(20, 50)
            meals.append({"minute": int(t), "key": m.key, "slot": "snack", "portion": 1.0,
                          "carbs_g": m.carbs_g, "f": m.bioavailability, "tau": m.absorption_tau})
    meals.sort(key=lambda d: d["minute"])

    # What the patient logs in the app: imperfect compliance and carb-estimation error
    for m in meals:
        m["logged"] = bool(rng.random() < life.log_compliance)
        m["logged_carbs_g"] = float(round(m["carbs_g"] * rng.lognormal(0, 0.18) / 5) * 5) if m["logged"] else np.nan
        m["logged_minute"] = int(m["minute"] + rng.normal(0, 4)) if m["logged"] else -1

    # Steps (cadence, steps/min)
    cadence = np.zeros(total)
    awake = ~in_bed

    def bout(start, dur, cad):
        s, e = int(max(start, 0)), int(min(start + dur, total))
        if e > s:
            cadence[s:e] = np.where(awake[s:e], np.maximum(cad + rng.normal(0, 6, e - s), 0), 0)

    walks = []
    for k in range(n_days):
        if rng.random() < life.morning_walk_p:
            st, du = wakes[k] + rng.uniform(10, 40), rng.uniform(20, 45)
            bout(st, du, rng.uniform(95, 118))
            walks.append({"minute": int(st), "duration": int(du), "type": "morning walk"})
        for m in meals:
            if m["slot"] in ("lunch", "dinner") and k * 1440 <= m["minute"] < (k + 1) * 1440 \
                    and rng.random() < life.post_meal_walk_p:
                st, du = m["minute"] + rng.uniform(10, 30), rng.uniform(10, 25)
                bout(st, du, rng.uniform(85, 105))
                walks.append({"minute": int(st), "duration": int(du), "type": "post-meal walk"})
        n_bouts = rng.poisson(life.daily_steps * 0.5 / 450)
        for _ in range(n_bouts):
            st = k * 1440 + rng.uniform(7 * 60, 22 * 60)
            bout(st, rng.uniform(3, 12), rng.uniform(55, 100))
    cadence[~awake] = 0

    # Drug action profiles (glimepiride with breakfast; glargine flat)
    drug_profile = np.zeros(total)
    phys_su, phys_ins = 0.0022 if rec.sulfonylurea else 0.0, 0.0016 if rec.basal_insulin else 0.0
    if phys_su + phys_ins > 0:
        su_prof = np.zeros(total)
        if phys_su:
            for k in range(n_days):
                bf = [m["minute"] for m in meals if m["slot"] == "breakfast" and k * 1440 <= m["minute"] < (k + 1) * 1440]
                dose = bf[0] - 10 if bf else k * 1440 + 480
                s = minute - dose
                mask = s >= 0
                su_prof[mask] += (s[mask] / 180.0) * np.exp(1 - s[mask] / 180.0)
        drug_profile = (phys_su * su_prof + phys_ins) / (phys_su + phys_ins)

    return {
        "minute": minute, "tod": tod, "stage": stage, "in_bed": in_bed, "sleep_debt": sleep_debt,
        "stress": stress, "cadence": cadence, "meals": meals, "nights": nights, "walks": walks,
        "drug_profile": drug_profile, "stress_days": stress_days,
    }


def wearable_signals(life: Lifestyle, lf: dict, rng: np.random.Generator, sample_min: int = 5) -> dict:
    """Aggregate minute-level life into 5-minute wearable streams: HR, HRV (RMSSD), steps, sleep stage."""
    n = len(lf["minute"]) // sample_min
    shape = (n, sample_min)
    cad = lf["cadence"][: n * sample_min].reshape(shape)
    steps = cad.sum(1)
    cad_mean = cad.mean(1)
    stage = lf["stage"][: n * sample_min].reshape(shape)[:, sample_min // 2]
    tod = lf["tod"][: n * sample_min: sample_min]
    stress = lf["stress"][: n * sample_min: sample_min]
    debt = lf["sleep_debt"][: n * sample_min: sample_min]

    post = np.zeros(n)
    for m in lf["meals"]:
        idx = m["minute"] // sample_min
        rel = np.arange(n) - idx
        win = (rel >= 0) & (rel < 36)
        post[win] += np.exp(-0.5 * ((rel[win] - 10) / 6.0) ** 2) * min(m["carbs_g"] / 70, 1.5)

    stage_hr = np.select([stage == LIGHT, stage == DEEP, stage == REM], [-6.0, -10.0, -3.0], 0.0)
    hr = (life.rhr + 3 * np.sin(2 * np.pi * (tod - 960) / 1440) + 0.33 * cad_mean + stage_hr
          + 4 * post + 5 * stress + 3 * debt + rng.normal(0, 1.8, n))
    stage_hrv = np.select([stage == LIGHT, stage == DEEP, stage == REM], [1.15, 1.35, 1.0], 1.0)
    hrv = (life.hrv_base * stage_hrv * np.exp(-cad_mean / 120) * np.where(stress > 0, 0.78, 1.0)
           * (1 - 0.15 * debt) * (1 - 0.06 * post) * rng.lognormal(0, 0.12, n))
    return {"hr": np.round(hr, 1), "hrv_rmssd": np.round(hrv, 1), "steps": np.round(steps).astype(int),
            "sleep_stage": stage.astype(int)}
