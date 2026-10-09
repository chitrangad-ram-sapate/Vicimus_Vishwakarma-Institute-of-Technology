"""Generate the full synthetic cohort and write it to ``data/synthetic``.

Outputs
-------
streams.parquet      5-min wearable + CGM stream (the dynamic data stream)
meals.parquet        meal events (true + what the patient logged)
sleep.parquet        nightly sleep summaries
activity.parquet     structured walks
ehr.parquet          flat EHR table (the static data stream)
fhir/P###.json       the same EHR as HL7 FHIR R4 bundles
true_params.parquet  hidden ground-truth physiology (for twin-fidelity validation only)
"""

from __future__ import annotations

import argparse
import json
from datetime import date

import numpy as np
import pandas as pd

from . import config as C
from .cohort import sample_patient, finalize_labs, write_fhir
from .lifestyle import assign_meal_preferences, generate_life, wearable_signals
from .physiology import meal_appearance, simulate, cgm_from_plasma


def generate(n_patients: int = C.N_PATIENTS, n_days: int = C.N_DAYS, seed: int = C.SEED, verbose: bool = True) -> None:
    C.ensure_dirs()
    rng = np.random.default_rng(seed)
    start = pd.Timestamp(C.START_DATE)
    ref_date = (start + pd.Timedelta(days=n_days)).date()

    streams, meals_rows, sleep_rows, walk_rows, ehr_rows, true_rows, hist_rows = [], [], [], [], [], [], []
    for i in range(1, n_patients + 1):
        rec, phys, life = sample_patient(i, rng)
        assign_meal_preferences(life, rng)
        lf = generate_life(rec, life, n_days, rng)

        ra = meal_appearance(len(lf["minute"]), lf["meals"], gut_scale=phys.gut_scale)
        g = simulate(phys, ra, lf["cadence"], lf["tod"].astype(float), lf["sleep_debt"], lf["stress"], lf["drug_profile"])
        cgm = cgm_from_plasma(g, rng, sample_min=C.SAMPLE_MIN)
        wear = wearable_signals(life, lf, rng, C.SAMPLE_MIN)

        fasting = g[(lf["tod"] >= 360) & (lf["tod"] < 420) & lf["in_bed"]]
        finalize_labs(rec, float(g.mean()), float(fasting.mean() if len(fasting) else g.min()), rng, ref_date)
        write_fhir(rec, ref_date, C.FHIR_DIR)

        n = len(cgm)
        ts = start + pd.to_timedelta(np.arange(n) * C.SAMPLE_MIN, unit="min")
        streams.append(pd.DataFrame({
            "patient_id": rec.patient_id, "ts": ts, "cgm": cgm,
            "glucose_true": np.round(g[:: C.SAMPLE_MIN][:n], 1),
            **wear,
        }))
        for m in lf["meals"]:
            meals_rows.append({
                "patient_id": rec.patient_id, "ts": start + pd.Timedelta(minutes=m["minute"]),
                "logged_ts": start + pd.Timedelta(minutes=m["logged_minute"]) if m["logged"] else pd.NaT,
                "meal_key": m["key"], "slot": m["slot"], "portion": round(m["portion"], 2),
                "true_carbs_g": round(m["carbs_g"], 1), "logged": m["logged"], "logged_carbs_g": m["logged_carbs_g"],
            })
        for nt in lf["nights"]:
            sleep_rows.append({
                "patient_id": rec.patient_id, "wake_date": (start + pd.Timedelta(days=nt["day"])).date(),
                "onset": start + pd.Timedelta(minutes=nt["onset_min"]), "wake": start + pd.Timedelta(minutes=nt["wake_min"]),
                "sleep_h": round(nt["sleep_h"], 2), "time_in_bed_h": round(nt["time_in_bed_h"], 2),
                "efficiency": round(nt["efficiency"], 3), "deep_frac": round(nt["deep_frac"], 3),
                "rem_frac": round(nt["rem_frac"], 3),
            })
        for w in lf["walks"]:
            walk_rows.append({"patient_id": rec.patient_id, "ts": start + pd.Timedelta(minutes=w["minute"]),
                              "duration_min": w["duration"], "type": w["type"]})
        ehr_rows.append(rec.to_flat())
        for h in rec.hba1c_history:
            hist_rows.append({"patient_id": rec.patient_id, **h})
        true_rows.append({"patient_id": rec.patient_id, **phys.to_dict(),
                          "stress_days": int(lf["stress_days"].sum()), "daily_steps_target": life.daily_steps})

        if verbose:
            tir = np.mean((g >= 70) & (g <= 180)) * 100
            print(f"{rec.patient_id} {rec.status:<11} HbA1c={rec.hba1c:4.1f}  mean={g.mean():5.0f}  "
                  f"max={g.max():4.0f}  min={g.min():4.0f}  TIR={tir:4.0f}%  meals={len(lf['meals'])}")

    pd.concat(streams).to_parquet(C.RAW_DIR / "streams.parquet", index=False)
    pd.DataFrame(meals_rows).to_parquet(C.RAW_DIR / "meals.parquet", index=False)
    pd.DataFrame(sleep_rows).to_parquet(C.RAW_DIR / "sleep.parquet", index=False)
    pd.DataFrame(walk_rows).to_parquet(C.RAW_DIR / "activity.parquet", index=False)
    pd.DataFrame(ehr_rows).to_parquet(C.RAW_DIR / "ehr.parquet", index=False)
    pd.DataFrame(hist_rows).to_parquet(C.RAW_DIR / "hba1c_history.parquet", index=False)
    pd.DataFrame(true_rows).to_parquet(C.RAW_DIR / "true_params.parquet", index=False)
    (C.RAW_DIR / "manifest.json").write_text(json.dumps({
        "n_patients": n_patients, "n_days": n_days, "seed": seed, "start": C.START_DATE,
        "sample_min": C.SAMPLE_MIN, "generated_on": date.today().isoformat(),
        "note": "Fully synthetic data. No real patient information.",
    }, indent=2))
    if verbose:
        print(f"Wrote cohort to {C.RAW_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate the synthetic GlucoTwin cohort")
    ap.add_argument("--patients", type=int, default=C.N_PATIENTS)
    ap.add_argument("--days", type=int, default=C.N_DAYS)
    ap.add_argument("--seed", type=int, default=C.SEED)
    a = ap.parse_args()
    generate(a.patients, a.days, a.seed)
