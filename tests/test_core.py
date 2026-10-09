"""Fast unit tests for the physiology, twin, metrics and feature layers (no pipeline run needed)."""

import numpy as np
import pandas as pd
import pytest

from glucotwin import config as C
from glucotwin.cohort import sample_patient, finalize_labs, to_fhir_bundle
from glucotwin.features import patient_features
from glucotwin.lifestyle import assign_meal_preferences, generate_life, wearable_signals
from glucotwin.meals import MEALS
from glucotwin.metrics import clarke_zones, event_lead_times
from glucotwin.physiology import meal_appearance, simulate, cgm_from_plasma
from glucotwin.twin import Twin, build_inputs, calibrate, ehr_prior


@pytest.fixture(scope="module")
def patient():
    rng = np.random.default_rng(7)
    rec, phys, life = sample_patient(1, rng)
    assign_meal_preferences(life, rng)
    days = 4
    lf = generate_life(rec, life, days, rng)
    ra = meal_appearance(len(lf["minute"]), lf["meals"], phys.gut_scale)
    g = simulate(phys, ra, lf["cadence"], lf["tod"].astype(float), lf["sleep_debt"], lf["stress"], lf["drug_profile"])
    cgm = cgm_from_plasma(g, rng)
    wear = wearable_signals(life, lf, rng)
    finalize_labs(rec, g.mean(), g.min(), rng, pd.Timestamp(C.START_DATE).date())
    start = pd.Timestamp(C.START_DATE)
    ts = start + pd.to_timedelta(np.arange(len(cgm)) * 5, unit="min")
    stream = pd.DataFrame({"patient_id": rec.patient_id, "ts": ts, "cgm": cgm, **wear})
    meals = pd.DataFrame([{
        "patient_id": rec.patient_id, "ts": start + pd.Timedelta(minutes=m["minute"]),
        "logged_ts": start + pd.Timedelta(minutes=m["logged_minute"]) if m["logged"] else pd.NaT,
        "meal_key": m["key"], "logged": m["logged"], "logged_carbs_g": m["logged_carbs_g"]} for m in lf["meals"]])
    sleep = pd.DataFrame([{"wake": start + pd.Timedelta(minutes=n["wake_min"]), "sleep_h": n["sleep_h"],
                           "efficiency": n["efficiency"], "deep_frac": n["deep_frac"], "rem_frac": n["rem_frac"]}
                          for n in lf["nights"]])
    ehr = pd.Series(rec.to_flat())
    return {"rec": rec, "g": g, "stream": stream, "meals": meals, "sleep": sleep, "ehr": ehr, "lf": lf}


def test_physiology_is_plausible(patient):
    g = patient["g"]
    assert 50 < g.min() and g.max() < 450
    assert 90 < g.mean() < 250


def test_meal_raises_glucose():
    m = MEALS["rice_dal_sabzi"]
    ra = meal_appearance(600, [{"minute": 60, "carbs_g": m.carbs_g, "f": m.bioavailability, "tau": m.absorption_tau}])
    assert ra[:60].sum() == 0 and ra.max() > 0
    # total appearance equals bioavailable carbs (mg), within truncation error
    assert abs(ra.sum() - m.bioavailability * m.carbs_g * 1000) / (m.carbs_g * 1000) < 0.02


def test_low_gi_meal_absorbs_slower():
    assert MEALS["roti_dal_sabzi"].absorption_tau > MEALS["rice_dal_sabzi"].absorption_tau


def test_fhir_bundle_is_valid_shape(patient):
    b = to_fhir_bundle(patient["rec"], pd.Timestamp(C.START_DATE).date())
    types = {e["resource"]["resourceType"] for e in b["entry"]}
    assert b["resourceType"] == "Bundle" and {"Patient", "Condition", "Observation"} <= types
    loinc = [e["resource"]["code"]["coding"][0]["code"] for e in b["entry"] if e["resource"]["resourceType"] == "Observation"]
    assert "4548-4" in loinc  # HbA1c


def test_twin_calibration_beats_prior(patient):
    inp = build_inputs(patient["stream"], patient["meals"], patient["rec"].weight_kg)
    prior = ehr_prior(patient["ehr"])
    params, info = calibrate(inp, prior, calib_end=3 * 288, maxfev=150)
    assert info["loss"] <= info["prior_loss"]
    tw = Twin(params, inp)
    fc = tw.forecast(np.arange(10, 100), 24)
    assert fc.shape == (90, 24) and np.isfinite(fc).all()


def test_forecast_does_not_peek_at_future_meals(patient):
    inp = build_inputs(patient["stream"], patient["meals"], patient["rec"].weight_kg)
    tw = Twin(ehr_prior(patient["ehr"]), inp)
    first_meal = int(np.ceil(inp.meal_step[0]))
    i = first_meal - 6
    fc = tw.forecast(np.array([i]), 24)[0]
    # with no known carbs on board, the twin must not predict a meal-sized rise
    assert fc.max() - tw.G[i] < 25


def test_walk_lowers_postprandial_peak(patient):
    inp = build_inputs(patient["stream"], patient["meals"], patient["rec"].weight_kg)
    tw = Twin(ehr_prior(patient["ehr"]), inp)
    meal = [{"meal_key": "rice_dal_sabzi", "offset_min": 0}]
    g_a = tw.simulate_scenario(500, 300, meal)
    g_b = tw.simulate_scenario(500, 300, meal, [{"offset_min": 15, "duration_min": 20, "cadence": 100}])
    assert g_b.max() < g_a.max()


def test_features_have_no_future_leakage(patient):
    f = patient_features(patient["stream"], patient["meals"], patient["sleep"], patient["ehr"], None)
    # perturbing future CGM must not change present features
    s2 = patient["stream"].copy()
    s2.loc[s2.index >= 600, "cgm"] += 50
    f2 = patient_features(s2, patient["meals"], patient["sleep"], patient["ehr"], None)
    feats = [c for c in f.columns if not c.startswith(("y_", "fut_")) and c not in ("cgm",)]
    pd.testing.assert_frame_equal(f.loc[:590, feats], f2.loc[:590, feats])


def test_clarke_zones():
    z = clarke_zones(np.array([100, 100, 250, 50]), np.array([105, 240, 120, 200]))
    assert list(z) == ["A", "C", "D", "E"]


def test_event_lead_time():
    g = np.r_[np.full(30, 120.0), np.linspace(120, 220, 20), np.full(20, 220.0)]
    p = np.zeros_like(g)
    p[20:40] = 0.9
    r = event_lead_times(g, p, 0.5)
    assert r["events"] == 1 and r["detected"] == 1 and r["median_lead_min"] > 0


def test_alert_policy_confirms_and_snoozes():
    from glucotwin.metrics import alert_notifications
    p = np.zeros(60)
    p[5] = 0.9                 # single-reading blip: not confirmed, no alert
    p[10:14] = 0.9             # confirmed at 11 -> alert
    p[16:20] = 0.9             # within 60-min snooze -> suppressed
    p[30:33] = 0.9             # after snooze -> alert at 31
    assert list(alert_notifications(p, 0.5)) == [11, 31]
