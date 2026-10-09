"""Ground-truth glucose-insulin physiology used to *generate* synthetic patients.

This is deliberately richer than the twin model in ``twin.py``: it adds the dawn
phenomenon, sleep-debt insulin resistance, stress hyperglycaemia, counter-
regulation and sulfonylurea/insulin drug action. The twin never sees these
mechanisms directly. It has to learn their footprint from wearable and EHR data
through the ML residual layer.

Model (extended Bergman oral minimal model, minutes, mg/dL):
    dG/dt = -(SG + X + A + D(t)) * G + SG * Gb_eff(t) + Ra(t) / (V * W) + CR(G)
    dX/dt =  p2 * (kx_eff(t) * max(G - Gb_eff(t), 0) - X)
    Ra(t) = sum_meals f * carbs * (t - t0) / tau^2 * exp(-(t - t0) / tau)   (mg/min)
    A(t)  = first-order filtered step cadence * k_ex
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class TruePhysiology:
    weight_kg: float
    gb: float            # basal (fasting) glucose, mg/dL, already reflecting metformin
    sg: float            # glucose effectiveness, 1/min
    kx: float            # lumped insulin secretion x sensitivity gain, 1/(min * mg/dL)
    p2: float            # insulin action rate, 1/min
    gut_scale: float     # individual gastric-emptying multiplier on meal tau
    k_ex: float          # glucose uptake per (steps/min), 1/min
    dawn_amp: float      # dawn phenomenon amplitude, mg/dL
    sleep_sens: float    # fractional loss of insulin action per unit sleep debt
    stress_gb: float     # basal glucose rise on high-stress days, mg/dL
    drug_x: float        # glucose-independent insulin action from SU / basal insulin, 1/min
    v_dl_per_kg: float = 1.6

    def to_dict(self) -> dict:
        return asdict(self)


def meal_appearance(n_min: int, meals: list[dict], gut_scale: float = 1.0, dt: float = 1.0) -> np.ndarray:
    """Rate of glucose appearance (mg/min) on a grid of ``n_min`` steps of ``dt`` minutes.

    ``meals`` items need: ``minute`` (onset), ``carbs_g``, ``f`` (bioavailability), ``tau`` (min).
    """
    t = np.arange(n_min) * dt
    ra = np.zeros(n_min)
    for m in meals:
        tau = m["tau"] * gut_scale
        s = t - m["minute"]
        mask = (s >= 0) & (s < 8 * tau)
        ss = s[mask]
        ra[mask] += m["f"] * m["carbs_g"] * 1000.0 * ss / tau**2 * np.exp(-ss / tau)
    return ra


def activity_effect(steps_per_min: np.ndarray, k_ex: float, tau_a: float = 15.0, dt: float = 1.0) -> np.ndarray:
    """Exercise-driven glucose uptake: first-order filtered step cadence."""
    a = np.zeros_like(steps_per_min, dtype=float)
    alpha = dt / tau_a
    acc = 0.0
    for i, s in enumerate(steps_per_min):
        acc += alpha * (k_ex * s - acc)
        a[i] = acc
    return a


def dawn_profile(minute_of_day: np.ndarray) -> np.ndarray:
    """Smooth bump peaking ~06:30, the hepatic glucose surge before waking."""
    return np.exp(-0.5 * ((minute_of_day - 390) / 75.0) ** 2)


def simulate(phys: TruePhysiology, ra: np.ndarray, steps_per_min: np.ndarray,
             minute_of_day: np.ndarray, sleep_debt: np.ndarray, stress: np.ndarray,
             drug_profile: np.ndarray, g0: float | None = None) -> np.ndarray:
    """Integrate the ground-truth model at 1-minute resolution. Returns plasma glucose (mg/dL)."""
    n = len(ra)
    vol = phys.v_dl_per_kg * phys.weight_kg
    a = activity_effect(steps_per_min, phys.k_ex)
    gb_eff = phys.gb + phys.dawn_amp * dawn_profile(minute_of_day) + phys.stress_gb * stress
    kx_eff = phys.kx * (1.0 - phys.sleep_sens * sleep_debt)
    drug = phys.drug_x * drug_profile

    g = np.empty(n)
    gi = g0 if g0 is not None else phys.gb
    x = 0.0
    sg, p2 = phys.sg, phys.p2
    for i in range(n):
        gbe = gb_eff[i]
        cr = 0.04 * (75.0 - gi) if gi < 75.0 else 0.0   # counter-regulatory hepatic output
        dg = -(sg + x + a[i] + drug[i]) * gi + sg * gbe + ra[i] / vol + cr
        dx = p2 * (kx_eff[i] * max(gi - gbe, 0.0) - x)
        gi = max(gi + dg, 30.0)
        x = max(x + dx, 0.0)
        g[i] = gi
    return g


def cgm_from_plasma(g_plasma_1min: np.ndarray, rng: np.random.Generator,
                    sample_min: int = 5, lag_min: float = 8.0,
                    noise_sd: float = 4.0, dropout_rate: float = 0.004) -> np.ndarray:
    """Interstitial lag + AR(1) sensor noise + sporadic dropouts, sampled every ``sample_min``."""
    gi = np.empty_like(g_plasma_1min)
    acc = g_plasma_1min[0]
    for i, v in enumerate(g_plasma_1min):
        acc += (v - acc) / lag_min
        gi[i] = acc
    sampled = gi[::sample_min].copy()
    noise = np.zeros(len(sampled))
    e = 0.0
    for i in range(len(sampled)):
        e = 0.7 * e + rng.normal(0, noise_sd * np.sqrt(1 - 0.49))
        noise[i] = e
    cgm = sampled * (1 + rng.normal(0, 0.02, len(sampled))) + noise
    cgm = np.clip(np.round(cgm), 40, 400)
    drop = rng.random(len(cgm)) < dropout_rate
    # dropouts come in short runs (sensor compression / Bluetooth loss)
    for idx in np.where(drop)[0]:
        cgm[idx: idx + rng.integers(1, 5)] = np.nan
    return cgm
