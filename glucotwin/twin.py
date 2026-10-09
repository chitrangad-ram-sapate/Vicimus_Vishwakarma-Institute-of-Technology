"""The patient digital twin: a personalised physiological model kept in sync with live data.

Model (oral minimal model on the 5-minute CGM grid):
    G' = -(SG + X + A) G + SG Gb + Ra(t) / V
    X' =  p2 (kx max(G - Gb, 0) - X)
    A' = (k_ex cadence - A) / 15
    Ra = sum over *logged* meals of f * carbs * s / tau^2 * exp(-s / tau),  tau = tau_meal * gut

Three steps make it a twin, not a generic model:
  1. Prior from the EHR: Gb is seeded from fasting plasma glucose, kx from HbA1c / therapy.
  2. Calibration: six parameters fitted to each patient's own history (MAP: forecast error + prior).
  3. Synchronisation: a Luenberger-style observer nudges the state toward every new CGM reading,
     so forecasts and what-if simulations always start from the patient's *current* state.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .meals import MEALS

DT = 5.0                      # minutes per step
OBS_GAIN = 0.5                # observer correction gain on G
PARAM_NAMES = ("gb", "sg", "kx", "p2", "gut", "k_ex")
PRIOR_SD = np.array([0.15, 0.35, 0.6, 0.4, 0.25, 0.6])   # log-space prior widths


@dataclass
class TwinParams:
    gb: float
    sg: float
    kx: float
    p2: float
    gut: float
    k_ex: float

    def to_log(self) -> np.ndarray:
        return np.log([getattr(self, k) for k in PARAM_NAMES])

    @classmethod
    def from_log(cls, v: np.ndarray) -> "TwinParams":
        v = np.clip(v, np.log([60, 2e-3, 5e-6, 3e-3, 0.4, 1e-6]), np.log([300, 6e-2, 2e-3, 0.2, 2.5, 1e-3]))
        return cls(*np.exp(v))

    def to_dict(self) -> dict:
        return asdict(self)


def ehr_prior(ehr_row: pd.Series) -> TwinParams:
    """Population prior conditioned on the patient's EHR, i.e. the static stream seeding the twin."""
    gb = float(ehr_row["fpg"]) if np.isfinite(ehr_row["fpg"]) else 120.0
    a1c = float(ehr_row["hba1c"])
    kx = 3.0e-4 if ehr_row["status"] == "Prediabetes" else 1.4e-4 * np.exp(-0.25 * (a1c - 7.0))
    return TwinParams(gb=gb, sg=0.013, kx=kx, p2=0.028, gut=1.0, k_ex=9e-5)


@dataclass
class TwinInputs:
    ts: pd.DatetimeIndex
    cgm: np.ndarray            # (n,) with NaN gaps
    cadence: np.ndarray        # (n,) steps/min averaged over each 5-min bin
    meal_step: np.ndarray      # (m,) logged meal onset in fractional steps
    meal_carbs: np.ndarray     # (m,) logged carbs (g)
    meal_f: np.ndarray         # (m,) bioavailability
    meal_tau: np.ndarray       # (m,) library absorption tau (min)
    vol: float                 # distribution volume (dL)

    @property
    def n(self) -> int:
        return len(self.cgm)


def build_inputs(stream: pd.DataFrame, meals: pd.DataFrame, weight_kg: float) -> TwinInputs:
    stream = stream.sort_values("ts")
    ts = pd.DatetimeIndex(stream["ts"])
    t0 = ts[0]
    logged = meals[meals["logged"]].sort_values("logged_ts")
    steps = ((logged["logged_ts"] - t0).dt.total_seconds() / 60.0 / DT).to_numpy()
    lib = [MEALS[k] for k in logged["meal_key"]]
    return TwinInputs(
        ts=ts, cgm=stream["cgm"].to_numpy(float), cadence=stream["steps"].to_numpy(float) / DT,
        meal_step=steps, meal_carbs=logged["logged_carbs_g"].to_numpy(float),
        meal_f=np.array([m.bioavailability for m in lib]), meal_tau=np.array([m.absorption_tau for m in lib]),
        vol=1.6 * weight_kg,
    )


def _meal_ra_matrix(p: TwinParams, steps: np.ndarray, carbs: np.ndarray, f: np.ndarray, tau_lib: np.ndarray,
                    n: int) -> np.ndarray:
    """Per-meal glucose appearance (mg/min) on the grid: shape (m, n)."""
    if len(steps) == 0:
        return np.zeros((0, n))
    grid = np.arange(n) * DT
    s = grid[None, :] - (steps * DT)[:, None]
    tau = (tau_lib * p.gut)[:, None]
    ra = np.where(s >= 0, (f * carbs * 1000.0)[:, None] * s / tau**2 * np.exp(-np.clip(s, 0, None) / tau), 0.0)
    return ra


class Twin:
    """A calibrated, synchronised twin for one patient."""

    def __init__(self, params: TwinParams, inputs: TwinInputs):
        self.p = params
        self.inp = inputs
        self._prepare()

    # ------------------------------------------------------------------ core simulation
    def _prepare(self) -> None:
        p, inp = self.p, self.inp
        self.ra_meals = _meal_ra_matrix(p, inp.meal_step, inp.meal_carbs, inp.meal_f, inp.meal_tau, inp.n)
        # cumulative over meals ordered by time: row k = Ra from the first k meals
        self.ra_cum = np.vstack([np.zeros((1, inp.n)), np.cumsum(self.ra_meals, axis=0)])
        self.ra_all = self.ra_cum[-1]
        # number of meals already logged at each grid step (what the twin "knows" at time i)
        self.meals_known = np.searchsorted(inp.meal_step, np.arange(inp.n), side="right")
        self.G, self.X, self.A = self._observe()

    def _observe(self):
        p, inp = self.p, self.inp
        n = inp.n
        G, X, A = np.empty(n), np.empty(n), np.empty(n)
        cgm, cad, ra = inp.cgm, inp.cadence, self.ra_all
        g = cgm[0] if np.isfinite(cgm[0]) else p.gb
        x = a = 0.0
        sg, gb, kx, p2, kex, vol = p.sg, p.gb, p.kx, p.p2, p.k_ex, inp.vol
        for i in range(n):
            if i > 0:
                dg = -(sg + x + a) * g + sg * gb + ra[i - 1] / vol
                dx = p2 * (kx * (g - gb if g > gb else 0.0) - x)
                a += (kex * cad[i - 1] - a) * DT / 15.0
                g = g + DT * dg
                x = max(x + DT * dx, 0.0)
                if g < 30.0:
                    g = 30.0
            c = cgm[i]
            if c == c:  # not NaN
                g += OBS_GAIN * (c - g)
            G[i], X[i], A[i] = g, x, a
        return G, X, A

    def forecast(self, idx: np.ndarray, horizon_steps: int, extra_ra: np.ndarray | None = None,
                 cadence: np.ndarray | None = None) -> np.ndarray:
        """Roll the twin forward from the synchronised state at each index in ``idx``.

        Only meals logged at or before each start time are used (no peeking at the future).
        ``extra_ra`` (len(idx), H) adds planned meals; ``cadence`` (len(idx), H) adds planned activity.
        Returns glucose of shape (len(idx), H) for steps 1..H ahead.
        """
        p, inp = self.p, self.inp
        idx = np.asarray(idx)
        H = horizon_steps
        k = self.meals_known[idx]
        cols = np.clip(idx[:, None] + np.arange(H)[None, :], 0, inp.n - 1)
        ra = self.ra_cum[k[:, None], cols]
        # meals beyond the recorded horizon: extend analytically for the last rows
        overflow = idx[:, None] + np.arange(H)[None, :] >= inp.n
        if overflow.any():
            ra = np.where(overflow, self._ra_extrapolate(k, idx, H), ra)
        if extra_ra is not None:
            ra = ra + extra_ra
        cad = np.zeros((len(idx), H)) if cadence is None else cadence
        g, x, a = self.G[idx].copy(), self.X[idx].copy(), self.A[idx].copy()
        out = np.empty((len(idx), H))
        for h in range(H):
            dg = -(p.sg + x + a) * g + p.sg * p.gb + ra[:, h] / inp.vol
            dx = p.p2 * (p.kx * np.maximum(g - p.gb, 0.0) - x)
            a = a + (p.k_ex * cad[:, h] - a) * DT / 15.0
            g = np.maximum(g + DT * dg, 30.0)
            x = np.maximum(x + DT * dx, 0.0)
            out[:, h] = g
        return out

    def _ra_extrapolate(self, k: np.ndarray, idx: np.ndarray, H: int) -> np.ndarray:
        inp, p = self.inp, self.p
        t = (idx[:, None] + np.arange(H)[None, :]) * DT
        out = np.zeros((len(idx), H))
        for j in range(len(inp.meal_step)):
            known = (k > j)[:, None]
            s = t - inp.meal_step[j] * DT
            tau = inp.meal_tau[j] * p.gut
            val = np.where(s >= 0, inp.meal_f[j] * inp.meal_carbs[j] * 1000 * s / tau**2 * np.exp(-np.clip(s, 0, None) / tau), 0)
            out += np.where(known, val, 0)
        return out

    # ------------------------------------------------------------------ derived quantities
    def carbs_on_board(self, idx: np.ndarray) -> np.ndarray:
        """Grams of logged carbohydrate still to be absorbed at each index."""
        inp, p = self.inp, self.p
        t = np.asarray(idx) * DT
        cob = np.zeros(len(t))
        for j in range(len(inp.meal_step)):
            s = t - inp.meal_step[j] * DT
            tau = inp.meal_tau[j] * p.gut
            frac_left = np.where(s >= 0, (1 + s / tau) * np.exp(-np.clip(s, 0, None) / tau), 0.0)
            cob += inp.meal_carbs[j] * frac_left
        return cob

    def planned_meal_ra(self, start_idx: int, H: int, offset_min: float, meal_key: str, portion: float = 1.0,
                        carbs_g: float | None = None) -> np.ndarray:
        m = MEALS[meal_key]
        carbs = (carbs_g if carbs_g is not None else m.carbs_g) * portion
        s = np.arange(H) * DT - offset_min
        tau = m.absorption_tau * self.p.gut
        return np.where(s >= 0, m.bioavailability * carbs * 1000 * s / tau**2 * np.exp(-np.clip(s, 0, None) / tau), 0.0)

    def simulate_scenario(self, i: int, horizon_min: int = 360, meals: list[dict] | None = None,
                          walks: list[dict] | None = None) -> np.ndarray:
        """What-if: roll forward from *now* (index i) with planned meals and walks."""
        H = int(horizon_min / DT)
        extra = np.zeros(H)
        for m in meals or []:
            extra += self.planned_meal_ra(i, H, m.get("offset_min", 0), m["meal_key"], m.get("portion", 1.0))
        cad = np.zeros(H)
        for w in walks or []:
            s = int(w.get("offset_min", 0) / DT)
            e = s + int(np.ceil(w.get("duration_min", 15) / DT))
            cad[max(s, 0): max(e, 0)] = w.get("cadence", 100)
        return self.forecast(np.array([i]), H, extra_ra=extra[None, :], cadence=cad[None, :])[0]


# ---------------------------------------------------------------------- calibration

def calibrate(inp: TwinInputs, prior: TwinParams, calib_end: int, stride: int = 3,
              horizons=(6, 12, 24), maxfev: int = 450) -> tuple[TwinParams, dict]:
    """MAP fit of the six twin parameters to the patient's own history (indices < calib_end)."""
    sub = TwinInputs(inp.ts[:calib_end], inp.cgm[:calib_end], inp.cadence[:calib_end],
                     *_meals_before(inp, calib_end), inp.vol)
    H = max(horizons)
    starts = np.arange(12, calib_end - H, stride)
    target = np.stack([sub.cgm[starts + h] for h in horizons], axis=1)
    mask = np.isfinite(target)
    prior_log = prior.to_log()
    w = np.array([1.0, 1.0, 1.5])

    def loss(v):
        p = TwinParams.from_log(v)
        tw = Twin(p, sub)
        fc = tw.forecast(starts, H)
        pred = np.stack([fc[:, h - 1] for h in horizons], axis=1)
        err = np.where(mask, (pred - np.nan_to_num(target)) ** 2, 0.0)
        mse = (err.sum(0) / mask.sum(0) * w).sum() / w.sum()
        return mse / 400.0 + 0.5 * np.sum(((v - prior_log) / PRIOR_SD) ** 2) / len(starts) * 50

    res = minimize(loss, prior_log, method="Nelder-Mead",
                   options={"maxfev": maxfev, "xatol": 1e-3, "fatol": 1e-4, "adaptive": True})
    best = TwinParams.from_log(res.x)
    return best, {"loss": float(res.fun), "prior_loss": float(loss(prior_log)), "nfev": int(res.nfev)}


def _meals_before(inp: TwinInputs, end: int):
    keep = inp.meal_step < end
    return inp.meal_step[keep], inp.meal_carbs[keep], inp.meal_f[keep], inp.meal_tau[keep]
