"""Clinical and statistical evaluation metrics for glucose forecasting and event alerts."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from . import config as C


def clarke_zones(ref: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Clarke Error Grid zone (A-E) for each (reference, prediction) pair in mg/dL (Clarke et al., 1987)."""
    ref, pred = np.asarray(ref, float), np.asarray(pred, float)
    z = np.full(ref.shape, "B", dtype="<U1")
    a = ((ref <= 70) & (pred <= 70)) | ((pred <= 1.2 * ref) & (pred >= 0.8 * ref))
    e = ((ref >= 180) & (pred <= 70)) | ((ref <= 70) & (pred >= 180))
    c = (((ref >= 70) & (ref <= 290)) & (pred >= ref + 110)) | \
        (((ref >= 130) & (ref <= 180)) & (pred <= (7 / 5) * ref - 182))
    d = ((ref >= 240) & ((pred >= 70) & (pred <= 180))) | \
        ((ref <= 175 / 3) & (pred <= 180) & (pred >= 70)) | \
        (((ref >= 175 / 3) & (ref <= 70)) & (pred >= (6 / 5) * ref))
    z[d] = "D"
    z[c] = "C"
    z[e] = "E"
    z[a] = "A"
    return z


def regression_report(y: np.ndarray, pred: np.ndarray, lo: np.ndarray | None = None, hi: np.ndarray | None = None) -> dict:
    ok = np.isfinite(y) & np.isfinite(pred)
    y, pred = y[ok], pred[ok]
    zones = clarke_zones(y, pred)
    out = {
        "n": int(ok.sum()),
        "rmse": float(np.sqrt(np.mean((pred - y) ** 2))),
        "mae": float(np.mean(np.abs(pred - y))),
        "mard_pct": float(np.mean(np.abs(pred - y) / y) * 100),
        "clarke_A_pct": float(np.mean(zones == "A") * 100),
        "clarke_AB_pct": float(np.mean(np.isin(zones, ["A", "B"])) * 100),
    }
    if lo is not None and hi is not None:
        lo, hi = lo[ok], hi[ok]
        out["coverage_pct"] = float(np.mean((y >= lo) & (y <= hi)) * 100)
        out["mean_width"] = float(np.mean(hi - lo))
    return out


def classification_report(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok].astype(int), p[ok]
    if y.sum() == 0 or y.sum() == len(y):
        return {"n": int(len(y)), "positives": int(y.sum())}
    pred = p >= thr
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    return {
        "n": int(len(y)), "positives": int(y.sum()), "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)), "threshold": float(thr),
        "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
    }


def best_f1_threshold(y: np.ndarray, p: np.ndarray, min_recall: float = 0.0) -> float:
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok].astype(int), p[ok]
    best, best_t = -1.0, 0.5
    for t in np.linspace(0.05, 0.95, 91):
        pred = p >= t
        tp = (pred & (y == 1)).sum()
        prec = tp / max(pred.sum(), 1)
        rec = tp / max(y.sum(), 1)
        if rec < min_recall:
            continue
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        if f1 > best:
            best, best_t = f1, t
    return float(best_t)


def choose_alert_threshold(per_patient: list[tuple[np.ndarray, np.ndarray]], max_false_per_day: float = 1.0) -> float:
    """Pick the operating point a clinician would: the most sensitive threshold whose false-alert
    burden stays under ``max_false_per_day`` per patient (alert fatigue is a patient-safety issue)."""
    best_t, best_sens = 0.9, -1.0
    for t in np.linspace(0.3, 0.95, 27):
        ev = det = 0
        fa = days = 0.0
        for cgm, prob in per_patient:
            r = event_lead_times(cgm, prob, t)
            d = len(cgm) * C.SAMPLE_MIN / 1440
            ev, det = ev + r["events"], det + r["detected"]
            fa, days = fa + r["false_alerts_per_day"] * d, days + d
        sens = det / max(ev, 1)
        if fa / days <= max_false_per_day and sens > best_sens:
            best_t, best_sens = float(t), sens
    return best_t


def excursion_onsets(cgm: np.ndarray, refractory_steps: int = 12) -> list[int]:
    """Indices where glucose newly crosses 180 mg/dL after at least ``refractory_steps`` below it,
    so sensor noise around the threshold is not counted as repeated events."""
    g = np.asarray(cgm, float)
    above = g >= C.HYPER_THRESHOLD
    onsets, last_above = [], -10**9
    for i in range(1, len(g)):
        if above[i] and not above[i - 1] and np.isfinite(g[i - 1]) and i - last_above > refractory_steps:
            onsets.append(i)
        if above[i]:
            last_above = i
    return onsets


ALERT_CONFIRM_STEPS = 2    # risk must stay above threshold for 2 readings (10 min) before notifying
ALERT_SNOOZE_STEPS = 12    # no repeat notification within 60 min, like CGM alarm snooze


def alert_notifications(prob: np.ndarray, thr: float, confirm: int = ALERT_CONFIRM_STEPS,
                        snooze: int = ALERT_SNOOZE_STEPS) -> np.ndarray:
    """Indices at which a patient/clinician would actually be notified under the alert policy:
    a confirmed rise of risk above ``thr`` that is not inside the snooze period of a previous alert."""
    hi = np.nan_to_num(np.asarray(prob, float)) >= thr
    confirmed = hi.copy()
    for k in range(1, confirm):
        confirmed &= np.r_[np.zeros(k, bool), hi[:-k]]
    rising = confirmed & ~np.r_[False, confirmed[:-1]]
    out, last = [], -10**9
    for i in np.where(rising)[0]:
        if i - last >= snooze:
            out.append(i)
            last = i
    return np.array(out, dtype=int)


def event_lead_times(cgm: np.ndarray, prob: np.ndarray, thr: float, window_steps: int = 24,
                     refractory_steps: int = 12) -> dict:
    """Event-level evaluation: for every new excursion above 180 mg/dL, did an alert fire beforehand and how early?

    Also counts false alert episodes (alert onsets with no excursion in the following window).
    """
    g = np.asarray(cgm, float)
    above = g >= C.HYPER_THRESHOLD
    onsets = excursion_onsets(g, refractory_steps)
    fired = alert_notifications(prob, thr)
    leads, detected = [], 0
    for o in onsets:
        hits = fired[(fired >= o - window_steps) & (fired < o)]
        if len(hits):
            detected += 1
            leads.append((o - hits[0]) * C.SAMPLE_MIN)
    false_alerts = sum(1 for a in fired if not above[a: a + window_steps + 1].any())
    days = len(g) * C.SAMPLE_MIN / 1440
    return {
        "events": len(onsets), "detected": detected,
        "sensitivity_pct": 100 * detected / max(len(onsets), 1),
        "median_lead_min": float(np.median(leads)) if leads else 0.0,
        "pct_detected_30min_early": 100 * float(np.mean(np.array(leads) >= 30)) if leads else 0.0,
        "false_alerts_per_day": false_alerts / max(days, 1e-9),
    }
