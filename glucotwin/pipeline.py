"""End-to-end pipeline: calibrate twins -> fuse features -> train -> evaluate -> report.

    python -m glucotwin.pipeline            # full run (generates data if missing)
    python -m glucotwin.pipeline --regen    # regenerate the synthetic cohort first

Evaluation protocol (no leakage):
  * Subjects are split 60/20/20 into train / conformal-calibration / test. Test patients are never
    seen by the ML models.
  * Every patient's twin is calibrated only on days 1-10 of their own data; metrics are reported on
    days 11-14 of test patients, i.e. forward in time for the twin and on unseen people for the ML.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor

import joblib
import numpy as np
import pandas as pd

from . import config as C
from .features import feature_groups, patient_features
from .metrics import (best_f1_threshold, choose_alert_threshold, classification_report, event_lead_times,
                      regression_report)
from .models import EventClassifier, GlucoseForecaster
from .twin import Twin, TwinParams, build_inputs, calibrate, ehr_prior

CALIB_END = C.CALIB_DAYS * 1440 // C.SAMPLE_MIN


def load_data() -> dict[str, pd.DataFrame]:
    names = ["streams", "meals", "sleep", "ehr", "true_params", "hba1c_history", "activity"]
    return {n: pd.read_parquet(C.RAW_DIR / f"{n}.parquet") for n in names}


def split_subjects(ids: list[str], seed: int = C.SEED) -> dict[str, list[str]]:
    rng = np.random.default_rng(seed + 1)
    ids = list(rng.permutation(sorted(ids)))
    n = len(ids)
    a = int(round(C.SPLIT_FRACTIONS[0] * n))
    b = a + int(round(C.SPLIT_FRACTIONS[1] * n))
    return {"train": sorted(ids[:a]), "calib": sorted(ids[a:b]), "test": sorted(ids[b:])}


def _calibrate_one(args):
    pid, stream, meals, ehr_row = args
    inp = build_inputs(stream, meals, float(ehr_row["weight_kg"]))
    prior = ehr_prior(ehr_row)
    params, info = calibrate(inp, prior, CALIB_END)
    return pid, prior.to_dict(), params.to_dict(), info


def calibrate_all(data: dict, workers: int | None = None) -> pd.DataFrame:
    ehr = data["ehr"].set_index("patient_id")
    jobs = [(pid, data["streams"][data["streams"].patient_id == pid], data["meals"][data["meals"].patient_id == pid],
             ehr.loc[pid]) for pid in ehr.index]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for pid, prior, params, info in ex.map(_calibrate_one, jobs):
            rows.append({"patient_id": pid, **{f"prior_{k}": v for k, v in prior.items()}, **params, **info})
            print(f"  twin {pid}: loss {info['prior_loss']:.2f} -> {info['loss']:.2f}")
    return pd.DataFrame(rows).sort_values("patient_id").reset_index(drop=True)


def load_twin(pid: str, data: dict, twin_params: pd.DataFrame) -> Twin:
    ehr = data["ehr"].set_index("patient_id")
    row = twin_params.set_index("patient_id").loc[pid]
    p = TwinParams(**{k: float(row[k]) for k in ("gb", "sg", "kx", "p2", "gut", "k_ex")})
    inp = build_inputs(data["streams"][data["streams"].patient_id == pid],
                       data["meals"][data["meals"].patient_id == pid], float(ehr.loc[pid, "weight_kg"]))
    return Twin(p, inp)


def build_feature_table(data: dict, twin_params: pd.DataFrame) -> pd.DataFrame:
    ehr = data["ehr"].set_index("patient_id")
    frames = []
    for pid in ehr.index:
        tw = load_twin(pid, data, twin_params)
        frames.append(patient_features(
            data["streams"][data["streams"].patient_id == pid], data["meals"][data["meals"].patient_id == pid],
            data["sleep"][data["sleep"].patient_id == pid], ehr.loc[pid], tw))
    return pd.concat(frames, ignore_index=True)


ABLATIONS = {
    "CGM only": ["CGM"],
    "+ Wearables & meal log": ["CGM", "Wearables", "Meal log"],
    "+ EHR (stream fusion)": ["CGM", "Wearables", "Meal log", "EHR"],
    "+ Twin physiology (full hybrid)": ["CGM", "Wearables", "Meal log", "EHR", "Twin"],
}


def train_and_evaluate(feat: pd.DataFrame, split: dict, data: dict, twin_params: pd.DataFrame) -> dict:
    groups = feature_groups(feat.columns)
    tr = feat[feat.patient_id.isin(split["train"])]
    ca = feat[feat.patient_id.isin(split["calib"]) & (feat.day >= C.CALIB_DAYS)]
    te = feat[feat.patient_id.isin(split["test"]) & (feat.day >= C.CALIB_DAYS)]
    results: dict = {"split": split, "n_rows": {"train": len(tr), "calib": len(ca), "test": len(te)}}

    # ---------------- baselines
    base = {}
    for h in C.HORIZONS_MIN:
        y = te[f"y_{h}"].to_numpy()
        base.setdefault("Persistence", {})[h] = regression_report(y, te["g0"].to_numpy())
        lin = np.clip(te["g0"] + te["d_15"].fillna(0) / 15 * h, 40, 400).to_numpy()
        base.setdefault("Linear extrapolation", {})[h] = regression_report(y, lin)
        base.setdefault("Twin only (physiology)", {})[h] = regression_report(y, te[f"twin_g{h}"].to_numpy())
    results["baselines"] = base
    results["baseline_spike_auroc"] = {
        "Persistence (current glucose)": classification_report(te["y_spike"].to_numpy(), te["g0"].to_numpy() / 400, 0.45).get("auroc"),
        "Twin only (simulated peak)": classification_report(
            te["y_spike"].to_numpy(), np.clip((te["g0"] + te["twin_peak_2h"]).to_numpy() / 400, 0, 1), 0.45).get("auroc"),
    }

    # ---------------- ablation: which data streams matter?
    ablation = {}
    for name, gs in ABLATIONS.items():
        feats = [f for g in gs for f in groups[g]]
        t0 = time.time()
        fc = GlucoseForecaster(feats, quantiles=False, n_estimators=300).fit(tr)
        pr = fc.predict(te)
        clf = EventClassifier(feats, "y_spike", n_estimators=250).fit(tr)
        ps = clf.predict_proba(te)
        ablation[name] = {
            **{h: regression_report(te[f"y_{h}"].to_numpy(), pr[h]["point"]) for h in C.HORIZONS_MIN},
            "spike": classification_report(te["y_spike"].to_numpy(), ps, 0.5),
        }
        print(f"  ablation '{name}': RMSE@60={ablation[name][60]['rmse']:.1f} "
              f"spike AUROC={ablation[name]['spike'].get('auroc', float('nan')):.3f} ({time.time() - t0:.0f}s)")
    results["ablation"] = ablation

    # ---------------- final hybrid model
    feats = [f for g in ABLATIONS["+ Twin physiology (full hybrid)"] for f in groups[g]]
    fc = GlucoseForecaster(feats).fit(tr)
    results["conformal_q"] = fc.calibrate_intervals(ca)
    spike = EventClassifier(feats, "y_spike").fit(tr)
    ca_p = ca.assign(p=spike.predict_proba(ca))
    spike.threshold = choose_alert_threshold([(d["cgm"].to_numpy(), d["p"].to_numpy()) for _, d in ca_p.groupby("patient_id")])
    hypo = None
    if tr["y_hypo"].sum() >= 50:
        hypo = EventClassifier(feats, "y_hypo").fit(tr)
        hypo.threshold = best_f1_threshold(ca["y_hypo"].to_numpy(), hypo.predict_proba(ca))

    pr = fc.predict(te)
    results["final"] = {h: regression_report(te[f"y_{h}"].to_numpy(), pr[h]["point"], pr[h]["lo"], pr[h]["hi"])
                        for h in C.HORIZONS_MIN}
    p_spike = spike.predict_proba(te)
    results["final"]["spike"] = classification_report(te["y_spike"].to_numpy(), p_spike, spike.threshold)
    if hypo is not None:
        results["final"]["hypo"] = classification_report(te["y_hypo"].to_numpy(), hypo.predict_proba(te), hypo.threshold)

    # event-level: per test patient, then pooled
    ev = {"events": 0, "detected": 0, "leads": [], "false_alerts": 0.0, "days": 0.0}
    te2 = te.assign(p_spike=p_spike)
    per_patient = []
    for pid, d in te2.groupby("patient_id"):
        r = event_lead_times(d["cgm"].to_numpy(), d["p_spike"].to_numpy(), spike.threshold)
        per_patient.append(r)
        days = len(d) * C.SAMPLE_MIN / 1440
        ev["events"] += r["events"]
        ev["detected"] += r["detected"]
        ev["false_alerts"] += r["false_alerts_per_day"] * days
        ev["days"] += days
    results["final"]["events"] = {
        "excursions": ev["events"], "detected": ev["detected"],
        "sensitivity_pct": 100 * ev["detected"] / max(ev["events"], 1),
        "median_lead_min": float(np.median([p["median_lead_min"] for p in per_patient if p["detected"]])),
        "false_alerts_per_patient_day": ev["false_alerts"] / ev["days"],
    }

    # global explanation: mean |SHAP| by data stream
    contrib = spike.contributions(te.sample(min(len(te), 20000), random_state=0)).drop(columns="_bias").abs().mean()
    results["shap_by_stream"] = {g: float(contrib[[f for f in fs if f in contrib.index]].sum()) for g, fs in groups.items()}
    results["shap_top_features"] = contrib.sort_values(ascending=False).head(15).round(4).to_dict()

    # twin fidelity: recovered vs hidden ground-truth physiology
    tp = data["true_params"].set_index("patient_id")
    tw = twin_params.set_index("patient_id")
    results["twin_fidelity"] = {
        "spearman_gb": float(tw["gb"].rank().corr(tp["gb"].rank())),
        "spearman_kx": float(tw["kx"].rank().corr(tp["kx"].rank())),
        "spearman_gut": float(tw["gut"].rank().corr(tp["gut_scale"].rank())),
        "mean_loss_reduction_pct": float(100 * (1 - (tw["loss"] / tw["prior_loss"]).mean())),
    }

    C.ARTIFACT_DIR.mkdir(exist_ok=True, parents=True)
    joblib.dump({"forecaster": fc, "spike": spike, "hypo": hypo, "features": feats, "groups": groups,
                 "split": split}, C.ARTIFACT_DIR / "models.joblib")
    return results


def make_figures(results: dict, feat: pd.DataFrame, data: dict, twin_params: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .metrics import clarke_zones

    fig_dir = C.REPORT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    blue, grey = "#2563eb", "#9ca3af"

    # 1. ablation
    names = list(results["baselines"]) + list(results["ablation"])
    fig, ax = plt.subplots(figsize=(10, 4.2))
    w = 0.12
    for j, name in enumerate(names):
        src = results["baselines"].get(name) or results["ablation"][name]
        vals = [src[h]["rmse"] if h in src else src[str(h)]["rmse"] for h in C.HORIZONS_MIN]
        color = grey if name in results["baselines"] else plt.cm.Blues(0.35 + 0.18 * (j - 3))
        ax.bar(np.arange(3) + (j - len(names) / 2) * w, vals, w, label=name, color=color, edgecolor="white")
    ax.set_xticks(range(3), [f"{h} min" for h in C.HORIZONS_MIN])
    ax.set_ylabel("RMSE (mg/dL), lower is better")
    ax.set_title("Forecast error by data stream: test patients, unseen days")
    ax.legend(fontsize=8, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "ablation_rmse.png", dpi=150)
    plt.close(fig)

    # 2. Clarke error grid at 60 min (final model)
    obj = joblib.load(C.ARTIFACT_DIR / "models.joblib")
    te = feat[feat.patient_id.isin(obj["split"]["test"]) & (feat.day >= C.CALIB_DAYS)]
    pr = obj["forecaster"].predict(te)
    y = te["y_60"].to_numpy()
    p = pr[60]["point"]
    ok = np.isfinite(y)
    z = clarke_zones(y[ok], p[ok])
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    colors = {"A": "#16a34a", "B": "#2563eb", "C": "#f59e0b", "D": "#dc2626", "E": "#7c3aed"}
    for zone, col in colors.items():
        m = z == zone
        ax.scatter(y[ok][m], p[ok][m], s=2, alpha=0.35, color=col, label=f"{zone}: {m.mean() * 100:.1f}%")
    ax.plot([0, 400], [0, 400], color="black", lw=0.6)
    ax.set_xlim(40, 400), ax.set_ylim(40, 400)
    ax.set_xlabel("Reference CGM (mg/dL)"), ax.set_ylabel("Predicted (mg/dL)")
    ax.set_title("Clarke error grid: 60-min forecast")
    ax.legend(markerscale=6, frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "clarke_60min.png", dpi=150)
    plt.close(fig)

    # 3. example forecast for one test patient-day
    pid = sorted(obj["split"]["test"])[0]
    d = te[(te.patient_id == pid) & (te.day == C.CALIB_DAYS + 1)]
    pr_d = obj["forecaster"].predict(d)
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.axhspan(70, 180, color="#16a34a", alpha=0.06)
    ax.plot(d["ts"], d["cgm"], ".", ms=3, color="black", label="CGM (observed)")
    shift = pd.Timedelta(minutes=60)
    ax.plot(d["ts"] + shift, pr_d[60]["point"], color=blue, lw=1.4, label="60-min-ahead forecast")
    ax.fill_between(d["ts"] + shift, pr_d[60]["lo"], pr_d[60]["hi"], color=blue, alpha=0.15, label="90% conformal interval")
    ax.set_ylabel("mg/dL")
    ax.set_title(f"{pid}: 60-minute forecasts on an unseen day")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "forecast_example.png", dpi=150)
    plt.close(fig)

    # 4. twin parameter recovery
    tp = data["true_params"].set_index("patient_id")
    tw = twin_params.set_index("patient_id").loc[tp.index]
    fig, axs = plt.subplots(1, 2, figsize=(9, 4))
    axs[0].scatter(tp["kx"] * 1e4, tw["kx"] * 1e4, s=14, color=blue)
    axs[0].set_xlabel("true insulin-action gain (×1e-4)"), axs[0].set_ylabel("twin estimate")
    axs[0].set_title(f"Insulin action, Spearman ρ={results['twin_fidelity']['spearman_kx']:.2f}")
    axs[1].scatter(tp["gb"], tw["gb"], s=14, color=blue)
    axs[1].set_xlabel("true basal glucose (mg/dL)"), axs[1].set_ylabel("twin estimate")
    axs[1].set_title(f"Basal glucose, Spearman ρ={results['twin_fidelity']['spearman_gb']:.2f}")
    fig.tight_layout()
    fig.savefig(fig_dir / "twin_recovery.png", dpi=150)
    plt.close(fig)

    # 5. SHAP by stream
    s = pd.Series(results["shap_by_stream"]).sort_values()
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh(s.index, s.values / s.sum() * 100, color=blue)
    ax.set_xlabel("share of mean |SHAP| for 2-h spike alerts (%)")
    ax.set_title("What drives the alerts")
    fig.tight_layout()
    fig.savefig(fig_dir / "shap_by_stream.png", dpi=150)
    plt.close(fig)


def write_report(results: dict) -> None:
    def row(name, src):
        cells = []
        for h in C.HORIZONS_MIN:
            r = src[h]
            cells.append(f"{r['rmse']:.1f} / {r['clarke_AB_pct']:.1f}%")
        spike = src.get("spike", {}).get("auroc")
        return f"| {name} | " + " | ".join(cells) + f" | {spike:.3f} |" if spike else f"| {name} | " + " | ".join(cells) + " | n/a |"

    lines = ["# GlucoTwin: evaluation results", "",
             "_Auto-generated by `python -m glucotwin.pipeline`. Test set = unseen patients, unseen days._", "",
             "## Forecast error and spike-alert discrimination", "",
             "| Model | RMSE / Clarke A+B @30 min | @60 min | @120 min | 2-h spike AUROC |",
             "|---|---|---|---|---|"]
    for name, src in results["baselines"].items():
        lines.append(row(name, src))
    for name, src in results["ablation"].items():
        lines.append(row(name, src))
    f = results["final"]
    lines += ["", "## Final hybrid twin", "",
              "| Horizon | RMSE | MAE | MARD | Clarke A | Clarke A+B | 90% interval coverage | mean width |",
              "|---|---|---|---|---|---|---|---|"]
    for h in C.HORIZONS_MIN:
        r = f[h]
        lines.append(f"| {h} min | {r['rmse']:.1f} | {r['mae']:.1f} | {r['mard_pct']:.1f}% | {r['clarke_A_pct']:.1f}% | "
                     f"{r['clarke_AB_pct']:.1f}% | {r['coverage_pct']:.1f}% | {r['mean_width']:.0f} mg/dL |")
    s, e = f["spike"], f["events"]
    lines += ["", f"**2-hour hyperglycaemia alert:** AUROC {s['auroc']:.3f}, AUPRC {s['auprc']:.3f}, "
              f"precision {s['precision']:.2f}, recall {s['recall']:.2f} at threshold {s['threshold']:.2f}.", "",
              f"**Event level:** {e['detected']}/{e['excursions']} excursions above 180 mg/dL flagged in advance "
              f"({e['sensitivity_pct']:.0f}%), median lead time **{e['median_lead_min']:.0f} min**, "
              f"{e['false_alerts_per_patient_day']:.2f} false alerts per patient-day."]
    if "hypo" in f and "auroc" in f["hypo"]:
        hh = f["hypo"]
        lines += ["", f"**2-hour hypoglycaemia alert:** AUROC {hh['auroc']:.3f} ({hh['positives']} positive windows)."]
    tf = results["twin_fidelity"]
    lines += ["", "## Twin fidelity (parameter recovery against hidden ground truth)", "",
              f"- Spearman ρ basal glucose: {tf['spearman_gb']:.2f}",
              f"- Spearman ρ insulin-action gain: {tf['spearman_kx']:.2f}",
              f"- Spearman ρ gut absorption speed: {tf['spearman_gut']:.2f}",
              f"- Personal calibration reduces twin forecast loss by {tf['mean_loss_reduction_pct']:.0f}% vs EHR-only prior", "",
              "## Share of alert explanations by data stream", ""]
    tot = sum(results["shap_by_stream"].values())
    for g, v in sorted(results["shap_by_stream"].items(), key=lambda x: -x[1]):
        lines.append(f"- {g}: {100 * v / tot:.0f}%")
    lines += ["", "![ablation](figures/ablation_rmse.png)", "![clarke](figures/clarke_60min.png)",
              "![forecast](figures/forecast_example.png)", "![twin](figures/twin_recovery.png)",
              "![shap](figures/shap_by_stream.png)"]
    (C.REPORT_DIR / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)


def main(regen: bool = False, workers: int | None = None) -> None:
    C.ensure_dirs()
    t0 = time.time()
    if regen or not (C.RAW_DIR / "streams.parquet").exists():
        from .synth import generate
        print("[1/5] Generating synthetic cohort ...")
        generate(verbose=False)
    data = load_data()
    print("[2/5] Calibrating a personal twin for every patient (days 1-10) ...")
    twin_params = calibrate_all(data, workers)
    twin_params.to_parquet(C.ARTIFACT_DIR / "twin_params.parquet", index=False)
    print("[3/5] Fusing EHR + wearable + twin features ...")
    feat = build_feature_table(data, twin_params)
    feat.to_parquet(C.ARTIFACT_DIR / "features.parquet", index=False)
    split = split_subjects(data["ehr"]["patient_id"].tolist())
    print(f"[4/5] Training and evaluating (train {len(split['train'])} / calib {len(split['calib'])} / "
          f"test {len(split['test'])} patients) ...")
    results = train_and_evaluate(feat, split, data, twin_params)
    (C.REPORT_DIR / "metrics.json").write_text(json.dumps(results, indent=2, default=_json_default))
    print("[5/5] Writing figures and report ...")
    make_figures(results, feat, data, twin_params)
    write_report(results)
    print(f"Done in {time.time() - t0:.0f}s -> {C.REPORT_DIR / 'RESULTS.md'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--regen", action="store_true", help="regenerate the synthetic cohort")
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    main(a.regen, a.workers)
