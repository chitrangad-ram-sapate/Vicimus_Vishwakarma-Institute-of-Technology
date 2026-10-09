"""Render docs/architecture.pdf and docs/presentation.pdf from the HTML sources, filling the
slide numbers from reports/metrics.json and reports/real_data_metrics.json so the deck always
matches the code. Requires Google Chrome or Microsoft Edge (headless).

    python scripts/build_docs.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
SRC = DOCS / "src"
BROWSERS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "google-chrome", "chromium", "chromium-browser", "msedge",
]


def browser() -> str:
    for b in BROWSERS:
        if Path(b).exists() or shutil.which(b):
            return b
    raise SystemExit("Chrome/Edge not found: needed for headless PDF rendering")


def render(html: Path, pdf: Path, png: Path | None = None, size: str | None = None) -> None:
    b = browser()
    url = html.resolve().as_uri()
    subprocess.run([b, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf}", url], check=True, capture_output=True)
    if png and size:
        subprocess.run([b, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--window-size={size}",
                        f"--screenshot={png}", url], check=True, capture_output=True)
    print(f"  wrote {pdf.relative_to(ROOT)}")


def fill_presentation() -> Path:
    m = json.loads((ROOT / "reports" / "metrics.json").read_text())
    f = m["final"]
    vals = {
        "lead": f"{f['events']['median_lead_min']:.0f}",
        "rmse60": f"{f['60']['rmse']:.1f}", "pers60": f"{m['baselines']['Persistence']['60']['rmse']:.1f}",
        "mard60": f"{f['60']['mard_pct']:.1f}", "clarke": f"{f['60']['clarke_AB_pct']:.1f}",
        "coverage": f"{f['60']['coverage_pct']:.0f}",
        "detected_pct": f"{f['events']['sensitivity_pct']:.0f}",
        "detected": f"{f['events']['detected']}/{f['events']['excursions']}",
        "fa": f"{f['events']['false_alerts_per_patient_day']:.1f}",
        "rho_kx": f"{m['twin_fidelity']['spearman_kx']:.2f}", "rho_gb": f"{m['twin_fidelity']['spearman_gb']:.2f}",
        "calib_gain": f"{m['twin_fidelity']['mean_loss_reduction_pct']:.0f}",
        "final_rows": "".join(
            f"<tr><td>{h} min</td><td class='n'>{f[h]['rmse']:.1f}</td><td class='n'>{f[h]['mard_pct']:.1f}%</td>"
            f"<td class='n'>{f[h]['clarke_A_pct']:.1f}%</td><td class='n'>{f[h]['coverage_pct']:.1f}%</td></tr>"
            for h in ("30", "60", "120")),
    }
    rp = ROOT / "reports" / "real_data_metrics.json"
    if rp.exists():
        r = json.loads(rp.read_text())
        sc = r["status_counts"]
        rows = []
        for name, src in list(r["baselines"].items()) + list(r["ablation"].items()):
            cls = " class='hl'" if name.startswith("+ Twin") else ""
            rows.append(f"<tr{cls}><td>{name}</td>" + "".join(f"<td class='n'>{src[h]['rmse']:.1f}</td>" for h in ("30", "60", "120")) + "</tr>")
        e = r.get("events_180", {})
        vals.update({
            "real_n": str(r["participants"]),
            "real_mix": f"{sc.get('Healthy', 0)} healthy, {sc.get('Prediabetes', 0)} prediabetes, {sc.get('T2D', 0)} T2D",
            "real_rows": "".join(rows),
            "real_events": (f"Spikes &gt;180 flagged in advance: <b>{e['detected']}/{e['excursions']}</b>, median lead "
                            f"<b>{e['median_lead_min']:.0f} min</b>." if e.get("median_lead_min") else ""),
        })
    html = (SRC / "presentation.html").read_text(encoding="utf-8")
    for k, v in vals.items():
        html = html.replace("{{" + k + "}}", v)
    missing = [seg.split("}}")[0] for seg in html.split("{{")[1:]]
    if missing:
        raise SystemExit(f"unfilled placeholders: {missing}")
    out = SRC / "_presentation.rendered.html"
    out.write_text(html, encoding="utf-8")
    return out


def main() -> None:
    render(SRC / "architecture.html", DOCS / "architecture.pdf", DOCS / "architecture.png", "1587,1122")
    rendered = fill_presentation()
    render(rendered, DOCS / "presentation.pdf")
    rendered.unlink()


if __name__ == "__main__":
    main()
