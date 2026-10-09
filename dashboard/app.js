/* GlucoTwin clinician dashboard: vanilla JS + Plotly, served by FastAPI. */
"use strict";

const state = { meta: null, now: null, pid: null, timer: null, patient: null };
const $ = (sel) => document.querySelector(sel);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const parseTs = (s) => new Date(s.replace(" ", "T"));
const fmtTs = (d) => {
  const p = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}T${p(d.getHours())}:${p(d.getMinutes())}:00`;
};
const fmtHuman = (d) => d.toLocaleString("en-IN", { weekday: "short", day: "numeric", month: "short", hour: "2-digit", minute: "2-digit", hour12: false });
const arrow = (d15) => d15 == null ? "" : d15 > 15 ? "⇈" : d15 > 6 ? "↗" : d15 < -15 ? "⇊" : d15 < -6 ? "↘" : "→";
const pct = (p) => `${Math.min(99, Math.max(1, Math.round(p * 100)))}%`;
const ordinal = (n) => n + ((n % 100 >= 10 && n % 100 <= 20) ? "th" : ({ 1: "st", 2: "nd", 3: "rd" }[n % 10] || "th"));

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
}

function plotLayout(extra = {}) {
  return Object.assign({
    margin: { l: 44, r: 12, t: 8, b: 30 },
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: css("--font"), size: 11, color: css("--muted") },
    xaxis: { gridcolor: css("--border"), linecolor: css("--border"), zeroline: false },
    yaxis: { gridcolor: css("--border"), linecolor: css("--border"), zeroline: false },
    showlegend: true, legend: { orientation: "h", y: 1.12, x: 0, font: { size: 11 } },
    hovermode: "x unified",
  }, extra);
}
const plotCfg = { displayModeBar: false, responsive: true };

// ------------------------------------------------------------------ ward
async function loadWard() {
  const w = await api(`/api/ward?at=${encodeURIComponent(fmtTs(state.now))}`);
  const list = $("#wardList");
  list.innerHTML = w.patients.map((p) => `
    <button class="ward-item ${p.patient_id === state.pid ? "active" : ""}" data-pid="${p.patient_id}">
      <span class="nm">${esc(p.name)}</span>
      <span class="g">${p.glucose == null ? "-" : Math.round(p.glucose)} ${arrow(p.trend_15)}</span>
      <span class="sub">${p.age}${p.sex} · ${esc(p.status)} · TIR ${p.tir_24h ?? "-"}%</span>
      <span><span class="tier ${p.tier}">${p.glucose >= 180 ? "above 180" : pct(p.p_spike)}</span></span>
    </button>`).join("");
  list.querySelectorAll(".ward-item").forEach((el) => el.addEventListener("click", () => selectPatient(el.dataset.pid)));
  if (!state.pid && w.patients.length) selectPatient(w.patients[0].patient_id);
}

async function selectPatient(pid) {
  state.pid = pid;
  document.querySelectorAll(".ward-item").forEach((el) => el.classList.toggle("active", el.dataset.pid === pid));
  await loadPatient(true);
}

// ------------------------------------------------------------------ patient
async function loadPatient(full = false) {
  if (!state.pid) return;
  const main = $("#main");
  main.classList.add("loading");
  const d = await api(`/api/patients/${state.pid}?at=${encodeURIComponent(fmtTs(state.now))}`);
  state.patient = d;
  if (full || !$("#chartGlucose")) renderShell(d);
  renderHeader(d);
  renderKpis(d);
  renderGlucose(d);
  renderReasons(d);
  renderWearables(d);
  renderInsights(d);
  main.classList.remove("loading");
}

function renderShell(d) {
  const mealOpts = ["breakfast", "lunch", "dinner", "snack"].map((slot) =>
    `<optgroup label="${slot}">${state.meta.meals.filter((m) => m.slot === slot)
      .map((m) => `<option value="${m.key}">${esc(m.name)} · ${Math.round(m.carbs_g)} g carbs, GI ${m.gi}</option>`).join("")}</optgroup>`).join("");
  $("#main").innerHTML = `
    <section class="ph" id="ph"></section>
    <section class="kpis" id="kpis"></section>
    <section class="grid">
      <div class="card span-8">
        <h3>Glucose · past 12 h and the twin's 2-h forecast</h3>
        <p class="desc">CGM (dots), logged meals (▲), ML forecast with 90% conformal interval, and the physiological twin's own trajectory.</p>
        <div id="chartGlucose" class="chart" style="height:320px"></div>
      </div>
      <div class="card span-4">
        <h3>Why this alert</h3>
        <p class="desc">Patient-specific drivers of the 2-h hyperglycaemia risk (TreeSHAP).</p>
        <div id="reasons"></div>
      </div>
      <div class="card span-8">
        <h3>Wearable stream · heart rate, HRV, activity, sleep</h3>
        <p class="desc">Simulated smartwatch data (Apple Health / Google Fit format), 5-min resolution.</p>
        <div id="chartWear" class="chart" style="height:300px"></div>
      </div>
      <div class="card span-4">
        <h3>Twin phenotype</h3>
        <p class="desc">Physiology calibrated to this patient, as percentile vs. cohort.</p>
        <div id="twinPheno"></div>
      </div>
      <div class="card span-12">
        <h3>What-if simulator · ask the twin before you prescribe</h3>
        <p class="desc">Simulates this patient's next 5 hours from their current physiological state.</p>
        <div class="sim-form">
          <label>Planned meal<select id="simMeal">${mealOpts}</select></label>
          <label>Portion <span id="simPortionV">1.0×</span><input type="range" id="simPortion" min="0.5" max="1.5" step="0.25" value="1"></label>
          <label>Eaten in <span id="simOffsetV">15 min</span><input type="range" id="simOffset" min="0" max="120" step="15" value="15"></label>
          <label>Walk after meal <span id="simWalkV">15 min</span><input type="range" id="simWalk" min="0" max="45" step="5" value="15"></label>
          <div style="display:flex;gap:8px"><button class="btn primary" id="simRun">Simulate</button><button class="btn" id="simRank">Best swaps</button></div>
        </div>
        <div class="grid">
          <div class="span-8"><div id="chartSim" class="chart" style="height:280px"></div><div class="sim-out" id="simOut"></div></div>
          <div class="span-4"><div id="simRankOut" class="muted" style="font-size:13px">Press <b>Best swaps</b> to rank alternative meals for this slot by simulated time above 180 mg/dL.</div></div>
        </div>
      </div>
      <div class="card span-6">
        <h3>Personal food response</h3>
        <p class="desc">Average glucose rise in the 2 h after each logged meal type.</p>
        <div id="food"></div>
      </div>
      <div class="card span-6">
        <h3>Sleep and next-day glucose</h3>
        <p class="desc">Last 7 nights from the wearable; bars = sleep hours, line = next-day mean glucose.</p>
        <div id="chartSleep" class="chart" style="height:220px"></div>
        <div id="sleepNote" class="muted" style="font-size:12px"></div>
      </div>
      <div class="card span-6">
        <h3>EHR · labs, genetics and history</h3>
        <p class="desc">Static stream, ingested as FHIR R4.</p>
        <div id="ehr"></div>
        <div id="chartA1c" class="chart" style="height:150px"></div>
      </div>
      <div class="card span-6">
        <h3>Pre-consultation summary</h3>
        <p class="desc">Auto-drafted from the twin for the doctor to verify.</p>
        <div id="summary"></div>
      </div>
    </section>`;
  const bind = (id, fmt) => { const el = $(id); const out = $(id + "V"); el.addEventListener("input", () => { out.textContent = fmt(el.value); }); };
  bind("#simPortion", (v) => `${(+v).toFixed(2).replace(/0$/, "")}×`);
  bind("#simOffset", (v) => `${v} min`);
  bind("#simWalk", (v) => (+v ? `${v} min` : "none"));
  $("#simRun").addEventListener("click", runSim);
  $("#simRank").addEventListener("click", runRank);
  const slot = d.forecast && guessSlot(state.now);
  const def = { breakfast: "idli_sambar", lunch: "rice_dal_sabzi", dinner: "rice_dal_sabzi", snack: "chai_biscuits" }[slot];
  $("#simMeal").value = def;
  runSim();
}

function guessSlot(dt) {
  const h = dt.getHours();
  return h < 10 ? "breakfast" : h < 15 ? "lunch" : h < 19 ? "snack" : "dinner";
}

function renderHeader(d) {
  const p = d.patient;
  $("#ph").innerHTML = `
    <div>
      <h1>${esc(p.name)}</h1>
      <div class="meta">${p.age} y · ${p.sex === "M" ? "Male" : "Female"} · ${esc(p.city)} · ABHA ${esc(p.abha_masked)} · ${esc(p.patient_id)}</div>
      <div class="chips">
        ${p.conditions.map((c) => `<span class="chip">${esc(c)}</span>`).join("")}
        ${p.medications.map((m) => `<span class="chip med">${esc(m)}</span>`).join("") || '<span class="chip">Lifestyle therapy only</span>'}
      </div>
    </div>
    <div class="ph-actions">
      <button class="btn" id="btnFhir">View FHIR record</button>
      <button class="btn" id="btnCopy">Copy summary</button>
    </div>`;
  $("#btnFhir").addEventListener("click", async () => {
    const f = await api(`/api/patients/${p.patient_id}/fhir`);
    $("#fhirPre").textContent = JSON.stringify(f, null, 2);
    $("#fhirDialog").showModal();
  });
  $("#btnCopy").addEventListener("click", () => {
    const s = d.insights.summary;
    navigator.clipboard?.writeText(`${s.text}\n\nSuggested actions:\n- ${s.actions.join("\n- ")}\n\n${s.disclaimer}`);
    $("#btnCopy").textContent = "Copied ✓";
    setTimeout(() => ($("#btnCopy").textContent = "Copy summary"), 1500);
  });
}

function renderKpis(d) {
  const c = d.current, r = d.risk, m = d.insights.metrics_7d || {};
  const g = c.glucose;
  const gCls = g == null ? "" : g > 250 || g < 70 ? "alert" : g > 180 ? "watch" : "";
  const sCls = r.p_spike >= 0.8 ? "alert" : r.p_spike >= r.spike_threshold ? "watch" : "";
  const hCls = r.hypo_threshold && r.p_hypo >= r.hypo_threshold ? "alert" : "";
  const f60 = d.forecast.horizons[60];
  $("#kpis").innerHTML = `
    <div class="kpi ${gCls}"><div class="lbl">Glucose now</div>
      <div class="val">${g == null ? "-" : Math.round(g)} <small>mg/dL ${arrow(c.trend_15)}</small></div>
      <div class="hint">60 min: ${Math.round(f60.point)} (${Math.round(f60.lo)}–${Math.round(f60.hi)})</div></div>
    <div class="kpi ${sCls}"><div class="lbl">2-h spike risk (&gt;180)</div>
      <div class="val">${pct(r.p_spike)}</div><div class="bar"><i style="width:${r.p_spike * 100}%;background:${r.p_spike >= r.spike_threshold ? css("--bad") : css("--accent")}"></i></div></div>
    <div class="kpi ${hCls}"><div class="lbl">2-h hypo risk (&lt;70)</div>
      <div class="val">${pct(r.p_hypo)}</div><div class="hint">${d.patient.medications.some((m) => /Glimepiride|Insulin/.test(m)) ? "on hypoglycaemic agents" : "low-risk therapy"}</div></div>
    <div class="kpi"><div class="lbl">Time in range · 7 d</div>
      <div class="val">${m.tir_pct ?? "-"}<small>%</small></div><div class="hint">target &gt; 70% · above ${m.tar_pct ?? "-"}%</div></div>
    <div class="kpi"><div class="lbl">GMI · CV</div>
      <div class="val">${m.gmi ?? "-"}<small>%</small></div><div class="hint">variability ${m.cv_pct ?? "-"}% ${m.cv_pct >= 36 ? "(labile)" : "(stable)"}</div></div>
    <div class="kpi"><div class="lbl">Carbs on board · steps today</div>
      <div class="val">${Math.round(c.cob_g ?? 0)}<small>g</small></div><div class="hint">${c.steps_today.toLocaleString("en-IN")} steps · slept ${c.sleep_last_h == null ? "-" : c.sleep_last_h.toFixed(1)} h</div></div>`;
}

function renderGlucose(d) {
  const h = d.history, f = d.forecast;
  const ts = h.ts.map(parseTs);
  const now = parseTs(d.at);
  const fx = [now, ...[30, 60, 120].map((m) => new Date(now.getTime() + m * 60000))];
  const g0 = d.current.glucose;
  const pts = [g0, ...[30, 60, 120].map((m) => f.horizons[m].point)];
  const lo = [g0, ...[30, 60, 120].map((m) => f.horizons[m].lo)];
  const hi = [g0, ...[30, 60, 120].map((m) => f.horizons[m].hi)];
  const accent = css("--accent"), violet = css("--violet");
  const mealY = d.meals.map(() => 52);
  const traces = [
    { x: ts, y: h.cgm, mode: "markers", name: "CGM", marker: { size: 4, color: css("--text") }, hovertemplate: "%{y:.0f} mg/dL<extra>CGM</extra>" },
    { x: [...fx, ...fx.slice().reverse()], y: [...hi, ...lo.slice().reverse()], fill: "toself", fillcolor: accent + "26", line: { width: 0 }, name: "90% interval", hoverinfo: "skip" },
    { x: fx, y: pts, mode: "lines+markers", name: "ML forecast", line: { color: accent, width: 2.5 }, marker: { size: 7 }, hovertemplate: "%{y:.0f} mg/dL<extra>forecast</extra>" },
    { x: f.twin_ts.map(parseTs), y: f.twin, mode: "lines", name: "Twin physiology", line: { color: violet, width: 1.5, dash: "dot" }, hovertemplate: "%{y:.0f}<extra>twin</extra>" },
    { x: d.meals.map((m) => parseTs(m.ts)), y: mealY, mode: "markers", name: "Meal logged", marker: { symbol: "triangle-up", size: 12, color: css("--warn") },
      text: d.meals.map((m) => `${m.name} · ${Math.round(m.carbs_g)} g`), hovertemplate: "%{text}<extra></extra>" },
  ];
  const end = new Date(now.getTime() + 125 * 60000);
  Plotly.react("chartGlucose", traces, plotLayout({
    yaxis: { title: { text: "mg/dL" }, range: [40, Math.max(320, ...pts.filter(Boolean)) + 10], gridcolor: css("--border"), zeroline: false },
    xaxis: { range: [ts[0], end], gridcolor: css("--border"), zeroline: false },
    shapes: [
      { type: "rect", xref: "paper", x0: 0, x1: 1, y0: 70, y1: 180, fillcolor: css("--range"), line: { width: 0 }, layer: "below" },
      { type: "line", x0: now, x1: now, yref: "paper", y0: 0, y1: 1, line: { color: css("--muted"), width: 1, dash: "dash" } },
      { type: "line", xref: "paper", x0: 0, x1: 1, y0: 180, y1: 180, line: { color: css("--bad"), width: 0.8, dash: "dot" } },
    ],
    annotations: [{ x: now, y: 1, yref: "paper", text: "now", showarrow: false, yanchor: "bottom", font: { size: 10, color: css("--muted") } }],
  }), plotCfg);
}

function renderReasons(d) {
  const r = d.risk;
  const max = Math.max(...r.reasons.map((x) => Math.abs(x.impact)), 0.01);
  const fmtVal = (x) => x.value == null ? "" : Math.abs(x.value) >= 100 ? Math.round(x.value) : Math.round(x.value * 100) / 100;
  const status = r.p_spike >= r.spike_threshold ? `<span class="tier high">ALERT</span>` : `<span class="tier low">no alert</span>`;
  $("#reasons").innerHTML = `
    <div class="alert-head"><span class="big">${pct(r.p_spike)}</span><span class="muted">chance of &gt;180 mg/dL in 2 h</span>${status}</div>
    ${r.reasons.map((x) => {
      const w = Math.abs(x.impact) / max * 50;
      const col = x.impact > 0 ? css("--bad") : css("--accent");
      const pos = x.impact > 0 ? `left:50%;width:${w}%` : `right:50%;width:${w}%`;
      return `<div class="reason"><div class="r-l">${esc(x.label)}<span class="stream-tag">${esc(x.stream)}</span><small>${x.value == null ? "" : `value ${fmtVal(x)} · `}${x.impact > 0 ? "raises" : "lowers"} risk</small></div>
        <div class="r-b"><i style="${pos};background:${col}"></i></div></div>`;
    }).join("")}`;
}

function renderWearables(d) {
  const h = d.history;
  const ts = h.ts.map(parseTs);
  const stageName = ["awake", "light", "deep", "REM"];
  Plotly.react("chartWear", [
    { x: ts, y: h.hr, name: "Heart rate (bpm)", mode: "lines", line: { color: css("--bad"), width: 1.3 }, yaxis: "y" },
    { x: ts, y: h.hrv, name: "HRV RMSSD (ms)", mode: "lines", line: { color: css("--ok"), width: 1.1 }, yaxis: "y" },
    { x: ts, y: h.steps, name: "Steps / 5 min", type: "bar", marker: { color: css("--accent") }, yaxis: "y2" },
    // plotted levels: deep = 1, light = 2, REM = 3; awake = gap
    { x: ts, y: h.sleep_stage.map((s) => ({ 1: 2, 2: 1, 3: 3 }[s] ?? null)), name: "Sleep stage",
      mode: "lines", line: { color: css("--violet"), shape: "hv", width: 1.5 }, yaxis: "y3",
      text: h.sleep_stage.map((s) => stageName[s]), hovertemplate: "%{text}<extra>sleep</extra>" },
  ], plotLayout({
    grid: { rows: 3, columns: 1, pattern: "coupled", roworder: "top to bottom" },
    yaxis: { domain: [0.48, 1], gridcolor: css("--border"), zeroline: false },
    yaxis2: { domain: [0.2, 0.44], gridcolor: css("--border"), zeroline: false },
    yaxis3: { domain: [0, 0.16], tickvals: [1, 2, 3], ticktext: ["deep", "light", "REM"], range: [0.5, 3.5], gridcolor: css("--border") },
    xaxis: { gridcolor: css("--border"), anchor: "y3" },
    bargap: 0,
  }), plotCfg);
}

function renderInsights(d) {
  const ins = d.insights, p = d.patient;
  // twin phenotype
  const t = p.twin;
  const rows = [["Insulin action", t.insulin_action_pct], ["Glucose effectiveness", t.glucose_effectiveness_pct],
    ["Carb absorption speed", t.absorption_speed_pct], ["Exercise response", t.exercise_response_pct]];
  $("#twinPheno").innerHTML = rows.map(([l, v]) => `<div class="pct-row"><span>${l}</span><span class="track"><i style="left:${v}%"></i></span><span class="muted">${ordinal(v)}</span></div>`).join("") +
    `<dl class="kv" style="margin-top:12px"><dt>Twin basal glucose</dt><dd>${t.basal_glucose} mg/dL</dd>
     <dt>Personal calibration</dt><dd>−${t.calibration_gain_pct}% forecast loss vs EHR prior</dd></dl>`;

  // food response
  const fr = ins.food_response;
  const maxR = Math.max(...fr.map((x) => x.rise), 1);
  $("#food").innerHTML = fr.length ? `<table><thead><tr><th>Meal</th><th>n</th><th style="text-align:right">avg rise</th><th></th></tr></thead><tbody>
    ${fr.slice(0, 8).map((x) => `<tr><td>${esc(x.name)}</td><td class="num">${x.n}</td><td class="num">+${x.rise}</td>
      <td style="width:35%"><span class="pill-bar" style="width:${x.rise / maxR * 100}%;background:${x.rise > 80 ? css("--bad") : x.rise > 50 ? css("--warn") : css("--ok")}"></span></td></tr>`).join("")}
    </tbody></table>` : '<p class="muted">Not enough logged meals yet.</p>';

  // sleep
  const n = ins.nights;
  Plotly.react("chartSleep", [
    { x: n.map((x) => x.date), y: n.map((x) => x.sleep_h), type: "bar", name: "Sleep (h)", marker: { color: n.map((x) => (x.sleep_h < 6 ? css("--warn") : css("--violet"))) } },
    { x: n.map((x) => x.date), y: n.map((x) => x.next_day_mean), name: "Next-day mean glucose", mode: "lines+markers", yaxis: "y2", line: { color: css("--bad") } },
  ], plotLayout({ yaxis: { title: { text: "h" }, gridcolor: css("--border"), range: [0, 10] },
    yaxis2: { overlaying: "y", side: "right", title: { text: "mg/dL" }, showgrid: false }, margin: { l: 36, r: 44, t: 8, b: 30 } }), plotCfg);
  $("#sleepNote").textContent = ins.sleep_effect_mgdl > 5 ? `Nights under 6 h were followed by +${ins.sleep_effect_mgdl} mg/dL mean glucose the next day.` : "No consistent sleep effect over the last 7 nights.";

  // EHR
  const l = p.labs, gnt = p.genetics;
  $("#ehr").innerHTML = `<dl class="kv">
    <dt>HbA1c</dt><dd>${l.hba1c}%</dd><dt>Fasting glucose</dt><dd>${l.fpg} mg/dL</dd>
    <dt>BP</dt><dd>${l.bp} mmHg</dd><dt>LDL · eGFR</dt><dd>${l.ldl} mg/dL · ${l.egfr} mL/min</dd>
    <dt>BMI · weight</dt><dd>${p.bmi} · ${p.weight_kg} kg</dd>
    <dt>TCF7L2 rs7903146</dt><dd>${gnt.tcf7l2_risk_alleles} risk allele${gnt.tcf7l2_risk_alleles === 1 ? "" : "s"}</dd>
    <dt>Polygenic risk</dt><dd>z = ${gnt.prs_z} ${gnt.family_history ? "· family history +" : ""}</dd></dl>`;
  Plotly.react("chartA1c", [{ x: p.hba1c_history.map((x) => x.date), y: p.hba1c_history.map((x) => x.hba1c), mode: "lines+markers", name: "HbA1c %", line: { color: css("--accent") } }],
    plotLayout({ showlegend: false, margin: { l: 36, r: 8, t: 14, b: 24 }, yaxis: { title: { text: "HbA1c %" }, gridcolor: css("--border") },
      shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: 7, y1: 7, line: { color: css("--ok"), dash: "dot", width: 1 } }] }), plotCfg);

  // summary
  const s = ins.summary;
  $("#summary").innerHTML = `<div class="summary-text">${esc(s.text)}</div>
    ${s.actions.length ? `<b style="display:block;margin-top:10px">Suggested discussion points</b><ul class="actions">${s.actions.map((a) => `<li>${esc(a)}</li>`).join("")}</ul>` : ""}
    <div class="disclaimer">${esc(s.disclaimer)}</div>`;
}

// ------------------------------------------------------------------ simulator
async function runSim() {
  const meal = $("#simMeal").value, portion = +$("#simPortion").value, offset = +$("#simOffset").value, walk = +$("#simWalk").value;
  const name = state.meta.meals.find((m) => m.key === meal).name;
  const scenarios = [
    { label: "No further meal", meals: [], walks: [] },
    { label: `${name}`, meals: [{ meal_key: meal, portion, offset_min: offset }], walks: [] },
  ];
  if (walk > 0) scenarios.push({ label: `${name} + ${walk}-min walk`, meals: [{ meal_key: meal, portion, offset_min: offset }], walks: [{ offset_min: offset + 15, duration_min: walk, cadence: 100 }] });
  const r = await api(`/api/patients/${state.pid}/whatif`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ at: fmtTs(state.now), horizon_min: 300, scenarios }),
  });
  const colors = [css("--muted"), css("--bad"), css("--ok")];
  const x = r.ts.map(parseTs);
  Plotly.react("chartSim", r.scenarios.map((s, i) => ({ x, y: s.glucose, name: s.label, mode: "lines", line: { color: colors[i], width: i ? 2.5 : 1.5, dash: i ? "solid" : "dot" } })),
    plotLayout({ yaxis: { title: { text: "mg/dL" }, gridcolor: css("--border"), zeroline: false },
      shapes: [{ type: "rect", xref: "paper", x0: 0, x1: 1, y0: 70, y1: 180, fillcolor: css("--range"), line: { width: 0 }, layer: "below" }] }), plotCfg);
  $("#simOut").innerHTML = r.scenarios.slice(1).map((s) => `<div class="s">${esc(s.label)}<b>peak ${s.peak} mg/dL</b>${s.time_above_180_min} min above 180</div>`).join("") +
    (r.scenarios.length === 3 ? `<div class="s">Walk benefit<b style="color:${css("--ok")}">−${r.scenarios[1].peak - r.scenarios[2].peak} mg/dL peak</b>−${r.scenarios[1].time_above_180_min - r.scenarios[2].time_above_180_min} min above 180</div>` : "");
}

async function runRank() {
  const meal = state.meta.meals.find((m) => m.key === $("#simMeal").value);
  const slot = meal.slot === "snack" ? "snack" : meal.slot === "breakfast" ? "breakfast" : "lunch";
  const r = await api(`/api/patients/${state.pid}/meal-ranking?slot=${slot}&at=${encodeURIComponent(fmtTs(state.now))}&offset_min=${$("#simOffset").value}&walk_min=${$("#simWalk").value}`);
  $("#simRankOut").innerHTML = `<table><thead><tr><th>Option (${slot}, region-aware)</th><th style="text-align:right">peak</th><th style="text-align:right">&gt;180</th></tr></thead><tbody>
    ${r.slice(0, 8).map((x) => `<tr><td>${esc(x.name)}</td><td class="num">${x.peak}</td><td class="num">${x.time_above_180_min}′</td></tr>`).join("")}</tbody></table>`;
}

// ------------------------------------------------------------------ clock & playback
function setNow(d) {
  const min = parseTs(state.meta.start).getTime(), max = parseTs(state.meta.end).getTime() - 125 * 60000;
  state.now = new Date(Math.min(Math.max(d.getTime(), min + 24 * 3600000), max));
  $("#now").textContent = fmtHuman(state.now);
}

async function refresh() {
  try { await Promise.all([loadWard(), loadPatient()]); } catch (e) { console.error(e); }
}

function togglePlay() {
  const btn = $("#play");
  if (state.timer) {
    clearInterval(state.timer); state.timer = null;
    btn.textContent = "▶ Replay live stream"; $("#clock").classList.remove("playing");
    return;
  }
  btn.textContent = "⏸ Pause"; $("#clock").classList.add("playing");
  let busy = false;
  state.timer = setInterval(async () => {
    if (busy) return; busy = true;
    setNow(new Date(state.now.getTime() + +$("#speed").value * 60000));
    await refresh(); busy = false;
  }, 1200);
}

async function init() {
  const saved = (() => { try { return localStorage.getItem("gt-theme"); } catch { return null; } })();
  if (saved) document.documentElement.dataset.theme = saved;
  state.meta = await api("/api/meta");
  const mc = state.meta.model_card;
  $("#modelChip").innerHTML = `Test set: RMSE@60 <b>${mc.rmse_60}</b> · spike AUROC <b>${mc.spike_auroc}</b> · lead <b>${Math.round(mc.median_lead_min)} min</b>`;
  setNow(parseTs(state.meta.default_now));
  $("#play").addEventListener("click", togglePlay);
  $("#fwd").addEventListener("click", () => { setNow(new Date(state.now.getTime() + 3600000)); refresh(); });
  $("#back").addEventListener("click", () => { setNow(new Date(state.now.getTime() - 3600000)); refresh(); });
  $("#theme").addEventListener("click", () => {
    const dark = getComputedStyle(document.documentElement).getPropertyValue("--bg").trim() === "#0b1120";
    const next = dark ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("gt-theme", next); } catch {}
    loadPatient(true);
  });
  await loadWard();
}

init().catch((e) => { $("#main").innerHTML = `<div class="empty">Could not load the API: ${esc(e.message)}<br>Run <code>python -m glucotwin.pipeline</code> first.</div>`; });
