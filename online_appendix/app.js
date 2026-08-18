"use strict";

// ---------------------------------------------------------------- config
// Main tabs. "run1" (Opinions) and "messages" share the interactive dashboard
// <main>; the others render into the figures pane.
const APPX_TABS = [
  { id: "run1",      label: "Opinions" },
  { id: "lattices",  label: "Lattices" },
  { id: "messages",  label: "Messages" },
  { id: "questions", label: "Questions" },
  { id: "personas",  label: "Personas" },
];
const DASH_SUBTABS = new Set(["run1", "messages"]);
const dashActive = () => DASH_SUBTABS.has(state.appendixTab);
const messagesActive = () => state.appendixTab === "messages";
// grid views on the Opinions tab
const VIEWS = [
  { id: "heatmap", label: "Group Trajectories" },  // per-agent mean opinion heatmap
  { id: "reps",    label: "Net Opinions" },        // net opinion n(t) per episode + mean
];
// display names for the graph families ("fresh" stays the internal id)
const GROUP_LABEL = { seen: "seen", fresh: "held out", lattice: "lattice" };
// episode selector: episode-average or a single 0-based episode
const REP_OPTS = [
  { id: "avg", label: "Avg" }, { id: "0", label: "E1" }, { id: "1", label: "E2" },
  { id: "2", label: "E3" }, { id: "3", label: "E4" },
];

// index.json plus its extraModels/extraFiles (e.g. Qwen3.5-9B)
const dashIdxCache = new WeakMap();
function dashIndex(idx) {
  if (!idx) return idx;
  if (!dashIdxCache.has(idx)) {
    dashIdxCache.set(idx, {
      ...idx,
      models: idx.models.concat(idx.extraModels || []),
      files: idx.files.concat(idx.extraFiles || []),
    });
  }
  return dashIdxCache.get(idx);
}

// line colors: episode-mean net opinion (bold) and per-episode curves
const COL = { net: "#1f4e9c", netRep: "#a9c0e0", zero: "#cfcfca" };

const state = {
  appendixTab: "run1",
  model: null, regime: "objective", view: "heatmap", rep: "avg",
  appendixData: null,  // question bank + personas (lazy-loaded)
  latModel: null, latDataset: "objective", latGraph: "square", latRep: "avg",
  bankRegime: "objective",  // Questions/Personas: which bank is shown
  index: null,   // data/index.json (+ extras via dashIndex)
  data: {},      // "model__regime" -> payload
  // Messages: per-society caches keyed by "<model>__<regime>[/society]"
  inspect: { qid: null, graph: null, rep: null, agent: 0, idx: {}, soc: {} },
};

// reduce a cell's episode list to the user's selection: all (avg) or one episode
function selectReps(reps) {
  if (state.rep === "avg") return reps;
  const i = +state.rep;
  return reps.map((r, k) => (k === i ? r : null));
}

const $ = (id) => document.getElementById(id);
const dpr = Math.max(1, window.devicePixelRatio || 1);

// shared offscreen canvas for heatmap upscaling
const offc = document.createElement("canvas");
const offx = offc.getContext("2d");

// ------------------------------------------------------------- colormap
// softened RdBu diverging map, v in [-1,1]: blue → neutral → red
const STOPS = [
  [-1.00, [ 56, 120, 184]],
  [-0.75, [ 86, 148, 200]],
  [-0.50, [124, 178, 214]],
  [-0.25, [176, 210, 230]],
  [ 0.00, [247, 247, 247]],
  [ 0.25, [248, 184, 158]],
  [ 0.50, [236, 138, 116]],
  [ 0.75, [216,  92,  84]],
  [ 1.00, [196,  62,  62]],
];
function interpStops(stops, v) {
  v = Math.max(-1, Math.min(1, v));
  for (let i = 1; i < stops.length; i++) {
    if (v <= stops[i][0]) {
      const [v0, c0] = stops[i - 1], [v1, c1] = stops[i];
      const t = (v - v0) / (v1 - v0);
      return [0, 1, 2].map((k) => Math.round(c0[k] + t * (c1[k] - c0[k])));
    }
  }
  return stops[stops.length - 1][1];
}
function cmap(v) { return interpStops(STOPS, v); }

// vertical colorbar (top = +1, bottom = -1) driven by cmap, with -1/0/+1 ticks
function drawColorbar(ctx, x, y, w, h) {
  for (let i = 0; i < h; i++) {
    const v = 1 - 2 * (i / (h - 1));            // top row -> +1, bottom -> -1
    const [r, g, b] = cmap(v);
    ctx.fillStyle = `rgb(${r},${g},${b})`;
    ctx.fillRect(x, y + i, w, 1);
  }
  ctx.strokeStyle = "#d2d2cc"; ctx.lineWidth = 1;
  ctx.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);
  ctx.fillStyle = "#9a9a9a"; ctx.font = "9px -apple-system, sans-serif";
  ctx.textAlign = "left"; ctx.textBaseline = "middle";
  const ticks = [[1, "+1"], [0, "0"], [-1, "−1"]];
  for (const [v, lab] of ticks) {
    const ty = y + (1 - (v + 1) / 2) * (h - 1);
    ctx.fillText(lab, x + w + 3, ty);
  }
}

// --------------------------------------------------------------- loading
async function init() {
  state.index = dashIndex(await fetch("data/index.json").then((r) => r.json()));

  // deep-link: "#<tab>". Legacy forms map onto the current tabs.
  const hash = decodeURIComponent(location.hash.replace(/^#/, ""));
  if (hash) {
    const [tab, sub, view] = hash.split("/");
    if (tab === "online" || (tab === "appendix" && (sub === "run1" || sub === "run2"))) {
      state.appendixTab = (sub === "run1" ? view : sub) === "inspect" ? "messages" : "run1";
    } else if (APPX_TABS.some((a) => a.id === tab)) {
      state.appendixTab = tab;
    } else if (tab === "appendix" && sub && APPX_TABS.some((a) => a.id === sub)) {
      state.appendixTab = sub;
    }
  }

  state.model = state.index.models[0].id;
  buildSeg("appendix-seg", APPX_TABS, () => state.appendixTab, (v) => { state.appendixTab = v; refresh(); });
  buildSeg("view-seg", VIEWS, () => state.view, (v) => { state.view = v; render(); });
  buildSeg("model-seg", state.index.models.map((m) => ({ id: m.id, label: m.label })),
    () => state.model, (v) => { state.model = v; syncRegimeSeg(); refresh(); });
  syncRegimeSeg();
  buildSeg("rep-seg", REP_OPTS, () => state.rep, (v) => { state.rep = v; render(); });

  $("detail-close").onclick = closeDetail;
  $("detail").onclick = (e) => { if (e.target === $("detail")) closeDetail(); };
  $("qmodal-close").onclick = closeQuestion;
  $("qmodal").onclick = (e) => { if (e.target === $("qmodal")) closeQuestion(); };
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") { closeDetail(); closeQuestion(); }
  });

  refresh();
}

function buildSeg(elId, items, getCur, onPick) {
  const el = $(elId);
  if (!el) return;                       // segment not present in this build
  el.innerHTML = "";
  for (const it of items) {
    const b = document.createElement("button");
    b.textContent = it.label;
    b.dataset.id = it.id;
    b.onclick = () => { onPick(it.id); syncSeg(elId, getCur); };
    el.appendChild(b);
  }
  syncSeg(elId, getCur);
}
function syncSeg(elId, getCur) {
  const el = $(elId);
  if (!el) return;
  const cur = getCur();
  for (const b of el.children) b.classList.toggle("on", b.dataset.id === cur);
}

// the regime selector is rebuilt (and state.regime clamped) per model
function regimesFor(idx, model) {
  return idx.regimes.filter((rg) => idx.files.some((f) => f.model === model && f.regime === rg.id));
}
function syncRegimeSeg() {
  const regs = regimesFor(state.index, state.model);
  if (!regs.some((rg) => rg.id === state.regime)) state.regime = regs[0].id;
  buildSeg("regime-seg", regs.map((rg) => ({ id: rg.id, label: rg.label })),
    () => state.regime, (v) => { state.regime = v; refresh(); });
}

async function loadPayload(model, regime) {
  const key = `${model}__${regime}`;
  if (!state.data[key]) {
    const f = state.index.files.find((x) => x.model === model && x.regime === regime);
    if (!f) return null;
    state.data[key] = await (await fetch(`data/${f.file}`)).json();
  }
  return state.data[key];
}

async function refresh() {
  $("status").textContent = "loading…";
  await loadPayload(state.model, state.regime);
  render();
}

// show/hide the per-tab controls and swap the dashboard vs figures <main>
function syncControls() {
  const dash = dashActive();
  syncSeg("appendix-seg", () => state.appendixTab);
  $("view-ctrl").style.display = messagesActive() ? "none" : "";
  syncSeg("view-seg", () => state.view);
  $("dashboard-view").hidden = !dash;
  $("figures-view").hidden = dash;
}

function popMean(flat, A, S, scale) {
  const out = new Array(S).fill(0);
  for (let s = 0; s < S; s++) {
    let acc = 0;
    for (let a = 0; a < A; a++) acc += flat[a * S + s];
    out[s] = acc / A / scale;
  }
  return out;
}

// rep-averaged matrix (flat) over non-null reps; null if all empty
function meanMatrix(reps, A, S) {
  const valid = reps.filter(Boolean);
  if (!valid.length) return null;
  const out = new Array(A * S).fill(0);
  for (const r of valid) for (let i = 0; i < r.length; i++) out[i] += r[i];
  for (let i = 0; i < out.length; i++) out[i] /= valid.length;
  return out;
}

// elementwise mean of equal-length arrays
function avgArrays(arrs) {
  const out = new Array(arrs[0].length).fill(0);
  for (const a of arrs) for (let i = 0; i < a.length; i++) out[i] += a[i];
  for (let i = 0; i < out.length; i++) out[i] /= arrs.length;
  return out;
}

// --------------------------------------------------------------- drawing
function setCanvas(cv, w, h, scale) {
  const s = scale || dpr;
  cv.width = Math.round(w * s); cv.height = Math.round(h * s);
  cv.style.width = w + "px"; cv.style.height = h + "px";
  const ctx = cv.getContext("2d");
  ctx.setTransform(s, 0, 0, s, 0, 0);
  return ctx;
}

function drawHeatmap(ctx, w, h, flat, A, S, scale) {
  offc.width = S; offc.height = A;
  const img = offx.createImageData(S, A);
  for (let a = 0; a < A; a++) for (let s = 0; s < S; s++) {
    const [r, g, b] = cmap(flat[a * S + s] / scale);
    const o = (a * S + s) * 4;
    img.data[o] = r; img.data[o + 1] = g; img.data[o + 2] = b; img.data[o + 3] = 255;
  }
  offx.putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.clearRect(0, 0, w, h);
  ctx.drawImage(offc, 0, 0, w, h);
}

// coupling-matrix thumbnail (grayscale, distinct from the opinion map):
// no edge → near-white, negative → mid gray, positive → near-black
const JSIZE = 52;  // rendered side in CSS px
function jGray(v) { return v > 0 ? 38 : v < 0 ? 150 : 247; }
function drawJMatrix(cv, w, h, flat, n) {
  const ctx = setCanvas(cv, w, h);
  offc.width = n; offc.height = n;
  const img = offx.createImageData(n, n);
  for (let i = 0; i < n * n; i++) {
    const g = jGray(flat[i]), o = i * 4;
    img.data[o] = g; img.data[o + 1] = g; img.data[o + 2] = g; img.data[o + 3] = 255;
  }
  offx.putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.clearRect(0, 0, w, h);
  ctx.drawImage(offc, 0, 0, w, h);
}

function lineY(v, h, pad) { return pad + (1 - (v + 1) / 2) * (h - 2 * pad); }
function lineX(s, S, w, pad) { return pad + (s / (S - 1)) * (w - 2 * pad); }

function plotLine(ctx, arr, S, w, h, pad, color, width, dash) {
  ctx.beginPath();
  ctx.lineWidth = width; ctx.strokeStyle = color;
  ctx.setLineDash(dash || []);
  for (let s = 0; s < S; s++) {
    const x = lineX(s, S, w, pad), y = lineY(arr[s], h, pad);
    s ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  }
  ctx.stroke(); ctx.setLineDash([]);
}

function zeroLine(ctx, w, h, pad) {
  ctx.strokeStyle = COL.zero; ctx.lineWidth = 1;
  ctx.beginPath(); const y0 = lineY(0, h, pad);
  ctx.moveTo(pad, y0); ctx.lineTo(w - pad, y0); ctx.stroke();
}

// net opinion m(t) per episode (faint) + episode-mean (bold)
function drawReps(ctx, w, h, reps, A, S, scale, pad) {
  ctx.clearRect(0, 0, w, h);
  zeroLine(ctx, w, h, pad);
  const valid = reps.filter(Boolean);
  if (!valid.length) return;
  const nets = valid.map((r) => popMean(r, A, S, scale));
  for (const a of nets) plotLine(ctx, a, S, w, h, pad, COL.netRep, 0.8);
  plotLine(ctx, avgArrays(nets), S, w, h, pad, COL.net, 1.8);
}

function paintCell(cv, w, h, reps, d) {
  const ctx = setCanvas(cv, w, h);
  const { n_agents: A, n_steps: S, scale } = d;
  if (state.view === "heatmap") {
    const mm = meanMatrix(reps, A, S);
    if (mm) drawHeatmap(ctx, w, h, mm, A, S, scale);
  } else {
    drawReps(ctx, w, h, reps, A, S, scale, 4);
  }
}

// --------------------------------------------------------------- grid render
function visibleGraphIdx(d) {
  return d.graphs.map((g, i) => i);   // every graph, in payload order
}

function render() {
  syncControls();
  if (!dashActive()) { renderFigures(); return; }
  // Messages swaps its own panel in for the grid
  const inspecting = messagesActive();
  $("grid-panel").hidden = inspecting;
  $("inspect-panel").hidden = !inspecting;
  $("rep-ctrl").style.display = inspecting ? "none" : "";
  $("legend").style.display = inspecting ? "none" : "";
  $("insp-bar").style.display = inspecting ? "" : "none";
  if (inspecting) { renderInspect(); return; }
  const d = state.data[`${state.model}__${state.regime}`];
  if (!d) return;
  const gi = visibleGraphIdx(d);
  const cellW = 80, labelW = 132;
  const cellH = state.view === "heatmap" ? 70 : 52;

  renderLegend($("legend"), state.view, true);

  const grid = $("grid");
  grid.innerHTML = "";
  grid.style.gridTemplateColumns = `${labelW}px repeat(${gi.length}, ${cellW}px)`;

  // header row
  const corner = document.createElement("div");
  corner.className = "corner colhead"; corner.textContent = "question";
  grid.appendChild(corner);
  for (const i of gi) {
    const c = document.createElement("div");
    c.className = `colhead ${d.graph_groups[i]}`;
    c.innerHTML = `<span class="grp">${GROUP_LABEL[d.graph_groups[i]]}</span>${d.graphs[i]}`;
    const J = d.graph_J && d.graph_J[i];        // grayscale coupling-matrix thumbnail
    if (J) {
      const jc = document.createElement("canvas");
      jc.className = "jmat"; jc.title = `coupling matrix J (${d.n_agents}×${d.n_agents})`;
      c.appendChild(jc);
      drawJMatrix(jc, JSIZE, JSIZE, J, d.n_agents);
    }
    grid.appendChild(c);
  }

  // body rows
  d.qids.forEach((qid, qi) => {
    const rh = document.createElement("div");
    const split = d.splits[qi];
    const hasText = !!(d.questions && d.questions[qi]);
    rh.className = "rowhead" + (hasText ? " clickable" : "");
    rh.innerHTML = `<span>${shortQid(qid)}</span><span class="badge ${split}">${split}</span>`;
    if (hasText) {
      rh.title = "Click to read the question";
      rh.onclick = () => openQuestion(qid, qi, d);
    }
    grid.appendChild(rh);

    for (const i of gi) {
      const reps = d.cells[qi][i];
      const shown = selectReps(reps);
      const cell = document.createElement("div");
      const has = shown.some(Boolean);
      cell.className = "cell" + (has ? "" : " empty");
      if (has) {
        const cv = document.createElement("canvas");
        cell.appendChild(cv);
        paintCell(cv, cellW, cellH, shown, d);
        cell.onclick = () => openDetail(qid, d.graphs[i], reps, d);
      }
      grid.appendChild(cell);
    }
  });

  $("status").textContent = statusText(d, gi);
}

function statusText(d, gi) {
  const rep = state.rep === "avg"
    ? "episode-averaged" : `episode ${+state.rep + 1} of ${d.n_reps}`;
  return `${state.model} · ${state.regime} · ${d.qids.length} questions × ${gi.length} graphs · ${rep}`;
}

function figTabBar(items, current, onPick) {
  const bar = document.createElement("div"); bar.className = "fig-tabs";
  for (const it of items) {
    const b = document.createElement("button");
    b.textContent = it.label;
    if (it.id === current) b.classList.add("on");
    b.onclick = () => onPick(it.id);
    bar.appendChild(b);
  }
  return bar;
}
function appendNote(grid, text, wide) {
  const p = document.createElement("p");
  p.className = "desc" + (wide ? " desc-wide" : "");
  p.innerHTML = text;
  grid.appendChild(p);
}
function figW() {
  const wrapW = $("figpane").clientWidth || 900;
  return Math.max(440, Math.min(980, wrapW - 4));
}

// figures pane: Lattices / Questions / Personas
function renderFigures() {
  const pane = $("figpane");
  pane.innerHTML = "";
  const content = document.createElement("div");
  content.className = "fig-content";
  pane.appendChild(content);
  renderAppendix(content);
  const tab = APPX_TABS.find((t) => t.id === state.appendixTab);
  $("status").textContent = tab ? tab.label : "";
}

// Lattices / Questions / Personas (the dashboard tabs never reach here)
function renderAppendix(grid) {
  if (state.appendixTab === "lattices") { renderLattices(grid); return; }

  // questions + personas come from the lazily-loaded appendix payload
  if (!state.appendixData) {
    appendNote(grid, "loading appendix data…");
    fetch(`data/appendix.json?v=${DATA_V}`).then((r) => r.json())
      .then((d) => { state.appendixData = d; render(); });
    return;
  }
  const d = state.appendixData;

  // Dataset selector shared by the Questions and Personas tabs
  latCtrlRow(grid, "Dataset", BANK_REGIMES, state.bankRegime,
    (v) => { state.bankRegime = v; render(); });
  const reg = state.bankRegime, bank = d[reg];
  const cap = reg === "objective" ? "Objective" : "Subjective";

  if (state.appendixTab === "personas") {
    appendPersonaList(grid, `${cap} personas (${bank.personas.length})`,
      bank.personas, reg);
    return;
  }

  appendQuestionTable(grid, `${cap} questions (${bank.questions.length})`,
    bank.questions);
}
const BANK_REGIMES = [
  { id: "objective", label: "Objective" }, { id: "subjective", label: "Subjective" },
];

// Lattices: societies drawn in real space — one row per question, one column
// per timestep; agents at their 4×8 lattice coordinates, colored by mean
// opinion. Square lattice = grid of squares, triangular = honeycomb.
const LAT_ROWS = 4, LAT_COLS = 8;
const LAT_OPTS = [{ id: "square", label: "Square" }, { id: "triangular", label: "Triangular" }];

// pointy-top hexagon path centered at (cx, cy) with circumradius R
function hexPath(ctx, cx, cy, R) {
  ctx.beginPath();
  for (let k = 0; k < 6; k++) {
    const a = Math.PI / 6 + k * Math.PI / 3;
    const x = cx + R * Math.cos(a), y = cy + R * Math.sin(a);
    if (k === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.closePath();
}

function latCtrlRow(grid, label, items, cur, onPick) {
  const row = document.createElement("div");
  row.style.cssText = "display:flex;align-items:baseline;gap:12px;margin:2px 0;";
  const lab = document.createElement("span");
  lab.textContent = label;
  lab.style.cssText =
    "font-size:10px;letter-spacing:.08em;text-transform:uppercase;color:#8a8a84;min-width:58px;";
  row.appendChild(lab);
  const bar = figTabBar(items, cur, onPick);
  bar.style.margin = "0";
  row.appendChild(bar);
  grid.appendChild(row);
}

function renderLattices(grid) {
  const idx = state.index;
  if (!state.latModel || !idx.models.some((m) => m.id === state.latModel))
    state.latModel = idx.models[0].id;
  const dsets = regimesFor(idx, state.latModel);
  if (!dsets.some((r) => r.id === state.latDataset)) state.latDataset = dsets[0].id;

  latCtrlRow(grid, "Model", idx.models.map((m) => ({ id: m.id, label: m.label })),
    state.latModel, (v) => { state.latModel = v; render(); });
  latCtrlRow(grid, "Dataset", dsets.map((r) => ({ id: r.id, label: r.label })),
    state.latDataset, (v) => { state.latDataset = v; render(); });
  latCtrlRow(grid, "Lattice", LAT_OPTS, state.latGraph,
    (v) => { state.latGraph = v; render(); });
  latCtrlRow(grid, "Episode", REP_OPTS, state.latRep,
    (v) => { state.latRep = v; render(); });

  const d = state.data[`${state.latModel}__${state.latDataset}`];
  if (!d) {
    appendNote(grid, "loading lattice data…");
    loadPayload(state.latModel, state.latDataset).then(() => {
      if (state.appendixTab === "lattices") render();
    });
    return;
  }

  const gi = d.graphs.indexOf(state.latGraph);
  if (gi < 0) { appendNote(grid, `no ${state.latGraph} lattice in this dataset`); return; }

  const S = d.n_steps, scale = d.scale, nQ = d.qids.length;
  const tri = state.latGraph === "triangular";
  const unitsW = LAT_COLS + (tri ? (LAT_ROWS - 1) * 0.5 : 0);   // row offset widens rows
  const W = figW();
  const labelW = 128, gap = 8, top = 26, rowGap = 12;
  const cellW = (W - labelW - (S - 1) * gap - 6) / S;
  const unit = cellW / unitsW;
  // honeycomb: hex width √3·R equals the site spacing; row pitch 1.5·R
  const hexR = unit / Math.sqrt(3);
  const subH = tri ? (1.5 * (LAT_ROWS - 1) + 2) * hexR : LAT_ROWS * unit;
  const H = top + nQ * (subH + rowGap);

  const repLabel = state.latRep === "avg"
    ? "episode-averaged" : `episode ${+state.latRep + 1} of ${d.n_reps}`;
  const head = document.createElement("h3");
  head.className = "fig-title";
  head.textContent =
    `${(idx.models.find((m) => m.id === state.latModel) || {}).label} · ` +
    `${(dsets.find((r) => r.id === state.latDataset) || {}).label} · ` +
    `${state.latGraph} lattice · ${repLabel}`;
  grid.appendChild(head);
  const cv = document.createElement("canvas");
  grid.appendChild(cv);
  const ctx = setCanvas(cv, W, H, Math.max(2, dpr));
  ctx.fillStyle = "#ffffff"; ctx.fillRect(0, 0, W, H);

  ctx.fillStyle = "#8a8a84"; ctx.font = "10px system-ui, sans-serif"; ctx.textAlign = "center";
  for (let t = 0; t < S; t++)
    ctx.fillText(`t = ${t}`, labelW + t * (cellW + gap) + cellW / 2, top - 9);

  for (let qi = 0; qi < nQ; qi++) {
    const y0 = top + qi * (subH + rowGap);
    ctx.textAlign = "right"; ctx.font = "10px system-ui, sans-serif";
    ctx.fillStyle = "#44443f";
    ctx.fillText(d.qids[qi].replace(/_/g, " "), labelW - 12, y0 + subH / 2 - 2);
    ctx.fillStyle = "#9a9a94";
    ctx.fillText(d.splits[qi], labelW - 12, y0 + subH / 2 + 9);

    const all = d.cells[qi][gi] || [];
    const reps = (state.latRep === "avg" ? all : [all[+state.latRep]]).filter(Boolean);
    for (let t = 0; t < S; t++) {
      const x0 = labelW + t * (cellW + gap);
      if (!reps.length) {   // cell (or this episode) never collected
        ctx.fillStyle = "#f4f4f1"; ctx.fillRect(x0, y0, cellW, subH);
        continue;
      }
      for (let i = 0; i < d.n_agents; i++) {
        const r = (i / LAT_COLS) | 0, c = i % LAT_COLS;
        let acc = 0;
        for (const f of reps) acc += f[i * S + t];
        const [R, G, B] = cmap(acc / reps.length / scale);
        ctx.fillStyle = `rgb(${R},${G},${B})`;
        if (tri) {
          // row r offset by r/2 sites: touching hexagons = the six J-neighbors
          const cx = x0 + (c + 0.5 * r + 0.5) * unit;
          const cy = y0 + hexR * (1 + 1.5 * r);
          hexPath(ctx, cx, cy, hexR - 0.35);
          ctx.fill();
        } else {
          ctx.fillRect(x0 + c * unit + 0.4, y0 + r * unit + 0.4, unit - 0.8, unit - 0.8);
        }
      }
    }
  }
  ctx.textAlign = "left";

  // footer note under the figure, spanning its full width
  appendNote(grid,
    "The two lattice graphs are the only societies with a literal geometry: 32 agents on a " +
    "4 × 8 grid, each wired to its nearest neighbors — 4 on the square lattice (drawn " +
    "as squares) and 6 on the triangular one, drawn as a honeycomb: each agent is a " +
    "hexagon and its six J-neighbors are exactly the six touching hexagons. Each " +
    "subplot places every agent at its spatial grid position and colors it by its mean " +
    "opinion — blue (−1) → neutral → red (+1), averaged over the 4 episodes or showing a " +
    "single one (Episode selector) — so each column shows the spatial opinion field at one " +
    "timestep, evolving left to right from t = 0 (pre-social) to t = 8. One row per question.",
    true);
}

// question bank table: qid, split, text, answer poles (ground truth flagged)
function appendQuestionTable(grid, title, questions) {
  const head = document.createElement("h3");
  head.className = "fig-title"; head.textContent = title;
  grid.appendChild(head);

  const pct = (k) => `${k}/${questions.length} (${Math.round(100 * k / questions.length)}%)`;
  const cnt = (pred) => questions.filter(pred).length;
  // balance note, appended below the table at the table's full width
  const note = document.createElement("p");
  note.className = "desc desc-wide";

  // class balance of the ground-truth answers over the two poles (objective only)
  if (questions.some((q) => q.answer)) {
    const isPos = (q) => q.answer === q.pos.key;
    const bySplit = (split) =>
      `${cnt((q) => q.split === split && isPos(q))}/${cnt((q) => q.split === split && !isPos(q))}`;
    note.innerHTML = `Ground-truth class balance: +1 ${pct(cnt(isPos))} vs ` +
      `−1 ${pct(cnt((q) => !isPos(q)))} — per split (+/−): ` +
      `train ${bySplit("train")}, test ${bySplit("test")}.`;
  }
  // political balance of the Agree pole (subjective only)
  if (questions.some((q) => q.lean)) {
    const byLean = (l) => cnt((q) => q.lean === l);
    const bySplit = (split) => ["left", "right", "ambiguous"]
      .map((l) => cnt((q) => q.split === split && q.lean === l)).join("/");
    note.innerHTML = `Political balance of the Agree (+1) position, hand-coded: ` +
      `left ${pct(byLean("left"))} vs right ${pct(byLean("right"))}, ` +
      `with ${pct(byLean("ambiguous"))} cross-cutting — per split (left/right/amb.): ` +
      `train ${bySplit("train")}, test ${bySplit("test")}. "Always Agree" is therefore not a ` +
      `coherent ideology: net opinion m measures agreement, not political direction.`;
  }

  const pole = (q, p) => {
    const label = p.text ? `${p.key}: ${escHtml(p.text)}` : escHtml(p.key);
    return q.answer === p.key ? `<b>${label}</b> <span class="pole-flag">✓</span>` : label;
  };
  const hasLean = questions.some((q) => q.lean);
  const leanCell = (q) => q.lean === "ambiguous"
    ? `<td class="lean-amb">—</td>` : `<td class="lean-${q.lean}">${q.lean}</td>`;
  const tbl = document.createElement("table");
  tbl.className = "cat-table q-table";
  tbl.innerHTML = "<tr><th>#</th><th>qid</th><th>split</th><th>question</th>" +
    "<th>+1</th><th>−1</th>" + (hasLean ? "<th>Agree leans</th>" : "") + "</tr>" +
    questions.map((q, i) =>
      `<tr><td>${i + 1}</td><td>${escHtml(q.qid)}</td><td>${q.split}</td>` +
      `<td class="q-text">${escHtml(q.text)}</td>` +
      `<td>${pole(q, q.pos)}</td><td>${pole(q, q.neg)}</td>` +
      (hasLean ? leanCell(q) : "") + `</tr>`).join("");
  grid.appendChild(tbl);
  if (note.innerHTML) grid.appendChild(note);
}

// personas as a collapsible list with a short per-agent summary tag
function appendPersonaList(grid, title, personas, regime) {
  const head = document.createElement("h3");
  head.className = "fig-title"; head.textContent = title;
  grid.appendChild(head);

  const tag = (p) => {
    if (regime === "objective") {
      const m = p.match(/Example problem \(([^)]+)\)/);
      return m ? m[1] : "";
    }
    return p.slice(0, 96).trimEnd() + "…";
  };
  const list = document.createElement("div");
  list.className = "pers-list";
  list.innerHTML = personas.map((p, i) =>
    `<details class="insp-persona"><summary>Agent ${i}` +
    `<span class="pers-tag">${escHtml(tag(p))}</span></summary>` +
    `<div>${escHtml(p)}</div></details>`).join("");
  grid.appendChild(list);
}

// cache-buster for data/appendix.json
const DATA_V = "111";

function shortQid(q) { return q.replace(/_/g, " "); }

// --------------------------------------------------------------- legend
function renderLegend(el, view, compact) {
  const items = {
    heatmap: [
      `<span class="item cbar"><span class="cbar-cap">opinion</span><span class="cbar-lab">−1</span><span class="grad grad-lo"></span><span class="cbar-lab">0</span><span class="grad grad-hi"></span><span class="cbar-lab">+1</span></span>`,
    ],
    reps: [
      sw(COL.net, "n(t) (episode-mean)"),
      sw(COL.netRep, "n(t) per episode"),
    ],
  }[view];
  el.innerHTML = items.join("");
}
function sw(color, label, dash) {
  return `<span class="item" style="color:${color}"><span class="swatch ${dash ? "dash" : ""}" style="background:${dash ? "" : color}"></span><span style="color:var(--muted)">${label}</span></span>`;
}

// --------------------------------------------------------------- detail
function openDetail(qid, graph, reps, d) {
  $("detail-title").textContent = `${shortQid(qid)} · ${graph} · ${state.regime} · ${state.model}`;
  const body = $("detail-body");
  body.innerHTML = "";
  const { n_agents: A, n_steps: S, scale } = d;
  if (state.view === "heatmap") {
    reps.forEach((r, k) => {
      const wrap = document.createElement("div");
      if (!r) { wrap.className = "rep empty"; wrap.textContent = `rep ${k} · n/a`; body.appendChild(wrap); return; }
      wrap.className = "rep";
      const cap = document.createElement("h4");
      cap.textContent = `episode ${k}`;
      wrap.appendChild(cap);
      const cv = document.createElement("canvas");
      wrap.appendChild(cv);
      const W = 190, H = 150;
      const ctx = setCanvas(cv, W, H);
      drawDetail(ctx, W, H, r, A, S, scale);
      body.appendChild(wrap);
    });
  } else {
    const wrap = document.createElement("div");
    wrap.className = "rep wide";
    const cv = document.createElement("canvas");
    wrap.appendChild(cv);
    const W = 540, H = 300;
    const ctx = setCanvas(cv, W, H);
    drawDetailBig(ctx, W, H, reps, A, S, scale);
    body.appendChild(wrap);
  }
  renderLegend($("detail-legend"), state.view, false);
  $("detail").classList.remove("hidden");
}

// large single-panel detail: net opinion n(t) per episode + mean
function drawDetailBig(ctx, w, h, reps, A, S, scale) {
  const pad = 30;
  axes(ctx, w, h, pad);
  const valid = reps.filter(Boolean);
  if (!valid.length) return;
  const nets = valid.map((r) => popMean(r, A, S, scale));
  for (const a of nets) plotLineP(ctx, a, S, w, h, pad, COL.netRep, 1.0);
  plotLineP(ctx, avgArrays(nets), S, w, h, pad, COL.net, 2.2);
}

function closeDetail() { $("detail").classList.add("hidden"); }

// --------------------------------------------------------------- question text
function openQuestion(qid, qi, d) {
  const split = d.splits[qi];
  $("qmodal-title").innerHTML =
    `${shortQid(qid)} <span class="badge ${split}">${split}</span>`;
  $("qmodal-text").textContent = d.questions[qi] || "(question text unavailable)";
  $("qmodal-poles").innerHTML = polesHtml(d.poles && d.poles[qi]);
  $("qmodal").classList.remove("hidden");
}

// the two opinion poles (+1 / -1) with their swatch, choice and correct flag
function polesHtml(p) {
  if (!p || !p.pos || !p.neg) return "";
  const row = (v, pole) => {
    const [r, g, b] = cmap(v);
    const correct = p.answer && pole.key === p.answer;
    const text = pole.text ? `: ${pole.text}` : "";
    return `<div class="pole${correct ? " correct" : ""}">
      <span class="pole-sw" style="background:rgb(${r},${g},${b})"></span>
      <span class="pole-val">${v > 0 ? "+1" : "−1"}</span>
      <span class="pole-key">${pole.key}${text}</span>
      ${correct ? `<span class="pole-flag">✓ correct</span>` : ""}
    </div>`;
  };
  return row(1, p.pos) + row(-1, p.neg);
}
function closeQuestion() { $("qmodal").classList.add("hidden"); }

// ===================================================================
// Messages tab — who said what to whom, round by round. Societies are
// fetched from the repo's raw runs in data/models/<model>/<regime>_energy/
// (one level up, so the site must be served from the repository root).
// ===================================================================

const insp = state.inspect;
const MODELS_BASE = "../data/models";
function inspMR() {
  return `${state.model}__${state.regime}`;
}

function escHtml(s) {
  return String(s).replace(/[&<>"]/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

// data/models path pieces for a "model__regime" id
function modelsDir(mr) {
  const i = mr.lastIndexOf("__");
  const model = mr.slice(0, i), regime = mr.slice(i + 2);
  return { model, regime, dir: `${MODELS_BASE}/${model}/${regime}_energy` };
}

// fetch JSON, preferring the gzipped deploy artifact (<path>.gz, decompressed
// in the browser) and falling back to the plain file (local raw runs)
async function fetchJsonGz(path) {
  try {
    const r = await fetch(`${path}.gz`);
    if (r.ok && typeof DecompressionStream !== "undefined") {
      const body = r.body.pipeThrough(new DecompressionStream("gzip"));
      return JSON.parse(await new Response(body).text());
    }
  } catch { /* fall through to the plain file */ }
  const r = await fetch(path);
  return r.ok ? r.json() : null;
}

// question bank (choices + correct answer) from data/obj|subj
async function loadQuestionBank(regime) {
  const base = regime === "objective" ? "../data/obj" : "../data/subj";
  const bank = {};
  for (const split of ["train", "test"]) {
    try {
      const r = await fetch(`${base}/${split}.jsonl`);
      if (!r.ok) continue;
      for (const line of (await r.text()).split("\n")) {
        if (!line.trim()) continue;
        const d = JSON.parse(line);
        bank[d.qid] = { choices: d.choices || {}, answer: d.answer || null };
      }
    } catch { /* keep whatever splits did load */ }
  }
  return bank;
}

// graph families: J* seen in training, Jf* held-out fresh, rest lattices
function graphGroup(id) {
  return /^Jf/.test(id) ? "fresh" : /^J\d/.test(id) ? "seen" : "lattice";
}
function numTail(s) { const m = s.match(/(\d+)$/); return m ? +m[1] : -1; }

// build (and cache) the per-(model,regime) index of available societies
// from the run directory's manifest.json plus the question bank
async function loadInspectIndex(mr) {
  const key = mr;
  if (insp.idx[key] === undefined) {
    try {
      const { model, regime, dir } = modelsDir(mr);
      const man = await fetchJsonGz(`${dir}/manifest.json`);
      if (!man) { insp.idx[key] = null; return null; }
      // deploy ships the (per-directory) personas once; local raw runs carry
      // them in every society file instead — lifted off the first read
      const pers = await fetch(`${dir}/personas.json`)
        .then((r) => (r.ok ? r.json() : null)).catch(() => null);
      const bank = await loadQuestionBank(regime);
      const questions = {}, runs = [], graphSet = new Set();
      for (const rep of Object.values(man.replicas || {})) {
        runs.push({ qid: rep.qid, graph: rep.graph_id, rep: rep.repeat });
        graphSet.add(rep.graph_id);
        if (!questions[rep.qid]) {
          questions[rep.qid] = Object.assign(
            { statement: rep.question, split: rep.split }, bank[rep.qid] || {});
        }
      }
      const rank = { seen: 0, fresh: 1, lattice: 2 };
      const graphs = [...graphSet]
        .sort((a, b) => (rank[graphGroup(a)] - rank[graphGroup(b)]) ||
          (numTail(a) - numTail(b)) || a.localeCompare(b))
        .map((id) => ({ id, group: graphGroup(id) }));
      const qids = Object.keys(questions).sort((a, b) =>
        ((questions[a].split !== "train") - (questions[b].split !== "train")) ||
        (numTail(a) - numTail(b)) || a.localeCompare(b));
      insp.idx[key] = {
        model, regime,
        n_agents: (man.meta || {}).num_agents,
        personas: pers || [],
        qids, questions, graphs, runs,
      };
    } catch { insp.idx[key] = null; }
  }
  return insp.idx[key];
}

// fetch (and cache) one society's raw run file, reshaped for the view
async function loadSociety(mr, qid, graph, rep) {
  const file = `${qid}__${graph}__rep${String(rep).padStart(2, "0")}.json`;
  const key = `${mr}/${file}`;
  if (insp.soc[key] === undefined) {
    try {
      const raw = await fetchJsonGz(`${modelsDir(mr).dir}/${file}`);
      if (!raw) { insp.soc[key] = null; return null; }
      const spins = raw.spins_history || [];
      insp.soc[key] = {
        n_agents: spins.length ? spins[0].length : 0,
        n_steps: spins.length,
        adj: (raw.J || []).map((row) =>
          row.flatMap((v, j) => (v ? [[j, v]] : []))),
        spins,
        means: (raw.spins_raw_history || []).map((step) =>
          step.map((ss) => ss.reduce((s, v) => s + v, 0) / ss.length)),
        messages: raw.messages_history || [],
      };
      const idx = insp.idx[mr];
      if (idx && !idx.personas.length) idx.personas = raw.personas || [];
    } catch { insp.soc[key] = null; }
  }
  return insp.soc[key];
}

// graphs / reps available for a given qid, from the index's run list
function graphsForQid(idx, qid) {
  const set = new Set(idx.runs.filter((r) => r.qid === qid).map((r) => r.graph));
  return idx.graphs.filter((g) => set.has(g.id));
}
function repsFor(idx, qid, graph) {
  return idx.runs.filter((r) => r.qid === qid && r.graph === graph)
    .map((r) => r.rep).sort((a, b) => a - b);
}

async function renderInspect() {
  const mr = inspMR();
  const idx = await loadInspectIndex(mr);
  // a later async hop may have changed the tab/selection; bail if so
  if (!messagesActive() || mr !== inspMR()) return;

  const bar = $("insp-bar");
  if (!idx) {
    bar.innerHTML = "";
    $("insp-question").innerHTML = "";
    $("insp-agent-grid").innerHTML = "";
    $("insp-agent-legend").innerHTML = "";
    $("insp-focus").innerHTML =
      `<div class="insp-empty">No inspect data for <b>${escHtml(mr)}</b>.<br>` +
      `Expected raw runs at <code>data/models/${escHtml(mr).replace("__", "/")}_energy/</code> — ` +
      `serve the site from the repository root.</div>`;
    $("insp-status").textContent = "";
    $("status").textContent = "Messages — no data";
    return;
  }

  // clamp selections to what exists in this index
  if (!idx.qids.includes(insp.qid)) insp.qid = idx.qids[0];
  let graphs = graphsForQid(idx, insp.qid);
  if (!graphs.some((g) => g.id === insp.graph)) insp.graph = graphs[0].id;
  let reps = repsFor(idx, insp.qid, insp.graph);
  if (!reps.includes(insp.rep)) insp.rep = reps[0];

  // ---- selector bar (question / graph / episode) ----
  const qOpts = idx.qids.map((q) => {
    const sp = (idx.questions[q] || {}).split || "";
    return `<option value="${escHtml(q)}"${q === insp.qid ? " selected" : ""}>${escHtml(shortQid(q))}${sp ? ` · ${sp}` : ""}</option>`;
  }).join("");
  const gOpts = graphs.map((g) =>
    `<option value="${escHtml(g.id)}"${g.id === insp.graph ? " selected" : ""}>${escHtml(g.id)} · ${escHtml(GROUP_LABEL[g.group] || g.group)}</option>`).join("");
  const rOpts = reps.map((r) =>
    `<option value="${r}"${r === insp.rep ? " selected" : ""}>ep ${r + 1}</option>`).join("");
  bar.innerHTML =
    `<span class="insp-field"><label>Question</label><select id="insp-q">${qOpts}</select></span>` +
    `<span class="insp-field"><label>Graph</label><select id="insp-g">${gOpts}</select></span>` +
    `<span class="insp-field"><label>Episode</label><select id="insp-r">${rOpts}</select></span>`;
  $("insp-q").onchange = (e) => { insp.qid = e.target.value; insp.graph = null; insp.rep = null; render(); };
  $("insp-g").onchange = (e) => { insp.graph = e.target.value; insp.rep = null; render(); };
  $("insp-r").onchange = (e) => { insp.rep = +e.target.value; render(); };

  // ---- question banner ----
  const q = idx.questions[insp.qid] || {};
  $("insp-question").innerHTML =
    `<div class="insp-qhead">${escHtml(shortQid(insp.qid))}` +
    (q.split ? ` <span class="badge ${q.split}">${q.split}</span>` : "") + `</div>` +
    `<div class="insp-qtext">${escHtml(q.statement || "(question text unavailable)")}</div>` +
    inspectPoles(q);

  // ---- load the society, then paint agents + timeline ----
  const soc = await loadSociety(mr, insp.qid, insp.graph, insp.rep);
  if (!messagesActive() || mr !== inspMR()) return;
  if (!soc) {
    $("insp-agent-grid").innerHTML = "";
    $("insp-focus").innerHTML = `<div class="insp-empty">Society file missing.</div>`;
    $("status").textContent = "Messages — society file missing";
    return;
  }
  if (insp.agent >= soc.n_agents) insp.agent = 0;
  renderAgentGrid(idx, soc);
  renderFocusAgent(idx, soc);
  $("insp-status").textContent =
    `${state.model} · ${state.regime} · ${shortQid(insp.qid)} · graph ${insp.graph} · episode ${insp.rep + 1} · ` +
    `${soc.n_agents} agents · ${soc.n_steps} steps · ${soc.messages.length} message rounds`;
  $("status").textContent = "Messages";  // clear the footer "loading…"
}

// the two opinion poles (+1 = A, −1 = B) with choice text and correctness flag
function inspectPoles(q) {
  const ch = q.choices || {};
  if (!ch.A && !ch.B) return "";
  const row = (v, key) => {
    const [r, g, b] = cmap(v);
    const correct = q.answer && key === q.answer;
    const txt = ch[key] ? `: ${escHtml(ch[key])}` : "";
    return `<div class="pole${correct ? " correct" : ""}">
      <span class="pole-sw" style="background:rgb(${r},${g},${b})"></span>
      <span class="pole-val">${v > 0 ? "+1" : "−1"}</span>
      <span class="pole-key">${key}${txt}</span>
      ${correct ? `<span class="pole-flag">✓ correct</span>` : ""}</div>`;
  };
  return `<div class="poles">${row(1, "A")}${row(-1, "B")}</div>`;
}

// 32-agent map (colored by final opinion); click to focus
function renderAgentGrid(idx, soc) {
  const last = soc.n_steps - 1;
  const grid = $("insp-agent-grid");
  grid.innerHTML = "";
  for (let a = 0; a < soc.n_agents; a++) {
    const mean = soc.means[last][a];
    const [r, g, b] = cmap(mean);
    const cell = document.createElement("button");
    cell.className = "insp-agent" + (a === insp.agent ? " on" : "");
    cell.style.background = `rgb(${r},${g},${b})`;
    cell.style.color = Math.abs(mean) > 0.55 ? "#fff" : "#1a1a1a";
    cell.textContent = a;
    cell.title = `agent ${a} · final m=${mean.toFixed(2)}`;
    cell.onclick = () => { insp.agent = a; renderAgentGrid(idx, soc); renderFocusAgent(idx, soc); };
    grid.appendChild(cell);
  }
  $("insp-agents-sub").textContent = `· colored by final opinion`;
  $("insp-agent-legend").innerHTML =
    `<span class="item cbar"><span class="cbar-lab">B −1</span>` +
    `<span class="grad grad-lo"></span><span class="cbar-lab">0</span>` +
    `<span class="grad grad-hi"></span><span class="cbar-lab">+1 A</span></span>`;
}

// a small colored opinion chip: stance (A/B), thermal mean, discrete spin
function opChip(mean, spin) {
  const [r, g, b] = cmap(mean);
  const choice = spin > 0 ? "A" : "B";
  const dark = Math.abs(mean) > 0.55;
  return `<span class="op-chip" style="background:rgb(${r},${g},${b});color:${dark ? "#fff" : "#1a1a1a"}">` +
    `${spin > 0 ? "+1" : "−1"} · ${choice} <span class="op-m">m=${mean.toFixed(2)}</span></span>`;
}

// focused agent: persona, sparkline, round-by-round message timeline
function renderFocusAgent(idx, soc) {
  const a = insp.agent;
  const S = soc.n_steps, rounds = soc.messages.length;
  const persona = (idx.personas || [])[a] || "";
  const adj = soc.adj[a] || [];
  const nbrs = adj.length;
  // coupling sign J[a][j] from the focus agent to each neighbor j
  const signOf = {};
  for (const [j, s] of adj) signOf[j] = s;
  const nPos = adj.filter(([, s]) => s > 0).length;
  const nNeg = adj.filter(([, s]) => s < 0).length;
  const out = [];

  // header: persona + trajectory sparkline
  out.push(`<div class="insp-focus-head">`);
  out.push(`<div class="insp-focus-title">Agent ${a} <span class="insp-sub">· ${nbrs} neighbors (` +
    `<b class="cpl-pos">+${nPos}</b> / <b class="cpl-neg">−${nNeg}</b>)</span></div>`);
  let spark = "";
  for (let s = 0; s < S; s++) {
    const [r, g, b] = cmap(soc.means[s][a]);
    spark += `<span class="spark-cell" title="step ${s} · m=${soc.means[s][a].toFixed(2)}" style="background:rgb(${r},${g},${b})"></span>`;
  }
  out.push(`<div class="insp-spark"><span class="insp-sub">trajectory</span>${spark}</div>`);
  out.push(`</div>`);
  if (persona) out.push(`<details class="insp-persona"><summary>Persona</summary><div>${escHtml(persona)}</div></details>`);

  // initial position
  out.push(`<div class="insp-round">`);
  out.push(`<div class="insp-step-line"><span class="insp-step-label">Initial position (step 0)</span>${opChip(soc.means[0][a], soc.spins[0][a])}</div>`);
  out.push(`</div>`);

  // one block per message round
  for (let t = 0; t < rounds; t++) {
    const msgs = soc.messages[t] || {};
    const received = [], sentByText = new Map();
    for (const k in msgs) {
      const [src, dst] = k.split(",").map(Number);
      if (dst === a) received.push({ src, text: msgs[k] });
      if (src === a) {
        const tx = msgs[k];
        if (!sentByText.has(tx)) sentByText.set(tx, []);
        sentByText.get(tx).push(dst);
      }
    }
    received.sort((x, y) => x.src - y.src);

    out.push(`<div class="insp-round">`);
    out.push(`<div class="insp-round-head">Round ${t + 1} <span class="insp-sub">· step ${t} → ${t + 1}</span></div>`);

    // received, split into two inboxes by the coupling sign J[a][src]
    const msgCard = (m) =>
      `<div class="msg in">` +
      `<div class="msg-from">from <b>${m.src}</b> ${opChip(soc.means[t][m.src], soc.spins[t][m.src])}</div>` +
      `<div class="msg-text">${escHtml(m.text)}</div></div>`;
    const inbox = (cls, label, list) =>
      `<div class="insp-inbox ${cls}">` +
      `<div class="insp-inbox-cap">${label} <span class="insp-sub">${list.length}</span></div>` +
      (list.length ? list.map(msgCard).join("") : `<div class="insp-none">— none —</div>`) +
      `</div>`;
    const pos = received.filter((m) => signOf[m.src] > 0);
    const neg = received.filter((m) => signOf[m.src] < 0);
    const other = received.filter((m) => !signOf[m.src]);

    out.push(`<div class="insp-msgs">`);
    out.push(`<div class="insp-msgs-cap">Received <span class="insp-sub">${received.length}</span></div>`);
    out.push(`<div class="insp-inboxes">`);
    out.push(inbox("pos", "<b>+1</b> coupled", pos));
    out.push(inbox("neg", "<b>−1</b> coupled", neg));
    out.push(`</div>`);
    if (other.length) out.push(inbox("other", "uncoupled", other));
    out.push(`</div>`);

    // sent
    const sentCount = [...sentByText.values()].reduce((n, v) => n + v.length, 0);
    out.push(`<div class="insp-msgs">`);
    out.push(`<div class="insp-msgs-cap">Sent <span class="insp-sub">${sentCount}</span></div>`);
    if (!sentCount) out.push(`<div class="insp-none">— no outgoing messages —</div>`);
    for (const [text, dsts] of sentByText) {
      out.push(`<div class="msg out">` +
        `<div class="msg-from">to <b>${dsts.sort((x, y) => x - y).join(", ")}</b></div>` +
        `<div class="msg-text">${escHtml(text)}</div></div>`);
    }
    out.push(`</div>`);

    // resulting opinion (step t+1), flagged if the discrete stance flipped
    const flipped = soc.spins[t + 1][a] !== soc.spins[t][a];
    out.push(`<div class="insp-step-line after">` +
      `<span class="insp-step-label">After round ${t + 1} (step ${t + 1})</span>` +
      `${opChip(soc.means[t + 1][a], soc.spins[t + 1][a])}` +
      (flipped ? `<span class="insp-flip">⟲ stance flipped</span>` : "") + `</div>`);
    out.push(`</div>`);
  }

  $("insp-focus").innerHTML = out.join("");
}

// enlarged single-episode heatmap (with axis labels + colorbar) for the detail modal
function drawDetail(ctx, w, h, rep, A, S, scale) {
  const pad = 22;
  ctx.clearRect(0, 0, w, h);
  const cbW = 9, cbGap = 8, cbLab = 18;         // colorbar strip + gap + tick labels
  const x0 = pad, y0 = 8, pw = w - pad - 8 - cbW - cbGap - cbLab, ph = h - 28;
  ctx.save();
  ctx.translate(x0, y0);
  drawHeatmap(ctx, pw, ph, rep, A, S, scale);
  ctx.restore();
  ctx.strokeStyle = "#d2d2cc"; ctx.lineWidth = 1;
  ctx.strokeRect(x0 + 0.5, y0 + 0.5, pw - 1, ph - 1);
  ctx.fillStyle = "#9a9a9a"; ctx.font = "9px -apple-system, sans-serif";
  ctx.textAlign = "center"; ctx.textBaseline = "top";
  ctx.fillText("timestep →", x0 + pw / 2, y0 + ph + 6);
  ctx.save(); ctx.translate(x0 - 7, y0 + ph / 2); ctx.rotate(-Math.PI / 2);
  ctx.textBaseline = "alphabetic"; ctx.fillText("32 agents", 0, 0); ctx.restore();
  drawColorbar(ctx, x0 + pw + cbGap, y0, cbW, ph);
}

// axes helpers for detail (margin pad on left/bottom)
function mapY(v, h, pad) { return pad / 2 + (1 - (v + 1) / 2) * (h - pad - 6); }
function mapX(s, S, w, pad) { return pad + (s / (S - 1)) * (w - pad - 6); }
function axes(ctx, w, h, pad) {
  ctx.clearRect(0, 0, w, h);
  ctx.strokeStyle = "#cfcfca"; ctx.lineWidth = 1; ctx.fillStyle = "#9a9a9a";
  ctx.font = "9px -apple-system, sans-serif"; ctx.textAlign = "right"; ctx.textBaseline = "middle";
  for (const v of [-1, 0, 1]) {
    const y = mapY(v, h, pad);
    ctx.fillText(v.toFixed(0), pad - 4, y);
    ctx.globalAlpha = v === 0 ? 0.7 : 0.3;
    ctx.beginPath(); ctx.moveTo(pad, y); ctx.lineTo(w - 6, y); ctx.stroke();
    ctx.globalAlpha = 1;
  }
  ctx.textAlign = "center"; ctx.textBaseline = "top";
  ctx.fillText("t", (w + pad) / 2, h - 11);
}
function plotLineP(ctx, arr, S, w, h, pad, color, width, dash) {
  ctx.beginPath(); ctx.lineWidth = width; ctx.strokeStyle = color; ctx.setLineDash(dash || []);
  for (let s = 0; s < S; s++) {
    const x = mapX(s, S, w, pad), y = mapY(arr[s], h, pad);
    s ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  }
  ctx.stroke(); ctx.setLineDash([]);
}

init().catch((e) => { $("status").textContent = "error: " + e.message; console.error(e); });

