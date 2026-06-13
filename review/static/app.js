import * as pdfjsLib from "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/4.4.168/pdf.min.mjs";
pdfjsLib.GlobalWorkerOptions.workerSrc =
  "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/4.4.168/pdf.worker.min.mjs";

// Color per region type (legend + box borders).
const COLORS = {
  title: "#d81b60", section_header: "#8e24aa", paragraph: "#1e88e5",
  list: "#3949ab", caption: "#00897b", table: "#43a047",
  chart: "#fb8c00", figure: "#6d4c41", diagram: "#f4511e",
  logo: "#c0ca33", photo: "#00acc1",
  page_header: "#757575", page_footer: "#757575", footnote: "#9e9e9e",
  formula: "#5e35b1",
};
const colorFor = (t) => COLORS[t] || "#455a64";

const els = {
  jobSelect: document.getElementById("job-select"),
  prev: document.getElementById("prev"),
  next: document.getElementById("next"),
  pageLabel: document.getElementById("page-label"),
  conf: document.getElementById("conf"),
  confVal: document.getElementById("conf-val"),
  status: document.getElementById("status"),
  canvas: document.getElementById("pdf-canvas"),
  overlay: document.getElementById("overlay"),
  legend: document.getElementById("legend"),
  detail: document.getElementById("detail"),
};

const state = {
  doc: null,         // document.json
  pdf: null,         // pdfjs document
  pageIdx: 0,        // index into doc.pages
  hidden: new Set(), // region types toggled off
  minConf: 0,
  selected: null,
};

async function loadJobs() {
  const { jobs } = await (await fetch("/api/jobs")).json();
  els.jobSelect.innerHTML = "";
  if (!jobs.length) {
    els.status.textContent = "No jobs found in store.";
    return;
  }
  for (const id of jobs) {
    const opt = document.createElement("option");
    opt.value = id;
    opt.textContent = id.slice(0, 12) + "…";
    els.jobSelect.appendChild(opt);
  }
  await selectJob(jobs[0]);
}

async function selectJob(id) {
  els.status.textContent = "loading…";
  state.doc = await (await fetch(`/api/jobs/${id}`)).json();
  const buf = await (await fetch(`/api/jobs/${id}/pdf`)).arrayBuffer();
  state.pdf = await pdfjsLib.getDocument({ data: buf }).promise;
  state.pageIdx = 0;
  state.selected = null;
  buildLegend();
  await render();
  els.status.textContent =
    `${state.doc.source.filename} · ${state.doc.pages.length} pages`;
}

function typesPresent() {
  const s = new Set();
  for (const p of state.doc.pages) for (const c of p.chunks) s.add(c.type);
  return [...s].sort();
}

function buildLegend() {
  els.legend.innerHTML = "";
  for (const t of typesPresent()) {
    const chip = document.createElement("span");
    chip.className = "chip" + (state.hidden.has(t) ? " off" : "");
    chip.innerHTML =
      `<span class="swatch" style="background:${colorFor(t)}"></span>${t}`;
    chip.onclick = () => {
      state.hidden.has(t) ? state.hidden.delete(t) : state.hidden.add(t);
      buildLegend();
      drawBoxes();
    };
    els.legend.appendChild(chip);
  }
}

async function render() {
  const page = state.doc.pages[state.pageIdx];
  const pdfPage = await state.pdf.getPage(page.page_index + 1);

  const unscaled = pdfPage.getViewport({ scale: 1 });
  const target = Math.min(950, els.canvas.parentElement.parentElement.clientWidth - 40);
  const scale = target / unscaled.width;
  const viewport = pdfPage.getViewport({ scale });

  els.canvas.width = viewport.width;
  els.canvas.height = viewport.height;
  els.overlay.style.width = viewport.width + "px";
  els.overlay.style.height = viewport.height + "px";

  await pdfPage.render({ canvasContext: els.canvas.getContext("2d"), viewport }).promise;

  els.pageLabel.textContent = `${state.pageIdx + 1} / ${state.doc.pages.length}`;
  els.prev.disabled = state.pageIdx === 0;
  els.next.disabled = state.pageIdx === state.doc.pages.length - 1;
  drawBoxes();
}

function drawBoxes() {
  const page = state.doc.pages[state.pageIdx];
  const W = els.canvas.width, H = els.canvas.height;
  els.overlay.innerHTML = "";
  for (const c of page.chunks) {
    if (state.hidden.has(c.type)) continue;
    if ((c.confidence ?? 1) < state.minConf) continue;
    const b = c.bbox;
    const div = document.createElement("div");
    div.className = "box" + (state.selected === c.id ? " selected" : "");
    div.style.left = b.x0 * W + "px";
    div.style.top = b.y0 * H + "px";
    div.style.width = (b.x1 - b.x0) * W + "px";
    div.style.height = (b.y1 - b.y0) * H + "px";
    div.style.borderColor = colorFor(c.type);
    div.innerHTML = `<span class="tag" style="background:${colorFor(c.type)}">${c.type}</span>`;
    div.onclick = () => { state.selected = c.id; drawBoxes(); showDetail(c); };
    els.overlay.appendChild(div);
  }
}

function confPill(v) {
  const pct = Math.round((v ?? 0) * 100);
  const color = v >= 0.8 ? "#43a047" : v >= 0.5 ? "#fb8c00" : "#e53935";
  return `<span class="conf-pill" style="background:${color}">${pct}%</span>`;
}

function renderContent(c) {
  const ct = c.content || {};
  if ("text" in ct) return document.createTextNode(ct.text || "(empty)").textContent;
  if ("html" in ct) return ct.html;               // table HTML, rendered as-is (local data)
  if ("description" in ct) return ct.description || "(no description)";
  return `<pre>${escapeHtml(JSON.stringify(ct, null, 2))}</pre>`; // chart etc.
}

function escapeHtml(s) {
  return s.replace(/[&<>]/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[ch]));
}

function showDetail(c) {
  els.detail.className = "";
  const body = renderContent(c);
  const isHtml = "html" in (c.content || {});
  const attrs = c.attributes || {};
  const attrStr = Object.keys(attrs).length
    ? Object.entries(attrs).map(([k, v]) => `${k}=${v}`).join(", ")
    : "(none)";
  const boilerplate = c.is_boilerplate
    ? `<div class="kv"><b>boilerplate</b> <span style="color:#e53935;font-weight:600">excluded from chunks</span></div>`
    : "";
  els.detail.innerHTML = `
    <div class="kv"><b>id</b> ${c.id}</div>
    <div class="kv"><b>type</b> <span style="color:${colorFor(c.type)};font-weight:600">${c.type}</span></div>
    <div class="kv"><b>attributes</b> ${escapeHtml(attrStr)}</div>
    ${boilerplate}
    <div class="kv"><b>reading order</b> ${c.reading_order}</div>
    <div class="kv"><b>source</b> ${c.source}</div>
    <div class="kv"><b>confidence</b> ${confPill(c.confidence)}</div>
    <div class="kv"><b>bbox</b> [${c.bbox.x0.toFixed(3)}, ${c.bbox.y0.toFixed(3)}, ${c.bbox.x1.toFixed(3)}, ${c.bbox.y1.toFixed(3)}]</div>
    <h2>content</h2>
    <div class="content-box">${isHtml ? body : escapeHtml(body)}</div>`;
}

els.jobSelect.onchange = (e) => selectJob(e.target.value);
els.prev.onclick = () => { if (state.pageIdx > 0) { state.pageIdx--; state.selected = null; render(); } };
els.next.onclick = () => { if (state.pageIdx < state.doc.pages.length - 1) { state.pageIdx++; state.selected = null; render(); } };
els.conf.oninput = (e) => {
  state.minConf = parseFloat(e.target.value);
  els.confVal.textContent = state.minConf.toFixed(2);
  drawBoxes();
};

loadJobs().catch((err) => { els.status.textContent = "error: " + err.message; });
