const STRATUM_COLOR = {
  financial_table: "#43a047", spec_table: "#2e7d32", landscape_table: "#00897b",
  diagram_figure: "#fb8c00", dense_narrative: "#1e88e5", narrative: "#5c6bc0",
  very_dense: "#8e24aa", near_empty: "#9e9e9e", cover_toc: "#6d4c41",
};

const nav = document.getElementById("nav");
const img = document.getElementById("page-img");
const info = document.getElementById("page-info");
const summary = document.getElementById("summary");
const counts = document.getElementById("counts");

let activeEl = null;
let selections = {};   // gold_id -> "in" | "out"

function show(doc_id, page, label, stratum, gold_id) {
  img.src = `/api/page/${doc_id}/${page}.png`;
  img.style.display = "block";
  info.innerHTML = `<b>${doc_id}</b> &nbsp;·&nbsp; page ${label} &nbsp;·&nbsp; ` +
                   `<span style="color:${STRATUM_COLOR[stratum] || '#555'}">${stratum}</span> ` +
                   `&nbsp;·&nbsp; <code>${gold_id}</code>`;
}

function renderCounts() {
  const vals = Object.values(selections);
  const nin = vals.filter((v) => v === "in").length;
  const nout = vals.filter((v) => v === "out").length;
  counts.innerHTML = `<span class="in">${nin} in</span> · <span class="out">${nout} out</span>`;
}

async function setDecision(gold_id, decision, rowEl) {
  // toggle off if the same button is clicked again
  const next = selections[gold_id] === decision ? "none" : decision;
  await fetch("/api/selections", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ gold_id, decision: next }),
  });
  if (next === "none") delete selections[gold_id];
  else selections[gold_id] = next;
  applyRowState(rowEl, selections[gold_id]);
  renderCounts();
}

function applyRowState(rowEl, state) {
  rowEl.classList.toggle("sel-in", state === "in");
  rowEl.classList.toggle("sel-out", state === "out");
  rowEl.querySelector(".up").classList.toggle("on", state === "in");
  rowEl.querySelector(".down").classList.toggle("on", state === "out");
}

async function load() {
  const [docs, sels] = await Promise.all([
    fetch("/api/candidates").then((r) => r.json()),
    fetch("/api/selections").then((r) => r.json()),
  ]);
  selections = sels || {};
  const total = docs.reduce((n, d) => n + d.pages.length, 0);
  summary.textContent = `${total} candidates · ${docs.length} docs`;
  renderCounts();

  for (const doc of docs) {
    const dh = document.createElement("div");
    dh.className = "doc";
    dh.innerHTML = `${doc.doc_id}<br><span class="src">${doc.source}</span>`;
    nav.appendChild(dh);

    for (const p of doc.pages) {
      const row = document.createElement("div");
      row.className = "page";
      const color = STRATUM_COLOR[p.stratum] || "#7986a3";
      row.innerHTML =
        `<span class="pg">${p.label}</span>` +
        `<span class="stratum" style="background:${color}">${p.stratum}</span>` +
        `<span class="spacer"></span>` +
        `<button class="thumb up" title="add to gold set">👍</button>` +
        `<button class="thumb down" title="exclude">👎</button>`;

      row.onclick = () => {
        if (activeEl) activeEl.classList.remove("active");
        row.classList.add("active");
        activeEl = row;
        show(doc.doc_id, p.page, p.label, p.stratum, p.gold_id);
      };
      row.querySelector(".up").onclick = (e) => { e.stopPropagation(); setDecision(p.gold_id, "in", row); };
      row.querySelector(".down").onclick = (e) => { e.stopPropagation(); setDecision(p.gold_id, "out", row); };

      nav.appendChild(row);
      applyRowState(row, selections[p.gold_id]);
    }
  }
}

load().catch((e) => { info.textContent = "error: " + e.message; });
