// Inférence du MLP exporté par train.py : standardisation puis 3 couches denses.

const FR = {
  "radius": "Rayon", "texture": "Texture", "perimeter": "Périmètre", "area": "Aire",
  "smoothness": "Lissité", "compactness": "Compacité", "concavity": "Concavité",
  "concave points": "Points concaves", "symmetry": "Symétrie", "fractal dimension": "Dimension fractale",
};
const GROUPS = [
  { title: "Moyenne", from: 0 },
  { title: "Erreur standard", from: 10 },
  { title: "Pire valeur", from: 20 },
];

function predict(model, x) {
  let h = x.map((v, i) => (v - model.scaler.mean[i]) / model.scaler.scale[i]);
  for (const layer of model.layers) {
    h = layer.W.map((row, j) => row.reduce((s, w, i) => s + w * h[i], layer.b[j]));
    h = layer.activation === "relu" ? h.map(v => Math.max(0, v)) : h.map(v => 1 / (1 + Math.exp(-v)));
  }
  return h[0]; // probabilité que la tumeur soit maligne
}

function baseName(name) {
  return name.replace(/^(mean|worst) /, "").replace(/ error$/, "");
}

if (typeof module !== "undefined") module.exports = { predict };

if (typeof document !== "undefined") {
  const $ = id => document.getElementById(id);
  const fmt = v => (Math.round(v * 10000) / 10000).toString();
  const pct = v => `${(v * 100).toFixed(1).replace(".", ",")} %`;
  const RING = 2 * Math.PI * 84;
  let model, rows = [], current = null, activeTab = 0, values = [];

  fetch("model.json").then(r => r.json()).then(m => {
    model = m;
    model.test_set.forEach(p => { p.prob = predict(model, p.x); });
    buildTabs();
    buildFeatures();
    renderKpis();
    renderConfusion();
    renderStrip();
    loadPatient(pick(model.test_set.filter(p => p.label === 1)));
  });

  // ---------- Tooltip ----------
  const tip = $("tooltip");
  function showTip(e, html) {
    tip.innerHTML = html;
    tip.classList.add("show");
    tip.style.left = `${Math.min(e.clientX + 14, innerWidth - 270)}px`;
    tip.style.top = `${e.clientY + 14}px`;
  }
  const hideTip = () => tip.classList.remove("show");

  // ---------- Saisie des caractéristiques ----------
  function buildTabs() {
    GROUPS.forEach((g, k) => {
      const b = document.createElement("button");
      b.className = "tab" + (k === 0 ? " active" : "");
      b.textContent = g.title;
      b.onclick = () => {
        activeTab = k;
        document.querySelectorAll(".tab").forEach((t, j) => t.classList.toggle("active", j === k));
        buildFeatures();
      };
      $("tabs").appendChild(b);
    });
  }

  function buildFeatures() {
    const box = $("features");
    box.innerHTML = "";
    rows = [];
    const from = GROUPS[activeTab].from;
    for (let i = from; i < from + 10; i++) {
      const row = document.createElement("div");
      row.className = "feat";
      row.innerHTML = `
        <span class="feat-name" title="${model.feature_names[i]}">${FR[baseName(model.feature_names[i])]}</span>
        <div class="zbar"><div class="zbar-fill"></div></div>
        <input type="number" step="any">`;
      const input = row.querySelector("input");
      input.value = values[i] !== undefined ? fmt(values[i]) : "";
      input.addEventListener("input", () => {
        values[i] = parseFloat(input.value);
        current = null;
        update();
      });
      const bar = row.querySelector(".zbar");
      bar.addEventListener("mousemove", e => showTip(e, `z = ${zscore(i).toFixed(2)} écart-type(s)<br><span style="color:#a9b4c8">moyenne train : ${fmt(model.scaler.mean[i])}</span>`));
      bar.addEventListener("mouseleave", hideTip);
      rows[i] = { input, fill: row.querySelector(".zbar-fill") };
      box.appendChild(row);
    }
    updateBars();
  }

  function zscore(i) { return (values[i] - model.scaler.mean[i]) / model.scaler.scale[i]; }

  function updateBars() {
    for (const i in rows) {
      const z = Math.max(-3, Math.min(3, zscore(+i) || 0));
      const w = Math.abs(z) / 3 * 50;
      rows[i].fill.style.left = `${z < 0 ? 50 - w : 50}%`;
      rows[i].fill.style.width = `${w}%`;
    }
  }

  function pick(list) { return list[Math.floor(Math.random() * list.length)]; }

  function loadPatient(p) {
    current = p;
    values = p.x.slice();
    for (const i in rows) rows[i].input.value = fmt(values[i]);
    $("source").textContent = `Patient n°${p.index} · jeu de test`;
    update();
  }

  // ---------- Prédiction ----------
  function update() {
    updateBars();
    document.querySelectorAll(".strip .pt").forEach(el => el.classList.toggle("current", current && +el.dataset.index === current.index));
    if (values.length !== 30 || values.some(Number.isNaN)) {
      $("verdict").textContent = "Valeurs incomplètes";
      $("verdict").className = "verdict";
      return;
    }
    const prob = predict(model, values);
    const malignant = prob >= 0.5;
    const ring = $("ring-fill");
    ring.style.strokeDashoffset = RING * (1 - prob);
    ring.style.stroke = malignant ? "var(--malignant)" : "var(--benign)";
    $("ring-value").textContent = pct(prob);
    $("verdict").innerHTML = `<i class="dot ${malignant ? "malignant" : "benign"}"></i>${malignant ? "Tumeur maligne" : "Tumeur bénigne"}`;
    $("verdict").className = "verdict " + (malignant ? "malignant" : "benign");
    if (current) {
      const ok = (current.label === 1) === malignant;
      $("truth").innerHTML = `Diagnostic réel : <b>${current.label === 1 ? "malin" : "bénin"}</b> · ` +
        (ok ? `<span class="ok">✓ bonne prédiction</span>` : `<span class="ko">✗ erreur du modèle</span>`);
    } else {
      $("source").textContent = "Valeurs modifiées à la main";
      $("truth").textContent = "Diagnostic réel inconnu (saisie manuelle).";
    }
  }

  // ---------- Évaluation ----------
  function counts() {
    const c = { tp: 0, tn: 0, fp: 0, fn: 0 };
    for (const p of model.test_set) {
      const pred = p.prob >= 0.5 ? 1 : 0;
      c[pred === p.label ? (pred ? "tp" : "tn") : (pred ? "fp" : "fn")]++;
    }
    return c;
  }

  function renderKpis() {
    const { tp, tn, fp, fn } = counts();
    const kpis = [
      ["Accuracy", (tp + tn) / model.test_set.length, `${tp + tn} / ${model.test_set.length} patients bien classés`],
      ["Rappel (malin)", tp / (tp + fn), `${tp} cancers détectés sur ${tp + fn}`],
      ["Précision (malin)", tp / (tp + fp), `${fp} fausse(s) alerte(s)`],
      ["AUC-ROC", model.test_metrics.auc, "capacité de séparation"],
    ];
    $("kpis").innerHTML = kpis.map(([label, v, note]) =>
      `<div class="kpi"><div class="kpi-label">${label}</div><div class="kpi-value">${label === "AUC-ROC" ? v.toFixed(3).replace(".", ",") : pct(v)}</div><div class="kpi-note">${note}</div></div>`).join("");
  }

  function renderConfusion() {
    const c = counts();
    const max = Math.max(c.tp, c.tn, c.fp, c.fn);
    const cell = (n, name, desc) => {
      const a = 0.08 + 0.5 * n / max;
      return `<div class="cell" style="background:rgba(124,140,255,${a.toFixed(2)})" data-tip="<b>${name}</b><br>${desc}"><b>${n}</b><span>${name}</span></div>`;
    };
    $("cm").innerHTML = `
      <div></div><div class="axis">Prédit bénin</div><div class="axis">Prédit malin</div>
      <div class="axis">Réel<br>bénin</div>
      ${cell(c.tn, "Vrais négatifs", "Bénins correctement identifiés")}
      ${cell(c.fp, "Faux positifs", "Bénins signalés à tort comme malins")}
      <div class="axis">Réel<br>malin</div>
      ${cell(c.fn, "Faux négatifs", "Cancers manqués : l'erreur la plus grave")}
      ${cell(c.tp, "Vrais positifs", "Cancers correctement détectés")}`;
    $("cm").querySelectorAll(".cell").forEach(el => {
      el.addEventListener("mousemove", e => showTip(e, el.dataset.tip));
      el.addEventListener("mouseleave", hideTip);
    });
  }

  function renderStrip() {
    const strip = $("strip");
    const lanes = { 0: 42, 1: 118 };
    let html = `<div class="axis-line"></div>
      <div class="thr" style="left:50%"><span class="thr-label">seuil 50 %</span></div>`;
    for (const t of [0, 0.25, 0.5, 0.75, 1]) html += `<span class="tick" style="left:${t * 100}%">${t * 100} %</span>`;
    for (const p of model.test_set) {
      const jitter = ((p.index * 9301 + 49297) % 233280) / 233280 - 0.5;
      const color = p.label === 1 ? "var(--malignant)" : "var(--benign)";
      html += `<div class="pt" data-index="${p.index}" style="left:${p.prob * 100}%;top:${lanes[p.label] + jitter * 48}px;background:${color}"></div>`;
    }
    strip.innerHTML = html;
    strip.querySelectorAll(".pt").forEach(el => {
      const p = model.test_set.find(q => q.index === +el.dataset.index);
      const ok = (p.prob >= 0.5 ? 1 : 0) === p.label;
      el.addEventListener("mousemove", e => showTip(e,
        `<b>Patient n°${p.index}</b><br>P(maligne) : ${pct(p.prob)}<br>Réel : ${p.label ? "malin" : "bénin"} · ${ok ? "✓ correct" : "✗ erreur"}`));
      el.addEventListener("mouseleave", hideTip);
      el.addEventListener("click", () => { loadPatient(p); window.scrollTo({ top: 0, behavior: "smooth" }); });
    });
  }

  $("btn-random").onclick = () => loadPatient(pick(model.test_set));
  $("btn-benign").onclick = () => loadPatient(pick(model.test_set.filter(p => p.label === 0)));
  $("btn-malignant").onclick = () => loadPatient(pick(model.test_set.filter(p => p.label === 1)));
  $("btn-error").onclick = () => loadPatient(pick(model.test_set.filter(p => (p.prob >= 0.5 ? 1 : 0) !== p.label)));
}
