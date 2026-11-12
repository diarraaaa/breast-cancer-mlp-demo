// Inférence du MLP exporté par train.py : standardisation puis 3 couches denses.

const FR = {
  "radius": "Rayon", "texture": "Texture", "perimeter": "Périmètre", "area": "Aire",
  "smoothness": "Lissité", "compactness": "Compacité", "concavity": "Concavité",
  "concave points": "Points concaves", "symmetry": "Symétrie", "fractal dimension": "Dimension fractale",
};
const GROUPS = [
  { title: "Moyenne (mean)", from: 0 },
  { title: "Erreur standard (error)", from: 10 },
  { title: "Pire valeur (worst)", from: 20 },
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
  let model, inputs, current = null;

  fetch("model.json").then(r => r.json()).then(m => {
    model = m;
    buildForm();
    renderEvaluation();
    loadPatient(pick(model.test_set));
  });

  function buildForm() {
    inputs = [];
    for (const g of GROUPS) {
      const fs = document.createElement("fieldset");
      fs.innerHTML = `<legend>${g.title}</legend>`;
      for (let i = g.from; i < g.from + 10; i++) {
        const label = document.createElement("label");
        label.textContent = FR[baseName(model.feature_names[i])];
        label.title = model.feature_names[i];
        const input = document.createElement("input");
        input.type = "number";
        input.step = "any";
        input.addEventListener("input", () => { current = null; update(); });
        label.appendChild(input);
        fs.appendChild(label);
        inputs[i] = input;
      }
      $("form").appendChild(fs);
    }
  }

  function pick(list) { return list[Math.floor(Math.random() * list.length)]; }

  function loadPatient(p) {
    current = p;
    p.x.forEach((v, i) => { inputs[i].value = fmt(v); });
    $("source").textContent = `Patient n°${p.index} du jeu de test (jamais vu pendant l'entraînement).`;
    update();
  }

  function update() {
    const x = inputs.map(el => parseFloat(el.value));
    if (x.some(Number.isNaN)) {
      $("verdict").textContent = "Remplissez toutes les valeurs.";
      $("verdict").className = "verdict";
      return;
    }
    const prob = predict(model, x);
    const malignant = prob >= 0.5;
    $("gauge-fill").style.width = `${prob * 100}%`;
    $("gauge-label").textContent = `P(maligne) = ${(prob * 100).toFixed(1)} %`;
    $("verdict").textContent = malignant ? "Prédiction : tumeur MALIGNE" : "Prédiction : tumeur BÉNIGNE";
    $("verdict").className = "verdict " + (malignant ? "malignant" : "benign");
    if (current) {
      const truth = current.label === 1 ? "maligne" : "bénigne";
      const ok = (current.label === 1) === malignant;
      $("truth").textContent = `Vraie classe : ${truth}. ${ok ? "✅ Bonne prédiction" : "❌ Erreur du modèle"}`;
    } else {
      $("source").textContent = "Valeurs saisies manuellement.";
      $("truth").textContent = "";
    }
  }

  function renderEvaluation() {
    let tp = 0, tn = 0, fp = 0, fn = 0;
    for (const p of model.test_set) {
      const pred = predict(model, p.x) >= 0.5 ? 1 : 0;
      if (pred === 1 && p.label === 1) tp++;
      else if (pred === 0 && p.label === 0) tn++;
      else if (pred === 1) fp++;
      else fn++;
    }
    $("cm").innerHTML = `
      <tr><th></th><th>Prédit bénin</th><th>Prédit malin</th></tr>
      <tr><th>Réel bénin</th><td class="hit">${tn}</td><td class="miss">${fp}</td></tr>
      <tr><th>Réel malin</th><td class="miss">${fn}</td><td class="hit">${tp}</td></tr>`;
    const precision = tp / (tp + fp), recall = tp / (tp + fn);
    const rows = [
      ["Accuracy", (tp + tn) / model.test_set.length],
      ["Précision (maligne)", precision],
      ["Rappel (maligne)", recall],
      ["F1-score (maligne)", 2 * precision * recall / (precision + recall)],
      ["AUC-ROC", model.test_metrics.auc],
    ];
    $("metrics").innerHTML = rows.map(([k, v]) => `<tr><td>${k}</td><td>${(v * 100).toFixed(1)} %</td></tr>`).join("");
  }

  $("btn-random").onclick = () => loadPatient(pick(model.test_set));
  $("btn-benign").onclick = () => loadPatient(pick(model.test_set.filter(p => p.label === 0)));
  $("btn-malignant").onclick = () => loadPatient(pick(model.test_set.filter(p => p.label === 1)));
  $("btn-error").onclick = () => {
    const errors = model.test_set.filter(p => (predict(model, p.x) >= 0.5 ? 1 : 0) !== p.label);
    if (errors.length) loadPatient(pick(errors));
  };
}
