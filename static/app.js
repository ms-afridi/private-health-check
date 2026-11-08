const $ = (id) => document.getElementById(id);

let sessionId = null;
let config = null;
let lastCiphertext = null;   // what an eavesdropper would capture
let sigmoidChart = null;

// ---------- Helpers ----------

async function api(path, options = {}) {
  const res = await fetch(path, options);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (_) {}
    const err = new Error(detail);
    err.status = res.status;
    throw err;
  }
  return res.json();
}

function postJSON(path, body) {
  return api(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

function fmtBytes(n) {
  if (n < 1024) return n + " B";
  if (n < 1024 * 1024) return (n / 1024).toFixed(1) + " KB";
  return (n / (1024 * 1024)).toFixed(2) + " MB";
}

function fmtProb(p) { return p.toFixed(5); }

function setStatus(text, cls) {
  const el = $("key-status");
  el.textContent = text;
  el.className = "status-bar" + (cls ? " " + cls : "");
}

// ---------- Sliders ----------

function stepFor(feature) {
  if (feature === "oldpeak") return 0.1;
  return 1;
}

function buildSliders() {
  const box = $("sliders");
  box.innerHTML = "";
  config.features.forEach((f) => {
    const r = config.ranges[f];
    const row = document.createElement("div");
    row.className = "slider-row";
    row.innerHTML = `
      <div class="slider-head">
        <span>${r.label}</span>
        <span class="slider-value" id="val-${f}">${r.default}</span>
      </div>
      <input type="range" id="in-${f}" min="${r.min}" max="${r.max}"
             step="${stepFor(f)}" value="${r.default}" />`;
    box.appendChild(row);
    $("in-" + f).addEventListener("input", (e) => {
      $("val-" + f).textContent = e.target.value;
    });
  });
}

function readValues() {
  const values = {};
  config.features.forEach((f) => { values[f] = parseFloat($("in-" + f).value); });
  return values;
}

// ---------- Keys ----------

async function createKeys() {
  setStatus("Creating encryption keys on the patient device...");
  $("btn-check").disabled = true;
  $("btn-spy").disabled = true;
  try {
    const s = await postJSON("/patient/session", {});
    sessionId = s.session_id;
    setStatus(
      `Keys ready (${s.keygen_seconds}s). Secret key stays on the patient side. ` +
      `Hospital received only the public context (${fmtBytes(s.public_context_bytes)}). ` +
      `Parameters: N=${s.poly_modulus_degree}, scale 2^${s.scale_bits}.`,
      "ok"
    );
    $("hospital-key").textContent = s.hospital_has_secret_key
      ? "WARNING: hospital has a secret key."
      : "The hospital holds NO secret key. It cannot decrypt anything.";
    $("btn-check").disabled = false;
    $("btn-spy").disabled = false;
  } catch (e) {
    setStatus("Could not create keys: " + e.message, "error");
  }
}

// ---------- Main flow ----------

async function runCheck(isRetry = false) {
  const btn = $("btn-check");
  btn.disabled = true;
  btn.textContent = "Working...";
  try {
    const values = readValues();

    // 1) Patient side: encrypt locally
    const enc = await postJSON("/patient/encrypt", { session_id: sessionId, values });
    lastCiphertext = enc.ciphertext;

    // 2) Hospital side shows what it received (only locked data)
    $("hospital-received").textContent = enc.preview.join(" ") + " ...";
    $("hospital-received-note").textContent =
      `Received ${fmtBytes(enc.ciphertext_bytes)} of locked data. It looks like random numbers and cannot be read.`;

    // 3) Hospital computes on the ciphertext
    const hosp = await postJSON("/hospital/predict", {
      session_id: sessionId,
      ciphertext: enc.ciphertext,
    });
    const ops = $("hospital-ops");
    ops.innerHTML = "";
    hosp.operations.forEach((o) => {
      const li = document.createElement("li");
      li.textContent = o;
      ops.appendChild(li);
    });
    $("hospital-note").textContent =
      `Finished in ${hosp.compute_seconds}s without seeing any patient value. ` +
      `Returned ${fmtBytes(hosp.result_bytes)} of locked result.`;

    // 4) Patient side: unlock the answer
    const dec = await postJSON("/patient/decrypt", {
      session_id: sessionId,
      result_ciphertext: hosp.result_ciphertext,
    });

    // 5) Comparison with the normal (unencrypted) model
    const plain = await postJSON("/predict_plain", { session_id: sessionId, values });

    showResult(dec);
    showProof(enc, hosp, dec, plain);
  } catch (e) {
    if (e.status === 410 && !isRetry) {
      // Session was evicted from server memory: make new keys and retry once.
      await createKeys();
      btn.textContent = "Check my risk privately";
      return runCheck(true);
    }
    setStatus("Error: " + e.message, "error");
  } finally {
    btn.disabled = false;
    btn.textContent = "Check my risk privately";
  }
}

function showResult(dec) {
  const pct = Math.max(0, Math.min(100, dec.probability * 100));
  $("result").classList.remove("hidden");
  $("res-percent").textContent = pct.toFixed(0) + "%";
  const lab = $("res-label");
  lab.textContent = dec.label + " risk";
  lab.className = "result-label " + dec.label.toLowerCase();
}

function showProof(enc, hosp, dec, plain) {
  $("proof-plain").textContent = fmtProb(plain.probability_poly);
  $("proof-enc").textContent = fmtProb(dec.probability);
  const diff = Math.abs(dec.probability - plain.probability_poly);
  $("proof-diff").textContent = diff.toExponential(2);
  $("proof-true").textContent = fmtProb(plain.probability);
  $("proof-time").textContent =
    `${enc.encrypt_seconds} / ${hosp.compute_seconds} / ${dec.decrypt_seconds}`;
  $("proof-size").textContent =
    `${fmtBytes(enc.plaintext_bytes)} vs ${fmtBytes(enc.ciphertext_bytes)}`;
}

// ---------- Spy ----------

async function runSpy() {
  const btn = $("btn-spy");
  btn.disabled = true;
  try {
    if (!lastCiphertext) {
      const enc = await postJSON("/patient/encrypt", {
        session_id: sessionId,
        values: readValues(),
      });
      lastCiphertext = enc.ciphertext;
    }
    const out = $("spy-output");
    out.classList.remove("hidden");
    out.textContent = "";
    out.append(
      "POST /hospital/predict\n\n" +
      "ciphertext: " + lastCiphertext.slice(0, 700) + " ...\n"
    );
    const msg = document.createElement("span");
    msg.className = "spy-msg";
    msg.textContent =
      "This is everything an eavesdropper captures. Without the patient's secret key " +
      "it is just noise: no age, no blood pressure, no cholesterol, and no result.";
    out.appendChild(msg);
  } catch (e) {
    setStatus("Error: " + e.message, "error");
  } finally {
    btn.disabled = false;
  }
}

// ---------- Math chart ----------

function drawChart(a) {
  const ctx = $("sigmoid-chart").getContext("2d");
  if (sigmoidChart) sigmoidChart.destroy();
  sigmoidChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: a.z.map((v) => v.toFixed(1)),
      datasets: [
        { label: "True sigmoid", data: a.sigmoid, borderColor: "#2b5fa8",
          borderWidth: 3, pointRadius: 0 },
        { label: `Polynomial (degree ${a.degree})`, data: a.approx,
          borderColor: "#c0392b", borderDash: [6, 4], borderWidth: 2, pointRadius: 0 },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { title: { display: true, text: "model score z" } },
        y: { title: { display: true, text: "risk probability" } },
      },
    },
  });
  const c = a.coefficients.map((v) => v.toFixed(5));
  $("math-info").textContent =
    `Fitted on z in [${a.interval[0].toFixed(2)}, ${a.interval[1].toFixed(2)}]. ` +
    `Polynomial: ${c[0]} + ${c[1]}·z + ${c[2]}·z³. ` +
    `Maximum error on this interval: ${a.max_abs_error.toFixed(4)}.`;
}

function showModelInfo(m) {
  $("model-info").textContent =
    `${m.dataset}: ${m.n_samples} patients (${m.n_train} train, ${m.n_test} test). ` +
    `Logistic regression, test accuracy ${(m.accuracy * 100).toFixed(1)}%, AUC ${m.auc}. ` +
    `Fixed random seed ${m.seed}, so results are reproducible.`;
}

// ---------- Start ----------

async function init() {
  try {
    config = await api("/api/config");
    buildSliders();
    showModelInfo(config.metrics);
    drawChart(await api("/api/approximation"));
  } catch (e) {
    setStatus("Could not load model settings: " + e.message, "error");
    return;
  }
  $("btn-check").addEventListener("click", () => runCheck());
  $("btn-spy").addEventListener("click", runSpy);
  await createKeys();
}

init();