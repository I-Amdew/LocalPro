let currentRunId = null;
let evtSource = null;
let evidenceToggle = false;
let cachedArtifacts = null;
let sttRecognition = null;
let sttActive = false;

function el(id) { return document.getElementById(id); }

function depthLabelFromValue(v) {
  return ["LOW", "MED", "HIGH", "ULTRA"][Number(v) || 0];
}

function appendChat(role, text) {
  const wrap = document.createElement("div");
  wrap.className = `chat-bubble ${role === "user" ? "bubble-user" : "bubble-assistant"}`;
  wrap.textContent = `${role}: ${text}`;
  el("chatThread").appendChild(wrap);
  el("chatThread").scrollTop = el("chatThread").scrollHeight;
}

function appendActivity(text) {
  const div = document.createElement("div");
  div.className = "event";
  div.textContent = text;
  el("activityFeed").appendChild(div);
  el("activityFeed").scrollTop = el("activityFeed").scrollHeight;
}

async function loadSettings() {
  const res = await fetch("/settings");
  const data = await res.json();
  const s = data.settings;
  el("cfgBaseUrl").value = s.lm_studio_base_url || "";
  el("cfgOrchBase").value = s.orch_endpoint.base_url || "";
  el("cfgOrch").value = s.orch_endpoint.model_id || "";
  el("cfgWorkerABase").value = s.worker_a_endpoint.base_url || "";
  el("cfgQwen8").value = s.worker_a_endpoint.model_id || "";
  el("cfgWorkerBBase").value = s.worker_b_endpoint.base_url || "";
  el("cfgQwen8B").value = s.worker_b_endpoint.model_id || "";
  el("cfgWorkerCBase").value = s.worker_c_endpoint.base_url || "";
  el("cfgQwen8C").value = s.worker_c_endpoint.model_id || "";
  el("cfgRouterBase").value = s.router_endpoint.base_url || "";
  el("cfgQwen4").value = s.router_endpoint.model_id || "";
  el("cfgSummarizerBase").value = s.summarizer_endpoint.base_url || "";
  el("cfgSummarizer").value = s.summarizer_endpoint.model_id || "";
  el("cfgVerifierBase").value = s.verifier_endpoint.base_url || "";
  el("cfgVerifier").value = s.verifier_endpoint.model_id || "";
  el("cfgTavily").value = s.tavily_api_key || "";
  el("cfgSearchMode").value = s.search_depth_mode || "auto";
  el("cfgMaxBase").value = s.max_results_base || 6;
  el("cfgMaxHigh").value = s.max_results_high || 10;
  el("cfgExtract").value = s.extract_depth || "basic";
  el("cfgDiscovery").value = (s.discovery_base_urls || []).join(", ");
  el("modelWarning").textContent = "";
  if (data.model_check) {
    const bad = Object.entries(data.model_check).filter(([, v]) => v.ok === false);
    if (bad.length) {
      el("modelWarning").textContent = `Model issues: ${bad.map(([r, v]) => `${r}:${v.missing || v.error || ""}`).join(", ")}`;
    }
  }
}

function toggleModal(show) {
  el("settingsModal").classList[show ? "remove" : "add"]("hidden");
}

async function saveSettings() {
  const tavKey = el("cfgTavily").value;
  const payload = {
    lm_studio_base_url: el("cfgBaseUrl").value,
    orch_endpoint: { base_url: el("cfgOrchBase").value, model_id: el("cfgOrch").value },
    worker_a_endpoint: { base_url: el("cfgWorkerABase").value, model_id: el("cfgQwen8").value },
    worker_b_endpoint: { base_url: el("cfgWorkerBBase").value, model_id: el("cfgQwen8B").value },
    worker_c_endpoint: { base_url: el("cfgWorkerCBase").value, model_id: el("cfgQwen8C").value },
    router_endpoint: { base_url: el("cfgRouterBase").value, model_id: el("cfgQwen4").value },
    summarizer_endpoint: { base_url: el("cfgSummarizerBase").value, model_id: el("cfgSummarizer").value },
    verifier_endpoint: { base_url: el("cfgVerifierBase").value, model_id: el("cfgVerifier").value },
    tavily_api_key: tavKey === "********" ? undefined : tavKey,
    search_depth_mode: el("cfgSearchMode").value,
    max_results_base: Number(el("cfgMaxBase").value),
    max_results_high: Number(el("cfgMaxHigh").value),
    extract_depth: el("cfgExtract").value,
    discovery_base_urls: el("cfgDiscovery").value.split(",").map((v) => v.trim()).filter(Boolean),
  };
  const res = await fetch("/settings", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  el("settingsStatus").textContent = res.ok ? "Saved. Restart runs to apply." : "Save failed.";
  if (res.ok) loadSettings();
}

async function discoverModels() {
  const base_urls = el("cfgDiscovery").value.split(",").map((v) => v.trim()).filter(Boolean);
  const res = await fetch("/api/discover", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ base_urls }),
  });
  const data = await res.json();
  el("settingsStatus").textContent = JSON.stringify(data.results);
}

function subscribeEvents(runId) {
  if (evtSource) evtSource.close();
  evtSource = new EventSource(`/runs/${runId}/events`);
  evtSource.onmessage = async (evt) => {
    const data = JSON.parse(evt.data);
    const type = data.event_type;
    const p = data.payload || {};
    switch (type) {
      case "run_started":
        appendActivity(`Run started: ${p.question || ""}`); break;
      case "router_decision":
        appendActivity(`Router -> needs_web=${p.needs_web} level=${p.reasoning_level} max=${p.max_results}`); break;
      case "router_skip_web":
        appendActivity(`Router decided no web research`); break;
      case "strict_mode":
        appendActivity(`Strict mode enabled`); break;
      case "plan_created":
        appendActivity(`Plan created with ${p.steps} steps`); break;
      case "step_started":
        appendActivity(`Step ${p.step_id}: ${p.name} started`); break;
      case "step_completed":
        appendActivity(`Step ${p.step_id}: ${p.name} completed`); break;
      case "control_action":
        appendActivity(`Control: ${p.control} ${p.reason || ""}`); break;
      case "tavily_search":
        appendActivity(`Search step ${p.step} query: ${p.query}`); break;
      case "tavily_extract":
        appendActivity(`Extract step ${p.step} urls=${(p.urls || []).length}`); break;
      case "merge_summary":
        appendActivity(`Merge: ${p.sources} sources, ${p.claims} claims, conflicts ${p.conflicts}`); break;
      case "draft_ready":
        appendActivity(`Draft ready (${p.chars} chars)`); break;
      case "verifier_verdict":
        appendActivity(`Verifier: ${p.verdict} (${p.issues} issues, extra ${p.extra_queries})`); break;
      case "loop_iteration":
        appendActivity(`Loop ${p.iteration}: extra queries ${p.extra_queries}`); break;
      case "archived":
        appendActivity(`Archived run ${p.run_id}`);
        await fetchArtifacts(runId);
        break;
      case "memory_retrieved":
        appendActivity(`Memory retrieved: ${p.count}`); break;
      case "memory_saved":
        appendActivity(`Memory saved: ${p.count}`); break;
      case "error":
        appendActivity(`Error: ${p.message}`); break;
      default:
        appendActivity(`${type}`);
    }
  };
}

async function fetchArtifacts(runId) {
  const res = await fetch(`/api/run/${runId}/artifacts`);
  if (!res.ok) return;
  const data = await res.json();
  cachedArtifacts = data;
  const run = data.run;
  const sources = data.sources || [];
  const claims = data.claims || [];
  const draft = data.draft || "";
  const verifier = data.verifier || {};
  el("finalAnswer").textContent = run.final_answer || draft || "(no answer yet)";
  el("confidence").textContent = `Confidence: ${run.confidence || "n/a"}`;
  el("sources").innerHTML = "<strong>Sources</strong>: " + sources.map((s) => `<a href="${s.url}" target="_blank">${s.url}</a>`).join(" • ");
  if (evidenceToggle) {
    const wrap = el("evidence");
    wrap.classList.remove("hidden");
    wrap.innerHTML = "<h4>Claims</h4>" + claims.map((c) => `<div>${c.claim} — ${c.support_urls.join(", ")}</div>`).join("");
  } else {
    el("evidence").classList.add("hidden");
  }
  appendChat("assistant", run.final_answer || draft || "");
}

async function startRun(evt) {
  evt.preventDefault();
  const question = el("question").value.trim();
  if (!question) return;
  const depthAuto = el("depthAuto").checked;
  const depthValue = depthLabelFromValue(el("depthSlider").value);
  appendChat("user", question);
  el("finalAnswer").textContent = "";
  el("sources").textContent = "";
  el("confidence").textContent = "";
  el("evidence").classList.add("hidden");
  const payload = {
    question,
    reasoning_mode: depthAuto ? "auto" : "manual",
    manual_level: depthValue,
    evidence_dump: el("evidenceDump").checked,
    search_depth_mode: el("searchDepth").value,
    max_results: Number(el("maxResults").value),
    strict_mode: el("strictMode").checked,
    auto_memory: el("autoMemory").checked,
    reasoning_auto: depthAuto,
  };
  const res = await fetch("/api/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    appendActivity("Failed to start run.");
    return;
  }
  const data = await res.json();
  currentRunId = data.run_id;
  appendActivity(`Run id: ${currentRunId}`);
  subscribeEvents(currentRunId);
}

async function loadMemory(searchTerm = "") {
  const url = searchTerm ? `/api/memory?q=${encodeURIComponent(searchTerm)}` : "/api/memory";
  const res = await fetch(url);
  if (!res.ok) return;
  const data = await res.json();
  const list = el("memoryList");
  list.innerHTML = "";
  (data.items || []).forEach((m) => {
    const div = document.createElement("div");
    div.className = "event";
    div.innerHTML = `<strong>${m.title}</strong> [${m.kind}] ${m.content}<br><small>${m.tags.join(", ")}</small>
    <button data-id="${m.id}" class="pin-btn">${m.pinned ? "Unpin" : "Pin"}</button>
    <button data-id="${m.id}" class="del-btn">Delete</button>`;
    list.appendChild(div);
  });
  list.querySelectorAll(".pin-btn").forEach((btn) => {
    btn.onclick = async () => {
      const id = btn.getAttribute("data-id");
      await fetch(`/api/memory/${id}`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ pinned: btn.textContent === "Pin" }) });
      loadMemory(searchTerm);
    };
  });
  list.querySelectorAll(".del-btn").forEach((btn) => {
    btn.onclick = async () => {
      const id = btn.getAttribute("data-id");
      await fetch(`/api/memory/${id}`, { method: "DELETE" });
      loadMemory(searchTerm);
    };
  });
}

function setupSTT() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    appendActivity("Speech recognition not supported in this browser.");
    return;
  }
  sttRecognition = new SpeechRecognition();
  sttRecognition.continuous = true;
  sttRecognition.interimResults = true;
  sttRecognition.lang = el("sttLang").value || "en-US";
  sttRecognition.onresult = (event) => {
    let transcript = "";
    for (let i = event.resultIndex; i < event.results.length; ++i) {
      transcript += event.results[i][0].transcript;
    }
    el("question").value = transcript;
  };
  sttRecognition.onend = () => { sttActive = false; el("micBtn").textContent = "🎤"; };
  sttRecognition.start();
  sttActive = true;
  el("micBtn").textContent = "■";
}

document.addEventListener("DOMContentLoaded", () => {
  loadSettings();
  loadMemory();
  el("chatForm").addEventListener("submit", startRun);
  el("settingsBtn").addEventListener("click", () => toggleModal(true));
  el("closeSettings").addEventListener("click", () => toggleModal(false));
  el("saveSettings").addEventListener("click", saveSettings);
  el("discoverBtn").addEventListener("click", discoverModels);
  el("evidenceDump").addEventListener("change", (e) => {
    evidenceToggle = e.target.checked;
    if (cachedArtifacts && evidenceToggle) {
      const claims = cachedArtifacts.claims || [];
      el("evidence").classList.remove("hidden");
      el("evidence").innerHTML = "<h4>Claims</h4>" + claims.map((c) => `<div>${c.claim} — ${c.support_urls.join(", ")}</div>`).join("");
    } else {
      el("evidence").classList.add("hidden");
    }
  });
  el("memorySearch").addEventListener("input", (e) => loadMemory(e.target.value));
  el("refreshMemory").addEventListener("click", () => loadMemory(el("memorySearch").value));
  el("toggleActivity").addEventListener("click", () => {
    el("activityFeed").classList.toggle("hidden");
  });
  el("depthSlider").addEventListener("input", (e) => {
    const lbl = depthLabelFromValue(e.target.value);
    el("depthLabel").textContent = lbl;
    el("manualLevel").value = lbl;
  });
  el("micBtn").addEventListener("click", () => {
    if (sttActive && sttRecognition) {
      sttRecognition.stop();
      sttActive = false;
      el("micBtn").textContent = "🎤";
    } else {
      setupSTT();
    }
  });
  document.addEventListener("keydown", (e) => {
    if (e.ctrlKey && e.key.toLowerCase() === "k") {
      e.preventDefault();
      toggleModal(true);
    }
    if (e.key === "Enter" && !e.shiftKey && document.activeElement === el("question")) {
      e.preventDefault();
      el("chatForm").dispatchEvent(new Event("submit"));
    }
  });
});
