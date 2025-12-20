let currentRunId = null;
let evtSource = null;
let evidenceToggle = false;
let cachedArtifacts = null;

function el(id) { return document.getElementById(id); }

function appendChat(role, text) {
  const wrap = document.createElement("div");
  wrap.className = "chat-msg";
  wrap.innerHTML = `<span class="role">${role}:</span> ${text}`;
  el("chatThread").appendChild(wrap);
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
  el("cfgBaseUrl").value = s.lm_studio_base_url;
  el("cfgOrch").value = s.model_orch;
  el("cfgQwen8").value = s.model_qwen8;
  el("cfgQwen4").value = s.model_qwen4;
  el("cfgTavily").value = s.tavily_api_key || "";
  el("cfgSearchMode").value = s.search_depth_mode;
  el("cfgMaxBase").value = s.max_results_base;
  el("cfgMaxHigh").value = s.max_results_high;
  el("cfgExtract").value = s.extract_depth;
  el("modelWarning").textContent = "";
  if (data.model_check && data.model_check.ok === false) {
    if (data.model_check.missing && data.model_check.missing.length) {
      el("modelWarning").textContent = `Missing models: ${data.model_check.missing.join(", ")}`;
    } else if (data.model_check.error) {
      el("modelWarning").textContent = data.model_check.error;
    }
  }
}

function toggleModal(show) {
  el("settingsModal").classList[show ? "remove" : "add"]("hidden");
}

async function saveSettings() {
  const payload = {
    lm_studio_base_url: el("cfgBaseUrl").value,
    model_orch: el("cfgOrch").value,
    model_qwen8: el("cfgQwen8").value,
    model_qwen4: el("cfgQwen4").value,
    tavily_api_key: el("cfgTavily").value,
    search_depth_mode: el("cfgSearchMode").value,
    max_results_base: Number(el("cfgMaxBase").value),
    max_results_high: Number(el("cfgMaxHigh").value),
    extract_depth: el("cfgExtract").value,
  };
  const res = await fetch("/settings", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (res.ok) {
    el("settingsStatus").textContent = "Saved. Restart runs to apply.";
    loadSettings();
  } else {
    el("settingsStatus").textContent = "Save failed.";
  }
}

function subscribeEvents(runId) {
  if (evtSource) {
    evtSource.close();
  }
  evtSource = new EventSource(`/runs/${runId}/events`);
  evtSource.onmessage = async (evt) => {
    const data = JSON.parse(evt.data);
    const type = data.event_type;
    const p = data.payload || {};
    switch (type) {
      case "run_started":
        appendActivity(`Run started: ${p.question || ""}`);
        break;
      case "router_decision":
        appendActivity(`Router -> needs_web=${p.needs_web} level=${p.reasoning_level} max=${p.max_results}`);
        break;
      case "router_skip_web":
        appendActivity(`Router decided no web research`);
        break;
      case "strict_mode":
        appendActivity(`Strict mode enabled`);
        break;
      case "plan":
        appendActivity(`Plan tasks: ${(p.tasks || []).join(" | ")}`);
        break;
      case "lane_started":
        appendActivity(`Lane ${p.lane} started`);
        break;
      case "lane_queries":
        appendActivity(`Lane ${p.lane} queries: ${(p.queries || []).join("; ")}`);
        break;
      case "lane_extract":
        appendActivity(`Lane ${p.lane} extracting ${p.urls ? p.urls.length : 0} URLs`);
        break;
      case "lane_finished":
        appendActivity(`Lane ${p.lane} finished (${p.sources || 0} sources)`);
        break;
      case "merge_summary":
        appendActivity(`Merge: ${p.sources} sources, ${p.claims} claims, conflicts ${p.conflicts}`);
        break;
      case "draft_ready":
        appendActivity(`Draft ready (${p.chars} chars)`);
        break;
      case "verifier_verdict":
        appendActivity(`Verifier: ${p.verdict} (${p.issues} issues, extra ${p.extra_queries})`);
        break;
      case "loop_iteration":
        appendActivity(`Loop ${p.iteration}: extra queries ${p.extra_queries}`);
        break;
      case "archived":
        appendActivity(`Archived run ${p.run_id}`);
        await fetchArtifacts(runId);
        break;
      case "error":
        appendActivity(`Error: ${p.message}`);
        break;
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
  }
  appendChat("assistant", run.final_answer || draft || "");
}

async function startRun(evt) {
  evt.preventDefault();
  const question = el("question").value.trim();
  if (!question) return;
  appendChat("user", question);
  el("finalAnswer").textContent = "";
  el("sources").textContent = "";
  el("confidence").textContent = "";
  el("evidence").classList.add("hidden");
  const payload = {
    question,
    reasoning_mode: document.querySelector('input[name="mode"]:checked').value,
    manual_level: el("manualLevel").value,
    evidence_dump: el("evidenceDump").checked,
    search_depth_mode: el("searchDepth").value,
    max_results: Number(el("maxResults").value),
    strict_mode: el("strictMode").checked,
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

document.addEventListener("DOMContentLoaded", () => {
  loadSettings();
  el("chatForm").addEventListener("submit", startRun);
  el("settingsBtn").addEventListener("click", () => toggleModal(true));
  el("closeSettings").addEventListener("click", () => toggleModal(false));
  el("saveSettings").addEventListener("click", saveSettings);
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
});
