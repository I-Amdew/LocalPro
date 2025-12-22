let currentRunId = null;
let evtSource = null;
let cachedArtifacts = null;
let sttRecognition = null;
let sttActive = false;
let sttBuffer = "";
let timerInterval = null;
let timerStartedAt = null;
let etaTargetAt = null;
let lastAssistantRunId = null;
let historyLog = [];
let totalSteps = 0;
let completedSteps = 0;
let triedAutoDiscover = false;
let liveEventBuffer = [];
let liveFlushTimer = null;
let pendingQuestion = "";
let questionShownInLive = false;
let latestRunPoll = null;
let pendingUploads = [];
let renderedMessages = new Set();
let pendingLocalMessages = [];
let conversationResetAt = null;
let currentTier = "pro";
let runDetails = {};
let runEvents = {};
let selectedRunId = null;
let tabScrollPositions = { activity: 0, reasoning: 0, sources: 0 };
let thinkingPlaceholders = {};
let activeDrawerTab = "activity";
let uploadPanelVisible = false;
let draggingUploads = false;
let settingsDefaults = {
  search_depth_mode: "auto",
  strict_mode: false,
  auto_memory: true,
  evidence_dump: false,
  max_results_override: 0,
  stt_lang: "en-US",
};

function el(id) {
  return document.getElementById(id);
}

function formatTime(ms) {
  const total = Math.max(0, Math.floor(ms / 1000));
  const m = String(Math.floor(total / 60)).padStart(2, "0");
  const s = String(total % 60).padStart(2, "0");
  return `${m}:${s}`;
}

function formatBytes(bytes) {
  if (!bytes && bytes !== 0) return "";
  const units = ["B", "KB", "MB", "GB"];
  let val = bytes;
  let idx = 0;
  while (val >= 1024 && idx < units.length - 1) {
    val /= 1024;
    idx += 1;
  }
  return `${val.toFixed(val >= 10 ? 0 : 1)} ${units[idx]}`;
}

function setStatus(text, tone = "idle") {
  const chip = el("runStatus");
  if (!chip) return;
  chip.textContent = text;
  if (tone === "live") {
    chip.style.background = "rgba(122,215,255,0.18)";
    chip.style.borderColor = "rgba(122,215,255,0.4)";
    chip.style.color = "#eaf1ff";
  } else if (tone === "error") {
    chip.style.background = "rgba(255,125,125,0.14)";
    chip.style.borderColor = "rgba(255,125,125,0.4)";
    chip.style.color = "#ffd6d6";
  } else if (tone === "done") {
    chip.style.background = "rgba(125,255,163,0.12)";
    chip.style.borderColor = "rgba(125,255,163,0.35)";
    chip.style.color = "#eaf1ff";
  } else {
    chip.style.background = "rgba(255,255,255,0.06)";
    chip.style.borderColor = "rgba(255,255,255,0.15)";
    chip.style.color = "#eaf1ff";
  }
}

function startTimer() {
  clearInterval(timerInterval);
  timerStartedAt = Date.now();
  el("runTimer").textContent = "00:00";
  timerInterval = setInterval(() => {
    if (timerStartedAt) {
      el("runTimer").textContent = formatTime(Date.now() - timerStartedAt);
    }
    updateEtaCountdown();
  }, 500);
}

function stopTimer(label) {
  clearInterval(timerInterval);
  timerInterval = null;
  if (label) {
    el("runTimer").textContent = label;
  } else if (timerStartedAt) {
    el("runTimer").textContent = formatTime(Date.now() - timerStartedAt);
  }
  timerStartedAt = null;
}

function updateEtaCountdown() {
  if (etaTargetAt === null) return;
  const etaEl = el("etaText");
  if (!etaEl) return;
  const remaining = Math.max(etaTargetAt - Date.now(), 0);
  etaEl.textContent = `ETA: ${formatTime(remaining)}`;
  if (remaining === 0) {
    etaTargetAt = null;
  }
}

function startNewConversation(opts = {}) {
  const keepQuestion = opts.keepQuestion || false;
  const keepUploads = opts.keepUploads || false;
  const silent = opts.silent || false;
  const preserveChat = opts.preserveChat || false;
  if (evtSource) {
    evtSource.close();
    evtSource = null;
  }
  currentRunId = null;
  cachedArtifacts = null;
  lastAssistantRunId = null;
  historyLog = [];
  runDetails = {};
  runEvents = {};
  selectedRunId = null;
  thinkingPlaceholders = {};
  renderHistory();
  resetLiveStreamState("");
  updateLiveTicker("Idle");
  setStatus("Idle");
  stopTimer("00:00");
  totalSteps = 0;
  completedSteps = 0;
  updateProgressUI();
  el("progressText").textContent = "Progress: 0%";
  el("etaText").textContent = "ETA: --";
  el("confidence").textContent = "";
  el("sources").textContent = "";
  if (!keepQuestion) {
    el("question").value = "";
    updateCharCount();
  }
  if (!keepUploads) {
    pendingUploads = [];
    uploadPanelVisible = false;
    draggingUploads = false;
    renderAttachments();
  }
  pendingLocalMessages = [];
  if (!preserveChat) {
    renderedMessages = new Set();
    pendingLocalMessages = [];
    el("chatThread").innerHTML = "";
  }
  closeDrawer();
  updateEmptyState();
  if (!silent) appendActivity("Ready for a new shared conversation.");
}

async function syncConversationHistory() {
  try {
    const res = await fetch("/api/conversation");
    if (!res.ok) return;
    const data = await res.json();
    conversationResetAt = data.reset_at || conversationResetAt;
    (data.messages || []).forEach((msg) => {
      appendChat(msg.role || "assistant", msg.content || "", { messageId: msg.id, runId: msg.run_id });
    });
  } catch (err) {
    // ignore hydration errors
  }
}

async function resetConversation() {
  try {
    const res = await fetch("/api/conversation", { method: "DELETE" });
    const data = await res.json().catch(() => ({}));
    conversationResetAt = data.reset_at || new Date().toISOString();
  } catch (err) {
    conversationResetAt = new Date().toISOString();
  }
  renderedMessages = new Set();
  pendingLocalMessages = [];
  startNewConversation({ keepQuestion: false, keepUploads: false, silent: false, preserveChat: false });
}

function escapeAndBreak(text) {
  const div = document.createElement("div");
  div.textContent = text || "";
  return div.innerHTML.replace(/\n/g, "<br>");
}

function updateEmptyState() {
  const empty = el("emptyState");
  const thread = el("chatThread");
  if (!empty || !thread) return;
  const hasMessages = thread.children.length > 0;
  empty.classList.toggle("hidden", hasMessages);
}

function appendChat(role, text, opts = {}) {
  const messageId = opts.messageId || opts.id || null;
  const runId = opts.runId || null;
  const cleanText = text || "";
  const html = opts.html || null;
  const skipPending = opts.skipPending === true;
  const state = opts.state || null;
  const idKey = messageId ? String(messageId) : null;
  if (idKey && renderedMessages.has(idKey)) return;
  if (idKey) {
    const pendingIdx = pendingLocalMessages.findIndex((m) => m.role === role && m.text === cleanText);
    if (pendingIdx >= 0) {
      const pending = pendingLocalMessages[pendingIdx];
      pending.el.dataset.messageId = idKey;
      if (runId) pending.el.dataset.runId = runId;
      pending.el.dataset.role = role;
      renderedMessages.add(idKey);
      pendingLocalMessages.splice(pendingIdx, 1);
      return;
    }
  }
  const wrap = document.createElement("div");
  wrap.className = `chat-bubble ${role === "user" ? "bubble-user" : "bubble-assistant"}`;
  wrap.dataset.role = role;
  if (state) wrap.classList.add(state);
  if (runId) wrap.dataset.runId = runId;
  if (idKey) {
    wrap.dataset.messageId = idKey;
    renderedMessages.add(idKey);
  }
  const head = document.createElement("div");
  head.className = "bubble-head";
  head.textContent = role === "user" ? "You" : "LocalPro";
  const body = document.createElement("div");
  body.className = "bubble-body";
  body.innerHTML = html !== null ? html : escapeAndBreak(cleanText);
  wrap.appendChild(head);
  wrap.appendChild(body);
  el("chatThread").appendChild(wrap);
  if (!idKey && !skipPending) pendingLocalMessages.push({ role, text: cleanText, el: wrap });
  el("chatThread").scrollTop = el("chatThread").scrollHeight;
  updateEmptyState();
}

function hasMessageForRun(role, runId) {
  if (!runId) return false;
  return !!document.querySelector(`.chat-bubble[data-run-id="${runId}"][data-role="${role}"]`);
}

function getAssistantBubble(runId) {
  if (!runId) return null;
  return document.querySelector(`.chat-bubble[data-run-id="${runId}"][data-role="assistant"]`);
}

function buildThinkingHTML() {
  const hints = ["Parsing your request", "Planning the path", "Looking up context", "Drafting a reply"];
  const steps = hints.map((h) => `<li><span>•</span><span>${h}</span></li>`).join("");
  return `
    <div class="thinking-block">
      <div>
        <div class="thinking-title">Thinking... <span class="thinking-dots"><span></span><span></span><span></span></span></div>
        <ul class="thinking-steps">${steps}</ul>
      </div>
      <button type="button" class="stop-btn" data-action="stop">Stop generating</button>
    </div>
  `;
}

function attachThinkingActions(bubble, runId) {
  if (!bubble) return;
  const stopBtn = bubble.querySelector(".stop-btn");
  if (stopBtn) {
    stopBtn.onclick = () => stopStreaming(runId || currentRunId);
  }
}

function ensureThinkingPlaceholder(runId) {
  if (!runId) return null;
  let bubble = getAssistantBubble(runId);
  const html = buildThinkingHTML();
  if (!bubble) {
    bubble = appendChat("assistant", "", { runId, html, skipPending: true, state: "thinking" });
  } else {
    const body = bubble.querySelector(".bubble-body");
    if (body) body.innerHTML = html;
    bubble.classList.add("thinking");
  }
  bubble.dataset.state = "thinking";
  attachThinkingActions(bubble, runId);
  return bubble;
}

function stopStreaming(runId) {
  if (evtSource) {
    try {
      evtSource.close();
    } catch (_) {}
    evtSource = null;
  }
  setStatus("Stopped", "error");
  updateLiveTicker("Generation stopped");
  const bubble = getAssistantBubble(runId || currentRunId);
  if (bubble) {
    bubble.classList.remove("thinking");
    bubble.dataset.state = "stopped";
    const body = bubble.querySelector(".bubble-body");
    if (body && !body.textContent.trim()) {
      body.innerHTML = escapeAndBreak("Stopped. Partial work kept.");
    }
  }
}

function appendActivity(entry) {
  // Streamlined: activity echoes into the reasoning feed; keep list only if element exists.
  const payload = typeof entry === "string" ? { text: entry } : entry || {};
  if (!payload.text) return;
  pushReasoning(payload.text, payload.tone || "info", payload.lane || "orch", payload.urls || []);
  const feed = el("activityFeed");
  if (!feed) return;
  const div = document.createElement("div");
  div.className = "event";
  div.textContent = payload.text;
  feed.appendChild(div);
  feed.scrollTop = feed.scrollHeight;
}

function renderAttachments() {
  const strip = el("attachmentStrip");
  const list = el("attachmentList");
  if (!list || !strip) return;
  list.innerHTML = "";
  const hasUploads = pendingUploads.length > 0;
  list.classList.toggle("hidden", !hasUploads);
  if (hasUploads) {
    strip.classList.remove("hidden");
    const show = el("showUploadDrop");
    if (show) show.classList.add("hidden");
    const drop = el("uploadDrop");
    if (drop) drop.classList.remove("hidden");
  }
  if (!pendingUploads.length) {
    hideUploadPanelIfIdle();
    return;
  }
  pendingUploads.forEach((u) => {
    const row = document.createElement("div");
    row.className = `attachment-chip ${u.status || ""}`;
    const thumb = document.createElement("div");
    thumb.className = "thumb";
    thumb.textContent =
      (u.mime && u.mime.startsWith("image/")) || (u.name || "").match(/\.(png|jpe?g|gif|webp)$/i) ? "IMG" : "FILE";
    const meta = document.createElement("div");
    meta.className = "meta";
    const name = document.createElement("div");
    name.className = "name";
    name.textContent = u.name;
    const size = document.createElement("div");
    size.className = "size";
    size.textContent = formatBytes(u.size || 0);
    meta.appendChild(name);
    meta.appendChild(size);
    const status = document.createElement("div");
    status.className = "status";
    status.textContent =
      u.status === "uploading"
        ? "Uploading..."
        : u.status === "ready"
        ? "Ready"
        : u.status === "processed"
        ? "Processed"
        : u.status === "failed"
        ? "Failed"
        : u.status === "queued"
        ? "Queued"
        : u.status;
    meta.appendChild(status);
    const actions = document.createElement("div");
    actions.className = "actions";
    if (!currentRunId || u.status === "ready" || u.status === "failed" || u.status === "queued") {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "ghost icon-btn small";
      btn.textContent = "x";
      btn.title = "Remove";
      btn.onclick = () => removeUpload(u.localId || u.serverId);
      actions.appendChild(btn);
    }
    row.appendChild(thumb);
    row.appendChild(meta);
    row.appendChild(actions);
    if (u.summary) {
      const summary = document.createElement("div");
      summary.className = "summary";
      summary.innerHTML = escapeAndBreak(u.summary);
      row.appendChild(summary);
    }
    list.appendChild(row);
  });
  hideUploadPanelIfIdle();
}

function hideUploadPanelIfIdle() {
  const strip = el("attachmentStrip");
  const show = el("showUploadDrop");
  const list = el("attachmentList");
  const shouldHide = !pendingUploads.length && !draggingUploads && !uploadPanelVisible;
  const drop = el("uploadDrop");
  if (strip) strip.classList.remove("hidden");
  if (drop) drop.classList.toggle("hidden", shouldHide);
  if (show) show.classList.toggle("hidden", !shouldHide);
  if (list) list.classList.toggle("hidden", !pendingUploads.length);
}

function showUploadPanel() {
  uploadPanelVisible = true;
  const strip = el("attachmentStrip");
  const show = el("showUploadDrop");
  if (strip) strip.classList.remove("hidden");
  if (show) show.classList.add("hidden");
  const drop = el("uploadDrop");
  if (drop) drop.classList.remove("hidden");
}

function removeUpload(id) {
  pendingUploads = pendingUploads.filter((u) => (u.localId || u.serverId) !== id);
  renderAttachments();
}

function setUploadStatus(serverId, status, summary) {
  const target = pendingUploads.find((u) => u.serverId === serverId || u.id === serverId || u.localId === serverId);
  if (!target) return;
  target.status = status;
  if (summary) target.summary = summary;
  renderAttachments();
}

function syncUploadsFromServer(list = []) {
  const keepLocals = pendingUploads.filter((u) => !u.serverId);
  const mapped = (list || []).map((u) => ({
    id: u.id,
    serverId: u.id,
    localId: `srv-${u.id}`,
    name: u.original_name || u.filename || `Upload ${u.id}`,
    size: u.size_bytes,
    status: u.status || "processed",
    summary: u.summary_text || "",
    mime: u.mime,
  }));
  pendingUploads = [...keepLocals, ...mapped];
  renderAttachments();
}

async function uploadFiles(fileList) {
  if (!fileList || !fileList.length) return;
  for (const file of fileList) {
    const localId = `local-${Date.now()}-${Math.random().toString(16).slice(2)}`;
    const entry = {
      localId,
      name: file.name,
      size: file.size,
      status: "uploading",
      serverId: null,
      mime: file.type,
      summary: "",
    };
    pendingUploads.push(entry);
    renderAttachments();
    const form = new FormData();
    form.append("file", file);
    try {
      const res = await fetch("/api/uploads", { method: "POST", body: form });
      if (!res.ok) {
        let detail = "Upload failed";
        try {
          const err = await res.json();
          if (err?.detail) detail = err.detail;
        } catch (_) {}
        throw new Error(detail);
      }
      const data = await res.json();
      entry.serverId = data.id;
      entry.status = "ready";
      entry.size = data.size || entry.size;
      entry.mime = data.mime || entry.mime;
    } catch (err) {
      entry.status = "failed";
      entry.summary = err.message || "Upload failed";
    }
    renderAttachments();
  }
}

function handleFileInput(files) {
  if (!files || !files.length) return;
  uploadPanelVisible = true;
  showUploadPanel();
  uploadFiles(files);
  const fileInput = el("fileInput");
  if (fileInput) fileInput.value = "";
}

const SEVERITY_RANK = { low: 0, medium: 1, high: 2 };

function resetLiveStreamState(question = "") {
  liveEventBuffer = [];
  clearTimeout(liveFlushTimer);
  liveFlushTimer = null;
  pendingQuestion = question;
  questionShownInLive = false;
}

function eventSeverity(type) {
  if (["archived", "error", "tavily_error"].includes(type)) return "high";
  if (
    [
      "run_started",
      "router_decision",
      "plan_created",
      "step_completed",
      "control_action",
      "loop_iteration",
      "strict_mode",
      "client_note",
      "step_error",
      "upload_received",
      "upload_processed",
      "upload_failed",
    ].includes(type)
  ) {
    return "medium";
  }
  return "low";
}

function condenseList(list, limit = 2) {
  const uniq = Array.from(new Set((list || []).filter(Boolean)));
  if (!uniq.length) return "";
  if (uniq.length <= limit) return uniq.join(", ");
  const tail = uniq.slice(-limit);
  return `${tail.join(", ")} (+${uniq.length - limit} more)`;
}

function queueLiveEvent(type, detail = {}, lane = "orch") {
  const severity = eventSeverity(type);
  const urls = [];
  if (detail.url) urls.push(detail.url);
  if (Array.isArray(detail.urls)) urls.push(...detail.urls);
  liveEventBuffer.push({ type, detail, lane, severity, urls });
  if (SEVERITY_RANK[severity] >= SEVERITY_RANK.medium) {
    flushLiveEvents();
  } else {
    scheduleLiveFlush();
  }
}

function queueLiveNote(note, lane = "orch") {
  if (!note) return;
  queueLiveEvent("client_note", { note }, lane);
}

function scheduleLiveFlush() {
  if (liveFlushTimer) return;
  liveFlushTimer = setTimeout(() => flushLiveEvents(), 2000);
}

function summarizeLiveEvents(events) {
  const stats = {
    searches: 0,
    searchQueries: [],
    extracts: 0,
    stepsStarted: [],
    stepsCompleted: [],
    memHits: 0,
    memSaved: 0,
    controls: [],
    loops: 0,
    router: null,
    planSteps: null,
    strict: false,
    finished: null,
    errors: [],
    warnings: [],
    notes: [],
    question: "",
    urls: [],
    uploads: [],
  };
  let lane = "orch";
  (events || []).forEach((ev) => {
    if (ev.lane) lane = ev.lane;
    if (Array.isArray(ev.urls)) stats.urls.push(...ev.urls);
    const d = ev.detail || {};
    switch (ev.type) {
      case "run_started":
        stats.question = d.question || stats.question || pendingQuestion;
        break;
      case "router_decision":
        stats.router = d;
        break;
      case "plan_created":
        stats.planSteps = Number(d.steps || stats.planSteps || 0);
        break;
      case "step_started":
        stats.stepsStarted.push(d.name || (d.step_id ? `Step ${d.step_id}` : "Step started"));
        break;
      case "step_completed":
        stats.stepsCompleted.push(d.name || (d.step_id ? `Step ${d.step_id}` : "Step completed"));
        break;
      case "tavily_search":
        stats.searches += 1;
        if (d.query) stats.searchQueries.push(d.query);
        break;
      case "tavily_extract":
        stats.extracts += d.urls && d.urls.length ? d.urls.length : 1;
        break;
      case "tavily_error":
        stats.errors.push(d.message || "Tavily search unavailable");
        break;
      case "memory_retrieved":
        stats.memHits += Number(d.count || 0);
        break;
      case "memory_saved":
        stats.memSaved += Number(d.count || 0);
        break;
      case "control_action":
        if (d.control || d.action_type) stats.controls.push(d.control || d.action_type);
        break;
      case "loop_iteration":
        stats.loops += 1;
        break;
      case "strict_mode":
        stats.strict = true;
        break;
      case "archived":
        stats.finished = d;
        break;
      case "error":
        if (d.message) stats.errors.push(d.message);
        break;
      case "step_error": {
        const label = d.name || (d.step ? `Step ${d.step}` : "Step");
        const msg = d.message || "error";
        stats.warnings.push(`${label}: ${msg}`);
        break;
      }
      case "upload_received":
        stats.uploads.push(d.name || `Upload ${d.upload_id || ""}`);
        break;
      case "upload_processed":
        stats.uploads.push(d.name || `Upload ${d.upload_id || ""}`);
        if (d.summary) stats.notes.push(`Upload: ${d.summary}`);
        break;
      case "upload_failed":
        stats.errors.push(d.error || `Upload failed: ${d.name || ""}`);
        break;
      default:
        break;
    }
    if (d.note && !stats.notes.includes(d.note)) {
      stats.notes.push(d.note);
    }
  });
  if (!stats.question && pendingQuestion) {
    stats.question = pendingQuestion;
  }
  const hasErrors = stats.errors.length > 0;
  const hasWarnings = stats.warnings.length > 0;
  let tone = hasErrors ? "error" : hasWarnings ? "warn" : "info";
  const snippets = [];
  const usedQuestion = !!(stats.question && !questionShownInLive);
  if (usedQuestion) snippets.push(`Plan: ${stats.question}`);
  if (stats.router) {
    const r = stats.router;
    const tier = r.model_tier ? tierLabel(r.model_tier) : "Router";
    const web = r.needs_web ? " + web" : "";
    const max = r.max_results ? ` (max ${r.max_results})` : "";
    const route = r.deep_route ? ` via ${r.deep_route}` : "";
    snippets.push(`Router chose ${tier} ${r.reasoning_level || ""}${web}${max}${route}`.trim());
  }
  if (stats.uploads.length) snippets.push(`Uploads: ${condenseList(stats.uploads, 3)}`);
  if (stats.strict) snippets.push("Strict verify on");
  if (stats.planSteps !== null && stats.planSteps !== undefined) snippets.push(`${stats.planSteps} steps planned`);
  const started = condenseList(stats.stepsStarted, 3);
  if (started) snippets.push(`Starting: ${started}`);
  const completed = condenseList(stats.stepsCompleted, 3);
  if (completed) snippets.push(`Finished: ${completed}`);
  if (stats.searches) {
    const qPreview = condenseList(stats.searchQueries, 2);
    snippets.push(qPreview ? `Searching ${qPreview}` : `Searching (${stats.searches} queries)`);
  }
  if (stats.extracts) snippets.push(`Digesting ${stats.extracts} source${stats.extracts > 1 ? "s" : ""}`);
  if (stats.memHits) snippets.push(`Memory hits: ${stats.memHits}`);
  if (stats.memSaved) snippets.push(`Saved to memory (${stats.memSaved})`);
  if (stats.loops) snippets.push(`Verifier rerun x${stats.loops}`);
  if (stats.controls.length) snippets.push(`Control checks: ${stats.controls.join(", ")}`);
  if (stats.finished) snippets.push(`Wrap-up (confidence ${stats.finished.confidence || "n/a"})`);
  if (stats.errors.length) {
    snippets.push(`Issue: ${stats.errors[0]}`);
    tone = "error";
  } else if (stats.warnings.length) {
    snippets.push(`Check: ${stats.warnings[stats.warnings.length - 1]}`);
  } else if (stats.notes.length) {
    snippets.push(stats.notes.slice(-2).join(" | "));
  }
  if (!snippets.length) {
    snippets.push(`Activity (${events.length} update${events.length === 1 ? "" : "s"})`);
  }
  const text = snippets.join(" · ");
  const urls = Array.from(new Set(stats.urls)).slice(0, 5);
  return { text, tone, lane, urls, usedQuestion };
}

function flushLiveEvents() {
  clearTimeout(liveFlushTimer);
  liveFlushTimer = null;
  if (!liveEventBuffer.length) return;
  const summary = summarizeLiveEvents(liveEventBuffer);
  liveEventBuffer = [];
  if (summary.usedQuestion) questionShownInLive = true;
  appendActivity(summary);
}

function laneFrom(name = "", stepId) {
  const lowered = name.toLowerCase();
  if (lowered.includes("primary")) return "primary";
  if (lowered.includes("recency")) return "recency";
  if (lowered.includes("adversarial")) return "adversarial";
  if (lowered.includes("router")) return "router";
  if (lowered.includes("verify")) return "verifier";
  if (lowered.includes("draft") || lowered.includes("merge")) return "orch";
  if (stepId) {
    const lanes = ["primary", "recency", "adversarial"];
    return lanes[(stepId - 1) % lanes.length];
  }
  return "orch";
}

function updateLiveTicker(text, urls = []) {
  const ticker = el("liveTicker");
  if (!ticker) return;
  const links = urls && urls.length ? " | " + urls.slice(0, 3).map((u) => `<a href="${u}" target="_blank">${u}</a>`).join(", ") : "";
  ticker.innerHTML = escapeAndBreak(text) + links;
  ticker.classList.remove("pulse");
  void ticker.offsetWidth;
  ticker.classList.add("pulse");
}

function renderHistory(runId = selectedRunId || currentRunId) {
  const feed = el("reasoningFeed");
  if (!feed) return;
  feed.innerHTML = "";
  const log = runId
    ? runEvents[runId] || []
    : historyLog;
  (log || []).forEach((h) => {
    const row = document.createElement("div");
    row.className = `thought ${h.tone || "info"}`;
    const laneTag = document.createElement("span");
    laneTag.className = "lane";
    laneTag.textContent = (h.lane || "orch").toUpperCase();
    const body = document.createElement("div");
    const links =
      h.urls && h.urls.length
        ? "<br><small>" + h.urls.slice(0, 5).map((u) => `<a href="${u}" target="_blank">${u}</a>`).join(", ") + "</small>"
        : "";
    body.innerHTML = escapeAndBreak(h.text) + links;
    row.appendChild(laneTag);
    row.appendChild(body);
    feed.appendChild(row);
  });
  feed.scrollTop = feed.scrollHeight;
}

function addHistory(text, lane = "orch", urls = [], tone = "info") {
  historyLog.push({ text, lane, urls, tone });
  if (historyLog.length > 150) historyLog.shift();
  if (currentRunId) {
    if (!runEvents[currentRunId]) runEvents[currentRunId] = [];
    runEvents[currentRunId].push({ text, lane, urls, tone });
    if (runEvents[currentRunId].length > 200) runEvents[currentRunId].shift();
  }
  updateLiveTicker(text, urls);
  renderHistory();
}

function pushReasoning(text, tone = "info", lane = "orch", urls = []) {
  addHistory(text, lane, urls, tone);
}

async function loadSettings() {
  const res = await fetch("/settings");
  const data = await res.json();
  const s = data.settings;
  settingsDefaults = {
    search_depth_mode: s.search_depth_mode || "auto",
    strict_mode: !!s.strict_mode,
    auto_memory: true,
    evidence_dump: false,
    max_results_override: 0,
    stt_lang: "en-US",
  };
  el("cfgBaseUrl").value = s.lm_studio_base_url || "";
  el("cfgOrchBase").value = s.orch_endpoint.base_url || "";
  el("cfgOrch").value = s.orch_endpoint.model_id || "";
  el("cfgWorkerABase").value = s.worker_a_endpoint.base_url || "";
  el("cfgQwen8").value = s.worker_a_endpoint.model_id || "";
  el("cfgWorkerBBase").value = s.worker_b_endpoint.base_url || "";
  el("cfgQwen8B").value = s.worker_b_endpoint.model_id || "";
  el("cfgWorkerCBase").value = s.worker_c_endpoint.base_url || "";
  el("cfgQwen8C").value = s.worker_c_endpoint.model_id || "";
  el("cfgFastBase").value = s.fast_endpoint?.base_url || s.worker_a_endpoint.base_url || "";
  el("cfgFastModel").value = s.fast_endpoint?.model_id || s.worker_a_endpoint.model_id || "";
  el("cfgDeepPlannerBase").value = s.deep_planner_endpoint?.base_url || s.worker_a_endpoint.base_url || "";
  el("cfgDeepPlanner").value = s.deep_planner_endpoint?.model_id || s.worker_a_endpoint.model_id || "";
  el("cfgDeepOrchBase").value = s.deep_orchestrator_endpoint?.base_url || s.router_endpoint.base_url || "";
  el("cfgDeepOrch").value = s.deep_orchestrator_endpoint?.model_id || s.router_endpoint.model_id || "";
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
  const modelWarning = el("modelWarning");
  if (modelWarning) {
    modelWarning.textContent = "";
    modelWarning.classList.add("hidden");
  }
  let missingRoles = [];
  if (data.model_check) {
    const bad = Object.entries(data.model_check).filter(([, v]) => v.ok === false);
    if (bad.length) {
      missingRoles = bad.map(([r, v]) => `${r}:${(v.missing || []).join(",") || v.error || ""}`);
      if (modelWarning) {
        modelWarning.textContent = `Model issues: ${missingRoles.join(", ")}`;
        modelWarning.title = modelWarning.textContent;
        modelWarning.classList.remove("hidden");
      }
    }
  }
  // Auto-fill defaults if core fields are empty to smooth setup.
  if (!s.orch_endpoint.model_id || !s.router_endpoint.model_id) {
    applyRecommended();
    el("settingsStatus").textContent = "Auto-filled recommended defaults (not saved yet).";
  }
  // Auto-discover if models are missing and we haven't tried yet.
  const baseUrls = el("cfgDiscovery").value.split(",").map((v) => v.trim()).filter(Boolean);
  if (!triedAutoDiscover && baseUrls.length && missingRoles.length) {
    triedAutoDiscover = true;
    el("settingsStatus").textContent = "Auto-discovering available models...";
    await autoDiscoverAndReport(baseUrls);
  }
  toggleModal(false);
}

function toggleModal(show) {
  const modal = el("settingsModal");
  modal.classList[show ? "remove" : "add"]("hidden");
}

async function saveSettings() {
  const tavKey = el("cfgTavily").value;
  const payload = {
    lm_studio_base_url: el("cfgBaseUrl").value,
    orch_endpoint: { base_url: el("cfgOrchBase").value, model_id: el("cfgOrch").value },
    worker_a_endpoint: { base_url: el("cfgWorkerABase").value, model_id: el("cfgQwen8").value },
    worker_b_endpoint: { base_url: el("cfgWorkerBBase").value, model_id: el("cfgQwen8B").value },
    worker_c_endpoint: { base_url: el("cfgWorkerCBase").value, model_id: el("cfgQwen8C").value },
    fast_endpoint: { base_url: el("cfgFastBase").value, model_id: el("cfgFastModel").value },
    deep_planner_endpoint: { base_url: el("cfgDeepPlannerBase").value, model_id: el("cfgDeepPlanner").value },
    deep_orchestrator_endpoint: { base_url: el("cfgDeepOrchBase").value, model_id: el("cfgDeepOrch").value },
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

async function autoDiscoverAndReport(base_urls) {
  try {
    const res = await fetch("/api/discover", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ base_urls }),
    });
    const data = await res.json();
    const summary = Object.entries(data.results || {})
      .map(([url, val]) => `${url}: ${val.ok ? (val.models || []).slice(0, 3).join("|") : "error"}`)
      .join(" | ");
    el("settingsStatus").textContent = `Discovery: ${summary}`;
    if (!el("cfgDiscovery").value) {
      el("cfgDiscovery").value = base_urls.join(", ");
    }
  } catch (err) {
    el("settingsStatus").textContent = "Discovery failed.";
  }
}

function applyRecommended() {
  el("cfgBaseUrl").value = "http://127.0.0.1:1234/v1";
  el("cfgOrchBase").value = "http://127.0.0.1:1234/v1";
  el("cfgOrch").value = "openai/gpt-oss-20b";
  el("cfgWorkerABase").value = "http://127.0.0.1:1234/v1";
  el("cfgQwen8").value = "qwen/qwen3-vl-8b";
  el("cfgWorkerBBase").value = "http://127.0.0.1:1234/v1";
  el("cfgQwen8B").value = "qwen/qwen3-vl-8b:2";
  el("cfgWorkerCBase").value = "http://127.0.0.1:1234/v1";
  el("cfgQwen8C").value = "qwen/qwen3-vl-8b:3";
  el("cfgRouterBase").value = "http://127.0.0.1:1234/v1";
  el("cfgQwen4").value = "qwen/qwen3-vl-4b";
  el("cfgSummarizerBase").value = "http://127.0.0.1:1234/v1";
  el("cfgSummarizer").value = "qwen/qwen3-vl-4b";
  el("cfgVerifierBase").value = "http://127.0.0.1:1234/v1";
  el("cfgVerifier").value = "qwen/qwen3-vl-8b";
  el("cfgFastBase").value = "http://127.0.0.1:1234/v1";
  el("cfgFastModel").value = "qwen/qwen3-vl-8b";
  el("cfgDeepPlannerBase").value = "http://127.0.0.1:1234/v1";
  el("cfgDeepPlanner").value = "qwen/qwen3-vl-8b";
  el("cfgDeepOrchBase").value = "http://127.0.0.1:1234/v1";
  el("cfgDeepOrch").value = "qwen/qwen3-vl-4b";
  el("cfgDiscovery").value = "http://127.0.0.1:1234/v1";
  el("settingsStatus").textContent = "Filled with recommended defaults.";
}

const REASONING_BY_TIER = {
  fast: [{ value: "LOW", label: "Fast (8B linear)" }],
  deep: [{ value: "AUTO", label: "Auto (router)" }],
  auto: [{ value: "AUTO", label: "Auto (router)" }],
  pro: [
    { value: "AUTO", label: "Auto" },
    { value: "LOW", label: "Quick" },
    { value: "MED", label: "Balanced" },
    { value: "HIGH", label: "Deep" },
    { value: "ULTRA", label: "Ultra" },
  ],
};

function renderReasoningOptions(tier) {
  const select = el("reasoningLevel");
  const opts = REASONING_BY_TIER[tier] || REASONING_BY_TIER.pro;
  select.innerHTML = "";
  opts.forEach((o, idx) => {
    const opt = document.createElement("option");
    opt.value = o.value;
    opt.textContent = o.label;
    if (idx === 0) opt.selected = true;
    select.appendChild(opt);
  });
  select.disabled = tier === "fast" || tier === "deep" || tier === "auto";
}

function deepRouteLabel(route) {
  if (!route) return "LocalDeep";
  if (route === "auto") return "Auto lane";
  if (route === "cluster") return "Mini Pro";
  if (route === "oss") return "OSS linear";
  return route;
}

function tierLabel(tier) {
  if (tier === "fast") return "LocalFast";
  if (tier === "deep") return "LocalDeep";
  if (tier === "auto") return "LocalAuto";
  return "LocalPro";
}

function setTier(tier) {
  currentTier = tier;
  document.querySelectorAll("#modelTierGroup .seg-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.tier === tier);
  });
  const deepWrap = el("deepRouteWrap");
  if (deepWrap) deepWrap.classList.toggle("hidden", tier !== "deep");
  const reasoningControl = el("reasoningControl");
  if (reasoningControl) {
    const hideReasoningControl = tier === "deep" || tier === "auto";
    reasoningControl.classList.toggle("hidden", hideReasoningControl);
  }
  renderReasoningOptions(tier);
  updateReasoningBadge();
}

function updateReasoningBadge() {
  const badge = el("reasoningBadge");
  if (!badge) return;
  const select = el("reasoningLevel");
  if (currentTier === "deep") {
    const route = el("deepRoute");
    const routeText = route?.selectedOptions[0]?.textContent || route?.value || "Lane";
    badge.textContent = `${tierLabel(currentTier)} - ${routeText}`;
    return;
  }
  const selectedText = select?.selectedOptions[0]?.textContent || select?.value || "";
  badge.textContent = `${tierLabel(currentTier)} - ${selectedText}`;
}

function updateProgressUI() {
  const pct = totalSteps > 0 ? Math.min(100, Math.round((completedSteps / totalSteps) * 100)) : 0;
  const progressText = el("progressText");
  const fill = el("runProgressFill");
  if (progressText) progressText.textContent = `Progress: ${pct}%`;
  if (fill) fill.style.width = `${pct}%`;
  const etaEl = el("etaText");
  if (etaEl) {
    if (timerStartedAt && totalSteps > 0 && completedSteps > 0) {
      const elapsed = Date.now() - timerStartedAt;
      const remaining = Math.max(totalSteps - completedSteps, 0);
      const perStep = elapsed / Math.max(completedSteps, 1);
      const etaMs = perStep * remaining;
      etaTargetAt = Date.now() + etaMs;
      etaEl.textContent = `ETA: ${formatTime(etaMs)}`;
    } else if (totalSteps > 0) {
      etaTargetAt = null;
      etaEl.textContent = "ETA: ...";
    } else {
      etaTargetAt = null;
      etaEl.textContent = "ETA: --";
    }
  }
}

async function startRun(evt) {
  evt.preventDefault();
  const question = el("question").value.trim();
  if (!question) return;
  const uploading = pendingUploads.some((u) => u.status === "uploading");
  if (uploading) {
    appendActivity("Wait for uploads to finish before starting the run.");
    return;
  }
  startNewConversation({ keepQuestion: true, silent: true, keepUploads: true, preserveChat: true });
  const selected = el("reasoningLevel").value;
  const deepRoute = (el("deepRoute") && el("deepRoute").value) || "auto";
  let reasoningMode = "manual";
  let manualLevel = "LOW";
  if (currentTier === "fast") {
    reasoningMode = "manual";
    manualLevel = "LOW";
  } else if (currentTier === "auto") {
    reasoningMode = "auto";
    manualLevel = "MED";
  } else {
    const reasoningAuto = selected === "AUTO";
    reasoningMode = reasoningAuto ? "auto" : "manual";
    manualLevel = reasoningAuto ? (currentTier === "deep" ? "HIGH" : "MED") : selected;
  }
  const reasoningAuto = reasoningMode === "auto";
  appendChat("user", question);
  setStatus("Queued", "live");
  el("confidence").textContent = "";
  el("sources").textContent = "";
  cachedArtifacts = null;
  historyLog = [];
  renderHistory();
  resetLiveStreamState(question);
  updateLiveTicker("Planning...");
  const activity = el("activityFeed");
  if (activity) activity.innerHTML = "";
  totalSteps = 0;
  completedSteps = 0;
  updateProgressUI();
  queueLiveNote("Queued. Planner warming up...", "orch");
  updateReasoningBadge();

  const payload = {
    question,
    reasoning_mode: reasoningMode,
    manual_level: manualLevel,
    model_tier: currentTier,
    deep_mode: currentTier === "deep" ? deepRoute : "auto",
    evidence_dump: settingsDefaults.evidence_dump,
    search_depth_mode: settingsDefaults.search_depth_mode,
    max_results: settingsDefaults.max_results_override || 0,
    strict_mode: settingsDefaults.strict_mode,
    auto_memory: settingsDefaults.auto_memory,
    reasoning_auto: reasoningAuto,
    upload_ids: pendingUploads.filter((u) => u.serverId && u.status !== "failed").map((u) => u.serverId),
  };
  pendingUploads.forEach((u) => {
    if (u.serverId && u.status === "ready") u.status = "queued";
  });
  renderAttachments();

  const res = await fetch("/api/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    appendActivity("Failed to start run.");
    setStatus("Error", "error");
    stopTimer("--");
    return;
  }
  const data = await res.json();
  await switchToRun(data.run_id, { clearChat: false, force: true, resetState: false });
  el("question").value = "";
  updateCharCount();
}

function handleEvent(type, p) {
  switch (type) {
    case "message_added": {
      const role = p.role === "assistant" ? "assistant" : "user";
      if (role === "assistant") {
        const envelope = buildAnswerEnvelope({
          run: { run_id: p.run_id, final_answer: p.content || "", confidence: p.confidence || "" },
          sources: [],
          claims: [],
        });
        renderAssistantAnswer(p.run_id, envelope);
      } else {
        appendChat("user", p.content || "", {
          messageId: p.id,
          runId: p.run_id,
        });
      }
      break;
    }
    case "run_started":
      setStatus("Thinking", "live");
      startTimer();
      totalSteps = 0;
      completedSteps = 0;
      updateProgressUI();
      ensureThinkingPlaceholder(currentRunId || p.run_id);
      queueLiveEvent("run_started", { question: p.question }, "orch");
      break;
    case "router_decision": {
      const requestedTier = p.requested_tier || currentTier;
      const actualTier = p.model_tier || requestedTier;
      const displayTier =
        requestedTier !== actualTier
          ? `${tierLabel(requestedTier)}->${tierLabel(actualTier)}`
          : tierLabel(actualTier);
      let detailText = p.reasoning_level || "AUTO";
      if (actualTier === "deep" && p.deep_route) {
        detailText = deepRouteLabel(p.deep_route);
      }
      el("reasoningBadge").textContent = `${displayTier} - ${detailText}`;
      queueLiveEvent("router_decision", p, "router");
      break;
    }
    case "strict_mode":
      queueLiveEvent("strict_mode", p, "verifier");
      break;
    case "plan_created":
      totalSteps = Number(p.expected_total_steps || p.steps || 0);
      completedSteps = Number(p.completed_reset_to || 0);
      updateProgressUI();
      queueLiveEvent("plan_created", p, "orch");
      break;
    case "upload_received":
      setUploadStatus(p.upload_id, "ready");
      queueLiveEvent("upload_received", p, "orch");
      break;
    case "upload_processed":
      setUploadStatus(p.upload_id, "processed", p.summary);
      queueLiveEvent("upload_processed", p, "orch");
      break;
    case "upload_failed":
      setUploadStatus(p.upload_id, "failed", p.error);
      queueLiveEvent("upload_failed", p, "orch");
      break;
    case "step_started":
      queueLiveEvent("step_started", { step_id: p.step_id, name: p.name }, laneFrom(p.name, p.step_id));
      break;
    case "tavily_search":
      queueLiveEvent("tavily_search", { query: p.query }, laneFrom("", p.step));
      break;
    case "tavily_extract":
      queueLiveEvent("tavily_extract", { urls: p.urls || [] }, laneFrom("", p.step));
      break;
    case "tavily_error": {
      const lane = laneFrom("", p.step);
      queueLiveEvent("tavily_error", { message: p.message || "Tavily unavailable" }, lane);
      pushReasoning(`Search helper issue: ${p.message || "Tavily unavailable"}`, "warn", lane);
      break;
    }
    case "step_completed":
      completedSteps = Math.min(totalSteps || completedSteps + 1, completedSteps + 1);
      updateProgressUI();
      if (currentRunId) {
        fetchArtifacts(currentRunId, { liveOnly: true, skipChat: true }).catch(() => {});
      }
      queueLiveEvent("step_completed", { step_id: p.step_id, name: p.name }, laneFrom(p.name, p.step_id));
      break;
    case "control_action":
      queueLiveEvent("control_action", { control: p.control || p.action_type || "" }, "orch");
      break;
    case "loop_iteration":
      if (typeof p.expected_total_steps === "number") {
        totalSteps = Number(p.expected_total_steps) || totalSteps;
      }
      if (typeof p.completed_reset_to === "number") {
        completedSteps = Number(p.completed_reset_to);
      }
      updateProgressUI();
      queueLiveEvent("loop_iteration", { iteration: p.iteration }, "verifier");
      break;
    case "memory_retrieved":
      queueLiveEvent("memory_retrieved", { count: p.count }, "memory");
      break;
    case "memory_saved":
      queueLiveEvent("memory_saved", { count: p.count }, "memory");
      break;
    case "archived":
      setStatus("Done", "done");
      stopTimer();
      completedSteps = totalSteps || completedSteps;
      updateProgressUI();
      fetchArtifacts(currentRunId);
      queueLiveEvent("archived", p, "orch");
      break;
    case "step_error": {
      const safe = p || {};
      const lane = laneFrom(safe.name || "", safe.step);
      const label = safe.name || (safe.step ? `Step ${safe.step}` : "Step");
      const msg = safe.message || "error encountered";
      queueLiveEvent("step_error", { step_id: safe.step, name: safe.name, message: msg }, lane);
      pushReasoning(`${label} hit an error: ${msg} (continuing)`, "warn", lane);
      completedSteps = Math.min(totalSteps || completedSteps + 1, completedSteps + 1);
      updateProgressUI();
      break;
    }
    case "error": {
      const safe = p || {};
      const recoverable = safe.fatal === false || typeof safe.step !== "undefined";
      const lane = laneFrom(safe.name || "", safe.step);
      if (recoverable) {
        queueLiveEvent(
          "step_error",
          { step_id: safe.step, name: safe.name, message: safe.message || "recoverable error" },
          lane
        );
        pushReasoning(
          `Recoverable error${safe.step ? ` on step ${safe.step}` : ""}: ${safe.message || "continuing"}`,
          "warn",
          lane
        );
        completedSteps = Math.min(totalSteps || completedSteps + 1, completedSteps + 1);
        updateProgressUI();
        break;
      }
      setStatus("Error", "error");
      stopTimer("error");
      updateProgressUI();
      queueLiveEvent("error", safe, "orch");
      break;
    }
    default:
      queueLiveEvent(type, p, "orch");
  }
}

function subscribeEvents(runId) {
  if (evtSource) evtSource.close();
  evtSource = new EventSource(`/runs/${runId}/events`);
  evtSource.onmessage = async (evt) => {
    const data = JSON.parse(evt.data);
    handleEvent(data.event_type, data.payload || {});
  };
}

async function followLatestRun(force = false) {
  try {
    const res = await fetch("/api/run/latest");
    if (!res.ok) return;
    const data = await res.json();
    const incomingReset = data.reset_at || null;
    if (incomingReset && incomingReset !== conversationResetAt) {
      conversationResetAt = incomingReset;
      startNewConversation({ keepQuestion: true, silent: true, keepUploads: false, preserveChat: false });
      syncConversationHistory();
    } else if (incomingReset) {
      conversationResetAt = incomingReset;
    }
    const latestId = data?.run?.run_id || null;
    if (!latestId) return;
    if (force || latestId !== currentRunId) {
      await switchToRun(latestId, { clearChat: false, fromPoll: true });
    }
  } catch (err) {
    // ignore polling errors
  }
}

function renderEvidence(claims) {
  // Evidence is now summarized into the reasoning stream; no separate panel.
  if (!claims || !claims.length) return;
  const sample = claims.slice(0, 3).map((c) => c.claim || "").join(" | ");
  pushReasoning(`Claims captured (${claims.length}): ${sample}`, "info", "web");
}

function hostFromUrl(url) {
  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch {
    return url;
  }
}

function summarizeFindings(data) {
  if (!data) return null;
  const sources = data.sources || [];
  const claims = data.claims || [];
  const topHosts = Array.from(new Set(sources.map((s) => hostFromUrl(s.url)).filter(Boolean))).slice(0, 3);
  const textParts = [];
  textParts.push(`${sources.length} sources, ${claims.length} claims so far`);
  if (topHosts.length) textParts.push(`key hosts: ${topHosts.join(", ")}`);
  return { text: `Findings: ${textParts.join(" | ")}`, urls: sources.slice(-3).map((s) => s.url) };
}

function tryParseJson(text) {
  try {
    return JSON.parse(text);
  } catch (_) {
    return null;
  }
}

function extractFinalText(raw) {
  if (raw === null || raw === undefined) return "";
  if (typeof raw === "string") {
    const trimmed = raw.trim();
    if ((trimmed.startsWith("{") && trimmed.endsWith("}")) || (trimmed.startsWith("[") && trimmed.endsWith("]"))) {
      const parsed = tryParseJson(trimmed);
      if (parsed) return extractFinalText(parsed) || "";
      return "";
    }
    return trimmed;
  }
  if (typeof raw === "object") {
    return (
      raw.final_text ||
      raw.draft_answer ||
      raw.answer ||
      raw.text ||
      raw.message ||
      raw.content ||
      ""
    );
  }
  return String(raw);
}

function buildReasoningSummary(run = {}, sources = [], claims = [], routerDecision = {}) {
  const items = [];
  const route = routerDecision.deep_route || routerDecision.route;
  if (routerDecision.reasoning_level) items.push(`Reasoning: ${routerDecision.reasoning_level}`);
  if (route) items.push(`Route: ${deepRouteLabel(route)}`);
  if (sources.length) items.push(`Cited ${sources.length} source${sources.length === 1 ? "" : "s"}`);
  if (claims.length) items.push(`Claims recorded: ${claims.length}`);
  if (run.confidence) items.push(`Confidence: ${run.confidence}`);
  return items.length ? items : ["Working on it..."];
}

function buildAnswerEnvelope(data = {}) {
  const run = data.run || {};
  const sources = data.sources || [];
  const claims = data.claims || [];
  const routerDecision = run.router_decision || {};
  const finalText = extractFinalText(run.final_answer) || extractFinalText(data.draft) || "";
  return {
    run_id: run.run_id,
    final_text: finalText,
    final_format: "markdown",
    message_meta: {
      used_model: routerDecision.model_tier ? tierLabel(routerDecision.model_tier) : tierLabel(currentTier),
      reasoning_level: routerDecision.reasoning_level || run.reasoning_mode || "AUTO",
      confidence: run.confidence || "",
    },
    reasoning_summary: buildReasoningSummary(run, sources, claims, routerDecision),
    activity_events: runEvents[run.run_id] || [],
    sources,
    visited_sources: data.visited_sources || [],
  };
}

function addMessageActions(bubble, runId, envelope) {
  if (!bubble) return;
  const existing = bubble.querySelector(".bubble-actions");
  if (existing) existing.remove();
  const actions = document.createElement("div");
  actions.className = "bubble-actions";
  const reasoningBtn = document.createElement("button");
  reasoningBtn.type = "button";
  reasoningBtn.className = "bubble-action";
  reasoningBtn.textContent = "Reasoning";
  reasoningBtn.onclick = () => openDrawer("reasoning", runId);
  const sourcesBtn = document.createElement("button");
  sourcesBtn.type = "button";
  sourcesBtn.className = "bubble-action";
  sourcesBtn.textContent = "Sources";
  sourcesBtn.onclick = () => openDrawer("sources", runId);
  const copyBtn = document.createElement("button");
  copyBtn.type = "button";
  copyBtn.className = "bubble-action";
  copyBtn.textContent = "Copy";
  copyBtn.onclick = async () => {
    try {
      await navigator.clipboard.writeText(envelope?.final_text || "");
    } catch (_) {}
  };
  actions.appendChild(reasoningBtn);
  actions.appendChild(sourcesBtn);
  actions.appendChild(copyBtn);
  bubble.appendChild(actions);
}

function renderAssistantAnswer(runId, envelope) {
  if (!runId || !envelope) return;
  runDetails[runId] = envelope;
  const displayText = envelope.final_text || "...";
  let bubble = getAssistantBubble(runId);
  const bodyHtml = escapeAndBreak(displayText);
  if (!bubble) {
    bubble = appendChat("assistant", "", { runId, html: bodyHtml, skipPending: true });
  } else {
    const body = bubble.querySelector(".bubble-body");
    if (body) body.innerHTML = bodyHtml;
    bubble.classList.remove("thinking");
  }
  bubble.dataset.state = "answered";
  addMessageActions(bubble, runId, envelope);
  updateEmptyState();
  if (!selectedRunId) {
    selectedRunId = runId;
  }
}

function getResponseIndex(runId) {
  if (!runId) return null;
  const assistants = Array.from(document.querySelectorAll('.chat-bubble[data-role="assistant"]'));
  const idx = assistants.findIndex((b) => b.dataset.runId === String(runId));
  return idx >= 0 ? idx + 1 : null;
}

function highlightSelectedMessage(runId) {
  document.querySelectorAll(".chat-bubble.bubble-assistant").forEach((b) => {
    b.classList.toggle("selected", runId && b.dataset.runId === String(runId));
  });
}

function isDrawerOpen() {
  const panel = el("reasoningPanel");
  return panel && panel.classList.contains("open");
}

function populateDrawer(runId = selectedRunId, tab = activeDrawerTab) {
  const panel = el("reasoningPanel");
  if (!panel || !runId) return;
  const envelope = runDetails[runId] || {};
  const sub = el("drawerSubline");
  const idx = getResponseIndex(runId);
  if (sub) sub.textContent = idx ? `For response #${idx}` : "";
  renderHistory(runId);
  const reasoningPanel = el("tabReasoning");
  if (reasoningPanel) {
    reasoningPanel.innerHTML = "";
    const list = envelope.reasoning_summary || [];
    if (!list.length) {
      reasoningPanel.innerHTML = '<p class="sub">No reasoning available for this response.</p>';
    } else {
      const ul = document.createElement("ul");
      ul.className = "thinking-steps";
      list.forEach((item, idx2) => {
        const li = document.createElement("li");
        li.innerHTML = `<span>•</span><span>${escapeAndBreak(item)}</span>`;
        if (idx2 === 0) li.style.fontWeight = "700";
        ul.appendChild(li);
      });
      reasoningPanel.appendChild(ul);
    }
  }
  const sourcesPanel = el("tabSources");
  if (sourcesPanel) {
    sourcesPanel.innerHTML = "";
    const cited = envelope.sources || [];
    const visited = envelope.visited_sources || [];
    const citedHeader = document.createElement("p");
    citedHeader.className = "label";
    citedHeader.textContent = "Cited sources";
    sourcesPanel.appendChild(citedHeader);
    if (!cited.length) {
      const empty = document.createElement("div");
      empty.className = "sub";
      empty.textContent = "No sources used. This answer was generated without browsing.";
      sourcesPanel.appendChild(empty);
    } else {
      const list = document.createElement("ul");
      list.className = "thinking-steps";
      cited.forEach((s, idx3) => {
        const li = document.createElement("li");
        const label = s.title || s.url || `Source ${idx3 + 1}`;
        li.innerHTML = `<span>[${idx3 + 1}]</span><span><a href="${s.url}" target="_blank">${label}</a></span>`;
        list.appendChild(li);
      });
      sourcesPanel.appendChild(list);
    }
    const visitedHeader = document.createElement("p");
    visitedHeader.className = "label";
    visitedHeader.textContent = "Visited / considered";
    sourcesPanel.appendChild(visitedHeader);
    if (!visited.length) {
      const emptyVisited = document.createElement("div");
      emptyVisited.className = "sub";
      emptyVisited.textContent = "No additional pages were visited.";
      sourcesPanel.appendChild(emptyVisited);
    } else {
      const list = document.createElement("ul");
      list.className = "thinking-steps";
      visited.slice(0, 8).forEach((s, idx4) => {
        const li = document.createElement("li");
        const label = s.title || s.url || `Visited ${idx4 + 1}`;
        li.innerHTML = `<span>•</span><span><a href="${s.url}" target="_blank">${label}</a></span>`;
        list.appendChild(li);
      });
      sourcesPanel.appendChild(list);
    }
  }
  activeDrawerTab = tab;
  switchDrawerTab(tab);
  highlightSelectedMessage(runId);
}

function switchDrawerTab(tab) {
  const panels = document.querySelectorAll(".tab-panel");
  const tabs = document.querySelectorAll(".drawer-tabs .tab-btn");
  const currentPanel = document.querySelector('.tab-panel:not(.hidden)');
  if (currentPanel) {
    tabScrollPositions[activeDrawerTab] = currentPanel.scrollTop;
  }
  tabs.forEach((btn) => {
    const isActive = btn.dataset.tab === tab;
    btn.classList.toggle("active", isActive);
    btn.setAttribute("aria-selected", isActive ? "true" : "false");
  });
  panels.forEach((panel) => {
    const isMatch = panel.dataset.tab === tab;
    panel.classList.toggle("hidden", !isMatch);
    if (isMatch && tabScrollPositions[tab] && typeof tabScrollPositions[tab] === "number") {
      panel.scrollTop = tabScrollPositions[tab];
    }
  });
  activeDrawerTab = tab;
}

function openDrawer(tab = "activity", runId = null) {
  const panel = el("reasoningPanel");
  const overlay = el("drawerOverlay");
  if (!panel || !overlay) return;
  const targetRun = runId || selectedRunId || lastAssistantRunId || currentRunId;
  selectedRunId = targetRun;
  panel.classList.add("open");
  panel.classList.remove("hidden");
  overlay.classList.remove("hidden");
  switchDrawerTab(tab);
  populateDrawer(targetRun, tab);
}

function closeDrawer() {
  const panel = el("reasoningPanel");
  const overlay = el("drawerOverlay");
  if (panel) {
    panel.classList.remove("open");
    panel.classList.add("hidden");
  }
  if (overlay) overlay.classList.add("hidden");
}

async function fetchArtifacts(runId, opts = {}) {
  const res = await fetch(`/api/run/${runId}/artifacts`);
  if (!res.ok) return;
  const data = await res.json();
  cachedArtifacts = data;
  syncUploadsFromServer(data.uploads || []);
  if (opts.clearChat) {
    el("chatThread").innerHTML = "";
    lastAssistantRunId = null;
  }
  const run = data.run;
  const sources = data.sources || [];
  const claims = data.claims || [];
  const verifier = data.verifier || {};
  const envelope = buildAnswerEnvelope(data);
  runDetails[run.run_id] = envelope;
  if (!opts.skipChat) {
    renderAssistantAnswer(run.run_id, envelope);
  }
  lastAssistantRunId = run.run_id;
  el("confidence").textContent = run.confidence ? `Confidence: ${run.confidence}` : "";
  if (sources.length) {
    el("sources").innerHTML =
      "<strong>Sources</strong>: " + sources.map((s) => `<a href="${s.url}" target="_blank">${s.url}</a>`).join(" | ");
    if (!opts.liveOnly) {
      addHistory("Sources cited", "web", sources.map((s) => s.url));
    }
  } else {
    el("sources").textContent = "";
  }
  renderEvidence(claims);
  if (verifier && verifier.verdict) {
    pushReasoning(`Verifier verdict: ${verifier.verdict} (${(verifier.issues || []).length} issues)`, "info", "verifier");
  }
  if (opts.liveOnly) {
    const summary = summarizeFindings(data);
    if (summary) pushReasoning(summary.text, "info", "web", summary.urls);
  }
  if (selectedRunId === run.run_id && isDrawerOpen()) {
    populateDrawer(run.run_id, activeDrawerTab);
  }
}

async function switchToRun(runId, opts = {}) {
  if (!runId) return;
  const clearChat = opts.clearChat === true;
  const fromPoll = opts.fromPoll || false;
  const resetState = opts.resetState !== false;
  if (currentRunId === runId && !opts.force) return;
  if (evtSource) {
    evtSource.close();
    evtSource = null;
  }
  if (resetState) {
    startNewConversation({ keepQuestion: true, silent: true, keepUploads: true, preserveChat: !clearChat });
  }
  currentRunId = runId;
  subscribeEvents(runId);
  ensureThinkingPlaceholder(runId);
  await fetchArtifacts(runId, { clearChat });
  if (fromPoll) {
    setStatus("Syncing", "live");
    updateLiveTicker(`Attached to shared run ${runId}`);
  }
}

function updateCharCount() {
  const q = el("question");
  el("charCount").textContent = `${q.value.length} chars`;
}

function setupSTT() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    pushReasoning("Speech recognition not supported in this browser.", "warn", "ui");
    el("sttStatus").textContent = "Mic unsupported";
    return;
  }
  sttRecognition = new SpeechRecognition();
  sttRecognition.continuous = true;
  sttRecognition.interimResults = true;
  const langInput = el("sttLang");
  sttRecognition.lang = (langInput && langInput.value) || settingsDefaults.stt_lang || "en-US";
  sttBuffer = "";
  sttRecognition.onstart = () => {
    sttActive = true;
    el("micBtn").textContent = "Stop";
    el("sttStatus").textContent = "Listening...";
  };
  sttRecognition.onresult = (event) => {
    let interim = "";
    for (let i = event.resultIndex; i < event.results.length; ++i) {
      const res = event.results[i];
      if (res.isFinal) {
        sttBuffer += res[0].transcript + " ";
      } else {
        interim += res[0].transcript;
      }
    }
    el("question").value = (sttBuffer + interim).trim();
    updateCharCount();
  };
  sttRecognition.onerror = (evt) => {
    sttActive = false;
    el("micBtn").textContent = "Mic";
    el("sttStatus").textContent = `Mic error: ${evt.error}`;
  };
  sttRecognition.onend = () => {
    sttActive = false;
    el("micBtn").textContent = "Mic";
    el("sttStatus").textContent = "Mic stopped";
  };
  sttRecognition.start();
}

document.addEventListener("DOMContentLoaded", () => {
  loadSettings();
  document.querySelectorAll("#modelTierGroup .seg-btn").forEach((btn) => {
    btn.addEventListener("click", () => setTier(btn.dataset.tier));
  });
  setTier(currentTier);
  closeDrawer();
  hideUploadPanelIfIdle();
  updateEmptyState();
  document.querySelectorAll(".suggestion-card").forEach((card) => {
    card.addEventListener("click", () => {
      const prompt = card.dataset.prompt || card.textContent || "";
      el("question").value = prompt;
      updateCharCount();
      el("question").focus();
    });
  });
  el("chatForm").addEventListener("submit", startRun);
  const newChat = async () => {
    await resetConversation();
    await syncConversationHistory();
  };
  el("newConversationBtn").addEventListener("click", newChat);
  const panelNewConversationBtn = el("panelNewConversationBtn");
  if (panelNewConversationBtn) panelNewConversationBtn.addEventListener("click", newChat);
  const mobileNewConversationBtn = el("mobileNewConversationBtn");
  if (mobileNewConversationBtn) mobileNewConversationBtn.addEventListener("click", newChat);
  el("settingsBtn").addEventListener("click", () => toggleModal(true));
  el("closeSettings").addEventListener("click", () => toggleModal(false));
  el("closeSettingsFooter").addEventListener("click", () => toggleModal(false));
  document.getElementById("settingsModal").addEventListener("click", (e) => {
    if (e.target.id === "settingsModal") toggleModal(false);
  });
  document.addEventListener("keyup", (e) => {
    if (e.key === "Escape") {
      toggleModal(false);
      closeDrawer();
    }
  });
  el("saveSettings").addEventListener("click", saveSettings);
  el("discoverBtn").addEventListener("click", discoverModels);
  el("applyDefaults").addEventListener("click", applyRecommended);

  el("reasoningLevel").addEventListener("change", updateReasoningBadge);
  const deepRouteSelect = el("deepRoute");
  if (deepRouteSelect) {
    deepRouteSelect.addEventListener("change", updateReasoningBadge);
  }
  const liveTicker = el("liveTicker");
  if (liveTicker) liveTicker.addEventListener("click", () => openDrawer("activity", selectedRunId || currentRunId));
  const drawerClose = el("drawerClose");
  if (drawerClose) {
    drawerClose.addEventListener("click", () => closeDrawer());
  }
  const drawerOverlay = el("drawerOverlay");
  if (drawerOverlay) drawerOverlay.addEventListener("click", closeDrawer);
  document.querySelectorAll(".drawer-tabs .tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const tab = btn.dataset.tab || "activity";
      switchDrawerTab(tab);
      populateDrawer(selectedRunId, tab);
    });
  });

  const drop = el("uploadDrop");
  if (drop) {
    drop.addEventListener("dragover", (e) => {
      e.preventDefault();
      drop.classList.add("dragging");
    });
    drop.addEventListener("dragleave", () => {
      drop.classList.remove("dragging");
      draggingUploads = false;
      hideUploadPanelIfIdle();
    });
    drop.addEventListener("drop", (e) => {
      e.preventDefault();
      drop.classList.remove("dragging");
      draggingUploads = false;
      handleFileInput(e.dataTransfer.files);
    });
  }
  window.addEventListener("dragenter", () => {
    draggingUploads = true;
    showUploadPanel();
  });
  window.addEventListener("dragleave", () => {
    draggingUploads = false;
    hideUploadPanelIfIdle();
  });
  const hideUploadBtn = el("dismissUploadDrop");
  if (hideUploadBtn)
    hideUploadBtn.addEventListener("click", () => {
      uploadPanelVisible = false;
      hideUploadPanelIfIdle();
    });
  const showUploadBtn = el("showUploadDrop");
  if (showUploadBtn)
    showUploadBtn.addEventListener("click", () => {
      uploadPanelVisible = true;
      showUploadPanel();
    });
  const attachMenu = el("attachMenu");
  if (attachMenu)
    attachMenu.addEventListener("click", () => {
      uploadPanelVisible = true;
      showUploadPanel();
      el("fileInput").click();
    });
  el("attachBtn").addEventListener("click", () => {
    uploadPanelVisible = true;
    showUploadPanel();
    el("fileInput").click();
  });
  el("fileInput").addEventListener("change", (e) => {
    uploadPanelVisible = true;
    showUploadPanel();
    handleFileInput(e.target.files);
  });

  el("micBtn").addEventListener("click", () => {
    if (sttActive && sttRecognition) {
      sttRecognition.stop();
    } else {
      setupSTT();
    }
  });
  el("question").addEventListener("input", updateCharCount);
  const thread = el("chatThread");
  if (thread) {
    thread.addEventListener("click", (e) => {
      const bubble = e.target.closest(".bubble-assistant");
      if (!bubble) return;
      const runId = bubble.dataset.runId;
      selectedRunId = runId || selectedRunId;
      highlightSelectedMessage(runId);
      if (isDrawerOpen()) populateDrawer(runId, activeDrawerTab);
    });
  }

  syncConversationHistory();
  followLatestRun(true);
  latestRunPoll = setInterval(() => followLatestRun(false), 5000);

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

  updateCharCount();
  renderAttachments();
});
