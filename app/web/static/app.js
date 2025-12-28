let currentRunId = null;
let evtSource = null;
let globalEventSource = null;
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
let multiAgentMode = false;
let latestRunPoll = null;
let pendingUploads = [];
let renderedMessages = new Set();
let pendingLocalMessages = [];
let conversations = [];
let activeConversationId = null;
let unreadConversations = new Set();
let conversationSearch = "";
let conversationRefreshTimer = null;
let conversationSettingsTimer = null;
let memorySearchTimer = null;
let activePrompt = null;
let activePromptRunId = null;
let promptLocked = false;
let currentTier = "pro";
let runDetails = {};
let runEvents = {};
let selectedRunId = null;
let eventPollTimer = null;
let pollingEvents = false;
let lastEventSeq = 0;
let thinkingPlaceholders = {};
let latestLiveLine = "Thinking...";
let activeAgentSteps = new Map();
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
let settingsSnapshot = null;
let bootInProgress = false;
const LOCAL_PREFS_KEY = "localpro_prefs_v1";
const API_BASE_STORAGE_KEY = "localpro_api_base_v1";
const ACTIVE_CONVERSATION_KEY = "localpro_active_conversation_v1";
const DRAWER_MEDIA_QUERY = "(min-width: 1100px)";
const SIDEBAR_OPEN_CLASS = "sidebar-open";
const DEFAULT_HELPER_TEXT = "Enter to send. Shift+Enter for new line.";
const APP_BASE_URL = (() => {
  const normalized = normalizeBaseUrl(document.baseURI || window.location.href);
  if (normalized) return new URL(normalized);
  return new URL("/", window.location.origin);
})();
const NARRATION_PREFIX = "";
const EXECUTOR_AGENT_KEY = "__executor__";
const EXECUTOR_AGENT_LABEL = "Executor";
function getInitialApiBase() {
  const params = new URLSearchParams(window.location.search);
  const fromQuery = params.get("api") || params.get("api_base");
  let normalized = normalizeBaseUrl(fromQuery);
  if (normalized) {
    try {
      localStorage.setItem(API_BASE_STORAGE_KEY, normalized);
    } catch (_) {}
    return normalized;
  }
  normalized = normalizeBaseUrl(window.LOCALPRO_API_BASE_URL || window.LOCALPRO_API_BASE);
  if (normalized) return normalized;
  const meta = document.querySelector('meta[name="localpro-api-base"]');
  normalized = normalizeBaseUrl(meta ? meta.getAttribute("content") : null);
  if (normalized) return normalized;
  try {
    normalized = normalizeBaseUrl(localStorage.getItem(API_BASE_STORAGE_KEY));
  } catch (_) {
    normalized = null;
  }
  if (normalized) return normalized;
  return APP_BASE_URL.href;
}
let apiBaseUrl = getInitialApiBase();
let apiReady = false;

function resolveEndpoint(path) {
  const raw = path || "";
  const clean = raw.replace(/^\/+/, "");
  return new URL(clean, apiBaseUrl).href;
}

function normalizeBaseUrl(value) {
  if (!value) return null;
  const raw = value instanceof URL ? value.href : String(value);
  if (!raw.trim()) return null;
  try {
    const base = new URL(raw, window.location.href);
    base.search = "";
    base.hash = "";
    let path = (base.pathname || "/").replace(/\/+$/, "");
    if (path.endsWith("/api")) {
      path = path.slice(0, -4) || "/";
    }
    const lastSegment = path.split("/").pop() || "";
    const looksLikeFile = lastSegment.includes(".");
    if (looksLikeFile) {
      path = path.slice(0, path.lastIndexOf("/") + 1) || "/";
    } else if (!path.endsWith("/")) {
      path += "/";
    }
    base.pathname = path;
    return base.href;
  } catch {
    return null;
  }
}

function setApiBaseUrl(value, persist = true) {
  const normalized = normalizeBaseUrl(value);
  if (!normalized) return false;
  apiBaseUrl = normalized;
  if (persist) {
    try {
      localStorage.setItem(API_BASE_STORAGE_KEY, normalized);
    } catch (_) {}
  }
  updateShareLinks();
  syncApiBaseInput();
  return true;
}

function updateShareLinks() {
  let origin = "";
  try {
    origin = new URL(apiBaseUrl || APP_BASE_URL.href || window.location.origin).origin;
  } catch (_) {
    origin = window.location.origin;
  }
  if (!origin || origin === "null") return;
  const link = el("shareLink");
  if (link) {
    link.textContent = origin;
    link.href = origin;
  }
  const mobile = el("shareLinkMobile");
  if (mobile) {
    mobile.textContent = origin;
    mobile.href = origin;
  }
}

function syncApiBaseInput() {
  const input = el("cfgApiBase");
  if (input) input.value = apiBaseUrl || "";
}

async function checkApiBase(baseUrl) {
  const target = new URL("api/run/latest", baseUrl).href;
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 2500);
  try {
    const res = await fetch(target, { signal: controller.signal });
    return res.ok;
  } catch (_) {
    return false;
  } finally {
    clearTimeout(timeout);
  }
}

async function ensureApiBase() {
  const candidates = [apiBaseUrl, APP_BASE_URL.href, window.location.origin];
  const seen = new Set();
  for (const candidate of candidates) {
    const normalized = normalizeBaseUrl(candidate);
    if (!normalized || seen.has(normalized)) continue;
    seen.add(normalized);
    if (await checkApiBase(normalized)) {
      if (normalized !== apiBaseUrl) setApiBaseUrl(normalized);
      apiReady = true;
      return true;
    }
  }
  apiReady = false;
  return false;
}

async function bootApp() {
  if (bootInProgress) return apiReady;
  bootInProgress = true;
  try {
    const apiOk = await ensureApiBase();
    if (apiOk) {
      await loadSettings();
      await fetchConversations();
      const saved = loadActiveConversationId();
      if (saved && getConversationById(saved)) {
        await selectConversation(saved, { force: true });
      } else if (conversations.length) {
        await selectConversation(conversations[0].id, { force: true });
      } else {
        await createConversation();
      }
      await syncPromptState({ clear: false });
      subscribeGlobalEvents();
      followLatestRun(true);
      if (latestRunPoll) clearInterval(latestRunPoll);
      latestRunPoll = setInterval(() => followLatestRun(false), 7000);
    } else {
      const settingsStatus = el("settingsStatus");
      if (settingsStatus) settingsStatus.textContent = "API offline. Check the server URL.";
      setStatus("API offline", "error");
      updateLiveTicker("API offline. Check the server URL.");
    }
    return apiOk;
  } finally {
    bootInProgress = false;
  }
}

function loadLocalPrefs() {
  try {
    const raw = localStorage.getItem(LOCAL_PREFS_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw);
    const prefs = {};
    if (typeof parsed.auto_memory === "boolean") prefs.auto_memory = parsed.auto_memory;
    if (typeof parsed.evidence_dump === "boolean") prefs.evidence_dump = parsed.evidence_dump;
    if (typeof parsed.max_results_override === "number" && Number.isFinite(parsed.max_results_override)) {
      prefs.max_results_override = parsed.max_results_override;
    }
    if (typeof parsed.stt_lang === "string" && parsed.stt_lang.trim()) {
      prefs.stt_lang = parsed.stt_lang.trim();
    }
    return prefs;
  } catch (_) {
    return {};
  }
}

function saveLocalPrefs(prefs) {
  try {
    localStorage.setItem(LOCAL_PREFS_KEY, JSON.stringify(prefs));
  } catch (_) {}
}

function isDesktopDrawer() {
  return window.matchMedia(DRAWER_MEDIA_QUERY).matches;
}

function syncDrawerLayout() {
  const panel = el("reasoningPanel");
  const overlay = el("drawerOverlay");
  if (!panel || !overlay) return;
  if (isDesktopDrawer()) {
    if (panel.classList.contains("open")) {
      panel.classList.remove("hidden");
    } else {
      panel.classList.add("hidden");
    }
    overlay.classList.add("hidden");
  } else if (!panel.classList.contains("open")) {
    panel.classList.add("hidden");
  }
}

function setSidebarOpen(open) {
  const overlay = el("sidebarOverlay");
  document.body.classList.toggle(SIDEBAR_OPEN_CLASS, open);
  if (overlay) overlay.classList.toggle("hidden", !open);
}

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
    chip.style.background = "rgba(16,163,127,0.18)";
    chip.style.borderColor = "rgba(16,163,127,0.4)";
    chip.style.color = "#0c3a2f";
  } else if (tone === "error") {
    chip.style.background = "rgba(180,35,24,0.14)";
    chip.style.borderColor = "rgba(180,35,24,0.35)";
    chip.style.color = "#6d1b11";
  } else if (tone === "done") {
    chip.style.background = "rgba(15,138,95,0.14)";
    chip.style.borderColor = "rgba(15,138,95,0.35)";
    chip.style.color = "#0d3b2b";
  } else {
    chip.style.background = "rgba(16,163,127,0.1)";
    chip.style.borderColor = "rgba(16,163,127,0.22)";
    chip.style.color = "#0f1115";
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

function setQuestionValue(value, opts = {}) {
  const input = el("question");
  if (!input) return false;
  if (promptLocked && !opts.force) return false;
  input.value = value || "";
  updateCharCount();
  return true;
}

function updatePromptUI(opts = {}) {
  const input = el("question");
  const endBtn = el("endPromptBtn");
  const helper = el("promptHelper");
  const sendBtn = document.querySelector("#chatForm .send-btn");
  const clear = opts.clear !== false;
  if (input) {
    if (activePrompt) {
      input.value = activePrompt;
      input.readOnly = true;
      input.classList.add("prompt-locked");
    } else {
      input.readOnly = false;
      input.classList.remove("prompt-locked");
      if (clear) input.value = "";
    }
  }
  if (sendBtn) sendBtn.disabled = !!activePrompt;
  if (endBtn) endBtn.classList.toggle("hidden", !activePrompt);
  if (helper) helper.textContent = activePrompt ? "Prompt locked. End it to start a new one." : DEFAULT_HELPER_TEXT;
  updateCharCount();
}

function applyPromptState(prompt, opts = {}) {
  if (prompt && typeof prompt.prompt_text === "string" && prompt.prompt_text.trim()) {
    activePrompt = prompt.prompt_text.trim();
    activePromptRunId = prompt.run_id || null;
    promptLocked = true;
  } else {
    activePrompt = null;
    activePromptRunId = null;
    promptLocked = false;
  }
  updatePromptUI(opts);
}

async function syncPromptState(opts = {}) {
  try {
    const res = await fetch(resolveEndpoint("/api/prompt"));
    if (!res.ok) return;
    const data = await res.json();
    applyPromptState(data.prompt || null, opts);
  } catch (_) {
    // ignore prompt sync failures
  }
}

async function endPrompt() {
  try {
    const res = await fetch(resolveEndpoint("/api/prompt"), { method: "DELETE" });
    if (!res.ok) throw new Error("End prompt failed");
    applyPromptState(null, { clear: true });
  } catch (_) {
    appendActivity("Failed to end the prompt.");
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
  stopEventPolling();
  lastEventSeq = 0;
  currentRunId = null;
  cachedArtifacts = null;
  lastAssistantRunId = null;
  historyLog = [];
  runDetails = {};
  runEvents = {};
  selectedRunId = null;
  thinkingPlaceholders = {};
  renderHistory();
  {
    const activity = el("activityFeed");
    if (activity) activity.innerHTML = "";
  }
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
    if (!promptLocked) setQuestionValue("", { force: true });
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
  if (promptLocked) updatePromptUI({ clear: false });
  if (!silent) appendActivity("Ready for a new shared conversation.");
}

function loadActiveConversationId() {
  try {
    return localStorage.getItem(ACTIVE_CONVERSATION_KEY);
  } catch (_) {
    return null;
  }
}

function storeActiveConversationId(id) {
  try {
    if (id) {
      localStorage.setItem(ACTIVE_CONVERSATION_KEY, id);
    } else {
      localStorage.removeItem(ACTIVE_CONVERSATION_KEY);
    }
  } catch (_) {}
}

function formatConversationTime(value) {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  const now = new Date();
  const sameDay = date.toDateString() === now.toDateString();
  if (sameDay) {
    return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  }
  const sameYear = date.getFullYear() === now.getFullYear();
  const base = date.toLocaleDateString([], { month: "short", day: "numeric" });
  return sameYear ? base : `${base} '${String(date.getFullYear()).slice(-2)}`;
}

function conversationPreview(convo) {
  const snippet = (convo.latest_message || "").trim();
  if (!snippet) return "No messages yet.";
  const prefix = convo.latest_role === "user" ? "You: " : "LocalPro: ";
  return `${prefix}${snippet}`;
}

function getConversationById(conversationId) {
  return conversations.find((c) => c.id === conversationId) || null;
}

function renderConversationList() {
  const list = el("conversationList");
  if (!list) return;
  list.innerHTML = "";
  const needle = conversationSearch.trim().toLowerCase();
  const filtered = conversations.filter((convo) => {
    if (!needle) return true;
    const hay = `${convo.title || ""} ${convo.latest_message || ""}`.toLowerCase();
    return hay.includes(needle);
  });
  if (!filtered.length) {
    const empty = document.createElement("div");
    empty.className = "conversation-empty";
    empty.textContent = needle ? "No chats match that search." : "No chats yet.";
    list.appendChild(empty);
    return;
  }
  filtered.forEach((convo) => {
    const item = document.createElement("div");
    item.className = "conversation-item";
    if (convo.id === activeConversationId) item.classList.add("active");
    if (unreadConversations.has(convo.id)) item.classList.add("unread");
    const select = document.createElement("button");
    select.type = "button";
    select.className = "conversation-select";
    select.onclick = () => selectConversation(convo.id);
    const title = document.createElement("div");
    title.className = "conversation-title";
    title.textContent = convo.title || "New chat";
    const meta = document.createElement("div");
    meta.className = "conversation-meta";
    meta.textContent = conversationPreview(convo);
    select.appendChild(title);
    select.appendChild(meta);
    const time = document.createElement("div");
    time.className = "conversation-time";
    time.textContent = formatConversationTime(convo.latest_message_at || convo.updated_at);
    const actions = document.createElement("div");
    actions.className = "conversation-actions";
    const delBtn = document.createElement("button");
    delBtn.type = "button";
    delBtn.className = "icon-btn ghost conversation-delete";
    delBtn.textContent = "Delete";
    delBtn.title = "Delete chat";
    delBtn.onclick = (e) => {
      e.stopPropagation();
      deleteConversation(convo.id);
    };
    actions.appendChild(delBtn);
    item.appendChild(select);
    item.appendChild(time);
    item.appendChild(actions);
    list.appendChild(item);
  });
}

function updateConversationHeader(convo) {
  const titleEl = el("conversationTitle");
  if (titleEl) titleEl.textContent = convo?.title || "New chat";
  const metaEl = el("conversationMeta");
  if (metaEl) {
    const updated = formatConversationTime(convo?.updated_at);
    metaEl.textContent = updated ? `Updated ${updated}` : "";
  }
}

function scheduleConversationRefresh() {
  if (conversationRefreshTimer) clearTimeout(conversationRefreshTimer);
  conversationRefreshTimer = setTimeout(() => {
    fetchConversations();
  }, 400);
}

async function fetchConversations(opts = {}) {
  try {
    const res = await fetch(resolveEndpoint("/api/conversations"));
    if (!res.ok) return;
    const data = await res.json();
    conversations = data.conversations || [];
    renderConversationList();
    if (activeConversationId) {
      const convo = getConversationById(activeConversationId);
      if (convo) updateConversationHeader(convo);
    }
    if (opts.ensureActive && !activeConversationId && conversations.length) {
      await selectConversation(conversations[0].id, { force: true });
    }
  } catch (_) {
    // ignore list failures
  }
}

function getReasoningSettings() {
  const selected = el("reasoningLevel")?.value || "AUTO";
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
  return { reasoningMode, manualLevel, deepRoute };
}

function getConversationSettingsFromUI() {
  const settings = getReasoningSettings();
  return {
    model_tier: currentTier,
    reasoning_mode: settings.reasoningMode,
    manual_level: settings.manualLevel,
    deep_mode: currentTier === "deep" ? settings.deepRoute : "auto",
  };
}

async function persistConversationSettings() {
  if (!activeConversationId) return;
  const payload = getConversationSettingsFromUI();
  try {
    const res = await fetch(resolveEndpoint(`/api/conversations/${activeConversationId}`), {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (res.ok) {
      const data = await res.json();
      const updated = data.conversation;
      if (updated) {
        const idx = conversations.findIndex((c) => c.id === updated.id);
        if (idx >= 0) conversations[idx] = updated;
        activeConversationId = updated.id;
        updateConversationHeader(updated);
        renderConversationList();
      }
    }
  } catch (_) {
    // ignore update failures
  }
}

function scheduleConversationSettingsUpdate() {
  if (!activeConversationId) return;
  if (conversationSettingsTimer) clearTimeout(conversationSettingsTimer);
  conversationSettingsTimer = setTimeout(() => {
    persistConversationSettings();
  }, 250);
}

function applyConversationSettings(convo) {
  if (!convo) return;
  setTier(convo.model_tier || "pro", { persist: false });
  const deepRoute = el("deepRoute");
  if (deepRoute) deepRoute.value = convo.deep_mode || "auto";
  const select = el("reasoningLevel");
  if (select) {
    const useManual = (convo.reasoning_mode || "auto") === "manual";
    const level = useManual ? convo.manual_level || "MED" : "AUTO";
    select.value = level;
  }
  updateReasoningBadge();
}

async function loadConversationMessages(conversationId) {
  try {
    const res = await fetch(resolveEndpoint(`/api/conversations/${conversationId}/messages`));
    if (!res.ok) return;
    const data = await res.json();
    (data.messages || []).forEach((msg) => {
      appendChat(msg.role || "assistant", msg.content || "", { messageId: msg.id, runId: msg.run_id });
    });
  } catch (_) {
    // ignore hydration errors
  }
}

async function selectConversation(conversationId, opts = {}) {
  if (!conversationId) return;
  if (activeConversationId === conversationId && !opts.force) return;
  activeConversationId = conversationId;
  storeActiveConversationId(conversationId);
  unreadConversations.delete(conversationId);
  const convo = opts.conversation || getConversationById(conversationId);
  updateConversationHeader(convo);
  applyConversationSettings(convo);
  startNewConversation({ keepQuestion: false, keepUploads: false, silent: true, preserveChat: false });
  await loadConversationMessages(conversationId);
  if (convo && convo.latest_run_id) {
    await switchToRun(convo.latest_run_id, { clearChat: false, resetState: false, skipThinking: true });
  }
  renderConversationList();
}

async function createConversation() {
  const payload = getConversationSettingsFromUI();
  try {
    const res = await fetch(resolveEndpoint("/api/conversations"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error("create failed");
    const data = await res.json();
    const convo = data.conversation;
    if (convo) {
      conversations = [convo, ...conversations.filter((c) => c.id !== convo.id)];
      await selectConversation(convo.id, { conversation: convo, force: true });
      renderConversationList();
      return convo;
    }
  } catch (_) {
    appendActivity("Failed to create a new chat.");
  }
  return null;
}

async function deleteConversation(conversationId) {
  const convo = getConversationById(conversationId);
  const title = convo?.title || "this chat";
  if (!confirm(`Delete "${title}"? This removes its runs and uploads.`)) return;
  try {
    const res = await fetch(resolveEndpoint(`/api/conversations/${conversationId}`), { method: "DELETE" });
    if (!res.ok) throw new Error("delete failed");
    unreadConversations.delete(conversationId);
    conversations = conversations.filter((c) => c.id !== conversationId);
    if (conversationId === activeConversationId) {
      activeConversationId = null;
      storeActiveConversationId(null);
      if (conversations.length) {
        await selectConversation(conversations[0].id, { force: true });
      } else {
        await createConversation();
      }
    } else {
      renderConversationList();
    }
  } catch (_) {
    appendActivity("Failed to delete chat.");
  }
}

async function renameConversation(conversationId, title) {
  if (!conversationId) return;
  const nextTitle = (title || "").trim();
  if (!nextTitle) return;
  try {
    const res = await fetch(resolveEndpoint(`/api/conversations/${conversationId}`), {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title: nextTitle }),
    });
    if (!res.ok) throw new Error("rename failed");
    const data = await res.json();
    const convo = data.conversation;
    if (convo) {
      const idx = conversations.findIndex((c) => c.id === convo.id);
      if (idx >= 0) conversations[idx] = convo;
      if (convo.id === activeConversationId) updateConversationHeader(convo);
      renderConversationList();
    }
  } catch (_) {
    appendActivity("Failed to rename chat.");
  }
}

function escapeAndBreak(text) {
  const div = document.createElement("div");
  div.textContent = text || "";
  return div.innerHTML.replace(/\n/g, "<br>");
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text || "";
  return div.innerHTML;
}

async function copyToClipboard(text) {
  const value = String(text || "");
  if (!value) return false;
  if (navigator.clipboard && window.isSecureContext) {
    try {
      await navigator.clipboard.writeText(value);
      return true;
    } catch (_) {}
  }
  const area = document.createElement("textarea");
  area.value = value;
  area.setAttribute("readonly", "");
  area.style.position = "fixed";
  area.style.top = "-1000px";
  area.style.left = "-1000px";
  area.style.opacity = "0";
  document.body.appendChild(area);
  area.select();
  area.setSelectionRange(0, area.value.length);
  let ok = false;
  try {
    ok = document.execCommand("copy");
  } catch (_) {}
  document.body.removeChild(area);
  return ok;
}

function withNarrationPrefix(text, lane) {
  if (!text || lane !== "executor" || !NARRATION_PREFIX) return text;
  const prefix = `${NARRATION_PREFIX}: `;
  return text.startsWith(prefix) ? text : `${prefix}${text}`;
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
  return wrap;
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
  const line = latestLiveLine && latestLiveLine.trim() ? latestLiveLine.trim() : "Thinking...";
  return `
    <div class="thinking-block">
      <div class="thinking-head">
        <div class="thinking-status">
          <span class="thinking-orb" aria-hidden="true"></span>
          <span class="thinking-title">Thinking</span>
        </div>
        <span class="thinking-divider" aria-hidden="true"></span>
        <span class="thinking-line" aria-live="polite">${escapeHtml(line)}</span>
        <button type="button" class="stop-btn" data-action="stop" title="Stop generating">Stop</button>
      </div>
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
  const targetRun = runId || currentRunId;
  if (evtSource) {
    try {
      evtSource.close();
    } catch (_) {}
    evtSource = null;
  }
  stopTimer();
  setStatus("Stopped", "error");
  updateLiveTicker("Generation stopped");
  resetLiveAgentState();
  if (targetRun) {
    void requestRunStop(targetRun);
  }
  const bubble = getAssistantBubble(targetRun);
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
  // Activity entries now flow through the merged live log.
  const payload = typeof entry === "string" ? { text: entry } : entry || {};
  const runId = payload.runId || payload.run_id || null;
  const updateTicker = (text, urls, lane) => {
    if (!text) return;
    updateLiveTicker(withNarrationPrefix(text, lane), urls || []);
  };
  const pushLine = (line) => {
    if (!line || !line.text) return null;
    const lane = line.lane || payload.lane || "executor";
    const tone = line.tone || payload.tone || "info";
    const urls = line.urls || payload.urls || [];
    addHistory(line.text, lane, urls, tone, runId);
    return { text: line.text, lane, urls };
  };
  if (Array.isArray(payload.lines)) {
    let lastLine = null;
    payload.lines.forEach((line) => {
      const pushed = pushLine(line);
      if (pushed) lastLine = pushed;
    });
    if (lastLine) updateTicker(lastLine.text, lastLine.urls, lastLine.lane);
    return;
  }
  if (!payload.text) return;
  const pushed = pushLine(payload);
  if (pushed) updateTicker(pushed.text, pushed.urls, pushed.lane);
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
      const res = await fetch(resolveEndpoint("/api/uploads"), { method: "POST", body: form });
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
  multiAgentMode = false;
  resetLiveAgentState();
}

function stepKeyFrom(detail = {}) {
  const key = detail?.step_id ?? detail?.step;
  if (key === null || key === undefined) return null;
  return String(key);
}

function agentLabelFrom(detail = {}) {
  const name = String(detail?.name || "").trim();
  if (name) return name;
  const profile = String(detail?.agent_profile || "").trim();
  if (profile) return profile.replace(/([a-z0-9])([A-Z])/g, "$1 $2");
  const type = String(detail?.type || "").trim();
  if (type) return type;
  const key = stepKeyFrom(detail);
  return key ? `Step ${key}` : "Agent";
}

function resetLiveAgentState() {
  activeAgentSteps.clear();
  updateLiveAgentIndicator();
}

function setExecutorActive(active) {
  const hasExecutor = activeAgentSteps.has(EXECUTOR_AGENT_KEY);
  if (active && !hasExecutor) {
    activeAgentSteps.set(EXECUTOR_AGENT_KEY, { label: EXECUTOR_AGENT_LABEL });
  } else if (!active && hasExecutor) {
    activeAgentSteps.delete(EXECUTOR_AGENT_KEY);
  } else {
    return;
  }
  updateLiveAgentIndicator();
}

function syncExecutorFromRun(run = {}) {
  const status = String(run.status || "").toLowerCase();
  setExecutorActive(status === "running");
}

function updateLiveAgentIndicator() {
  const chip = el("liveAgentCount");
  if (!chip) return;
  const entries = Array.from(activeAgentSteps.entries()).filter(([key]) => key !== EXECUTOR_AGENT_KEY);
  const count = entries.length;
  const hasExecutor = activeAgentSteps.has(EXECUTOR_AGENT_KEY);
  chip.textContent = `Live agents: ${count}`;
  if (count === 0) {
    chip.title = hasExecutor ? "Executor active" : "No active agents";
    return;
  }
  const labels = entries
    .map(([, entry]) => entry.label)
    .filter(Boolean);
  const condensed = condenseList(labels, 3);
  chip.title = condensed ? `Active: ${condensed}` : "Agents active";
}

function noteAgentStepStarted(detail = {}) {
  const key = stepKeyFrom(detail);
  if (!key) return;
  activeAgentSteps.set(key, { label: agentLabelFrom(detail) });
  updateLiveAgentIndicator();
}

function noteAgentStepFinished(detail = {}) {
  const key = stepKeyFrom(detail);
  if (!key) return;
  activeAgentSteps.delete(key);
  updateLiveAgentIndicator();
}

function eventSeverity(type) {
  if (["archived", "error", "tavily_error"].includes(type)) return "high";
  if (
    [
      "run_started",
      "router_decision",
      "resource_budget",
      "plan_created",
      "plan_updated",
      "work_log",
      "step_completed",
      "control_action",
      "loop_iteration",
      "strict_mode",
      "client_note",
      "step_error",
      "upload_received",
      "upload_processed",
      "upload_failed",
      "executor_brief",
      "allocator_decision",
      "planner_verifier",
      "model_unavailable",
      "model_error",
      "source_found",
      "claim_found",
      "search_skipped",
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

function normalizeNote(note) {
  if (!note) return "";
  const match = note.match(/^([A-Za-z0-9_-]+) mode: [^()]+\(route ([^)]+)\)/i);
  if (match) {
    const tier = match[1].toUpperCase();
    const route = match[2];
    return `Running in ${tier} mode (route ${route}).`;
  }
  return note;
}

const LANE_LABELS = {
  orch: "Planner",
  executor: "Executor",
  primary: "Primary researcher",
  recency: "Recency checker",
  adversarial: "Caveat checker",
  router: "Router",
  verifier: "Verifier",
  memory: "Memory",
  web: "Web",
  ui: "UI",
  worklog: "Status",
};

function laneLabel(lane) {
  if (!lane) return "";
  return LANE_LABELS[lane] || lane;
}

const MULTI_AGENT_LANES = new Set(["executor", "primary", "recency", "adversarial", "verifier"]);

function hasMultipleLanes(events) {
  const lanes = new Set();
  (events || []).forEach((ev) => {
    const lane = ev.lane;
    if (!lane || lane === "worklog") return;
    if (!MULTI_AGENT_LANES.has(lane)) return;
    lanes.add(lane);
  });
  return lanes.size > 1;
}

function shouldPrefixLane(lane, multiLane) {
  if (!multiLane || !lane) return false;
  return lane !== "worklog";
}

function withLane(text, lane, multiLane) {
  if (!shouldPrefixLane(lane, multiLane)) return text;
  const label = laneLabel(lane);
  return label ? `${label}: ${text}` : text;
}

function inferStepType(detail) {
  const raw = String(detail?.type || "").toLowerCase();
  if (raw) return raw;
  const name = String(detail?.name || "").toLowerCase();
  if (name.includes("research")) return "research";
  if (name.includes("search")) return "search";
  if (name.includes("extract")) return "extract";
  if (name.includes("merge")) return "merge";
  if (name.includes("draft") || name.includes("write")) return "draft";
  if (name.includes("finalize") || name.includes("final")) return "finalize";
  if (name.includes("verify") || name.includes("check")) return "verify";
  if (name.includes("analysis") || name.includes("analy")) return "analysis";
  return "";
}

function stepAction(type, phase, detail = {}) {
  const topic = narrationTopic(detail);
  const actions = {
    analysis: {
      start: topic ? `Analyzing ${topic}` : "Analyzing the details",
      done: "Analysis complete",
    },
    research: {
      start: topic ? `Gathering sources on ${topic}` : "Gathering sources",
      done: "Sources gathered",
    },
    search: {
      start: topic ? `Searching for ${topic}` : "Searching for sources",
      done: "Search pass complete",
    },
    tavily_search: {
      start: topic ? `Searching for ${topic}` : "Searching for sources",
      done: "Search pass complete",
    },
    extract: { start: "Reading sources", done: "Source notes captured" },
    tavily_extract: { start: "Reading sources", done: "Source notes captured" },
    merge: { start: "Combining notes", done: "Notes combined" },
    draft: { start: "Drafting the response", done: "Draft ready" },
    finalize: { start: "Finalizing the response", done: "Response finalized" },
    verify: { start: "Double-checking the draft", done: "Checks complete" },
  };
  const entry = actions[type];
  if (!entry) return "";
  return phase === "done" ? entry.done : entry.start;
}

function shouldIncludeStepName(type, name) {
  if (!name) return false;
  const lower = name.toLowerCase();
  if (lower.includes("lane") || lower.includes("primary") || lower.includes("recency") || lower.includes("adversarial")) {
    return true;
  }
  if (!type) return true;
  return !lower.includes(type);
}

function describeStep(detail, phase) {
  const name = String(detail?.name || "").trim();
  const type = inferStepType(detail);
  const action = stepAction(type, phase, detail);
  if (action) {
    const suffix = shouldIncludeStepName(type, name) ? ` (${name})` : "";
    return `${action}${suffix}.`;
  }
  if (phase === "done") return `Finished ${name || "the step"}.`;
  return `Working on ${name || "the next step"}.`;
}

function narrationStepAction(detail, phase) {
  const type = inferStepType(detail);
  const action = stepAction(type, phase, detail);
  if (action) return action;
  const name = String(detail?.name || "").trim();
  if (name) {
    return phase === "done" ? `Finished ${name}` : `Working on ${name}`;
  }
  return phase === "done" ? "Finished a step" : "Working on the next step";
}

function normalizeAction(text) {
  return String(text || "").trim().replace(/\.$/, "");
}

const NARRATION_PREFIXES = [
  "how do i ",
  "how can i ",
  "how to ",
  "what is ",
  "what are ",
  "why is ",
  "why are ",
  "can you ",
  "could you ",
  "please ",
  "tell me ",
  "give me ",
  "show me ",
  "find ",
  "search for ",
  "search ",
  "look up ",
  "lookup ",
];
const IDLE_NARRATION_CHOICES = [
  (topic) => (topic ? `Still working on ${topic}` : "Still working on it"),
  (topic) => (topic ? `Making progress on ${topic}` : "Making steady progress"),
  (topic) => (topic ? `Thinking through ${topic}` : "Thinking through the details"),
  (topic) => (topic ? `Double-checking ${topic}` : "Double-checking the details"),
];
let idleNarrationIndex = 0;

function shortenWords(text, maxWords) {
  const words = String(text || "")
    .trim()
    .split(/\s+/)
    .filter(Boolean);
  if (!words.length) return "";
  if (words.length <= maxWords) return words.join(" ");
  return words.slice(0, maxWords).join(" ");
}

function cleanNarrationTopic(text) {
  if (!text) return "";
  let cleaned = String(text).replace(/\s+/g, " ").trim();
  cleaned = cleaned.replace(/^[\"'`]+|[\"'`]+$/g, "");
  cleaned = cleaned.replace(/[?.!]+$/, "");
  let lower = cleaned.toLowerCase();
  for (const prefix of NARRATION_PREFIXES) {
    if (lower.startsWith(prefix)) {
      cleaned = cleaned.slice(prefix.length).trim();
      break;
    }
  }
  cleaned = shortenWords(cleaned, 8);
  return cleaned;
}

function narrationTopic(detail = {}) {
  const candidates = [detail?.query, detail?.question, pendingQuestion, detail?.name];
  for (const candidate of candidates) {
    const cleaned = cleanNarrationTopic(candidate);
    if (cleaned) return cleaned;
  }
  return "";
}

function nextIdleNarration() {
  const topic = narrationTopic();
  const line = IDLE_NARRATION_CHOICES[idleNarrationIndex % IDLE_NARRATION_CHOICES.length](topic);
  idleNarrationIndex += 1;
  return line;
}

function formatSearchNarration(detail = {}) {
  const query = cleanNarrationTopic(detail.query);
  const topic = narrationTopic(detail);
  const base = query ? `Searching for ${query}` : topic ? `Looking up ${topic}` : "Searching the web";
  const resultCount = Number(detail.result_count || 0);
  const newCount = Number(detail.new_sources || 0);
  const dupes = Number(detail.duplicate_sources || 0);
  if (resultCount) {
    const parts = [];
    if (newCount) parts.push(`${newCount} new`);
    if (dupes) parts.push(`${dupes} already seen`);
    if (!parts.length) parts.push(`${resultCount} results`);
    return `${base} (${parts.join(", ")})`;
  }
  return base;
}

function formatExtractNarration(detail = {}) {
  const urls = Array.isArray(detail.urls) ? detail.urls : [];
  const hosts = condenseList(
    urls
      .map((url) => hostFromUrl(url))
      .filter(Boolean),
    2
  );
  if (hosts) return `Reading sources from ${hosts}`;
  return "Reading sources";
}

function formatToolRequest(detail = {}) {
  const requests = Array.isArray(detail.requests) ? detail.requests : [];
  if (!requests.length) return "Running a quick tool";
  const req = requests[0] || {};
  const tool = String(req.tool || req.type || req.name || "").toLowerCase();
  if (["calculator", "calc", "math"].includes(tool)) {
    const expr = String(req.expr || req.expression || req.input || "").trim();
    const cleanExpr = expr.replace(/\s+/g, " ");
    return cleanExpr ? `Computing ${cleanExpr}` : "Running a quick calculation";
  }
  if (["live_date", "time_now", "now", "date"].includes(tool)) {
    return "Checking the current time";
  }
  if (["code_eval", "code", "python", "execute_code", "exec_code", "code_exec", "execute", "python_exec", "local_code"].includes(tool)) {
    return "Running the code tool";
  }
  if (["read_text", "read_file", "file_read", "text_read", "read_bytes", "file_bytes", "read_file_bytes"].includes(tool)) {
    const path = String(req.path || req.file || req.filename || "").trim();
    const name = path ? path.split(/[\\/]/).pop() : "";
    return name ? `Reading ${name}` : "Reading a file";
  }
  if (["list_files", "list_dir", "list_directory", "ls"].includes(tool)) {
    return "Listing files";
  }
  if (["plot_chart", "plot_graph", "chart", "graph"].includes(tool)) {
    return "Drawing a quick chart";
  }
  if (["image_info", "image_metadata", "image_load", "image_open", "image_zoom", "image_crop", "image_eval"].includes(tool)) {
    return "Inspecting an image";
  }
  return "Running a quick tool";
}

function formatToolResult(detail = {}) {
  const results = Array.isArray(detail.results) ? detail.results : [];
  if (!results.length) return "Tool results ready";
  const res = results[0] || {};
  const tool = String(res.tool || res.type || res.name || "").toLowerCase();
  if (["calculator", "calc", "math"].includes(tool)) return "Calculation complete";
  if (["live_date", "time_now", "now", "date"].includes(tool)) return "Time check done";
  if (["code_eval", "code", "python", "execute_code", "exec_code", "code_exec", "execute", "python_exec", "local_code"].includes(tool)) {
    return "Code tool finished";
  }
  if (["read_text", "read_file", "file_read", "text_read", "read_bytes", "file_bytes", "read_file_bytes"].includes(tool)) {
    return "File read complete";
  }
  if (["list_files", "list_dir", "list_directory", "ls"].includes(tool)) {
    return "File list ready";
  }
  if (["plot_chart", "plot_graph", "chart", "graph"].includes(tool)) {
    return "Chart ready";
  }
  if (["image_info", "image_metadata", "image_load", "image_open", "image_zoom", "image_crop", "image_eval"].includes(tool)) {
    return "Image check done";
  }
  return "Tool results ready";
}

function extractToolMedia(detail = {}) {
  const results = Array.isArray(detail.results) ? detail.results : [];
  for (const res of results) {
    if (!res || typeof res !== "object") continue;
    const tool = String(res.tool || res.type || res.name || "").toLowerCase();
    if (!["plot_chart", "plot_graph", "chart", "graph"].includes(tool)) continue;
    const result = res.result && typeof res.result === "object" ? res.result : {};
    const src = result.data_url || res.data_url || "";
    if (!src) continue;
    const title = String(result.title || res.title || "Chart").trim();
    return [{ type: "image", src, title }];
  }
  return null;
}

function formatTraceDetail(detail, maxLen = 240) {
  if (detail === null || detail === undefined) return "";
  let text = "";
  if (typeof detail === "string") {
    text = detail;
  } else {
    try {
      text = JSON.stringify(detail);
    } catch (_) {
      text = String(detail);
    }
  }
  text = String(text || "").replace(/\s+/g, " ").trim();
  if (!text) return "";
  if (text.length > maxLen) return `${text.slice(0, maxLen - 3)}...`;
  return text;
}

function traceUrls(detail = {}) {
  const urls = [];
  if (!detail || typeof detail !== "object") return urls;
  if (detail.url) urls.push(detail.url);
  if (Array.isArray(detail.urls)) urls.push(...detail.urls);
  return urls;
}

function traceLineForEvent(type, detail = {}) {
  const safe = detail || {};
  switch (type) {
    case "work_log": {
      const text = String(safe.text || "").trim();
      if (!text) return null;
      return { text, tone: safe.tone || "info", lane: "worklog", urls: traceUrls(safe) };
    }
    case "narration": {
      const text = String(safe.text || "").trim();
      if (!text) return null;
      return { text, tone: safe.tone || "info", lane: "executor", urls: traceUrls(safe) };
    }
    case "dev_trace": {
      return null;
    }
    case "run_started": {
      const question = shortenWords(String(safe.question || ""), 18);
      return {
        text: question ? `Run started: ${question}` : "Run started",
        tone: "info",
        lane: "orch",
        urls: traceUrls(safe),
      };
    }
    case "router_decision": {
      const tier = safe.model_tier || safe.requested_tier || "";
      const level = safe.reasoning_level || "";
      let text = "Router decision";
      if (tier) text += `: ${tierLabel(tier)}`;
      if (safe.deep_route) {
        text += ` (${deepRouteLabel(safe.deep_route)})`;
      } else if (level) {
        text += ` (${level})`;
      }
      return { text, tone: "info", lane: "router", urls: traceUrls(safe) };
    }
    case "plan_created": {
      const steps = Number(safe.expected_total_steps || safe.steps || 0);
      const passes = Number(safe.expected_passes || 0);
      let text = steps ? `Plan created: ${steps} steps` : "Plan created";
      if (passes) text += `, ${passes} pass${passes === 1 ? "" : "es"}`;
      return { text, tone: "info", lane: "orch", urls: traceUrls(safe) };
    }
    case "plan_updated": {
      const steps = Number(safe.expected_total_steps || safe.steps || 0);
      const passes = Number(safe.expected_passes || 0);
      let text = steps ? `Plan updated: ${steps} steps` : "Plan updated";
      if (passes) text += `, ${passes} pass${passes === 1 ? "" : "es"}`;
      return { text, tone: "info", lane: "orch", urls: traceUrls(safe) };
    }
    case "step_started": {
      const base = describeStep(safe, "start");
      const clean = base.replace(/\.$/, "");
      const suffix = safe.step_id ? ` (step ${safe.step_id})` : "";
      const lane = laneFromProfile(safe.agent_profile || "") || laneFrom(safe.name || "", safe.step_id);
      return { text: `${clean}${suffix}.`, tone: "info", lane, urls: traceUrls(safe) };
    }
    case "step_completed": {
      const base = describeStep(safe, "done");
      const clean = base.replace(/\.$/, "");
      const suffix = safe.step_id ? ` (step ${safe.step_id})` : "";
      const lane = laneFromProfile(safe.agent_profile || "") || laneFrom(safe.name || "", safe.step_id);
      return { text: `${clean}${suffix}.`, tone: "info", lane, urls: traceUrls(safe) };
    }
    case "step_error": {
      const label = safe.name || (safe.step ? `Step ${safe.step}` : "Step");
      const msg = safe.message || "error encountered";
      const lane = laneFromProfile(safe.agent_profile || "") || laneFrom(safe.name || "", safe.step);
      const detail = String(safe.detail || "").trim();
      const suffix = detail ? ` (${shortenWords(detail, 16)})` : "";
      return { text: `${label} error: ${msg}${suffix}`, tone: "warn", lane, urls: traceUrls(safe) };
    }
    case "tool_request": {
      const action = formatToolRequest(safe);
      const lane = laneFrom("", safe.step);
      return { text: `Tool request: ${action}`, tone: "info", lane, urls: traceUrls(safe) };
    }
    case "tool_result": {
      const action = formatToolResult(safe);
      const lane = laneFrom("", safe.step);
      const media = extractToolMedia(safe);
      const text = action.startsWith("Chart") ? action : `Tool result: ${action}`;
      return { text, tone: "info", lane, urls: traceUrls(safe), media };
    }
    case "tavily_search": {
      const query = String(safe.query || "").trim();
      const resultCount = Number(safe.result_count || 0);
      const newCount = Number(safe.new_sources || 0);
      const dupes = Number(safe.duplicate_sources || 0);
      let line = query ? `Search: ${shortenWords(query, 14)}` : "Search pass complete";
      if (resultCount) {
        const parts = [];
        if (newCount) parts.push(`${newCount} new`);
        if (dupes) parts.push(`${dupes} already seen`);
        if (!parts.length) parts.push(`${resultCount} results`);
        line += ` (${parts.join(", ")})`;
      }
      return { text: line, tone: "info", lane: laneFrom("", safe.step), urls: traceUrls(safe) };
    }
    case "tavily_extract": {
      const action = formatExtractNarration(safe);
      return { text: `Extract: ${action}`, tone: "info", lane: laneFrom("", safe.step), urls: traceUrls(safe) };
    }
    case "tavily_error": {
      const msg = safe.message || "Tavily unavailable";
      return { text: `Search error: ${msg}`, tone: "warn", lane: laneFrom("", safe.step), urls: traceUrls(safe) };
    }
    case "search_skipped": {
      const query = String(safe.query || "").trim();
      const line = query ? `Skipped duplicate search: ${shortenWords(query, 12)}` : "Skipped a duplicate search";
      return { text: line, tone: "info", lane: laneFrom("", safe.step), urls: traceUrls(safe) };
    }
    case "source_found": {
      const title = String(safe.title || "").trim();
      const publisher = String(safe.publisher || "").trim();
      const url = String(safe.url || "").trim();
      let label = title || publisher || hostFromUrl(url);
      if (title && publisher && publisher !== title) {
        label = `${title} (${publisher})`;
      }
      const text = label ? `Source found: ${shortenWords(label, 16)}` : "Source found";
      const lane = laneFromProfile(safe.lane || safe.agent_profile || "");
      return { text, tone: "info", lane, urls: traceUrls(safe) };
    }
    case "claim_found": {
      const claim = String(safe.claim || "").trim();
      const text = claim ? `Finding: ${shortenWords(claim, 18)}` : "Finding noted";
      const lane = laneFromProfile(safe.lane || safe.agent_profile || "");
      return { text, tone: "info", lane, urls: traceUrls(safe) };
    }
    case "upload_processed": {
      const name = safe.name ? `: ${safe.name}` : "";
      return { text: `Upload processed${name}`, tone: "info", lane: "orch", urls: traceUrls(safe) };
    }
    case "upload_failed": {
      const name = safe.name ? `: ${safe.name}` : "";
      const err = safe.error ? ` (${safe.error})` : "";
      return { text: `Upload failed${name}${err}`, tone: "warn", lane: "orch", urls: traceUrls(safe) };
    }
    case "memory_retrieved": {
      const count = Number(safe.count || 0);
      return {
        text: count ? `Memory retrieved (${count})` : "Memory retrieved",
        tone: "info",
        lane: "memory",
        urls: traceUrls(safe),
      };
    }
    case "memory_saved": {
      const count = Number(safe.count || 0);
      return {
        text: count ? `Memory saved (${count})` : "Memory saved",
        tone: "info",
        lane: "memory",
        urls: traceUrls(safe),
      };
    }
    case "loop_iteration": {
      const iteration = Number(safe.iteration || 0);
      const text = iteration ? `Verifier requested another pass (loop ${iteration})` : "Verifier requested another pass";
      return { text, tone: "warn", lane: "verifier", urls: traceUrls(safe) };
    }
    case "control_action": {
      const origin = String(safe.origin || "").toLowerCase();
      const control = String(safe.control || safe.action_type || "").toUpperCase();
      let text = origin === "user" ? "Live plan update applied" : "Quality check applied";
      if (control && control !== "CONTINUE") text += ` (${control})`;
      return { text, tone: "info", lane: "orch", urls: traceUrls(safe) };
    }
    case "model_unavailable": {
      const model = safe.model || safe.profile || "model";
      return { text: `Model unavailable: ${model}`, tone: "warn", lane: "orch", urls: traceUrls(safe) };
    }
    case "model_error": {
      const model = safe.model || safe.profile || "model";
      return { text: `Model error: ${model}`, tone: "warn", lane: "orch", urls: traceUrls(safe) };
    }
    case "archived": {
      const base = safe.stopped ? "Run stopped" : "Run complete";
      const conf = safe.confidence ? ` (${safe.confidence})` : "";
      return { text: `${base}${conf}`, tone: "info", lane: "orch", urls: traceUrls(safe) };
    }
    case "error": {
      const msg = safe.message || "Run error";
      return { text: `Run error: ${msg}`, tone: "error", lane: "orch", urls: traceUrls(safe) };
    }
    default:
      return null;
  }
}

function narrationForEvent(ev) {
  const d = ev.detail || {};
  switch (ev.type) {
    case "run_started":
      if (d.question) pendingQuestion = d.question;
      {
        const topic = cleanNarrationTopic(d.question);
        return topic ? `Starting on ${topic}` : "Starting the run";
      }
    case "router_decision":
      return "Choosing a route";
    case "plan_created":
      return "Sketching the plan";
    case "allocator_decision":
      return "Handing out tasks";
    case "memory_retrieved":
      if (Number.isFinite(Number(d.count)) && Number(d.count) > 0) {
        return `Checking memory (${Number(d.count)})`;
      }
      return "Checking memory";
    case "memory_saved":
      if (Number.isFinite(Number(d.count)) && Number(d.count) > 0) {
        return `Saving notes (${Number(d.count)})`;
      }
      return "Saving notes";
    case "upload_processed":
      return "Reviewing the upload";
    case "upload_failed":
      return "Upload hit a snag";
    case "step_started":
      return narrationStepAction(d, "start");
    case "step_completed":
      return narrationStepAction(d, "done");
    case "tavily_search":
      return formatSearchNarration(d);
    case "tavily_extract":
      return formatExtractNarration(d);
    case "tool_request":
      return formatToolRequest(d);
    case "tool_result":
      return formatToolResult(d);
    case "control_action":
      return "Quick quality check";
    case "loop_iteration":
      return "Not sure about results yet";
    case "tavily_error":
      return "Web search hiccup, moving on";
    case "step_error":
      return "Hit a snag, continuing";
    case "archived":
      return "Wrapping up";
    case "model_unavailable":
      return "Model unavailable, switching gears";
    case "model_error":
      return "Model had trouble, continuing";
    case "error":
      return "Hit an error";
    default:
      return "";
  }
}

function summarizeLiveEvents(events) {
  const summary = { lines: [], usedQuestion: false };
  if (!events || !events.length) return summary;
  const multiLane = hasMultipleLanes(events);
  const lines = [];
  let usedQuestion = false;
  const pushLine = (text, ev, tone = "", urls = null, laneOverride = "", media = null) => {
    const clean = String(text || "").trim();
    if (!clean) return;
    const lane = laneOverride || ev?.lane || "orch";
    const severity = ev?.severity || "";
    let pickedTone = tone;
    if (!pickedTone) {
      if (severity === "high") {
        pickedTone = ev?.type === "error" ? "error" : "warn";
      } else if (severity === "medium") {
        pickedTone = "info";
      } else {
        pickedTone = "info";
      }
    }
    const line = {
      text: withLane(clean, lane, multiLane),
      tone: pickedTone,
      lane,
      urls: (Array.isArray(urls) && urls.length ? urls : ev?.urls) || [],
      media: media || null,
    };
    if (lines.length && lines[lines.length - 1].text === line.text) return;
    lines.push(line);
  };

  (events || []).forEach((ev) => {
    if (!ev) return;
    const d = ev.detail || {};
    if (ev.type === "run_started") {
      const question = String(d.question || pendingQuestion || "").trim();
      if (question && !questionShownInLive) {
        usedQuestion = true;
        pushLine(`Plan: ${shortenWords(question, 18)}`, ev);
        return;
      }
    }
    if (ev.type === "client_note") {
      const note = normalizeNote(d.note || d.text || "");
      if (note) pushLine(note, ev);
      return;
    }
    if (ev.type === "executor_brief" || ev.type === "allocator_decision") {
      const note = String(d.note || "").trim();
      if (note) pushLine(note, ev);
      return;
    }
    if (ev.type === "tavily_search") {
      pushLine(`Tool request: ${formatSearchNarration(d)}`, ev, "info", ev.urls);
      return;
    }
    if (ev.type === "tavily_extract") {
      pushLine(`Tool result: ${formatExtractNarration(d)}`, ev, "info", ev.urls);
      return;
    }
    if (ev.type === "tool_request") {
      pushLine(`Tool request: ${formatToolRequest(d)}`, ev);
      return;
    }
    if (ev.type === "tool_result") {
      const action = formatToolResult(d);
      const text = action.startsWith("Chart") ? action : `Tool result: ${action}`;
      pushLine(text, ev, "info", ev.urls, "", extractToolMedia(d));
      return;
    }
    const trace = traceLineForEvent(ev.type, d);
    if (trace && trace.text) {
      pushLine(
        trace.text,
        ev,
        trace.tone || "",
        trace.urls || ev.urls || [],
        trace.lane || ev.lane || "",
        trace.media || null
      );
      return;
    }
    const narration = narrationForEvent(ev);
    if (narration) pushLine(narration, ev);
  });

  summary.lines = lines;
  summary.usedQuestion = usedQuestion;
  return summary;
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
  if (lowered.includes("executor")) return "executor";
  if (lowered.includes("planner")) return "orch";
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

function laneFromProfile(profile = "") {
  const lowered = String(profile || "").toLowerCase();
  if (!lowered) return "orch";
  if (lowered.includes("executor")) return "executor";
  if (lowered.includes("summarizer")) return "executor";
  if (lowered.includes("planner") || lowered.includes("orch")) return "orch";
  if (lowered.includes("recency")) return "recency";
  if (lowered.includes("adversarial")) return "adversarial";
  if (lowered.includes("primary")) return "primary";
  if (lowered.includes("verify")) return "verifier";
  return "orch";
}

function updateThinkingLine(text, runId = currentRunId) {
  if (!text) return;
  const targetRun = runId || currentRunId;
  if (!targetRun) return;
  const bubble = getAssistantBubble(targetRun);
  if (!bubble || !bubble.classList.contains("thinking")) return;
  const line = bubble.querySelector(".thinking-line");
  if (line) line.textContent = text;
}

function updateLiveTicker(text, urls = []) {
  const safeText = (text || "").trim();
  if (safeText) latestLiveLine = safeText;
  const ticker = el("liveTicker");
  if (ticker) {
    const links = urls && urls.length ? " | " + urls.slice(0, 3).map((u) => `<a href="${u}" target="_blank">${u}</a>`).join(", ") : "";
    ticker.innerHTML = escapeAndBreak(text) + links;
    ticker.classList.remove("pulse");
    void ticker.offsetWidth;
    ticker.classList.add("pulse");
  }
  if (safeText) updateThinkingLine(safeText);
}

function registerCitation(url, map) {
  if (!url) return null;
  if (!map.has(url)) {
    map.set(url, map.size + 1);
  }
  return map.get(url);
}

function citationLabel(url, meta = {}) {
  const host = hostFromUrl(url);
  if (host) return host;
  const title = String(meta.title || "").trim();
  const publisher = String(meta.publisher || "").trim();
  return title || publisher || String(url || "");
}

function citationTitle(url, meta = {}) {
  const title = String(meta.title || "").trim();
  if (title && title !== url && !/^https?:\/\//i.test(title)) return title;
  const publisher = String(meta.publisher || "").trim();
  const host = hostFromUrl(url);
  if (publisher && publisher !== host) return publisher;
  return "";
}

function citationTooltip(url, meta = {}) {
  const title = String(meta.title || "").trim();
  if (title) return title;
  return String(url || "");
}

function buildCitationState({ log = [], sources = [], visited = [], claims = [] } = {}) {
  const sourceMeta = {};
  const logUrls = [];
  const logSet = new Set();
  const claimUrls = [];
  const claimSet = new Set();
  const addMeta = (entry) => {
    if (entry && entry.url && !sourceMeta[entry.url]) sourceMeta[entry.url] = entry;
  };
  const addLogUrl = (url) => {
    if (!url || logSet.has(url)) return;
    logSet.add(url);
    logUrls.push(url);
  };
  const addClaimUrl = (url) => {
    if (!url || claimSet.has(url)) return;
    claimSet.add(url);
    claimUrls.push(url);
  };
  (sources || []).forEach(addMeta);
  (visited || []).forEach(addMeta);
  (log || []).forEach((entry) => {
    if (!entry || entry.text === "Sources cited") return;
    (entry.urls || []).forEach(addLogUrl);
  });
  (claims || []).forEach((claim) => {
    const urls = Array.isArray(claim.support_urls)
      ? claim.support_urls
      : Array.isArray(claim.urls)
      ? claim.urls
      : [];
    (urls || []).forEach(addClaimUrl);
  });
  const usedUrls = claimUrls.length ? claimUrls : logUrls;
  const usedSet = new Set(usedUrls);
  const citationMap = new Map();
  const register = (url) => {
    if (!url || citationMap.has(url)) return;
    citationMap.set(url, citationMap.size + 1);
  };
  usedUrls.forEach(register);
  (sources || []).forEach((entry) => register(entry && entry.url));
  (visited || []).forEach((entry) => register(entry && entry.url));
  return { citationMap, sourceMeta, usedSet };
}

function formatDuration(log = []) {
  const times = (log || []).map((h) => h.ts).filter(Boolean);
  if (times.length < 2) return null;
  return formatTime(Math.max(...times) - Math.min(...times));
}

function renderInlineCitations(urls = [], citationMap, sourceMeta, usedSet) {
  const uniq = Array.from(new Set((urls || []).filter(Boolean)));
  if (!uniq.length) return "";
  const bits = uniq.map((url) => {
    const idx = registerCitation(url, citationMap);
    if (url && !sourceMeta[url]) {
      sourceMeta[url] = { url, title: hostFromUrl(url) };
    }
    const meta = sourceMeta[url] || {};
    const labelText = citationLabel(url, meta);
    const titleText = citationTooltip(url, meta);
    const safeLabel = escapeHtml(labelText);
    const safeTitle = escapeHtml(titleText || labelText);
    const safeUrl = escapeHtml(url);
    const ariaLabel = escapeHtml(idx ? `Source ${idx}: ${labelText}` : `Source: ${labelText}`);
    return `<a class="cite-pill" href="${safeUrl}" target="_blank" rel="noopener noreferrer" aria-label="${ariaLabel}" title="${safeTitle}">${safeLabel}</a>`;
  });
  return bits.length ? " " + bits.join(" ") : "";
}

function renderCitationList(container, citationMap, sourceMeta, usedSet = new Set()) {
  if (!container || !citationMap.size) return;
  const block = document.createElement("div");
  block.className = "citations-block";
  const title = document.createElement("div");
  title.className = "label";
  title.textContent = "Citations";
  block.appendChild(title);
  const hasUsageSignal = usedSet && usedSet.size > 0;
  const entries = Array.from(citationMap.entries())
    .map(([url, idx]) => ({ url, idx, used: hasUsageSignal ? usedSet.has(url) : true }))
    .sort((a, b) => {
      if (a.used === b.used) return a.idx - b.idx;
      return a.used ? -1 : 1;
    });
  const used = entries.filter((entry) => entry.used);
  const unused = entries.filter((entry) => !entry.used);
  const renderSection = (label, items) => {
    const section = document.createElement("div");
    section.className = "citation-section";
    if (label) {
      const subtitle = document.createElement("div");
      subtitle.className = "citation-subtitle";
      subtitle.textContent = label;
      section.appendChild(subtitle);
    }
    const list = document.createElement("ol");
    list.className = "citation-list";
    items.forEach((entry) => {
      const li = document.createElement("li");
      const link = document.createElement("a");
      link.className = `citation-card${entry.used ? "" : " unused"}`;
      link.href = entry.url;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      const badge = document.createElement("span");
      badge.className = "cite-index";
      badge.textContent = `[${entry.idx}]`;
      const meta = sourceMeta[entry.url] || {};
      const hostText = citationLabel(entry.url, meta);
      const titleText = citationTitle(entry.url, meta);
      const tooltipText = citationTooltip(entry.url, meta);
      const metaWrap = document.createElement("span");
      metaWrap.className = "citation-meta";
      const host = document.createElement("span");
      host.className = "citation-host";
      host.textContent = hostText;
      metaWrap.appendChild(host);
      if (titleText && titleText !== hostText) {
        const sub = document.createElement("span");
        sub.className = "citation-title";
        sub.textContent = titleText;
        metaWrap.appendChild(sub);
      }
      link.title = tooltipText || entry.url;
      link.appendChild(badge);
      link.appendChild(metaWrap);
      li.appendChild(link);
      list.appendChild(li);
    });
    section.appendChild(list);
    return section;
  };
  if (used.length && unused.length) {
    block.appendChild(renderSection("Used in response", used));
    block.appendChild(renderSection("Other sources", unused));
  } else {
    block.appendChild(renderSection("", entries));
  }
  container.appendChild(block);
}

function renderSourceChips(urls = [], sourceMeta = {}) {
  if (!urls.length) return "";
  return urls
    .map((url) => {
      const meta = sourceMeta[url] || {};
      const labelText = citationLabel(url, meta);
      const titleText = citationTooltip(url, meta);
      const safeLabel = escapeHtml(labelText);
      const safeTitle = escapeHtml(titleText || labelText);
      const safeUrl = escapeHtml(url);
      return `<a class="source-pill" href="${safeUrl}" target="_blank" rel="noopener noreferrer" title="${safeTitle}">${safeLabel}</a>`;
    })
    .join("");
}

function renderSourcesRow(urls = [], sourceMeta = {}) {
  if (!urls.length) return "";
  const chips = renderSourceChips(urls, sourceMeta);
  return `<div class="sources-row"><span class="label">Sources</span><div class="sources-list">${chips}</div></div>`;
}

function renderHistory(runId = selectedRunId || currentRunId, logOverride = null) {
  const feed = el("activityFeed") || el("reasoningFeed");
  if (!feed) return;
  const envelope = (runId && runDetails[runId]) || {};
  const log =
    logOverride ||
    (runId
      ? (runEvents[runId] && runEvents[runId].length ? runEvents[runId] : envelope.activity_events || [])
      : historyLog);
  const sources = envelope.sources || [];
  const visited = envelope.visited_sources || [];
  const claims = envelope.claims || [];
  const { citationMap, sourceMeta, usedSet } = buildCitationState({ log, sources, visited, claims });
  feed.innerHTML = "";
  if (feed.id === "reasoningFeed") {
    const header = document.createElement("div");
    header.className = "thinking-header";
    header.textContent = "Thinking";
    feed.appendChild(header);
  }
  (log || []).forEach((h) => {
    const row = document.createElement("div");
    row.className = `thought ${h.tone || "info"}`;
    const body = document.createElement("div");
    const inlineCites = renderInlineCitations(h.urls || [], citationMap, sourceMeta, usedSet);
    const lineText = withNarrationPrefix(h.text, h.lane);
    body.innerHTML = escapeAndBreak(lineText) + inlineCites;
    row.appendChild(body);
    const mediaItems = Array.isArray(h.media) ? h.media : h.media ? [h.media] : [];
    if (mediaItems.length) {
      const mediaWrap = document.createElement("div");
      mediaWrap.className = "thought-media";
      mediaItems.forEach((item) => {
        if (!item || typeof item !== "object") return;
        if (item.type === "image" && item.src) {
          const fig = document.createElement("figure");
          fig.className = "media-card";
          const img = document.createElement("img");
          img.src = item.src;
          img.alt = item.title || "Chart";
          img.loading = "lazy";
          fig.appendChild(img);
          if (item.title) {
            const cap = document.createElement("figcaption");
            cap.textContent = item.title;
            fig.appendChild(cap);
          }
          mediaWrap.appendChild(fig);
        }
      });
      if (mediaWrap.childNodes.length) {
        row.appendChild(mediaWrap);
      }
    }
    feed.appendChild(row);
  });
  renderCitationList(feed, citationMap, sourceMeta, usedSet);
  feed.scrollTop = feed.scrollHeight;
}
function addHistory(text, lane = "orch", urls = [], tone = "info", runId = null, media = null) {
  const trimmed = String(text || "").trim();
  if (!trimmed) return;
  const entry = { text: trimmed, lane, urls, tone, ts: Date.now(), media };
  historyLog.push(entry);
  if (historyLog.length > 150) historyLog.shift();
  const targetRunId = runId || currentRunId;
  if (targetRunId) {
    if (!runEvents[targetRunId]) runEvents[targetRunId] = [];
    runEvents[targetRunId].push(entry);
    if (runEvents[targetRunId].length > 200) runEvents[targetRunId].shift();
  }
  renderHistory();
}

function pushReasoning(text, tone = "info", lane = "orch", urls = [], runId = null, media = null) {
  addHistory(text, lane, urls, tone, runId, media);
}

function renderResourceSnapshot(modelCheck = {}) {
  const resEl = el("resourceStats");
  const budgetEl = el("agentBudget");
  if (!resEl || !budgetEl) return;
  const resources = modelCheck.resources || {};
  const ram = resources.ram || {};
  const gpus = Array.isArray(resources.gpus) ? resources.gpus : [];
  const fmt = (v) => (typeof v === "number" ? v.toFixed(v >= 10 ? 0 : 1) : v || "?");
  const ramPart = ram.total_gb ? `RAM free ${fmt(ram.available_gb)}/${fmt(ram.total_gb)} GB` : "RAM: n/a";
  const gpuHealthy = gpus.filter((g) => g && !g.error && g.free_gb !== undefined);
  let gpuPart = "GPU: not detected";
  if (gpuHealthy.length) {
    gpuPart = gpuHealthy
      .map((g) => `${g.name || "GPU"} free ${fmt(g.free_gb)} GB (total ${fmt(g.total_gb)} GB)`)
      .join("; ");
  } else if (gpus.length && gpus[0].error) {
    gpuPart = `GPU error: ${gpus[0].error}`;
  }
  resEl.textContent = `${ramPart} | ${gpuPart}`;
  const budget = modelCheck.worker_slots || modelCheck.budget || {};
  if (budget.max_parallel) {
    const desired = budget.desired_parallel ? `, target ${budget.desired_parallel}` : "";
    const readyWorkers = Number(budget.ready_workers || 0);
    const readyVariants = Number(budget.ready_variants || 0);
    const readyPart = readyWorkers
      ? `, ready ${readyWorkers}${readyVariants ? ` (${readyVariants} models)` : ""}`
      : "";
    budgetEl.textContent = `Agent slots: ${budget.max_parallel}${desired} (config ${
      budget.configured || "?"
    }, variants ${budget.variants || "?"}${readyPart}, RAM cap ${budget.ram_slots || "-"}, VRAM cap ${
      budget.vram_slots || "-"
    })`;
  } else {
    budgetEl.textContent = "Agent slots: estimating...";
  }
}

async function requestRunStop(runId) {
  if (!runId) return;
  try {
    const res = await fetch(resolveEndpoint(`/api/run/${runId}/stop`), { method: "POST" });
    if (!res.ok) {
      appendActivity("Failed to stop the run.");
    }
  } catch (_) {
    appendActivity("Failed to stop the run.");
  }
}

async function loadSettings() {
  let data = null;
  syncApiBaseInput();
  try {
    const res = await fetch(resolveEndpoint("/settings"));
    if (!res.ok) throw new Error(`Settings ${res.status}`);
    data = await res.json();
  } catch (err) {
    const settingsStatus = el("settingsStatus");
    if (settingsStatus) settingsStatus.textContent = "Settings unavailable. Check the API base URL.";
    setStatus("API offline", "error");
    updateLiveTicker("API offline. Check the server URL.");
    return false;
  }
  const s = data.settings || {};
  settingsSnapshot = s || {};
  settingsDefaults = {
    search_depth_mode: s.search_depth_mode || "auto",
    strict_mode: !!s.strict_mode,
    auto_memory: true,
    evidence_dump: false,
    max_results_override: 0,
    stt_lang: "en-US",
  };
  const localPrefs = loadLocalPrefs();
  settingsDefaults = {
    ...settingsDefaults,
    ...localPrefs,
    strict_mode: !!s.strict_mode,
  };
  const baseUrlInput = el("cfgBaseUrl");
  if (baseUrlInput) baseUrlInput.value = s.lm_studio_base_url || "";
  const tavilyInput = el("cfgTavily");
  if (tavilyInput) tavilyInput.value = s.tavily_api_key || "";
  const searchModeInput = el("cfgSearchMode");
  if (searchModeInput) searchModeInput.value = s.search_depth_mode || "auto";
  const maxBaseInput = el("cfgMaxBase");
  if (maxBaseInput) maxBaseInput.value = s.max_results_base || 6;
  const maxHighInput = el("cfgMaxHigh");
  if (maxHighInput) maxHighInput.value = s.max_results_high || 10;
  const extractInput = el("cfgExtract");
  if (extractInput) extractInput.value = s.extract_depth || "basic";
  const discoveryInput = el("cfgDiscovery");
  if (discoveryInput) discoveryInput.value = (s.discovery_base_urls || []).join(", ");
  const strictToggle = el("cfgStrictMode");
  if (strictToggle) strictToggle.checked = settingsDefaults.strict_mode;
  const autoMemoryToggle = el("cfgAutoMemory");
  if (autoMemoryToggle) autoMemoryToggle.checked = settingsDefaults.auto_memory;
  const evidenceToggle = el("cfgEvidenceDump");
  if (evidenceToggle) evidenceToggle.checked = settingsDefaults.evidence_dump;
  const maxOverrideInput = el("cfgMaxOverride");
  if (maxOverrideInput) maxOverrideInput.value = settingsDefaults.max_results_override || 0;
  const sttLangInput = el("sttLang");
  if (sttLangInput) sttLangInput.value = settingsDefaults.stt_lang || "en-US";
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
  renderModelSelectors(s, data.model_check || {});
  // Auto-discover if models are missing and we haven't tried yet.
  const baseUrls = discoveryInput ? discoveryInput.value.split(",").map((v) => v.trim()).filter(Boolean) : [];
  if (!triedAutoDiscover && baseUrls.length && missingRoles.length) {
    triedAutoDiscover = true;
    const settingsStatus = el("settingsStatus");
    if (settingsStatus) settingsStatus.textContent = "Auto-discovering available models...";
    await autoDiscoverAndReport(baseUrls);
  }
  renderResourceSnapshot(data.model_check || {});
  toggleModal(false);
  return true;
}

function renderModelSelectors(settings, modelCheck) {
  MODEL_ENDPOINT_FIELDS.forEach((field) => {
    const select = el(field.selectId);
    if (!select) return;
    const current = settings?.[field.key]?.model_id || "";
    const available = modelCheck?.[field.role]?.available || [];
    const options = Array.from(new Set([current, ...available].filter(Boolean)));
    select.innerHTML = "";
    if (!options.length) {
      const opt = document.createElement("option");
      opt.value = current || "";
      opt.textContent = current || "No models detected";
      select.appendChild(opt);
      select.disabled = true;
      return;
    }
    select.disabled = false;
    options.forEach((id) => {
      const opt = document.createElement("option");
      opt.value = id;
      opt.textContent = id;
      select.appendChild(opt);
    });
    select.value = current || options[0];
  });
}

function toggleModal(show) {
  const modal = el("settingsModal");
  if (!modal) return;
  modal.classList[show ? "remove" : "add"]("hidden");
}

function toggleMemoryModal(show) {
  const modal = el("memoryModal");
  if (!modal) return;
  modal.classList[show ? "remove" : "add"]("hidden");
}

function renderMemoryList(items = []) {
  const list = el("memoryList");
  if (!list) return;
  list.innerHTML = "";
  if (!items.length) {
    const empty = document.createElement("div");
    empty.className = "conversation-empty";
    empty.textContent = "No memory saved yet.";
    list.appendChild(empty);
    return;
  }
  items.forEach((item) => {
    const card = document.createElement("div");
    card.className = "memory-card";
    const title = document.createElement("h4");
    title.textContent = item.title || "Untitled memory";
    const content = document.createElement("div");
    content.className = "memory-content";
    content.textContent = item.content || "";
    const tags = document.createElement("div");
    tags.className = "memory-tags";
    const tagList = (item.tags || []).join(", ");
    tags.textContent = tagList ? `Tags: ${tagList}` : "Tags: none";
    const actions = document.createElement("div");
    actions.className = "memory-actions";
    const pinBtn = document.createElement("button");
    pinBtn.type = "button";
    pinBtn.className = "pill-btn ghost";
    pinBtn.textContent = item.pinned ? "Unpin" : "Pin";
    pinBtn.onclick = async () => {
      await fetch(resolveEndpoint(`/api/memory/${item.id}`), {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pinned: !item.pinned }),
      });
      loadMemory(el("memorySearch")?.value || "");
    };
    const delBtn = document.createElement("button");
    delBtn.type = "button";
    delBtn.className = "pill-btn ghost";
    delBtn.textContent = "Delete";
    delBtn.onclick = async () => {
      if (!confirm("Delete this memory item?")) return;
      await fetch(resolveEndpoint(`/api/memory/${item.id}`), { method: "DELETE" });
      loadMemory(el("memorySearch")?.value || "");
    };
    actions.appendChild(pinBtn);
    actions.appendChild(delBtn);
    card.appendChild(title);
    card.appendChild(content);
    card.appendChild(tags);
    card.appendChild(actions);
    list.appendChild(card);
  });
}

async function loadMemory(query = "") {
  try {
    const q = query.trim();
    const endpoint = q ? `/api/memory?q=${encodeURIComponent(q)}` : "/api/memory";
    const res = await fetch(resolveEndpoint(endpoint));
    if (!res.ok) return;
    const data = await res.json();
    renderMemoryList(data.items || []);
  } catch (_) {
    // ignore memory load errors
  }
}

async function saveSettings() {
  const apiBaseInput = el("cfgApiBase");
  const rawApiBase = apiBaseInput ? apiBaseInput.value.trim() : "";
  const wantsBaseUpdate = !!rawApiBase;
  if (apiBaseInput) {
    const nextApiBase = rawApiBase || apiBaseUrl || APP_BASE_URL.href;
    if (!setApiBaseUrl(nextApiBase)) {
      el("settingsStatus").textContent = "Invalid API base URL.";
      return;
    }
  }
  if (!settingsSnapshot) {
    const apiOk = await bootApp();
    const status = el("settingsStatus");
    if (status) {
      status.textContent = apiOk
        ? wantsBaseUpdate
          ? "API base updated."
          : "API connected."
        : wantsBaseUpdate
          ? "API base saved. Server still offline."
          : "API offline. Check the server URL.";
    }
    return;
  }
  const tavKey = el("cfgTavily").value;
  const baseUrl = el("cfgBaseUrl").value;
  const snapshot = settingsSnapshot || {};
  const priorBaseUrl = snapshot.lm_studio_base_url || baseUrl;
  const payload = {
    lm_studio_base_url: baseUrl,
    tavily_api_key: tavKey === "********" ? undefined : tavKey,
    search_depth_mode: el("cfgSearchMode").value,
    max_results_base: Number(el("cfgMaxBase").value),
    max_results_high: Number(el("cfgMaxHigh").value),
    extract_depth: el("cfgExtract").value,
    discovery_base_urls: el("cfgDiscovery").value.split(",").map((v) => v.trim()).filter(Boolean),
  };
  const strictToggle = el("cfgStrictMode");
  if (strictToggle) payload.strict_mode = strictToggle.checked;
  const maxOverrideRaw = el("cfgMaxOverride") ? Number(el("cfgMaxOverride").value) : settingsDefaults.max_results_override;
  const localPrefs = {
    auto_memory: el("cfgAutoMemory") ? el("cfgAutoMemory").checked : settingsDefaults.auto_memory,
    evidence_dump: el("cfgEvidenceDump") ? el("cfgEvidenceDump").checked : settingsDefaults.evidence_dump,
    max_results_override: Number.isFinite(maxOverrideRaw) ? maxOverrideRaw : 0,
    stt_lang: (el("sttLang")?.value || "en-US").trim() || "en-US",
  };
  settingsDefaults = {
    ...settingsDefaults,
    ...localPrefs,
    strict_mode: payload.strict_mode ?? settingsDefaults.strict_mode,
  };
  saveLocalPrefs(localPrefs);
  MODEL_ENDPOINT_FIELDS.forEach((field) => {
    const current = snapshot[field.key];
    if (!current) return;
    const select = el(field.selectId);
    const selectedModel = select?.value || current.model_id;
    if (!selectedModel) return;
    const nextBaseUrl = current.base_url && current.base_url !== priorBaseUrl ? current.base_url : baseUrl;
    payload[field.key] = { base_url: nextBaseUrl, model_id: selectedModel };
  });
  const res = await fetch(resolveEndpoint("/settings"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  el("settingsStatus").textContent = res.ok ? "Saved. Restart runs to apply." : "Save failed.";
  if (res.ok) loadSettings();
}

async function discoverModels() {
  const base_urls = el("cfgDiscovery").value.split(",").map((v) => v.trim()).filter(Boolean);
  const res = await fetch(resolveEndpoint("/api/discover"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ base_urls }),
  });
  const data = await res.json();
  el("settingsStatus").textContent = JSON.stringify(data.results);
}

async function autoDiscoverAndReport(base_urls) {
  try {
    const res = await fetch(resolveEndpoint("/api/discover"), {
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

const MODEL_ENDPOINT_FIELDS = [
  { key: "orch_endpoint", role: "orch", selectId: "cfgModelOrch" },
  { key: "worker_a_endpoint", role: "worker", selectId: "cfgModelWorker" },
  { key: "worker_b_endpoint", role: "worker_b", selectId: "cfgModelWorkerB" },
  { key: "worker_c_endpoint", role: "worker_c", selectId: "cfgModelWorkerC" },
  { key: "fast_endpoint", role: "fast", selectId: "cfgModelFast" },
  { key: "deep_planner_endpoint", role: "deep_planner", selectId: "cfgModelDeepPlanner" },
  { key: "deep_orchestrator_endpoint", role: "deep_orch", selectId: "cfgModelDeepOrch" },
  { key: "router_endpoint", role: "router", selectId: "cfgModelRouter" },
  { key: "summarizer_endpoint", role: "summarizer", selectId: "cfgModelSummarizer" },
  { key: "verifier_endpoint", role: "verifier", selectId: "cfgModelVerifier" },
];

function renderReasoningOptions(tier) {
  const select = el("reasoningLevel");
  if (!select) return;
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

function setTier(tier, opts = {}) {
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
  if (opts.persist !== false) scheduleConversationSettingsUpdate();
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
  if (promptLocked) {
    appendActivity("End the current prompt to start a new one.");
    return;
  }
  if (!activeConversationId) {
    const convo = await createConversation();
    if (!convo) return;
  }
  const question = el("question").value.trim();
  if (!question) return;
  const uploading = pendingUploads.some((u) => u.status === "uploading");
  if (uploading) {
    appendActivity("Wait for uploads to finish before starting the run.");
    return;
  }
  startNewConversation({ keepQuestion: true, silent: true, keepUploads: true, preserveChat: true });
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

  const settings = getReasoningSettings();
  const reasoningAuto = settings.reasoningMode === "auto";
  const payload = {
    question,
    conversation_id: activeConversationId,
    reasoning_mode: settings.reasoningMode,
    manual_level: settings.manualLevel,
    model_tier: currentTier,
    deep_mode: currentTier === "deep" ? settings.deepRoute : "auto",
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

  const res = await fetch(resolveEndpoint("/api/run"), {
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
  applyPromptState({ prompt_text: question, run_id: data.run_id }, { clear: false });
}

function handleEvent(type, p) {
  const trace = traceLineForEvent(type, p);
  if (trace) {
    pushReasoning(
      trace.text,
      trace.tone || "info",
      trace.lane || "orch",
      trace.urls || [],
      p.run_id,
      trace.media || null
    );
  }
  switch (type) {
    case "narration":
      if (p && p.text) {
        updateLiveTicker(withNarrationPrefix(p.text, "executor"), p.urls || []);
      }
      break;
    case "work_log":
      if (p && p.text) {
        updateLiveTicker(p.text, p.urls || []);
      }
      break;
    case "dev_trace":
      break;
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
      resetLiveAgentState();
      setExecutorActive(true);
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
    case "resource_budget":
      renderResourceSnapshot({ resources: p.resources, worker_slots: p.budget || p.worker_slots });
      {
        const budget = p.budget || p.worker_slots || p;
        const desired = Number(p.desired_parallel || budget?.desired_parallel || 0);
        const maxSlots = Number(budget?.max_parallel || budget?.max || budget?.slots || 0);
        multiAgentMode = Math.max(desired, maxSlots) > 1;
      }
      queueLiveEvent("resource_budget", p.budget || p, "orch");
      break;
    case "team_roster":
      queueLiveEvent("team_roster", p, "orch");
      break;
    case "executor_brief":
      queueLiveEvent("executor_brief", p, "executor");
      break;
    case "allocator_decision":
      queueLiveEvent("allocator_decision", p, "executor");
      break;
    case "tool_request": {
      const lane = laneFrom("", p.step);
      queueLiveEvent("tool_request", p, lane);
      break;
    }
    case "tool_result": {
      const lane = laneFrom("", p.step);
      queueLiveEvent("tool_result", p, lane);
      break;
    }
    case "planner_verifier":
      queueLiveEvent("planner_verifier", p, "verifier");
      break;
    case "model_selected":
    case "model_unavailable":
    case "model_error": {
      const lane = laneFromProfile(p.profile || "");
      queueLiveEvent(type, p, lane || "orch");
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
    case "plan_updated":
      if (typeof p.expected_total_steps !== "undefined") {
        totalSteps = Number(p.expected_total_steps || totalSteps);
      } else if (typeof p.steps !== "undefined") {
        totalSteps = Number(p.steps || totalSteps);
      }
      if (typeof p.completed_reset_to === "number") {
        completedSteps = Number(p.completed_reset_to);
      }
      updateProgressUI();
      queueLiveEvent("plan_updated", p, "orch");
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
      {
        noteAgentStepStarted(p);
        const lane = laneFromProfile(p.agent_profile || "") || laneFrom(p.name, p.step_id);
        queueLiveEvent(
          "step_started",
          { step_id: p.step_id, name: p.name, type: p.type, agent_profile: p.agent_profile },
          lane
        );
      }
      break;
    case "tavily_search":
      queueLiveEvent(
        "tavily_search",
        {
          query: p.query,
          urls: p.urls || [],
          result_count: p.result_count,
          new_sources: p.new_sources,
          duplicate_sources: p.duplicate_sources,
        },
        laneFrom("", p.step)
      );
      break;
    case "tavily_extract":
      queueLiveEvent("tavily_extract", { urls: p.urls || [] }, laneFrom("", p.step));
      break;
    case "tavily_error": {
      const lane = laneFrom("", p.step);
      queueLiveEvent("tavily_error", { message: p.message || "Tavily unavailable" }, lane);
      break;
    }
    case "search_skipped": {
      const lane = laneFrom("", p.step);
      queueLiveEvent("search_skipped", { query: p.query, reason: p.reason || "" }, lane);
      break;
    }
    case "source_found": {
      const lane = laneFromProfile(p.lane || p.agent_profile || "");
      queueLiveEvent("source_found", p, lane || "orch");
      break;
    }
    case "claim_found": {
      const lane = laneFromProfile(p.lane || p.agent_profile || "");
      queueLiveEvent("claim_found", p, lane || "orch");
      break;
    }
    case "step_completed":
      noteAgentStepFinished(p);
      completedSteps = Math.min(totalSteps || completedSteps + 1, completedSteps + 1);
      updateProgressUI();
      if (currentRunId) {
        fetchArtifacts(currentRunId, { liveOnly: true, skipChat: true }).catch(() => {});
      }
      {
        const lane = laneFromProfile(p.agent_profile || "") || laneFrom(p.name, p.step_id);
        queueLiveEvent(
          "step_completed",
          { step_id: p.step_id, name: p.name, type: p.type, agent_profile: p.agent_profile },
          lane
        );
      }
      break;
    case "control_action":
      queueLiveEvent(
        "control_action",
        { control: p.control || p.action_type || "", origin: p.origin || "" },
        "orch"
      );
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
      resetLiveAgentState();
      stopEventPolling();
      if (p?.stopped) {
        if (evtSource) {
          try {
            evtSource.close();
          } catch (_) {}
          evtSource = null;
        }
        setStatus("Stopped", "error");
        updateLiveTicker("Generation stopped");
      } else {
        setStatus("Done", "done");
      }
      stopTimer();
      completedSteps = totalSteps || completedSteps;
      updateProgressUI();
      fetchArtifacts(currentRunId);
      queueLiveEvent("archived", p, "orch");
      break;
    case "step_error": {
      const safe = p || {};
      noteAgentStepFinished(safe);
      const lane = laneFromProfile(safe.agent_profile || "") || laneFrom(safe.name || "", safe.step);
      const label = safe.name || (safe.step ? `Step ${safe.step}` : "Step");
      const msg = safe.message || "error encountered";
      queueLiveEvent(
        "step_error",
        { step_id: safe.step, name: safe.name, type: safe.type, agent_profile: safe.agent_profile, message: msg },
        lane
      );
      completedSteps = Math.min(totalSteps || completedSteps + 1, completedSteps + 1);
      updateProgressUI();
      break;
    }
    case "error": {
      const safe = p || {};
      const recoverable = safe.fatal === false || typeof safe.step !== "undefined";
      const lane = laneFromProfile(safe.agent_profile || "") || laneFrom(safe.name || "", safe.step);
      if (recoverable) {
        noteAgentStepFinished(safe);
        queueLiveEvent(
          "step_error",
          {
            step_id: safe.step,
            name: safe.name,
            type: safe.type,
            agent_profile: safe.agent_profile,
            message: safe.message || "recoverable error",
          },
          lane
        );
        completedSteps = Math.min(totalSteps || completedSteps + 1, completedSteps + 1);
        updateProgressUI();
        break;
      }
      resetLiveAgentState();
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

function ingestRunEvent(data) {
  if (!data || !data.event_type) return;
  const seq = Number(data.seq);
  if (Number.isFinite(seq) && seq > 0) {
    if (seq <= lastEventSeq) return;
    lastEventSeq = seq;
  }
  handleEvent(data.event_type, data.payload || {});
}

function stopEventPolling() {
  pollingEvents = false;
  if (eventPollTimer) {
    clearTimeout(eventPollTimer);
    eventPollTimer = null;
  }
}

async function pollRunEvents(runId) {
  if (!pollingEvents || !runId) return;
  try {
    const url = resolveEndpoint(`/api/run/${runId}/events?after_seq=${encodeURIComponent(lastEventSeq)}`);
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Events ${res.status}`);
    const data = await res.json();
    const events = Array.isArray(data.events) ? data.events : [];
    events.forEach((ev) => ingestRunEvent(ev));
  } catch (_) {
    // Ignore polling errors and try again.
  } finally {
    if (pollingEvents) {
      eventPollTimer = setTimeout(() => pollRunEvents(runId), 2000);
    }
  }
}

function startEventPolling(runId) {
  if (!runId || pollingEvents) return;
  pollingEvents = true;
  pollRunEvents(runId);
}

function subscribeGlobalEvents() {
  if (globalEventSource) globalEventSource.close();
  globalEventSource = new EventSource(resolveEndpoint("/events"));
  globalEventSource.onmessage = async (evt) => {
    let data = null;
    try {
      data = JSON.parse(evt.data);
    } catch (_) {
      return;
    }
    if (!data) return;
    if (data.event_type === "conversation_created" || data.event_type === "conversation_updated") {
      scheduleConversationRefresh();
      return;
    }
    if (data.event_type === "conversation_deleted") {
      const deletedId = data?.payload?.conversation_id;
      if (deletedId && deletedId === activeConversationId) {
        activeConversationId = null;
        storeActiveConversationId(null);
      }
      await fetchConversations({ ensureActive: true });
      return;
    }
    if (data.event_type === "conversation_reset") {
      scheduleConversationRefresh();
      return;
    }
    if (data.event_type === "prompt_updated") {
      applyPromptState(data.payload || null, { clear: false });
      return;
    }
    if (data.event_type === "prompt_cleared") {
      applyPromptState(null, { clear: true });
      return;
    }
    const runId = data?.payload?.run_id || data?.run_id || null;
    const convoId = data?.payload?.conversation_id || null;
    if (!runId || !convoId) return;
    if (convoId !== activeConversationId) {
      if (data.event_type === "message_added" || data.event_type === "run_started") {
        unreadConversations.add(convoId);
        renderConversationList();
        scheduleConversationRefresh();
      }
      return;
    }
    if (data.event_type === "message_added" || data.event_type === "run_started") {
      scheduleConversationRefresh();
    }
    if (runId !== currentRunId && (data.event_type === "message_added" || data.event_type === "run_started")) {
      await switchToRun(runId, { clearChat: false, fromPoll: true, resetState: false, skipThinking: true });
    }
  };
}

function subscribeEvents(runId) {
  stopEventPolling();
  lastEventSeq = 0;
  if (evtSource) evtSource.close();
  try {
    evtSource = new EventSource(resolveEndpoint(`/runs/${runId}/events`));
  } catch (_) {
    evtSource = null;
    startEventPolling(runId);
    return;
  }
  evtSource.onmessage = async (evt) => {
    let data = null;
    try {
      data = JSON.parse(evt.data);
    } catch (_) {
      return;
    }
    ingestRunEvent(data);
  };
  evtSource.onerror = () => {
    if (pollingEvents) return;
    if (evtSource) {
      try {
        evtSource.close();
      } catch (_) {}
    }
    evtSource = null;
    startEventPolling(runId);
  };
}

async function followLatestRun(force = false) {
  if (!activeConversationId) return;
  try {
    const res = await fetch(
      resolveEndpoint(`/api/run/latest?conversation_id=${encodeURIComponent(activeConversationId)}`)
    );
    if (!res.ok) return;
    const data = await res.json();
    const latestId = data?.run?.run_id || null;
    if (!latestId) return;
    if (force || latestId !== currentRunId) {
      await switchToRun(latestId, { clearChat: false, fromPoll: true, resetState: false, skipThinking: true });
    }
  } catch (err) {
    // ignore polling errors
  }
}

function renderEvidence(claims) {
  // Evidence is now summarized into the reasoning stream; no separate panel.
  if (!claims || !claims.length) return;
  // Keep narration focused on execution steps.
}

function hostFromUrl(url) {
  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch {
    const raw = String(url || "").trim();
    if (!raw) return "";
    const cleaned = raw.replace(/^[a-z]+:\/\//i, "");
    const host = cleaned.split("/")[0];
    return host.split("?")[0].split("#")[0].replace(/^www\./, "");
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

const FINAL_TEXT_KEYS = [
  "final_text",
  "final_answer",
  "finalAnswer",
  "final_response",
  "finalResponse",
  "draft_answer",
  "answer",
  "final",
  "result",
  "output",
  "response",
  "text",
  "message",
  "content",
];

function extractFinalText(raw) {
  if (raw === null || raw === undefined) return "";
  if (typeof raw === "string") {
    const trimmed = raw.trim();
    if (!trimmed) return "";
    if ((trimmed.startsWith("{") && trimmed.endsWith("}")) || (trimmed.startsWith("[") && trimmed.endsWith("]"))) {
      const parsed = tryParseJson(trimmed);
      if (parsed) {
        const extracted = extractFinalText(parsed);
        return extracted || trimmed;
      }
      return trimmed;
    }
    return trimmed;
  }
  if (Array.isArray(raw)) {
    for (const item of raw) {
      const extracted = extractFinalText(item);
      if (extracted) return extracted;
    }
    return "";
  }
  if (typeof raw === "object") {
    for (const key of FINAL_TEXT_KEYS) {
      if (!Object.prototype.hasOwnProperty.call(raw, key)) continue;
      const extracted = extractFinalText(raw[key]);
      if (extracted) return extracted;
    }
    return "";
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
  const finalText = extractFinalText(run.final_answer) || "";
  return {
    run_id: run.run_id,
    final_text: finalText,
    final_format: "markdown",
    message_meta: {
      used_model: routerDecision.model_tier ? tierLabel(routerDecision.model_tier) : tierLabel(currentTier),
      reasoning_level: routerDecision.reasoning_level || run.reasoning_mode || "AUTO",
      confidence: run.confidence || "",
    },
    run_meta: run,
    reasoning_summary: buildReasoningSummary(run, sources, claims, routerDecision),
    activity_events: runEvents[run.run_id] || [],
    sources,
    claims,
    visited_sources: data.visited_sources || [],
  };
}

function addMessageActions(bubble, runId, envelope) {
  if (!bubble) return;
  const existing = bubble.querySelector(".bubble-actions");
  if (existing) existing.remove();
  const actions = document.createElement("div");
  actions.className = "bubble-actions";
  const activityBtn = document.createElement("button");
  activityBtn.type = "button";
  activityBtn.className = "bubble-action";
  activityBtn.textContent = "Thinking";
  activityBtn.onclick = () => openDrawer(runId);
  const copyBtn = document.createElement("button");
  copyBtn.type = "button";
  copyBtn.className = "bubble-action";
  copyBtn.textContent = "Copy";
  copyBtn.onclick = async () => {
    const fallbackText = bubble?.querySelector(".bubble-body")?.textContent || "";
    const text = (envelope?.final_text || fallbackText || "").trim();
    if (!text) return;
    const ok = await copyToClipboard(text);
    if (ok) {
      copyBtn.textContent = "Copied";
      setTimeout(() => {
        copyBtn.textContent = "Copy";
      }, 1200);
    }
  };
  actions.appendChild(activityBtn);
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
  if (!panel) return false;
  if (isDesktopDrawer()) return !panel.classList.contains("hidden");
  return panel.classList.contains("open");
}

function populateDrawer(runId = selectedRunId, logOverride = null) {
  const panel = el("reasoningPanel");
  if (!panel || !runId) return;
  const envelope = runDetails[runId] || {};
  const sub = el("drawerSubline");
  const idx = getResponseIndex(runId);
  if (sub) sub.textContent = "";
  const log =
    logOverride ||
    (runEvents[runId] && runEvents[runId].length ? runEvents[runId] : envelope.activity_events || []);
  const durationEl = el("drawerDuration");
  if (durationEl) durationEl.textContent = formatDuration(log) || "";
  renderHistory(runId, log);
  highlightSelectedMessage(runId);
}

function openDrawer(runId = null) {
  const panel = el("reasoningPanel");
  const overlay = el("drawerOverlay");
  if (!panel || !overlay) return;
  const targetRun = runId || selectedRunId || lastAssistantRunId || currentRunId;
  selectedRunId = targetRun;
  panel.classList.add("open");
  panel.classList.remove("hidden");
  if (!isDesktopDrawer()) {
    overlay.classList.remove("hidden");
  }
  populateDrawer(targetRun);
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
  const res = await fetch(resolveEndpoint(`/api/run/${runId}/artifacts`));
  if (!res.ok) return;
  const data = await res.json();
  cachedArtifacts = data;
  syncUploadsFromServer(data.uploads || []);
  if (opts.clearChat) {
    el("chatThread").innerHTML = "";
    lastAssistantRunId = null;
  }
  const run = data.run;
  syncExecutorFromRun(run || {});
  const sources = data.sources || [];
  const claims = data.claims || [];
  const verifier = data.verifier || {};
  const envelope = buildAnswerEnvelope(data);
  runDetails[run.run_id] = envelope;
  const hasFinalText = Boolean((envelope.final_text || "").trim());
  if (!opts.skipChat && hasFinalText) {
    renderAssistantAnswer(run.run_id, envelope);
  }
  lastAssistantRunId = run.run_id;
  el("confidence").textContent = run.confidence ? `Confidence: ${run.confidence}` : "";
  if (sources.length) {
    const citationState = buildCitationState({ log: runEvents[run.run_id] || [], sources, claims });
    const sourceOrder = Array.from(citationState.citationMap.keys());
    el("sources").innerHTML = renderSourcesRow(sourceOrder, citationState.sourceMeta);
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
    // Keep narration focused on execution steps; skip findings summaries here.
  }
  if (selectedRunId === run.run_id && isDrawerOpen()) {
    populateDrawer(run.run_id);
  }
}

async function switchToRun(runId, opts = {}) {
  if (!runId) return;
  const clearChat = opts.clearChat === true;
  const fromPoll = opts.fromPoll || false;
  const resetState = opts.resetState !== false;
  const skipThinking = opts.skipThinking === true;
  if (currentRunId === runId && !opts.force) return;
  if (evtSource) {
    evtSource.close();
    evtSource = null;
  }
  if (resetState) {
    startNewConversation({ keepQuestion: true, silent: true, keepUploads: true, preserveChat: !clearChat });
  } else {
    resetLiveAgentState();
  }
  {
    const activity = el("activityFeed");
    if (activity) activity.innerHTML = "";
  }
  currentRunId = runId;
  subscribeEvents(runId);
  if (!skipThinking) ensureThinkingPlaceholder(runId);
  await fetchArtifacts(runId, { clearChat });
  if (isDesktopDrawer()) {
    openDrawer(runId);
  }
  if (fromPoll) {
    setStatus("Syncing", "live");
    updateLiveTicker(`Attached to shared run ${runId}`);
  }
}

function updateCharCount() {
  const q = el("question");
  const counter = el("charCount");
  if (!q || !counter) return;
  counter.textContent = `${q.value.length} chars`;
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
    setQuestionValue((sttBuffer + interim).trim());
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

document.addEventListener("DOMContentLoaded", async () => {
  updateShareLinks();
  syncApiBaseInput();
  document.querySelectorAll("#modelTierGroup .seg-btn").forEach((btn) => {
    btn.addEventListener("click", () => setTier(btn.dataset.tier));
  });
  setTier(currentTier, { persist: false });
  closeDrawer();
  syncDrawerLayout();
  hideUploadPanelIfIdle();
  updateEmptyState();
  const questionInput = el("question");
  document.querySelectorAll(".suggestion-card").forEach((card) => {
    card.addEventListener("click", () => {
      const prompt = card.dataset.prompt || card.textContent || "";
      if (!questionInput) return;
      if (setQuestionValue(prompt)) questionInput.focus();
    });
  });
  const chatForm = el("chatForm");
  if (chatForm) chatForm.addEventListener("submit", startRun);
  const newChat = async () => {
    await createConversation();
  };
  const newConversationBtn = el("newConversationBtn");
  if (newConversationBtn) newConversationBtn.addEventListener("click", newChat);
  const panelNewConversationBtn = el("panelNewConversationBtn");
  if (panelNewConversationBtn) panelNewConversationBtn.addEventListener("click", newChat);
  const mobileNewConversationBtn = el("mobileNewConversationBtn");
  if (mobileNewConversationBtn) mobileNewConversationBtn.addEventListener("click", newChat);
  const endPromptBtn = el("endPromptBtn");
  if (endPromptBtn) endPromptBtn.addEventListener("click", endPrompt);
  const settingsBtn = el("settingsBtn");
  if (settingsBtn) settingsBtn.addEventListener("click", () => toggleModal(true));
  const closeSettings = el("closeSettings");
  if (closeSettings) closeSettings.addEventListener("click", () => toggleModal(false));
  const closeSettingsFooter = el("closeSettingsFooter");
  if (closeSettingsFooter) closeSettingsFooter.addEventListener("click", () => toggleModal(false));
  const settingsModal = el("settingsModal");
  if (settingsModal) {
    settingsModal.addEventListener("click", (e) => {
      if (e.target.id === "settingsModal") toggleModal(false);
    });
  }
  const memoryBtn = el("memoryBtn");
  if (memoryBtn) memoryBtn.addEventListener("click", () => {
    toggleMemoryModal(true);
    loadMemory(el("memorySearch")?.value || "");
  });
  const closeMemory = el("closeMemory");
  if (closeMemory) closeMemory.addEventListener("click", () => toggleMemoryModal(false));
  const memoryModal = el("memoryModal");
  if (memoryModal) {
    memoryModal.addEventListener("click", (e) => {
      if (e.target.id === "memoryModal") toggleMemoryModal(false);
    });
  }
  const refreshMemory = el("refreshMemory");
  if (refreshMemory) refreshMemory.addEventListener("click", () => loadMemory(el("memorySearch")?.value || ""));
  const memorySearch = el("memorySearch");
  if (memorySearch) {
    memorySearch.addEventListener("input", (e) => {
      const value = e.target.value || "";
      if (memorySearchTimer) clearTimeout(memorySearchTimer);
      memorySearchTimer = setTimeout(() => {
        loadMemory(value);
      }, 250);
    });
  }
  document.addEventListener("keyup", (e) => {
    if (e.key === "Escape") {
      toggleModal(false);
      toggleMemoryModal(false);
      closeDrawer();
      setSidebarOpen(false);
    }
  });
  const saveSettingsBtn = el("saveSettings");
  if (saveSettingsBtn) saveSettingsBtn.addEventListener("click", saveSettings);
  const discoverBtn = el("discoverBtn");
  if (discoverBtn) discoverBtn.addEventListener("click", discoverModels);

  const reasoningLevel = el("reasoningLevel");
  if (reasoningLevel)
    reasoningLevel.addEventListener("change", () => {
      updateReasoningBadge();
      scheduleConversationSettingsUpdate();
    });
  const deepRouteSelect = el("deepRoute");
  if (deepRouteSelect) {
    deepRouteSelect.addEventListener("change", () => {
      updateReasoningBadge();
      scheduleConversationSettingsUpdate();
    });
  }
  const conversationSearchInput = el("conversationSearch");
  if (conversationSearchInput) {
    conversationSearchInput.addEventListener("input", (e) => {
      conversationSearch = e.target.value || "";
      renderConversationList();
    });
  }
  const conversationTitle = el("conversationTitle");
  if (conversationTitle) {
    conversationTitle.addEventListener("click", async () => {
      if (!activeConversationId) return;
      const current = conversationTitle.textContent || "";
      const next = prompt("Rename this chat:", current);
      if (next !== null) {
        await renameConversation(activeConversationId, next);
      }
    });
  }
  const liveTicker = el("liveTicker");
  if (liveTicker) liveTicker.addEventListener("click", () => openDrawer(selectedRunId || currentRunId));
  const drawerClose = el("drawerClose");
  if (drawerClose) {
    drawerClose.addEventListener("click", () => closeDrawer());
  }
  const activityToggleBtn = el("activityToggleBtn");
  if (activityToggleBtn) {
    activityToggleBtn.addEventListener("click", () => {
      if (isDrawerOpen()) {
        closeDrawer();
      } else {
        openDrawer(selectedRunId || currentRunId);
      }
    });
  }
  const drawerOverlay = el("drawerOverlay");
  if (drawerOverlay) drawerOverlay.addEventListener("click", closeDrawer);
  const sidebarToggleBtn = el("sidebarToggleBtn");
  if (sidebarToggleBtn) {
    sidebarToggleBtn.addEventListener("click", () => setSidebarOpen(true));
  }
  const sidebarOverlay = el("sidebarOverlay");
  if (sidebarOverlay) {
    sidebarOverlay.addEventListener("click", () => setSidebarOpen(false));
  }
  window.addEventListener("resize", () => {
    syncDrawerLayout();
    if (isDesktopDrawer()) setSidebarOpen(false);
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
  const fileInput = el("fileInput");
  if (attachMenu)
    attachMenu.addEventListener("click", () => {
      uploadPanelVisible = true;
      showUploadPanel();
      if (fileInput) fileInput.click();
    });
  const attachBtn = el("attachBtn");
  if (attachBtn)
    attachBtn.addEventListener("click", () => {
      uploadPanelVisible = true;
      showUploadPanel();
      if (fileInput) fileInput.click();
    });
  if (fileInput) {
    fileInput.addEventListener("change", (e) => {
      uploadPanelVisible = true;
      showUploadPanel();
      handleFileInput(e.target.files);
    });
  }

  const micBtn = el("micBtn");
  if (micBtn)
    micBtn.addEventListener("click", () => {
      if (sttActive && sttRecognition) {
        sttRecognition.stop();
      } else {
        setupSTT();
      }
    });
  if (questionInput) questionInput.addEventListener("input", updateCharCount);
  const thread = el("chatThread");
  if (thread) {
    thread.addEventListener("click", (e) => {
      const bubble = e.target.closest(".bubble-assistant");
      if (!bubble) return;
      const runId = bubble.dataset.runId;
      selectedRunId = runId || selectedRunId;
      highlightSelectedMessage(runId);
      if (isDrawerOpen()) populateDrawer(runId);
    });
  }

  await bootApp();

  document.addEventListener("keydown", (e) => {
    if (e.ctrlKey && e.key.toLowerCase() === "k") {
      e.preventDefault();
      toggleModal(true);
    }
    if (questionInput && e.key === "Enter" && !e.shiftKey && document.activeElement === questionInput) {
      e.preventDefault();
      if (chatForm) chatForm.dispatchEvent(new Event("submit"));
    }
  });

  updateCharCount();
  renderAttachments();
});
