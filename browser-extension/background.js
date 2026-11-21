importScripts("service-config.js");

const MAX_HTML_CHARS = 2 * 1024 * 1024;
const CACHE_TTL_MS = 10 * 60 * 1000;
const jobs = new Map();

function isCheckable(url) { return /^https?:\/\//i.test(url || ""); }

async function badge(tabId, text, color, title) {
  await Promise.allSettled([
    chrome.action.setBadgeText({ tabId, text }),
    chrome.action.setBadgeBackgroundColor({ tabId, color }),
    chrome.action.setTitle({ tabId, title }),
  ]);
}

async function grabHtml(tabId) {
  try {
    const res = await chrome.tabs.sendMessage(tabId, { type: "GET_HTML" });
    if (res?.html) return res;
  } catch (_) { /* A previously open tab may not have the content script yet. */ }
  const [out] = await chrome.scripting.executeScript({
    target: { tabId },
    func: () => ({ url: location.href, html: document.documentElement.outerHTML.slice(0, 2 * 1024 * 1024) }),
  });
  return out?.result;
}

async function checkTab(tabId, url, force = false) {
  if (!isCheckable(url)) return;
  const previous = jobs.get(tabId);
  if (!force && previous?.url === url && (previous.pending || Date.now() - previous.completed < CACHE_TTL_MS)) return;
  const job = { url, pending: true, completed: 0 };
  jobs.set(tabId, job);
  const current = async () => {
    if (jobs.get(tabId) !== job) return false;
    try { return (await chrome.tabs.get(tabId)).url === url; } catch (_) { return false; }
  };
  await chrome.storage.session.remove("tab_" + tabId);
  await badge(tabId, "…", "#64748b", "DeciPhish: Scanning page");
  try {
    const snapshot = await grabHtml(tabId);
    if (snapshot?.url !== url || !snapshot.html) throw new Error("Page content unavailable; reload and scan again");
    const service = await getServiceUrl();
    const response = await fetch(service + "/check", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url, html: snapshot.html.slice(0, MAX_HTML_CHARS) }),
      signal: AbortSignal.timeout(120000),
    });
    const data = await response.json();
    if (!response.ok || data.error) throw new Error(data.error || `HTTP ${response.status}`);
    if (!Number.isFinite(data.score) || data.score < 0 || data.score > 1
        || !["phishing", "benign"].includes(data.label)) throw new Error("Invalid model response");
    if (!await current()) return;
    const risky = data.label === "phishing";
    await chrome.storage.session.set({ ["tab_" + tabId]: { ...data, url, checked_at: Date.now() } });
    await badge(tabId, risky ? "RISK" : "OK", risky ? "#dc2626" : "#15803d",
      "DeciPhish: " + (risky ? "Suspected phishing" : "No phishing detected"));
    const type = risky ? "SHOW_WARNING" : "CLEAR_WARNING";
    await chrome.tabs.sendMessage(tabId, { type, score: data.score, url }).catch(() => {});
    job.completed = Date.now();
  } catch (error) {
    if (await current()) {
      await chrome.storage.session.set({ ["tab_" + tabId]: { url, error: error.message } });
      await badge(tabId, "?", "#64748b", "DeciPhish: Scan unavailable");
    }
  } finally {
    job.pending = false;
  }
}

chrome.tabs.onUpdated.addListener((tabId, info, tab) => {
  if (info.status === "loading") {
    jobs.delete(tabId);
    chrome.storage.session.remove("tab_" + tabId).catch(() => {});
    badge(tabId, "", "#64748b", "DeciPhish");
  }
  if (info.status === "complete" && tab) checkTab(tabId, tab.url).catch(console.error);
});
chrome.tabs.onRemoved.addListener(tabId => {
  jobs.delete(tabId);
  chrome.storage.session.remove("tab_" + tabId).catch(() => {});
});
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (sender.id !== chrome.runtime.id || sender.tab || msg?.type !== "RECHECK") return false;
  (async () => {
    const tab = await chrome.tabs.get(msg.tabId);
    await checkTab(tab.id, tab.url, true);
    sendResponse({ done: true });
  })().catch(() => sendResponse({ done: false }));
  return true;
});
chrome.storage.onChanged.addListener((changes, area) => {
  if (area === "local" && changes.serviceUrl) {
    jobs.clear();
    chrome.storage.session.clear().catch(() => {});
  }
});
chrome.tabs.query({ active: true, currentWindow: true }).then(tabs => {
  for (const tab of tabs) checkTab(tab.id, tab.url).catch(console.error);
}).catch(() => {});
