async function refresh() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  document.getElementById("url").textContent = tab?.url || "—";
  const key = "tab_" + tab?.id;
  const store = await chrome.storage.session.get(key);
  const rec = store[key];
  const verdict = document.getElementById("verdict");
  verdict.className = "";
  document.getElementById("score").textContent = "—";
  document.getElementById("latency").textContent = "—";
  if (rec?.url === tab?.url && rec?.error) {
    verdict.textContent = "Scan unavailable: " + rec.error;
  } else if (rec?.url === tab?.url && ["phishing", "benign"].includes(rec?.label)) {
    verdict.textContent = rec.label === "phishing" ? "Suspected phishing" : "No phishing detected";
    verdict.className = rec.label === "phishing" ? "risky" : "safe";
    document.getElementById("score").textContent = rec.score.toFixed(4);
    document.getElementById("latency").textContent = Number.isFinite(rec.latency_ms) ? rec.latency_ms + " ms" : "—";
  } else {
    verdict.textContent = "Not scanned yet";
  }
  try {
    const response = await fetch((await getServiceUrl()) + "/health", { signal: AbortSignal.timeout(5000) });
    if (!response.ok) throw new Error("Service unavailable");
    const health = await response.json();
    document.getElementById("health").textContent = health.model_ready
      ? `Ready (${health.device}/${health.dtype})` : "Model not ready";
  } catch (_) {
    document.getElementById("health").textContent = "Offline — start the model service or SSH tunnel";
  }
}

document.getElementById("recheck").addEventListener("click", async event => {
  const button = event.currentTarget;
  button.disabled = true;
  button.textContent = "Scanning…";
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (tab) await chrome.runtime.sendMessage({ type: "RECHECK", tabId: tab.id });
    await refresh();
  } finally {
    button.disabled = false;
    button.textContent = "Scan again";
  }
});
document.getElementById("save-service").addEventListener("click", async () => {
  const status = document.getElementById("settings-status");
  try {
    const serviceUrl = normalizeServiceUrl(document.getElementById("service-url").value.trim());
    await chrome.storage.local.set({ serviceUrl });
    status.textContent = "Saved. Scan the page again to use this service.";
    await refresh();
  } catch (error) {
    status.textContent = error.message;
  }
});
getServiceUrl().then(url => { document.getElementById("service-url").value = url; }).catch(() => {});
chrome.storage.onChanged.addListener((_changes, area) => {
  if (area === "session") refresh().catch(console.error);
});
refresh().catch(console.error);
