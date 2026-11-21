/* Shared by the extension worker and popup. No page scripts load this file. */
const DEFAULT_SERVICE_URL = "http://127.0.0.1:8765";

function normalizeServiceUrl(value) {
  const url = new URL(value);
  if (url.protocol !== "http:" || !["127.0.0.1", "localhost"].includes(url.hostname)
      || url.username || url.password || url.pathname !== "/" || url.search || url.hash) {
    throw new Error("Use a loopback HTTP address, for example http://127.0.0.1:8765");
  }
  return url.origin;
}

async function getServiceUrl() {
  const { serviceUrl = DEFAULT_SERVICE_URL } = await chrome.storage.local.get("serviceUrl");
  return normalizeServiceUrl(serviceUrl);
}
