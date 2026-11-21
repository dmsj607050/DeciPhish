const MAX_HTML_CHARS = 2 * 1024 * 1024;
const BANNER_ID = "deciphish-warning-host";

const warningStyles = `
  :host {
    all: initial;
    position: fixed;
    z-index: 2147483647;
    top: 16px;
    right: 16px;
    left: 16px;
    display: flex;
    justify-content: center;
    pointer-events: none;
    direction: ltr;
  }

  .warning {
    box-sizing: border-box;
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    align-items: start;
    gap: 16px;
    width: min(100%, 640px);
    padding: 16px 18px;
    border: 1px solid #d75a5a;
    border-left: 4px solid #ffc5bc;
    border-radius: 12px;
    background: #981b24;
    color: #fffafa;
    box-shadow: 0 12px 40px #410d1755, 0 2px 8px #410d1733;
    font: 400 14px/1.5 Inter, "Segoe UI", Arial, sans-serif;
    text-align: left;
    pointer-events: auto;
  }

  .notice { display: flex; align-items: flex-start; gap: 14px; min-width: 0; }
  .symbol {
    flex: 0 0 34px;
    height: 34px;
    border-radius: 9px;
    background: #68121a;
    color: #ffe0da;
    text-align: center;
    font-size: 22px;
    font-weight: 700;
    line-height: 34px;
  }
  .copy { min-width: 0; }
  .title { margin: 0; font-size: 18px; font-weight: 700; line-height: 1.3; }
  .advice { margin: 4px 0 0; color: #fff0ee; }
  .details { margin: 7px 0 0; color: #ffd2cc; font-size: 12px; overflow-wrap: anywhere; }

  .dismiss {
    box-sizing: border-box;
    min-height: 38px;
    padding: 6px 12px;
    border: 1px solid #e69b9d;
    border-radius: 7px;
    background: #76121a;
    color: inherit;
    font: 600 13px/1.3 Inter, "Segoe UI", Arial, sans-serif;
    cursor: pointer;
  }
  .dismiss:hover { background: #5c0f16; }
  .dismiss:focus-visible { outline: 2px solid #ffe0da; outline-offset: 3px; }

  @media (max-width: 420px) {
    :host { top: 8px; right: 8px; left: 8px; }
    .warning { grid-template-columns: minmax(0, 1fr); gap: 10px; padding: 12px; }
    .notice { gap: 10px; }
    .title { font-size: 16px; }
    .dismiss { justify-self: end; }
  }
`;

function warningElement(tag, className, text) {
  const element = document.createElement(tag);
  element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

function formatScore(value) {
  return typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 1
    ? value.toFixed(4)
    : null;
}

function showWarning(score) {
  if (document.getElementById(BANNER_ID)) return;
  const host = document.createElement("div");
  host.id = BANNER_ID;
  host.lang = "en";
  const root = host.attachShadow({ mode: "open" });
  const stylesheet = new CSSStyleSheet();
  stylesheet.replaceSync(warningStyles);
  root.adoptedStyleSheets = [stylesheet];

  const panel = warningElement("section", "warning");
  panel.setAttribute("aria-label", "DeciPhish warning");
  const notice = warningElement("div", "notice");
  notice.setAttribute("role", "alert");
  const symbol = warningElement("span", "symbol", "!");
  symbol.setAttribute("aria-hidden", "true");
  const copy = warningElement("div", "copy");
  const title = warningElement("h2", "title", "Suspected phishing site");
  const advice = warningElement("p", "advice", "Do not enter passwords, verification codes, or payment details on this site.");
  const modelScore = formatScore(score);
  const details = warningElement("p", "details", modelScore
    ? `DeciPhish · Model score ${modelScore}`
    : "Flagged by DeciPhish");
  const dismiss = warningElement("button", "dismiss", "Dismiss");
  dismiss.type = "button";
  dismiss.addEventListener("click", () => host.remove());

  copy.append(title, advice, details);
  notice.append(symbol, copy);
  panel.append(notice, dismiss);
  root.append(panel);
  document.documentElement.appendChild(host);
}

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (!msg) return false;
  if (msg.type === "GET_HTML") {
    sendResponse({ url: location.href, html: document.documentElement.outerHTML.slice(0, MAX_HTML_CHARS) });
    return true;
  }
  if (msg.type === "CLEAR_WARNING" && msg.url === location.href) {
    document.getElementById(BANNER_ID)?.remove();
  }
  if (msg.type === "SHOW_WARNING" && msg.url === location.href) {
    showWarning(msg.score);
  }
  return false;
});
