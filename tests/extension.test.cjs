const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const root = path.join(__dirname, "..", "browser-extension");

function environment(fetch) {
  const session = {};
  const local = {};
  const messages = [];
  const badges = [];
  const tab = { id: 1, url: "https://example.org/" };
  const event = () => ({ addListener() {} });
  function storage(values) {
    return {
      async get(key) { return { [key]: values[key] }; },
      async set(record) { Object.assign(values, record); },
      async remove(key) { delete values[key]; },
      async clear() { for (const key of Object.keys(values)) delete values[key]; },
    };
  }
  const chrome = {
    storage: { local: storage(local), session: storage(session), onChanged: event() },
    tabs: {
      async query() { return []; },
      async get() { return tab; },
      async sendMessage(_id, msg) {
        if (msg.type === "GET_HTML") return { url: tab.url, html: "<html>Snapshot</html>" };
        messages.push(msg);
      },
      onUpdated: event(), onRemoved: event(),
    },
    action: {
      async setBadgeText(value) { badges.push(value.text); },
      async setBadgeBackgroundColor() {}, async setTitle() {},
    },
    runtime: { id: "test", onMessage: event() },
  };
  const context = vm.createContext({ chrome, URL, AbortSignal, console, fetch });
  context.importScripts = filename => vm.runInContext(fs.readFileSync(path.join(root, filename), "utf8"), context);
  vm.runInContext(fs.readFileSync(path.join(root, "background.js"), "utf8"), context);
  return { context, session, local, tab, messages, badges,
    scan: () => vm.runInContext('checkTab(1, "https://example.org/")', context) };
}

const response = (label = "phishing", score = 0.9) => ({ ok: true, async json() { return { label, score, latency_ms: 90 }; } });

test("service settings accept only loopback HTTP origins", async () => {
  const { context } = environment(async () => response());
  assert.equal(vm.runInContext('normalizeServiceUrl("http://127.0.0.1:18765/")', context), "http://127.0.0.1:18765");
  for (const url of ["https://evil.example", "http://127.0.0.1.evil.example", "http://user@localhost", "http://localhost/path", "http://localhost?x=1"]) {
    context.inputUrl = url;
    assert.throws(() => vm.runInContext("normalizeServiceUrl(inputUrl)", context));
  }
});

test("submits URL/HTML to the configured API and shows phishing warning", async () => {
  let request;
  const env = environment(async (url, options) => { request = { url, options }; return response(); });
  env.local.serviceUrl = "http://127.0.0.1:18765";
  await env.scan();
  assert.equal(request.url, "http://127.0.0.1:18765/check");
  assert.deepEqual(JSON.parse(request.options.body), { url: env.tab.url, html: "<html>Snapshot</html>" });
  assert.equal(env.session.tab_1.label, "phishing");
  assert.equal(env.messages[0].type, "SHOW_WARNING");
  assert.equal(env.badges.at(-1), "RISK");
});

test("a benign recheck clears a previous warning", async () => {
  const env = environment(async () => response("benign", 0.1));
  await env.scan();
  assert.equal(env.messages[0].type, "CLEAR_WARNING");
  assert.equal(env.badges.at(-1), "OK");
});

test("offline requests do not become benign verdicts or cached successes", async () => {
  let calls = 0;
  const env = environment(async () => { calls++; throw new Error("Offline"); });
  await env.scan();
  assert.equal(env.session.tab_1.error, "Offline");
  assert.equal(env.session.tab_1.label, undefined);
  assert.equal(env.badges.at(-1), "?");
  await env.scan();
  assert.equal(calls, 2);
});

test("busy and malformed model replies remain unavailable", async () => {
  for (const reply of [
    { ok: false, async json() { return { error: "MODEL_BUSY" }; } },
    response("unknown", 0.1), response("benign", NaN),
  ]) {
    const env = environment(async () => reply);
    await env.scan();
    assert.ok(env.session.tab_1.error);
    assert.equal(env.badges.at(-1), "?");
    assert.equal(env.messages.length, 0);
  }
});

test("a model reply after navigation cannot warn the new page", async () => {
  let finish;
  let started;
  const ready = new Promise(resolve => { started = resolve; });
  const env = environment(() => { started(); return new Promise(resolve => { finish = resolve; }); });
  const scanning = env.scan();
  await ready;
  env.tab.url = "https://different.example/";
  finish(response());
  await scanning;
  assert.equal(env.messages.length, 0);
  assert.equal(env.session.tab_1, undefined);
});

test("successful automatic scans are cached within the same tab and URL", async () => {
  let calls = 0;
  const env = environment(async () => { calls++; return response(); });
  await env.scan();
  await env.scan();
  assert.equal(calls, 1);
});
