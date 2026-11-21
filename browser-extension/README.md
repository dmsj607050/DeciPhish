# DeciPhish Local Guard

A Manifest V3 extension for Chrome and Microsoft Edge. After an HTTP(S) page
finishes loading, it submits the page URL and current top-level DOM snapshot to
the configured loopback model service. It displays **RISK** and a red English
warning for suspected phishing, **OK** for a benign model verdict, and **?** when
scanning is unavailable. The popup shows the model score and scan latency.
Warnings do not display the classification threshold. They can be dismissed;
the extension does not block navigation or submit forms.

## 1. Start the model service

Install the repository dependencies and the appropriate PyTorch build as
described in the [main README](../README.md). From the repository root, run:

```sh
python -m deciphish serve --base-model model/base --checkpoint runs/my-training/checkpoint --device auto
```

The checkpoint must contain `adapter_config.json`, `adapter_model.safetensors`,
and `dcer.pt`. Weights are loaded once, from local files. The service reuses
`deciphish.evidence.make_view`, `RaceEncoder`, `load_detector`, and
`deciphish.evaluation.predict`: one HTML → URL forward pass at the repository's
fixed operating threshold. The default evidence profile is `phreshphish`; use
`--profile ablation` for a compatible PhishIntention-trained checkpoint.

`--device auto` selects CUDA when available, otherwise CPU. Use `--device cpu`
or `--device cuda:0` to select explicitly. To select a physical GPU on a shared
Linux server, set `CUDA_VISIBLE_DEVICES` before launching. Stop with Ctrl+C.

The service listens only on `127.0.0.1:8765`; `--port 18765` selects another
port. `GET /health` reports readiness after loading. `POST /check` accepts:

```json
{"url":"https://example.org/","html":"<html><body>Example</body></html>"}
```

A successful response contains `score`, `label` (`phishing` or `benign`), and
`latency_ms`. Empty/invalid snapshots return 400, non-JSON content returns 415,
requests over 16 MiB return 413, and a concurrent inference request returns
503 `MODEL_BUSY`. There are no automatic inference retries. The server does
not visit the URL or execute HTML/scripts; it extracts static evidence.

### Optional: use a server GPU through SSH

Run this repository's model service on the server, for example on port 18765,
then run a tunnel on the browser's computer:

```sh
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -L 127.0.0.1:8765:127.0.0.1:18765 USER@SERVER
```

Keep that SSH connection running. The extension still uses
`http://127.0.0.1:8765`; the URL/HTML travels through SSH to the server. A
dashboard tunnel on port 18766 does not forward the model API. The server
checkpoint and evidence profile determine the model being evaluated.

## 2. Load the extension

1. Open `chrome://extensions` or `edge://extensions`.
2. Enable **Developer mode**.
3. Choose **Load unpacked** and select this `browser-extension` directory.
4. Pin **DeciPhish Local Guard** if desired.
5. Reload already open web pages so the content script is available.

After updating extension files, click **Reload** on the extension management
page and reload the web page. Installation is manual; this repository does not
publish to the browser extension stores.

## 3. Use and configure

Open a normal HTTP(S) page. Click the extension icon to see the verdict and
service health. **Scan again** forces a fresh snapshot; automatic duplicate
checks for the same tab and URL are cached for ten minutes in the current
worker lifetime. A navigation or page reload starts a new check. Dynamic page
changes that do not trigger a navigation are not monitored continuously.

In **Model connection**, save a loopback HTTP address such as
`http://127.0.0.1:8765` or `http://127.0.0.1:18765`. Only `localhost` and
`127.0.0.1` are accepted. Settings are stored locally in the browser, not synced.
After changing the address, click **Scan again**.

## Data and permissions

- `tabs` identifies the page and discards results when the tab navigates.
- `scripting` obtains the DOM from tabs opened before extension installation.
- `storage` stores the service address locally and verdicts in session storage.
- HTTP(S) host access is necessary to inspect arbitrary pages and call the API.
  Browser internal pages, the extension store, `file://` pages, cross-origin
  frame contents, and closed shadow DOM are not included.
- The captured URL and DOM can contain private page text. The snapshot is
  limited to the first 2,097,152 JavaScript characters; large pages are truncated
  before the model's separate token budget is applied. It is sent only to the
  chosen loopback service (or a server behind your SSH tunnel). The extension
  does not read cookie APIs or record screenshots, keystrokes or browsing
  history. The model service does not persist snapshots or normal page URLs.
- Verdicts are model predictions, not independently verified phishing reports.
  An offline, busy or failed service is shown as unavailable, never as benign.

The HTTP service validates the loopback Host header, accepts extension origins
instead of website origins, and requires JSON. It has no authentication and is
intended for a trusted local environment or SSH forwarding, not public hosting.

## Troubleshooting

| Symptom | Action |
|---|---|
| Service offline | Start the model service or restore its SSH tunnel; verify the popup address and `/health`. |
| `MODEL_BUSY` | Another page is being scored. Click **Scan again** after it finishes. |
| Page content unavailable | Reload the page after loading/updating the extension. Browser internal/store pages cannot be scanned. |
| Loading fails mentioning `dcer.pt` | Supply a DeciPhish RACE/DCER checkpoint, not a plain LoRA adapter. |
| No warning after updating | Reload the extension and the page, then use **Scan again**. |

## Development checks

From the repository root:

```sh
python -m unittest discover -s tests -p "test_server.py"
node --test tests/extension.test.cjs
```

These checks use a deterministic test detector and mocked browser APIs; a real
checkpoint is required for model inference and manual browser verification.
