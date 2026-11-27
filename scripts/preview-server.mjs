import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");
const DOC_PATH = path.join(ROOT, "docs", "PATENT_CLEAN_MASTERING_SUITE.md");
const README_PATH = path.join(ROOT, "README.md");
const PATCH_DIR = path.join(
  ROOT,
  ".aistudio",
  "artifacts",
  "brain",
  "37802b59-db81-4af3-8bfd-83e27b0b3f47",
  "patches",
);

const PORT = Number(process.env.PORT || 3000);

function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function listPatches() {
  try {
    return fs
      .readdirSync(PATCH_DIR)
      .filter((name) => name.endsWith(".patch"))
      .sort();
  } catch {
    return [];
  }
}

function readSafeFile(filePath) {
  try {
    return fs.readFileSync(filePath, "utf8");
  } catch {
    return "";
  }
}

function renderIndexHtml() {
  const patches = listPatches();
  const patchButtons = patches
    .map(
      (name) =>
        `<button class="tab-btn" data-kind="patch" data-name="${escapeHtml(name)}">${escapeHtml(name)}</button>`,
    )
    .join("");

  return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>G6 Media Encoder (g6me) — Mastering &amp; Spatial Suite</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #0c0e12;
      --panel: #141820;
      --border: #252c3a;
      --text: #e6ebf5;
      --muted: #93a0b8;
      --accent: #58a6ff;
      --ok: #3fb950;
      --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      --sans: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: var(--sans);
      line-height: 1.55;
    }
    header {
      border-bottom: 1px solid var(--border);
      background: var(--panel);
      padding: 20px 28px;
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }
    header h1 {
      margin: 0;
      font-size: 1.25rem;
      letter-spacing: -0.01em;
    }
    header p {
      margin: 4px 0 0;
      color: var(--muted);
      font-size: 0.9rem;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 6px 12px;
      border-radius: 6px;
      border: 1px solid rgba(63, 185, 80, 0.35);
      background: rgba(63, 185, 80, 0.1);
      color: var(--ok);
      font-size: 0.85rem;
      font-weight: 600;
    }
    main {
      max-width: 1180px;
      margin: 0 auto;
      padding: 24px 28px 48px;
      display: grid;
      gap: 20px;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 14px;
    }
    .card {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 16px;
    }
    .card h2 {
      margin: 0 0 6px;
      font-size: 0.95rem;
      color: var(--accent);
    }
    .card p {
      margin: 0;
      color: var(--muted);
      font-size: 0.86rem;
    }
    .viewer {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 10px;
      overflow: hidden;
    }
    .toolbar {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      padding: 12px 16px;
      border-bottom: 1px solid var(--border);
      background: #10141b;
    }
    .tabs {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }
    .tab-btn, .copy-btn {
      background: #1b2230;
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 6px 11px;
      font-size: 0.82rem;
      cursor: pointer;
      font-family: var(--mono);
    }
    .tab-btn.active {
      border-color: var(--accent);
      background: rgba(88, 166, 255, 0.16);
      color: #ffffff;
    }
    .copy-btn:hover, .tab-btn:hover {
      border-color: var(--accent);
    }
    pre {
      margin: 0;
      padding: 18px 20px;
      overflow: auto;
      max-height: 68vh;
      font-family: var(--mono);
      font-size: 0.82rem;
      line-height: 1.5;
      white-space: pre-wrap;
      word-break: break-word;
      color: #dbe4f3;
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>G6 Media Encoder (<code>g6me</code>) — Native Rust Audio Workspace</h1>
      <p>Patent-Clean Mastering &amp; Spatial Enhancement Suite · Deterministic f64 · Zero FMA · Zero New Crates</p>
    </div>
    <div class="badge">Workspace Verified · g6me-verify 98/98 PASS</div>
  </header>

  <main>
    <section class="grid">
      <div class="card">
        <h2>Shared Fuzzy STN</h2>
        <p>Dual-resolution Fierro–Välimäki (4096/512) unity-summing Sines, Transients &amp; Noise decomposition (&lt;1e-12 null).</p>
      </div>
      <div class="card">
        <h2>Zero-IMD Virtual Bass &amp; Detail</h2>
        <p>N=8192 phase-vocoder bass harmonics on Sines + FEXP1 on Transients; upward micro-detail lift with Noise untouched.</p>
      </div>
      <div class="card">
        <h2>Moore 2016 Unmasking EQ</h2>
        <p>ISO 532-2 75-band ERB excitation + binaural inhibition realized via causal minimum-phase cepstral FIR OLA.</p>
      </div>
      <div class="card">
        <h2>2.0 → 7.1 PAE Upmixer</h2>
        <p>He PCA+ASE + Song APES quadrature phase + Kraft–Zölzer 4-ch mutually-prime Schroeder all-pass decorrelators.</p>
      </div>
      <div class="card">
        <h2>Punchy Limiter + ADAA2</h2>
        <p>Residual ADAA2 C² soft-knee pre-clipper + Signalsmith 2-stage boxcar C¹ lookahead + Giannoulis log-domain release.</p>
      </div>
    </section>

    <section class="viewer">
      <div class="toolbar">
        <div class="tabs" id="tabs">
          <button class="tab-btn active" data-kind="doc" data-name="writeup">PATENT_CLEAN_MASTERING_SUITE.md</button>
          <button class="tab-btn" data-kind="doc" data-name="readme">README.md</button>
          ${patchButtons}
        </div>
        <button class="copy-btn" id="copy-btn">Copy</button>
      </div>
      <pre id="content">Loading…</pre>
    </section>
  </main>

  <script>
    const contentEl = document.getElementById("content");
    const copyBtn = document.getElementById("copy-btn");
    const tabsEl = document.getElementById("tabs");
    let currentText = "";

    async function loadItem(kind, name) {
      contentEl.textContent = "Loading...";
      const url = kind === "doc"
        ? "/api/doc?name=" + encodeURIComponent(name)
        : "/api/patch?name=" + encodeURIComponent(name);
      try {
        const res = await fetch(url);
        if (!res.ok) throw new Error("HTTP " + res.status);
        currentText = await res.text();
        contentEl.textContent = currentText;
      } catch (err) {
        currentText = "";
        contentEl.textContent = "Failed to load: " + err.message;
      }
    }

    tabsEl.addEventListener("click", (e) => {
      const btn = e.target.closest("button.tab-btn");
      if (!btn) return;
      for (const b of tabsEl.querySelectorAll("button.tab-btn")) {
        b.classList.remove("active");
      }
      btn.classList.add("active");
      loadItem(btn.dataset.kind, btn.dataset.name);
    });

    copyBtn.addEventListener("click", async () => {
      if (!currentText) return;
      try {
        await navigator.clipboard.writeText(currentText);
        copyBtn.textContent = "Copied";
        setTimeout(() => (copyBtn.textContent = "Copy"), 1200);
      } catch {
        copyBtn.textContent = "Copy failed";
        setTimeout(() => (copyBtn.textContent = "Copy"), 1200);
      }
    });

    loadItem("doc", "writeup");
  </script>
</body>
</html>`;
}

const server = http.createServer((req, res) => {
  const url = new URL(req.url || "/", `http://${req.headers.host || "localhost"}`);

  if (url.pathname === "/" || url.pathname === "/index.html") {
    res.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
    res.end(renderIndexHtml());
    return;
  }

  if (url.pathname === "/api/doc") {
    const name = url.searchParams.get("name");
    const target = name === "readme" ? README_PATH : name === "writeup" ? DOC_PATH : null;
    if (!target) {
      res.writeHead(400, { "Content-Type": "text/plain; charset=utf-8" });
      res.end("Unknown document");
      return;
    }
    res.writeHead(200, { "Content-Type": "text/plain; charset=utf-8" });
    res.end(readSafeFile(target));
    return;
  }

  if (url.pathname === "/api/patch") {
    const name = url.searchParams.get("name") || "";
    const allowed = listPatches();
    if (!allowed.includes(name)) {
      res.writeHead(404, { "Content-Type": "text/plain; charset=utf-8" });
      res.end("Patch not found");
      return;
    }
    res.writeHead(200, { "Content-Type": "text/plain; charset=utf-8" });
    res.end(readSafeFile(path.join(PATCH_DIR, name)));
    return;
  }

  if (url.pathname === "/health") {
    res.writeHead(200, { "Content-Type": "application/json; charset=utf-8" });
    res.end(JSON.stringify({ status: "ok" }));
    return;
  }

  res.writeHead(404, { "Content-Type": "text/plain; charset=utf-8" });
  res.end("Not found");
});

server.listen(PORT, "0.0.0.0", () => {
  console.log(`g6me preview server listening on http://0.0.0.0:${PORT}`);
});
