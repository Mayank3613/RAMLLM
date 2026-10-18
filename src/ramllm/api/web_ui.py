"""HTML and assets for the RAMLLM Web UI and playground."""

FAVICON_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" fill="none">
  <defs>
    <linearGradient id="ramllm-grad" x1="0" y1="0" x2="64" y2="64" gradientUnits="userSpaceOnUse">
      <stop offset="0%" stop-color="#06b6d4" />
      <stop offset="50%" stop-color="#3b82f6" />
      <stop offset="100%" stop-color="#8b5cf6" />
    </linearGradient>
    <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur stdDeviation="3" result="blur" />
      <feComposite in="SourceGraphic" in2="blur" operator="over" />
    </filter>
  </defs>
  <rect width="64" height="64" rx="16" fill="#090b10" />
  <rect x="8" y="8" width="48" height="48" rx="10" stroke="url(#ramllm-grad)" stroke-width="2.5" fill="#121622" />
  <!-- RAM chip pins -->
  <line x1="16" y1="4" x2="16" y2="8" stroke="#06b6d4" stroke-width="2.5" stroke-linecap="round" />
  <line x1="24" y1="4" x2="24" y2="8" stroke="#06b6d4" stroke-width="2.5" stroke-linecap="round" />
  <line x1="32" y1="4" x2="32" y2="8" stroke="#3b82f6" stroke-width="2.5" stroke-linecap="round" />
  <line x1="40" y1="4" x2="40" y2="8" stroke="#8b5cf6" stroke-width="2.5" stroke-linecap="round" />
  <line x1="48" y1="4" x2="48" y2="8" stroke="#8b5cf6" stroke-width="2.5" stroke-linecap="round" />
  <line x1="16" y1="56" x2="16" y2="60" stroke="#06b6d4" stroke-width="2.5" stroke-linecap="round" />
  <line x1="24" y1="56" x2="24" y2="60" stroke="#06b6d4" stroke-width="2.5" stroke-linecap="round" />
  <line x1="32" y1="56" x2="32" y2="60" stroke="#3b82f6" stroke-width="2.5" stroke-linecap="round" />
  <line x1="40" y1="56" x2="40" y2="60" stroke="#8b5cf6" stroke-width="2.5" stroke-linecap="round" />
  <line x1="48" y1="56" x2="48" y2="60" stroke="#8b5cf6" stroke-width="2.5" stroke-linecap="round" />
  <!-- Brain / Neural nodes inside chip -->
  <circle cx="24" cy="24" r="3.5" fill="#06b6d4" filter="url(#glow)" />
  <circle cx="40" cy="24" r="3.5" fill="#3b82f6" filter="url(#glow)" />
  <circle cx="32" cy="38" r="4" fill="#8b5cf6" filter="url(#glow)" />
  <line x1="24" y1="24" x2="40" y2="24" stroke="rgba(255,255,255,0.4)" stroke-width="1.5" />
  <line x1="24" y1="24" x2="32" y2="38" stroke="rgba(255,255,255,0.4)" stroke-width="1.5" />
  <line x1="40" y1="24" x2="32" y2="38" stroke="rgba(255,255,255,0.4)" stroke-width="1.5" />
</svg>"""

WEB_UI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>RAMLLM Studio &middot; {{MODEL_NAME}}</title>
  <link rel="icon" type="image/svg+xml" href="/favicon.ico">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&family=Outfit:wght@500;600;700;800&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-base: #080a0f;
      --bg-surface: #0f131c;
      --bg-card: rgba(18, 23, 34, 0.7);
      --bg-input: #141824;
      --border-subtle: rgba(255, 255, 255, 0.08);
      --border-hover: rgba(6, 182, 212, 0.4);
      --text-main: #f1f5f9;
      --text-muted: #94a3b8;
      --text-dim: #64748b;
      --cyan: #06b6d4;
      --cyan-glow: rgba(6, 182, 212, 0.25);
      --purple: #8b5cf6;
      --emerald: #10b981;
      --rose: #f43f5e;
      --amber: #f59e0b;
      --font-heading: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
      --font-body: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      --font-mono: 'JetBrains Mono', monospace;
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }

    body {
      background-color: var(--bg-base);
      color: var(--text-main);
      font-family: var(--font-body);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      overflow-x: hidden;
    }

    /* Top Navigation */
    header {
      background: rgba(15, 19, 28, 0.85);
      backdrop-filter: blur(16px);
      border-bottom: 1px solid var(--border-subtle);
      position: sticky;
      top: 0;
      z-index: 50;
      padding: 0.75rem 1.5rem;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 0.75rem;
      text-decoration: none;
      color: var(--text-main);
    }

    .brand-icon {
      width: 32px;
      height: 32px;
      border-radius: 8px;
      background: linear-gradient(135deg, var(--cyan), var(--purple));
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: 0 0 16px var(--cyan-glow);
    }

    .brand-icon svg {
      width: 18px;
      height: 18px;
      fill: none;
      stroke: #fff;
      stroke-width: 2;
    }

    .brand-text {
      font-family: var(--font-heading);
      font-weight: 800;
      font-size: 1.25rem;
      letter-spacing: -0.02em;
      background: linear-gradient(90deg, #fff 0%, #cbd5e1 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }

    .badge-live {
      display: inline-flex;
      align-items: center;
      gap: 0.375rem;
      font-size: 0.75rem;
      font-weight: 600;
      padding: 0.2rem 0.6rem;
      border-radius: 9999px;
      background: rgba(16, 185, 129, 0.12);
      color: var(--emerald);
      border: 1px solid rgba(16, 185, 129, 0.25);
    }

    .badge-live .dot {
      width: 6px;
      height: 6px;
      background-color: var(--emerald);
      border-radius: 50%;
      animation: pulse 2s infinite;
    }

    @keyframes pulse {
      0%, 100% { opacity: 1; transform: scale(1); }
      50% { opacity: 0.4; transform: scale(0.85); }
    }

    .nav-actions {
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }

    .nav-link {
      color: var(--text-muted);
      text-decoration: none;
      font-size: 0.85rem;
      font-weight: 500;
      padding: 0.4rem 0.8rem;
      border-radius: 6px;
      border: 1px solid transparent;
      transition: all 0.2s;
    }

    .nav-link:hover {
      color: var(--text-main);
      background: rgba(255, 255, 255, 0.05);
      border-color: var(--border-subtle);
    }

    .btn-api-docs {
      background: linear-gradient(135deg, rgba(6, 182, 212, 0.15), rgba(139, 92, 246, 0.15));
      border: 1px solid rgba(6, 182, 212, 0.3);
      color: var(--cyan);
    }

    .btn-api-docs:hover {
      background: linear-gradient(135deg, rgba(6, 182, 212, 0.25), rgba(139, 92, 246, 0.25));
      border-color: var(--cyan);
      box-shadow: 0 0 12px var(--cyan-glow);
    }

    /* Main Container */
    main {
      flex: 1;
      display: grid;
      grid-template-columns: 320px 1fr 340px;
      height: calc(100vh - 58px);
      overflow: hidden;
    }

    @media (max-width: 1100px) {
      main {
        grid-template-columns: 280px 1fr;
      }
      .drawer-right {
        display: none;
      }
    }

    @media (max-width: 768px) {
      main {
        grid-template-columns: 1fr;
        height: auto;
      }
      .sidebar-left {
        display: none;
      }
    }

    /* Left Sidebar */
    .sidebar-left {
      background: var(--bg-surface);
      border-right: 1px solid var(--border-subtle);
      padding: 1.25rem;
      display: flex;
      flex-direction: column;
      gap: 1.25rem;
      overflow-y: auto;
    }

    .section-title {
      font-family: var(--font-heading);
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--text-dim);
      margin-bottom: 0.5rem;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }

    .hud-card {
      background: var(--bg-card);
      backdrop-filter: blur(12px);
      border: 1px solid var(--border-subtle);
      border-radius: 12px;
      padding: 1rem;
      transition: border-color 0.2s;
    }

    .hud-card:hover {
      border-color: rgba(255, 255, 255, 0.15);
    }

    .model-title {
      font-family: var(--font-heading);
      font-size: 1rem;
      font-weight: 700;
      color: #fff;
      word-break: break-all;
      line-height: 1.3;
      margin-bottom: 0.35rem;
    }

    .model-tag {
      display: inline-block;
      font-family: var(--font-mono);
      font-size: 0.7rem;
      padding: 0.15rem 0.5rem;
      border-radius: 4px;
      background: rgba(6, 182, 212, 0.15);
      color: var(--cyan);
      border: 1px solid rgba(6, 182, 212, 0.3);
    }

    /* Metric Bars */
    .metric-row {
      margin-top: 0.75rem;
    }

    .metric-header {
      display: flex;
      justify-content: space-between;
      font-size: 0.78rem;
      margin-bottom: 0.35rem;
    }

    .metric-label {
      color: var(--text-muted);
    }

    .metric-val {
      font-family: var(--font-mono);
      color: var(--text-main);
      font-weight: 600;
    }

    .progress-track {
      height: 6px;
      background: rgba(255, 255, 255, 0.06);
      border-radius: 9999px;
      overflow: hidden;
      position: relative;
    }

    .progress-fill {
      height: 100%;
      border-radius: 9999px;
      transition: width 0.4s ease, background 0.4s ease;
      background: linear-gradient(90deg, var(--cyan), #3b82f6);
    }

    .progress-fill.vram {
      background: linear-gradient(90deg, #3b82f6, var(--purple));
    }

    .governor-status {
      margin-top: 0.85rem;
      padding: 0.5rem 0.75rem;
      border-radius: 8px;
      font-size: 0.75rem;
      display: flex;
      align-items: center;
      gap: 0.5rem;
      background: rgba(16, 185, 129, 0.08);
      border: 1px solid rgba(16, 185, 129, 0.2);
      color: var(--emerald);
    }

    /* Controls */
    .control-group {
      display: flex;
      flex-direction: column;
      gap: 0.85rem;
    }

    .control-item label {
      display: flex;
      justify-content: space-between;
      font-size: 0.78rem;
      color: var(--text-muted);
      margin-bottom: 0.35rem;
    }

    .control-item input[type="range"] {
      width: 100%;
      accent-color: var(--cyan);
      cursor: pointer;
    }

    .control-item textarea, .control-item input[type="text"], .control-item input[type="password"] {
      width: 100%;
      background: var(--bg-input);
      border: 1px solid var(--border-subtle);
      border-radius: 6px;
      color: var(--text-main);
      padding: 0.5rem 0.65rem;
      font-size: 0.8rem;
      font-family: var(--font-mono);
      outline: none;
      transition: border-color 0.2s;
    }

    .control-item textarea:focus, .control-item input:focus {
      border-color: var(--cyan);
    }

    /* Center Chat Column */
    .chat-col {
      display: flex;
      flex-direction: column;
      height: 100%;
      background: radial-gradient(circle at 50% 0%, rgba(6, 182, 212, 0.04) 0%, transparent 60%);
    }

    .chat-messages {
      flex: 1;
      overflow-y: auto;
      padding: 1.5rem;
      display: flex;
      flex-direction: column;
      gap: 1.25rem;
    }

    /* Empty state */
    .empty-chat {
      margin: auto;
      max-width: 520px;
      text-align: center;
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 1rem;
      padding: 2rem 1rem;
    }

    .empty-icon {
      width: 56px;
      height: 56px;
      border-radius: 16px;
      background: linear-gradient(135deg, rgba(6, 182, 212, 0.15), rgba(139, 92, 246, 0.15));
      border: 1px solid rgba(6, 182, 212, 0.3);
      display: flex;
      align-items: center;
      justify-content: center;
      color: var(--cyan);
      box-shadow: 0 0 24px var(--cyan-glow);
    }

    .empty-title {
      font-family: var(--font-heading);
      font-size: 1.5rem;
      font-weight: 700;
      color: #fff;
    }

    .empty-subtitle {
      color: var(--text-muted);
      font-size: 0.9rem;
      line-height: 1.5;
    }

    .preset-chips {
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
      width: 100%;
      margin-top: 0.5rem;
    }

    .preset-btn {
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: 8px;
      color: var(--text-main);
      padding: 0.65rem 0.9rem;
      font-size: 0.85rem;
      text-align: left;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: space-between;
      transition: all 0.2s;
    }

    .preset-btn:hover {
      border-color: var(--border-hover);
      background: rgba(6, 182, 212, 0.08);
      transform: translateY(-1px);
    }

    /* Message Bubbles */
    .msg-row {
      display: flex;
      gap: 0.85rem;
      max-width: 85%;
      animation: fadeIn 0.2s ease-out;
    }

    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(6px); }
      to { opacity: 1; transform: translateY(0); }
    }

    .msg-row.user {
      align-self: flex-end;
      flex-direction: row-reverse;
    }

    .msg-row.assistant {
      align-self: flex-start;
    }

    .msg-avatar {
      width: 32px;
      height: 32px;
      border-radius: 8px;
      flex-shrink: 0;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 0.8rem;
      font-weight: 700;
    }

    .msg-row.user .msg-avatar {
      background: linear-gradient(135deg, #4f46e5, #7c3aed);
      color: #fff;
    }

    .msg-row.assistant .msg-avatar {
      background: linear-gradient(135deg, var(--cyan), var(--purple));
      color: #fff;
      box-shadow: 0 0 12px var(--cyan-glow);
    }

    .msg-bubble {
      padding: 0.85rem 1.15rem;
      border-radius: 12px;
      font-size: 0.92rem;
      line-height: 1.6;
      word-break: break-word;
    }

    .msg-row.user .msg-bubble {
      background: #2563eb;
      color: #fff;
      border-bottom-right-radius: 3px;
    }

    .msg-row.assistant .msg-bubble {
      background: var(--bg-card);
      backdrop-filter: blur(12px);
      border: 1px solid var(--border-subtle);
      color: var(--text-main);
      border-bottom-left-radius: 3px;
    }

    .msg-bubble pre {
      background: #090b10;
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 6px;
      padding: 0.75rem;
      margin: 0.65rem 0;
      overflow-x: auto;
      font-family: var(--font-mono);
      font-size: 0.82rem;
    }

    .msg-bubble code {
      font-family: var(--font-mono);
      font-size: 0.85em;
      background: rgba(255, 255, 255, 0.08);
      padding: 0.15rem 0.35rem;
      border-radius: 4px;
    }

    .msg-bubble pre code {
      background: none;
      padding: 0;
    }

    .msg-stats {
      margin-top: 0.5rem;
      font-size: 0.72rem;
      font-family: var(--font-mono);
      color: var(--text-dim);
      display: flex;
      gap: 0.75rem;
    }

    /* Cursor */
    .typing-cursor {
      display: inline-block;
      width: 7px;
      height: 14px;
      background: var(--cyan);
      margin-left: 3px;
      vertical-align: middle;
      animation: blink 0.8s infinite;
    }

    @keyframes blink {
      0%, 100% { opacity: 1; }
      50% { opacity: 0; }
    }

    /* Chat Input Bar */
    .chat-input-bar {
      padding: 1rem 1.5rem;
      background: var(--bg-surface);
      border-top: 1px solid var(--border-subtle);
    }

    .input-wrapper {
      display: flex;
      align-items: flex-end;
      gap: 0.75rem;
      background: var(--bg-input);
      border: 1px solid var(--border-subtle);
      border-radius: 12px;
      padding: 0.5rem 0.75rem;
      transition: border-color 0.2s, box-shadow 0.2s;
    }

    .input-wrapper:focus-within {
      border-color: var(--cyan);
      box-shadow: 0 0 12px var(--cyan-glow);
    }

    .input-wrapper textarea {
      flex: 1;
      background: transparent;
      border: none;
      color: var(--text-main);
      font-size: 0.92rem;
      font-family: var(--font-body);
      resize: none;
      outline: none;
      min-height: 24px;
      max-height: 160px;
      line-height: 1.5;
    }

    .action-btn {
      background: linear-gradient(135deg, var(--cyan), #3b82f6);
      border: none;
      border-radius: 8px;
      color: #fff;
      width: 36px;
      height: 36px;
      display: flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      transition: all 0.2s;
      flex-shrink: 0;
    }

    .action-btn:hover {
      transform: scale(1.05);
      box-shadow: 0 0 12px var(--cyan-glow);
    }

    .action-btn:disabled {
      opacity: 0.4;
      cursor: not-allowed;
      transform: none;
      box-shadow: none;
    }

    .btn-stop {
      background: linear-gradient(135deg, var(--rose), #e11d48);
    }

    .btn-clear {
      background: transparent;
      border: 1px solid var(--border-subtle);
      color: var(--text-dim);
    }

    .btn-clear:hover {
      color: var(--rose);
      border-color: rgba(244, 63, 94, 0.4);
      background: rgba(244, 63, 94, 0.1);
    }

    /* Right Drawer (API & Integration) */
    .drawer-right {
      background: var(--bg-surface);
      border-left: 1px solid var(--border-subtle);
      padding: 1.25rem;
      display: flex;
      flex-direction: column;
      gap: 1.25rem;
      overflow-y: auto;
    }

    .tab-nav {
      display: flex;
      gap: 0.35rem;
      background: var(--bg-input);
      padding: 0.25rem;
      border-radius: 8px;
    }

    .tab-btn {
      flex: 1;
      padding: 0.35rem 0.5rem;
      border: none;
      border-radius: 6px;
      background: transparent;
      color: var(--text-muted);
      font-size: 0.75rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s;
    }

    .tab-btn.active {
      background: var(--bg-card);
      color: var(--cyan);
      box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2);
    }

    .code-box {
      position: relative;
      background: #06080d;
      border: 1px solid var(--border-subtle);
      border-radius: 8px;
      padding: 0.85rem;
      overflow-x: auto;
    }

    .code-box pre {
      font-family: var(--font-mono);
      font-size: 0.72rem;
      color: #cbd5e1;
      line-height: 1.5;
    }

    .copy-btn {
      position: absolute;
      top: 0.5rem;
      right: 0.5rem;
      background: rgba(255, 255, 255, 0.08);
      border: 1px solid var(--border-subtle);
      border-radius: 4px;
      color: var(--text-muted);
      padding: 0.25rem 0.5rem;
      font-size: 0.7rem;
      cursor: pointer;
      transition: all 0.2s;
    }

    .copy-btn:hover {
      color: #fff;
      background: rgba(255, 255, 255, 0.16);
    }
  </style>
</head>
<body>
  <!-- Header -->
  <header>
    <a href="/" class="brand">
      <div class="brand-icon">
        <svg viewBox="0 0 24 24"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/></svg>
      </div>
      <div class="brand-text">RAMLLM Studio</div>
      <span class="badge-live"><span class="dot"></span> Serving</span>
    </a>
    <div class="nav-actions">
      <a href="/docs" target="_blank" class="nav-link btn-api-docs">Interactive API Docs</a>
      <a href="/ramllm/metrics" target="_blank" class="nav-link">JSON Metrics</a>
      <a href="/healthz" target="_blank" class="nav-link">Health Check</a>
    </div>
  </header>

  <!-- Main Layout -->
  <main>
    <!-- Left Sidebar: Model & Memory Governor HUD -->
    <aside class="sidebar-left">
      <div>
        <div class="section-title">Active Model</div>
        <div class="hud-card">
          <div class="model-title" id="hud-model-name">{{MODEL_NAME}}</div>
          <span class="model-tag" id="hud-model-arch">{{ARCH}}</span>
          <div style="font-size:0.75rem; color:var(--text-dim); margin-top:0.5rem;" id="hud-model-plan">Plan: Initializing...</div>
        </div>
      </div>

      <div>
        <div class="section-title">Memory Governor HUD</div>
        <div class="hud-card">
          <div class="metric-row">
            <div class="metric-header">
              <span class="metric-label">System RAM RSS</span>
              <span class="metric-val" id="ram-text">-- / -- GB</span>
            </div>
            <div class="progress-track">
              <div class="progress-fill" id="ram-bar" style="width: 0%;"></div>
            </div>
          </div>

          <div class="metric-row">
            <div class="metric-header">
              <span class="metric-label">GPU VRAM</span>
              <span class="metric-val" id="vram-text">-- / -- GB</span>
            </div>
            <div class="progress-track">
              <div class="progress-fill vram" id="vram-bar" style="width: 0%;"></div>
            </div>
          </div>

          <div class="governor-status" id="gov-status">
            <span>&#9679;</span> Governor: Active &amp; Monitoring
          </div>
        </div>
      </div>

      <div>
        <div class="section-title">Inference Parameters</div>
        <div class="hud-card control-group">
          <div class="control-item">
            <label>
              <span>Temperature</span>
              <span id="temp-val" class="metric-val">0.7</span>
            </label>
            <input type="range" id="param-temp" min="0" max="1.5" step="0.05" value="0.7">
          </div>

          <div class="control-item">
            <label>
              <span>Max Tokens</span>
              <span id="max-tokens-val" class="metric-val">512</span>
            </label>
            <input type="range" id="param-max-tokens" min="64" max="2048" step="64" value="512">
          </div>

          <div class="control-item">
            <label><span>System Prompt</span></label>
            <textarea id="param-system" rows="2" placeholder="You are a helpful AI assistant..."></textarea>
          </div>

          <div class="control-item">
            <label><span>API Key (Optional)</span></label>
            <input type="password" id="param-api-key" placeholder="Enter RAMLLM_API_KEY if set">
          </div>
        </div>
      </div>
    </aside>

    <!-- Center Chat Column -->
    <section class="chat-col">
      <div class="chat-messages" id="chat-messages">
        <!-- Empty State / Welcome -->
        <div class="empty-chat" id="empty-state">
          <div class="empty-icon">
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
            </svg>
          </div>
          <div class="empty-title">Chat with {{MODEL_NAME}}</div>
          <div class="empty-subtitle">
            RAMLLM is serving this model with active memory governor offload and OpenAI-compatible streaming.
          </div>
          <div class="preset-chips">
            <button class="preset-btn" onclick="sendPreset('Hello! Introduce yourself briefly and tell me your key capabilities.')">
              <span>Hello! Introduce yourself briefly.</span>
              <span style="color:var(--cyan);">&#8594;</span>
            </button>
            <button class="preset-btn" onclick="sendPreset('Explain how memory governor and partial offloading work in LLM inference.')">
              <span>Explain how memory governor offloading works.</span>
              <span style="color:var(--cyan);">&#8594;</span>
            </button>
            <button class="preset-btn" onclick="sendPreset('Write a concise Python script using httpx to call this /v1/chat/completions endpoint.')">
              <span>Write a Python script calling this API.</span>
              <span style="color:var(--cyan);">&#8594;</span>
            </button>
          </div>
        </div>
      </div>

      <!-- Input Bar -->
      <div class="chat-input-bar">
        <div class="input-wrapper">
          <button class="action-btn btn-clear" id="btn-clear" title="Clear chat history" onclick="clearChat()">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <polyline points="3 6 5 6 21 6"></polyline>
              <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
            </svg>
          </button>
          <textarea id="prompt-input" rows="1" placeholder="Type a message... (Press Enter to send, Shift+Enter for newline)"></textarea>
          <button class="action-btn" id="btn-send" onclick="handleSend()">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <line x1="22" y1="2" x2="11" y2="13"></line>
              <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
            </svg>
          </button>
        </div>
      </div>
    </section>

    <!-- Right Drawer: API & cURL Playground -->
    <aside class="drawer-right">
      <div>
        <div class="section-title">OpenAI Compatible API</div>
        <p style="font-size:0.8rem; color:var(--text-muted); line-height:1.4; margin-bottom:0.75rem;">
          Connect your apps, LangChain, LlamaIndex, or OpenAI SDK directly to this instance.
        </p>

        <div class="tab-nav">
          <button class="tab-btn active" onclick="switchSnippet('curl')">cURL</button>
          <button class="tab-btn" onclick="switchSnippet('python')">Python</button>
          <button class="tab-btn" onclick="switchSnippet('js')">JavaScript</button>
        </div>

        <div class="code-box" style="margin-top: 0.65rem;">
          <button class="copy-btn" onclick="copySnippet()">Copy</button>
          <pre id="snippet-code"></pre>
        </div>
      </div>

      <div class="hud-card" style="margin-top:auto;">
        <div class="section-title">Available Endpoints</div>
        <div style="font-size:0.75rem; font-family:var(--font-mono); line-height:1.8; color:var(--text-muted);">
          <div><strong style="color:var(--emerald);">POST</strong> /v1/chat/completions</div>
          <div><strong style="color:var(--emerald);">POST</strong> /v1/completions</div>
          <div><strong style="color:var(--cyan);">GET</strong>  /v1/models</div>
          <div><strong style="color:var(--cyan);">GET</strong>  /ramllm/plan</div>
          <div><strong style="color:var(--cyan);">GET</strong>  /ramllm/metrics</div>
        </div>
      </div>
    </aside>
  </main>

  <script>
    // State
    let conversation = [];
    let isGenerating = false;
    let abortController = null;
    let currentTab = 'curl';

    // Elements
    const chatContainer = document.getElementById('chat-messages');
    const emptyState = document.getElementById('empty-state');
    const promptInput = document.getElementById('prompt-input');
    const btnSend = document.getElementById('btn-send');
    const tempSlider = document.getElementById('param-temp');
    const tempVal = document.getElementById('temp-val');
    const maxTokensSlider = document.getElementById('param-max-tokens');
    const maxTokensVal = document.getElementById('max-tokens-val');

    tempSlider.addEventListener('input', (e) => tempVal.textContent = e.target.value);
    maxTokensSlider.addEventListener('input', (e) => maxTokensVal.textContent = e.target.value);

    // Auto-resize textarea
    promptInput.addEventListener('input', function() {
      this.style.height = 'auto';
      this.style.height = Math.min(this.scrollHeight, 160) + 'px';
    });

    // Enter to send
    promptInput.addEventListener('keydown', function(e) {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    });

    function sendPreset(text) {
      promptInput.value = text;
      handleSend();
    }

    function clearChat() {
      conversation = [];
      chatContainer.innerHTML = '';
      chatContainer.appendChild(emptyState);
      emptyState.style.display = 'flex';
    }

    function escapeHtml(unsafe) {
      return unsafe
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
    }

    function renderMarkdown(text) {
      // Safe, lightweight markdown renderer for code blocks and bold
      let html = escapeHtml(text);
      // Code blocks
      html = html.replace(/```([\\w-]*)\\n([\\s\\S]*?)```/g, function(match, lang, code) {
        return '<pre><code>' + code.trim() + '</code></pre>';
      });
      // Inline code
      html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
      // Bold
      html = html.replace(/\\*\\*([^\\*]+)\\*\\*/g, '<strong>$1</strong>');
      // Line breaks
      html = html.replace(/\\n/g, '<br>');
      return html;
    }

    async function handleSend() {
      const text = promptInput.value.trim();
      if (!text || isGenerating) return;

      if (emptyState && emptyState.style.display !== 'none') {
        emptyState.style.display = 'none';
      }

      // Append User message
      conversation.push({ role: 'user', content: text });
      appendMessage('user', text);
      promptInput.value = '';
      promptInput.style.height = 'auto';

      // Prepare Assistant placeholder
      const assistantMsgObj = { role: 'assistant', content: '' };
      conversation.push(assistantMsgObj);
      const { bubbleEl, statsEl, rowEl } = appendMessage('assistant', '');

      isGenerating = true;
      btnSend.innerHTML = `
        <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
          <rect x="6" y="6" width="12" height="12" rx="2"></rect>
        </svg>
      `;
      btnSend.classList.add('btn-stop');
      btnSend.onclick = stopGeneration;

      abortController = new AbortController();
      const startTime = performance.now();
      let tokenCount = 0;

      const apiKey = document.getElementById('param-api-key').value.trim();
      const headers = { 'Content-Type': 'application/json' };
      if (apiKey) {
        headers['Authorization'] = 'Bearer ' + apiKey;
      }

      // Build messages payload including system prompt
      const messages = [];
      const sysPrompt = document.getElementById('param-system').value.trim();
      if (sysPrompt) {
        messages.push({ role: 'system', content: sysPrompt });
      }
      // Include past turns
      for (const m of conversation.slice(0, -1)) {
        messages.push({ role: m.role, content: m.content });
      }

      try {
        const response = await fetch('/v1/chat/completions', {
          method: 'POST',
          headers: headers,
          signal: abortController.signal,
          body: JSON.stringify({
            messages: messages,
            temperature: parseFloat(tempSlider.value),
            max_tokens: parseInt(maxTokensSlider.value),
            stream: true
          })
        });

        if (!response.ok) {
          const errData = await response.json().catch(() => ({ detail: response.statusText }));
          throw new Error(errData.detail || errData.error?.message || 'Server error');
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
          const { value, done } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\\n');
          buffer = lines.pop(); // keep last incomplete line

          for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed || trimmed.startsWith(':')) continue;
            if (trimmed === 'data: [DONE]') continue;

            if (trimmed.startsWith('data: ')) {
              try {
                const json = JSON.parse(trimmed.slice(6));
                const delta = json.choices?.[0]?.delta?.content || '';
                if (delta) {
                  assistantMsgObj.content += delta;
                  tokenCount++;
                  bubbleEl.innerHTML = renderMarkdown(assistantMsgObj.content) + '<span class="typing-cursor"></span>';
                  chatContainer.scrollTop = chatContainer.scrollHeight;
                }
              } catch (e) {
                // partial chunk or keepalive
              }
            }
          }
        }

        // Finalize
        bubbleEl.innerHTML = renderMarkdown(assistantMsgObj.content);
        const durationSec = ((performance.now() - startTime) / 1000).toFixed(1);
        const tps = durationSec > 0 ? (tokenCount / durationSec).toFixed(1) : '0';
        statsEl.innerHTML = `<span>⚡ ${tps} tok/s</span> <span>${tokenCount} tokens</span> <span>${durationSec}s</span>`;

      } catch (err) {
        if (err.name === 'AbortError') {
          bubbleEl.innerHTML = renderMarkdown(assistantMsgObj.content) + '<span style="color:var(--rose); font-size:0.75rem; margin-left:8px;">(Stopped)</span>';
        } else {
          bubbleEl.innerHTML = `<span style="color:var(--rose);">Error: ${escapeHtml(err.message)}</span>`;
        }
      } finally {
        isGenerating = false;
        btnSend.classList.remove('btn-stop');
        btnSend.innerHTML = `
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <line x1="22" y1="2" x2="11" y2="13"></line>
            <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
          </svg>
        `;
        btnSend.onclick = handleSend;
      }
    }

    function stopGeneration() {
      if (abortController) {
        abortController.abort();
      }
    }

    function appendMessage(role, content) {
      const row = document.createElement('div');
      row.className = 'msg-row ' + role;

      const avatar = document.createElement('div');
      avatar.className = 'msg-avatar';
      avatar.textContent = role === 'user' ? 'U' : 'AI';

      const contentBox = document.createElement('div');
      contentBox.style.display = 'flex';
      contentBox.style.flexDirection = 'column';

      const bubble = document.createElement('div');
      bubble.className = 'msg-bubble';
      bubble.innerHTML = renderMarkdown(content);

      contentBox.appendChild(bubble);

      let stats = null;
      if (role === 'assistant') {
        stats = document.createElement('div');
        stats.className = 'msg-stats';
        contentBox.appendChild(stats);
      }

      row.appendChild(avatar);
      row.appendChild(contentBox);
      chatContainer.appendChild(row);
      chatContainer.scrollTop = chatContainer.scrollHeight;

      return { bubbleEl: bubble, statsEl: stats, rowEl: row };
    }

    // Snippet Playground
    const snippets = {
      curl: `curl http://127.0.0.1:8080/v1/chat/completions \\\\
  -H "Content-Type: application/json" \\\\
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "temperature": 0.7,
    "stream": false
  }'`,
      python: `from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="none"
)

response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True
)

for chunk in response:
    content = chunk.choices[0].delta.content or ""
    print(content, end="", flush=True)`,
      js: `const response = await fetch("http://127.0.0.1:8080/v1/chat/completions", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    messages: [{ role: "user", content: "Hello!" }],
    stream: false
  })
});

const data = await response.json();
console.log(data.choices[0].message.content);`
    };

    function switchSnippet(tab) {
      currentTab = tab;
      document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
      event.target.classList.add('active');
      document.getElementById('snippet-code').textContent = snippets[tab];
    }

    function copySnippet() {
      navigator.clipboard.writeText(snippets[currentTab]);
      const btn = event.target;
      btn.textContent = 'Copied!';
      setTimeout(() => btn.textContent = 'Copy', 1500);
    }

    // Initial snippet
    document.getElementById('snippet-code').textContent = snippets.curl;

    // Metrics & Plan Polling
    async function updatePlan() {
      try {
        const res = await fetch('/ramllm/plan');
        if (res.ok) {
          const plan = await res.json();
          document.getElementById('hud-model-plan').textContent =
            'Plan: ' + plan.id + ' (' + plan.gpu_layers + ' GPU layers)';
        }
      } catch (e) {}
    }

    async function pollMetrics() {
      try {
        const res = await fetch('/ramllm/metrics');
        if (res.ok) {
          const m = await res.json();
          const ramUsedGb = (m.ram_rss_bytes / (1024**3)).toFixed(1);
          const ramBudgetGb = m.budget_ram_bytes ? (m.budget_ram_bytes / (1024**3)).toFixed(1) : '16.0';
          const ramPct = m.budget_ram_bytes ? Math.min(100, Math.round((m.ram_rss_bytes / m.budget_ram_bytes) * 100)) : 0;

          document.getElementById('ram-text').textContent = ramUsedGb + ' / ' + ramBudgetGb + ' GB';
          document.getElementById('ram-bar').style.width = ramPct + '%';

          const vramUsedGb = (m.vram_used_bytes / (1024**3)).toFixed(1);
          const vramBudgetGb = m.budget_vram_bytes ? (m.budget_vram_bytes / (1024**3)).toFixed(1) : '8.0';
          const vramPct = m.budget_vram_bytes ? Math.min(100, Math.round((m.vram_used_bytes / m.budget_vram_bytes) * 100)) : 0;

          document.getElementById('vram-text').textContent = vramUsedGb + ' / ' + vramBudgetGb + ' GB';
          document.getElementById('vram-bar').style.width = vramPct + '%';

          const govStatus = document.getElementById('gov-status');
          if (m.budget_exceeded) {
            govStatus.style.color = 'var(--rose)';
            govStatus.innerHTML = '<span>&#9888;</span> Governor: Memory Pressure Detected';
          } else {
            govStatus.style.color = 'var(--emerald)';
            govStatus.innerHTML = '<span>&#9679;</span> Governor: Active &amp; Optimal';
          }
        }
      } catch (e) {}
    }

    updatePlan();
    pollMetrics();
    setInterval(pollMetrics, 2500);
  </script>
</body>
</html>"""


def get_web_ui_html(model_name: str = "RAMLLM Model", arch: str = "GGUF") -> str:
    """Return standalone, responsive HTML/CSS/JS for the RAMLLM Web UI."""
    return WEB_UI_HTML.replace("{{MODEL_NAME}}", model_name).replace("{{ARCH}}", arch)
