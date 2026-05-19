<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>NSE Options Strategy Advisor</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: #f9fafb;
      color: #111827;
      padding: 20px;
      line-height: 1.4;
    }
    .container { max-width: 1200px; margin: 0 auto; }

    header {
      background: linear-gradient(135deg, #1e3a8a 0%, #6d28d9 100%);
      color: #fff;
      padding: 24px 20px;
      border-radius: 12px;
      margin-bottom: 20px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 10px;
    }
    header h1 { font-size: 22px; font-weight: 700; margin-bottom: 4px; }
    header .subtitle { font-size: 13px; opacity: 0.9; }
    header a.back-link {
      background: rgba(255,255,255,0.15);
      padding: 8px 14px;
      border-radius: 8px;
      color: #fff;
      text-decoration: none;
      font-size: 13px;
      font-weight: 500;
    }
    header a.back-link:hover { background: rgba(255,255,255,0.25); }

    .input-card {
      background: #fff;
      border: 1px solid #e5e7eb;
      border-radius: 10px;
      padding: 20px;
      margin-bottom: 20px;
    }
    .input-row {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      align-items: center;
    }
    .input-row input[type="text"] {
      flex: 1;
      min-width: 200px;
      padding: 12px 16px;
      font-size: 16px;
      border: 1px solid #d1d5db;
      border-radius: 8px;
      font-family: ui-monospace, monospace;
      text-transform: uppercase;
    }
    .input-row input[type="text"]:focus {
      outline: none;
      border-color: #6d28d9;
      box-shadow: 0 0 0 3px rgba(109, 40, 217, 0.1);
    }
    .input-row button {
      padding: 12px 24px;
      background: #6d28d9;
      color: #fff;
      border: none;
      border-radius: 8px;
      cursor: pointer;
      font-size: 14px;
      font-weight: 600;
    }
    .input-row button:hover { background: #5b21b6; }
    .input-row button:disabled { background: #9ca3af; cursor: not-allowed; }

    .hint {
      font-size: 12px;
      color: #6b7280;
      margin-top: 8px;
    }

    .status {
      padding: 12px 16px;
      border-radius: 8px;
      font-size: 14px;
      margin-bottom: 16px;
    }
    .status.error   { background: #fee2e2; color: #991b1b; }
    .status.warning { background: #fef3c7; color: #92400e; }

    .loading-spinner {
      text-align: center;
      padding: 60px 20px;
      color: #6b7280;
    }
    .loading-spinner::after {
      content: "⏳";
      font-size: 32px;
      display: block;
      margin-top: 10px;
      animation: pulse 1.5s ease-in-out infinite;
    }
    @keyframes pulse { 0%, 100% { opacity: 0.4; } 50% { opacity: 1; } }

    /* Outlook summary card */
    .outlook-card {
      background: #fff;
      border: 1px solid #e5e7eb;
      border-radius: 10px;
      padding: 18px;
      margin-bottom: 18px;
    }
    .outlook-header {
      font-size: 14px;
      font-weight: 600;
      color: #6d28d9;
      margin-bottom: 12px;
    }
    .outlook-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 10px;
    }
    .outlook-item {
      padding: 10px 12px;
      background: #faf5ff;
      border-radius: 6px;
    }
    .outlook-label {
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      color: #6b7280;
      margin-bottom: 4px;
    }
    .outlook-value {
      font-weight: 700;
      font-size: 16px;
    }
    .outlook-value.bullish  { color: #16a34a; }
    .outlook-value.bearish  { color: #dc2626; }
    .outlook-value.neutral  { color: #6b7280; }
    .outlook-value.high     { color: #dc2626; }
    .outlook-value.low      { color: #16a34a; }
    .outlook-value.normal   { color: #6b7280; }
    .outlook-value.strong   { color: #6d28d9; }
    .outlook-value.moderate { color: #2563eb; }
    .outlook-value.weak     { color: #9ca3af; }

    /* Top pick — featured card */
    .top-pick-card {
      background: linear-gradient(135deg, #faf5ff 0%, #fff 100%);
      border: 2px solid #6d28d9;
      border-radius: 12px;
      padding: 20px;
      margin-bottom: 20px;
      position: relative;
    }
    .top-pick-badge {
      position: absolute;
      top: -10px;
      left: 20px;
      background: #6d28d9;
      color: #fff;
      padding: 4px 12px;
      border-radius: 99px;
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.5px;
    }
    .strategy-name {
      font-size: 22px;
      font-weight: 700;
      color: #111827;
      margin-bottom: 6px;
      margin-top: 6px;
    }
    .strategy-tags {
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
      margin-bottom: 14px;
    }
    .strategy-tag {
      padding: 3px 10px;
      border-radius: 4px;
      font-size: 11px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.3px;
    }
    .tag-bullish     { background: #d1fae5; color: #065f46; }
    .tag-bearish     { background: #fee2e2; color: #991b1b; }
    .tag-neutral     { background: #f3f4f6; color: #374151; }
    .tag-volatility  { background: #fef3c7; color: #92400e; }
    .tag-defined     { background: #dbeafe; color: #1e40af; }
    .tag-undefined   { background: #fed7aa; color: #9a3412; }
    .tag-fit         { background: #ede9fe; color: #5b21b6; }

    /* Legs table */
    .legs-section {
      margin: 14px 0;
    }
    .section-header-small {
      font-size: 12px;
      font-weight: 600;
      color: #6b7280;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      margin-bottom: 8px;
    }
    .legs-table {
      width: 100%;
      border-collapse: collapse;
      background: #fff;
      border: 1px solid #e5e7eb;
      border-radius: 6px;
      overflow: hidden;
    }
    .legs-table th {
      background: #f9fafb;
      padding: 8px 10px;
      text-align: left;
      font-size: 11px;
      font-weight: 600;
      color: #6b7280;
      text-transform: uppercase;
    }
    .legs-table td {
      padding: 8px 10px;
      font-size: 13px;
      font-family: ui-monospace, monospace;
      border-bottom: 1px solid #f3f4f6;
    }
    .legs-table tr:last-child td { border-bottom: none; }
    .leg-action-buy  { color: #16a34a; font-weight: 700; }
    .leg-action-sell { color: #dc2626; font-weight: 700; }

    /* P/L metrics grid */
    .pl-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 10px;
      margin: 14px 0;
    }
    .pl-metric {
      background: #fff;
      padding: 12px;
      border-radius: 6px;
      border: 1px solid #e5e7eb;
    }
    .pl-label {
      font-size: 11px;
      text-transform: uppercase;
      color: #6b7280;
      margin-bottom: 4px;
    }
    .pl-value {
      font-size: 18px;
      font-weight: 700;
      font-family: ui-monospace, monospace;
    }
    .pl-value.profit { color: #16a34a; }
    .pl-value.loss   { color: #dc2626; }
    .pl-value.cap    { color: #1f2937; }
    .pl-value-sm     { font-size: 14px; font-weight: 600; }

    /* Fit reason */
    .fit-reason {
      background: #faf5ff;
      border-left: 3px solid #6d28d9;
      padding: 8px 12px;
      font-size: 13px;
      color: #5b21b6;
      margin-top: 12px;
      border-radius: 0 6px 6px 0;
    }

    /* AI analysis */
    .ai-card {
      background: #fff;
      border: 1px solid #e5e7eb;
      border-radius: 10px;
      padding: 18px;
      margin-bottom: 20px;
    }
    .ai-card-header {
      font-size: 14px;
      font-weight: 700;
      color: #6d28d9;
      margin-bottom: 12px;
    }
    .ai-text { line-height: 1.7; font-size: 14px; }
    .ai-text h2 {
      font-size: 15px;
      margin-top: 16px;
      margin-bottom: 8px;
      color: #111827;
      padding-bottom: 4px;
      border-bottom: 1px solid #e5e7eb;
    }
    .ai-text h2:first-child { margin-top: 0; }
    .ai-text ul { margin: 6px 0 6px 22px; }
    .ai-text li { margin: 3px 0; }
    .ai-text strong { color: #111827; }
    .ai-text p { margin: 6px 0; }
    .ai-text em { color: #6b7280; font-size: 13px; }

    /* Alternatives */
    .alternatives-section {
      margin-top: 20px;
    }
    .alternatives-header {
      font-size: 14px;
      font-weight: 700;
      color: #374151;
      margin-bottom: 12px;
    }
    .alt-card {
      background: #fff;
      border: 1px solid #e5e7eb;
      border-radius: 10px;
      padding: 16px;
      margin-bottom: 12px;
      cursor: pointer;
    }
    .alt-card-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 8px;
    }
    .alt-name {
      font-size: 16px;
      font-weight: 600;
      color: #111827;
    }
    .alt-body {
      display: none;
      margin-top: 14px;
      padding-top: 14px;
      border-top: 1px solid #f3f4f6;
    }
    .alt-card.expanded .alt-body { display: block; }
    .alt-toggle {
      font-size: 12px;
      color: #6b7280;
    }

    /* All strategies summary */
    .all-strats-section {
      margin-top: 20px;
      background: #fff;
      border: 1px solid #e5e7eb;
      border-radius: 10px;
      padding: 16px;
    }
    .all-strats-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
      gap: 8px;
      margin-top: 10px;
    }
    .all-strat-item {
      padding: 8px 10px;
      background: #f9fafb;
      border-radius: 6px;
      font-size: 12px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .all-strat-item .strat-name { font-weight: 500; color: #374151; }
    .all-strat-item .fit-badge {
      padding: 2px 6px;
      border-radius: 4px;
      font-size: 11px;
      font-weight: 700;
    }
    .fit-badge.high   { background: #d1fae5; color: #065f46; }
    .fit-badge.mid    { background: #fef3c7; color: #92400e; }
    .fit-badge.low    { background: #fee2e2; color: #991b1b; }

    /* Disclaimer footer */
    .disclaimer {
      background: #fffbeb;
      border: 1px solid #fde68a;
      border-radius: 8px;
      padding: 12px 16px;
      font-size: 12px;
      color: #92400e;
      margin-top: 20px;
    }

    @media (max-width: 640px) {
      body { padding: 10px; }
      header { padding: 16px; }
      header h1 { font-size: 18px; }
      .strategy-name { font-size: 18px; }
      .pl-grid { grid-template-columns: repeat(2, 1fr); }
      .legs-table { font-size: 12px; }
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div>
        <h1>🎯 Options Strategy Advisor</h1>
        <div class="subtitle">AI-powered strategy recommendation with live option prices</div>
      </div>
      <a class="back-link" href="index.html">← Back to Dashboard</a>
    </header>

    <div class="input-card">
      <div class="input-row">
        <input type="text" id="symbol-input" placeholder="Enter NSE symbol (e.g. RELIANCE)" autocomplete="off">
        <button id="analyze-btn" type="button">Find Strategy</button>
      </div>
      <div class="hint">
        Analyses 26 strategies across directional, spreads, volatility, condor/butterfly, and hybrids. ~15-30 seconds.
        Stock must be in NSE F&O segment.
      </div>
    </div>

    <div id="result"></div>
  </div>

  <script>
    const STRATEGY_URL = "https://nse-dashboard-api.onrender.com/api/strategy";

    const $ = (id) => document.getElementById(id);
    const fmt = (n, places = 2) =>
      Number(n).toLocaleString("en-IN", { minimumFractionDigits: places, maximumFractionDigits: places });
    const fmtInt = (n) => Math.round(Number(n)).toLocaleString("en-IN");

    function escapeHtml(s) {
      return String(s ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    }

    // Minimal markdown → HTML for AI analysis output
    function mdToHtml(md) {
      const lines = String(md || "").split("\n");
      let html = "";
      let inUl = false;
      for (const raw of lines) {
        const line = raw.trim();
        if (!line) {
          if (inUl) { html += "</ul>"; inUl = false; }
          continue;
        }
        if (line.startsWith("## ")) {
          if (inUl) { html += "</ul>"; inUl = false; }
          html += `<h2>${escapeHtml(line.slice(3))}</h2>`;
        } else if (line.startsWith("- ") || line.startsWith("* ")) {
          if (!inUl) { html += "<ul>"; inUl = true; }
          html += `<li>${inlineMd(line.slice(2))}</li>`;
        } else {
          if (inUl) { html += "</ul>"; inUl = false; }
          html += `<p>${inlineMd(line)}</p>`;
        }
      }
      if (inUl) html += "</ul>";
      return html;
    }

    function inlineMd(text) {
      let t = escapeHtml(text);
      t = t.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
      t = t.replace(/\*([^*]+)\*/g,    "<em>$1</em>");
      return t;
    }

    // Submit handler
    $("analyze-btn").addEventListener("click", () => {
      const symbol = $("symbol-input").value.trim().toUpperCase();
      if (symbol) analyze(symbol);
    });
    $("symbol-input").addEventListener("keypress", (e) => {
      if (e.key === "Enter") {
        const symbol = e.target.value.trim().toUpperCase();
        if (symbol) analyze(symbol);
      }
    });
    $("symbol-input").focus();

    async function analyze(symbol) {
      const btn = $("analyze-btn");
      const result = $("result");

      btn.disabled = true;
      btn.textContent = "Analyzing…";
      result.innerHTML = `
        <div class="loading-spinner">
          Building strategy recommendation for ${escapeHtml(symbol)}…<br>
          <span style="font-size:13px;">Price → option chain → outlook → 26 strategy builders → AI explanation</span><br>
          <span style="font-size:12px;color:#9ca3af;">(First request may take ~30s if server is cold)</span>
        </div>
      `;

      try {
        const res = await fetch(STRATEGY_URL, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ symbol }),
        });
        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          throw new Error(err.message || err.error || `HTTP ${res.status}`);
        }
        const data = await res.json();
        renderResult(data);
      } catch (err) {
        result.innerHTML = `<div class="status error">⚠️ ${escapeHtml(err.message)}</div>`;
      } finally {
        btn.disabled = false;
        btn.textContent = "Find Strategy";
      }
    }

    function renderResult(data) {
      const result = $("result");
      const outlook = data.outlook;
      const top = data.top_pick;

      // ===== Outlook summary =====
      const outlookHTML = `
        <div class="outlook-card">
          <div class="outlook-header">📊 Market Outlook for ${escapeHtml(data.symbol)}</div>
          <div class="outlook-grid">
            <div class="outlook-item">
              <div class="outlook-label">Spot Price</div>
              <div class="outlook-value">₹${fmt(data.spot_price)}</div>
            </div>
            <div class="outlook-item">
              <div class="outlook-label">Direction</div>
              <div class="outlook-value ${outlook.direction}">${outlook.direction.toUpperCase()}</div>
            </div>
            <div class="outlook-item">
              <div class="outlook-label">Conviction</div>
              <div class="outlook-value ${outlook.conviction}">${outlook.conviction.toUpperCase()}</div>
            </div>
            <div class="outlook-item">
              <div class="outlook-label">IV Regime</div>
              <div class="outlook-value ${outlook.iv_regime}">${outlook.iv_regime.toUpperCase()}</div>
            </div>
            <div class="outlook-item">
              <div class="outlook-label">Expiry</div>
              <div class="outlook-value" style="font-size:13px;">${escapeHtml(data.expiry_date)}</div>
            </div>
            <div class="outlook-item">
              <div class="outlook-label">Days Left</div>
              <div class="outlook-value">${data.days_to_expiry}d</div>
            </div>
            <div class="outlook-item">
              <div class="outlook-label">Lot Size</div>
              <div class="outlook-value">${fmtInt(data.lot_size)}</div>
            </div>
            <div class="outlook-item">
              <div class="outlook-label">Signal Score</div>
              <div class="outlook-value" style="font-size:13px;">
                Bull ${outlook.bullish_signals} / Bear ${outlook.bearish_signals}
              </div>
            </div>
          </div>
        </div>
      `;

      // ===== Top pick card =====
      const topPickHTML = renderStrategyCard(top, true);

      // ===== AI analysis =====
      const aiHTML = `
        <div class="ai-card">
          <div class="ai-card-header">🤖 AI Strategy Analysis</div>
          <div class="ai-text">${mdToHtml(data.ai_analysis)}</div>
        </div>
      `;

      // ===== Alternatives =====
      const altsHTML = data.alternatives && data.alternatives.length > 0 ? `
        <div class="alternatives-section">
          <div class="alternatives-header">📋 Alternative Strategies</div>
          ${data.alternatives.map((alt, i) => `
            <div class="alt-card" data-alt-idx="${i}">
              <div class="alt-card-header">
                <div>
                  <div class="alt-name">${escapeHtml(alt.name)}</div>
                  <div class="strategy-tags" style="margin-top:6px;">
                    <span class="strategy-tag tag-${alt.direction_bias}">${alt.direction_bias}</span>
                    <span class="strategy-tag tag-${alt.risk_profile}">${alt.risk_profile} risk</span>
                    <span class="strategy-tag tag-fit">Fit: ${alt.fit_score}/100</span>
                  </div>
                </div>
                <div class="alt-toggle">▼ Details</div>
              </div>
              <div class="alt-body">${renderStrategyBody(alt)}</div>
            </div>
          `).join("")}
        </div>
      ` : "";

      // ===== All strategies summary =====
      const allStratsHTML = `
        <div class="all-strats-section">
          <div class="alternatives-header">🔍 All Strategies Ranked (${data.all_results_summary.length} analyzed)</div>
          <div class="all-strats-grid">
            ${data.all_results_summary.map(r => {
              const fitClass = r.fit_score >= 60 ? 'high' : r.fit_score >= 40 ? 'mid' : 'low';
              return `
                <div class="all-strat-item">
                  <span class="strat-name">${escapeHtml(r.name)}</span>
                  <span class="fit-badge ${fitClass}">${r.fit_score}</span>
                </div>
              `;
            }).join("")}
          </div>
        </div>
      `;

      // ===== Disclaimer =====
      const disclaimerHTML = `
        <div class="disclaimer">
          ⚠️ <strong>Important:</strong> Option prices shown use last-traded values, which can deviate from live bid/ask by 5-15%.
          For execution, always reference live quotes in your broker terminal. These are educational suggestions, not orders.
          Lot sizes from NSE; verify before placing orders. F&O strategies require margin — check your broker for actual requirements.
        </div>
      `;

      result.innerHTML = outlookHTML + topPickHTML + aiHTML + altsHTML + allStratsHTML + disclaimerHTML;

      // Attach alt-card toggle handlers
      document.querySelectorAll(".alt-card").forEach(card => {
        card.addEventListener("click", () => {
          card.classList.toggle("expanded");
          const toggle = card.querySelector(".alt-toggle");
          toggle.textContent = card.classList.contains("expanded") ? "▲ Hide" : "▼ Details";
        });
      });
    }

    function renderStrategyCard(strat, isTopPick) {
      return `
        <div class="top-pick-card">
          ${isTopPick ? '<div class="top-pick-badge">⭐ TOP PICK</div>' : ''}
          <div class="strategy-name">${escapeHtml(strat.name)}</div>
          <div class="strategy-tags">
            <span class="strategy-tag tag-${strat.direction_bias}">${strat.direction_bias}</span>
            <span class="strategy-tag tag-${strat.risk_profile}">${strat.risk_profile} risk</span>
            <span class="strategy-tag tag-fit">Fit: ${strat.fit_score}/100</span>
          </div>
          ${renderStrategyBody(strat)}
        </div>
      `;
    }

    function renderStrategyBody(strat) {
      const legsRows = strat.legs.map(leg => {
        const actionClass = leg.action === 'BUY' ? 'leg-action-buy' : 'leg-action-sell';
        const strikeDisplay = leg.strike !== null ? `₹${fmt(leg.strike, 2)}` : '—';
        const premiumDisplay = leg.instrument === 'STOCK'
          ? `₹${fmt(leg.premium)} (spot)`
          : `₹${fmt(leg.premium)}`;
        return `
          <tr>
            <td class="${actionClass}">${leg.action}</td>
            <td>${leg.quantity}× ${leg.instrument}</td>
            <td>${strikeDisplay}</td>
            <td>${premiumDisplay}</td>
          </tr>
        `;
      }).join("");

      const debitLabel = strat.net_debit > 0 ? 'Net Debit' : 'Net Credit';
      const debitValue = Math.abs(strat.net_debit);
      const debitClass = strat.net_debit > 0 ? 'loss' : 'profit';

      const maxProfitDisplay = strat.max_profit === null
        ? '<span class="profit">Unlimited</span>'
        : `<span class="profit">₹${fmtInt(strat.max_profit)}</span>`;
      const maxLossDisplay = strat.max_loss === null
        ? '<span class="loss">Unlimited</span>'
        : `<span class="loss">₹${fmtInt(strat.max_loss)}</span>`;

      const breakevensDisplay = (strat.breakevens || []).length > 0
        ? strat.breakevens.map(b => `₹${fmt(b)}`).join(" or ")
        : '—';

      return `
        <div class="legs-section">
          <div class="section-header-small">Strategy Legs</div>
          <table class="legs-table">
            <thead>
              <tr>
                <th>Action</th>
                <th>Quantity</th>
                <th>Strike</th>
                <th>Premium</th>
              </tr>
            </thead>
            <tbody>${legsRows}</tbody>
          </table>
          <div style="font-size:11px; color:#9ca3af; margin-top:6px;">
            Quantities × lot size (${fmtInt(strat.lot_size)}) for total exposure
          </div>
        </div>

        <div class="pl-grid">
          <div class="pl-metric">
            <div class="pl-label">${debitLabel}</div>
            <div class="pl-value ${debitClass}">₹${fmtInt(debitValue)}</div>
          </div>
          <div class="pl-metric">
            <div class="pl-label">Max Profit</div>
            <div class="pl-value">${maxProfitDisplay}</div>
          </div>
          <div class="pl-metric">
            <div class="pl-label">Max Loss</div>
            <div class="pl-value">${maxLossDisplay}</div>
          </div>
          <div class="pl-metric">
            <div class="pl-label">Breakeven(s)</div>
            <div class="pl-value pl-value-sm">${breakevensDisplay}</div>
          </div>
          <div class="pl-metric">
            <div class="pl-label">Capital Required</div>
            <div class="pl-value cap pl-value-sm">₹${fmtInt(strat.capital_required)}</div>
          </div>
          <div class="pl-metric">
            <div class="pl-label">Category</div>
            <div class="pl-value pl-value-sm">${escapeHtml(strat.category)}</div>
          </div>
        </div>

        ${strat.fit_reason ? `
          <div class="fit-reason">
            <strong>Why this fits:</strong> ${escapeHtml(strat.fit_reason)}
          </div>
        ` : ''}
      `;
    }
  </script>
</body>
</html>
