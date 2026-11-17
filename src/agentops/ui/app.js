"use strict";

/* multi-agent-ops console.
 *
 * One idea: every trace event the server emits is folded into a `run` model, and
 * the page is drawn from that model. A live run (SSE) and a replayed run
 * (GET /v1/runs/{id}) go through exactly the same code, so what you see while it
 * happens is what you can open again later.
 *
 * Everything the model or a tool produced is untrusted text: it is always
 * escaped before it touches innerHTML.
 */

/* ------------------------------------------------------------------ helpers */

const $ = (id) => document.getElementById(id);

/**
 * Escape text for safe use in HTML, in both element and attribute position.
 *
 * Every string that came from a model or a tool goes through this before it reaches
 * innerHTML; it is the page's only defence against injected markup.
 *
 * @param {*} value Anything; null and undefined become the empty string.
 * @returns {string} The escaped text.
 */
const esc = (value) =>
  String(value ?? "").replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));

/**
 * Format a duration for display: "840 ms", "1.41 s" or "13.9 s".
 *
 * @param {number|null} ms Milliseconds.
 * @returns {string} The formatted duration, or "" when unknown.
 */
const fmtMs = (ms) => {
  if (ms == null || Number.isNaN(ms)) return "";
  if (ms < 1000) return `${Math.max(0, Math.round(ms))} ms`;
  return `${(ms / 1000).toFixed(ms < 10000 ? 2 : 1)} s`;
};
/**
 * Format a count with thousands separators ("11,886"); missing values read as 0.
 *
 * @param {number|null} n
 * @returns {string}
 */
const fmtNum = (n) => Number(n ?? 0).toLocaleString();
/**
 * Turn an agent id into a display name: "code_reviewer" -> "Code Reviewer".
 *
 * @param {string} name
 * @returns {string}
 */
const pretty = (name) =>
  String(name).split("_").map((w) => (w ? w[0].toUpperCase() + w.slice(1) : w)).join(" ");
/**
 * Two-letter badge for an agent ("code_reviewer" -> "CR"); the orchestrator is "DL".
 *
 * The badge carries the identity in text as well as colour, so the page does not rely
 * on colour alone to tell agents apart.
 *
 * @param {string} name
 * @returns {string}
 */
const monogram = (name) =>
  name === "orchestrator"
    ? "DL"
    : String(name).split("_").map((w) => (w ? w[0].toUpperCase() : "")).join("").slice(0, 2);
/**
 * Truncate to at most n characters, ending with an ellipsis when cut.
 *
 * @param {*} text
 * @param {number} n
 * @returns {string}
 */
const clip = (text, n) => (String(text).length > n ? `${String(text).slice(0, n - 1)}…` : String(text));

const roster = {}; // agent name -> { title, mandate, tools }

const EXAMPLES = [
  {
    tag: "Fan-out",
    text: "Review PR 482 on checkout-api and tell me if it is safe to merge, and whether we can still ship 2.14.0 on Thursday.",
  },
  {
    tag: "Dependency",
    text: "Review PR 482 on checkout-api, then have someone write the regression test that the review finds is missing.",
  },
  {
    tag: "Single specialist",
    text: "checkout-api latency is spiking. What changed, and who should I page?",
  },
  {
    tag: "Answers directly",
    text: "What kinds of questions can you help me with?",
  },
];

/* ---------------------------------------------------------------- markdown */

/**
 * Render inline markdown (code, bold, italic, links) to safe HTML.
 *
 * Escape-first: each piece of text is escaped BEFORE any markup is added, so nothing the
 * model wrote can become a tag. Code spans are split out first and escaped verbatim, so
 * asterisks inside them are never treated as emphasis. Only http(s) URLs become links;
 * javascript: and data: URLs stay as plain text.
 *
 * @param {string} text One line or paragraph of markdown.
 * @returns {string} HTML.
 */
function inlineMd(text) {
  return String(text)
    .split(/(`[^`\n]+`)/g)
    .map((part) => {
      if (part.length > 2 && part.startsWith("`") && part.endsWith("`")) {
        return `<code>${esc(part.slice(1, -1))}</code>`;
      }
      return esc(part)
        .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
        .replace(/(^|[^*\w])\*([^*\s][^*\n]*?)\*(?![*\w])/g, "$1<em>$2</em>")
        .replace(
          /\[([^\]\n]+)\]\((https?:\/\/[^\s)]+)\)/g,
          '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>',
        );
    })
    .join("");
}

/**
 * Render the subset of markdown that agent reports use to safe HTML.
 *
 * Supported: fenced code, headings, rules, tables, nested bullet/numbered lists,
 * blockquotes and paragraphs, plus the inline forms in `inlineMd`. It is a deliberately
 * small hand-written renderer rather than a library, so the page has no dependency and
 * the escaping rule is easy to audit: every text node goes through `esc`.
 *
 * @param {string} src Markdown text.
 * @returns {string} HTML.
 */
function renderMarkdown(src) {
  const lines = String(src ?? "").replace(/\r\n?/g, "\n").split("\n");
  const out = [];
  let i = 0;
  let listStack = [];

  const closeLists = (toIndent = -1) => {
    while (listStack.length && listStack[listStack.length - 1].indent > toIndent) {
      out.push(`</${listStack.pop().tag}>`);
    }
  };
  const isTableSep = (line) => /^\s*\|?[\s:|-]*-[\s:|-]*\|[\s:|-]*$/.test(line || "");
  const cells = (line) =>
    line.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((c) => c.trim());

  while (i < lines.length) {
    const line = lines[i];

    const fence = line.match(/^\s*```/);
    if (fence) {
      closeLists();
      const body = [];
      i += 1;
      while (i < lines.length && !/^\s*```/.test(lines[i])) body.push(lines[i++]);
      i += 1;
      out.push(`<pre><code>${esc(body.join("\n"))}</code></pre>`);
      continue;
    }

    if (!line.trim()) { closeLists(); i += 1; continue; }

    const heading = line.match(/^(#{1,6})\s+(.*)$/);
    if (heading) {
      closeLists();
      const level = heading[1].length;
      out.push(`<h${level}>${inlineMd(heading[2])}</h${level}>`);
      i += 1;
      continue;
    }

    if (/^\s*([-*_])(\s*\1){2,}\s*$/.test(line)) { closeLists(); out.push("<hr>"); i += 1; continue; }

    if (line.trim().startsWith("|") && isTableSep(lines[i + 1])) {
      closeLists();
      const head = cells(line);
      i += 2;
      const rows = [];
      while (i < lines.length && lines[i].trim().startsWith("|")) rows.push(cells(lines[i++]));
      out.push(
        "<table><thead><tr>" + head.map((c) => `<th>${inlineMd(c)}</th>`).join("") +
        "</tr></thead><tbody>" +
        rows.map((r) => "<tr>" + r.map((c) => `<td>${inlineMd(c)}</td>`).join("") + "</tr>").join("") +
        "</tbody></table>",
      );
      continue;
    }

    const item = line.match(/^(\s*)([-*+]|\d+[.)])\s+(.*)$/);
    if (item) {
      const indent = item[1].length;
      const tag = /\d/.test(item[2]) ? "ol" : "ul";
      closeLists(indent);
      const top = listStack[listStack.length - 1];
      if (!top || top.indent < indent) { out.push(`<${tag}>`); listStack.push({ indent, tag }); }
      else if (top.tag !== tag) { out.push(`</${listStack.pop().tag}>`); out.push(`<${tag}>`); listStack.push({ indent, tag }); }
      out.push(`<li>${inlineMd(item[3])}</li>`);
      i += 1;
      continue;
    }

    if (line.startsWith(">")) {
      closeLists();
      const quote = [];
      while (i < lines.length && lines[i].startsWith(">")) quote.push(lines[i++].replace(/^>\s?/, ""));
      out.push(`<blockquote>${inlineMd(quote.join(" "))}</blockquote>`);
      continue;
    }

    closeLists();
    const para = [line];
    i += 1;
    while (
      i < lines.length && lines[i].trim() &&
      !/^(#{1,6}\s|\s*```|\s*([-*+]|\d+[.)])\s|>|\s*\|)/.test(lines[i])
    ) para.push(lines[i++]);
    out.push(`<p>${inlineMd(para.join(" "))}</p>`);
  }
  closeLists();
  return out.join("\n");
}

/* --------------------------------------------------------------- run model */

/**
 * Create the empty model of a run. Everything the page draws is derived from this.
 *
 * `status` is one of running, succeeded, partial, failed, stopped, error. `t0`/`tEnd` are
 * epoch milliseconds: from the browser's clock for a live run, from the server's event
 * timestamps for a replay. `steps` is keyed by plan step id and kept in plan order.
 *
 * @param {{id?: string, request: string, live: boolean}} init
 * @returns {object} The run model.
 */
function newRun({ id = null, request, live }) {
  return {
    id, request, live,
    t0: null, tEnd: null,
    status: "running",       // running | succeeded | partial | failed | stopped | error
    done: false,
    reasoning: "", direct: false, revisions: [],
    steps: new Map(),
    phases: { planEnd: null, synthStart: null },
    toolCalls: 0,
    events: [],
    answer: null, usage: null, durationMs: null,
    error: null,
  };
}

/**
 * Register plan steps in the model as "pending".
 *
 * Idempotent by step id, so receiving the same plan twice never resets a step's progress.
 *
 * @param {object} run
 * @param {Array} steps Step objects from a plan event ({id, agent, objective, depends_on}).
 * @param {boolean} followUp True for steps scheduled by the review round.
 */
function addSteps(run, steps, followUp) {
  for (const s of steps) {
    if (run.steps.has(s.id)) continue;
    run.steps.set(s.id, {
      id: s.id, agent: s.agent, objective: s.objective,
      dependsOn: s.depends_on || [], followUp,
      status: "pending", tools: [], turns: 0, thinking: "",
      startedAt: null, endedAt: null,
      tokens: null, truncated: false,
      output: null, error: null, durationMs: null, iterations: null,
    });
  }
}

/**
 * Return the step an event refers to, creating a placeholder if the plan never listed it.
 *
 * Defensive: it keeps the page working if events ever arrive for a step it has not seen.
 *
 * @param {object} run
 * @param {object} ev A trace event.
 * @returns {object|null} The step, or null when the event is not about a step.
 */
function ensureStep(run, ev) {
  if (!ev.step_id) return null;
  if (!run.steps.has(ev.step_id)) {
    addSteps(run, [{ id: ev.step_id, agent: ev.agent || "?", objective: ev.message || "", depends_on: (ev.data || {}).depends_on || [] }], false);
  }
  return run.steps.get(ev.step_id);
}

/**
 * Fold one trace event into the run model. This is the page's reducer.
 *
 * The same function handles a live stream and a replay of a stored run, which is why what
 * you watch live is exactly what you can reopen later. The event vocabulary is documented
 * on `EventType` in src/agentops/runtime/events.py.
 *
 * @param {object} run
 * @param {object} ev A trace event ({seq, type, agent, step_id, message, data, at}).
 */
function applyEvent(run, ev) {
  // Live: stamp events with the moment they ARRIVE, so the page's clock is self-consistent even if
  // the server's clock differs. Replay: use the server's own timestamps.
  const t = run.live ? Date.now() : Date.parse(ev.at);
  if (run.t0 === null) run.t0 = t;
  run.events.push({ ev, t });
  const d = ev.data || {};
  const s = ensureStep(run, ev);

  switch (ev.type) {
    case "plan.created":
      run.reasoning = ev.message || "";
      run.direct = !!d.direct;
      run.phases.planEnd = t;
      addSteps(run, d.steps || [], false);
      break;
    case "plan.revised":
      run.revisions.push({ reasoning: ev.message || "", at: t });
      addSteps(run, d.steps || [], true);
      break;
    case "step.started":
      s.status = "running"; s.startedAt = t;
      if (d.depends_on) s.dependsOn = d.depends_on;
      break;
    case "agent.thinking":
      s.turns += 1; s.thinking = ev.message || "";
      break;
    case "tool.called":
      s.tools.push({ name: ev.message, args: d.arguments || {}, status: "running" });
      run.toolCalls += 1;
      break;
    case "tool.result": {
      // LIMITATION: tool results carry no call id, so a result is paired with the oldest
      // still-running call of the same name. Two parallel calls to the SAME tool could swap their
      // previews; calls to different tools are always paired correctly.
      const tool = s.tools.find((x) => x.name === ev.message && x.status === "running");
      if (tool) {
        tool.status = d.ok ? "ok" : "error";
        tool.duration = d.duration_ms;
        tool.preview = d.preview || "";
      }
      break;
    }
    case "step.completed":
      s.status = "done"; s.endedAt = t; s.tokens = d.tokens ?? null; s.truncated = !!d.truncated;
      break;
    case "step.failed":
      s.status = "failed"; s.endedAt = t; s.error = ev.message || "Step failed";
      break;
    case "step.skipped":
      s.status = "skipped"; s.endedAt = t; s.error = ev.message || "Skipped";
      break;
    case "synthesis.started":
      run.phases.synthStart = t;
      break;
    case "run.completed":
      run.tEnd = t;
      break;
    case "run.failed":
      run.tEnd = t; run.error = ev.message || "The run failed";
      break;
    default: break;
  }
}

/**
 * Merge the final result into the model and mark the run finished.
 *
 * The result is authoritative for what the trace only hints at: each step's full report,
 * its server-measured duration and turn count, and the run's usage totals.
 *
 * @param {object} run
 * @param {object} res The `result` event / run response body.
 */
function applyResult(run, res) {
  run.status = res.status;
  run.answer = res.answer;
  run.usage = res.usage;
  run.durationMs = res.duration_ms;
  run.id = run.id || res.run_id;
  for (const sv of res.steps || []) {
    const s = run.steps.get(sv.step_id);
    if (!s) continue;
    s.output = sv.output;
    s.durationMs = sv.duration_ms;
    s.iterations = sv.iterations;
    s.truncated = sv.truncated;
    if (sv.error && !s.error) s.error = sv.error;
  }
  run.done = true;
  if (run.tEnd === null) run.tEnd = run.live ? Date.now() : run.t0 + (res.duration_ms || 0);
}

/**
 * The moment to draw "up to": the browser clock while a live run is in flight, and the
 * recorded end for a finished or replayed one (so a replay does not appear to keep running).
 *
 * @param {object} run
 * @returns {number} Epoch milliseconds.
 */
const nowOf = (run) => (run.done || !run.live ? (run.tEnd ?? run.t0) : Date.now());

/* ------------------------------------------------------------------ graph */

/**
 * Group steps into columns by dependency depth, which is how the plan graph is laid out.
 *
 * A step's column is the length of the longest dependency chain beneath it, so steps with
 * no dependencies share column 0 and anything that waits on them lands further right.
 * Steps in the same column are those the executor can run in parallel.
 *
 * @param {object} run
 * @returns {Array<Array<object>>} Steps grouped by column, left to right.
 */
function wavesOf(run) {
  const depth = new Map();
  const visit = (id, seen = new Set()) => {
    if (depth.has(id)) return depth.get(id);
    if (seen.has(id)) return 0;
    seen.add(id);
    const s = run.steps.get(id);
    const deps = (s?.dependsOn || []).filter((d) => run.steps.has(d));
    const value = deps.length ? 1 + Math.max(...deps.map((d) => visit(d, seen))) : 0;
    depth.set(id, value);
    return value;
  };
  for (const id of run.steps.keys()) visit(id);
  const waves = [];
  for (const [id, level] of depth) (waves[level] ||= []).push(run.steps.get(id));
  return waves.filter(Boolean);
}

/**
 * Glyph and label per step status. A glyph accompanies every colour so status is never
 * conveyed by colour alone.
 */
const STATUS = {
  pending: { glyph: "○", label: "Waiting" },
  running: { glyph: "", label: "Working" },
  done: { glyph: "✓", label: "Done" },
  failed: { glyph: "✕", label: "Failed" },
  skipped: { glyph: "⊘", label: "Skipped" },
  stopped: { glyph: "■", label: "Stopped" },
};

// Keys of <details> the user has opened (tool calls, reports). The graph is rebuilt on every event,
// so this is what stops them closing again.
const openState = new Set(); // keys of <details> the user has opened; survives re-renders

/**
 * One-line "key=value" summary of a tool call's arguments for its collapsed row.
 *
 * @param {object} args
 * @returns {string}
 */
function toolArgsSummary(args) {
  const entries = Object.entries(args || {});
  if (!entries.length) return "";
  return entries.map(([k, v]) => `${k}=${typeof v === "string" ? v : JSON.stringify(v)}`).join(", ");
}

/**
 * How long a step has run, as text.
 *
 * Uses the server-measured duration once the step has finished, since it excludes network
 * and rendering delay; before that it counts live from when the step started.
 *
 * @param {object} run
 * @param {object} s A step.
 * @returns {string} "" for a step that has not started.
 */
function stepElapsed(run, s) {
  if (s.durationMs != null && s.status !== "running") return fmtMs(s.durationMs);
  if (s.startedAt == null) return "";
  return fmtMs((s.endedAt ?? nowOf(run)) - s.startedAt);
}

/**
 * Build the card for one step: header, objective, tool calls and, when done, its report.
 *
 * `<details>` elements remember being open through `openState`, because the whole graph is
 * rebuilt on every event and would otherwise snap shut under the reader.
 *
 * @param {object} run
 * @param {object} s A step.
 * @returns {string} HTML (every dynamic value escaped).
 */
function nodeHTML(run, s) {
  const status = run.status === "stopped" && s.status === "running" ? "stopped" : s.status;
  const st = STATUS[status];
  const agent = roster[s.agent];
  const key = (suffix) => `${s.id}:${suffix}`;

  const tools = s.tools.map((t, idx) => {
    const k = key(`tool${idx}`);
    const icon = t.status === "running" ? "…" : t.status === "ok" ? "✓" : "✕";
    const args = Object.keys(t.args || {}).length ? JSON.stringify(t.args, null, 2) : "(no arguments)";
    return `<details class="tool ${t.status}" data-key="${esc(k)}"${openState.has(k) ? " open" : ""}>
      <summary><span class="ti">${icon}</span><code>${esc(t.name)}</code>
        <span class="targs">${esc(clip(toolArgsSummary(t.args), 80))}</span>
        <span class="tdur">${t.duration != null ? fmtMs(t.duration) : ""}</span></summary>
      <div class="body">
        <div class="lbl">Arguments</div><pre>${esc(args)}</pre>
        ${t.preview ? `<div class="lbl">Result</div><pre>${esc(t.preview)}</pre>` : ""}
      </div></details>`;
  }).join("");

  const reportKey = key("report");
  const report = s.output
    ? `<details class="report" data-key="${esc(reportKey)}"${openState.has(reportKey) ? " open" : ""}>
         <summary>Report from ${esc(pretty(s.agent))}</summary>
         <div class="md">${renderMarkdown(s.output)}</div></details>`
    : "";

  const meta = [];
  if (s.tools.length) meta.push(`${s.tools.length} tool call${s.tools.length === 1 ? "" : "s"}`);
  const turns = s.iterations ?? (s.turns ? s.turns + (s.status === "done" ? 1 : 0) : null);
  if (turns) meta.push(`${turns} turn${turns === 1 ? "" : "s"}`);
  if (s.tokens != null) meta.push(`${fmtNum(s.tokens)} tokens`);

  return `<article class="node ${status}" data-agent="${esc(s.agent)}" data-step="${esc(s.id)}">
    <div class="node-head">
      <span class="mono-badge" aria-hidden="true" style="width:26px;height:26px;font-size:11px;border-radius:7px">${esc(monogram(s.agent))}</span>
      <span class="who" title="${esc(agent?.title || "")}">${esc(pretty(s.agent))}</span>
      <span class="sid">${esc(s.id)}</span>
      ${s.followUp ? '<span class="follow" title="Scheduled after the Delivery Lead reviewed the first results">follow-up</span>' : ""}
      <span class="st ${status}">${status === "running" ? '<span class="spin" aria-hidden="true"></span>' : `<span aria-hidden="true">${st.glyph}</span>`}
        ${st.label}<span class="elapsed" data-step="${esc(s.id)}"> ${esc(stepElapsed(run, s))}</span></span>
    </div>
    <div class="objective">${esc(s.objective)}</div>
    ${s.dependsOn.length ? `<div class="after">after ${s.dependsOn.map(esc).join(", ")}</div>` : ""}
    ${s.status === "running" && s.thinking && !/^Calling \d+ tool/.test(s.thinking) ? `<div class="after">${esc(clip(s.thinking, 160))}</div>` : ""}
    ${meta.length ? `<div class="meta">${meta.map((m) => `<span>${esc(m)}</span>`).join("")}</div>` : ""}
    ${s.truncated ? '<div class="note warn">Ran out of tool budget — this report is partial and says what it could not verify.</div>' : ""}
    ${s.error ? `<div class="note err">${esc(s.error)}</div>` : ""}
    ${tools ? `<div class="tools">${tools}</div>` : ""}
    ${report}
  </article>`;
}

/**
 * Draw the plan: the Delivery Lead's reasoning and the step graph, or an explanatory state.
 *
 * States: answered directly (no steps), waiting for a plan, or the graph. Edges are drawn
 * after the DOM exists, since they are measured from it.
 */
function renderPlan() {
  const wrap = $("graph-wrap");
  const run = current;
  const lead = run.reasoning ? `<b>Delivery Lead:</b> ${esc(run.reasoning)}` : "";
  const revisions = run.revisions
    .map((r) => `<br><b>Follow-up:</b> ${esc(r.reasoning)}`).join("");
  $("plan-reasoning").innerHTML = lead + revisions;

  if (run.direct) {
    wrap.innerHTML = '<p class="direct">The Delivery Lead answered this itself — no specialist was needed.</p>';
    return;
  }
  if (!run.steps.size) {
    wrap.innerHTML = run.done
      ? '<p class="direct">No plan was produced.</p>'
      : '<div class="placeholder"><span class="spin"></span>The Delivery Lead is deciding who should do what…</div>';
    return;
  }

  const waves = wavesOf(run);
  wrap.innerHTML = `<div class="graph" id="graph">
    <svg class="edges" id="edges" aria-hidden="true"><defs>
      <marker id="arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto">
        <path d="M0 0 L8 4 L0 8 z"/></marker></defs></svg>
    <div class="waves" id="waves">${waves.map((wave, n) => `
      <div class="wave">
        <div class="wave-label">Step ${n + 1}${wave.length > 1 ? `<span class="par">${wave.length} in parallel</span>` : ""}</div>
        ${wave.map((s) => nodeHTML(run, s)).join("")}
      </div>`).join("")}
    </div></div>`;
  drawEdges();
}

/**
 * Draw the dependency arrows as an SVG overlay, from each step to the steps it waits on.
 *
 * Positions are measured from the rendered cards, so this must run after layout and again
 * whenever layout changes (see the ResizeObserver below). On a wide screen columns run left
 * to right and arrows leave a card's right edge; on a narrow screen they stack and arrows
 * run top to bottom. An arrow is solid once its upstream step is done, dashed while waiting.
 */
function drawEdges() {
  const graph = $("graph");
  const svg = $("edges");
  if (!graph || !svg || !current) return;
  svg.querySelectorAll("path.e").forEach((p) => p.remove());
  const box = graph.getBoundingClientRect();
  // The CSS stacks the columns on a narrow screen; the arrows must follow suit.
  const vertical = getComputedStyle($("waves")).flexDirection === "column";
  const at = (id) => graph.querySelector(`[data-step="${CSS.escape(id)}"].node`);

  for (const s of current.steps.values()) {
    const to = at(s.id);
    if (!to) continue;
    for (const dep of s.dependsOn) {
      const from = at(dep);
      if (!from) continue;
      const a = from.getBoundingClientRect();
      const b = to.getBoundingClientRect();
      let d;
      if (vertical) {
        const x1 = a.left + a.width / 2 - box.left, y1 = a.bottom - box.top;
        const x2 = b.left + b.width / 2 - box.left, y2 = b.top - box.top - 2;
        const m = (y1 + y2) / 2;
        d = `M${x1} ${y1} C${x1} ${m}, ${x2} ${m}, ${x2} ${y2}`;
      } else {
        const x1 = a.right - box.left, y1 = a.top + 24 - box.top;
        const x2 = b.left - box.left - 2, y2 = b.top + 24 - box.top;
        const m = (x1 + x2) / 2;
        d = `M${x1} ${y1} C${m} ${y1}, ${m} ${y2}, ${x2} ${y2}`;
      }
      const depStep = current.steps.get(dep);
      const cls = depStep.status === "done" ? "live" : "wait";
      const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      path.setAttribute("d", d);
      path.setAttribute("class", `e ${cls}`);
      path.setAttribute("marker-end", "url(#arrow)");
      svg.appendChild(path);
    }
  }
}

/* --------------------------------------------------------------- timeline */

/**
 * Pick a round tick interval so the time axis has at most `maxTicks` labels.
 *
 * @param {number} span Axis length in milliseconds.
 * @param {number} maxTicks
 * @returns {number} Interval in milliseconds.
 */
function niceStep(span, maxTicks) {
  const candidates = [100, 250, 500, 1000, 2000, 5000, 10000, 15000, 30000, 60000];
  return candidates.find((c) => span / c <= maxTicks) || 60000;
}

/**
 * Draw the timeline: one row per participant, one bar per stretch of work.
 *
 * This view exists to make parallelism visible: overlapping bars are agents working at the
 * same time. The axis runs from the first event to now (live) or to the end (finished) and
 * is at least one second wide so a very fast run does not collapse into a sliver.
 */
function renderTimeline() {
  const box = $("timeline");
  const run = current;
  if (run.t0 === null || run.direct || !run.steps.size) {
    box.innerHTML = `<p class="tl-empty">${run.direct ? "Nothing ran in parallel — the Delivery Lead answered directly." : "Waiting for the first delegation…"}</p>`;
    return;
  }
  const now = nowOf(run);
  const span = Math.max(1000, now - run.t0);
  const pct = (t) => `${(Math.max(0, t - run.t0) / span) * 100}%`;
  const width = (a, b) => `${(Math.max(0, b - a) / span) * 100}%`;

  const lead = [];
  if (run.phases.planEnd != null) {
    lead.push(`<div class="tl-bar plan" style="left:0;width:${width(run.t0, run.phases.planEnd)}" title="Planning ${fmtMs(run.phases.planEnd - run.t0)}"></div>`);
  }
  if (run.phases.synthStart != null) {
    lead.push(`<div class="tl-bar synth ${run.done ? "" : "running"}" style="left:${pct(run.phases.synthStart)};width:${width(run.phases.synthStart, now)}" title="Writing the answer ${fmtMs(now - run.phases.synthStart)}"></div>`);
  }

  const rows = [`<div class="tl-row">
      <div class="tl-name"><span class="mono-badge" data-agent="orchestrator">DL</span><span>Delivery Lead</span></div>
      <div class="tl-track">${lead.join("")}</div></div>`];

  for (const s of run.steps.values()) {
    const start = s.startedAt ?? now;
    const end = s.endedAt ?? now;
    const running = s.status === "running" && !(run.status === "stopped");
    const bar = s.startedAt == null ? "" :
      `<div class="tl-bar ${running ? "running" : ""} ${s.status === "failed" ? "failed" : ""}" data-agent="${esc(s.agent)}"
         style="left:${pct(start)};width:${width(start, end)}"
         title="${esc(pretty(s.agent))} ${s.id}: ${fmtMs(end - start)}"></div>`;
    rows.push(`<div class="tl-row" data-agent="${esc(s.agent)}">
      <div class="tl-name"><span class="mono-badge">${esc(monogram(s.agent))}</span><span>${esc(pretty(s.agent))} <span class="muted">${esc(s.id)}</span></span></div>
      <div class="tl-track">${bar}</div></div>`);
  }

  // Fewer labels on a narrow screen, or they run into each other.
  const step = niceStep(span, box.clientWidth < 560 ? 3 : 6);
  const ticks = [];
  for (let ms = 0; ms <= span; ms += step) {
    ticks.push(`<span style="left:${(ms / span) * 100}%">${ms === 0 ? "0" : fmtMs(ms)}</span>`);
  }

  box.innerHTML = `<div class="tl">${rows.join("")}
    <div class="tl-axis"><span></span><div class="tl-ticks">${ticks.join("")}</div></div></div>
    <p class="tl-legend">Overlapping bars are agents working at the same time. Pale bars are the Delivery Lead planning and writing the answer.</p>`;
}

/* ------------------------------------------------------- head, answer, raw */

/**
 * Run status -> [css class, label] for the pill in the run header.
 */
const STATUS_PILL = {
  running: ["run", "Running"], succeeded: ["ok", "Succeeded"], partial: ["warn", "Partial"],
  failed: ["err", "Failed"], stopped: ["", "Stopped"], error: ["err", "Error"],
};

/**
 * Draw the run header: status pill, run id, the request and the stats.
 */
function renderHead() {
  const run = current;
  const [cls, label] = STATUS_PILL[run.status] || ["", run.status];
  const pill = $("run-status");
  pill.className = `pill ${cls}`;
  pill.textContent = label;
  $("run-id").textContent = run.id || "starting…";
  $("run-request").textContent = run.request;
  renderStats();
}

/**
 * Draw the stat tiles. Tokens and LLM calls read "-" until the result arrives, because only
 * the server's final accounting is trustworthy.
 */
function renderStats() {
  const run = current;
  const elapsed = run.durationMs ?? (run.t0 != null ? nowOf(run) - run.t0 : 0);
  const stats = [
    [fmtMs(elapsed) || "0 ms", "Duration"],
    [run.usage ? fmtNum(run.usage.total_tokens) : "—", "Tokens"],
    [run.usage ? fmtNum(run.usage.llm_calls) : "—", "LLM calls"],
    [fmtNum(run.toolCalls), "Tool calls"],
    [fmtNum(run.steps.size), "Delegations"],
  ];
  $("stats").innerHTML = stats
    .map(([v, l]) => `<div class="stat"><b>${esc(v)}</b><span>${esc(l)}</span></div>`).join("");
}

/**
 * Show a banner for outcomes that need explaining: an error, a partial run or a stopped one.
 */
function renderBanner() {
  const run = current;
  const banner = $("banner");
  let html = "";
  let kind = "err";
  if (run.status === "error" || (run.status === "failed" && run.error)) {
    html = esc(run.error || "The run failed.");
  } else if (run.status === "partial") {
    kind = "warn";
    html = "Some specialists failed or were skipped. The answer below is built from the ones that finished.";
  } else if (run.status === "stopped") {
    kind = "info";
    html = "Stopped. The server cancelled the run when the connection closed, so no further tokens are being spent.";
  }
  banner.hidden = !html;
  banner.className = `banner ${kind}`;
  banner.innerHTML = html;
}

/**
 * Draw the answer card: the rendered answer, a "waiting" placeholder, or an empty note.
 */
function renderAnswer() {
  const run = current;
  const box = $("answer");
  $("copy").hidden = !run.answer;
  if (run.answer) {
    box.innerHTML = renderMarkdown(run.answer);
    return;
  }
  if (run.done) {
    box.innerHTML = '<p class="muted">No answer was produced.</p>';
    return;
  }
  const writing = run.phases.synthStart != null;
  box.innerHTML = `<div class="placeholder"><span class="spin"></span>${
    writing ? "The Delivery Lead is writing the answer…" : "Waiting for the specialists…"}</div>`;
}

/**
 * Draw the raw trace log, one line per event, keeping it scrolled to the bottom while the
 * reader has not scrolled up.
 */
function renderRaw() {
  const run = current;
  $("raw-count").textContent = `(${run.events.length} events)`;
  const log = $("raw-log");
  // Only follow the tail if the reader is already at the bottom; do not yank them back down.
  const stick = log.scrollTop + log.clientHeight >= log.scrollHeight - 24;
  log.textContent = run.events.map(({ ev, t }) => {
    const rel = `+${((t - run.t0) / 1000).toFixed(2)}s`.padEnd(9);
    const who = [ev.agent, ev.step_id].filter(Boolean).join("/");
    return `${rel}${String(ev.seq).padStart(3)}  ${ev.type.padEnd(18)} ${who.padEnd(24)} ${clip(ev.message || "", 90)}`;
  }).join("\n");
  if (stick) log.scrollTop = log.scrollHeight;
}

// The run being displayed: either the live one or a replay. Null before anything has run.
let current = null;
let frame = 0;

/**
 * Redraw the whole page from the current run model. Cheap enough to call on every event.
 */
function render() {
  if (!current) return;
  $("empty").hidden = true;
  $("run-view").hidden = false;
  renderHead();
  renderBanner();
  renderPlan();
  renderTimeline();
  renderAnswer();
  renderRaw();
}
/**
 * Request a redraw, coalescing a burst of events into one paint per animation frame.
 */
function schedule() {
  if (!frame) frame = requestAnimationFrame(() => { frame = 0; render(); });
}

// Live clocks: only text and bar widths change, so open <details> are not disturbed.
setInterval(() => {
  if (!current || current.done || !current.live) return;
  document.querySelectorAll(".elapsed").forEach((el) => {
    const s = current.steps.get(el.dataset.step);
    if (s) el.textContent = ` ${stepElapsed(current, s)}`;
  });
  renderStats();
  renderTimeline();
}, 250);

document.addEventListener("toggle", (e) => {
  const el = e.target;
  if (!(el instanceof HTMLDetailsElement) || !el.dataset.key) return;
  if (el.open) openState.add(el.dataset.key); else openState.delete(el.dataset.key);
}, true);

new ResizeObserver(() => drawEdges()).observe(document.body);

/* ----------------------------------------------------------------- network */

/**
 * An error response from the API (not a network failure), carrying the server's message.
 *
 * `code` is the server's stable error code, e.g. "missing_credentials".
 */
class ApiError extends Error {
  constructor(status, body) {
    super(body?.error || body?.detail?.[0]?.msg || `HTTP ${status}`);
    this.status = status;
    this.code = body?.code;
  }
}

/**
 * POST a request to the streaming endpoint and call `onEvent` for each server-sent event.
 *
 * EventSource cannot POST or set headers, so this reads the response body directly and
 * parses the SSE framing itself: events are separated by a blank line, and a network
 * chunk may end mid-event, so incomplete data is buffered until the rest arrives.
 *
 * @param {string} request The question to ask.
 * @param {string} key Optional OpenAI key, sent as X-OpenAI-Key.
 * @param {AbortSignal} signal Aborting closes the stream, which makes the server cancel
 *     the run.
 * @param {(event: string, data: object) => void} onEvent Called as "run", "trace", "result".
 * @throws {ApiError} The server rejected the request before streaming began.
 */
async function streamRun(request, key, signal, onEvent) {
  const headers = { "content-type": "application/json" };
  if (key) headers["X-OpenAI-Key"] = key;
  const res = await fetch("/v1/requests/stream", {
    method: "POST", headers, signal, body: JSON.stringify({ request }),
  });
  if (!res.ok) {
    let body = null;
    try { body = await res.json(); } catch { /* not JSON */ }
    throw new ApiError(res.status, body);
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
    // SSE events end with a blank line. A chunk can end anywhere, so keep the unfinished tail in
    // `buffer` until the rest of it arrives.
    let cut;
    while ((cut = buffer.indexOf("\n\n")) >= 0) {
      const frameText = buffer.slice(0, cut);
      buffer = buffer.slice(cut + 2);
      let event = "message";
      const data = [];
      for (const line of frameText.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data.push(line.slice(5).trimStart());
      }
      if (data.length) onEvent(event, JSON.parse(data.join("\n")));
    }
  }
}

// AbortController of the run in flight; non-null means a run is in progress.
let controller = null;

/**
 * Toggle the composer between idle and a run in flight (Run vs Stop).
 */
function setBusy(busy) {
  $("run").disabled = busy;
  $("run").textContent = busy ? "Running…" : "Run";
  $("stop").hidden = !busy;
  $("request").readOnly = busy;
}

/**
 * Start a live run and drive the page from its stream until it ends.
 *
 * Handles every way a run can end: finished, stopped by the user (abort), rejected by the
 * server (for example no key), or interrupted. Whatever happens, it always clears the busy
 * state and refreshes the history.
 *
 * @param {string} text The request.
 */
async function start(text) {
  if (controller || text.trim().length < 3) return;
  controller = new AbortController();
  setBusy(true);
  const run = newRun({ request: text.trim(), live: true });
  current = run;
  history.replaceState(null, "", location.pathname);
  highlightHistory(null);
  render();

  try {
    await streamRun(run.request, $("apikey").value.trim(), controller.signal, (event, data) => {
      if (event === "run") run.id = data.run_id;
      else if (event === "trace") applyEvent(run, data);
      else if (event === "result") applyResult(run, data);
      schedule();
    });
    if (!run.done) {
      run.status = "error";
      run.error = "The stream ended before the run produced a result.";
    }
  } catch (err) {
    if (err.name === "AbortError") {
      run.status = "stopped";
    } else {
      run.status = "error";
      run.error = err instanceof ApiError ? err.message : `Could not reach the server: ${err.message}`;
      if (err instanceof ApiError && err.code === "missing_credentials") {
        $("keyrow").hidden = false;
        run.error += " Enter a key in the panel on the left and run it again.";
      }
    }
  } finally {
    controller = null;
    run.done = true;
    if (run.tEnd === null) run.tEnd = Date.now();
    setBusy(false);
    render();
    refreshHistory(run.id);
  }
}

/**
 * Load a finished run from the server and show it exactly as it appeared live.
 *
 * Feeds the stored trace through the same reducer as a live stream.
 *
 * @param {string} id Run id.
 */
async function replay(id) {
  if (controller) return;
  const res = await fetch(`/v1/runs/${encodeURIComponent(id)}?include_trace=true`);
  if (!res.ok) return;
  const body = await res.json();
  const run = newRun({ id: body.run_id, request: body.request, live: false });
  for (const ev of body.trace || []) applyEvent(run, ev);
  applyResult(run, body);
  current = run;
  history.replaceState(null, "", `?run=${encodeURIComponent(id)}`);
  highlightHistory(id);
  render();
  window.scrollTo({ top: 0, behavior: "smooth" });
}

/* ------------------------------------------------------- sidebar and boot */

/**
 * Mark the history entry for run `id` as current (or none).
 */
function highlightHistory(id) {
  document.querySelectorAll(".hist").forEach((b) => {
    if (b.dataset.id === id) b.setAttribute("aria-current", "true");
    else b.removeAttribute("aria-current");
  });
}

/**
 * Reload the "Recent runs" list.
 *
 * The API lists ids only, so this fetches each run (without its trace) to show its request
 * and outcome. That is one request per entry, acceptable for a dozen in-memory runs but the
 * first thing to replace with a summary endpoint if history ever grows.
 *
 * @param {string|null} activeId Run to highlight.
 */
async function refreshHistory(activeId = null) {
  try {
    const ids = await (await fetch("/v1/runs?limit=12")).json();
    const runs = await Promise.all(ids.map(async (id) => {
      const r = await fetch(`/v1/runs/${encodeURIComponent(id)}?include_trace=false`);
      return r.ok ? r.json() : null;
    }));
    const list = $("history");
    const items = runs.filter(Boolean);
    if (!items.length) { list.innerHTML = '<li class="muted">No runs yet.</li>'; return; }
    list.innerHTML = items.map((r) => `<li><button class="hist" type="button" data-id="${esc(r.run_id)}">
      <span class="dot ${esc(r.status)}" title="${esc(r.status)}"></span>
      <span class="q">${esc(r.request)}</span>
      <span class="m">${esc(r.status)} · ${esc(fmtMs(r.duration_ms))} · ${esc(fmtNum(r.usage.total_tokens))} tokens</span>
    </button></li>`).join("");
    highlightHistory(activeId || (current && !current.live ? current.id : null));
  } catch { /* history is a convenience; the console works without it */ }
}

/**
 * Load the agent roster for the sidebar, and remember each agent's title and tools.
 */
async function loadRoster() {
  try {
    const data = await (await fetch("/v1/agents")).json();
    const lead = data.orchestrator;
    const members = [`<div class="member" data-agent="orchestrator">
        <span class="mono-badge">DL</span>
        <div><h3>Delivery Lead</h3><div class="title">Orchestrator · ${esc(lead.model)}</div>
          <details><summary>Responsibilities</summary>
            <ul class="mandate">${lead.responsibilities.map((r) => `<li>${esc(r)}</li>`).join("")}</ul>
          </details></div></div>`];
    for (const a of data.agents) {
      roster[a.name] = a;
      members.push(`<div class="member" data-agent="${esc(a.name)}">
        <span class="mono-badge">${esc(monogram(a.name))}</span>
        <div><h3>${esc(pretty(a.name))}</h3><div class="title">${esc(a.title)}</div>
          <details><summary>Mandate &amp; ${a.tools.length} tools</summary>
            <p class="mandate">${esc(a.mandate)}</p>
            <div class="chips">${a.tools.map((t) => `<span class="chip">${esc(t)}</span>`).join("")}</div>
          </details></div></div>`);
    }
    $("roster").innerHTML = members.join("");
  } catch {
    $("roster").innerHTML = '<p class="muted">Could not load the roster.</p>';
  }
}

/**
 * Show whether the server is up and whether it has its own OpenAI key. When it does not,
 * reveal the key field so the user can supply one.
 */
async function checkReady() {
  const pill = $("ready");
  try {
    const r = await (await fetch("/readyz")).json();
    const own = r.llm_credentials === "server-key";
    pill.className = "pill ok";
    pill.textContent = own ? "● ready · server key" : "● ready · bring your own key";
    $("keyrow").hidden = own;
  } catch {
    pill.className = "pill err";
    pill.textContent = "● server unreachable";
  }
}

/**
 * Wire up the page: examples, form, buttons, then load the roster and history.
 */
function boot() {
  $("examples").innerHTML = EXAMPLES.map((e, i) =>
    `<button class="ex" type="button" data-i="${i}"><b>${esc(e.tag)}</b><span>${esc(clip(e.text, 96))}</span></button>`).join("");
  $("examples").addEventListener("click", (e) => {
    const btn = e.target.closest(".ex");
    if (!btn) return;
    $("request").value = EXAMPLES[Number(btn.dataset.i)].text;
    $("request").focus();
  });

  $("form").addEventListener("submit", (e) => { e.preventDefault(); start($("request").value); });
  $("request").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) { e.preventDefault(); start($("request").value); }
  });
  $("stop").addEventListener("click", () => controller?.abort());
  $("history").addEventListener("click", (e) => {
    const btn = e.target.closest(".hist");
    if (btn) replay(btn.dataset.id);
  });
  $("copy").addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(current?.answer || "");
      $("copy").textContent = "Copied";
      setTimeout(() => { $("copy").textContent = "Copy"; }, 1500);
    } catch { /* clipboard unavailable */ }
  });

  checkReady();
  setInterval(checkReady, 20000);
  loadRoster().then(() => {
    const id = new URLSearchParams(location.search).get("run");
    if (id) replay(id);
  });
  refreshHistory();
}

boot();
