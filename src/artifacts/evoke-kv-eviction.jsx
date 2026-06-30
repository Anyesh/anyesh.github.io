import { useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";

export const meta = {
  title: "EVOKE: Evicting and Recovering the KV Cache",
  category: "LLM Systems",
  description:
    "An agent session outgrows its GPU memory budget fast. Watch every cache block earn a relevance score from how well it still matches the task in focus, see the lowest scorers evicted to host RAM, then spliced back by content identity when the history is re-sent, so almost nothing is recomputed.",
  date: "2026-06-11",
  tags: ["kv-cache", "llm-inference", "eviction", "agents", "memory-hierarchy"],
};

const C = {
  bg: "#f7f5f2",
  card: "#ffffff",
  border: "#e4dfd8",
  ink: "#1c1a17",
  muted: "#857c72",
  accent: "#c0561f",
  accentSoft: "#f6ece5",
  faint: "#efeae3",
  match: "#5a7d3c",
  matchSoft: "#eef3e6",
  splice: "#2e6f8e",
  spliceSoft: "#e8f0f5",
  decode: "#c0561f",
  decodeSoft: "#f9e9df",
  gen: "#9a5ba6",
  genSoft: "#f3ecf6",
  danger: "#a8453f",
};

const KIND_COLORS = {
  sys: "#3f5f9e",
  tools: "#2e6f8e",
  user: "#b8902a",
  tool: "#5a7d3c",
  assist: "#9a5ba6",
};

const EASE = "cubic-bezier(0.23, 1, 0.32, 1)";
const TOKENS_PER_BLOCK = 128;

function usePrefersReducedMotion() {
  const [reduce, setReduce] = useState(false);
  useEffect(() => {
    if (!window.matchMedia) return;
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    setReduce(mq.matches);
    const onChange = (e) => setReduce(e.matches);
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, []);
  return reduce;
}

// The scripted agent session: a coding agent builds a small webapp. Each turn
// brings new prompt blocks (tool results, user messages) and generates one
// assistant block. Mirrors the live demo run shape (system prompt + tools,
// then tool traffic). `focus` is the share of the turn's task relevance that
// landed in each block: in the headline config a block's relevance is how
// closely its embedding still matches the running task-focus vector (the
// coherence signal, the largest weight in the scorer). The values are scripted
// so the run is deterministic and scrubbable.
const SESSION = [
  {
    label: "Turn 1: the task arrives",
    newBlocks: [
      { id: "sys", kind: "sys", label: "system", pinned: true },
      { id: "tools", kind: "tools", label: "tools" },
      { id: "user1", kind: "user", label: "user: build it" },
    ],
    gen: { id: "a1", kind: "assist", label: "assistant ① write app.py" },
    focus: { user1: 0.6, tools: 0.25, sys: 0.15 },
  },
  {
    label: "Turn 2: tool result comes back",
    newBlocks: [
      { id: "t1", kind: "tool", label: "tool: wrote app.py" },
      { id: "f1", kind: "tool", label: "file: app.py" },
    ],
    gen: { id: "a2", kind: "assist", label: "assistant ② write README" },
    focus: { f1: 0.45, user1: 0.2, a1: 0.2, tools: 0.15 },
  },
  {
    label: "Turn 3: another result",
    newBlocks: [{ id: "t2", kind: "tool", label: "tool: wrote README" }],
    gen: { id: "a3", kind: "assist", label: "assistant ③ run check" },
    focus: { f1: 0.3, t2: 0.25, tools: 0.2, a2: 0.15, user1: 0.1 },
  },
  {
    label: "Turn 4: the check passes",
    newBlocks: [{ id: "t3", kind: "tool", label: "tool: check ok" }],
    gen: { id: "a4", kind: "assist", label: "assistant ④ done" },
    focus: { t3: 0.5, user1: 0.3, a3: 0.2 },
  },
];

// Scoring constants mirror the headline EvokeConfig (the evoke_kv_restore
// benchmark arm): task coherence is the dominant weight at 0.6, recency at 0.4,
// attention off by default. User turns floor at 0.6 and assistant turns at 0.5
// (conversation backbone outlives tool output), and recency decays
// exponentially.
const SCORE_W_COHERENCE = 0.6;
const SCORE_W_RECENCY = 0.4;
const RECENCY_DECAY = 0.45;
const FOCUS_EWMA = 0.5;
const FLOOR_USER = 0.6;
const FLOOR_ASSIST = 0.5;

// Pure simulator. Returns a list of steps; each step is a full snapshot the
// UI can scrub to. Block states: resident | archived | gone. A freshly
// generated assistant block is a raw emit; the client's next resend contains
// a re-templated echo of it that never byte-matches, so it re-decodes once
// and becomes canonical from then on (this is the echo-drift cost the real
// system pays).
function simulate(policy, budget) {
  const blocks = new Map();
  let order = [];
  let decoded = 0;
  let generated = 0;
  let evictions = 0;
  let recoveries = 0;
  let mismatches = 0;
  let peak = 0;
  const steps = [];
  const focusEwma = new Map();

  const resident = () => order.filter((id) => blocks.get(id).state === "resident");

  // Per-turn focus folds into a running average (the real scorer keeps a
  // decayed sliding window so the task-focus vector drifts rather than jumps;
  // one turn here stands in for that window) so a block that stops matching the
  // task fades over a couple of turns instead of dropping off a cliff.
  function absorbFocus(focus) {
    for (const id of order) {
      const prev = focusEwma.get(id) || 0;
      focusEwma.set(id, prev * FOCUS_EWMA + (focus[id] || 0) * (1 - FOCUS_EWMA));
    }
  }

  function rescore() {
    const res = resident();
    res.forEach((id, i) => {
      const b = blocks.get(id);
      if (b.pinned) {
        b.score = 1;
        return;
      }
      const coherence = focusEwma.get(id) || 0;
      const recency = Math.exp(-RECENCY_DECAY * (res.length - 1 - i));
      let raw = SCORE_W_COHERENCE * coherence + SCORE_W_RECENCY * recency;
      if (b.kind === "user") raw = Math.max(raw, FLOOR_USER);
      if (b.kind === "assist") raw = Math.max(raw, FLOOR_ASSIST);
      b.score = Math.min(raw, 1);
    });
  }

  function snapshot(phase, turn, note, highlightId, flash) {
    peak = Math.max(peak, resident().length);
    steps.push({
      phase,
      turn,
      note,
      highlightId: highlightId || null,
      flash: flash || null,
      order: [...order],
      blocks: new Map([...blocks].map(([k, v]) => [k, { ...v }])),
      counters: {
        decoded,
        generated,
        evictions,
        recoveries,
        mismatches,
        peak,
        resident: resident().length,
      },
    });
  }

  SESSION.forEach((turn, ti) => {
    const turnNo = ti + 1;

    // Phase 1: the client re-sends the whole conversation plus new content.
    for (const nb of turn.newBlocks) {
      if (!blocks.has(nb.id)) {
        blocks.set(nb.id, { ...nb, state: "incoming", rawEmit: false });
        order.push(nb.id);
      }
    }
    snapshot(
      "resend",
      turnNo,
      ti === 0
        ? `${turn.label}. The full prompt arrives; nothing is cached yet.`
        : `${turn.label}. The client re-sends the entire conversation plus the new content. The server walks it block by block against the cache.`,
      null
    );

    // Phase 2: the walk. Resident blocks match for free, archived blocks
    // splice back by identity, everything else decodes. Fresh splices are
    // remembered because the turn-end pass must not re-evict a block the
    // agent just brought back (the real system's recovery guard).
    const splicedThisTurn = new Set();
    for (const id of order) {
      const b = blocks.get(id);
      if (b.state === "resident") {
        if (b.rawEmit) {
          decoded += 1;
          b.rawEmit = false;
          snapshot(
            "walk",
            turnNo,
            `${b.label}: the re-templated echo never byte-matches the raw emit in cache, so this block re-decodes once (echo drift), then it is canonical.`,
            id,
            "decode"
          );
        } else {
          snapshot("walk", turnNo, `${b.label}: already resident, matches for free.`, id, "match");
        }
      } else if (b.state === "archived") {
        if (b.rawEmit) {
          mismatches += 1;
          decoded += 1;
          b.state = "resident";
          b.rawEmit = false;
          snapshot(
            "walk",
            turnNo,
            `${b.label}: archived as a raw emit, but the resend carries the re-templated echo. Identity mismatch, so it re-decodes.`,
            id,
            "decode"
          );
        } else {
          recoveries += 1;
          b.state = "resident";
          splicedThisTurn.add(id);
          snapshot(
            "walk",
            turnNo,
            `${b.label}: found in host RAM with identical content at the same position. The saved K/V splices straight back in. Zero tokens recomputed.`,
            id,
            "splice"
          );
        }
      } else {
        const wasGone = b.state === "gone";
        decoded += 1;
        b.state = "resident";
        b.rawEmit = false;
        snapshot(
          "walk",
          turnNo,
          wasGone
            ? `${b.label}: the saved copy was discarded at eviction, so the whole block re-decodes from scratch.`
            : `${b.label}: new content, decoded for the first time.`,
          id,
          "decode"
        );
      }
    }

    // Phase 3: generation appends an assistant block (a raw emit).
    blocks.set(turn.gen.id, { ...turn.gen, state: "resident", rawEmit: true });
    order.push(turn.gen.id);
    generated += 1;
    snapshot(
      "generate",
      turnNo,
      `The model generates: ${turn.gen.label}. Generation cost is the same under every policy; the contest is over prompt re-decoding.`,
      turn.gen.id,
      "gen"
    );

    // Phase 4: rescore. The manager refreshes every block's coherence to the
    // running task focus before the budget pass, so eviction acts on what the
    // session is working on now rather than on block age.
    absorbFocus(turn.focus || {});
    rescore();
    const ranked = Object.entries(turn.focus || {}).sort((a, b) => b[1] - a[1]);
    const topNames = ranked
      .slice(0, 2)
      .map(([id, share]) => `${blocks.get(id).label} (${Math.round(share * 100)}%)`)
      .join(" and ");
    snapshot(
      "score",
      turnNo,
      policy === "none"
        ? `This turn's task focus tracked mostly to ${topNames}. Every block rescores (bars below each chip), but with no eviction policy nothing acts on the scores.`
        : `This turn's task focus tracked mostly to ${topNames}. Every block rescores: coherence to the current task, blended with recency, floors holding up user and assistant turns. The lowest scorers are now the eviction candidates.`,
      ranked.length ? ranked[0][0] : null,
      "attend"
    );

    // Phase 5: turn-end eviction enforces the budget, lowest relevance first.
    if (policy !== "none") {
      const protectedIds = new Set([
        ...turn.newBlocks.map((b) => b.id),
        turn.gen.id,
        ...splicedThisTurn,
      ]);
      // The victim set is picked in one pass before anything is marked, the
      // way the real manager collects to_evict and only then evicts, so the
      // age note can name true survivors of the whole pass.
      const candidates = resident()
        .map((id) => blocks.get(id))
        .filter((b) => !b.pinned && b.score < 1 && !protectedIds.has(b.id))
        .sort((a, b) => a.score - b.score);
      const needed = Math.max(0, resident().length - budget);
      const victims = candidates.slice(0, needed);
      const victimIds = new Set(victims.map((b) => b.id));
      for (const v of victims) {
        const olderSurvives = resident().some(
          (id) =>
            !victimIds.has(id) &&
            !blocks.get(id).pinned &&
            !protectedIds.has(id) &&
            order.indexOf(id) < order.indexOf(v.id)
        );
        const ageNote =
          olderSurvives && v === victims[0]
            ? " An older block outscores it and stays; the victim is the lowest relevance, not the oldest block."
            : "";
        v.state = policy === "kv_restore" ? "archived" : "gone";
        evictions += 1;
        snapshot(
          "evict",
          turnNo,
          policy === "kv_restore"
            ? `${v.label} scores ${v.score.toFixed(2)}, the lowest in cache. Evicted; its K/V tensors are saved to host RAM, keyed by content.${ageNote}`
            : `${v.label} scores ${v.score.toFixed(2)}, the lowest in cache. Evicted and discarded; nothing is saved.${ageNote}`,
          v.id,
          "evict"
        );
      }
      if (!victims.length) {
        snapshot(
          "evict",
          turnNo,
          resident().length <= budget
            ? `Turn ends within budget (${resident().length}/${budget} blocks resident). Nothing to evict.`
            : `Everything resident is pinned, freshly recovered, or from this turn; eviction waits.`,
          null
        );
      }
    } else {
      snapshot(
        "evict",
        turnNo,
        `No eviction policy: the turn ends and the cache keeps growing (${resident().length} blocks resident).`,
        null
      );
    }
  });

  snapshot(
    "done",
    SESSION.length,
    `Session over. ${decoded * TOKENS_PER_BLOCK} prompt tokens decoded, ${recoveries} blocks spliced back recompute-free, peak ${peak} blocks resident.`,
    null
  );
  return steps;
}

function finalCounters(policy, budget) {
  const steps = simulate(policy, budget);
  return steps[steps.length - 1].counters;
}

function Card({ children, style }) {
  return (
    <div
      style={{
        background: C.card,
        border: `1px solid ${C.border}`,
        borderRadius: 14,
        padding: 20,
        ...style,
      }}
    >
      {children}
    </div>
  );
}

function Btn({ children, onClick, disabled, primary, ariaLabel }) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      aria-label={ariaLabel}
      style={{
        padding: primary ? "8px 18px" : "7px 14px",
        borderRadius: 8,
        border: primary ? "none" : `1px solid ${C.border}`,
        background: primary ? C.accent : "transparent",
        color: primary ? "#fff" : C.ink,
        fontSize: 13,
        fontWeight: primary ? 600 : 500,
        fontFamily: "inherit",
        cursor: disabled ? "not-allowed" : "pointer",
        opacity: disabled ? 0.4 : 1,
        transition: `transform 140ms ${EASE}, background 160ms ease`,
      }}
      onPointerDown={(e) => {
        if (!disabled) e.currentTarget.style.transform = "scale(0.97)";
      }}
      onPointerUp={(e) => {
        e.currentTarget.style.transform = "scale(1)";
      }}
      onPointerLeave={(e) => {
        e.currentTarget.style.transform = "scale(1)";
      }}
    >
      {children}
    </button>
  );
}

function Slider({ label, value, min, max, onChange, id }) {
  return (
    <div>
      <label
        htmlFor={id}
        style={{
          display: "flex",
          justifyContent: "space-between",
          fontSize: 12,
          color: C.muted,
          marginBottom: 6,
        }}
      >
        <span>{label}</span>
        <span style={{ fontWeight: 700, color: C.accent }}>{value} blocks</span>
      </label>
      <input
        id={id}
        type="range"
        min={min}
        max={max}
        value={value}
        onChange={(e) => onChange(+e.target.value)}
        style={{ width: "100%", accentColor: C.accent }}
      />
    </div>
  );
}

function Stat({ label, value, tone }) {
  return (
    <div
      style={{
        flex: "1 1 90px",
        background: C.faint,
        borderRadius: 10,
        padding: "10px 12px",
        textAlign: "center",
      }}
    >
      <div style={{ fontSize: 19, fontWeight: 700, color: tone || C.ink }}>{value}</div>
      <div style={{ fontSize: 10.5, color: C.muted, marginTop: 2 }}>{label}</div>
    </div>
  );
}

function flashStyle(flash, reduce) {
  const base = { transition: reduce ? "none" : `all 360ms ${EASE}` };
  switch (flash) {
    case "match":
      return { ...base, boxShadow: `0 0 0 3px ${C.match}55`, background: C.matchSoft };
    case "splice":
      return { ...base, boxShadow: `0 0 0 3px ${C.splice}66`, background: C.spliceSoft };
    case "decode":
      return { ...base, boxShadow: `0 0 0 3px ${C.decode}66`, background: C.decodeSoft };
    case "gen":
      return { ...base, boxShadow: `0 0 0 3px ${C.gen}55`, background: C.genSoft };
    case "evict":
      return { ...base, boxShadow: `0 0 0 3px ${C.danger}55`, opacity: 0.7 };
    case "attend":
      return { ...base, boxShadow: `0 0 0 3px ${C.accent}66`, background: C.accentSoft };
    default:
      return base;
  }
}

function scoreColor(score) {
  if (score >= 0.6) return C.match;
  if (score >= 0.3) return "#b8902a";
  return C.danger;
}

// One frozen decode step for the bridge diagram: a softmax row over nine
// cached tokens, grouped into the three blocks they belong to. Percentages
// sum to 100 so the "row sums to 1" claim is checkable by eye.
const ROW_DEMO = [
  {
    block: "system",
    sum: 6,
    color: KIND_COLORS.sys,
    tint: "#eef1f7",
    tokens: [
      ["You", 2],
      ["are", 1],
      ["an agent", 3],
    ],
  },
  {
    block: "file: app.py",
    sum: 54,
    color: KIND_COLORS.tool,
    tint: C.matchSoft,
    tokens: [
      ["def", 41],
      ["main", 8],
      ["():", 5],
    ],
  },
  {
    block: "tool log",
    sum: 40,
    color: KIND_COLORS.tools,
    tint: C.spliceSoft,
    tokens: [
      ["wrote", 30],
      ["app", 6],
      [".py", 4],
    ],
  },
];

function SoftmaxRowDiagram() {
  const maxSum = Math.max(...ROW_DEMO.map((g) => g.sum));
  return (
    <div
      role="img"
      aria-label="One decode step's softmax row over nine cached tokens, grouped into three blocks: system collects 6 percent, file app.py collects 54 percent, tool log collects 40 percent."
      style={{ display: "flex", gap: 14, flexWrap: "wrap", margin: "14px 0" }}
    >
      {ROW_DEMO.map((g) => (
        <div key={g.block} style={{ flex: "1 1 180px" }}>
          <div style={{ display: "flex", gap: 4 }}>
            {g.tokens.map(([tok, pct]) => (
              <div
                key={tok}
                style={{
                  flex: 1,
                  textAlign: "center",
                  border: `1.5px solid ${g.color}`,
                  borderRadius: 8,
                  background: g.tint,
                  padding: "6px 2px",
                }}
              >
                <div style={{ fontSize: 11, fontWeight: 600, color: C.ink }}>{tok}</div>
                <div style={{ fontSize: 10.5, color: C.muted, marginTop: 2 }}>{pct}%</div>
              </div>
            ))}
          </div>
          <div
            style={{
              borderLeft: `1.5px solid ${g.color}88`,
              borderRight: `1.5px solid ${g.color}88`,
              borderBottom: `1.5px solid ${g.color}88`,
              borderRadius: "0 0 6px 6px",
              height: 5,
              margin: "3px 4px 0",
            }}
          />
          <div
            style={{
              textAlign: "center",
              fontSize: 11.5,
              marginTop: 4,
              color: g.sum === maxSum ? C.match : C.muted,
              fontWeight: g.sum === maxSum ? 700 : 500,
            }}
          >
            {g.block} → {g.sum}%
          </div>
        </div>
      ))}
    </div>
  );
}

function CrossLink({ to, children }) {
  return (
    <Link
      to={to}
      style={{ color: C.accent, textDecorationColor: `${C.accent}66`, fontWeight: 600 }}
    >
      {children}
    </Link>
  );
}

function BlockChip({ block, highlighted, flash, reduce, shelf }) {
  const kc = KIND_COLORS[block.kind] || C.muted;
  const hasScore = typeof block.score === "number";
  return (
    <div
      role="img"
      aria-label={`${block.label}, ${block.state}${block.pinned ? ", pinned" : ""}${block.rawEmit ? ", raw emit" : ""}${hasScore ? `, relevance ${block.score.toFixed(2)}` : ""}`}
      style={{
        minWidth: 74,
        maxWidth: 118,
        padding: "7px 9px",
        borderRadius: 9,
        border: shelf ? `1.5px dashed ${kc}88` : `1.5px solid ${kc}`,
        background: shelf ? "transparent" : "#fff",
        opacity: shelf ? 0.85 : 1,
        fontSize: 10.5,
        lineHeight: 1.35,
        position: "relative",
        ...(highlighted ? flashStyle(flash, reduce) : { transition: reduce ? "none" : `all 360ms ${EASE}` }),
      }}
    >
      <div
        style={{
          width: 8,
          height: 8,
          borderRadius: 3,
          background: kc,
          display: "inline-block",
          marginRight: 5,
          verticalAlign: -1,
        }}
      />
      <span style={{ fontWeight: 600, color: C.ink }}>{block.label}</span>
      <div style={{ color: C.muted, marginTop: 2 }}>
        {TOKENS_PER_BLOCK} tok
        {block.pinned ? " · pinned" : ""}
        {block.rawEmit ? " · raw" : ""}
        {hasScore && !shelf ? ` · ${block.score.toFixed(2)}` : ""}
      </div>
      {hasScore && !shelf && (
        <div
          aria-hidden="true"
          style={{
            marginTop: 4,
            height: 3,
            borderRadius: 2,
            background: C.faint,
            overflow: "hidden",
          }}
        >
          <div
            style={{
              height: "100%",
              width: `${Math.round(block.score * 100)}%`,
              background: scoreColor(block.score),
              transition: reduce ? "none" : `width 360ms ${EASE}`,
            }}
          />
        </div>
      )}
    </div>
  );
}

const POLICIES = [
  { id: "kv_restore", label: "EVOKE (kv_restore)" },
  { id: "discard", label: "Evict, no recovery" },
  { id: "none", label: "No eviction" },
];

const PHASE_LABELS = {
  resend: "client re-sends",
  walk: "prefix walk",
  generate: "generate",
  score: "relevance rescore",
  evict: "turn-end eviction",
  done: "session over",
};

export default function App() {
  const reduce = usePrefersReducedMotion();
  const [policy, setPolicy] = useState("kv_restore");
  const [budget, setBudget] = useState(6);
  const [stepIdx, setStepIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const timerRef = useRef(null);

  const steps = useMemo(() => simulate(policy, budget), [policy, budget]);
  const step = steps[Math.min(stepIdx, steps.length - 1)];
  const atEnd = stepIdx >= steps.length - 1;

  useEffect(() => {
    setStepIdx(0);
    setPlaying(false);
  }, [policy, budget]);

  useEffect(() => {
    if (!playing) return;
    timerRef.current = setInterval(
      () => {
        setStepIdx((i) => {
          if (i >= steps.length - 1) {
            setPlaying(false);
            return i;
          }
          return i + 1;
        });
      },
      reduce ? 900 : 700
    );
    return () => clearInterval(timerRef.current);
  }, [playing, steps.length, reduce]);

  const residentBlocks = step.order
    .map((id) => step.blocks.get(id))
    .filter((b) => b.state === "resident" || b.state === "incoming");
  const archivedBlocks = step.order
    .map((id) => step.blocks.get(id))
    .filter((b) => b.state === "archived");
  const goneCount = step.order.filter((id) => step.blocks.get(id).state === "gone").length;
  const overBudget = residentBlocks.length > budget;

  const comparison = useMemo(
    () => POLICIES.map((p) => ({ ...p, final: finalCounters(p.id, budget) })),
    [budget]
  );
  const maxDecoded = Math.max(...comparison.map((c) => c.final.decoded));
  const maxResident = Math.max(...comparison.map((c) => c.final.resident));

  return (
    <div style={{ background: C.bg, minHeight: "100vh", color: C.ink, fontFamily: "system-ui, -apple-system, sans-serif" }}>
      <div style={{ maxWidth: 860, margin: "0 auto", padding: "28px 16px 60px" }}>
        <div style={{ marginBottom: 18 }}>
          <div
            style={{
              fontSize: 11,
              fontWeight: 700,
              letterSpacing: 1.2,
              textTransform: "uppercase",
              color: C.accent,
              marginBottom: 5,
            }}
          >
            LLM Systems · KV Cache
          </div>
          <h1 style={{ fontSize: 25, fontWeight: 700, margin: 0 }}>
            EVOKE: Evicting and Recovering the KV Cache
          </h1>
          <p style={{ color: C.muted, fontSize: 13.5, margin: "7px 0 0", lineHeight: 1.6, maxWidth: "64ch" }}>
            An agent re-sends its whole conversation every turn, but the GPU can only hold so much
            attention state. EVOKE treats the KV cache like an OS treats memory. Every block
            carries a relevance score: how well it still matches the task in focus, blended with
            recency, with floors under conversation turns. At turn ends the lowest scorers are
            evicted to host RAM, and when the same bytes come back, the saved tensors splice in
            instead of being recomputed.
          </p>
        </div>

        <Card style={{ marginBottom: 16 }}>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 14 }}>
            {POLICIES.map((p) => (
              <button
                key={p.id}
                onClick={() => setPolicy(p.id)}
                aria-pressed={policy === p.id}
                style={{
                  padding: "7px 14px",
                  borderRadius: 999,
                  border: `1.5px solid ${policy === p.id ? C.accent : C.border}`,
                  background: policy === p.id ? C.accentSoft : "transparent",
                  color: policy === p.id ? C.accent : C.muted,
                  fontWeight: policy === p.id ? 700 : 500,
                  fontSize: 12.5,
                  cursor: "pointer",
                  fontFamily: "inherit",
                }}
              >
                {p.label}
              </button>
            ))}
          </div>

          <div style={{ marginBottom: 16, maxWidth: 320 }}>
            <Slider id="budget" label="GPU budget" value={budget} min={3} max={12} onChange={setBudget} />
          </div>

          <div style={{ fontSize: 11, fontWeight: 700, color: C.muted, textTransform: "uppercase", letterSpacing: 0.8, marginBottom: 6 }}>
            GPU KV cache
            <span style={{ fontWeight: 500, textTransform: "none", letterSpacing: 0, marginLeft: 8 }}>
              {residentBlocks.length}/{budget} blocks
              {overBudget ? " (over budget, trims at turn end)" : ""}
            </span>
          </div>
          <div
            style={{
              display: "flex",
              gap: 6,
              flexWrap: "wrap",
              padding: 10,
              borderRadius: 10,
              border: `1.5px solid ${overBudget ? C.danger + "88" : C.border}`,
              background: overBudget ? "#fdf4f3" : C.faint,
              minHeight: 56,
              transition: reduce ? "none" : `border 300ms ${EASE}, background 300ms ${EASE}`,
            }}
          >
            {residentBlocks.length === 0 && (
              <span style={{ fontSize: 12, color: C.muted, alignSelf: "center" }}>empty</span>
            )}
            {residentBlocks.map((b) => (
              <BlockChip
                key={b.id}
                block={b}
                highlighted={step.highlightId === b.id}
                flash={step.flash}
                reduce={reduce}
              />
            ))}
          </div>

          <div style={{ fontSize: 11, fontWeight: 700, color: C.muted, textTransform: "uppercase", letterSpacing: 0.8, margin: "14px 0 6px" }}>
            {policy === "kv_restore" ? "Host RAM archive (saved K/V, keyed by content)" : policy === "discard" ? "Discarded" : "Archive unused"}
          </div>
          <div
            style={{
              display: "flex",
              gap: 6,
              flexWrap: "wrap",
              padding: 10,
              borderRadius: 10,
              border: `1.5px dashed ${C.border}`,
              minHeight: 44,
            }}
          >
            {policy === "kv_restore" &&
              archivedBlocks.map((b) => (
                <BlockChip
                  key={b.id}
                  block={b}
                  highlighted={step.highlightId === b.id}
                  flash={step.flash}
                  reduce={reduce}
                  shelf
                />
              ))}
            {policy === "kv_restore" && archivedBlocks.length === 0 && (
              <span style={{ fontSize: 12, color: C.muted, alignSelf: "center" }}>empty</span>
            )}
            {policy === "discard" && (
              <span style={{ fontSize: 12, color: C.muted, alignSelf: "center" }}>
                {goneCount === 0 ? "nothing discarded yet" : `${goneCount} block${goneCount === 1 ? "" : "s"} gone for good (${goneCount * TOKENS_PER_BLOCK} tokens must re-decode if re-sent)`}
              </span>
            )}
            {policy === "none" && (
              <span style={{ fontSize: 12, color: C.muted, alignSelf: "center" }}>
                nothing leaves the GPU; the cache strip above just keeps growing
              </span>
            )}
          </div>

          <div
            aria-live="polite"
            style={{
              marginTop: 14,
              padding: "10px 14px",
              borderRadius: 10,
              background: C.accentSoft,
              border: `1px solid ${C.accent}33`,
              fontSize: 12.5,
              lineHeight: 1.55,
              minHeight: 40,
            }}
          >
            <b style={{ color: C.accent }}>
              Turn {step.turn} · {PHASE_LABELS[step.phase]}:
            </b>{" "}
            {step.note}
          </div>

          <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginTop: 14 }}>
            <Btn primary onClick={() => setStepIdx((i) => Math.min(i + 1, steps.length - 1))} disabled={atEnd} ariaLabel="Advance one step">
              Step
            </Btn>
            <Btn onClick={() => setPlaying((p) => !p)} disabled={atEnd} ariaLabel={playing ? "Pause playback" : "Play all steps"}>
              {playing ? "Pause" : "Play"}
            </Btn>
            <Btn onClick={() => setStepIdx((i) => Math.max(i - 1, 0))} disabled={stepIdx === 0} ariaLabel="Go back one step">
              Back
            </Btn>
            <Btn
              onClick={() => {
                setStepIdx(0);
                setPlaying(false);
              }}
              ariaLabel="Reset to the first step"
            >
              Reset
            </Btn>
            <div style={{ marginLeft: "auto", alignSelf: "center", fontSize: 12, color: C.muted }}>
              step <b style={{ color: C.ink }}>{stepIdx + 1}</b> / {steps.length}
            </div>
          </div>
        </Card>

        <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginBottom: 16 }}>
          <Stat label="prompt tokens decoded" value={step.counters.decoded * TOKENS_PER_BLOCK} tone={C.decode} />
          <Stat label="blocks spliced back" value={step.counters.recoveries} tone={C.splice} />
          <Stat label="evictions" value={step.counters.evictions} />
          <Stat label="peak resident" value={`${step.counters.peak} blk`} />
          <Stat label="echo mismatches" value={step.counters.mismatches} />
        </div>

        <Card style={{ marginBottom: 16 }}>
          <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 4 }}>
            Same session, three policies
          </div>
          <p style={{ fontSize: 12, color: C.muted, lineHeight: 1.6, margin: "0 0 14px" }}>
            Run to the end at the current budget. The contest: prompt tokens re-decoded (left,
            lower is cheaper) against what stays resident on the GPU after the final turn (right,
            lower is leaner). Every policy spikes through the full working set mid-turn; the
            difference is what remains.
          </p>
          {comparison.map((c) => (
            <div key={c.id} style={{ marginBottom: 12 }}>
              <div style={{ fontSize: 12, fontWeight: c.id === policy ? 700 : 500, marginBottom: 4 }}>
                {c.label}
                {c.id === policy ? " (showing)" : ""}
              </div>
              <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
                <div style={{ flex: 1 }}>
                  <div
                    style={{
                      height: 14,
                      width: `${(c.final.decoded / maxDecoded) * 100}%`,
                      minWidth: 30,
                      background: C.decode,
                      opacity: 0.85,
                      borderRadius: 4,
                      transition: reduce ? "none" : `width 400ms ${EASE}`,
                    }}
                  />
                  <div style={{ fontSize: 10.5, color: C.muted, marginTop: 2 }}>
                    {c.final.decoded * TOKENS_PER_BLOCK} tokens decoded
                    {c.final.recoveries > 0 ? ` · ${c.final.recoveries} splices` : ""}
                  </div>
                </div>
                <div style={{ flex: 1 }}>
                  <div
                    style={{
                      height: 14,
                      width: `${(c.final.resident / maxResident) * 100}%`,
                      minWidth: 30,
                      background: c.final.resident > budget ? C.danger : C.match,
                      opacity: 0.85,
                      borderRadius: 4,
                      transition: reduce ? "none" : `width 400ms ${EASE}`,
                    }}
                  />
                  <div style={{ fontSize: 10.5, color: C.muted, marginTop: 2 }}>
                    {c.final.resident} blocks resident at end
                    {c.final.resident > budget ? " (over budget, still growing)" : ""}
                  </div>
                </div>
              </div>
            </div>
          ))}
          <p style={{ fontSize: 12, color: C.muted, lineHeight: 1.6, margin: "6px 0 0" }}>
            Discard respects the budget but re-decodes history every turn. No-eviction decodes
            cheaply (an intact cache is a perfect prefix cache) but its footprint grows with the
            session. EVOKE holds both: budget enforced, and recovery makes the resend nearly free.
          </p>
        </Card>

        <Card style={{ marginBottom: 16 }}>
          <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 4 }}>
            An optional signal: the model's own attention
          </div>
          <p style={{ fontSize: 12, color: C.muted, lineHeight: 1.7, margin: "0 0 4px" }}>
            The score bars above come from coherence and recency, the weights the headline
            benchmark arm uses. EVOKE can also read a second signal: the attention the model still
            pays each block. Each generated token runs one lookup over everything cached, its query
            vector is dotted against every cached key, and softmax turns the raw scores into a row
            of percentages that sums to 1. That row is the model's own record of where it looked
            while choosing the next token.{" "}
            <CrossLink to="/a/attention-explainer">Attention, From the Ground Up</CrossLink>{" "}
            builds that machinery (queries, keys, values, softmax) interactively; this page only
            needs its output. Below, one frozen decode step over nine cached tokens:
          </p>
          <SoftmaxRowDiagram />
          <p style={{ fontSize: 12, color: C.muted, lineHeight: 1.7, margin: 0 }}>
            The forked llama.cpp sums each decode step's row inside a block's token range, so the
            block gets its share of the model's attention. An attention-weighted config (attention
            0.5, coherence 0.3, recency 0.2) blends that share into the score; the headline config
            leaves attention at zero weight and scores on coherence and recency alone. Either way,
            here the work is clearly centered on <b>file: app.py</b>, so that block stays while a
            block left cold for long enough becomes the next eviction.
          </p>
        </Card>

        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: 14 }}>
          <Card>
            <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 8 }}>How a block earns its score</div>
            <ul style={{ fontSize: 12, color: C.muted, lineHeight: 1.7, margin: 0, paddingLeft: 18 }}>
              <li>
                <b>Task coherence</b> (the dominant weight, 0.60): each block's embedding is
                compared against a running task-focus embedding, so a block earns its score by
                still matching what the session is working on. This is the signal driving the
                score bars and rescore phase above; switch tasks and the focus drifts, so the old
                task's blocks go cold within a turn or two.
              </li>
              <li>
                <b>Recency</b> (0.40): an exponential decay that keeps just-arrived blocks safe and
                acts as a stability prior, so one spike cannot thrash the cache.
              </li>
              <li>
                <b>The model's own attention</b> (off by default): the forked llama.cpp can copy
                each decode step's post-softmax attention row into a host buffer and hand each
                block its share. An attention-weighted variant blends it into the score; the
                headline arm leaves it at zero.
              </li>
              <li>
                <b>Protections</b>: the first tokens are attention sinks and never leave, user
                turns floor at 0.60 and assistant turns at 0.50 so the conversation backbone
                outlives tool output, and a freshly spliced block cannot be re-evicted in the
                turn it came back.
              </li>
            </ul>
          </Card>
          <Card>
            <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 8 }}>What to watch for</div>
            <ul style={{ fontSize: 12, color: C.muted, lineHeight: 1.7, margin: 0, paddingLeft: 18 }}>
              <li>
                <b>Score bars under every chip</b>: eviction takes the lowest score, not the
                oldest block. At the default budget, turn 3's first victim is a tool ack from the
                turn before, while the turn-1 user message outlives it, because the ack no longer
                matches the task in focus.
              </li>
              <li>
                <b style={{ color: C.splice }}>Blue splices</b>: an evicted block whose exact bytes
                reappear at the same position comes back as a tensor copy, not a forward pass.
                Identity match, never similarity search; that distinction is what keeps this from
                being retrieval-augmented generation.
              </li>
              <li>
                <b style={{ color: C.decode }}>Orange decodes</b>: the only blocks that cost
                compute. Under EVOKE they are the genuinely new content plus one echo per
                assistant turn.
              </li>
              <li>
                The <b>system block is an attention sink</b>: pinned at score 1.00, it never
                leaves.
              </li>
              <li>
                Each <b>assistant block re-decodes once</b>: the client echoes a re-templated copy
                of the reply, which never byte-matches the raw emit (whitespace and tool-call JSON
                re-rendering differ). After that one decode it is canonical and recoverable.
              </li>
              <li>
                Drop the budget to 3 and watch the discard arm pay for it; raise it to 12 and the
                policies converge because nothing needs to leave.
              </li>
            </ul>
          </Card>
          <Card>
            <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 8 }}>The real measurement</div>
            <p style={{ fontSize: 12, color: C.muted, lineHeight: 1.7, margin: 0 }}>
              This simulation mirrors a live run: a real coding agent (opencode, 9 tools) built a
              notes webapp through an EVOKE server running Qwen3-8B on a 16 GB GPU at a
              2,048-token budget. The agent session saw 17,397 prompt tokens but decoded only
              9,719; all 59 evicted blocks spliced back recompute-free with zero identity
              mismatches. The eviction control re-decoded every prompt in full each turn, and the
              no-eviction control finished with 10,952 tokens resident and growing. The mechanism
              behind the blue splice is a pair of C++ primitives added to llama.cpp that copy a
              block's K/V tensors off-GPU at eviction and splice them back with positions
              re-anchored.
            </p>
          </Card>
        </div>
      </div>
    </div>
  );
}
