import { useState, useEffect, useRef, useMemo, useCallback } from "react";

export const meta = {
  title: "Dynamic Programming: the Explosion and the Collapse",
  category: "Algorithms",
  description:
    "Naive fib(20) makes 21,891 calls because the same subproblems keep coming back. Hand the recursion a cache and watch the tree collapse to 39 calls.",
  date: "2026-04-25",
  tags: ["dynamic-programming", "memoization", "recursion", "fibonacci"],
};

const C = {
  bg: "#f7f5f2",
  card: "#ffffff",
  border: "#e4dfd8",
  ink: "#1c1a17",
  muted: "#6f675e",
  faint: "#9b938a",
  accent: "#c0561f",
  accentSoft: "#f5e7df",
  hit: "#2f6d4f",
  hitSoft: "#dceae3",
  fresh: "#c0561f",
  freshSoft: "#f5e7df",
  table: "#5a6f8f",
  tableSoft: "#e1e6ee",
  repeat: "#b3341b",
};

const EASE = "cubic-bezier(0.23, 1, 0.32, 1)";
const MONO = "ui-monospace, 'SF Mono', Menlo, monospace";

const NAIVE_CAP = 12;
const N_MAX = 25;

function useReducedMotion() {
  const [reduced, setReduced] = useState(false);
  useEffect(() => {
    if (typeof window === "undefined" || !window.matchMedia) return;
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    const update = () => setReduced(mq.matches);
    update();
    mq.addEventListener("change", update);
    return () => mq.removeEventListener("change", update);
  }, []);
  return reduced;
}

function buildNaive(n) {
  const events = [];
  const nodes = [];
  let nextId = 0;
  const counts = new Array(n + 1).fill(0);

  function rec(arg, parentId, depth) {
    const id = nextId++;
    counts[arg] = (counts[arg] || 0) + 1;
    nodes.push({ id, arg, parentId, depth, children: [], value: null });
    events.push({ kind: "call", id, arg });
    let value;
    if (arg < 2) {
      value = arg;
    } else {
      const a = rec(arg - 1, id, depth + 1);
      const b = rec(arg - 2, id, depth + 1);
      value = a + b;
    }
    nodes[id].value = value;
    events.push({ kind: "return", id, arg, value });
    if (parentId !== null) nodes[parentId].children.push(id);
    return value;
  }

  const result = rec(n, null, 0);
  return { events, nodes, counts, result, callCount: nextId };
}

function buildMemoized(n) {
  const events = [];
  const nodes = [];
  let nextId = 0;
  const memo = new Map();
  let calls = 0;
  let hits = 0;

  function rec(arg, parentId, depth) {
    const id = nextId++;
    calls += 1;
    const cached = memo.has(arg);
    nodes.push({ id, arg, parentId, depth, children: [], value: null, hit: cached });
    events.push({ kind: "call", id, arg, hit: cached });
    let value;
    if (cached) {
      hits += 1;
      value = memo.get(arg);
      nodes[id].value = value;
      events.push({ kind: "hit", id, arg, value });
      if (parentId !== null) nodes[parentId].children.push(id);
      return value;
    }
    if (arg < 2) {
      value = arg;
    } else {
      const a = rec(arg - 1, id, depth + 1);
      const b = rec(arg - 2, id, depth + 1);
      value = a + b;
    }
    memo.set(arg, value);
    nodes[id].value = value;
    events.push({ kind: "store", id, arg, value });
    if (parentId !== null) nodes[parentId].children.push(id);
    return value;
  }

  const result = rec(n, null, 0);
  return { events, nodes, result, callCount: calls, hits };
}

function buildTabulated(n) {
  const table = new Array(n + 1).fill(null);
  const events = [];
  for (let i = 0; i <= n; i++) {
    if (i < 2) {
      table[i] = i;
      events.push({ i, value: i, from: null, base: true });
    } else {
      table[i] = table[i - 1] + table[i - 2];
      events.push({ i, value: table[i], from: [i - 1, i - 2], base: false });
    }
  }
  return { table: [...table], events, result: n >= 0 ? table[n] : 0, callCount: Math.max(n + 1, 1) };
}

function fmtInt(x) {
  return x.toLocaleString("en-US");
}

function layoutTree(nodes, rootId) {
  let leaf = 0;
  const xs = new Array(nodes.length).fill(0);
  const ys = new Array(nodes.length).fill(0);
  const vGap = 58;

  function place(id) {
    const node = nodes[id];
    ys[id] = node.depth * vGap + 28;
    const kids = node.children;
    if (kids.length === 0) {
      xs[id] = leaf * 46 + 24;
      leaf += 1;
    } else {
      for (const k of kids) place(k);
      xs[id] = (xs[kids[0]] + xs[kids[kids.length - 1]]) / 2;
    }
  }
  place(rootId);
  const width = Math.max(leaf * 46, 80);
  const maxDepth = nodes.reduce((m, n) => Math.max(m, n.depth), 0);
  const height = maxDepth * vGap + 70;
  return { xs, ys, width, height };
}

function Btn({ children, onClick, variant = "ghost", disabled, ariaLabel }) {
  const base = {
    fontFamily: "inherit",
    fontSize: 13,
    fontWeight: variant === "primary" ? 600 : 500,
    borderRadius: 9,
    padding: variant === "primary" ? "9px 18px" : "9px 14px",
    cursor: disabled ? "not-allowed" : "pointer",
    opacity: disabled ? 0.4 : 1,
    transition: `transform 140ms ${EASE}, background 160ms ease, border-color 160ms ease`,
  };
  const variants = {
    primary: { ...base, background: C.accent, color: "#fff", border: "none", boxShadow: "0 1px 3px rgba(192,86,31,0.3)" },
    ghost: { ...base, background: C.card, color: C.ink, border: `1.5px solid ${C.border}` },
  };
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      aria-label={ariaLabel}
      style={variants[variant]}
      onPointerDown={(e) => {
        if (!disabled) e.currentTarget.style.transform = "scale(0.97)";
      }}
      onPointerUp={(e) => (e.currentTarget.style.transform = "scale(1)")}
      onPointerLeave={(e) => (e.currentTarget.style.transform = "scale(1)")}
    >
      {children}
    </button>
  );
}

function StatBox({ label, value, sub, color, bg }) {
  return (
    <div style={{ background: bg, borderRadius: 10, padding: "12px 14px", flex: 1, minWidth: 120 }}>
      <div
        style={{
          fontSize: 10.5,
          color,
          fontWeight: 700,
          letterSpacing: "0.06em",
          textTransform: "uppercase",
          marginBottom: 4,
        }}
      >
        {label}
      </div>
      <div style={{ fontSize: 26, fontWeight: 700, color, lineHeight: 1, fontVariantNumeric: "tabular-nums" }}>
        {value}
      </div>
      {sub && <div style={{ fontSize: 11, color, opacity: 0.78, marginTop: 4 }}>{sub}</div>}
    </div>
  );
}

function NaiveTree({ build, visibleCalls, repeatArg, reduced }) {
  const { nodes } = build;
  const { xs, ys, width, height } = useMemo(() => layoutTree(nodes, 0), [nodes]);
  const callOrder = useMemo(() => {
    const order = new Array(nodes.length).fill(Infinity);
    let i = 0;
    for (const ev of build.events) {
      if (ev.kind === "call") {
        order[ev.id] = i;
        i += 1;
      }
    }
    return order;
  }, [build, nodes.length]);

  const r = 15;
  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      width="100%"
      role="img"
      aria-label={`Recursive call tree for fib. ${nodes.length} calls total.`}
      style={{ display: "block", maxHeight: 360 }}
    >
      {nodes.map((node) => {
        if (node.parentId === null) return null;
        const shown = callOrder[node.id] < visibleCalls;
        return (
          <line
            key={`e${node.id}`}
            x1={xs[node.parentId]}
            y1={ys[node.parentId]}
            x2={xs[node.id]}
            y2={ys[node.id]}
            stroke={C.border}
            strokeWidth={1.3}
            opacity={shown ? 0.6 : 0.12}
            style={{ transition: reduced ? "none" : "opacity 200ms ease" }}
          />
        );
      })}
      {nodes.map((node) => {
        const shown = callOrder[node.id] < visibleCalls;
        const isRepeat = repeatArg !== null && node.arg === repeatArg && node.arg >= 2;
        const base = node.arg < 2;
        let fill = C.card;
        let stroke = C.border;
        let ink = C.muted;
        if (shown) {
          if (isRepeat) {
            fill = "#f7dcd6";
            stroke = C.repeat;
            ink = C.repeat;
          } else if (base) {
            fill = C.hitSoft;
            stroke = C.hit;
            ink = C.hit;
          } else {
            fill = C.freshSoft;
            stroke = C.accent;
            ink = C.accent;
          }
        }
        return (
          <g
            key={`n${node.id}`}
            opacity={shown ? 1 : 0.18}
            style={{ transition: reduced ? "none" : "opacity 200ms ease" }}
          >
            <circle
              cx={xs[node.id]}
              cy={ys[node.id]}
              r={r}
              fill={fill}
              stroke={stroke}
              strokeWidth={isRepeat && shown ? 2.2 : 1.4}
              style={{ transition: reduced ? "none" : `fill 200ms ease, stroke 200ms ease` }}
            />
            <text
              x={xs[node.id]}
              y={ys[node.id] + 3.5}
              textAnchor="middle"
              fontSize={10.5}
              fontFamily={MONO}
              fontWeight={isRepeat && shown ? 700 : 500}
              fill={ink}
            >
              {node.arg}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

function MemoTree({ build, visibleCalls, reduced }) {
  const { nodes } = build;
  const { xs, ys, width, height } = useMemo(() => layoutTree(nodes, 0), [nodes]);
  const callOrder = useMemo(() => {
    const order = new Array(nodes.length).fill(Infinity);
    let i = 0;
    for (const ev of build.events) {
      if (ev.kind === "call") {
        order[ev.id] = i;
        i += 1;
      }
    }
    return order;
  }, [build, nodes.length]);

  const r = 15;
  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      width="100%"
      role="img"
      aria-label={`Memoized call tree for fib. Cache hits are served without recursing.`}
      style={{ display: "block", maxHeight: 360 }}
    >
      {nodes.map((node) => {
        if (node.parentId === null) return null;
        const shown = callOrder[node.id] < visibleCalls;
        return (
          <line
            key={`e${node.id}`}
            x1={xs[node.parentId]}
            y1={ys[node.parentId]}
            x2={xs[node.id]}
            y2={ys[node.id]}
            stroke={C.border}
            strokeWidth={1.3}
            opacity={shown ? 0.6 : 0.12}
            style={{ transition: reduced ? "none" : "opacity 200ms ease" }}
          />
        );
      })}
      {nodes.map((node) => {
        const shown = callOrder[node.id] < visibleCalls;
        const base = node.arg < 2;
        let fill = C.card;
        let stroke = C.border;
        let ink = C.muted;
        if (shown) {
          if (node.hit) {
            fill = C.hitSoft;
            stroke = C.hit;
            ink = C.hit;
          } else if (base) {
            fill = C.tableSoft;
            stroke = C.table;
            ink = C.table;
          } else {
            fill = C.freshSoft;
            stroke = C.accent;
            ink = C.accent;
          }
        }
        return (
          <g
            key={`n${node.id}`}
            opacity={shown ? 1 : 0.18}
            style={{ transition: reduced ? "none" : "opacity 200ms ease" }}
          >
            <circle
              cx={xs[node.id]}
              cy={ys[node.id]}
              r={r}
              fill={fill}
              stroke={stroke}
              strokeWidth={1.5}
            />
            <text
              x={xs[node.id]}
              y={ys[node.id] + 3.5}
              textAnchor="middle"
              fontSize={10.5}
              fontFamily={MONO}
              fontWeight={500}
              fill={ink}
            >
              {node.arg}
            </text>
            {shown && node.hit && (
              <text
                x={xs[node.id]}
                y={ys[node.id] - 20}
                textAnchor="middle"
                fontSize={8}
                fontWeight={700}
                fill={C.hit}
                letterSpacing="0.04em"
              >
                HIT
              </text>
            )}
          </g>
        );
      })}
    </svg>
  );
}

function Table({ build, filled, n, reduced }) {
  const cell = 46;
  const gap = 6;
  const total = n + 1;
  const cols = Math.min(total, 13);
  const rows = Math.ceil(total / cols);
  const w = cols * (cell + gap) - gap;
  const h = rows * (cell + 30);
  const lastFrom = filled > 0 ? build.events[filled - 1].from : null;

  return (
    <svg
      viewBox={`0 0 ${w} ${h}`}
      width="100%"
      role="img"
      aria-label={`Bottom-up table for fib, filled left to right. ${filled} of ${total} cells filled.`}
      style={{ display: "block", margin: "0 auto", maxWidth: cols * (cell + gap) }}
    >
      {build.events.map((ev) => {
        const idx = ev.i;
        const row = Math.floor(idx / cols);
        const col = idx % cols;
        const x = col * (cell + gap);
        const y = row * (cell + 30) + 18;
        const shown = idx < filled;
        const isLast = idx === filled - 1;
        const isSource = lastFrom && (idx === lastFrom[0] || idx === lastFrom[1]);
        let fill = C.card;
        let stroke = C.border;
        let ink = C.faint;
        if (shown) {
          if (isLast) {
            fill = C.freshSoft;
            stroke = C.accent;
            ink = C.accent;
          } else if (isSource) {
            fill = C.hitSoft;
            stroke = C.hit;
            ink = C.hit;
          } else {
            fill = C.tableSoft;
            stroke = C.table;
            ink = C.table;
          }
        }
        return (
          <g key={idx} style={{ transition: reduced ? "none" : "opacity 200ms ease" }} opacity={shown ? 1 : 0.4}>
            <text x={x + cell / 2} y={y - 5} textAnchor="middle" fontSize={9} fontFamily={MONO} fill={C.faint}>
              fib({idx})
            </text>
            <rect
              x={x}
              y={y}
              width={cell}
              height={cell}
              rx={8}
              fill={fill}
              stroke={stroke}
              strokeWidth={isLast && shown ? 2.2 : 1.4}
              style={{ transition: reduced ? "none" : "fill 200ms ease, stroke 200ms ease" }}
            />
            <text
              x={x + cell / 2}
              y={y + cell / 2 + 4}
              textAnchor="middle"
              fontSize={shown && ev.value > 9999 ? 9.5 : 13}
              fontFamily={MONO}
              fontWeight={isLast && shown ? 700 : 500}
              fill={ink}
            >
              {shown ? fmtInt(ev.value) : ""}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

function GrowthChart({ n, reduced }) {
  const w = 320;
  const h = 130;
  const pad = 28;
  const ns = [];
  for (let k = 2; k <= Math.max(n, 6); k++) ns.push(k);
  const naiveAt = (k) => 2 * fibClosed(k + 1) - 1;
  const linAt = (k) => 2 * k - 1;
  const maxV = naiveAt(ns[ns.length - 1]);
  const logMax = Math.log10(Math.max(maxV, 10));
  const xAt = (k) => pad + ((k - ns[0]) / (ns[ns.length - 1] - ns[0] || 1)) * (w - 2 * pad);
  const yAt = (v) => h - pad - (Math.log10(Math.max(v, 1)) / logMax) * (h - 2 * pad);
  const naivePts = ns.map((k) => `${xAt(k).toFixed(1)},${yAt(naiveAt(k)).toFixed(1)}`).join(" ");
  const linPts = ns.map((k) => `${xAt(k).toFixed(1)},${yAt(linAt(k)).toFixed(1)}`).join(" ");
  const markX = xAt(n);
  return (
    <svg viewBox={`0 0 ${w} ${h}`} width="100%" role="img" aria-label="Call count growth on a log scale: naive grows exponentially, memoized grows linearly.">
      <line x1={pad} y1={h - pad} x2={w - pad} y2={h - pad} stroke={C.border} strokeWidth={1} />
      <line x1={pad} y1={pad} x2={pad} y2={h - pad} stroke={C.border} strokeWidth={1} />
      {n >= 2 && n <= N_MAX && (
        <line x1={markX} y1={pad} x2={markX} y2={h - pad} stroke={C.faint} strokeWidth={1} strokeDasharray="3 3" opacity={0.6} />
      )}
      <polyline points={naivePts} fill="none" stroke={C.accent} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
      <polyline points={linPts} fill="none" stroke={C.hit} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
      <text x={w - pad} y={yAt(naiveAt(ns[ns.length - 1])) - 6} textAnchor="end" fontSize={9} fill={C.accent} fontWeight={700}>
        naive
      </text>
      <text x={w - pad} y={yAt(linAt(ns[ns.length - 1])) + 12} textAnchor="end" fontSize={9} fill={C.hit} fontWeight={700}>
        memoized
      </text>
      <text x={pad} y={h - 8} fontSize={8.5} fill={C.faint}>
        n = {ns[0]}
      </text>
      <text x={w - pad} y={h - 8} textAnchor="end" fontSize={8.5} fill={C.faint}>
        n = {ns[ns.length - 1]}
      </text>
      <text x={6} y={pad + 4} fontSize={8.5} fill={C.faint} transform={`rotate(-90 6 ${pad + 4})`}>
        calls (log)
      </text>
    </svg>
  );
}

function fibClosed(k) {
  let a = 0;
  let b = 1;
  for (let i = 0; i < k; i++) {
    const t = a + b;
    a = b;
    b = t;
  }
  return a;
}

const MODES = [
  { id: "naive", label: "Naive", sub: "no cache" },
  { id: "memoized", label: "Memoized", sub: "top-down" },
  { id: "tabulated", label: "Tabulated", sub: "bottom-up" },
];

export default function App() {
  const reduced = useReducedMotion();
  const [mode, setMode] = useState("naive");
  const [n, setN] = useState(8);
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const playRef = useRef(null);

  const naiveTooBig = n > NAIVE_CAP;
  const effectiveN = mode === "naive" && naiveTooBig ? NAIVE_CAP : n;

  const naiveBuild = useMemo(() => buildNaive(effectiveN), [effectiveN]);
  const memoBuild = useMemo(() => buildMemoized(n), [n]);
  const tabBuild = useMemo(() => buildTabulated(n), [n]);

  const naiveFull = useMemo(() => buildNaive(n <= NAIVE_CAP ? n : NAIVE_CAP), [n]);
  const naiveTrueCount = useMemo(() => 2 * fibClosed(n + 1) - 1, [n]);
  const memoCount = memoBuild.callCount;

  const totalSteps =
    mode === "naive"
      ? naiveBuild.callCount
      : mode === "memoized"
      ? memoBuild.callCount
      : tabBuild.events.length;

  const clampedStep = Math.min(step, totalSteps);

  useEffect(() => {
    setStep(0);
    setPlaying(false);
  }, [mode, n]);

  useEffect(() => {
    if (!playing) {
      clearInterval(playRef.current);
      return;
    }
    playRef.current = setInterval(
      () => {
        setStep((s) => {
          if (s >= totalSteps) {
            setPlaying(false);
            return s;
          }
          return s + 1;
        });
      },
      reduced ? 16 : mode === "tabulated" ? 220 : 90
    );
    return () => clearInterval(playRef.current);
  }, [playing, totalSteps, reduced, mode]);

  const repeatArg = useMemo(() => {
    if (mode !== "naive") return null;
    let best = null;
    let bestCount = 1;
    for (let k = 2; k <= effectiveN - 1; k++) {
      if ((naiveBuild.counts[k] || 0) > bestCount) {
        bestCount = naiveBuild.counts[k];
        best = k;
      }
    }
    return bestCount > 1 ? best : null;
  }, [mode, naiveBuild, effectiveN]);

  const repeatCount = repeatArg !== null ? naiveBuild.counts[repeatArg] : 0;

  const liveCount = mode === "tabulated" ? clampedStep : clampedStep;

  const stage = useMemo(() => {
    if (mode === "naive") {
      if (naiveTooBig) {
        return {
          title: `Naive is capped at n = ${NAIVE_CAP} so the tree stays drawable`,
          body: `At n = ${n} the naive tree would have ${fmtInt(2 * fibClosed(n + 1) - 1)} nodes, far too many to render. The tree below shows fib(${NAIVE_CAP}); the growth chart still reports the true count for your chosen n. This cap is honest: the explosion is real, it just does not fit on screen.`,
        };
      }
      if (clampedStep === 0) {
        return {
          title: `Naive fib(${effectiveN}): one call per node, no memory`,
          body: `Every node is a fresh function call. fib(k) calls fib(k-1) and fib(k-2), and neither remembers the other's work. Step through and watch identical subtrees appear again and again. The repeated arg fib(${repeatArg ?? "-"}) is highlighted; count how many times it recomputes.`,
        };
      }
      if (clampedStep >= totalSteps) {
        return {
          title: `Done: ${fmtInt(naiveBuild.callCount)} calls to compute fib(${effectiveN}) = ${fmtInt(naiveBuild.result)}`,
          body: `Nothing was reused. fib(${repeatArg ?? "-"}) alone was recomputed ${repeatCount} times, each time rebuilding its whole subtree from scratch. The call count follows the recurrence T(n) = T(n-1) + T(n-2) + 1, which is itself Fibonacci-shaped, so it grows exponentially. This is the explosion.`,
        };
      }
      return {
        title: `Call ${clampedStep} of ${naiveBuild.callCount}`,
        body: `Each call recurses into two smaller calls until it hits a base case (0 or 1). The same arguments keep reappearing because the two branches overlap heavily. Highlighted in red: fib(${repeatArg ?? "-"}), recomputed ${repeatCount} times in this run.`,
      };
    }
    if (mode === "memoized") {
      if (clampedStep === 0) {
        return {
          title: `Memoized fib(${n}): same recursion, one cache`,
          body: `The recursion is identical, but before recomputing fib(k) it checks a cache. The first time it sees an argument it computes and stores it; every later request for that argument is a cache hit served instantly without recursing. Step through: the tree that exploded now collapses.`,
        };
      }
      if (clampedStep >= totalSteps) {
        return {
          title: `Done: ${fmtInt(memoBuild.callCount)} calls, ${fmtInt(memoBuild.hits)} of them cache hits`,
          body: `Each distinct subproblem fib(0) through fib(${n}) is computed exactly once; the green HIT nodes are returns straight from the cache, no subtree underneath. The call count is about 2n - 1, linear in n. Same answer, fib(${n}) = ${fmtInt(memoBuild.result)}, reached without the explosion. This is the collapse.`,
        };
      }
      const ev = memoBuild.events.filter((e) => e.kind === "call")[clampedStep - 1];
      return {
        title: ev && ev.hit ? `fib(${ev.arg}) is a cache hit` : `fib(${ev ? ev.arg : ""}) computed and stored`,
        body:
          ev && ev.hit
            ? `fib(${ev.arg}) was already solved, so the cache returns its value with no recursion. The entire subtree that the naive version would have rebuilt is skipped.`
            : `First time seeing this argument: compute it from its children, store the result, and the next request for fib(${ev ? ev.arg : ""}) will be a hit.`,
      };
    }
    if (clampedStep === 0) {
      return {
        title: `Tabulated fib(${n}): fill a table, no recursion at all`,
        body: `Bottom-up tabulation drops recursion entirely. Start from the base cases fib(0) = 0 and fib(1) = 1, then fill the table left to right: each cell is the sum of the two before it. By the time you need fib(k), its inputs are already sitting in the table.`,
      };
    }
    if (clampedStep >= totalSteps) {
      return {
        title: `Done: ${n + 1} cells filled, fib(${n}) = ${fmtInt(tabBuild.result)}`,
        body: `One left-to-right pass, ${n + 1} additions, no call stack. Each cell read its two predecessors (highlighted green as you filled). Memoization caches subproblems on demand top-down; tabulation precomputes them in order bottom-up. Both visit each subproblem once, both are linear.`,
      };
    }
    const ev = tabBuild.events[clampedStep - 1];
    return {
      title: ev.base ? `Base case: fib(${ev.i}) = ${ev.i}` : `fib(${ev.i}) = fib(${ev.from[0]}) + fib(${ev.from[1]}) = ${fmtInt(ev.value)}`,
      body: ev.base
        ? `Base cases are filled directly. They anchor everything that follows.`
        : `This cell reads the two cells immediately to its left, both already filled. No recomputation: the values are just there, waiting.`,
    };
  }, [mode, clampedStep, totalSteps, effectiveN, n, naiveTooBig, naiveBuild, memoBuild, tabBuild, repeatArg, repeatCount]);

  return (
    <div
      className="dp-root"
      style={{
        fontFamily: "Georgia, 'Iowan Old Style', serif",
        background: C.bg,
        minHeight: "100vh",
        padding: "26px 14px 56px",
        color: C.ink,
      }}
    >
      <style>{`
        .dp-root :focus-visible { outline: 3px solid ${C.accent}; outline-offset: 2px; border-radius: 8px; }
        .dp-root input[type=range] { accent-color: ${C.accent}; }
        @media (prefers-reduced-motion: reduce) {
          .dp-root * { transition-duration: 0.001ms !important; animation-duration: 0.001ms !important; }
        }
      `}</style>

      <div style={{ maxWidth: 720, margin: "0 auto" }}>
        <header style={{ marginBottom: 22 }}>
          <div style={{ fontSize: 11, letterSpacing: "0.14em", textTransform: "uppercase", color: C.faint, marginBottom: 6 }}>
            Algorithms / Dynamic programming
          </div>
          <h1 style={{ fontSize: 29, fontWeight: 700, margin: 0, lineHeight: 1.12, letterSpacing: "-0.02em", textWrap: "balance" }}>
            The Explosion and the Collapse
          </h1>
          <p style={{ color: C.muted, fontSize: 15, lineHeight: 1.6, margin: "10px 0 0", maxWidth: "62ch" }}>
            Computing Fibonacci by the plain recurrence redoes the same work over and over: fib(20) takes 21,891 calls.
            Hand that recursion a place to remember answers and the same call tree collapses to 39 calls.
            Pick a mode, set n, and step through to feel where the cost goes.
          </p>
        </header>

        <div
          style={{
            display: "inline-flex",
            background: C.card,
            border: `1px solid ${C.border}`,
            borderRadius: 11,
            padding: 4,
            marginBottom: 16,
            gap: 4,
            flexWrap: "wrap",
          }}
          role="radiogroup"
          aria-label="Choose computation mode"
        >
          {MODES.map((opt) => {
            const active = mode === opt.id;
            return (
              <button
                key={opt.id}
                type="button"
                role="radio"
                aria-checked={active}
                onClick={() => setMode(opt.id)}
                style={{
                  padding: "8px 16px",
                  borderRadius: 8,
                  border: "none",
                  background: active ? C.accent : "transparent",
                  color: active ? "#fff" : C.muted,
                  fontSize: 14,
                  fontWeight: active ? 600 : 500,
                  fontFamily: "inherit",
                  cursor: "pointer",
                  transition: `background 160ms ease, color 160ms ease`,
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  gap: 1,
                  lineHeight: 1.2,
                }}
              >
                {opt.label}
                <span style={{ fontSize: 10.5, fontWeight: 500, opacity: active ? 0.85 : 0.65, fontFamily: MONO }}>
                  {opt.sub}
                </span>
              </button>
            );
          })}
        </div>

        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 12,
            marginBottom: 16,
            background: C.card,
            border: `1px solid ${C.border}`,
            borderRadius: 11,
            padding: "12px 16px",
            flexWrap: "wrap",
          }}
        >
          <label htmlFor="dp-n" style={{ fontSize: 13, color: C.muted, minWidth: 84, fontWeight: 600 }}>
            n = fib({n})
          </label>
          <input
            id="dp-n"
            type="range"
            min={1}
            max={N_MAX}
            step={1}
            value={n}
            onChange={(e) => setN(+e.target.value)}
            style={{ flex: 1, minWidth: 150 }}
            aria-valuetext={`n equals ${n}`}
          />
          <span style={{ fontFamily: MONO, fontSize: 14, fontWeight: 700, color: C.accent, minWidth: 88, textAlign: "right" }}>
            = {fmtInt(fibClosed(n))}
          </span>
        </div>

        {mode === "naive" && naiveTooBig && (
          <div
            role="note"
            style={{
              background: C.freshSoft,
              border: `1px solid ${C.accent}55`,
              borderRadius: 11,
              padding: "10px 14px",
              marginBottom: 16,
              fontSize: 13,
              lineHeight: 1.55,
              color: "#6b4b32",
            }}
          >
            Naive is capped at n = {NAIVE_CAP}. The real fib({n}) tree would need {fmtInt(naiveTrueCount)} nodes, more
            than can be drawn. The chart and counters below still use the true count for n = {n}.
          </div>
        )}

        <div style={{ display: "flex", gap: 10, marginBottom: 16, flexWrap: "wrap" }}>
          <StatBox
            label={mode === "tabulated" ? "Cells filled" : "Calls so far"}
            value={fmtInt(liveCount)}
            sub={`of ${fmtInt(totalSteps)} total`}
            color={C.accent}
            bg={C.accentSoft}
          />
          {mode === "naive" ? (
            <StatBox
              label={`fib(${repeatArg ?? "-"}) recomputed`}
              value={repeatArg !== null ? `${repeatCount}x` : "--"}
              sub="same subtree, rebuilt"
              color={C.repeat}
              bg="#f7dcd6"
            />
          ) : mode === "memoized" ? (
            <StatBox label="Cache hits" value={fmtInt(memoBuild.hits)} sub="served, no recursion" color={C.hit} bg={C.hitSoft} />
          ) : (
            <StatBox label="Recursion" value="none" sub="one bottom-up pass" color={C.table} bg={C.tableSoft} />
          )}
          <StatBox
            label={mode === "naive" ? "Naive total" : "vs naive total"}
            value={fmtInt(naiveTrueCount)}
            sub={`memoized about ${fmtInt(memoCount)}`}
            color="#8a6f3f"
            bg="#efe6d6"
          />
        </div>

        <div
          style={{
            background: C.card,
            border: `1px solid ${C.border}`,
            borderRadius: 14,
            padding: "16px 14px",
            marginBottom: 16,
            overflowX: "auto",
          }}
        >
          {mode === "naive" && (
            <NaiveTree build={naiveBuild} visibleCalls={clampedStep} repeatArg={repeatArg} reduced={reduced} />
          )}
          {mode === "memoized" && <MemoTree build={memoBuild} visibleCalls={clampedStep} reduced={reduced} />}
          {mode === "tabulated" && <Table build={tabBuild} filled={clampedStep} n={n} reduced={reduced} />}
        </div>

        <div style={{ display: "flex", gap: 9, flexWrap: "wrap", justifyContent: "center", marginBottom: 18 }}>
          <Btn onClick={() => setStep((s) => Math.min(s + 1, totalSteps))} disabled={clampedStep >= totalSteps} ariaLabel="Advance one step">
            Step
          </Btn>
          <Btn
            variant="primary"
            onClick={() => {
              if (clampedStep >= totalSteps) setStep(0);
              setPlaying((p) => !p);
            }}
            disabled={totalSteps === 0}
            ariaLabel={playing ? "Pause" : "Play"}
          >
            {playing ? "Pause" : clampedStep >= totalSteps && totalSteps > 0 ? "Replay" : "Play"}
          </Btn>
          <Btn
            onClick={() => {
              setStep(totalSteps);
              setPlaying(false);
            }}
            disabled={clampedStep >= totalSteps}
            ariaLabel="Skip to the end"
          >
            Skip to end
          </Btn>
          <Btn
            onClick={() => {
              setStep(0);
              setPlaying(false);
            }}
            ariaLabel="Reset to the start"
          >
            Reset
          </Btn>
        </div>

        <section
          aria-live="polite"
          style={{
            background: mode === "naive" ? C.freshSoft : mode === "memoized" ? C.hitSoft : C.tableSoft,
            border: `1px solid ${mode === "naive" ? C.accent + "44" : mode === "memoized" ? C.hit + "44" : C.table + "44"}`,
            borderRadius: 14,
            padding: "18px 20px",
            marginBottom: 16,
            minHeight: 110,
            transition: reduced ? "none" : "background 240ms ease, border-color 240ms ease",
          }}
        >
          <div
            style={{
              fontSize: 11,
              letterSpacing: "0.08em",
              textTransform: "uppercase",
              color: mode === "naive" ? C.accent : mode === "memoized" ? C.hit : C.table,
              fontWeight: 700,
              marginBottom: 7,
            }}
          >
            {MODES.find((m) => m.id === mode).label}
            {totalSteps > 0 && ` · step ${clampedStep} of ${totalSteps}`}
          </div>
          <h2 style={{ fontSize: 17, fontWeight: 700, margin: "0 0 8px", letterSpacing: "-0.01em" }}>{stage.title}</h2>
          <p style={{ fontSize: 14, lineHeight: 1.65, color: C.ink, margin: 0, maxWidth: "66ch" }}>{stage.body}</p>
        </section>

        <section
          style={{
            background: C.card,
            border: `1px solid ${C.border}`,
            borderRadius: 14,
            padding: "18px 20px",
            marginBottom: 16,
          }}
        >
          <h2 style={{ fontSize: 15, fontWeight: 700, margin: "0 0 4px" }}>Why the count drops from exponential to linear</h2>
          <p style={{ fontSize: 13.5, lineHeight: 1.6, color: C.muted, margin: "0 0 14px", maxWidth: "66ch" }}>
            Naive recursion grows as Fibonacci itself, so doubling the work for a few extra n. Memoization and tabulation
            both touch each subproblem once, so the count rises in a straight line. Read the two curves on a log scale: a
            straight line that keeps climbing is exponential, the nearly flat line is linear.
          </p>
          <GrowthChart n={n} reduced={reduced} />
          <div style={{ display: "flex", gap: 16, flexWrap: "wrap", marginTop: 14 }}>
            <div style={{ flex: 1, minWidth: 130 }}>
              <div style={{ fontSize: 12, color: C.muted, marginBottom: 3 }}>Naive calls, fib({n})</div>
              <div style={{ fontSize: 28, fontWeight: 700, color: C.accent, lineHeight: 1, fontVariantNumeric: "tabular-nums" }}>
                {fmtInt(naiveTrueCount)}
              </div>
              <div style={{ fontSize: 11, color: C.faint }}>2·fib(n+1) − 1</div>
            </div>
            <div style={{ flex: 1, minWidth: 130 }}>
              <div style={{ fontSize: 12, color: C.muted, marginBottom: 3 }}>Memoized calls, fib({n})</div>
              <div style={{ fontSize: 28, fontWeight: 700, color: C.hit, lineHeight: 1, fontVariantNumeric: "tabular-nums" }}>
                {fmtInt(memoCount)}
              </div>
              <div style={{ fontSize: 11, color: C.faint }}>about 2n − 1</div>
            </div>
            <div style={{ flex: 1, minWidth: 130 }}>
              <div style={{ fontSize: 12, color: C.muted, marginBottom: 3 }}>Work saved</div>
              <div style={{ fontSize: 28, fontWeight: 700, color: C.table, lineHeight: 1, fontVariantNumeric: "tabular-nums" }}>
                {naiveTrueCount > 0 ? `${Math.round((1 - memoCount / naiveTrueCount) * 100)}%` : "--"}
              </div>
              <div style={{ fontSize: 11, color: C.faint }}>fewer calls</div>
            </div>
          </div>
        </section>

        <section
          style={{
            background: C.accentSoft,
            border: `1px solid ${C.accent}33`,
            borderRadius: 14,
            padding: "18px 20px",
          }}
        >
          <h2 style={{ fontSize: 15, fontWeight: 700, margin: "0 0 10px" }}>The two ingredients</h2>
          <p style={{ fontSize: 14, lineHeight: 1.65, color: "#6b4b32", margin: 0, maxWidth: "66ch" }}>
            Dynamic programming pays off when a problem has <i>overlapping subproblems</i> (the same smaller question is
            asked many times) and <i>optimal substructure</i> (the answer is built from answers to those smaller
            questions). Fibonacci has both: fib(k) appears all over the naive tree, and it is always fib(k-1) + fib(k-2).
            Memoization is top-down caching, you keep the recursion and remember answers as you meet them; tabulation is
            bottom-up, you drop recursion and fill a table in dependency order. Either way each subproblem is solved once
            instead of once per appearance, which is exactly why the call count falls from exponential to linear.
          </p>
        </section>

        <p style={{ textAlign: "center", fontSize: 11.5, color: C.faint, marginTop: 22, lineHeight: 1.6 }}>
          fib(0) = 0, fib(1) = 1, fib(k) = fib(k-1) + fib(k-2). Call counts are measured by instrumenting the actual
          recursion; the naive tree is capped at n = {NAIVE_CAP} so it stays drawable.
        </p>
      </div>
    </div>
  );
}
