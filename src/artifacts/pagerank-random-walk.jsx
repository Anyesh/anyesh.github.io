import { useState, useEffect, useMemo, useRef, useCallback } from "react";

export const meta = {
  title: "PageRank by Random Walk",
  category: "Graphs",
  description:
    "One random surfer, clicking forever, ranks a whole web. Build a graph and watch a surfer's visit frequencies converge to the same PageRank vector that power iteration computes.",
  date: "2026-04-03",
  tags: ["pagerank", "graphs", "markov-chain", "power-iteration", "random-walk"],
};

const C = {
  bg: "#f7f5f2",
  card: "#ffffff",
  border: "#e4dfd8",
  ink: "#1c1a17",
  muted: "#6f665c",
  faintInk: "#857c72",
  accent: "#c0561f",
  accentSoft: "#f6ece5",
  faint: "#efeae3",
  walk: "#2e6f8e",
  walkSoft: "#e9f1f4",
  edge: "#cfc7bc",
};

const EASE = "cubic-bezier(0.23, 1, 0.32, 1)";
const NODE_FILL = "#3a3631";

function prefersReducedMotion() {
  return (
    typeof window !== "undefined" &&
    window.matchMedia &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches
  );
}

function buildAdjacency(nodes, edges) {
  const idx = new Map(nodes.map((nd, i) => [nd.id, i]));
  const out = nodes.map(() => []);
  for (const [from, to] of edges) {
    const fi = idx.get(from);
    const ti = idx.get(to);
    if (fi != null && ti != null) out[fi].push(ti);
  }
  return { idx, out };
}

// G = d*M + (1-d)/N, with M column-stochastic. A column for a dangling node
// (no out-links) is filled uniformly so probability is conserved, otherwise the
// rank vector would leak mass and stop summing to one.
function googleMatrix(nodes, edges, d) {
  const n = nodes.length;
  const { out } = buildAdjacency(nodes, edges);
  const M = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let j = 0; j < n; j++) {
    if (out[j].length === 0) {
      for (let i = 0; i < n; i++) M[i][j] = 1 / n;
    } else {
      const w = 1 / out[j].length;
      for (const i of out[j]) M[i][j] += w;
    }
  }
  const G = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) G[i][j] = d * M[i][j] + (1 - d) / n;
  }
  return G;
}

function applyMatrix(G, r) {
  const n = r.length;
  const next = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    let acc = 0;
    for (let j = 0; j < n; j++) acc += G[i][j] * r[j];
    next[i] = acc;
  }
  return next;
}

function l1(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += Math.abs(a[i] - b[i]);
  return s;
}

function solvePageRank(nodes, edges, d, iters = 200, tol = 1e-12) {
  const n = nodes.length;
  if (n === 0) return [];
  const G = googleMatrix(nodes, edges, d);
  let r = new Array(n).fill(1 / n);
  for (let t = 0; t < iters; t++) {
    const next = applyMatrix(G, r);
    const change = l1(next, r);
    r = next;
    if (change < tol) break;
  }
  return r;
}

const SVG_W = 460;
const SVG_H = 360;

function nodeRadius(rank, maxRank) {
  const base = 15;
  const span = 24;
  if (maxRank <= 0) return base + span * 0.5;
  return base + span * Math.sqrt(rank / maxRank);
}

function heatFill(rank, maxRank, hue) {
  const t = maxRank > 0 ? rank / maxRank : 0;
  const light = hue === "walk" ? "#dcebf1" : "#f3e0d3";
  const deep = hue === "walk" ? C.walk : C.accent;
  return mixHex(light, deep, Math.sqrt(t));
}

function mixHex(a, b, t) {
  const pa = [1, 3, 5].map((i) => parseInt(a.slice(i, i + 2), 16));
  const pb = [1, 3, 5].map((i) => parseInt(b.slice(i, i + 2), 16));
  const c = pa.map((v, i) => Math.round(v + (pb[i] - v) * t));
  return `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
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

function Btn({ children, onClick, disabled, primary, tone, ariaLabel, active }) {
  const accent = tone === "walk" ? C.walk : C.accent;
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      aria-label={ariaLabel}
      aria-pressed={active}
      style={{
        padding: primary ? "8px 16px" : "7px 13px",
        borderRadius: 8,
        border: primary || active ? "none" : `1px solid ${C.border}`,
        background: primary ? accent : active ? accent + "1f" : "transparent",
        color: primary ? "#fff" : active ? accent : C.ink,
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

const DEFAULT_NODES = [
  { id: "A", x: 110, y: 90 },
  { id: "B", x: 300, y: 70 },
  { id: "C", x: 380, y: 210 },
  { id: "D", x: 230, y: 200 },
  { id: "E", x: 90, y: 270 },
];

const DEFAULT_EDGES = [
  ["A", "B"],
  ["A", "D"],
  ["B", "D"],
  ["B", "C"],
  ["C", "A"],
  ["D", "C"],
  ["D", "A"],
  ["E", "A"],
  ["E", "D"],
];

function edgeKey(a, b) {
  return `${a}->${b}`;
}

function Graph({
  nodes,
  edges,
  ranks,
  maxRank,
  hue,
  surfer,
  pendingFrom,
  onNodeClick,
  flashEdge,
}) {
  const reduce = prefersReducedMotion();
  const pos = useMemo(() => new Map(nodes.map((n) => [n.id, n])), [nodes]);
  const edgeSet = useMemo(() => new Set(edges.map(([a, b]) => edgeKey(a, b))), [edges]);

  function edgePath(a, b, curve) {
    const p = pos.get(a);
    const q = pos.get(b);
    if (!p || !q) return null;
    const dx = q.x - p.x;
    const dy = q.y - p.y;
    const len = Math.hypot(dx, dy) || 1;
    const ux = dx / len;
    const uy = dy / len;
    const rB = nodeRadius(ranks[nodes.findIndex((n) => n.id === b)] ?? 0, maxRank);
    const rA = nodeRadius(ranks[nodes.findIndex((n) => n.id === a)] ?? 0, maxRank);
    const sx = p.x + ux * rA;
    const sy = p.y + uy * rA;
    const ex = q.x - ux * (rB + 7);
    const ey = q.y - uy * (rB + 7);
    const nx = -uy;
    const ny = ux;
    const bow = curve ? 18 : 0;
    const mx = (sx + ex) / 2 + nx * bow;
    const my = (sy + ey) / 2 + ny * bow;
    return { d: `M ${sx} ${sy} Q ${mx} ${my} ${ex} ${ey}`, ex, ey, mx, my };
  }

  return (
    <svg
      viewBox={`0 0 ${SVG_W} ${SVG_H}`}
      width="100%"
      style={{ maxWidth: SVG_W, display: "block", margin: "0 auto", touchAction: "manipulation" }}
      role="img"
      aria-label="Directed graph. Node size and color show PageRank; click a node then another to add or remove a directed edge."
    >
      <defs>
        <marker id="pr-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill={C.edge} />
        </marker>
        <marker id="pr-arrow-hot" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill={hue === "walk" ? C.walk : C.accent} />
        </marker>
      </defs>

      {edges.map(([a, b]) => {
        const hasReverse = edgeSet.has(edgeKey(b, a));
        const geo = edgePath(a, b, hasReverse);
        if (!geo) return null;
        const hot = flashEdge === edgeKey(a, b);
        return (
          <path
            key={edgeKey(a, b)}
            d={geo.d}
            fill="none"
            stroke={hot ? (hue === "walk" ? C.walk : C.accent) : C.edge}
            strokeWidth={hot ? 3 : 1.6}
            markerEnd={hot ? "url(#pr-arrow-hot)" : "url(#pr-arrow)"}
            style={{ transition: reduce ? "none" : `stroke 200ms ease, stroke-width 200ms ease` }}
          />
        );
      })}

      {nodes.map((n, i) => {
        const r = nodeRadius(ranks[i] ?? 0, maxRank);
        const fill = heatFill(ranks[i] ?? 0, maxRank, hue);
        const isPending = pendingFrom === n.id;
        const isSurfer = surfer === n.id;
        const accent = hue === "walk" ? C.walk : C.accent;
        return (
          <g
            key={n.id}
            onClick={() => onNodeClick && onNodeClick(n.id)}
            style={{ cursor: onNodeClick ? "pointer" : "default" }}
          >
            {isSurfer && (
              <circle
                cx={n.x}
                cy={n.y}
                r={r + 9}
                fill="none"
                stroke={C.walk}
                strokeWidth={2.5}
                opacity={0.9}
                style={{ transition: reduce ? "none" : `cx 260ms ${EASE}, cy 260ms ${EASE}` }}
              />
            )}
            <circle
              cx={n.x}
              cy={n.y}
              r={r}
              fill={fill}
              stroke={isPending ? accent : NODE_FILL}
              strokeWidth={isPending ? 3 : 1.5}
              strokeDasharray={isPending ? "4 3" : "none"}
              style={{ transition: reduce ? "none" : `r 320ms ${EASE}, fill 320ms ${EASE}` }}
            />
            <text
              x={n.x}
              y={n.y}
              textAnchor="middle"
              dominantBaseline="central"
              fontSize={14}
              fontWeight={700}
              fill={(ranks[i] ?? 0) / (maxRank || 1) > 0.55 ? "#fff" : NODE_FILL}
              style={{ pointerEvents: "none", userSelect: "none" }}
            >
              {n.id}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

function RankBars({ nodes, ranks, hue, analytic, topId }) {
  const accent = hue === "walk" ? C.walk : C.accent;
  const order = nodes
    .map((n, i) => ({ id: n.id, rank: ranks[i] ?? 0, ref: analytic ? analytic[i] : null }))
    .sort((a, b) => b.rank - a.rank);
  const max = Math.max(0.0001, ...order.map((o) => o.rank), ...(analytic || [0]));
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      {order.map((o) => {
        const isTop = o.id === topId;
        return (
          <div key={o.id} style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <span
              style={{
                width: 20,
                fontWeight: 700,
                fontSize: 13,
                color: isTop ? accent : C.ink,
              }}
            >
              {o.id}
            </span>
            <div
              style={{
                flex: 1,
                position: "relative",
                background: C.faint,
                borderRadius: 6,
                height: 18,
                overflow: "hidden",
              }}
            >
              <div
                style={{
                  width: `${(o.rank / max) * 100}%`,
                  height: "100%",
                  background: isTop ? accent : accent + "99",
                  borderRadius: 6,
                  transition: `width 320ms ${EASE}, background 200ms ease`,
                }}
              />
              {o.ref != null && (
                <div
                  aria-hidden="true"
                  title="Analytic PageRank"
                  style={{
                    position: "absolute",
                    top: -2,
                    bottom: -2,
                    left: `${(o.ref / max) * 100}%`,
                    width: 2,
                    background: C.ink,
                    opacity: 0.55,
                  }}
                />
              )}
            </div>
            <span
              style={{
                width: 56,
                textAlign: "right",
                fontVariantNumeric: "tabular-nums",
                fontSize: 12.5,
                color: C.muted,
              }}
            >
              {(o.rank * 100).toFixed(2)}%
            </span>
          </div>
        );
      })}
    </div>
  );
}

export default function App() {
  const [nodes, setNodes] = useState(DEFAULT_NODES);
  const [edges, setEdges] = useState(DEFAULT_EDGES);
  const [damping, setDamping] = useState(0.85);
  const [pendingFrom, setPendingFrom] = useState(null);
  const [flashEdge, setFlashEdge] = useState(null);

  const [rankIter, setRankIter] = useState(() => new Array(DEFAULT_NODES.length).fill(1 / DEFAULT_NODES.length));
  const [iterCount, setIterCount] = useState(0);
  const [lastDelta, setLastDelta] = useState(null);
  const [playing, setPlaying] = useState(false);

  const [surferIdx, setSurferIdx] = useState(0);
  const [visits, setVisits] = useState(() => new Array(DEFAULT_NODES.length).fill(0));
  const [walkSteps, setWalkSteps] = useState(0);
  const [teleports, setTeleports] = useState(0);
  const [walking, setWalking] = useState(false);

  const playRef = useRef(null);
  const walkRef = useRef(null);
  const surferIdxRef = useRef(0);

  const analytic = useMemo(
    () => solvePageRank(nodes, edges, damping),
    [nodes, edges, damping]
  );

  const G = useMemo(() => googleMatrix(nodes, edges, damping), [nodes, edges, damping]);
  const adjacency = useMemo(() => buildAdjacency(nodes, edges), [nodes, edges]);

  const resetIteration = useCallback(() => {
    setRankIter(new Array(nodes.length).fill(1 / nodes.length));
    setIterCount(0);
    setLastDelta(null);
    setPlaying(false);
  }, [nodes.length]);

  const resetWalk = useCallback(() => {
    setVisits(new Array(nodes.length).fill(0));
    setWalkSteps(0);
    setTeleports(0);
    setSurferIdx(0);
    setWalking(false);
  }, [nodes.length]);

  useEffect(() => {
    resetIteration();
    resetWalk();
    setPendingFrom(null);
  }, [nodes, edges, damping, resetIteration, resetWalk]);

  const stepIteration = useCallback(() => {
    setRankIter((prev) => {
      const next = applyMatrix(G, prev);
      setLastDelta(l1(next, prev));
      return next;
    });
    setIterCount((c) => c + 1);
  }, [G]);

  useEffect(() => {
    if (!playing) {
      clearInterval(playRef.current);
      return;
    }
    playRef.current = setInterval(() => {
      setRankIter((prev) => {
        const next = applyMatrix(G, prev);
        const change = l1(next, prev);
        setLastDelta(change);
        if (change < 1e-9) setPlaying(false);
        return next;
      });
      setIterCount((c) => c + 1);
    }, 420);
    return () => clearInterval(playRef.current);
  }, [playing, G]);

  const stepWalk = useCallback(
    (batch = 1) => {
      const { out } = adjacency;
      const n = nodes.length;
      setVisits((prevVisits) => {
        const v = prevVisits.slice();
        let cur = surferIdxRef.current;
        let tp = 0;
        for (let s = 0; s < batch; s++) {
          v[cur]++;
          const teleport = out[cur].length === 0 || Math.random() >= damping;
          if (teleport) {
            cur = Math.floor(Math.random() * n);
            tp++;
          } else {
            cur = out[cur][Math.floor(Math.random() * out[cur].length)];
          }
        }
        surferIdxRef.current = cur;
        setSurferIdx(cur);
        setTeleports((t) => t + tp);
        return v;
      });
      setWalkSteps((w) => w + batch);
    },
    [adjacency, nodes.length, damping]
  );

  useEffect(() => {
    surferIdxRef.current = surferIdx;
  }, [surferIdx]);

  useEffect(() => {
    if (!walking) {
      clearInterval(walkRef.current);
      return;
    }
    const reduce = prefersReducedMotion();
    walkRef.current = setInterval(() => stepWalk(1), reduce ? 80 : 360);
    return () => clearInterval(walkRef.current);
  }, [walking, stepWalk]);

  function handleNodeClick(id) {
    if (pendingFrom == null) {
      setPendingFrom(id);
      return;
    }
    if (pendingFrom === id) {
      setPendingFrom(null);
      return;
    }
    const key = edgeKey(pendingFrom, id);
    const exists = edges.some(([a, b]) => edgeKey(a, b) === key);
    if (exists) {
      setEdges((es) => es.filter(([a, b]) => edgeKey(a, b) !== key));
    } else {
      setEdges((es) => [...es, [pendingFrom, id]]);
      setFlashEdge(key);
      setTimeout(() => setFlashEdge(null), 700);
    }
    setPendingFrom(null);
  }

  function addNode() {
    if (nodes.length >= 8) return;
    const used = new Set(nodes.map((n) => n.id));
    let code = 65;
    while (used.has(String.fromCharCode(code))) code++;
    const id = String.fromCharCode(code);
    const angle = (nodes.length / 8) * Math.PI * 2;
    const x = SVG_W / 2 + Math.cos(angle) * 150 + (Math.random() - 0.5) * 30;
    const y = SVG_H / 2 + Math.sin(angle) * 120 + (Math.random() - 0.5) * 30;
    setNodes((ns) => [
      ...ns,
      { id, x: Math.max(40, Math.min(SVG_W - 40, x)), y: Math.max(40, Math.min(SVG_H - 40, y)) },
    ]);
  }

  function removeNode() {
    if (nodes.length <= 2) return;
    const victim = nodes[nodes.length - 1].id;
    setNodes((ns) => ns.slice(0, -1));
    setEdges((es) => es.filter(([a, b]) => a !== victim && b !== victim));
  }

  function fullReset() {
    setNodes(DEFAULT_NODES);
    setEdges(DEFAULT_EDGES);
    setDamping(0.85);
    setPendingFrom(null);
  }

  const maxIter = Math.max(0.0001, ...rankIter);
  const maxVisit = Math.max(1, ...visits);
  const empirical = visits.map((v) => (walkSteps > 0 ? v / walkSteps : 0));
  const maxEmp = Math.max(0.0001, ...empirical);

  const topAnalyticId = useMemo(() => {
    let best = 0;
    for (let i = 1; i < analytic.length; i++) if (analytic[i] > analytic[best]) best = i;
    return nodes[best]?.id;
  }, [analytic, nodes]);

  const iterError = l1(rankIter, analytic);
  const walkError = empirical.length ? l1(empirical, analytic) : null;
  const dangling = adjacency.out.filter((o) => o.length === 0).length;

  return (
    <div
      style={{
        fontFamily: "'Palatino Linotype', 'Book Antiqua', Palatino, serif",
        background: C.bg,
        minHeight: "100vh",
        padding: "24px 14px",
        color: C.ink,
      }}
    >
      <style>{`
        input[type="range"]:focus-visible { outline: 2px solid ${C.accent}; outline-offset: 3px; border-radius: 4px; }
        button:focus-visible { outline: 2px solid ${C.accent}; outline-offset: 2px; border-radius: 8px; }
        @media (prefers-reduced-motion: reduce) {
          * { transition-duration: 0.001ms !important; }
        }
      `}</style>

      <div style={{ maxWidth: 720, margin: "0 auto" }}>
        <div style={{ marginBottom: 22 }}>
          <div
            style={{
              fontSize: 10,
              letterSpacing: "0.15em",
              textTransform: "uppercase",
              color: C.faintInk,
              marginBottom: 5,
            }}
          >
            Graphs · Markov Chains
          </div>
          <h1 style={{ fontSize: 25, fontWeight: 700, margin: 0 }}>PageRank by Random Walk</h1>
          <p
            style={{
              color: C.muted,
              fontSize: 13.5,
              margin: "7px 0 0",
              lineHeight: 1.6,
              maxWidth: "64ch",
            }}
          >
            PageRank scores a page by how often a surfer who clicks random links, and occasionally
            jumps to a page at random, ends up there. Two methods reach the same numbers: power
            iteration on the Google matrix, and a single surfer counting its own visits. Edit the
            graph and watch both views track each other.
          </p>
        </div>

        <Card style={{ marginBottom: 16 }}>
          <div
            style={{
              display: "flex",
              gap: 8,
              flexWrap: "wrap",
              alignItems: "center",
              marginBottom: 14,
            }}
          >
            <Btn onClick={addNode} disabled={nodes.length >= 8} ariaLabel="Add a node">
              Add node
            </Btn>
            <Btn onClick={removeNode} disabled={nodes.length <= 2} ariaLabel="Remove the last node">
              Remove node
            </Btn>
            <Btn onClick={fullReset} ariaLabel="Reset the graph to its starting layout">
              Reset graph
            </Btn>
            <div style={{ marginLeft: "auto", fontSize: 12, color: C.muted }}>
              <b style={{ color: C.ink }}>{nodes.length}</b> nodes ·{" "}
              <b style={{ color: C.ink }}>{edges.length}</b> edges
            </div>
          </div>

          <Graph
            nodes={nodes}
            edges={edges}
            ranks={analytic}
            maxRank={Math.max(0.0001, ...analytic)}
            hue="rank"
            surfer={walkSteps > 0 ? nodes[surferIdx]?.id : null}
            pendingFrom={pendingFrom}
            onNodeClick={handleNodeClick}
            flashEdge={flashEdge}
          />

          <div
            aria-live="polite"
            style={{
              marginTop: 6,
              minHeight: 22,
              textAlign: "center",
              fontSize: 12.5,
              color: pendingFrom ? C.accent : C.muted,
              fontWeight: pendingFrom ? 600 : 400,
            }}
          >
            {pendingFrom
              ? `From ${pendingFrom}: tap a second node to add or remove that directed edge, or tap ${pendingFrom} again to cancel.`
              : "Tap one node, then another, to add a directed edge. Tap an existing edge's endpoints to remove it. Node size and shade follow PageRank."}
          </div>
        </Card>

        <Card style={{ marginBottom: 16 }}>
          <label
            htmlFor="damping"
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "baseline",
              fontSize: 13,
              fontWeight: 700,
              marginBottom: 8,
            }}
          >
            <span>Damping factor d</span>
            <span style={{ color: C.accent, fontVariantNumeric: "tabular-nums" }}>
              {damping.toFixed(2)}
            </span>
          </label>
          <input
            id="damping"
            type="range"
            min={0.05}
            max={0.99}
            step={0.01}
            value={damping}
            onChange={(e) => setDamping(+e.target.value)}
            style={{ width: "100%", accentColor: C.accent }}
            aria-valuetext={`damping ${damping.toFixed(2)}`}
          />
          <p style={{ fontSize: 12.5, color: C.muted, lineHeight: 1.6, margin: "10px 0 0" }}>
            With probability <b style={{ color: C.ink }}>d</b> the surfer follows a random out-link;
            with probability <b style={{ color: C.ink }}>1 - d = {(1 - damping).toFixed(2)}</b> it
            teleports to a node picked uniformly at random. Teleportation is what makes the chain
            irreducible, so a unique ranking exists no matter how the links are wired. The classic
            web value is 0.85.
          </p>
        </Card>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))",
            gap: 16,
            marginBottom: 16,
          }}
        >
          <Card>
            <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 3 }}>Power iteration</div>
            <div style={{ fontSize: 12, color: C.muted, marginBottom: 12, lineHeight: 1.5 }}>
              Start uniform, then repeatedly apply r &larr; G r. Each pass moves probability along
              the links until it stops changing.
            </div>

            <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 14 }}>
              <Btn onClick={stepIteration} ariaLabel="Apply one power-iteration step">
                Step
              </Btn>
              <Btn
                primary
                onClick={() => setPlaying((p) => !p)}
                ariaLabel={playing ? "Pause power iteration" : "Run power iteration"}
                active={playing}
              >
                {playing ? "Pause" : "Run"}
              </Btn>
              <Btn onClick={resetIteration} ariaLabel="Reset power iteration to uniform">
                Reset
              </Btn>
            </div>

            <RankBars
              nodes={nodes}
              ranks={rankIter}
              hue="rank"
              analytic={analytic}
              topId={topAnalyticId}
            />

            <div
              style={{
                display: "flex",
                gap: 10,
                marginTop: 14,
                flexWrap: "wrap",
                fontSize: 12,
                color: C.muted,
              }}
            >
              <span>
                Iteration <b style={{ color: C.ink, fontVariantNumeric: "tabular-nums" }}>{iterCount}</b>
              </span>
              <span>
                L1 change{" "}
                <b style={{ color: C.accent, fontVariantNumeric: "tabular-nums" }}>
                  {lastDelta == null ? "—" : lastDelta.toExponential(2)}
                </b>
              </span>
              <span style={{ marginLeft: "auto" }}>
                gap to exact{" "}
                <b style={{ color: C.ink, fontVariantNumeric: "tabular-nums" }}>
                  {iterError.toExponential(2)}
                </b>
              </span>
            </div>
            <div style={{ fontSize: 11.5, color: C.faintInk, marginTop: 6 }}>
              Thin dark mark on each bar is the exact PageRank. The bars slide onto it within a few
              iterations.
            </div>
          </Card>

          <Card>
            <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 3, color: C.walk }}>
              Random surfer
            </div>
            <div style={{ fontSize: 12, color: C.muted, marginBottom: 12, lineHeight: 1.5 }}>
              One walker hops the real edges, teleporting with probability 1 - d. Its visit
              frequencies approach the exact ranks as steps pile up.
            </div>

            <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 14 }}>
              <Btn tone="walk" onClick={() => stepWalk(1)} ariaLabel="Take one surfer step">
                Step
              </Btn>
              <Btn
                tone="walk"
                onClick={() => stepWalk(2000)}
                ariaLabel="Take two thousand surfer steps at once"
              >
                +2000
              </Btn>
              <Btn
                primary
                tone="walk"
                onClick={() => setWalking((w) => !w)}
                ariaLabel={walking ? "Pause the surfer" : "Run the surfer"}
                active={walking}
              >
                {walking ? "Pause" : "Run"}
              </Btn>
              <Btn tone="walk" onClick={resetWalk} ariaLabel="Reset the surfer and its counts">
                Reset
              </Btn>
            </div>

            <RankBars
              nodes={nodes}
              ranks={empirical}
              hue="walk"
              analytic={analytic}
              topId={topAnalyticId}
            />

            <div
              style={{
                display: "flex",
                gap: 10,
                marginTop: 14,
                flexWrap: "wrap",
                fontSize: 12,
                color: C.muted,
              }}
            >
              <span>
                Steps{" "}
                <b style={{ color: C.ink, fontVariantNumeric: "tabular-nums" }}>
                  {walkSteps.toLocaleString()}
                </b>
              </span>
              <span>
                Teleports{" "}
                <b style={{ color: C.walk, fontVariantNumeric: "tabular-nums" }}>
                  {teleports.toLocaleString()}
                </b>
              </span>
              <span style={{ marginLeft: "auto" }}>
                gap to exact{" "}
                <b style={{ color: C.ink, fontVariantNumeric: "tabular-nums" }}>
                  {walkError == null || walkSteps === 0 ? "—" : walkError.toFixed(4)}
                </b>
              </span>
            </div>
            <div style={{ fontSize: 11.5, color: C.faintInk, marginTop: 6 }}>
              {walkSteps === 0
                ? "Press Run or +2000 to start counting visits. The bar chart fills in as the walk explores."
                : "Same dark marks as the left panel. Noisy at first, the empirical bars settle onto the exact ranks."}
            </div>
          </Card>
        </div>

        <Card style={{ marginBottom: 16, background: C.accentSoft, border: `1px solid ${C.accent}22` }}>
          <div
            style={{
              fontSize: 11,
              fontWeight: 700,
              letterSpacing: "0.08em",
              textTransform: "uppercase",
              color: C.accent,
              marginBottom: 10,
            }}
          >
            Why the two views agree
          </div>
          <div style={{ fontSize: 13, color: C.ink, lineHeight: 1.7, display: "grid", gap: 10 }}>
            <p style={{ margin: 0 }}>
              <b>The Google matrix.</b> Column j of M says where the surfer goes from node j: an
              even split across its out-links. The full operator is G = d M + (1 - d)/N, mixing those
              links with a uniform teleport. PageRank is the vector r that G leaves unchanged, the
              stationary distribution of the chain. Power iteration finds it by applying G over and
              over from any start.
            </p>
            <p style={{ margin: 0 }}>
              <b>Dangling nodes.</b>{" "}
              {dangling > 0 ? (
                <>
                  This graph has <b>{dangling}</b> node{dangling > 1 ? "s" : ""} with no out-links.
                </>
              ) : (
                <>A node with no out-links is a dangling node.</>
              )}{" "}
              A surfer landing there has nowhere to click, so its column is filled uniformly: from a
              dead end the walker restarts anywhere. Without that fix probability drains out of the
              system and the ranks no longer sum to one.
            </p>
            <p style={{ margin: 0 }}>
              <b>Walk equals matrix.</b> By the ergodic theorem the long-run fraction of time the
              surfer spends at each node is exactly that stationary vector. So counting visits and
              powering up the matrix are two routes to the same ranking; the surfer is just a slow,
              noisy way to read off r.
            </p>
          </div>
        </Card>

        <div style={{ marginTop: 18, textAlign: "center", fontSize: 11, color: C.faintInk }}>
          Power iteration on G = d M + (1 - d)/N, with dangling columns spread uniformly. The
          surfer is a Monte Carlo sample of the same chain.
        </div>
      </div>
    </div>
  );
}
