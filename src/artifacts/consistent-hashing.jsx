import { useState, useMemo, useEffect } from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts";

export const meta = {
  title: "Consistent Hashing",
  category: "Distributed Systems",
  description:
    "A hash ring you can poke at: place nodes and keys by a real hash, add and remove servers, and watch why only a thin slice of keys ever moves while mod-N reshuffles almost everything.",
  date: "2026-06-01",
  tags: ["consistent-hashing", "distributed-systems", "sharding", "load-balancing"],
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
};

const NODE_COLORS = [
  "#c0561f",
  "#2e6f8e",
  "#5a7d3c",
  "#9a5ba6",
  "#b8902a",
  "#3f5f9e",
  "#a8453f",
  "#2f7d68",
];

const RING_MAX = 0x100000000;
const EASE = "cubic-bezier(0.23, 1, 0.32, 1)";

function fnv1a(str) {
  let h = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

function angleOf(hash) {
  return (hash / RING_MAX) * 360;
}

function pointOnRing(angleDeg, radius, cx, cy) {
  const rad = ((angleDeg - 90) * Math.PI) / 180;
  return { x: cx + radius * Math.cos(rad), y: cy + radius * Math.sin(rad) };
}

function buildNodePositions(nodeIds, replicas) {
  const positions = [];
  for (const id of nodeIds) {
    for (let r = 0; r < replicas; r++) {
      const hash = fnv1a(`${id}#${r}`);
      positions.push({ nodeId: id, replica: r, hash, angle: angleOf(hash) });
    }
  }
  positions.sort((a, b) => a.hash - b.hash);
  return positions;
}

function ownerOf(keyHash, sortedPositions) {
  if (sortedPositions.length === 0) return null;
  for (const p of sortedPositions) {
    if (p.hash >= keyHash) return p.nodeId;
  }
  return sortedPositions[0].nodeId;
}

function buildKeys(count) {
  const keys = [];
  for (let i = 0; i < count; i++) {
    const id = `key:${i}`;
    const hash = fnv1a(id);
    keys.push({ id, hash, angle: angleOf(hash) });
  }
  return keys;
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
        <span style={{ fontWeight: 700, color: C.accent }}>{value}</span>
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

function colorForNode(nodeId, nodeIds) {
  const idx = nodeIds.indexOf(nodeId);
  return NODE_COLORS[idx % NODE_COLORS.length];
}

function Ring({ nodeIds, positions, keys, ownerMap, radius, highlightKeys }) {
  const size = 360;
  const cx = size / 2;
  const cy = size / 2;
  const reduce =
    typeof window !== "undefined" &&
    window.matchMedia &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const arcs = useMemo(() => {
    if (positions.length === 0) return [];
    return positions.map((p, i) => {
      const next = positions[(i + 1) % positions.length];
      let start = p.angle;
      let end = next.angle;
      if (i === positions.length - 1) end += 360;
      return { start, end, nodeId: next.nodeId };
    });
  }, [positions]);

  function arcPath(startDeg, endDeg) {
    const a = pointOnRing(startDeg, radius, cx, cy);
    const b = pointOnRing(endDeg, radius, cx, cy);
    const large = endDeg - startDeg > 180 ? 1 : 0;
    return `M ${a.x.toFixed(2)} ${a.y.toFixed(2)} A ${radius} ${radius} 0 ${large} 1 ${b.x.toFixed(2)} ${b.y.toFixed(2)}`;
  }

  return (
    <svg
      viewBox={`0 0 ${size} ${size}`}
      width="100%"
      style={{ maxWidth: 360, display: "block", margin: "0 auto" }}
      role="img"
      aria-label="Hash ring showing nodes and keys arranged by hash position, each key colored by its owning node"
    >
      <circle cx={cx} cy={cy} r={radius} fill="none" stroke={C.faint} strokeWidth={14} />
      {arcs.map((arc, i) => (
        <path
          key={i}
          d={arcPath(arc.start, arc.end)}
          fill="none"
          stroke={colorForNode(arc.nodeId, nodeIds)}
          strokeWidth={6}
          strokeLinecap="round"
          opacity={0.55}
          style={{ transition: reduce ? "none" : `stroke 280ms ${EASE}` }}
        />
      ))}

      <g style={{ fontSize: 9 }}>
        {pointOnRing(0, radius, cx, cy) && (
          <text x={cx} y={cy - radius - 18} textAnchor="middle" fill={C.muted} fontSize={9}>
            0 / 2³²
          </text>
        )}
      </g>

      {keys.map((k) => {
        const pt = pointOnRing(k.angle, radius, cx, cy);
        const owner = ownerMap[k.id];
        const moved = highlightKeys && highlightKeys.has(k.id);
        return (
          <circle
            key={k.id}
            cx={pt.x}
            cy={pt.y}
            r={moved ? 5.5 : 3.2}
            fill={owner ? colorForNode(owner, nodeIds) : C.muted}
            stroke={moved ? C.ink : "#fff"}
            strokeWidth={moved ? 1.5 : 0.8}
            style={{
              transition: reduce ? "none" : `fill 280ms ${EASE}, r 200ms ${EASE}`,
            }}
          />
        );
      })}

      {positions.map((p) => {
        const pt = pointOnRing(p.angle, radius, cx, cy);
        const col = colorForNode(p.nodeId, nodeIds);
        const isVirtual = p.replica > 0;
        return (
          <g key={`${p.nodeId}#${p.replica}`}>
            <rect
              x={pt.x - (isVirtual ? 3.5 : 5)}
              y={pt.y - (isVirtual ? 3.5 : 5)}
              width={isVirtual ? 7 : 10}
              height={isVirtual ? 7 : 10}
              rx={2}
              transform={`rotate(45 ${pt.x} ${pt.y})`}
              fill={isVirtual ? "#fff" : col}
              stroke={col}
              strokeWidth={2}
              style={{ transition: reduce ? "none" : `fill 200ms ${EASE}` }}
            />
            {!isVirtual && (
              <text
                x={pointOnRing(p.angle, radius + 18, cx, cy).x}
                y={pointOnRing(p.angle, radius + 18, cx, cy).y}
                textAnchor="middle"
                dominantBaseline="middle"
                fontSize={11}
                fontWeight={700}
                fill={col}
              >
                {p.nodeId}
              </text>
            )}
          </g>
        );
      })}
    </svg>
  );
}

function LoadChart(props) {
  const { data } = props;
  return (
    <ResponsiveContainer width="100%" height={150}>
      <BarChart data={data} margin={{ left: -18, right: 8, top: 6 }}>
        <XAxis dataKey="node" tick={{ fontSize: 11, fill: C.muted }} />
        <YAxis tick={{ fontSize: 10, fill: C.muted }} allowDecimals={false} />
        <Tooltip
          cursor={{ fill: C.faint }}
          contentStyle={{ borderRadius: 8, border: `1px solid ${C.border}`, fontSize: 12 }}
          formatter={(v) => [`${v} keys`, "load"]}
        />
        <Bar dataKey="keys" radius={[4, 4, 0, 0]} isAnimationActive={false}>
          {data.map((d, i) => (
            <Cell key={i} fill={d.color} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

function loadStats(ownerMap, nodeIds, keys) {
  const counts = Object.fromEntries(nodeIds.map((n) => [n, 0]));
  for (const k of keys) {
    const o = ownerMap[k.id];
    if (o != null && counts[o] != null) counts[o]++;
  }
  const vals = nodeIds.map((n) => counts[n]);
  const max = Math.max(1, ...vals);
  const min = Math.min(...vals);
  const ideal = keys.length / Math.max(1, nodeIds.length);
  const spread = ideal > 0 ? ((max - min) / ideal) * 100 : 0;
  return { counts, max, min, spread };
}

export default function App() {
  const [nodeIds, setNodeIds] = useState(["A", "B", "C"]);
  const [keyCount, setKeyCount] = useState(40);
  const [replicas, setReplicas] = useState(1);
  const [highlight, setHighlight] = useState(null);
  const [nextLetter, setNextLetter] = useState(3);

  const keys = useMemo(() => buildKeys(keyCount), [keyCount]);
  const positions = useMemo(
    () => buildNodePositions(nodeIds, replicas),
    [nodeIds, replicas]
  );
  const ownerMap = useMemo(() => {
    const m = {};
    for (const k of keys) m[k.id] = ownerOf(k.hash, positions);
    return m;
  }, [keys, positions]);

  useEffect(() => {
    if (!highlight) return;
    const t = setTimeout(() => setHighlight(null), 3500);
    return () => clearTimeout(t);
  }, [highlight]);

  const stats = loadStats(ownerMap, nodeIds, keys);

  const chartData = nodeIds.map((n) => ({
    node: n,
    keys: stats.counts[n],
    color: colorForNode(n, nodeIds),
  }));

  function ownerMapFor(ids) {
    const pos = buildNodePositions(ids, replicas);
    const m = {};
    for (const k of keys) m[k.id] = ownerOf(k.hash, pos);
    return m;
  }

  function addNode() {
    const letter = String.fromCharCode(65 + nextLetter);
    const before = ownerMap;
    const nextIds = [...nodeIds, letter];
    const after = ownerMapFor(nextIds);
    const moved = new Set(keys.filter((k) => before[k.id] !== after[k.id]).map((k) => k.id));
    setNodeIds(nextIds);
    setNextLetter((n) => n + 1);
    setHighlight({ moved, label: `Added node ${letter}`, count: moved.size });
  }

  function removeNode() {
    if (nodeIds.length <= 1) return;
    const removed = nodeIds[nodeIds.length - 1];
    const before = ownerMap;
    const nextIds = nodeIds.slice(0, -1);
    const after = ownerMapFor(nextIds);
    const moved = new Set(keys.filter((k) => before[k.id] !== after[k.id]).map((k) => k.id));
    setNodeIds(nextIds);
    setHighlight({ moved, label: `Removed node ${removed}`, count: moved.size });
  }

  function reset() {
    setNodeIds(["A", "B", "C"]);
    setKeyCount(40);
    setReplicas(1);
    setNextLetter(3);
    setHighlight(null);
  }

  const naiveRemap = useMemo(() => {
    const n = nodeIds.length;
    if (n <= 1) return { count: 0, pct: 0 };
    const before = keys.map((k) => k.hash % n);
    const after = keys.map((k) => k.hash % (n + 1));
    let moved = 0;
    for (let i = 0; i < keys.length; i++) if (before[i] !== after[i]) moved++;
    return { count: moved, pct: Math.round((moved / keys.length) * 100) };
  }, [keys, nodeIds.length]);

  const consistentRemapPct = keys.length
    ? Math.round((100 / Math.max(1, nodeIds.length)))
    : 0;

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
        button:focus-visible { outline: 2px solid ${C.accent}; outline-offset: 2px; }
        @media (prefers-reduced-motion: reduce) {
          * { transition: none !important; animation: none !important; }
        }
      `}</style>

      <div style={{ maxWidth: 720, margin: "0 auto" }}>
        <div style={{ marginBottom: 22 }}>
          <div
            style={{
              fontSize: 10,
              letterSpacing: "0.15em",
              textTransform: "uppercase",
              color: C.muted,
              marginBottom: 5,
            }}
          >
            Distributed Systems · Sharding
          </div>
          <h1 style={{ fontSize: 25, fontWeight: 700, margin: 0 }}>Consistent Hashing</h1>
          <p
            style={{
              color: C.muted,
              fontSize: 13.5,
              margin: "7px 0 0",
              lineHeight: 1.6,
              maxWidth: "62ch",
            }}
          >
            Every node and every key gets hashed onto one circle. A key belongs to the first node
            you meet going clockwise. That single rule is what keeps data put when the cluster grows
            or shrinks.
          </p>
        </div>

        <Card style={{ marginBottom: 16 }}>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 18 }}>
            <Btn primary onClick={addNode} ariaLabel="Add a node to the ring">
              Add node
            </Btn>
            <Btn onClick={removeNode} disabled={nodeIds.length <= 1} ariaLabel="Remove a node">
              Remove node
            </Btn>
            <Btn onClick={reset} ariaLabel="Reset everything to the starting state">
              Reset
            </Btn>
            <div
              style={{
                marginLeft: "auto",
                alignSelf: "center",
                fontSize: 12,
                color: C.muted,
              }}
            >
              <b style={{ color: C.ink }}>{nodeIds.length}</b> nodes ·{" "}
              <b style={{ color: C.ink }}>{keyCount}</b> keys
            </div>
          </div>

          <Ring
            nodeIds={nodeIds}
            positions={positions}
            keys={keys}
            ownerMap={ownerMap}
            radius={132}
            highlightKeys={highlight?.moved}
          />

          <div
            aria-live="polite"
            style={{
              marginTop: 8,
              minHeight: 38,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            {highlight ? (
              <div
                style={{
                  background: C.accentSoft,
                  border: `1px solid ${C.accent}33`,
                  color: C.accent,
                  borderRadius: 999,
                  padding: "6px 16px",
                  fontSize: 12.5,
                  fontWeight: 600,
                  transition: `opacity 200ms ${EASE}`,
                }}
              >
                {highlight.label}: <b>{highlight.count}</b> of {keyCount} keys remapped (ringed in
                black)
              </div>
            ) : (
              <div style={{ fontSize: 11.5, color: C.muted }}>
                Filled diamonds are physical nodes, small open diamonds are virtual replicas, dots
                are keys
              </div>
            )}
          </div>
        </Card>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
            gap: 14,
            marginBottom: 16,
          }}
        >
          <Card>
            <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 12 }}>Tune the ring</div>
            <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
              <Slider
                id="keys-slider"
                label="Keys on the ring"
                value={keyCount}
                min={5}
                max={120}
                onChange={(v) => {
                  setHighlight(null);
                  setKeyCount(v);
                }}
              />
              <Slider
                id="vnode-slider"
                label="Virtual nodes per physical node"
                value={replicas}
                min={1}
                max={40}
                onChange={(v) => {
                  setHighlight(null);
                  setReplicas(v);
                }}
              />
            </div>
            <p style={{ fontSize: 12, color: C.muted, lineHeight: 1.6, margin: "14px 0 0" }}>
              With one position per node the arcs are lumpy, so load is uneven. Each virtual node
              drops another diamond for the same server, slicing the circle into many small arcs
              that average out.
            </p>
          </Card>

          <Card>
            <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 4 }}>Load per node</div>
            <div style={{ fontSize: 12, color: C.muted, marginBottom: 10 }}>
              Keys owned by each physical server. Even bars mean balanced load.
            </div>
            <LoadChart data={chartData} />
            <div style={{ fontSize: 12, color: C.muted, marginTop: 8 }}>
              Imbalance:{" "}
              <b style={{ color: stats.spread > 60 ? C.accent : "#2e6f8e" }}>
                {Math.round(stats.spread)}%
              </b>{" "}
              gap between busiest and quietest, relative to a perfectly even split. Raise virtual
              nodes and watch it fall.
            </div>
          </Card>
        </div>

        <Card style={{ marginBottom: 16 }}>
          <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 4 }}>
            The point: adding a node moves almost nothing
          </div>
          <div style={{ fontSize: 12.5, color: C.muted, lineHeight: 1.6, marginBottom: 16 }}>
            Compare what happens to your {keyCount} keys when the cluster grows from {nodeIds.length}{" "}
            to {nodeIds.length + 1} servers.
          </div>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
              gap: 12,
            }}
          >
            <div
              style={{
                background: "#fdf2ee",
                border: `1px solid ${C.accent}33`,
                borderRadius: 10,
                padding: "14px 16px",
              }}
            >
              <div style={{ fontSize: 12, fontWeight: 700, color: C.accent, marginBottom: 6 }}>
                Naive: hash(key) mod N
              </div>
              <div style={{ fontSize: 30, fontWeight: 700, color: C.accent, lineHeight: 1 }}>
                ~{naiveRemap.pct}%
              </div>
              <div style={{ fontSize: 12, color: C.muted, marginTop: 6 }}>
                of keys land on a different server (about {naiveRemap.count} of {keyCount}). Changing
                the divisor reshuffles nearly the whole table, so every cache misses at once.
              </div>
            </div>
            <div
              style={{
                background: "#eef5f4",
                border: "1px solid #2e6f8e33",
                borderRadius: 10,
                padding: "14px 16px",
              }}
            >
              <div style={{ fontSize: 12, fontWeight: 700, color: "#2e6f8e", marginBottom: 6 }}>
                Consistent hashing
              </div>
              <div style={{ fontSize: 30, fontWeight: 700, color: "#2e6f8e", lineHeight: 1 }}>
                ~{consistentRemapPct}%
              </div>
              <div style={{ fontSize: 12, color: C.muted, marginTop: 6 }}>
                only the keys in the new node's arc move, roughly K/N. Everyone else keeps their
                owner. Press <b>Add node</b> above to see exactly which dots get ringed.
              </div>
            </div>
          </div>
        </Card>

        <Card style={{ background: C.accentSoft, border: `1px solid ${C.accent}22` }}>
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
            Why it works
          </div>
          <div style={{ fontSize: 13, color: C.ink, lineHeight: 1.7, display: "grid", gap: 10 }}>
            <p style={{ margin: 0 }}>
              <b>Why clockwise.</b> Picking the next node clockwise is just a rule everyone agrees
              on, so any client computes the same owner from the same ring with no coordinator. The
              direction itself does not matter, only that it is fixed.
            </p>
            <p style={{ margin: 0 }}>
              <b>Why only a slice moves.</b> A new node lands at one spot and captures only the keys
              between it and the previous node clockwise. Every key outside that arc still meets the
              same node it did before, so it stays put. That is the K/N intuition: with N nodes each
              owns about 1/N of the circle, so a new node steals roughly 1/(N+1) of the keys and
              leaves the rest alone.
            </p>
            <p style={{ margin: 0 }}>
              <b>Why virtual nodes.</b> A handful of random points split a circle into wildly uneven
              arcs, so a few servers get overloaded by luck. Giving each server many positions makes
              its total ownership the sum of many small arcs, and those averages converge toward an
              even share. The bar chart flattens as you raise the slider.
            </p>
          </div>
        </Card>

        <div style={{ marginTop: 18, textAlign: "center", fontSize: 11, color: C.muted }}>
          Positions are real FNV-1a hashes of the node and key ids, mapped from 0..2³² onto 0..360°
        </div>
      </div>
    </div>
  );
}
