import { useState, useMemo, useEffect, useRef } from "react";

export const meta = {
  title: "Huffman Coding",
  category: "Information Theory",
  description:
    "Type a string and watch a real Huffman tree assemble itself bottom up: count symbols, merge the two rarest nodes again and again, then read prefix-free codewords off the branches. See the encoded size, the compression ratio, and how the average code length lands between the source entropy and entropy plus one bit.",
  date: "2026-06-02",
  tags: ["huffman", "compression", "entropy", "information-theory"],
};

const C = {
  bg: "#f7f5f2",
  card: "#ffffff",
  border: "#e4dfd8",
  ink: "#1c1a17",
  muted: "#6f675d",
  accent: "#c0561f",
  accentSoft: "#f6ece5",
  faint: "#efeae3",
  leaf: "#c0561f",
  branch: "#2e6f8e",
  good: "#2f7d53",
  goldL: "#f4ecd6",
  gold: "#8a6a1f",
};

const EASE = "cubic-bezier(0.23, 1, 0.32, 1)";

const PRESETS = [
  {
    id: "sentence",
    label: "Natural language",
    text: "the quick brown fox jumps over the lazy dog",
    note: "English text: a handful of common letters dominate, so Huffman shaves real bits.",
  },
  {
    id: "skewed",
    label: "Skewed",
    text: "aaaaaaaaaaaaaaaaaaaaaaaabbbbbbcccdde",
    note: "One symbol swamps the rest. Huffman gives it a 1 bit code and the long tail pays.",
  },
  {
    id: "uniform",
    label: "Uniform",
    text: "abcdefghabcdefgh",
    note: "Eight equally likely symbols. Huffman degenerates to a flat 3 bit code, matching log2(8).",
  },
  {
    id: "single",
    label: "One symbol",
    text: "aaaaaaaa",
    note: "A degenerate alphabet. By convention the lone symbol still needs a 1 bit code.",
  },
];

function symbolLabel(ch) {
  if (ch === " ") return "space";
  if (ch === "\n") return "\\n";
  if (ch === "\t") return "\\t";
  return ch;
}

function frequencies(text) {
  const map = new Map();
  for (const ch of text) map.set(ch, (map.get(ch) || 0) + 1);
  return map;
}

// Ties are broken by insertion order (a monotonic counter) so the tree is fully
// reproducible: two nodes of equal frequency always pop in the order they were
// created, which keeps codeword assignment deterministic across runs.
function buildHuffman(freqMap) {
  const symbols = [...freqMap.entries()];
  if (symbols.length === 0) return { root: null, steps: [], nodes: [], order: [] };

  let counter = 0;
  const order = [];
  let forest = symbols.map(([sym, freq]) => {
    const node = {
      id: counter,
      sym,
      freq,
      left: null,
      right: null,
      seq: counter,
    };
    order.push(counter);
    counter++;
    return node;
  });

  const lessThan = (a, b) => (a.freq !== b.freq ? a.freq < b.freq : a.seq < b.seq);

  const steps = [];
  const snapshotForest = (f) => f.map((n) => n.id);

  if (forest.length === 1) {
    const only = forest[0];
    const root = {
      id: counter,
      sym: null,
      freq: only.freq,
      left: only,
      right: null,
      seq: counter,
    };
    order.push(counter);
    counter++;
    steps.push({
      merged: [only.id],
      parent: root.id,
      forestBefore: [only.id],
      forestAfter: [root.id],
      single: true,
    });
    const nodeIndex = indexNodes(root);
    return { root, steps, nodes: nodeIndex, order };
  }

  while (forest.length > 1) {
    forest.sort((a, b) => (lessThan(a, b) ? -1 : 1));
    const a = forest[0];
    const b = forest[1];
    const forestBefore = snapshotForest(forest);
    const parent = {
      id: counter,
      sym: null,
      freq: a.freq + b.freq,
      left: a,
      right: b,
      seq: counter,
    };
    order.push(counter);
    counter++;
    forest = forest.slice(2);
    forest.push(parent);
    steps.push({
      merged: [a.id, b.id],
      parent: parent.id,
      forestBefore,
      forestAfter: snapshotForest([...forest].sort((x, y) => (lessThan(x, y) ? -1 : 1))),
    });
  }

  const root = forest[0];
  const nodeIndex = indexNodes(root);
  return { root, steps, nodes: nodeIndex, order };
}

function indexNodes(root) {
  const map = new Map();
  const walk = (node) => {
    if (!node) return;
    map.set(node.id, node);
    walk(node.left);
    walk(node.right);
  };
  walk(root);
  return map;
}

function assignCodes(root) {
  const codes = new Map();
  if (!root) return codes;
  const walk = (node, prefix) => {
    if (!node) return;
    const isLeaf = node.sym !== null;
    if (isLeaf) {
      // A single distinct symbol yields a root with no real branch depth, so it
      // must still receive a 1 bit code by convention rather than the empty string.
      codes.set(node.id, prefix === "" ? "0" : prefix);
      return;
    }
    walk(node.left, prefix + "0");
    walk(node.right, prefix + "1");
  };
  walk(root, "");
  return codes;
}

function entropyBits(freqMap, total) {
  if (total === 0) return 0;
  let h = 0;
  for (const freq of freqMap.values()) {
    const p = freq / total;
    h -= p * Math.log2(p);
  }
  return h;
}

// Treemap-free layered layout: leaves are spread evenly across the width and each
// internal node centers over its children, so the final tree reads top down.
function layoutTree(root) {
  if (!root) return { positions: new Map(), width: 0, depth: 0 };
  let leafCount = 0;
  let maxDepth = 0;
  const positions = new Map();

  const measure = (node, depth) => {
    if (!node) return;
    maxDepth = Math.max(maxDepth, depth);
    const hasLeft = !!node.left;
    const hasRight = !!node.right;
    if (!hasLeft && !hasRight) {
      positions.set(node.id, { col: leafCount, depth });
      leafCount++;
      return;
    }
    measure(node.left, depth + 1);
    measure(node.right, depth + 1);
    const kids = [node.left, node.right].filter(Boolean).map((k) => positions.get(k.id).col);
    const col = kids.reduce((a, b) => a + b, 0) / kids.length;
    positions.set(node.id, { col, depth });
  };

  measure(root, 0);
  return { positions, leafCount, depth: maxDepth };
}

function usePrefersReducedMotion() {
  const [reduce, setReduce] = useState(false);
  useEffect(() => {
    if (typeof window === "undefined" || !window.matchMedia) return;
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    const update = () => setReduce(mq.matches);
    update();
    mq.addEventListener("change", update);
    return () => mq.removeEventListener("change", update);
  }, []);
  return reduce;
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

function Btn({ children, onClick, disabled, primary, ariaLabel, type }) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      type={type || "button"}
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

function Stat({ label, value, sub, color, bg }) {
  return (
    <div style={{ background: bg, borderRadius: 10, padding: "12px 14px", flex: 1, minWidth: 130 }}>
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
      <div style={{ fontSize: 22, fontWeight: 700, color, lineHeight: 1.05 }}>{value}</div>
      {sub && <div style={{ fontSize: 11, color: C.muted, marginTop: 4 }}>{sub}</div>}
    </div>
  );
}

const NODE_W = 52;
const ROW_H = 78;
const PAD_X = 34;
const PAD_Y = 28;

function HuffmanTree({ build, codes, visibleNodeIds, highlightPath, reduce, onPickLeaf }) {
  const { root } = build;
  const layout = useMemo(() => layoutTree(root), [root]);
  if (!root) return null;

  const cols = Math.max(1, layout.leafCount);
  const width = (cols - 1) * (NODE_W + 16) + NODE_W + PAD_X * 2;
  const height = (layout.depth + 1) * ROW_H + PAD_Y * 2;

  const xOf = (col) => PAD_X + NODE_W / 2 + col * (NODE_W + 16);
  const yOf = (depth) => PAD_Y + NODE_W / 2 + depth * ROW_H;

  const highlightSet = new Set(highlightPath || []);

  const edges = [];
  const renderNodes = [];

  const walk = (node) => {
    if (!node) return;
    const pos = layout.positions.get(node.id);
    if (!pos) return;
    const visible = visibleNodeIds.has(node.id);
    const x = xOf(pos.col);
    const y = yOf(pos.depth);

    for (const [child, bit] of [
      [node.left, "0"],
      [node.right, "1"],
    ]) {
      if (!child) continue;
      const cpos = layout.positions.get(child.id);
      if (!cpos) continue;
      const cx = xOf(cpos.col);
      const cy = yOf(cpos.depth);
      const edgeVisible = visible && visibleNodeIds.has(child.id);
      const onPath = highlightSet.has(node.id) && highlightSet.has(child.id);
      edges.push({
        key: `${node.id}-${child.id}`,
        x1: x,
        y1: y,
        x2: cx,
        y2: cy,
        bit,
        visible: edgeVisible,
        onPath,
        labelX: (x + cx) / 2 + (bit === "0" ? -10 : 10),
        labelY: (y + cy) / 2,
      });
    }

    renderNodes.push({ node, x, y, visible, pos });
    walk(node.left);
    walk(node.right);
  };
  walk(root);

  return (
    <div style={{ overflowX: "auto", paddingBottom: 4 }}>
      <svg
        width={width}
        height={height}
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label="Huffman tree. Leaves carry symbols and codewords; edges are labelled 0 for left and 1 for right."
        style={{ display: "block", margin: "0 auto", maxWidth: "100%" }}
      >
        {edges.map((e) => (
          <g key={e.key} style={{ opacity: e.visible ? 1 : 0, transition: reduce ? "none" : `opacity 360ms ${EASE}` }}>
            <line
              x1={e.x1}
              y1={e.y1}
              x2={e.x2}
              y2={e.y2}
              stroke={e.onPath ? C.accent : C.border}
              strokeWidth={e.onPath ? 3 : 1.5}
              strokeLinecap="round"
            />
            <text
              x={e.labelX}
              y={e.labelY}
              fontSize={12}
              fontWeight={700}
              fill={e.onPath ? C.accent : C.muted}
              textAnchor="middle"
              dominantBaseline="middle"
              fontFamily="'IBM Plex Mono', monospace"
            >
              {e.bit}
            </text>
          </g>
        ))}

        {renderNodes.map(({ node, x, y, visible }) => {
          const isLeaf = node.sym !== null;
          const onPath = highlightSet.has(node.id);
          const code = isLeaf ? codes.get(node.id) : null;
          return (
            <g
              key={node.id}
              style={{
                opacity: visible ? 1 : 0,
                transform: visible ? "translateY(0)" : "translateY(-8px)",
                transformBox: "fill-box",
                transition: reduce ? "none" : `opacity 360ms ${EASE}, transform 360ms ${EASE}`,
                cursor: isLeaf ? "pointer" : "default",
              }}
              onClick={isLeaf ? () => onPickLeaf(node.id) : undefined}
              role={isLeaf ? "button" : undefined}
              aria-label={isLeaf ? `Symbol ${symbolLabel(node.sym)}, code ${code}` : undefined}
            >
              {isLeaf ? (
                <rect
                  x={x - NODE_W / 2}
                  y={y - NODE_W / 2}
                  width={NODE_W}
                  height={NODE_W}
                  rx={9}
                  fill={onPath ? C.accent : C.accentSoft}
                  stroke={C.leaf}
                  strokeWidth={onPath ? 2.5 : 1.5}
                />
              ) : (
                <circle
                  cx={x}
                  cy={y}
                  r={NODE_W / 2 - 4}
                  fill={onPath ? "#e7f0f4" : "#fff"}
                  stroke={C.branch}
                  strokeWidth={onPath ? 2.5 : 1.5}
                />
              )}
              {isLeaf ? (
                <>
                  <text
                    x={x}
                    y={y - 6}
                    fontSize={node.sym && node.sym.length > 2 ? 10 : 14}
                    fontWeight={700}
                    fill={onPath ? "#fff" : C.leaf}
                    textAnchor="middle"
                    dominantBaseline="middle"
                    fontFamily="'IBM Plex Mono', monospace"
                  >
                    {symbolLabel(node.sym)}
                  </text>
                  <text
                    x={x}
                    y={y + 10}
                    fontSize={9}
                    fill={onPath ? "#fff" : C.muted}
                    textAnchor="middle"
                    dominantBaseline="middle"
                    fontFamily="'IBM Plex Mono', monospace"
                  >
                    {code}
                  </text>
                </>
              ) : (
                <text
                  x={x}
                  y={y}
                  fontSize={12}
                  fontWeight={700}
                  fill={C.branch}
                  textAnchor="middle"
                  dominantBaseline="middle"
                  fontFamily="'IBM Plex Mono', monospace"
                >
                  {node.freq}
                </text>
              )}
            </g>
          );
        })}
      </svg>
    </div>
  );
}

function Forest({ build, step, reduce }) {
  const { nodes, steps } = build;
  const current = step >= 0 && step < steps.length ? steps[step] : null;
  const forestIds = current
    ? current.forestBefore
    : steps.length > 0
      ? steps[steps.length - 1].forestAfter
      : [...nodes.keys()].filter((id) => {
          const n = nodes.get(id);
          return n && n.sym !== null;
        });

  const mergedSet = new Set(current ? current.merged : []);
  const items = forestIds
    .map((id) => nodes.get(id))
    .filter(Boolean)
    .sort((a, b) => (a.freq !== b.freq ? a.freq - b.freq : a.seq - b.seq));

  return (
    <div style={{ display: "flex", gap: 8, flexWrap: "wrap", alignItems: "flex-end", minHeight: 64 }}>
      {items.map((node) => {
        const merging = mergedSet.has(node.id);
        const isLeaf = node.sym !== null;
        return (
          <div
            key={node.id}
            style={{
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              gap: 4,
              padding: "8px 10px",
              borderRadius: 10,
              minWidth: 46,
              background: merging ? C.accent : isLeaf ? C.accentSoft : "#fff",
              border: `1.5px solid ${merging ? C.accent : isLeaf ? C.leaf : C.branch}`,
              transform: merging && !reduce ? "translateY(-4px)" : "translateY(0)",
              boxShadow: merging ? `0 6px 16px ${C.accent}33` : "0 1px 3px rgba(0,0,0,0.05)",
              transition: reduce ? "none" : `transform 240ms ${EASE}, background 240ms ${EASE}`,
            }}
          >
            <span
              style={{
                fontSize: isLeaf ? 14 : 11,
                fontWeight: 700,
                fontFamily: "'IBM Plex Mono', monospace",
                color: merging ? "#fff" : isLeaf ? C.leaf : C.branch,
              }}
            >
              {isLeaf ? symbolLabel(node.sym) : "•"}
            </span>
            <span
              style={{
                fontSize: 11,
                fontFamily: "'IBM Plex Mono', monospace",
                color: merging ? "#fff" : C.muted,
              }}
            >
              {node.freq}
            </span>
          </div>
        );
      })}
    </div>
  );
}

export default function App() {
  const reduce = usePrefersReducedMotion();
  const [text, setText] = useState(PRESETS[0].text);
  const [presetNote, setPresetNote] = useState(PRESETS[0].note);
  const [step, setStep] = useState(-1);
  const [playing, setPlaying] = useState(false);
  const [pickedLeaf, setPickedLeaf] = useState(null);
  const timer = useRef(null);

  const freqMap = useMemo(() => frequencies(text), [text]);
  const total = text.length;
  const build = useMemo(() => buildHuffman(freqMap), [freqMap]);
  const codes = useMemo(() => assignCodes(build.root), [build]);

  const totalSteps = build.steps.length;

  useEffect(() => {
    setStep(-1);
    setPlaying(false);
    setPickedLeaf(null);
  }, [text]);

  useEffect(() => {
    if (!playing) {
      clearInterval(timer.current);
      return;
    }
    timer.current = setInterval(() => {
      setStep((s) => {
        if (s >= totalSteps - 1) {
          setPlaying(false);
          return s;
        }
        return s + 1;
      });
    }, reduce ? 520 : 820);
    return () => clearInterval(timer.current);
  }, [playing, totalSteps, reduce]);

  useEffect(() => () => clearInterval(timer.current), []);

  const visibleNodeIds = useMemo(() => {
    const set = new Set();
    if (!build.root) return set;
    for (const [id, node] of build.nodes) {
      if (node.sym !== null) set.add(id);
    }
    const shown = Math.max(0, step + 1);
    for (let i = 0; i < shown && i < build.steps.length; i++) {
      set.add(build.steps[i].parent);
    }
    return set;
  }, [build, step]);

  const treeComplete = step >= totalSteps - 1;

  const sortedRows = useMemo(() => {
    const rows = [...freqMap.entries()].map(([sym, freq]) => {
      const node = [...build.nodes.values()].find((n) => n.sym === sym);
      const code = node ? codes.get(node.id) : "";
      return { sym, freq, code, nodeId: node ? node.id : null, len: code ? code.length : 0 };
    });
    rows.sort((a, b) => (b.freq !== a.freq ? b.freq - a.freq : a.sym.localeCompare(b.sym)));
    return rows;
  }, [freqMap, codes, build]);

  const stats = useMemo(() => {
    const alphabet = freqMap.size;
    const fixedWidth = alphabet <= 1 ? 1 : Math.ceil(Math.log2(alphabet));
    let encodedBits = 0;
    for (const row of sortedRows) encodedBits += row.freq * row.len;
    const fixedBits = total * fixedWidth;
    const asciiBits = total * 8;
    const H = entropyBits(freqMap, total);
    const avgLen = total > 0 ? encodedBits / total : 0;
    const ratio = encodedBits > 0 ? fixedBits / encodedBits : 0;
    return { alphabet, fixedWidth, encodedBits, fixedBits, asciiBits, H, avgLen, ratio };
  }, [freqMap, sortedRows, total]);

  const encodedString = useMemo(() => {
    if (!treeComplete || total === 0) return "";
    let out = "";
    for (const ch of text) {
      const node = [...build.nodes.values()].find((n) => n.sym === ch);
      out += node ? codes.get(node.id) : "";
    }
    return out;
  }, [text, build, codes, treeComplete, total]);

  const highlightPath = useMemo(() => {
    if (pickedLeaf === null || !build.root) return [];
    const path = [];
    const find = (node, trail) => {
      if (!node) return false;
      const next = [...trail, node.id];
      if (node.id === pickedLeaf) {
        path.push(...next);
        return true;
      }
      return find(node.left, next) || find(node.right, next);
    };
    find(build.root, []);
    return path;
  }, [pickedLeaf, build]);

  const pickedSummary = useMemo(() => {
    if (pickedLeaf === null) return null;
    const node = build.nodes.get(pickedLeaf);
    if (!node) return null;
    return { sym: node.sym, code: codes.get(node.id) };
  }, [pickedLeaf, build, codes]);

  function choosePreset(p) {
    setText(p.text);
    setPresetNote(p.note);
  }

  function stepForward() {
    setPlaying(false);
    setStep((s) => Math.min(totalSteps - 1, s + 1));
  }
  function stepBack() {
    setPlaying(false);
    setStep((s) => Math.max(-1, s - 1));
    setPickedLeaf(null);
  }
  function resetBuild() {
    setPlaying(false);
    setStep(-1);
    setPickedLeaf(null);
  }
  function skipToEnd() {
    setPlaying(false);
    setStep(totalSteps - 1);
  }

  const currentStep = step >= 0 && step < totalSteps ? build.steps[step] : null;
  const mergeCaption = currentStep
    ? currentStep.single
      ? "Only one symbol exists, so it becomes the root on its own with a single 1 bit code."
      : (() => {
          const a = build.nodes.get(currentStep.merged[0]);
          const b = build.nodes.get(currentStep.merged[1]);
          const parent = build.nodes.get(currentStep.parent);
          const lbl = (n) => (n.sym !== null ? `"${symbolLabel(n.sym)}"` : `subtree(${n.freq})`);
          return `Merge the two rarest nodes ${lbl(a)} (${a.freq}) and ${lbl(b)} (${b.freq}) into a parent of frequency ${parent.freq}.`;
        })()
    : step >= totalSteps && totalSteps > 0
      ? "The forest has collapsed to a single root. Every symbol now has a prefix-free codeword."
      : "Press Step or Play to start merging the two lowest-frequency nodes.";

  const progress = totalSteps === 0 ? 0 : (step + 1) / totalSteps;

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
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&display=swap');
        input[type="range"]:focus-visible { outline: 2px solid ${C.accent}; outline-offset: 3px; border-radius: 4px; }
        textarea:focus-visible { outline: 2px solid ${C.accent}; outline-offset: 1px; }
        button:focus-visible { outline: 2px solid ${C.accent}; outline-offset: 2px; }
        g[role="button"]:focus-visible { outline: 2px solid ${C.accent}; }
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
            Information Theory · Compression
          </div>
          <h1 style={{ fontSize: 25, fontWeight: 700, margin: 0, textWrap: "balance" }}>Huffman Coding</h1>
          <p
            style={{
              color: C.ink,
              fontSize: 13.5,
              margin: "8px 0 0",
              lineHeight: 1.65,
              maxWidth: "66ch",
            }}
          >
            Huffman coding assigns short binary codes to frequent symbols and long ones to rare
            symbols. It counts how often each character appears, then greedily merges the two
            least frequent nodes over and over until a single tree remains. Reading 0 for every
            left branch and 1 for every right branch gives each symbol a codeword that is
            prefix-free: no codeword is the start of another, so the bitstream decodes with no
            separators.
          </p>
        </div>

        <Card style={{ marginBottom: 16 }}>
          <label
            htmlFor="huffman-input"
            style={{ display: "block", fontSize: 12, color: C.muted, marginBottom: 8, fontWeight: 600 }}
          >
            Text to encode
          </label>
          <textarea
            id="huffman-input"
            value={text}
            onChange={(e) => {
              setText(e.target.value);
              setPresetNote("");
            }}
            rows={2}
            spellCheck={false}
            aria-label="Text to encode with Huffman coding"
            style={{
              width: "100%",
              boxSizing: "border-box",
              resize: "vertical",
              padding: "10px 12px",
              borderRadius: 8,
              border: `1px solid ${C.border}`,
              fontSize: 14,
              fontFamily: "'IBM Plex Mono', monospace",
              color: C.ink,
              background: "#fff",
              lineHeight: 1.5,
            }}
          />
          <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginTop: 12 }}>
            {PRESETS.map((p) => (
              <button
                key={p.id}
                onClick={() => choosePreset(p)}
                style={{
                  padding: "5px 12px",
                  borderRadius: 999,
                  border: `1px solid ${text === p.text ? C.accent : C.border}`,
                  background: text === p.text ? C.accentSoft : "#fff",
                  color: text === p.text ? C.accent : C.ink,
                  fontSize: 12,
                  fontWeight: text === p.text ? 700 : 500,
                  fontFamily: "inherit",
                  cursor: "pointer",
                  transition: `transform 140ms ${EASE}`,
                }}
                onPointerDown={(e) => (e.currentTarget.style.transform = "scale(0.96)")}
                onPointerUp={(e) => (e.currentTarget.style.transform = "scale(1)")}
                onPointerLeave={(e) => (e.currentTarget.style.transform = "scale(1)")}
              >
                {p.label}
              </button>
            ))}
          </div>
          {presetNote && (
            <p style={{ fontSize: 12, color: C.muted, lineHeight: 1.6, margin: "12px 0 0" }}>{presetNote}</p>
          )}
          {total === 0 && (
            <p style={{ fontSize: 12.5, color: C.accent, lineHeight: 1.6, margin: "12px 0 0", fontWeight: 600 }}>
              The input is empty, so there is nothing to encode. Type a string or pick a preset.
            </p>
          )}
        </Card>

        {total > 0 && (
          <>
            <Card style={{ marginBottom: 16 }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", flexWrap: "wrap", gap: 8 }}>
                <div style={{ fontSize: 14, fontWeight: 700 }}>Build the tree</div>
                <div style={{ fontSize: 12, color: C.muted, fontFamily: "'IBM Plex Mono', monospace" }}>
                  merge {Math.max(0, step + 1)} of {totalSteps}
                </div>
              </div>

              <div
                style={{
                  height: 4,
                  background: C.faint,
                  borderRadius: 4,
                  margin: "10px 0 16px",
                  overflow: "hidden",
                }}
              >
                <div
                  style={{
                    height: "100%",
                    width: `${progress * 100}%`,
                    background: C.accent,
                    borderRadius: 4,
                    transition: reduce ? "none" : `width 320ms ${EASE}`,
                  }}
                />
              </div>

              <div style={{ fontSize: 11, color: C.muted, marginBottom: 8, textTransform: "uppercase", letterSpacing: "0.06em", fontWeight: 700 }}>
                Priority queue (lowest frequency first)
              </div>
              <Forest build={build} step={step} reduce={reduce} />

              <div
                aria-live="polite"
                style={{
                  marginTop: 14,
                  background: currentStep ? C.accentSoft : C.faint,
                  border: `1px solid ${currentStep ? C.accent + "33" : C.border}`,
                  borderRadius: 10,
                  padding: "10px 14px",
                  fontSize: 12.5,
                  lineHeight: 1.55,
                  color: currentStep ? C.accent : C.muted,
                  minHeight: 40,
                }}
              >
                {mergeCaption}
              </div>

              <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginTop: 14 }}>
                <Btn onClick={stepBack} disabled={step < 0} ariaLabel="Step back one merge">
                  Back
                </Btn>
                <Btn onClick={stepForward} disabled={treeComplete} ariaLabel="Step forward one merge">
                  Step
                </Btn>
                <Btn
                  primary
                  onClick={() => setPlaying((p) => !p)}
                  disabled={treeComplete && !playing}
                  ariaLabel={playing ? "Pause the build animation" : "Play the build animation"}
                >
                  {playing ? "Pause" : step < 0 ? "Play" : "Resume"}
                </Btn>
                <Btn onClick={skipToEnd} disabled={treeComplete} ariaLabel="Skip to the finished tree">
                  Skip to end
                </Btn>
                <Btn onClick={resetBuild} disabled={step < 0} ariaLabel="Reset the build to the start">
                  Reset
                </Btn>
              </div>
            </Card>

            <Card style={{ marginBottom: 16 }}>
              <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 4 }}>The tree</div>
              <div style={{ fontSize: 12.5, color: C.muted, lineHeight: 1.6, marginBottom: 14 }}>
                {treeComplete
                  ? "Click any leaf to trace its codeword down from the root."
                  : "Internal nodes show their combined frequency. The tree fills in merge by merge."}
              </div>
              <HuffmanTree
                build={build}
                codes={codes}
                visibleNodeIds={visibleNodeIds}
                highlightPath={highlightPath}
                reduce={reduce}
                onPickLeaf={(id) => setPickedLeaf((cur) => (cur === id ? null : id))}
              />
              {pickedSummary && (
                <div
                  style={{
                    marginTop: 12,
                    textAlign: "center",
                    fontSize: 13,
                    color: C.accent,
                    fontFamily: "'IBM Plex Mono', monospace",
                  }}
                >
                  "{symbolLabel(pickedSummary.sym)}" = {pickedSummary.code} ({pickedSummary.code.length} bits)
                </div>
              )}
            </Card>

            <Card style={{ marginBottom: 16 }}>
              <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 12 }}>Code table</div>
              <div style={{ overflowX: "auto" }}>
                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
                  <thead>
                    <tr style={{ textAlign: "left", color: C.muted, fontSize: 11, textTransform: "uppercase", letterSpacing: "0.05em" }}>
                      <th style={{ padding: "6px 8px", fontWeight: 700 }}>Symbol</th>
                      <th style={{ padding: "6px 8px", fontWeight: 700, textAlign: "right" }}>Freq</th>
                      <th style={{ padding: "6px 8px", fontWeight: 700 }}>Codeword</th>
                      <th style={{ padding: "6px 8px", fontWeight: 700, textAlign: "right" }}>Bits</th>
                      <th style={{ padding: "6px 8px", fontWeight: 700, textAlign: "right" }}>Total</th>
                    </tr>
                  </thead>
                  <tbody>
                    {sortedRows.map((row) => {
                      const active = pickedLeaf !== null && pickedLeaf === row.nodeId;
                      return (
                        <tr
                          key={row.sym}
                          onClick={() => treeComplete && setPickedLeaf((cur) => (cur === row.nodeId ? null : row.nodeId))}
                          style={{
                            borderTop: `1px solid ${C.faint}`,
                            background: active ? C.accentSoft : "transparent",
                            cursor: treeComplete ? "pointer" : "default",
                          }}
                        >
                          <td style={{ padding: "6px 8px", fontFamily: "'IBM Plex Mono', monospace", fontWeight: 700 }}>
                            {symbolLabel(row.sym)}
                          </td>
                          <td style={{ padding: "6px 8px", textAlign: "right", color: C.muted }}>{row.freq}</td>
                          <td style={{ padding: "6px 8px", fontFamily: "'IBM Plex Mono', monospace", color: treeComplete ? C.accent : C.muted }}>
                            {treeComplete ? row.code : "…"}
                          </td>
                          <td style={{ padding: "6px 8px", textAlign: "right", color: C.muted }}>
                            {treeComplete ? row.len : "—"}
                          </td>
                          <td style={{ padding: "6px 8px", textAlign: "right", fontWeight: 600 }}>
                            {treeComplete ? row.freq * row.len : "—"}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
              {!treeComplete && (
                <p style={{ fontSize: 12, color: C.muted, margin: "12px 0 0" }}>
                  Finish building the tree to read off the codewords.
                </p>
              )}
            </Card>

            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))",
                gap: 12,
                marginBottom: 16,
              }}
            >
              <Stat
                label="Huffman size"
                value={treeComplete ? `${stats.encodedBits} bits` : "—"}
                sub={treeComplete ? `${stats.avgLen.toFixed(2)} bits/symbol` : "build the tree first"}
                color={C.accent}
                bg={C.accentSoft}
              />
              <Stat
                label="Fixed width baseline"
                value={`${stats.fixedBits} bits`}
                sub={`${stats.fixedWidth} bits/symbol, ${stats.alphabet} symbol alphabet`}
                color={C.branch}
                bg="#e9f1f4"
              />
              <Stat
                label="Compression vs fixed"
                value={treeComplete ? `${stats.ratio.toFixed(2)}x` : "—"}
                sub={treeComplete ? `saves ${Math.max(0, stats.fixedBits - stats.encodedBits)} bits` : ""}
                color={C.good}
                bg="#e7f1ea"
              />
              <Stat
                label="Source entropy"
                value={`${stats.H.toFixed(2)} bits`}
                sub="Shannon H, the theoretical floor"
                color={C.gold}
                bg={C.goldL}
              />
            </div>

            {treeComplete && (
              <Card style={{ marginBottom: 16 }}>
                <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 6 }}>Average length sits between H and H + 1</div>
                <div style={{ fontSize: 12.5, color: C.muted, lineHeight: 1.6, marginBottom: 14 }}>
                  Shannon's source coding theorem bounds any prefix-free code from below by the entropy
                  H. Huffman is optimal among such codes, and its average length L always satisfies
                  H &le; L &lt; H + 1. Per symbol grouping closes that gap; here it is one symbol at a time.
                </div>
                {(() => {
                  const lo = stats.H;
                  const hi = stats.H + 1;
                  const span = Math.max(hi, stats.avgLen, stats.fixedWidth) + 0.5;
                  const pct = (v) => `${(v / span) * 100}%`;
                  return (
                    <div style={{ position: "relative", height: 56, marginBottom: 8 }}>
                      <div
                        style={{
                          position: "absolute",
                          top: 22,
                          left: pct(lo),
                          width: `calc(${pct(hi)} - ${pct(lo)})`,
                          height: 12,
                          background: C.goldL,
                          border: `1px solid ${C.gold}55`,
                          borderRadius: 6,
                        }}
                      />
                      {[
                        { v: stats.H, label: `H = ${stats.H.toFixed(2)}`, color: C.gold, top: 0 },
                        { v: stats.avgLen, label: `L = ${stats.avgLen.toFixed(2)}`, color: C.accent, top: 38 },
                        { v: stats.fixedWidth, label: `fixed = ${stats.fixedWidth}`, color: C.branch, top: 38 },
                      ].map((m, i) => (
                        <div
                          key={i}
                          style={{
                            position: "absolute",
                            left: pct(m.v),
                            top: m.top,
                            transform: "translateX(-50%)",
                            fontSize: 10.5,
                            fontWeight: 700,
                            color: m.color,
                            fontFamily: "'IBM Plex Mono', monospace",
                            whiteSpace: "nowrap",
                          }}
                        >
                          <div style={{ width: 2, height: 18, background: m.color, margin: m.top === 0 ? "14px auto 0" : "0 auto 2px", order: m.top === 0 ? 1 : 0, display: "flex" }} />
                          <span>{m.label}</span>
                        </div>
                      ))}
                    </div>
                  );
                })()}
                <p style={{ fontSize: 12, color: C.muted, lineHeight: 1.6, margin: "8px 0 0" }}>
                  The shaded band runs from H to H + 1. The Huffman average L falls inside it. When the
                  distribution is uniform, L meets the fixed width baseline and entropy alike; when it is
                  skewed, L drops well below fixed width and hugs H.
                </p>
              </Card>
            )}

            {treeComplete && encodedString && (
              <Card style={{ marginBottom: 16 }}>
                <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 6 }}>Encoded bitstream</div>
                <div style={{ fontSize: 12.5, color: C.muted, lineHeight: 1.6, marginBottom: 12 }}>
                  Concatenating the codewords with no separators. It is uniquely decodable precisely
                  because the code is prefix-free: scan left to right and the first matching codeword is
                  always the right one.
                </div>
                <div
                  style={{
                    fontFamily: "'IBM Plex Mono', monospace",
                    fontSize: 12,
                    lineHeight: 1.7,
                    color: C.ink,
                    background: C.faint,
                    borderRadius: 8,
                    padding: "10px 12px",
                    wordBreak: "break-all",
                    maxHeight: 140,
                    overflowY: "auto",
                  }}
                >
                  {encodedString}
                </div>
                <div style={{ fontSize: 11.5, color: C.muted, marginTop: 8, fontFamily: "'IBM Plex Mono', monospace" }}>
                  {encodedString.length} bits versus {stats.asciiBits} bits at 8 bits per ASCII character
                </div>
              </Card>
            )}

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
                  <b>The greedy merge is optimal.</b> At each step the two least frequent symbols can be
                  placed as deepest siblings without loss, because any optimal tree can be rearranged so
                  they sit there. Merging them into one node of summed frequency and recursing on the
                  smaller problem builds the shortest possible average code.
                </p>
                <p style={{ margin: 0 }}>
                  <b>Prefix-free means uniquely decodable.</b> Every symbol lives at a leaf, so no
                  codeword is a prefix of another. A decoder walks the tree from the root, turning left on
                  0 and right on 1, and emits a symbol the instant it reaches a leaf. No commas, lengths,
                  or escape characters are needed in the stream.
                </p>
                <p style={{ margin: 0 }}>
                  <b>Entropy is the floor.</b> The source entropy H is the average information per symbol.
                  No prefix-free code can beat it, and Huffman gets within one bit of it. The gap shrinks
                  toward zero when probabilities are close to powers of one half, or when you Huffman-code
                  blocks of symbols at once.
                </p>
              </div>
            </Card>

            <div style={{ marginTop: 18, textAlign: "center", fontSize: 11, color: C.muted }}>
              Frequencies, tree, codewords, and entropy are computed live from your text. Ties break by
              insertion order so the tree is reproducible.
            </div>
          </>
        )}
      </div>
    </div>
  );
}
