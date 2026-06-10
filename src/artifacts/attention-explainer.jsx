import { useState, useEffect, useRef, useCallback } from "react";

export const meta = {
  title: "Attention, From the Ground Up",
  category: "Transformers",
  description:
    "The word bank means a riverside in one sentence and a lender in another, yet the model stores just one meaning for it. Attention fixes that by letting each word borrow meaning from its neighbors: drag the weights around, read a hand-built head doing grammar, then watch a real one learn where to look.",
  date: "2026-06-11",
  tags: ["attention", "transformers", "kv-cache", "softmax", "training"],
};

const MONO = "'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, monospace";
const SERIF = "Georgia, serif";
const EASE = "cubic-bezier(0.23, 1, 0.32, 1)";

const C = {
  bg: "#fafaf8",
  card: "#ffffff",
  border: "#ebebeb",
  ink: "#1a1a1a",
  muted: "#5f5a52",
  faint: "#9b958c",
  grid: "#eceae5",
  q: "#1d4ed8",
  k: "#15803d",
  v: "#b45309",
  pos: "#15803d",
  neg: "#b91c1c",
};

const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
const matVec = (M, x) => M.map((row) => dot(row, x));
const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));

function softmax(arr) {
  const m = Math.max(...arr);
  const e = arr.map((x) => Math.exp(x - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((x) => x / s);
}

function mulberry32(a) {
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function usePrefersReducedMotion() {
  const [reduce, setReduce] = useState(false);
  useEffect(() => {
    if (typeof window === "undefined" || !window.matchMedia) return;
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    const apply = () => setReduce(mq.matches);
    apply();
    mq.addEventListener("change", apply);
    return () => mq.removeEventListener("change", apply);
  }, []);
  return reduce;
}

function fmt(x, places = 2) {
  const v = Math.abs(x) < 5e-7 ? 0 : x;
  return v.toFixed(places);
}

function rgba(hex, a) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}

function heatColor(w) {
  const r = Math.round(255 - w * 180);
  const g = Math.round(255 - w * 160);
  const b = Math.round(255 - w * 60);
  return `rgb(${r},${g},${b})`;
}
const textOnHeat = (w) => (w > 0.45 ? "#fff" : C.ink);

const DIMS = ["noun", "verb", "adj", "animate", "edible", "definite"];
const SQRT_D = Math.sqrt(6);

const EMB = {
  the: [0, 0, 0, 0, 0, 1],
  cat: [1, 0, 0, 1, 0, 0],
  ate: [0, 1, 0, 0, 0, 0],
  fish: [1, 0, 0, 0.2, 1, 0],
  because: [0, 0, 0, 0, 0, 0],
  it: [0.6, 0, 0, 0, 0, 0.4],
  was: [0, 0.8, 0, 0, 0, 0],
  hungry: [0, 0, 1, 0.9, 0, 0],
  tasty: [0, 0, 1, 0, 0.9, 0],
};

const W_Q = [
  [0, 0.8, 1.0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0],
  [0, 0, -2.0, 0, 0, 0],
  [0, 0.6, 0, 2.0, 0, 0],
  [0, 0, 0, 0, 2.0, 0],
  [0, 0, 0, 0, 0, 0],
];
const W_K = [
  [1.0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0],
  [0, 0, 1.0, 0, 0, 0],
  [0, 0, 0, 2.0, 0, 0],
  [0, 0, 0, 0, 2.0, 0],
  [0, 0, 0, 0, 0, 0],
];
const W_V = [
  [1, 0, 0, 0, 0, 0],
  [0, 0.3, 0, 0, 0, 0],
  [0, 0, 0.3, 0, 0, 0],
  [0, 0, 0, 1, 0, 0],
  [0, 0, 0, 0, 1, 0],
  [0, 0, 0, 0, 0, 0.3],
];

const BASE_TOKENS = ["the", "cat", "ate", "the", "fish", "because", "it", "was"];
const sentence = (adj) => [...BASE_TOKENS, adj];

const qOf = (w) => matVec(W_Q, EMB[w]);
const kOf = (w) => matVec(W_K, EMB[w]);
const vOf = (w) => matVec(W_V, EMB[w]);

function attnRow(tokens, i, causal) {
  const q = qOf(tokens[i]);
  const scores = tokens.map((t, j) =>
    causal && j > i ? -Infinity : dot(q, kOf(t)) / SQRT_D
  );
  return softmax(scores);
}

function outputAt(tokens, i, causal) {
  const w = attnRow(tokens, i, causal);
  const o = [0, 0, 0, 0, 0, 0];
  tokens.forEach((t, j) => {
    const v = vOf(t);
    for (let d = 0; d < 6; d++) o[d] += w[j] * v[d];
  });
  return { weights: w, out: o };
}

const TOK_STYLE = {
  the: ["#ece5d6", "#7a6638"],
  cat: ["#d6ead9", "#1f6b35"],
  ate: ["#d4e2f7", "#1a3d8c"],
  fish: ["#e6d8f5", "#5a1a8c"],
  because: ["#ebebe7", "#6e6a62"],
  it: ["#ebebe7", "#6e6a62"],
  was: ["#ebebe7", "#6e6a62"],
  hungry: ["#f8dcd2", "#9c3318"],
  tasty: ["#f5e6c2", "#8a5a00"],
};

function Chip({ word, onClick, selected, dim, label, pressed }) {
  const [bg, fg] = TOK_STYLE[word] || ["#eee", "#555"];
  const inner = (
    <>
      {word}
      {label && (
        <span style={{ fontSize: 8, fontWeight: 400, opacity: 0.7, display: "block" }}>
          {label}
        </span>
      )}
    </>
  );
  const base = {
    background: bg,
    color: fg,
    borderRadius: 8,
    padding: "5px 9px",
    fontFamily: MONO,
    fontWeight: 700,
    fontSize: 12,
    lineHeight: 1.25,
    opacity: dim ? 0.3 : 1,
    boxShadow: selected ? `0 0 0 2px ${fg}` : "0 1px 3px rgba(0,0,0,0.08)",
    transform: selected ? "scale(1.06)" : "none",
    transition: `transform 160ms ${EASE}, box-shadow 160ms ${EASE}, opacity 240ms ${EASE}`,
    textAlign: "center",
    minWidth: 34,
  };
  if (!onClick) return <div style={base}>{inner}</div>;
  return (
    <button
      type="button"
      className="ae-press"
      onClick={onClick}
      aria-pressed={pressed ?? selected}
      style={{ ...base, border: "none", cursor: "pointer" }}
    >
      {inner}
    </button>
  );
}

function SectionLabel({ children }) {
  return (
    <div
      style={{
        fontSize: 10,
        letterSpacing: 1.5,
        textTransform: "uppercase",
        color: C.faint,
        fontFamily: MONO,
        marginBottom: 6,
      }}
    >
      {children}
    </div>
  );
}

function Prose({ children, style }) {
  return (
    <p style={{ fontSize: 13.5, lineHeight: 1.65, color: C.ink, margin: 0, fontFamily: SERIF, ...style }}>
      {children}
    </p>
  );
}

function InfoBox({ children, tone = "blue" }) {
  const tones = {
    blue: ["#eaf2fc", "#b9d3f2", "#1e3a5f"],
    green: ["#eef8ef", "#b5dfba", "#1d4d28"],
    amber: ["#fdf6e0", "#ecd28a", "#6b4e08"],
  };
  const [bg, bd, fg] = tones[tone];
  return (
    <div
      style={{
        background: bg,
        border: `1px solid ${bd}`,
        borderRadius: 10,
        padding: "10px 14px",
        fontSize: 12.5,
        color: fg,
        lineHeight: 1.6,
        fontFamily: SERIF,
      }}
    >
      {children}
    </div>
  );
}

function pillStyle(active) {
  return {
    flex: 1,
    padding: "8px 10px",
    borderRadius: 9,
    border: `1.5px solid ${active ? C.ink : "#ddd"}`,
    background: active ? C.ink : "#fff",
    color: active ? "#fff" : "#888",
    fontFamily: MONO,
    fontSize: 11,
    cursor: "pointer",
    transition: `background 160ms ${EASE}, color 160ms ${EASE}, border-color 160ms ${EASE}`,
  };
}

function NamedRow({ label, vec, color, max = 2, faded }) {
  return (
    <>
      <div
        style={{
          fontFamily: MONO,
          fontSize: 9.5,
          color: faded ? C.faint : color,
          fontWeight: 700,
          display: "flex",
          alignItems: "center",
          paddingRight: 6,
          whiteSpace: "nowrap",
        }}
      >
        {label}
      </div>
      {vec.map((x, d) => {
        const a = clamp(Math.abs(x) / max, 0, 1) * (faded ? 0.35 : 0.8);
        return (
          <div
            key={d}
            style={{
              background: x >= 0 ? rgba(color, a) : rgba(C.neg, a),
              borderRadius: 5,
              padding: "5px 2px",
              textAlign: "center",
              fontFamily: MONO,
              fontSize: 10,
              color: a > 0.45 ? "#fff" : C.ink,
              fontVariantNumeric: "tabular-nums",
              transition: `background 240ms ${EASE}`,
            }}
          >
            {fmt(x, 1)}
          </div>
        );
      })}
    </>
  );
}

function DimGrid({ children, extraCol }) {
  return (
    <div style={{ overflowX: "auto" }}>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: `minmax(72px, auto) repeat(6, minmax(44px, 1fr))${extraCol ? " minmax(58px, auto)" : ""}`,
          gap: 4,
          minWidth: 380,
          alignItems: "stretch",
        }}
      >
        <div />
        {DIMS.map((d) => (
          <div
            key={d}
            style={{ fontFamily: MONO, fontSize: 8.5, color: C.faint, textAlign: "center", letterSpacing: 0.4 }}
          >
            {d}
          </div>
        ))}
        {extraCol && (
          <div style={{ fontFamily: MONO, fontSize: 8.5, color: C.faint, textAlign: "center" }}>{extraCol}</div>
        )}
        {children}
      </div>
    </div>
  );
}

const MAP_WORDS = [
  { w: "river", p: [-2.8, 1.6], c: "river" },
  { w: "water", p: [-3.3, 0.55], c: "river" },
  { w: "shore", p: [-2.1, 0.75], c: "river" },
  { w: "stream", p: [-2.9, 2.2], c: "river" },
  { w: "loan", p: [2.6, 1.55], c: "money" },
  { w: "cash", p: [3.25, 0.6], c: "money" },
  { w: "rates", p: [2.2, 0.65], c: "money" },
  { w: "teller", p: [3.0, 2.15], c: "money" },
];
const BANK_E = [0.05, 1.15];
const SENTS = {
  river: { parts: ["we sat on the ", "bank", " of the ", "river", ", near the ", "water", ""], ctx: ["river", "water"] },
  money: { parts: ["the ", "bank", " raised the ", "rates", " on the ", "loan", ""], ctx: ["rates", "loan"] },
};
const CLUSTER_COL = { river: "#1d6fa3", money: "#2e7d46" };

const mX = (x) => 260 + x * 58;
const mY = (y) => 258 - y * 82;

function StageProblem({ reduce }) {
  const [sent, setSent] = useState("river");
  const [t, setT] = useState(0);
  const ctx = SENTS[sent].ctx;
  const ctxVecs = ctx.map((w) => MAP_WORDS.find((m) => m.w === w).p);
  const mean = [
    ctxVecs.reduce((s, p) => s + p[0], 0) / ctxVecs.length,
    ctxVecs.reduce((s, p) => s + p[1], 0) / ctxVecs.length,
  ];
  const bp = [
    (1 - t) * BANK_E[0] + t * mean[0],
    (1 - t) * BANK_E[1] + t * mean[1],
  ];
  const moveT = reduce ? "none" : `transform 480ms ${EASE}`;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      <Prose>
        Two sentences share the word <strong>bank</strong>. Below is a small map of meaning space: words
        about rivers cluster on the left, words about money on the right. Every word gets one fixed
        embedding, the list of numbers the model stores for it, so <strong>bank</strong> is stuck at a
        single point between the clusters no matter which sentence it appears in. Toggle the sentence
        and watch nothing happen. That is the problem.
      </Prose>

      <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
        {["river", "money"].map((s) => (
          <button
            key={s}
            type="button"
            className="ae-press"
            onClick={() => setSent(s)}
            aria-pressed={sent === s}
            style={{ ...pillStyle(sent === s), minWidth: 200, fontFamily: SERIF, fontSize: 12 }}
          >
            {SENTS[s].parts.map((part, i) =>
              i % 2 === 1 ? (
                <strong key={i} style={{ fontFamily: MONO, fontSize: 11 }}>
                  {part}
                </strong>
              ) : (
                <span key={i}>{part}</span>
              )
            )}
          </button>
        ))}
      </div>

      <svg
        viewBox="0 0 520 300"
        style={{ width: "100%", height: "auto", background: C.bg, borderRadius: 12, border: `1px solid ${C.border}`, display: "block" }}
        role="img"
        aria-label={`Semantic map. River words cluster left, money words right. The bank point sits at (${fmt(bp[0])}, ${fmt(bp[1])}), pulled ${Math.round(t * 100)} percent toward the ${sent} cluster.`}
      >
        <ellipse cx={mX(-2.78)} cy={mY(1.28)} rx={96} ry={104} fill={rgba(CLUSTER_COL.river, 0.07)} />
        <ellipse cx={mX(2.76)} cy={mY(1.24)} rx={96} ry={104} fill={rgba(CLUSTER_COL.money, 0.07)} />
        <text x={mX(-2.78)} y={mY(2.85)} textAnchor="middle" fontFamily={MONO} fontSize="9" fill={CLUSTER_COL.river}>
          river things
        </text>
        <text x={mX(2.76)} y={mY(2.85)} textAnchor="middle" fontFamily={MONO} fontSize="9" fill={CLUSTER_COL.money}>
          money things
        </text>
        {MAP_WORDS.map((m) => {
          const inCtx = ctx.includes(m.w);
          return (
            <g key={m.w}>
              <circle
                cx={mX(m.p[0])}
                cy={mY(m.p[1])}
                r={inCtx ? 6 : 4.5}
                fill={CLUSTER_COL[m.c]}
                stroke={inCtx ? C.ink : "none"}
                strokeWidth={inCtx ? 1.5 : 0}
              />
              <text
                x={mX(m.p[0]) + 8}
                y={mY(m.p[1]) + 3}
                fontFamily={MONO}
                fontSize="10"
                fontWeight={inCtx ? 700 : 400}
                fill={inCtx ? C.ink : C.muted}
              >
                {m.w}
              </text>
            </g>
          );
        })}
        <line
          x1={mX(BANK_E[0])}
          y1={mY(BANK_E[1])}
          x2={mX(bp[0])}
          y2={mY(bp[1])}
          stroke={C.faint}
          strokeWidth="1.2"
          strokeDasharray="4 4"
        />
        <circle cx={mX(BANK_E[0])} cy={mY(BANK_E[1])} r="5" fill="none" stroke={C.faint} strokeWidth="1.5" strokeDasharray="3 2" />
        <text x={mX(BANK_E[0])} y={mY(BANK_E[1]) - 11} textAnchor="middle" fontFamily={MONO} fontSize="9" fill={C.faint}>
          bank (static)
        </text>
        <g style={{ transform: `translate(${mX(bp[0])}px, ${mY(bp[1])}px)`, transition: moveT }}>
          <circle r="7" fill={C.ink} />
          <circle r="11" fill="none" stroke={C.ink} strokeWidth="1" opacity="0.25" />
          <text x="0" y="24" textAnchor="middle" fontFamily={MONO} fontSize="11" fontWeight="700" fill={C.ink}>
            bank&#8242;
          </text>
        </g>
      </svg>

      <div>
        <label
          htmlFor="ae-borrow"
          style={{ fontFamily: MONO, fontSize: 11, color: C.muted, display: "block", marginBottom: 6 }}
        >
          borrow from context: t = {t.toFixed(2)}
        </label>
        <input
          id="ae-borrow"
          type="range"
          min="0"
          max="100"
          value={Math.round(t * 100)}
          onChange={(e) => setT(Number(e.target.value) / 100)}
          style={{ width: "100%", accentColor: C.ink }}
          aria-label="How much bank borrows from its context words"
        />
      </div>

      <div
        style={{
          background: "#f5f5f0",
          borderRadius: 9,
          padding: "9px 12px",
          fontFamily: MONO,
          fontSize: 11,
          color: C.muted,
          lineHeight: 1.7,
          overflowX: "auto",
          whiteSpace: "nowrap",
        }}
      >
        bank&#8242; = (1&#8722;t)&#183;bank + t&#183;mean({ctx.join(", ")})
        <br />
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; = {fmt(1 - t)}&#183;({fmt(BANK_E[0])}, {fmt(BANK_E[1])}) + {fmt(t)}&#183;(
        {fmt(mean[0])}, {fmt(mean[1])}) = <strong style={{ color: C.ink }}>({fmt(bp[0])}, {fmt(bp[1])})</strong>
      </div>

      <InfoBox tone="green">
        Drag the slider and the dot lands in the right cluster, computed as a plain average of the
        context words. Attention&apos;s output is exactly this: a weighted average of the other tokens&apos;
        vectors (a token is roughly one word, and its vector is that embedding, the same list of
        numbers). Everything that follows, the queries, the keys, the softmax, exists to answer one
        question: what should the weights be?
      </InfoBox>
    </div>
  );
}

const CTX2 = [
  { w: "sun", k: [1.45, 0.5], v: [1.8, 1.55], col: "#b45309" },
  { w: "rain", k: [-0.45, 1.5], v: [-1.75, 1.15], col: "#0e7490" },
  { w: "snow", k: [-1.4, -0.55], v: [-1.15, -1.7], col: "#7c3aed" },
  { w: "wind", k: [0.7, -1.35], v: [1.55, -1.25], col: "#15803d" },
];
const P_SIZE = 340;
const P_UNIT = 68;
const pX = (x) => P_SIZE / 2 + x * P_UNIT;
const pY = (y) => P_SIZE / 2 - y * P_UNIT;

function ArrowSvg({ to, color, width = 2.5 }) {
  const tx = pX(to[0]);
  const ty = pY(to[1]);
  const ang = Math.atan2(ty - pY(0), tx - pX(0));
  const h = 8;
  return (
    <g>
      <line x1={pX(0)} y1={pY(0)} x2={tx} y2={ty} stroke={color} strokeWidth={width} />
      <polygon
        points={`${tx},${ty} ${tx - h * Math.cos(ang - 0.42)},${ty - h * Math.sin(ang - 0.42)} ${tx - h * Math.cos(ang + 0.42)},${ty - h * Math.sin(ang + 0.42)}`}
        fill={color}
      />
    </g>
  );
}

function StagePlayground() {
  const [q, setQ] = useState([1.0, 0.85]);
  const [scaled, setScaled] = useState(true);
  const [dragging, setDragging] = useState(false);
  const svgRef = useRef(null);

  const div = scaled ? Math.SQRT2 : 1;
  const scores = CTX2.map((c) => dot(q, c.k) / div);
  const weights = softmax(scores);
  const out = CTX2.reduce((acc, c, j) => [acc[0] + weights[j] * c.v[0], acc[1] + weights[j] * c.v[1]], [0, 0]);
  const qLen = Math.hypot(q[0], q[1]);
  const qu = qLen > 1e-9 ? [q[0] / qLen, q[1] / qLen] : [1, 0];
  const maxW = Math.max(...weights);

  const toWorld = (e) => {
    const rect = svgRef.current.getBoundingClientRect();
    const sx = P_SIZE / rect.width;
    const px = (e.clientX - rect.left) * sx;
    const py = (e.clientY - rect.top) * sx;
    return [(px - P_SIZE / 2) / P_UNIT, (P_SIZE / 2 - py) / P_UNIT];
  };

  const setQClamped = (p) => {
    const len = Math.hypot(p[0], p[1]);
    if (len < 0.15) return;
    if (len > 2.3) {
      setQ([(p[0] / len) * 2.3, (p[1] / len) * 2.3]);
    } else {
      setQ(p);
    }
  };

  const onDown = (e) => {
    const [mx, my] = toWorld(e);
    if (Math.hypot(mx - q[0], my - q[1]) < 34 / P_UNIT) {
      e.currentTarget.setPointerCapture(e.pointerId);
      setDragging(true);
      setQClamped([mx, my]);
    }
  };
  const onMove = (e) => {
    if (dragging) setQClamped(toWorld(e));
  };
  const onUp = (e) => {
    if (!dragging) return;
    setDragging(false);
    try {
      e.currentTarget.releasePointerCapture(e.pointerId);
    } catch {
      return;
    }
  };
  const onKey = (e) => {
    const step = 0.12;
    const moves = { ArrowUp: [0, step], ArrowDown: [0, -step], ArrowLeft: [-step, 0], ArrowRight: [step, 0] };
    if (moves[e.key]) {
      e.preventDefault();
      setQClamped([q[0] + moves[e.key][0], q[1] + moves[e.key][1]]);
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      <Prose>
        So how relevant is each word? Attention measures it by direction. Each context word puts up a
        key arrow advertising itself, and you hold a query arrow saying what you want. The score is
        their dot product, a single number that is large when they point the same way, near zero at
        right angles, negative when opposed. Softmax then squashes the scores into percentages that
        sum to 100, and those weights blend the value points, what each word contributes, into the
        output. Drag the tip of{" "}
        <strong style={{ fontFamily: MONO, color: C.q }}>q</strong> and steer the blend.
      </Prose>

      <div style={{ display: "flex", gap: 14, flexWrap: "wrap", alignItems: "flex-start", justifyContent: "center" }}>
        <svg
          ref={svgRef}
          viewBox={`0 0 ${P_SIZE} ${P_SIZE}`}
          tabIndex={0}
          onPointerDown={onDown}
          onPointerMove={onMove}
          onPointerUp={onUp}
          onPointerCancel={onUp}
          onKeyDown={onKey}
          style={{
            width: P_SIZE,
            maxWidth: "100%",
            height: "auto",
            background: C.bg,
            borderRadius: 12,
            border: `1px solid ${C.border}`,
            touchAction: "none",
            cursor: dragging ? "grabbing" : "grab",
            display: "block",
          }}
          role="img"
          aria-label={`Attention playground. Query points to (${fmt(q[0])}, ${fmt(q[1])}). Strongest match is ${CTX2[weights.indexOf(maxW)].w} at ${Math.round(maxW * 100)} percent. Drag the query tip or use arrow keys.`}
        >
          <line x1={pX(-2.4)} y1={pY(0)} x2={pX(2.4)} y2={pY(0)} stroke={C.grid} strokeWidth="1" />
          <line x1={pX(0)} y1={pY(-2.4)} x2={pX(0)} y2={pY(2.4)} stroke={C.grid} strokeWidth="1" />
          <line
            x1={pX(-qu[0] * 2.35)}
            y1={pY(-qu[1] * 2.35)}
            x2={pX(qu[0] * 2.35)}
            y2={pY(qu[1] * 2.35)}
            stroke={rgba(C.ink, 0.14)}
            strokeWidth="1"
            strokeDasharray="3 4"
          />
          {CTX2.map((c, j) => {
            const tproj = dot(c.k, qu);
            const foot = [qu[0] * tproj, qu[1] * tproj];
            return (
              <g key={c.w}>
                <line
                  x1={pX(c.k[0])}
                  y1={pY(c.k[1])}
                  x2={pX(foot[0])}
                  y2={pY(foot[1])}
                  stroke={c.col}
                  strokeWidth="1"
                  strokeDasharray="2 3"
                  opacity="0.5"
                />
                <circle cx={pX(foot[0])} cy={pY(foot[1])} r="2.5" fill={c.col} opacity="0.6" />
                <line
                  x1={pX(out[0])}
                  y1={pY(out[1])}
                  x2={pX(c.v[0])}
                  y2={pY(c.v[1])}
                  stroke={c.col}
                  strokeWidth={1 + weights[j] * 4}
                  opacity={0.12 + weights[j] * 0.7}
                />
                <ArrowSvg to={c.k} color={c.col} width={2.2} />
                <text
                  x={pX(c.k[0]) + (c.k[0] >= 0 ? 7 : -7)}
                  y={pY(c.k[1]) - 6}
                  textAnchor={c.k[0] >= 0 ? "start" : "end"}
                  fontFamily={MONO}
                  fontSize="10"
                  fontWeight="700"
                  fill={c.col}
                >
                  k {c.w}
                </text>
                <rect
                  x={pX(c.v[0]) - 5}
                  y={pY(c.v[1]) - 5}
                  width="10"
                  height="10"
                  transform={`rotate(45 ${pX(c.v[0])} ${pY(c.v[1])})`}
                  fill={rgba(c.col, 0.85)}
                />
                <text
                  x={pX(c.v[0]) + (c.v[0] >= 0 ? 9 : -9)}
                  y={pY(c.v[1]) + 11}
                  textAnchor={c.v[0] >= 0 ? "start" : "end"}
                  fontFamily={MONO}
                  fontSize="9"
                  fill={c.col}
                >
                  v {c.w}
                </text>
              </g>
            );
          })}
          <circle cx={pX(out[0])} cy={pY(out[1])} r="7" fill={C.ink} />
          <circle cx={pX(out[0])} cy={pY(out[1])} r="11" fill="none" stroke={C.ink} strokeWidth="1" opacity="0.3" />
          <text x={pX(out[0])} y={pY(out[1]) + 23} textAnchor="middle" fontFamily={MONO} fontSize="10" fontWeight="700" fill={C.ink}>
            output
          </text>
          <ArrowSvg to={q} color={C.q} width={3.2} />
          <circle cx={pX(q[0])} cy={pY(q[1])} r="9" fill="#fff" stroke={C.q} strokeWidth="2.5" />
          <text x={pX(q[0]) + 12} y={pY(q[1]) - 10} fontFamily={MONO} fontSize="12" fontWeight="700" fill={C.q}>
            q
          </text>
        </svg>

        <div style={{ flex: "1 1 200px", minWidth: 196 }}>
          <SectionLabel>score &#8594; softmax weight</SectionLabel>
          <div style={{ display: "flex", flexDirection: "column", gap: 7 }}>
            {CTX2.map((c, j) => (
              <div key={c.w}>
                <div style={{ display: "flex", justifyContent: "space-between", fontFamily: MONO, fontSize: 10.5, marginBottom: 3 }}>
                  <span style={{ color: c.col, fontWeight: 700 }}>{c.w}</span>
                  <span style={{ color: C.muted }}>
                    q&#183;k = {fmt(dot(q, c.k))}
                    {scaled ? ` /√2 = ${fmt(scores[j])}` : ""}
                  </span>
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                  <div style={{ flex: 1, background: "#efefea", borderRadius: 5, height: 12, overflow: "hidden" }}>
                    <div
                      style={{
                        width: `${weights[j] * 100}%`,
                        height: "100%",
                        background: c.col,
                        borderRadius: 5,
                        transition: `width 120ms ${EASE}`,
                      }}
                    />
                  </div>
                  <span style={{ fontFamily: MONO, fontSize: 10.5, color: C.ink, width: 36, textAlign: "right" }}>
                    {(weights[j] * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
            ))}
          </div>

          <div style={{ marginTop: 12, fontFamily: MONO, fontSize: 10.5, color: C.muted, lineHeight: 1.8 }}>
            |q| = {fmt(qLen)} &nbsp; max weight = {(maxW * 100).toFixed(0)}%
            <br />
            output = ({fmt(out[0])}, {fmt(out[1])})
          </div>

          <button
            type="button"
            className="ae-press"
            onClick={() => setScaled((s) => !s)}
            aria-pressed={scaled}
            style={{ ...pillStyle(scaled), marginTop: 10, width: "100%", flex: "none" }}
          >
            divide scores by &#8730;d {scaled ? "(on)" : "(off)"}
          </button>
        </div>
      </div>

      <InfoBox tone="amber">
        Steer q toward one key and the dashed line through q tilts, the perpendicular feet of the keys
        slide along it, and the output glides around inside the value points. Now drag q as long as it
        goes: the scores stretch with it and softmax collapses toward one-hot, all the weight piling
        onto a single winner. Turn the &#8730;d division off and the collapse comes even sooner. Real
        attention divides by &#8730;d&#8342; (d&#8342; is just how many numbers each vector holds)
        precisely to keep softmax in this soft, blendable regime.
      </InfoBox>
    </div>
  );
}

const QK_ROUTES = [
  ["adj → seeks noun (+1.0)", "an adjective goes looking for a noun"],
  ["verb → seeks noun, animate (+0.8, +0.6)", "a verb looks back for its subject"],
  ["animate → seeks animate (+2.0)", "a word about living things seeks living things"],
  ["edible → seeks edible (+2.0)", "a word about food seeks food"],
  ["adj → avoids adj (−2.0)", "an adjective is not looking for another adjective"],
];

function StageQKV() {
  const tokens = sentence("hungry");
  const [qi, setQi] = useState(8);
  const [ki, setKi] = useState(1);
  const qw = tokens[qi];
  const kw = tokens[ki];
  const q = qOf(qw);
  const k = kOf(kw);
  const prod = q.map((x, d) => x * k[d]);
  const sum = prod.reduce((a, b) => a + b, 0);
  const kv = vOf(kw);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      <Prose>
        Why not compare raw embeddings directly? Because raw similarity is symmetric: each word would
        mostly match itself and its synonyms, and <strong>ate</strong> could never seek{" "}
        <strong>cat</strong>. So attention multiplies each embedding by three learned grids of numbers,
        the weight matrices W_Q, W_K, W_V, giving it three roles: a query (what I am looking for), a
        key (what I advertise), and a value (what I hand over if picked). It is a hash map lookup where
        every key matches a little instead of exactly one. Here is one attention head, the unit that
        does this, small enough to read: six dimensions, each a named slot in the vector, and matrices
        that encode a few grammar rules:
      </Prose>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: 6 }}>
        {QK_ROUTES.map(([rule, why]) => (
          <div key={rule} style={{ background: "#f6f6f1", borderRadius: 8, padding: "7px 10px" }}>
            <div style={{ fontFamily: MONO, fontSize: 10, color: C.ink, fontWeight: 700 }}>{rule}</div>
            <div style={{ fontFamily: SERIF, fontSize: 11.5, color: C.muted }}>{why}</div>
          </div>
        ))}
      </div>

      <div>
        <SectionLabel>pick a query (who is asking)</SectionLabel>
        <div style={{ display: "flex", gap: 5, flexWrap: "wrap" }}>
          {tokens.map((t, i) => (
            <Chip key={i} word={t} onClick={() => setQi(i)} selected={qi === i} />
          ))}
        </div>
      </div>
      <div>
        <SectionLabel>and a key (who is being scored)</SectionLabel>
        <div style={{ display: "flex", gap: 5, flexWrap: "wrap" }}>
          {tokens.map((t, i) => (
            <Chip key={i} word={t} onClick={() => setKi(i)} selected={ki === i} />
          ))}
        </div>
      </div>

      <div>
        <SectionLabel>the dot product, dimension by dimension</SectionLabel>
        <DimGrid extraCol="sum">
          <NamedRow label={`e ${qw}`} vec={EMB[qw]} color={C.q} max={1} faded />
          <div />
          <NamedRow label={`e ${kw}`} vec={EMB[kw]} color={C.k} max={1} faded />
          <div />
          <NamedRow label="q = W_Q e" vec={q} color={C.q} />
          <div />
          <NamedRow label="k = W_K e" vec={k} color={C.k} />
          <div />
          <NamedRow label="q &#215; k" vec={prod} color={C.v} max={4} />
          <div
            style={{
              fontFamily: MONO,
              fontSize: 11,
              fontWeight: 700,
              color: C.ink,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              background: "#f6f6f1",
              borderRadius: 5,
            }}
          >
            {fmt(sum)}
          </div>
        </DimGrid>
        <div style={{ fontFamily: MONO, fontSize: 11, color: C.muted, marginTop: 8 }}>
          score({qw}, {kw}) = {fmt(sum)} / &#8730;6 = <strong style={{ color: C.ink }}>{fmt(sum / SQRT_D)}</strong>
        </div>
      </div>

      <InfoBox>
        Try query <strong>hungry</strong> against key <strong>cat</strong>: nearly the whole score
        comes from one cell, animate &#215; animate. Hungry&apos;s query carries &quot;animate&quot;
        because hungriness is a property of living things, and cat&apos;s key advertises it. Against
        key <strong>hungry</strong> itself, the adj &#215; adj cell goes negative and pushes the match
        down. Every score has a cell you can point at.
      </InfoBox>

      <div>
        <SectionLabel>and the value: what &quot;{kw}&quot; hands over if picked</SectionLabel>
        <DimGrid>
          <NamedRow label={`e ${kw}`} vec={EMB[kw]} color={C.faint} max={1} faded />
          <NamedRow label="v = W_V e" vec={kv} color={C.v} max={1} />
        </DimGrid>
        <Prose style={{ fontSize: 12, color: C.muted, marginTop: 8 }}>
          W_V passes content (noun, animate, edible) through at full strength and damps the grammar
          features to 0.3. The matched token hands over a selected payload, not its whole self.
        </Prose>
      </div>
    </div>
  );
}

function StageLinguistics() {
  const [adj, setAdj] = useState("hungry");
  const tokens = sentence(adj);
  const last = tokens.length - 1;
  const { weights, out } = outputAt(tokens, last, true);
  const ranked = weights.map((w, j) => ({ j, w })).sort((a, b) => b.w - a.w);
  const winner = ranked[0];
  const runner = ranked[1];
  const vCat = vOf("cat");
  const vFish = vOf("fish");
  const dist = (v) => Math.sqrt(out.reduce((s, x, d) => s + (x - v[d]) ** 2, 0));
  const dCat = dist(vCat);
  const dFish = dist(vFish);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      <Prose>
        Now the payoff. Same head, full sentence, computed live all the way through. The final word
        asks the question; swap it and watch where the head looks.
      </Prose>

      <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
        {["hungry", "tasty"].map((a) => (
          <button
            key={a}
            type="button"
            className="ae-press"
            onClick={() => setAdj(a)}
            aria-pressed={adj === a}
            style={{ ...pillStyle(adj === a), minWidth: 220, fontFamily: SERIF, fontSize: 12 }}
          >
            the cat ate the fish because it was <strong style={{ fontFamily: MONO, fontSize: 11 }}>{a}</strong>
          </button>
        ))}
      </div>

      <div>
        <SectionLabel>where &quot;{adj}&quot; attends (it can only see earlier words and itself)</SectionLabel>
        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
          {tokens.map((t, j) => (
            <div key={j} style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <div style={{ width: 70, flexShrink: 0, display: "flex" }}>
                <Chip word={t} selected={j === winner.j} />
              </div>
              <div style={{ flex: 1, background: "#efefea", borderRadius: 5, height: 14, overflow: "hidden" }}>
                <div
                  style={{
                    width: `${weights[j] * 100}%`,
                    height: "100%",
                    borderRadius: 5,
                    background: j === winner.j ? C.ink : C.faint,
                    transition: `width 360ms ${EASE}, background 360ms ${EASE}`,
                  }}
                />
              </div>
              <span
                style={{
                  fontFamily: MONO,
                  fontSize: 10.5,
                  color: j === winner.j ? C.ink : C.muted,
                  width: 42,
                  textAlign: "right",
                  fontVariantNumeric: "tabular-nums",
                }}
              >
                {(weights[j] * 100).toFixed(1)}%
              </span>
            </div>
          ))}
        </div>
        <div style={{ fontFamily: MONO, fontSize: 11, color: C.muted, marginTop: 8 }}>
          winner <strong style={{ color: C.ink }}>{tokens[winner.j]}</strong> at {(winner.w * 100).toFixed(1)}%,
          runner-up {tokens[runner.j]} at {(runner.w * 100).toFixed(1)}% ({(winner.w / runner.w).toFixed(1)}&#215;)
        </div>
      </div>

      <div>
        <SectionLabel>the output vector moves with it</SectionLabel>
        <DimGrid>
          <NamedRow label={`output ${adj}`} vec={out} color={C.q} max={1} />
          <NamedRow label="v cat" vec={vCat} color={C.k} max={1} faded={dCat > dFish} />
          <NamedRow label="v fish" vec={vFish} color={C.v} max={1} faded={dFish > dCat} />
        </DimGrid>
        <Prose style={{ fontSize: 12, color: C.muted, marginTop: 8 }}>
          The output now sits closest to <strong>{dCat < dFish ? "cat" : "fish"}</strong>&apos;s value
          (distance {fmt(Math.min(dCat, dFish))} vs {fmt(Math.max(dCat, dFish))}). With{" "}
          <strong>hungry</strong> the animate dimension dominates; with <strong>tasty</strong> the
          edible dimension takes over.
        </Prose>
      </div>

      <InfoBox tone="amber">
        This head was designed by hand so every number is readable. Real heads learn patterns exactly
        like this from data, which is stage 7, and a full model stacks many heads across many layers.
        Resolving what &quot;it&quot; refers to in this sentence is the same kind of match, performed
        across those layers.
      </InfoBox>
    </div>
  );
}

function StageHeatmap() {
  const [adj, setAdj] = useState("hungry");
  const [causal, setCausal] = useState(true);
  const [hover, setHover] = useState(null);
  const tokens = sentence(adj);
  const matrix = tokens.map((_, i) => attnRow(tokens, i, causal));

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      <Prose>
        Every row of the grid below is one token&apos;s attention, computed through the same head.
        During generation a model writes left to right, so while it is producing word <em>i</em> the
        later words do not exist yet. The causal mask enforces that: every cell above the diagonal is
        set to &#8722;&#8734; before softmax, which softmax turns into a weight of exactly 0%. Rows
        always sum to 100%.
      </Prose>

      <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
        <button type="button" className="ae-press" onClick={() => setCausal(false)} aria-pressed={!causal} style={pillStyle(!causal)}>
          no mask (see everything)
        </button>
        <button type="button" className="ae-press" onClick={() => setCausal(true)} aria-pressed={causal} style={pillStyle(causal)}>
          causal mask &#9699;
        </button>
        <button
          type="button"
          className="ae-press"
          onClick={() => setAdj(adj === "hungry" ? "tasty" : "hungry")}
          style={{ ...pillStyle(false), flex: "none" }}
          aria-label={`Switch final word to ${adj === "hungry" ? "tasty" : "hungry"}`}
        >
          last word: {adj}
        </button>
      </div>

      <div style={{ overflowX: "auto", paddingTop: 26 }}>
        <div style={{ display: "inline-flex", flexDirection: "column" }}>
          <div style={{ display: "flex", marginLeft: 60 }}>
            {tokens.map((t, j) => (
              <div key={j} style={{ width: 44, textAlign: "center", fontSize: 8.5, color: C.faint, fontFamily: MONO, paddingBottom: 4 }}>
                {t}
              </div>
            ))}
          </div>
          {matrix.map((row, i) => (
            <div key={i} style={{ display: "flex", alignItems: "center" }}>
              <div
                style={{
                  width: 60,
                  fontSize: 9,
                  color: TOK_STYLE[tokens[i]][1],
                  fontFamily: MONO,
                  fontWeight: 700,
                  textAlign: "right",
                  paddingRight: 7,
                }}
              >
                {tokens[i]}
              </div>
              {row.map((w, j) => {
                const masked = causal && j > i;
                return (
                  <div
                    key={j}
                    onMouseEnter={() => setHover([i, j])}
                    onMouseLeave={() => setHover(null)}
                    style={{
                      width: 44,
                      height: 34,
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      background: masked ? "#f4f4f1" : heatColor(w),
                      color: masked ? "#ccc" : textOnHeat(w),
                      fontFamily: MONO,
                      fontSize: 9,
                      border: "1px solid #fff",
                      position: "relative",
                      transition: `background 320ms ${EASE}`,
                    }}
                  >
                    {masked ? "—" : `${Math.round(w * 100)}`}
                    {hover && hover[0] === i && hover[1] === j && !masked && (
                      <div
                        style={{
                          position: "absolute",
                          top: -26,
                          left: "50%",
                          transform: "translateX(-50%)",
                          background: C.ink,
                          color: "#fff",
                          borderRadius: 4,
                          padding: "3px 7px",
                          fontSize: 9,
                          whiteSpace: "nowrap",
                          zIndex: 2,
                          fontFamily: MONO,
                          pointerEvents: "none",
                        }}
                      >
                        {tokens[i]} &#8594; {tokens[j]}: {(w * 100).toFixed(1)}%
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          ))}
        </div>
      </div>

      <InfoBox>
        Read the stories in the rows: <strong>ate</strong> leans on <strong>cat</strong>, its subject;
        the function words spread evenly because their queries match nothing in particular; and the
        bottom row is stage 4&apos;s flip, {adj} finding {adj === "hungry" ? "cat" : "fish"}. Cell
        numbers are percentages of each row.
      </InfoBox>
    </div>
  );
}

const KV_TOKENS = ["the", "cat", "ate", "the", "fish"];
const KV_K = KV_TOKENS.map((t) => kOf(t));
const KV_V = KV_TOKENS.map((t) => vOf(t));

function CachePanel({ title, rows, step }) {
  return (
    <div style={{ background: "#f8f8f5", borderRadius: 10, padding: 12, border: `1px solid ${C.border}` }}>
      <SectionLabel>{title}</SectionLabel>
      {KV_TOKENS.slice(0, step + 1).map((t, i) => (
        <div
          key={i}
          style={{
            display: "flex",
            alignItems: "center",
            gap: 6,
            marginBottom: 4,
            background: i === step ? "#fff3cd" : rgba(TOK_STYLE[t][1], 0.1),
            borderRadius: 6,
            padding: "4px 8px",
            border: i === step ? "1px solid #fbbf24" : "1px solid transparent",
            transition: `background 280ms ${EASE}`,
          }}
        >
          <span style={{ fontFamily: MONO, fontSize: 10, color: TOK_STYLE[t][1], fontWeight: 700, width: 32 }}>{t}</span>
          {rows[i].map((x, d) => (
            <div
              key={d}
              title={x.toFixed(2)}
              style={{
                flex: 1,
                height: 8,
                borderRadius: 2,
                background: i === step ? "#fbbf24" : rgba(TOK_STYLE[t][1], 0.4),
                opacity: 0.35 + clamp(Math.abs(x), 0, 1) * 0.65,
              }}
            />
          ))}
          <span style={{ fontSize: 8, color: i === step ? "#92400e" : C.faint, fontFamily: MONO, width: 26 }}>
            {i === step ? "NEW" : "HIT"}
          </span>
        </div>
      ))}
    </div>
  );
}

function StageKVCache() {
  const [step, setStep] = useState(0);
  const recomputeWithout = (n) => n * n;
  const recomputeWith = (n) => n;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      <Prose>
        Generation produces one token at a time, and every new token attends over all the old ones.
        But the keys and values of old tokens never change, so there is no reason to recompute them.
        The KV cache is memoization for attention: store K and V once, and each step computes them
        only for the single new token.
      </Prose>

      <div style={{ display: "flex", gap: 8, justifyContent: "center", flexWrap: "wrap" }}>
        {KV_TOKENS.map((t, i) => (
          <Chip key={i} word={t} selected={i === step} label={i < step ? "cached" : i === step ? "new" : ""} dim={i > step} />
        ))}
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: 8 }}>
        <CachePanel title="K cache (keys)" rows={KV_K} step={step} />
        <CachePanel title="V cache (values)" rows={KV_V} step={step} />
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: 8 }}>
        <div style={{ background: "#eef8ef", border: "1px solid #b5dfba", borderRadius: 9, padding: "10px 12px" }}>
          <div style={{ fontSize: 10, color: "#1d4d28", fontFamily: MONO, marginBottom: 4 }}>with cache</div>
          <div style={{ fontSize: 20, fontFamily: MONO, fontWeight: 700, color: C.pos }}>
            {recomputeWith(step + 1)} op{recomputeWith(step + 1) !== 1 ? "s" : ""}
          </div>
          <div style={{ fontSize: 10.5, color: "#1d4d28", fontFamily: SERIF }}>compute K, V for 1 new token only</div>
        </div>
        <div style={{ background: "#fdf0ef", border: "1px solid #f0bcb6", borderRadius: 9, padding: "10px 12px" }}>
          <div style={{ fontSize: 10, color: "#7e2417", fontFamily: MONO, marginBottom: 4 }}>without cache</div>
          <div style={{ fontSize: 20, fontFamily: MONO, fontWeight: 700, color: C.neg }}>{recomputeWithout(step + 1)} ops</div>
          <div style={{ fontSize: 10.5, color: "#7e2417", fontFamily: SERIF }}>
            recompute all {step + 1} token{step ? "s" : ""} every step
          </div>
        </div>
      </div>

      <InfoBox tone="amber">
        The cache grows by 2 &#215; d&#8342; floats per token per layer per head. At 32k context, 32
        layers, 32 heads, d&#8342; = 128 in fp16 (two bytes per float), that is roughly 8&#8202;GB
        just for KV. This is why eviction schemes like EVOKE, which drop entries and recompute them
        when needed, matter.
      </InfoBox>

      <div style={{ display: "flex", gap: 10, justifyContent: "center" }}>
        <button
          type="button"
          className="ae-press"
          onClick={() => setStep((s) => Math.max(0, s - 1))}
          disabled={step === 0}
          style={{
            background: "#fff",
            border: "1px solid #ddd",
            borderRadius: 9,
            padding: "8px 18px",
            cursor: step === 0 ? "not-allowed" : "pointer",
            fontFamily: MONO,
            fontSize: 11.5,
            opacity: step === 0 ? 0.4 : 1,
            color: C.muted,
          }}
        >
          &#8592; prev token
        </button>
        <button
          type="button"
          className="ae-press"
          onClick={() => setStep((s) => Math.min(KV_TOKENS.length - 1, s + 1))}
          disabled={step === KV_TOKENS.length - 1}
          style={{
            background: C.ink,
            color: "#fff",
            border: "none",
            borderRadius: 9,
            padding: "8px 18px",
            cursor: step === KV_TOKENS.length - 1 ? "not-allowed" : "pointer",
            fontFamily: MONO,
            fontSize: 11.5,
            opacity: step === KV_TOKENS.length - 1 ? 0.4 : 1,
          }}
        >
          generate next &#8594;
        </button>
      </div>
    </div>
  );
}

const TR = { NK: 6, NV: 6, DM: 12, DK: 8, BATCH: 32, LR: 0.002, MAX: 1500, CHUNK: 5, EVAL_EVERY: 25 };
const KEY_SYMS = ["A", "B", "C", "D", "E", "F"];
const VAL_SYMS = ["1", "2", "3", "4", "5", "6"];
const PROBE = { ks: [0, 2, 4], vs: [1, 3, 5] };

function trInit(rnd) {
  const mat = (r, c) => Array.from({ length: r }, () => Array.from({ length: c }, () => (rnd() * 2 - 1) * 0.4));
  return { Wq: mat(TR.DM, TR.DK), Wk: mat(TR.DM, TR.DK), Wv: mat(TR.DM, TR.DK), Wo: mat(TR.DK, TR.NV) };
}
function trZeros(P) {
  const z = {};
  for (const k in P) z[k] = P[k].map((r) => r.map(() => 0));
  return z;
}
function trEmbedPair(k, v) {
  const x = new Array(TR.DM).fill(0);
  x[k] = 1;
  x[6 + v] = 1;
  return x;
}
function trEmbedQuery(k) {
  const x = new Array(TR.DM).fill(0);
  x[k] = 1;
  return x;
}
function trSample(rnd) {
  const keys = [0, 1, 2, 3, 4, 5];
  for (let i = 5; i > 0; i--) {
    const j = Math.floor(rnd() * (i + 1));
    [keys[i], keys[j]] = [keys[j], keys[i]];
  }
  const ks = keys.slice(0, 3);
  const vs = ks.map(() => Math.floor(rnd() * TR.NV));
  const qi = Math.floor(rnd() * 3);
  return { ks, vs, qk: ks[qi], y: vs[qi] };
}
function trVecMat(v, M) {
  const out = new Array(M[0].length).fill(0);
  for (let i = 0; i < v.length; i++) {
    if (v[i] === 0) continue;
    const row = M[i];
    for (let j = 0; j < row.length; j++) out[j] += v[i] * row[j];
  }
  return out;
}
function trForward(P, ex) {
  const xs = ex.ks.map((k, i) => trEmbedPair(k, ex.vs[i]));
  const xq = trEmbedQuery(ex.qk);
  const q = trVecMat(xq, P.Wq);
  const K = xs.map((x) => trVecMat(x, P.Wk));
  const V = xs.map((x) => trVecMat(x, P.Wv));
  const s = K.map((k) => dot(q, k) / Math.sqrt(TR.DK));
  const a = softmax(s);
  const o = new Array(TR.DK).fill(0);
  for (let j = 0; j < 3; j++) for (let d = 0; d < TR.DK; d++) o[d] += a[j] * V[j][d];
  const z = trVecMat(o, P.Wo);
  const p = softmax(z);
  return { xs, xq, q, K, V, a, o, z, p, loss: -Math.log(p[ex.y] + 1e-12) };
}
function trBackward(P, ex, f, G) {
  const dz = f.p.slice();
  dz[ex.y] -= 1;
  for (let d = 0; d < TR.DK; d++) for (let c = 0; c < TR.NV; c++) G.Wo[d][c] += f.o[d] * dz[c];
  const dO = new Array(TR.DK).fill(0);
  for (let d = 0; d < TR.DK; d++) for (let c = 0; c < TR.NV; c++) dO[d] += P.Wo[d][c] * dz[c];
  const da = f.V.map((v) => dot(dO, v));
  let inner = 0;
  for (let j = 0; j < 3; j++) inner += f.a[j] * da[j];
  const ds = f.a.map((aj, j) => aj * (da[j] - inner));
  const dq = new Array(TR.DK).fill(0);
  for (let j = 0; j < 3; j++) {
    const cj = ds[j] / Math.sqrt(TR.DK);
    for (let d = 0; d < TR.DK; d++) dq[d] += cj * f.K[j][d];
    for (let i = 0; i < TR.DM; i++) {
      const xi = f.xs[j][i];
      if (xi === 0) continue;
      for (let d = 0; d < TR.DK; d++) {
        G.Wk[i][d] += xi * cj * f.q[d];
        G.Wv[i][d] += xi * f.a[j] * dO[d];
      }
    }
  }
  for (let i = 0; i < TR.DM; i++) {
    const xi = f.xq[i];
    if (xi === 0) continue;
    for (let d = 0; d < TR.DK; d++) G.Wq[i][d] += xi * dq[d];
  }
}
function trAdamStep(P, G, st) {
  st.t++;
  const b1 = 0.9;
  const b2 = 0.999;
  const eps = 1e-8;
  const bc1 = 1 - Math.pow(b1, st.t);
  const bc2 = 1 - Math.pow(b2, st.t);
  for (const k in P) {
    const M = P[k];
    const Gm = G[k];
    const Mm = st.m[k];
    const Mv = st.v[k];
    for (let i = 0; i < M.length; i++) {
      for (let j = 0; j < M[0].length; j++) {
        const g = Gm[i][j];
        Mm[i][j] = b1 * Mm[i][j] + (1 - b1) * g;
        Mv[i][j] = b2 * Mv[i][j] + (1 - b2) * g * g;
        M[i][j] -= (TR.LR * (Mm[i][j] / bc1)) / (Math.sqrt(Mv[i][j] / bc2) + eps);
      }
    }
  }
}
function trEvalAcc(P, seed, n = 200) {
  const rnd = mulberry32(seed);
  let c = 0;
  for (let i = 0; i < n; i++) {
    const ex = trSample(rnd);
    const f = trForward(P, ex);
    let best = 0;
    for (let j = 1; j < TR.NV; j++) if (f.z[j] > f.z[best]) best = j;
    if (best === ex.y) c++;
  }
  return c / n;
}
function trProbe(P) {
  return PROBE.ks.map((qk) => trForward(P, { ks: PROBE.ks, vs: PROBE.vs, qk, y: 0 }).a);
}

function LossChart({ lossHist, accHist, maxLoss }) {
  const W = 520;
  const H = 150;
  const padL = 8;
  const padR = 8;
  const padT = 10;
  const padB = 16;
  const x = (s) => padL + (s / TR.MAX) * (W - padL - padR);
  const yLoss = (l) => padT + (1 - clamp(l / maxLoss, 0, 1)) * (H - padT - padB);
  const yAcc = (a) => padT + (1 - a) * (H - padT - padB);
  const stride = Math.max(1, Math.floor(lossHist.length / 280));
  const lossPath = lossHist
    .filter((_, i) => i % stride === 0 || i === lossHist.length - 1)
    .map((p, i) => `${i === 0 ? "M" : "L"}${x(p.s).toFixed(1)},${yLoss(p.l).toFixed(1)}`)
    .join("");
  const accPath = accHist.map((p, i) => `${i === 0 ? "M" : "L"}${x(p.s).toFixed(1)},${yAcc(p.a).toFixed(1)}`).join("");
  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      style={{ width: "100%", height: "auto", background: C.bg, borderRadius: 10, border: `1px solid ${C.border}`, display: "block" }}
      role="img"
      aria-label={`Training curves. ${lossHist.length ? `Loss ${lossHist[lossHist.length - 1].l.toFixed(2)}.` : "Not started."} ${accHist.length ? `Accuracy ${(accHist[accHist.length - 1].a * 100).toFixed(0)} percent.` : ""}`}
    >
      <line x1={padL} y1={yAcc(0.98)} x2={W - padR} y2={yAcc(0.98)} stroke={C.pos} strokeWidth="1" strokeDasharray="3 4" opacity="0.5" />
      <text x={W - padR - 2} y={yAcc(0.98) - 4} textAnchor="end" fontFamily={MONO} fontSize="8.5" fill={C.pos}>
        98% accuracy
      </text>
      {lossPath && <path d={lossPath} fill="none" stroke={C.neg} strokeWidth="1.8" />}
      {accPath && <path d={accPath} fill="none" stroke={C.pos} strokeWidth="1.8" />}
      <text x={padL + 2} y={H - 4} fontFamily={MONO} fontSize="8.5" fill={C.faint}>
        0
      </text>
      <text x={W - padR - 2} y={H - 4} textAnchor="end" fontFamily={MONO} fontSize="8.5" fill={C.faint}>
        {TR.MAX} steps
      </text>
      <text x={padL + 2} y={padT + 8} fontFamily={MONO} fontSize="8.5" fill={C.neg}>
        loss
      </text>
    </svg>
  );
}

function StageTraining({ reduce }) {
  const [seed, setSeed] = useState(1);
  const [running, setRunning] = useState(false);
  const [, setTick] = useState(0);
  const ref = useRef(null);

  const reset = useCallback((sd) => {
    const rnd = mulberry32(sd);
    const P = trInit(rnd);
    ref.current = {
      P,
      adam: { m: trZeros(P), v: trZeros(P), t: 0 },
      rnd,
      step: 0,
      ema: null,
      lossHist: [],
      accHist: [{ s: 0, a: trEvalAcc(P, sd * 7 + 1234) }],
      probe: trProbe(P),
    };
  }, []);

  if (ref.current === null) reset(seed);

  const runSteps = useCallback(
    (n) => {
      const m = ref.current;
      for (let i = 0; i < n && m.step < TR.MAX; i++) {
        const G = trZeros(m.P);
        let loss = 0;
        for (let b = 0; b < TR.BATCH; b++) {
          const ex = trSample(m.rnd);
          const f = trForward(m.P, ex);
          loss += f.loss;
          trBackward(m.P, ex, f, G);
        }
        loss /= TR.BATCH;
        for (const k in G) for (const r of G[k]) for (let j = 0; j < r.length; j++) r[j] /= TR.BATCH;
        trAdamStep(m.P, G, m.adam);
        m.step++;
        m.ema = m.ema === null ? loss : 0.92 * m.ema + 0.08 * loss;
        m.lossHist.push({ s: m.step, l: m.ema });
        if (m.step % TR.EVAL_EVERY === 0) m.accHist.push({ s: m.step, a: trEvalAcc(m.P, seed * 7 + 1234) });
      }
      m.probe = trProbe(m.P);
    },
    [seed]
  );

  useEffect(() => {
    if (!running) return;
    let raf;
    const frame = () => {
      runSteps(TR.CHUNK);
      setTick((t) => t + 1);
      if (ref.current.step >= TR.MAX) {
        setRunning(false);
        return;
      }
      raf = requestAnimationFrame(frame);
    };
    raf = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(raf);
  }, [running, runSteps]);

  const instant = () => {
    setRunning(false);
    runSteps(TR.MAX - ref.current.step);
    setTick((t) => t + 1);
  };
  const doReset = () => {
    setRunning(false);
    const next = seed + 1;
    setSeed(next);
    reset(next);
    setTick((t) => t + 1);
  };
  const onPlay = () => {
    if (reduce) {
      instant();
      return;
    }
    setRunning((r) => !r);
  };

  const m = ref.current;
  const acc = m.accHist[m.accHist.length - 1].a;
  const loss = m.lossHist.length ? m.lossHist[m.lossHist.length - 1].l : null;
  const done = m.step >= TR.MAX;
  const maxLoss = m.lossHist.length ? Math.max(2, m.lossHist[0].l) : 2;

  const statBox = {
    background: "#f6f6f1",
    borderRadius: 8,
    padding: "7px 10px",
    fontFamily: MONO,
    fontSize: 10.5,
    color: C.muted,
    minWidth: 86,
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      <Prose>
        The head in stages 3 to 5 was designed. Real ones are learned, and this one learns right here.
        The task: each example shows three key&#8594;value pairs, then asks for the value of one of
        the keys. Predicting it requires attending to the right pair; nothing else says where to look.
        One attention head, weights initialized at random, trains in your browser by gradient descent:
        every step measures the loss, a single number scoring how wrong the guesses are, then nudges
        each weight slightly in the direction that shrinks it.
      </Prose>

      <div style={{ display: "flex", gap: 6, alignItems: "center", flexWrap: "wrap", fontFamily: MONO, fontSize: 11, color: C.muted }}>
        <span>probe example:</span>
        {PROBE.ks.map((k, i) => (
          <span key={i} style={{ background: "#f0f0eb", borderRadius: 6, padding: "3px 8px", color: C.ink, fontWeight: 700 }}>
            {KEY_SYMS[k]}&#8594;{VAL_SYMS[PROBE.vs[i]]}
          </span>
        ))}
        <span>then ask each key</span>
      </div>

      <div style={{ display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center" }}>
        <button
          type="button"
          className="ae-press"
          onClick={onPlay}
          disabled={done && !reduce}
          style={{
            background: C.ink,
            color: "#fff",
            border: "none",
            borderRadius: 9,
            padding: "9px 22px",
            fontFamily: MONO,
            fontSize: 12,
            cursor: done && !reduce ? "not-allowed" : "pointer",
            opacity: done && !reduce ? 0.4 : 1,
          }}
          aria-label={running ? "Pause training" : "Start training"}
        >
          {reduce ? "train (instant)" : running ? "pause" : m.step > 0 && !done ? "resume" : "train"}
        </button>
        {!reduce && (
          <button
            type="button"
            className="ae-press"
            onClick={instant}
            disabled={done}
            style={{
              background: "#fff",
              border: "1px solid #ddd",
              borderRadius: 9,
              padding: "9px 14px",
              fontFamily: MONO,
              fontSize: 11,
              color: C.muted,
              cursor: done ? "not-allowed" : "pointer",
              opacity: done ? 0.4 : 1,
            }}
            aria-label="Run all remaining training steps instantly"
          >
            instant
          </button>
        )}
        <button
          type="button"
          className="ae-press"
          onClick={doReset}
          style={{
            background: "#fff",
            border: "1px solid #ddd",
            borderRadius: 9,
            padding: "9px 14px",
            fontFamily: MONO,
            fontSize: 11,
            color: C.muted,
            cursor: "pointer",
          }}
          aria-label="Reset weights with a new random seed"
        >
          reset (seed {seed})
        </button>
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginLeft: "auto" }}>
          <div style={statBox}>
            step <strong style={{ color: C.ink }}>{m.step}</strong>
          </div>
          <div style={statBox}>
            loss <strong style={{ color: C.neg }}>{loss === null ? "—" : loss.toFixed(3)}</strong>
          </div>
          <div style={statBox}>
            acc <strong style={{ color: C.pos }}>{(acc * 100).toFixed(1)}%</strong>
          </div>
        </div>
      </div>

      <LossChart lossHist={m.lossHist} accHist={m.accHist} maxLoss={maxLoss} />

      <div>
        <SectionLabel>attention on the probe (row = which key is asked)</SectionLabel>
        <div style={{ display: "inline-flex", flexDirection: "column" }}>
          <div style={{ display: "flex", marginLeft: 70 }}>
            {PROBE.ks.map((k, i) => (
              <div key={i} style={{ width: 72, textAlign: "center", fontFamily: MONO, fontSize: 9, color: C.faint, paddingBottom: 4 }}>
                {KEY_SYMS[k]}&#8594;{VAL_SYMS[PROBE.vs[i]]}
              </div>
            ))}
          </div>
          {m.probe.map((row, i) => (
            <div key={i} style={{ display: "flex", alignItems: "center" }}>
              <div style={{ width: 70, fontFamily: MONO, fontSize: 9.5, color: C.muted, textAlign: "right", paddingRight: 8 }}>
                ask {KEY_SYMS[PROBE.ks[i]]}?
              </div>
              {row.map((w, j) => (
                <div
                  key={j}
                  style={{
                    width: 72,
                    height: 36,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    background: heatColor(w),
                    color: textOnHeat(w),
                    fontFamily: MONO,
                    fontSize: 10,
                    border: "1px solid #fff",
                  }}
                >
                  {(w * 100).toFixed(0)}%
                </div>
              ))}
            </div>
          ))}
        </div>
        <Prose style={{ fontSize: 12, color: C.muted, marginTop: 8 }}>
          At step 0 the rows are mush, roughly a third each. As loss falls the diagonal sharpens:
          asking for a key makes the head attend to that key&apos;s pair, because that is the only
          place the answer lives.
        </Prose>
      </div>

      <InfoBox tone="green">
        Nobody told this head where to look. W_Q, W_K and W_V started as random noise, and the lookup
        pattern emerged because predicting the value requires it. That is exactly how the matrices in
        a real transformer get their meaning, just at a much larger scale.
      </InfoBox>
    </div>
  );
}

const STAGES = [
  { id: "problem", title: "1. The Problem", subtitle: "one word, two meanings", Comp: StageProblem },
  { id: "scores", title: "2. Scores", subtitle: "direction match, then blend", Comp: StagePlayground },
  { id: "qkv", title: "3. Q K V", subtitle: "three jobs, one embedding", Comp: StageQKV },
  { id: "head", title: "4. One Head", subtitle: "grammar you can compute", Comp: StageLinguistics },
  { id: "mask", title: "5. The Mask", subtitle: "no peeking ahead", Comp: StageHeatmap },
  { id: "kvcache", title: "6. KV Cache", subtitle: "old tokens never change", Comp: StageKVCache },
  { id: "learn", title: "7. Learning", subtitle: "watch a head find the pattern", Comp: StageTraining },
];

export default function App() {
  const [stage, setStage] = useState(0);
  const reduce = usePrefersReducedMotion();
  const { Comp, title, subtitle } = STAGES[stage];

  const css = `
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&display=swap');
    .ae-root :focus-visible { outline: 2px solid ${C.q}; outline-offset: 2px; border-radius: 8px; }
    .ae-press:active { transform: scale(0.97); }
    @media (prefers-reduced-motion: reduce) {
      .ae-root * { transition-duration: 0.001ms !important; animation-duration: 0.001ms !important; }
    }
  `;

  return (
    <div
      className="ae-root"
      style={{
        minHeight: "100vh",
        background: C.bg,
        fontFamily: SERIF,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        padding: "28px 12px 48px",
        color: C.ink,
      }}
    >
      <style>{css}</style>

      <div style={{ textAlign: "center", marginBottom: 22, maxWidth: 580 }}>
        <h1 style={{ fontSize: 23, fontWeight: 400, margin: 0, textWrap: "balance" }}>Attention, From the Ground Up</h1>
        <div style={{ fontSize: 12.5, color: C.muted, marginTop: 6, lineHeight: 1.5 }}>
          Start with the problem attention solves, end by watching attention learn to solve it.
        </div>
      </div>

      <div style={{ display: "flex", gap: 5, marginBottom: 20, overflowX: "auto", maxWidth: "100%", paddingBottom: 4 }}>
        {STAGES.map((s, i) => (
          <button
            key={s.id}
            type="button"
            className="ae-press"
            onClick={() => setStage(i)}
            aria-pressed={stage === i}
            style={{
              background: stage === i ? C.ink : "#fff",
              color: stage === i ? "#fff" : "#888",
              border: `1px solid ${stage === i ? C.ink : "#ddd"}`,
              borderRadius: 20,
              padding: "5px 12px",
              fontFamily: MONO,
              fontSize: 10,
              cursor: "pointer",
              whiteSpace: "nowrap",
              transition: `background 160ms ${EASE}, color 160ms ${EASE}`,
            }}
          >
            {s.title}
          </button>
        ))}
      </div>

      <div
        style={{
          width: "100%",
          maxWidth: 580,
          background: C.card,
          border: `1px solid ${C.border}`,
          borderRadius: 16,
          padding: "22px 18px",
          boxShadow: "0 2px 24px rgba(0,0,0,0.05)",
        }}
      >
        <div style={{ marginBottom: 16 }}>
          <div
            style={{
              fontSize: 10,
              color: C.faint,
              letterSpacing: 1.5,
              textTransform: "uppercase",
              marginBottom: 3,
              fontFamily: MONO,
            }}
          >
            {subtitle}
          </div>
          <h2 style={{ fontSize: 17, margin: 0, fontWeight: 500 }}>{title}</h2>
        </div>
        <div style={{ borderTop: "1px solid #f0f0f0", paddingTop: 16 }}>
          <Comp reduce={reduce} />
        </div>
      </div>

      <div style={{ display: "flex", gap: 10, marginTop: 18 }}>
        <button
          type="button"
          className="ae-press"
          onClick={() => setStage((s) => Math.max(0, s - 1))}
          disabled={stage === 0}
          style={{
            background: "#fff",
            border: "1px solid #ddd",
            borderRadius: 10,
            padding: "9px 20px",
            fontFamily: SERIF,
            fontSize: 13,
            cursor: stage === 0 ? "not-allowed" : "pointer",
            color: C.muted,
            opacity: stage === 0 ? 0.4 : 1,
          }}
        >
          &#8592; back
        </button>
        <button
          type="button"
          className="ae-press"
          onClick={() => setStage((s) => Math.min(STAGES.length - 1, s + 1))}
          disabled={stage === STAGES.length - 1}
          style={{
            background: C.ink,
            color: "#fff",
            border: "none",
            borderRadius: 10,
            padding: "9px 24px",
            fontFamily: SERIF,
            fontSize: 13,
            cursor: stage === STAGES.length - 1 ? "not-allowed" : "pointer",
            opacity: stage === STAGES.length - 1 ? 0.4 : 1,
          }}
        >
          next &#8594;
        </button>
      </div>
    </div>
  );
}
