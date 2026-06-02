import { useState, useEffect, useRef, useMemo } from "react";

export const meta = {
  title: "Backpropagation, by Hand",
  category: "Machine Learning",
  description:
    "A tiny network learns XOR while you watch. Step the forward pass neuron by neuron, then trace the gradient backward edge by edge as the chain rule assembles itself.",
  date: "2026-02-09",
  tags: ["backpropagation", "neural-networks", "gradients", "autodiff"],
};

const C = {
  bg: "#f7f5f2",
  card: "#ffffff",
  border: "#e7e1d8",
  ink: "#1b1916",
  muted: "#857c70",
  accent: "#c0561f",
  accentSoft: "#f6e9e0",
  pos: "#2f7d54",
  posSoft: "#e6f1ea",
  neg: "#b23b5e",
  negSoft: "#f7e6ec",
  wire: "#cfc7ba",
};

const EASE = "cubic-bezier(0.23, 1, 0.32, 1)";

const tanh = (x) => Math.tanh(x);
const dtanh = (y) => 1 - y * y;
const sigmoid = (x) => 1 / (1 + Math.exp(-x));
const dsigmoid = (y) => y * (1 - y);
const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));

const DATA = [
  { x: [0, 0], y: 0 },
  { x: [0, 1], y: 1 },
  { x: [1, 0], y: 1 },
  { x: [1, 1], y: 0 },
];

function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function initWeights(seed) {
  const r = mulberry32(seed);
  const rand = () => (r() * 2 - 1) * 1.4;
  return {
    W1: [
      [rand(), rand()],
      [rand(), rand()],
    ],
    b1: [rand(), rand()],
    W2: [rand(), rand()],
    b2: rand(),
  };
}

function forwardSample(p, x) {
  const z1 = [
    p.W1[0][0] * x[0] + p.W1[0][1] * x[1] + p.b1[0],
    p.W1[1][0] * x[0] + p.W1[1][1] * x[1] + p.b1[1],
  ];
  const h = [tanh(z1[0]), tanh(z1[1])];
  const z2 = p.W2[0] * h[0] + p.W2[1] * h[1] + p.b2;
  const yhat = sigmoid(z2);
  return { z1, h, z2, yhat };
}

function backwardSample(p, x, target, fwd) {
  const { h, yhat } = fwd;
  const dLoss_dyhat = yhat - target;
  const dyhat_dz2 = dsigmoid(yhat);
  const dz2 = dLoss_dyhat * dyhat_dz2;

  const gW2 = [dz2 * h[0], dz2 * h[1]];
  const gb2 = dz2;

  const dh = [dz2 * p.W2[0], dz2 * p.W2[1]];
  const dz1 = [dh[0] * dtanh(h[0]), dh[1] * dtanh(h[1])];

  const gW1 = [
    [dz1[0] * x[0], dz1[0] * x[1]],
    [dz1[1] * x[0], dz1[1] * x[1]],
  ];
  const gb1 = [dz1[0], dz1[1]];

  return { dLoss_dyhat, dyhat_dz2, dz2, dh, dz1, gW1, gb1, gW2, gb2 };
}

function lossAndGrads(p) {
  const acc = {
    W1: [
      [0, 0],
      [0, 0],
    ],
    b1: [0, 0],
    W2: [0, 0],
    b2: 0,
  };
  let loss = 0;
  for (const d of DATA) {
    const fwd = forwardSample(p, d.x);
    const err = fwd.yhat - d.y;
    loss += 0.5 * err * err;
    const g = backwardSample(p, d.x, d.y, fwd);
    acc.W1[0][0] += g.gW1[0][0];
    acc.W1[0][1] += g.gW1[0][1];
    acc.W1[1][0] += g.gW1[1][0];
    acc.W1[1][1] += g.gW1[1][1];
    acc.b1[0] += g.gb1[0];
    acc.b1[1] += g.gb1[1];
    acc.W2[0] += g.gW2[0];
    acc.W2[1] += g.gW2[1];
    acc.b2 += g.gb2;
  }
  const n = DATA.length;
  acc.W1 = acc.W1.map((row) => row.map((v) => v / n));
  acc.b1 = acc.b1.map((v) => v / n);
  acc.W2 = acc.W2.map((v) => v / n);
  acc.b2 = acc.b2 / n;
  return { loss: loss / n, grads: acc };
}

function applyStep(p, grads, lr) {
  return {
    W1: [
      [p.W1[0][0] - lr * grads.W1[0][0], p.W1[0][1] - lr * grads.W1[0][1]],
      [p.W1[1][0] - lr * grads.W1[1][0], p.W1[1][1] - lr * grads.W1[1][1]],
    ],
    b1: [p.b1[0] - lr * grads.b1[0], p.b1[1] - lr * grads.b1[1]],
    W2: [p.W2[0] - lr * grads.W2[0], p.W2[1] - lr * grads.W2[1]],
    b2: p.b2 - lr * grads.b2,
  };
}

const fmt = (v, d = 3) => (v >= 0 ? "+" : "") + v.toFixed(d);

function gradColor(v, eps = 1e-4) {
  if (v > eps) return C.neg;
  if (v < -eps) return C.pos;
  return C.muted;
}

const NODE_POS = {
  x0: { x: 70, y: 70 },
  x1: { x: 70, y: 200 },
  h0: { x: 250, y: 70 },
  h1: { x: 250, y: 200 },
  o: { x: 430, y: 135 },
};

const SVG_W = 500;
const SVG_H = 270;

function useReducedMotion() {
  const [reduced, setReduced] = useState(false);
  useEffect(() => {
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    const update = () => setReduced(mq.matches);
    update();
    mq.addEventListener("change", update);
    return () => mq.removeEventListener("change", update);
  }, []);
  return reduced;
}

function Card({ children, style }) {
  return (
    <div
      style={{
        background: C.card,
        border: `1px solid ${C.border}`,
        borderRadius: 14,
        padding: 18,
        ...style,
      }}
    >
      {children}
    </div>
  );
}

function btnStyle(variant, disabled) {
  const base = {
    fontFamily: "inherit",
    fontSize: 13,
    fontWeight: 600,
    borderRadius: 9,
    padding: "8px 14px",
    cursor: disabled ? "not-allowed" : "pointer",
    opacity: disabled ? 0.4 : 1,
    transition: `transform 140ms ${EASE}, background 160ms ease, border-color 160ms ease`,
  };
  if (variant === "primary")
    return { ...base, background: C.accent, color: "#fff", border: `1px solid ${C.accent}` };
  if (variant === "dark")
    return { ...base, background: C.ink, color: "#fff", border: `1px solid ${C.ink}` };
  return { ...base, background: "transparent", color: C.ink, border: `1px solid ${C.border}` };
}

function Btn({ children, onClick, variant = "ghost", disabled, ariaLabel }) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      aria-label={ariaLabel}
      style={btnStyle(variant, disabled)}
      onMouseDown={(e) => {
        if (!disabled) e.currentTarget.style.transform = "scale(0.97)";
      }}
      onMouseUp={(e) => (e.currentTarget.style.transform = "scale(1)")}
      onMouseLeave={(e) => (e.currentTarget.style.transform = "scale(1)")}
    >
      {children}
    </button>
  );
}

const STAGES = [
  { phase: "idle", title: "Ready", blurb: "" },
  {
    phase: "fwd",
    key: "input",
    title: "Forward: inputs enter",
    blurb:
      "We feed one training example into the network. The two inputs x0 and x1 are just numbers; nothing has been computed yet. For XOR the answer should be 1 when exactly one input is 1.",
  },
  {
    phase: "fwd",
    key: "hidden",
    title: "Forward: hidden layer fires",
    blurb:
      "Each hidden neuron forms a weighted sum of the inputs plus a bias (its pre-activation z), then squashes it through tanh. The pre-activation is the raw score; the activation h is what gets passed on.",
  },
  {
    phase: "fwd",
    key: "output",
    title: "Forward: output and loss",
    blurb:
      "The output neuron sums the hidden activations, runs them through sigmoid to land between 0 and 1, and we compare the prediction to the target with squared error. That single loss number is what we will push backward.",
  },
  {
    phase: "bwd",
    key: "dout",
    title: "Backward: seed the gradient",
    blurb:
      "Backprop starts at the loss. dL/dyhat is (yhat - target). Multiply by sigmoid's local derivative yhat(1-yhat) and you get dL/dz2, the gradient at the output neuron's pre-activation. This is the upstream signal everything downstream will reuse.",
  },
  {
    phase: "bwd",
    key: "w2",
    title: "Backward: output weights",
    blurb:
      "Each output weight's gradient is its local derivative times the upstream gradient. The local derivative of z2 with respect to W2[i] is just h[i], so dL/dW2[i] = h[i] x dL/dz2. Large activations earn large weight updates. That is the chain rule in one line.",
  },
  {
    phase: "bwd",
    key: "dhidden",
    title: "Backward: into the hidden layer",
    blurb:
      "The gradient flows back through the output weights into each hidden activation: dL/dh[i] = W2[i] x dL/dz2. Then tanh's local derivative (1 - h^2) converts it to dL/dz1[i], the gradient at each hidden pre-activation.",
  },
  {
    phase: "bwd",
    key: "w1",
    title: "Backward: input weights",
    blurb:
      "Same rule, one layer deeper. dL/dW1[i][j] = x[j] x dL/dz1[i]. The gradient has now reached every weight in the network. Notice an input of 0 kills its weight's gradient: that weight did not affect this prediction.",
  },
  {
    phase: "bwd",
    key: "update",
    title: "Descend the gradient",
    blurb:
      "Every gradient points uphill in loss, so we step the opposite way: w := w - learning_rate x dL/dw. Small steps, repeated over all four examples, walk the weights downhill until the network separates the XOR classes.",
  },
];

function Edge({ from, to, weight, grad, mode, active, reduced }) {
  const a = NODE_POS[from];
  const b = NODE_POS[to];
  const showGrad = mode === "bwd" && active;
  const stroke = showGrad ? gradColor(grad) : C.wire;
  const mag = showGrad ? Math.min(Math.abs(grad) * 9, 5) : Math.min(Math.abs(weight) * 1.6 + 0.6, 5);
  const midX = (a.x + b.x) / 2;
  const midY = (a.y + b.y) / 2;
  const label = showGrad ? fmt(grad, 3) : fmt(weight, 2);
  return (
    <g>
      <line
        x1={a.x}
        y1={a.y}
        x2={b.x}
        y2={b.y}
        stroke={stroke}
        strokeWidth={mag}
        strokeLinecap="round"
        opacity={active ? 0.95 : mode === "bwd" ? 0.25 : 0.55}
        style={{ transition: reduced ? "none" : `stroke 220ms ease, stroke-width 220ms ${EASE}, opacity 220ms ease` }}
      />
      {active && (
        <g style={{ transition: reduced ? "none" : `opacity 200ms ease` }}>
          <rect
            x={midX - 22}
            y={midY - 9}
            width={44}
            height={18}
            rx={5}
            fill={C.card}
            stroke={showGrad ? gradColor(grad) : C.border}
            strokeWidth={1}
          />
          <text
            x={midX}
            y={midY + 4}
            textAnchor="middle"
            fontSize={10}
            fontFamily="ui-monospace, 'SF Mono', Menlo, monospace"
            fill={showGrad ? gradColor(grad) : C.muted}
          >
            {label}
          </text>
        </g>
      )}
    </g>
  );
}

function Node({ id, label, value, sub, color, active, mode, reduced }) {
  const p = NODE_POS[id];
  const lit = active;
  const ring = mode === "bwd" ? C.accent : color;
  return (
    <g style={{ transition: reduced ? "none" : `transform 240ms ${EASE}` }}>
      <circle
        cx={p.x}
        cy={p.y}
        r={26}
        fill={lit ? (mode === "bwd" ? C.accentSoft : color + "22") : C.card}
        stroke={lit ? ring : C.border}
        strokeWidth={lit ? 2.4 : 1.4}
        style={{
          transition: reduced ? "none" : `fill 220ms ease, stroke 220ms ease, stroke-width 220ms ${EASE}`,
          filter: lit && !reduced ? `drop-shadow(0 0 8px ${ring}55)` : "none",
        }}
      />
      <text x={p.x} y={p.y - 4} textAnchor="middle" fontSize={11} fontWeight={700} fill={C.ink}>
        {label}
      </text>
      <text
        x={p.x}
        y={p.y + 10}
        textAnchor="middle"
        fontSize={10}
        fontFamily="ui-monospace, 'SF Mono', Menlo, monospace"
        fill={lit ? (mode === "bwd" ? C.accent : C.ink) : C.muted}
      >
        {value}
      </text>
      {sub && (
        <text x={p.x} y={p.y + 42} textAnchor="middle" fontSize={9} fill={C.muted}>
          {sub}
        </text>
      )}
    </g>
  );
}

function Diagram({ params, sampleIdx, stage, grads, fwd, reduced }) {
  const x = DATA[sampleIdx].x;
  const st = STAGES[stage];
  const mode = st.phase === "bwd" ? "bwd" : st.phase === "fwd" ? "fwd" : "idle";
  const key = st.key;

  const fwdShowHidden = mode === "fwd" && (key === "hidden" || key === "output");
  const fwdShowOut = mode === "fwd" && key === "output";

  const bwdOut = mode === "bwd";
  const bwdW2 = mode === "bwd" && ["w2", "dhidden", "w1", "update"].includes(key);
  const bwdHidden = mode === "bwd" && ["dhidden", "w1", "update"].includes(key);
  const bwdW1 = mode === "bwd" && ["w1", "update"].includes(key);

  const edges = [
    { from: "x0", to: "h0", w: params.W1[0][0], g: grads?.gW1[0][0], active: mode === "fwd" ? fwdShowHidden : bwdW1 },
    { from: "x1", to: "h0", w: params.W1[0][1], g: grads?.gW1[0][1], active: mode === "fwd" ? fwdShowHidden : bwdW1 },
    { from: "x0", to: "h1", w: params.W1[1][0], g: grads?.gW1[1][0], active: mode === "fwd" ? fwdShowHidden : bwdW1 },
    { from: "x1", to: "h1", w: params.W1[1][1], g: grads?.gW1[1][1], active: mode === "fwd" ? fwdShowHidden : bwdW1 },
    { from: "h0", to: "o", w: params.W2[0], g: grads?.gW2[0], active: mode === "fwd" ? fwdShowOut : bwdW2 },
    { from: "h1", to: "o", w: params.W2[1], g: grads?.gW2[1], active: mode === "fwd" ? fwdShowOut : bwdW2 },
  ];

  const nodeVal = (id) => {
    if (mode === "bwd") {
      if (id === "o") return bwdOut ? `g ${fmt(grads.dz2, 2)}` : "";
      if (id === "h0") return bwdHidden ? `g ${fmt(grads.dz1[0], 2)}` : "";
      if (id === "h1") return bwdHidden ? `g ${fmt(grads.dz1[1], 2)}` : "";
      return String(x[id === "x0" ? 0 : 1]);
    }
    if (id === "x0") return String(x[0]);
    if (id === "x1") return String(x[1]);
    if (id === "h0") return fwdShowHidden ? fwd.h[0].toFixed(2) : "";
    if (id === "h1") return fwdShowHidden ? fwd.h[1].toFixed(2) : "";
    if (id === "o") return fwdShowOut ? fwd.yhat.toFixed(2) : "";
    return "";
  };

  const nodeActive = (id) => {
    if (id === "x0" || id === "x1") return mode === "fwd" ? key === "input" || true : false;
    if (id === "h0" || id === "h1") return mode === "fwd" ? fwdShowHidden : bwdHidden;
    if (id === "o") return mode === "fwd" ? fwdShowOut : bwdOut;
    return false;
  };

  const nodeSub = (id) => {
    if (mode === "fwd" && fwdShowHidden && id === "h0") return `z=${fwd.z1[0].toFixed(2)}`;
    if (mode === "fwd" && fwdShowHidden && id === "h1") return `z=${fwd.z1[1].toFixed(2)}`;
    if (mode === "fwd" && fwdShowOut && id === "o") return `z=${fwd.z2.toFixed(2)}`;
    return null;
  };

  return (
    <svg
      viewBox={`0 0 ${SVG_W} ${SVG_H}`}
      width="100%"
      role="img"
      aria-label="Network diagram showing two inputs, two hidden tanh neurons, and one sigmoid output"
      style={{ display: "block" }}
    >
      <text x={70} y={250} textAnchor="middle" fontSize={9} fill={C.muted} letterSpacing="0.06em">
        INPUTS
      </text>
      <text x={250} y={250} textAnchor="middle" fontSize={9} fill={C.muted} letterSpacing="0.06em">
        HIDDEN (tanh)
      </text>
      <text x={430} y={250} textAnchor="middle" fontSize={9} fill={C.muted} letterSpacing="0.06em">
        OUTPUT (sigmoid)
      </text>
      {edges.map((e, i) => (
        <Edge key={i} from={e.from} to={e.to} weight={e.w} grad={e.g} mode={mode} active={e.active} reduced={reduced} />
      ))}
      {["x0", "x1", "h0", "h1", "o"].map((id) => (
        <Node
          key={id}
          id={id}
          label={id === "o" ? "ŷ" : id === "x0" ? "x0" : id === "x1" ? "x1" : id}
          value={nodeVal(id)}
          sub={nodeSub(id)}
          color={id === "o" ? C.accent : id.startsWith("h") ? C.pos : C.ink}
          active={nodeActive(id)}
          mode={mode}
          reduced={reduced}
        />
      ))}
    </svg>
  );
}

function LossChart({ history, reduced }) {
  const w = 320;
  const h = 90;
  const pad = 4;
  if (history.length < 2) {
    return (
      <div style={{ height: h, display: "flex", alignItems: "center", justifyContent: "center", color: C.muted, fontSize: 12 }}>
        Run an update or train to plot the loss.
      </div>
    );
  }
  const max = Math.max(...history);
  const min = Math.min(...history);
  const span = max - min || 1;
  const pts = history.map((v, i) => {
    const px = pad + (i / (history.length - 1)) * (w - 2 * pad);
    const py = pad + (1 - (v - min) / span) * (h - 2 * pad);
    return `${px.toFixed(1)},${py.toFixed(1)}`;
  });
  const last = history[history.length - 1];
  const lastPt = pts[pts.length - 1].split(",");
  return (
    <svg viewBox={`0 0 ${w} ${h}`} width="100%" role="img" aria-label={`Loss curve, current loss ${last.toFixed(4)}`}>
      <polyline
        points={pts.join(" ")}
        fill="none"
        stroke={C.accent}
        strokeWidth={2}
        strokeLinejoin="round"
        strokeLinecap="round"
        style={{ transition: reduced ? "none" : `all 200ms ${EASE}` }}
      />
      <circle cx={lastPt[0]} cy={lastPt[1]} r={3} fill={C.accent} />
    </svg>
  );
}

const SAMPLE_LABEL = (i) => `(${DATA[i].x[0]}, ${DATA[i].x[1]}) -> ${DATA[i].y}`;

export default function App() {
  const reduced = useReducedMotion();
  const [seed, setSeed] = useState(7);
  const [params, setParams] = useState(() => initWeights(7));
  const [sampleIdx, setSampleIdx] = useState(1);
  const [stage, setStage] = useState(0);
  const [lr, setLr] = useState(2.5);
  const [history, setHistory] = useState([]);
  const [epoch, setEpoch] = useState(0);
  const [training, setTraining] = useState(false);
  const trainRef = useRef(null);

  const fwd = useMemo(() => forwardSample(params, DATA[sampleIdx].x), [params, sampleIdx]);
  const grads = useMemo(() => backwardSample(params, DATA[sampleIdx].x, DATA[sampleIdx].y, fwd), [params, sampleIdx, fwd]);
  const { loss: batchLoss, grads: batchGrads } = useMemo(() => lossAndGrads(params), [params]);

  useEffect(() => {
    setHistory([batchLoss]);
    setEpoch(0);
  }, [seed]);

  useEffect(() => {
    if (!training) return;
    let stop = false;
    let local = params;
    const tick = () => {
      if (stop) return;
      let cur = local;
      for (let k = 0; k < 6; k++) {
        const { grads: g } = lossAndGrads(cur);
        cur = applyStep(cur, g, lr);
      }
      local = cur;
      const { loss } = lossAndGrads(cur);
      setParams(cur);
      setEpoch((e) => e + 6);
      setHistory((prev) => {
        const next = [...prev, loss];
        return next.length > 240 ? next.slice(next.length - 240) : next;
      });
      if (loss < 0.002) {
        setTraining(false);
        return;
      }
      trainRef.current = requestAnimationFrame(tick);
    };
    trainRef.current = requestAnimationFrame(tick);
    return () => {
      stop = true;
      cancelAnimationFrame(trainRef.current);
    };
  }, [training, lr]);

  const reset = () => {
    const ns = (seed * 1103515245 + 12345) % 2147483647;
    setTraining(false);
    setSeed(ns);
    setParams(initWeights(ns));
    setStage(0);
  };

  const stepForward = () => {
    setTraining(false);
    setStage((s) => {
      if (s >= 3) return s;
      return Math.max(1, s + 1);
    });
  };
  const stepBackward = () => {
    setTraining(false);
    setStage((s) => {
      if (s < 3) return 4;
      if (s >= STAGES.length - 1) return s;
      return s + 1;
    });
  };

  const runUpdate = () => {
    setTraining(false);
    setParams((p) => {
      const { grads: g } = lossAndGrads(p);
      const np = applyStep(p, g, lr);
      const { loss } = lossAndGrads(np);
      setHistory((prev) => {
        const next = [...prev, loss];
        return next.length > 240 ? next.slice(next.length - 240) : next;
      });
      setEpoch((e) => e + 1);
      return np;
    });
    setStage(8);
  };

  const st = STAGES[stage];
  const phaseLabel = st.phase === "fwd" ? "Forward pass" : st.phase === "bwd" ? "Backward pass" : "Idle";

  const focusCss = `
    .bp-root :focus-visible { outline: 2px solid ${C.accent}; outline-offset: 2px; border-radius: 9px; }
    .bp-root input[type=range] { accent-color: ${C.accent}; }
    @media (prefers-reduced-motion: reduce) {
      .bp-root * { transition-duration: 0.001ms !important; animation-duration: 0.001ms !important; }
    }
  `;

  const predictions = DATA.map((d) => forwardSample(params, d.x).yhat);
  const solved = DATA.every((d, i) => (d.y === 1 ? predictions[i] > 0.5 : predictions[i] < 0.5));

  return (
    <div
      className="bp-root"
      style={{
        fontFamily: "Georgia, 'Iowan Old Style', serif",
        background: C.bg,
        minHeight: "100vh",
        padding: "26px 14px 56px",
        color: C.ink,
      }}
    >
      <style>{focusCss}</style>
      <div style={{ maxWidth: 720, margin: "0 auto" }}>
        <header style={{ marginBottom: 22 }}>
          <div style={{ fontSize: 11, letterSpacing: "0.14em", textTransform: "uppercase", color: C.muted, marginBottom: 6 }}>
            Machine Learning, computed live
          </div>
          <h1 style={{ fontSize: 30, fontWeight: 700, margin: 0, lineHeight: 1.1, letterSpacing: "-0.02em", textWrap: "balance" }}>
            Backpropagation, by Hand
          </h1>
          <p style={{ color: C.muted, fontSize: 15, margin: "10px 0 0", lineHeight: 1.6, maxWidth: "62ch" }}>
            XOR is the smallest problem a single neuron cannot solve, so it needs a hidden layer, which makes it the
            perfect place to watch backprop work. This net has two tanh neurons and one sigmoid output; walk the forward
            pass and the chain rule backward one step at a time, and read every gradient as it forms.
          </p>
        </header>

        <Card style={{ marginBottom: 16, padding: "16px 18px 8px" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", flexWrap: "wrap", gap: 8 }}>
            <div style={{ fontSize: 12, fontWeight: 700, letterSpacing: "0.04em", color: st.phase === "bwd" ? C.accent : C.ink }}>
              {phaseLabel}
            </div>
            <div style={{ fontSize: 12, color: C.muted }}>
              Example <b style={{ color: C.ink }}>{SAMPLE_LABEL(sampleIdx)}</b>
            </div>
          </div>
          <Diagram params={params} sampleIdx={sampleIdx} stage={stage} grads={grads} fwd={fwd} reduced={reduced} />
          <div style={{ display: "flex", gap: 6, justifyContent: "center", flexWrap: "wrap", marginTop: 4, marginBottom: 8 }}>
            <span style={{ fontSize: 11, color: C.muted, alignSelf: "center", marginRight: 4 }}>Pick example:</span>
            {DATA.map((d, i) => (
              <button
                key={i}
                onClick={() => {
                  setSampleIdx(i);
                  setStage(0);
                  setTraining(false);
                }}
                aria-pressed={sampleIdx === i}
                style={{
                  fontFamily: "ui-monospace, monospace",
                  fontSize: 11,
                  padding: "4px 9px",
                  borderRadius: 7,
                  cursor: "pointer",
                  border: `1px solid ${sampleIdx === i ? C.accent : C.border}`,
                  background: sampleIdx === i ? C.accentSoft : "transparent",
                  color: sampleIdx === i ? C.accent : C.muted,
                  fontWeight: sampleIdx === i ? 700 : 400,
                  transition: `all 150ms ${EASE}`,
                }}
              >
                {SAMPLE_LABEL(i)}
              </button>
            ))}
          </div>
        </Card>

        <Card
          style={{
            marginBottom: 16,
            background: st.phase === "bwd" ? C.accentSoft : st.phase === "fwd" ? C.posSoft : C.card,
            border: `1px solid ${st.phase === "bwd" ? C.accent + "55" : st.phase === "fwd" ? C.pos + "44" : C.border}`,
            minHeight: 96,
            transition: reduced ? "none" : `background 240ms ease, border-color 240ms ease`,
          }}
        >
          <div
            style={{
              fontSize: 13,
              fontWeight: 700,
              marginBottom: 6,
              color: st.phase === "bwd" ? C.accent : st.phase === "fwd" ? C.pos : C.ink,
            }}
          >
            {st.title}
          </div>
          <p style={{ fontSize: 14, lineHeight: 1.6, margin: 0, color: C.ink, maxWidth: "64ch" }}>
            {stage === 0
              ? "Step the forward pass to push this example through the network, then step the backward pass to watch the chain rule assign a gradient to every weight. Or skip ahead and run a gradient-descent update."
              : st.blurb}
          </p>
          {stage >= 4 && stage <= 7 && (
            <div
              style={{
                marginTop: 10,
                fontFamily: "ui-monospace, 'SF Mono', Menlo, monospace",
                fontSize: 12,
                background: C.card,
                border: `1px solid ${C.border}`,
                borderRadius: 8,
                padding: "8px 10px",
                color: C.ink,
                overflowX: "auto",
              }}
            >
              {stage === 4 && `dL/dz2 = (yhat - y) x yhat(1-yhat) = ${fmt(grads.dLoss_dyhat, 3)} x ${fwd.yhat.toFixed(3)}(1-${fwd.yhat.toFixed(3)}) = ${fmt(grads.dz2, 3)}`}
              {stage === 5 && `dL/dW2 = [h0, h1] x dL/dz2 = [${fwd.h[0].toFixed(2)}, ${fwd.h[1].toFixed(2)}] x ${fmt(grads.dz2, 3)} = [${fmt(grads.gW2[0], 3)}, ${fmt(grads.gW2[1], 3)}]`}
              {stage === 6 && `dL/dz1 = (W2 x dL/dz2)(1 - h^2) = [${fmt(grads.dz1[0], 3)}, ${fmt(grads.dz1[1], 3)}]`}
              {stage === 7 && `dL/dW1 = outer(dL/dz1, x), e.g. dL/dW1[0][0] = x0 x dL/dz1[0] = ${DATA[sampleIdx].x[0]} x ${fmt(grads.dz1[0], 3)} = ${fmt(grads.gW1[0][0], 3)}`}
            </div>
          )}
        </Card>

        <Card style={{ marginBottom: 16 }}>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 8, alignItems: "center" }}>
            <Btn onClick={stepForward} variant="ghost" disabled={stage >= 3} ariaLabel="Step the forward pass forward one stage">
              Step forward
            </Btn>
            <Btn onClick={stepBackward} variant="dark" disabled={stage >= STAGES.length - 1 && stage >= 4} ariaLabel="Step the backward pass one node deeper">
              Step backward
            </Btn>
            <Btn onClick={runUpdate} variant="primary" ariaLabel="Run one gradient descent update on all examples">
              Run one update
            </Btn>
            <Btn
              onClick={() => setTraining((t) => !t)}
              variant={training ? "ghost" : "primary"}
              ariaLabel={training ? "Pause training" : "Train the network"}
            >
              {training ? "Pause" : "Train"}
            </Btn>
            <Btn onClick={reset} variant="ghost" ariaLabel="Reset weights to a new random initialization">
              Reset weights
            </Btn>
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: 12, marginTop: 16, flexWrap: "wrap" }}>
            <label htmlFor="lr" style={{ fontSize: 13, color: C.muted, minWidth: 92 }}>
              Learning rate
            </label>
            <input
              id="lr"
              type="range"
              min={0.2}
              max={6}
              step={0.1}
              value={lr}
              onChange={(e) => setLr(+e.target.value)}
              style={{ flex: 1, minWidth: 130 }}
            />
            <span
              style={{
                fontFamily: "ui-monospace, monospace",
                fontSize: 14,
                fontWeight: 700,
                color: C.accent,
                minWidth: 42,
                textAlign: "right",
              }}
            >
              {lr.toFixed(1)}
            </span>
          </div>
          <p style={{ fontSize: 12, color: C.muted, margin: "8px 0 0", lineHeight: 1.5 }}>
            Larger steps descend faster but can overshoot and oscillate. Too small and the loss crawls. Watch the curve
            below react when you change it mid-training.
          </p>
        </Card>

        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: 14, marginBottom: 16 }}>
          <Card>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 8 }}>
              <span style={{ fontSize: 13, fontWeight: 700 }}>Loss over time</span>
              <span style={{ fontFamily: "ui-monospace, monospace", fontSize: 13, color: C.accent, fontWeight: 700 }}>
                {batchLoss.toFixed(4)}
              </span>
            </div>
            <LossChart history={history} reduced={reduced} />
            <div style={{ fontSize: 11, color: C.muted, marginTop: 6 }}>
              Mean squared error over all 4 examples after {epoch} update{epoch === 1 ? "" : "s"}.
            </div>
          </Card>

          <Card>
            <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 10 }}>Predictions vs target</div>
            <div style={{ display: "flex", flexDirection: "column", gap: 7 }}>
              {DATA.map((d, i) => {
                const yh = predictions[i];
                const right = d.y === 1 ? yh > 0.5 : yh < 0.5;
                return (
                  <div key={i} style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 12 }}>
                    <span style={{ fontFamily: "ui-monospace, monospace", color: C.muted, width: 70 }}>
                      ({d.x[0]}, {d.x[1]})
                    </span>
                    <div style={{ flex: 1, height: 10, background: C.bg, borderRadius: 5, overflow: "hidden", border: `1px solid ${C.border}` }}>
                      <div
                        style={{
                          width: `${clamp(yh, 0, 1) * 100}%`,
                          height: "100%",
                          background: right ? C.pos : C.neg,
                          transition: reduced ? "none" : `width 240ms ${EASE}, background 240ms ease`,
                        }}
                      />
                    </div>
                    <span style={{ fontFamily: "ui-monospace, monospace", width: 64, textAlign: "right", color: right ? C.pos : C.neg }}>
                      {yh.toFixed(2)} / {d.y}
                    </span>
                  </div>
                );
              })}
            </div>
            <div
              style={{
                fontSize: 12,
                marginTop: 10,
                color: solved ? C.pos : C.muted,
                fontWeight: solved ? 700 : 400,
              }}
            >
              {solved ? "All four classified correctly." : "Not yet separating the classes. Keep training."}
            </div>
          </Card>
        </div>

        <Card>
          <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 4 }}>Why we step opposite the gradient</div>
          <p style={{ fontSize: 14, lineHeight: 1.6, margin: 0, color: C.ink, maxWidth: "64ch" }}>
            A gradient is the direction of steepest increase in the loss. Each weight's gradient answers one question: if I
            nudge this weight up a little, does the loss rise or fall, and how sharply? Backprop computes all of those
            answers in a single backward sweep by reusing the upstream gradient at each layer, which is why it is far
            cheaper than poking each weight one at a time. To shrink the loss we move every weight a small step in the
            opposite direction of its gradient, scaled by the learning rate. Repeat over the four examples and the network
            bends its decision boundary until XOR comes apart.
          </p>
        </Card>
      </div>
    </div>
  );
}
