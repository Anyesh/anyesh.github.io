import { useState, useEffect, useMemo, useRef } from "react";

export const meta = {
  title: "Inside an LSTM Cell",
  category: "Deep Learning",
  description:
    "Open one LSTM cell and run it on a short sequence. Watch the forget, input, and output gates compute for real, see the cell state carry memory along a near-identity highway, and trace why an additive gated update lets gradients survive the long time horizons where a vanilla recurrent net vanishes.",
  date: "2026-06-02",
  tags: ["lstm", "rnn", "gates", "gradients", "deep-learning", "memory"],
};

const C = {
  bg: "#f7f5f2",
  card: "#ffffff",
  border: "#e6e1d9",
  ink: "#1b1916",
  muted: "#7c7368",
  accent: "#c0561f",
  accentSoft: "#f6ece5",
  forget: "#2a5298",
  forgetSoft: "#e7eef8",
  input: "#2e7d51",
  inputSoft: "#e6f1ea",
  output: "#9a4bb0",
  outputSoft: "#f1e7f5",
  cand: "#c0561f",
  candSoft: "#f6ece5",
  cell: "#b8821a",
  cellSoft: "#f6efdc",
  neg: "#b23b5e",
  grid: "#eee9e1",
};

const EASE = "cubic-bezier(0.23, 1, 0.32, 1)";

const sigmoid = (x) => 1 / (1 + Math.exp(-x));
const tanh = (x) => Math.tanh(x);
const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));

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

// One LSTM gate is governed by an input weight Wx, a recurrent weight Uh, and a
// bias b. With a single hidden unit and a single scalar input per step the whole
// cell is six such triples plus the four nonlinearities, which keeps every number
// on screen legible while staying the exact textbook computation.
const GATE_KEYS = ["f", "i", "g", "o"];

function makeCell(p) {
  return {
    f: { Wx: p.fWx, Uh: p.fUh, b: p.fb, act: sigmoid },
    i: { Wx: p.iWx, Uh: p.iUh, b: p.ib, act: sigmoid },
    g: { Wx: p.gWx, Uh: p.gUh, b: p.gb, act: tanh },
    o: { Wx: p.oWx, Uh: p.oUh, b: p.ob, act: sigmoid },
  };
}

// Forward pass of a single-unit LSTM over a scalar sequence. Returns, per step,
// the pre-activations, the four gate values, the new cell state, and the hidden
// output, exactly as c_t = f * c_{t-1} + i * g and h_t = o * tanh(c_t).
function runLSTM(cell, xs, c0 = 0, h0 = 0) {
  const steps = [];
  let c = c0;
  let h = h0;
  for (let t = 0; t < xs.length; t++) {
    const x = xs[t];
    const zf = cell.f.Wx * x + cell.f.Uh * h + cell.f.b;
    const zi = cell.i.Wx * x + cell.i.Uh * h + cell.i.b;
    const zg = cell.g.Wx * x + cell.g.Uh * h + cell.g.b;
    const zo = cell.o.Wx * x + cell.o.Uh * h + cell.o.b;
    const f = sigmoid(zf);
    const i = sigmoid(zi);
    const g = tanh(zg);
    const o = sigmoid(zo);
    const cPrev = c;
    const cNew = f * cPrev + i * g;
    const cTanh = tanh(cNew);
    const hNew = o * cTanh;
    steps.push({
      t,
      x,
      cPrev,
      hPrev: h,
      z: { f: zf, i: zi, g: zg, o: zo },
      f,
      i,
      g,
      o,
      add: i * g,
      kept: f * cPrev,
      c: cNew,
      cTanh,
      h: hNew,
    });
    c = cNew;
    h = hNew;
  }
  return steps;
}

// Vanilla recurrent cell h_t = tanh(W h_{t-1} + V x_t + b), run for contrast. Its
// gradient path through time multiplies by W * tanh'(z) at every step, the very
// product that the vanishing-gradients artifact shows decaying geometrically.
function runVanilla(W, V, b, xs, h0 = 0) {
  const steps = [];
  let h = h0;
  for (let t = 0; t < xs.length; t++) {
    const z = W * h + V * xs[t] + b;
    const hNew = tanh(z);
    const jac = W * (1 - hNew * hNew);
    steps.push({ t, x: xs[t], hPrev: h, z, h: hNew, jac });
    h = hNew;
  }
  return steps;
}

const PRESETS = {
  hold: {
    label: "Hold the first value",
    blurb:
      "A high forget bias pins the gate open near 1, the input gate stays shut after the first step, and the cell state carries that opening value almost unchanged to the end.",
    xs: [0.9, 0, 0, 0, 0, 0, 0, 0],
    weights: { fWx: 0, fUh: 0, fb: 5, iWx: 5, iUh: -6, ib: -3, gWx: 1.4, gUh: 0, gb: 0, oWx: 0, oUh: 0, ob: 2.2 },
  },
  sum: {
    label: "Accumulate a running sum",
    blurb:
      "Forget held at 1 and input held at 1 turn the cell update into c_t = c_{t-1} + g, so the cell state integrates the candidate across the sequence.",
    xs: [0.3, 0.3, 0.3, -0.2, 0.4, 0.1, 0.2, 0.2],
    weights: { fWx: 0, fUh: 0, fb: 4, iWx: 0, iUh: 0, ib: 4, gWx: 1, gUh: 0, gb: 0, oWx: 0, oUh: 0, ob: 2.5 },
  },
  gate: {
    label: "Input-driven gate",
    blurb:
      "Here the input itself drives the gates: a large positive x opens the input gate to write, a negative x opens the forget gate to clear. The cell reacts to the signal it sees.",
    xs: [1, 0.2, -1.2, 0.5, 0.8, -0.9, 0.3, 0.6],
    weights: { fWx: 2, fUh: 0, fb: 1, iWx: 3, iUh: 0, ib: -0.5, gWx: 1.4, gUh: 0.3, gb: 0, oWx: 0.5, oUh: 0, ob: 1.5 },
  },
};

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

function Segmented({ options, value, onChange, ariaLabel }) {
  return (
    <div role="group" aria-label={ariaLabel} style={{ display: "inline-flex", gap: 4, flexWrap: "wrap" }}>
      {options.map((o) => {
        const active = o.value === value;
        return (
          <button
            key={o.value}
            onClick={() => onChange(o.value)}
            aria-pressed={active}
            style={{
              fontFamily: "inherit",
              fontSize: 12,
              fontWeight: active ? 700 : 500,
              padding: "6px 12px",
              borderRadius: 8,
              cursor: "pointer",
              border: `1px solid ${active ? C.accent : C.border}`,
              background: active ? C.accentSoft : "transparent",
              color: active ? C.accent : C.muted,
              transition: `all 150ms ${EASE}`,
            }}
          >
            {o.label}
          </button>
        );
      })}
    </div>
  );
}

const fmt = (v, d = 2) => {
  if (!isFinite(v)) return "inf";
  return v.toFixed(d);
};

const fmtSci = (v) => {
  if (!isFinite(v)) return "inf";
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a >= 1e4 || a < 1e-3) return v.toExponential(2);
  if (a >= 1) return v.toFixed(3);
  return v.toFixed(4);
};

const GATE_META = {
  f: { name: "forget", color: C.forget, soft: C.forgetSoft, range: [0, 1], desc: "how much old memory to keep" },
  i: { name: "input", color: C.input, soft: C.inputSoft, range: [0, 1], desc: "how much new candidate to write" },
  g: { name: "candidate", color: C.cand, soft: C.candSoft, range: [-1, 1], desc: "the value proposed for writing" },
  o: { name: "output", color: C.output, soft: C.outputSoft, range: [0, 1], desc: "how much of the cell to expose" },
};

// A horizontal bar for a gate value. Sigmoid gates fill from the left over [0,1];
// the candidate is signed over [-1,1] and fills out from a center zero line so a
// reader can see its sign at a glance.
function GateBar({ gkey, value, reduced }) {
  const meta = GATE_META[gkey];
  const signed = meta.range[0] < 0;
  const trackH = 14;
  let fill;
  if (signed) {
    const half = 50;
    const w = Math.abs(value) * half;
    const left = value >= 0 ? half : half - w;
    fill = (
      <div
        style={{
          position: "absolute",
          left: `${left}%`,
          width: `${w}%`,
          top: 0,
          bottom: 0,
          background: value < 0 ? C.neg : meta.color,
          borderRadius: 3,
          transition: reduced ? "none" : `left 320ms ${EASE}, width 320ms ${EASE}`,
        }}
      />
    );
  } else {
    fill = (
      <div
        style={{
          position: "absolute",
          left: 0,
          width: `${clamp(value, 0, 1) * 100}%`,
          top: 0,
          bottom: 0,
          background: meta.color,
          borderRadius: 3,
          transition: reduced ? "none" : `width 320ms ${EASE}`,
        }}
      />
    );
  }
  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 4 }}>
        <span style={{ fontSize: 12 }}>
          <span style={{ fontWeight: 700, color: meta.color }}>{gkey}</span>
          <span style={{ color: C.muted, marginLeft: 6 }}>{meta.name}</span>
        </span>
        <span style={{ fontFamily: "ui-monospace, monospace", fontSize: 13, fontWeight: 700, color: value < 0 ? C.neg : meta.color }}>
          {fmt(value)}
        </span>
      </div>
      <div
        style={{ position: "relative", height: trackH, background: meta.soft, borderRadius: 4, overflow: "hidden" }}
        role="img"
        aria-label={`${meta.name} gate value ${fmt(value)}, ${meta.desc}`}
      >
        {signed && (
          <div style={{ position: "absolute", left: "50%", top: 0, bottom: 0, width: 1, background: `${meta.color}55` }} />
        )}
        {fill}
      </div>
    </div>
  );
}

// SVG schematic of the cell: the cell-state line runs straight across the top as
// the memory highway, the forget gate pinches it, and i * g is added in. The
// stroke width of the highway tracks how open the forget gate is, so a reader sees
// the near-identity path widen as f approaches 1.
function CellDiagram({ step, reduced }) {
  const W = 640;
  const H = 250;
  const beltY = 52;
  const xIn = 70;
  const xForget = 230;
  const xAdd = 400;
  const xOut = 560;
  if (!step) return null;
  const fOpen = clamp(step.f, 0, 1);
  const beltStroke = 2 + fOpen * 7;
  const tr = reduced ? "none" : `all 320ms ${EASE}`;

  const node = (x, y, label, color, val) => (
    <g>
      <circle cx={x} cy={y} r={15} fill={C.card} stroke={color} strokeWidth={2} style={{ transition: tr }} />
      <text x={x} y={y + 4} textAnchor="middle" fontSize={13} fontWeight={700} fill={color} fontFamily="ui-monospace, monospace">
        {label}
      </text>
      {val !== undefined && (
        <text x={x} y={y - 24} textAnchor="middle" fontSize={11} fontFamily="ui-monospace, monospace" fill={color} fontWeight={700}>
          {fmt(val)}
        </text>
      )}
    </g>
  );

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      width="100%"
      role="img"
      aria-label={`LSTM cell diagram at step ${step.t + 1}. Forget gate ${fmt(step.f)}, input gate ${fmt(step.i)}, candidate ${fmt(step.g)}, output gate ${fmt(step.o)}. Cell state ${fmt(step.c)}, hidden output ${fmt(step.h)}.`}
      style={{ display: "block" }}
    >
      <text x={xIn - 40} y={beltY - 18} fontSize={11} fill={C.muted} fontFamily="ui-monospace, monospace">
        c(t-1) = {fmt(step.cPrev)}
      </text>
      <text x={xOut + 8} y={beltY - 18} fontSize={11} fill={C.cell} fontWeight={700} fontFamily="ui-monospace, monospace">
        c(t) = {fmt(step.c)}
      </text>

      <line x1={xIn} y1={beltY} x2={xForget} y2={beltY} stroke={C.cell} strokeWidth={beltStroke} strokeLinecap="round" style={{ transition: tr }} />
      <line x1={xForget} y1={beltY} x2={xAdd} y2={beltY} stroke={C.cell} strokeWidth={beltStroke} strokeLinecap="round" style={{ transition: tr }} />
      <line x1={xAdd} y1={beltY} x2={xOut + 30} y2={beltY} stroke={C.cell} strokeWidth={beltStroke} strokeLinecap="round" style={{ transition: tr }} />

      <g>
        <circle cx={xForget} cy={beltY} r={13} fill={C.forgetSoft} stroke={C.forget} strokeWidth={2} style={{ transition: tr }} />
        <text x={xForget} y={beltY + 4} textAnchor="middle" fontSize={13} fontWeight={700} fill={C.forget}>
          x
        </text>
        <text x={xForget} y={beltY + 30} textAnchor="middle" fontSize={10} fill={C.forget} fontFamily="ui-monospace, monospace">
          x f={fmt(step.f)}
        </text>
      </g>
      <g>
        <circle cx={xAdd} cy={beltY} r={13} fill={C.inputSoft} stroke={C.input} strokeWidth={2} style={{ transition: tr }} />
        <text x={xAdd} y={beltY + 5} textAnchor="middle" fontSize={15} fontWeight={700} fill={C.input}>
          +
        </text>
        <text x={xAdd} y={beltY + 30} textAnchor="middle" fontSize={10} fill={C.input} fontFamily="ui-monospace, monospace">
          + i*g={fmt(step.add)}
        </text>
      </g>

      {node(xForget, 150, "f", C.forget, step.f)}
      {node(xForget - 56, 150, "i", C.input, step.i)}
      {node(xAdd - 40, 150, "g", C.cand, step.g)}
      {node(xOut, 150, "o", C.output, step.o)}

      <line x1={xForget} y1={135} x2={xForget} y2={beltY + 13} stroke={C.forget} strokeWidth={1.5} strokeDasharray="3 3" />
      <line x1={xAdd - 40} y1={135} x2={xAdd - 8} y2={beltY + 8} stroke={C.input} strokeWidth={1.5} strokeDasharray="3 3" />
      <line x1={xForget - 56} y1={135} x2={xAdd - 10} y2={beltY + 10} stroke={C.input} strokeWidth={1.5} strokeDasharray="3 3" />

      <text x={xIn - 40} y={195} fontSize={11} fill={C.muted} fontFamily="ui-monospace, monospace">
        x(t) = {fmt(step.x)}
      </text>
      <text x={xIn - 40} y={212} fontSize={11} fill={C.muted} fontFamily="ui-monospace, monospace">
        h(t-1) = {fmt(step.hPrev)}
      </text>

      <line x1={xOut} y1={beltY + 13} x2={xOut} y2={135} stroke={C.cell} strokeWidth={1.5} strokeDasharray="3 3" />
      <text x={xOut + 22} y={150} fontSize={10} fill={C.cell} fontFamily="ui-monospace, monospace">
        tanh(c)={fmt(step.cTanh)}
      </text>
      <g>
        <circle cx={xOut} cy={218} r={16} fill={C.outputSoft} stroke={C.output} strokeWidth={2} />
        <text x={xOut} y={222} textAnchor="middle" fontSize={12} fontWeight={700} fill={C.output} fontFamily="ui-monospace, monospace">
          h
        </text>
      </g>
      <line x1={xOut} y1={166} x2={xOut} y2={202} stroke={C.output} strokeWidth={1.5} strokeDasharray="3 3" />
      <text x={xOut + 24} y={222} fontSize={11} fill={C.output} fontWeight={700} fontFamily="ui-monospace, monospace">
        h(t) = {fmt(step.h)}
      </text>
    </svg>
  );
}

// Two overlaid traces of c_t and h_t across the whole sequence, with the current
// step marked. Plain y axis, no log scale, since these stay in a small range.
function SequenceChart({ steps, current, reduced }) {
  const W = 640;
  const H = 180;
  const padL = 40;
  const padR = 16;
  const padT = 14;
  const padB = 26;
  const n = steps.length;
  const cs = steps.map((s) => s.c);
  const hs = steps.map((s) => s.h);
  const all = cs.concat(hs);
  let lo = Math.min(...all, 0);
  let hi = Math.max(...all, 0);
  if (hi - lo < 0.5) {
    hi += 0.25;
    lo -= 0.25;
  }
  const pad = (hi - lo) * 0.1;
  lo -= pad;
  hi += pad;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;
  const xAt = (i) => padL + (n === 1 ? plotW / 2 : (i / (n - 1)) * plotW);
  const yAt = (v) => padT + (1 - (v - lo) / (hi - lo)) * plotH;
  const path = (arr) => arr.map((v, i) => `${i === 0 ? "M" : "L"}${xAt(i).toFixed(1)},${yAt(v).toFixed(1)}`).join("");
  const zeroY = yAt(0);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label="Cell state and hidden output across the sequence" style={{ display: "block" }}>
      <line x1={padL} y1={zeroY} x2={W - padR} y2={zeroY} stroke={C.grid} strokeWidth={1} />
      <text x={padL - 6} y={zeroY + 3} textAnchor="end" fontSize={9} fill={C.muted} fontFamily="ui-monospace, monospace">
        0
      </text>
      <text x={padL - 6} y={yAt(hi) + 8} textAnchor="end" fontSize={9} fill={C.muted} fontFamily="ui-monospace, monospace">
        {fmt(hi, 1)}
      </text>
      <text x={padL - 6} y={yAt(lo) + 2} textAnchor="end" fontSize={9} fill={C.muted} fontFamily="ui-monospace, monospace">
        {fmt(lo, 1)}
      </text>
      <line x1={xAt(current)} y1={padT} x2={xAt(current)} y2={H - padB} stroke={C.accent} strokeWidth={1} strokeDasharray="3 3" opacity={0.6} />
      <path d={path(cs)} fill="none" stroke={C.cell} strokeWidth={2.4} strokeLinejoin="round" strokeLinecap="round" style={{ transition: reduced ? "none" : "stroke 200ms ease" }} />
      <path d={path(hs)} fill="none" stroke={C.output} strokeWidth={2.4} strokeLinejoin="round" strokeLinecap="round" />
      {steps.map((s, i) => (
        <g key={i}>
          <circle cx={xAt(i)} cy={yAt(s.c)} r={i === current ? 4 : 2.4} fill={C.cell} />
          <circle cx={xAt(i)} cy={yAt(s.h)} r={i === current ? 4 : 2.4} fill={C.output} />
        </g>
      ))}
      {steps.map((s, i) => (
        <text key={`x${i}`} x={xAt(i)} y={H - padB + 16} textAnchor="middle" fontSize={9} fill={i === current ? C.accent : C.muted} fontFamily="ui-monospace, monospace">
          {i + 1}
        </text>
      ))}
    </svg>
  );
}

function Stat({ label, value, color, hint }) {
  return (
    <div style={{ background: C.bg, border: `1px solid ${C.border}`, borderRadius: 10, padding: "9px 11px" }}>
      <div style={{ fontSize: 10.5, color: C.muted, letterSpacing: "0.02em", marginBottom: 3 }}>{label}</div>
      <div style={{ display: "flex", alignItems: "baseline", gap: 6 }}>
        <span style={{ fontFamily: "ui-monospace, monospace", fontSize: 15, fontWeight: 700, color }}>{value}</span>
        {hint && <span style={{ fontSize: 10.5, color: C.muted }}>{hint}</span>}
      </div>
    </div>
  );
}

const LSTM = "lstm";
const COMPARE = "compare";

export default function App() {
  const reduced = useReducedMotion();
  const [view, setView] = useState(LSTM);
  const [presetKey, setPresetKey] = useState("hold");
  const [xs, setXs] = useState(PRESETS.hold.xs);
  const [forgetBias, setForgetBias] = useState(PRESETS.hold.weights.fb);
  const [inputBias, setInputBias] = useState(PRESETS.hold.weights.ib);
  const [cur, setCur] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [vanillaW, setVanillaW] = useState(1.4);
  const timer = useRef(null);

  const preset = PRESETS[presetKey];

  const weights = useMemo(() => ({ ...preset.weights, fb: forgetBias, ib: inputBias }), [preset, forgetBias, inputBias]);
  const cell = useMemo(() => makeCell(weights), [weights]);
  const steps = useMemo(() => runLSTM(cell, xs), [cell, xs]);

  const vanilla = useMemo(() => runVanilla(vanillaW, 1, 0, xs), [vanillaW, xs]);

  useEffect(() => {
    if (cur > xs.length - 1) setCur(xs.length - 1);
  }, [xs, cur]);

  useEffect(() => {
    if (!playing) return;
    if (cur >= xs.length - 1) {
      setPlaying(false);
      return;
    }
    timer.current = setTimeout(() => setCur((c) => Math.min(xs.length - 1, c + 1)), reduced ? 220 : 700);
    return () => clearTimeout(timer.current);
  }, [playing, cur, xs.length, reduced]);

  const applyPreset = (key) => {
    const p = PRESETS[key];
    setPresetKey(key);
    setXs(p.xs);
    setForgetBias(p.weights.fb);
    setInputBias(p.weights.ib);
    setCur(0);
    setPlaying(false);
  };

  const step = steps[cur];

  // The cell-state gradient path: dc_T/dc_t is the product of the forget gates
  // between t and T (the dominant additive-path term). A vanilla RNN instead
  // multiplies the per-step Jacobian W * tanh'(z), which is what decays. Both are
  // computed on the live run so the contrast is real, not asserted.
  const forgetProduct = useMemo(() => {
    const prod = new Array(steps.length).fill(1);
    let acc = 1;
    for (let t = steps.length - 1; t >= 0; t--) {
      prod[t] = acc;
      acc *= steps[t].f;
    }
    return prod;
  }, [steps]);

  const vanillaProduct = useMemo(() => {
    const prod = new Array(vanilla.length).fill(1);
    let acc = 1;
    for (let t = vanilla.length - 1; t >= 0; t--) {
      prod[t] = acc;
      acc *= vanilla[t].jac;
    }
    return prod;
  }, [vanilla]);

  const lstmPathToStart = forgetProduct[0];
  const vanillaPathToStart = vanillaProduct[0];

  const setX = (idx, val) => {
    setXs((arr) => arr.map((v, i) => (i === idx ? val : v)));
  };
  const addStep = () => xs.length < 12 && setXs((arr) => [...arr, 0]);
  const removeStep = () => xs.length > 2 && setXs((arr) => arr.slice(0, -1));

  const focusCss = `
    .lstm-root :focus-visible { outline: 2px solid ${C.accent}; outline-offset: 2px; border-radius: 9px; }
    .lstm-root input[type=range] { accent-color: ${C.accent}; }
    .lstm-root input[type=number] { accent-color: ${C.accent}; }
    @media (prefers-reduced-motion: reduce) {
      .lstm-root * { transition-duration: 0.001ms !important; animation-duration: 0.001ms !important; }
    }
  `;

  const sliderRow = (id, label, value, min, max, stepv, onChange, display, hint) => (
    <div style={{ marginBottom: 12 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
        <label htmlFor={id} style={{ fontSize: 13, color: C.muted, minWidth: 130 }}>
          {label}
        </label>
        <input id={id} type="range" min={min} max={max} step={stepv} value={value} onChange={(e) => onChange(+e.target.value)} style={{ flex: 1, minWidth: 110 }} />
        <span style={{ fontFamily: "ui-monospace, monospace", fontSize: 13, fontWeight: 700, color: C.accent, minWidth: 48, textAlign: "right" }}>
          {display(value)}
        </span>
      </div>
      {hint && <div style={{ fontSize: 11, color: C.muted, margin: "3px 0 0 142px" }}>{hint}</div>}
    </div>
  );

  return (
    <div
      className="lstm-root"
      style={{ fontFamily: "Georgia, 'Iowan Old Style', serif", background: C.bg, minHeight: "100vh", padding: "26px 14px 56px", color: C.ink }}
    >
      <style>{focusCss}</style>
      <div style={{ maxWidth: 720, margin: "0 auto" }}>
        <header style={{ marginBottom: 22 }}>
          <div style={{ fontSize: 11, letterSpacing: "0.14em", textTransform: "uppercase", color: C.muted, marginBottom: 6 }}>
            Deep learning, computed live
          </div>
          <h1 style={{ fontSize: 30, fontWeight: 700, margin: 0, lineHeight: 1.1, letterSpacing: "-0.02em", textWrap: "balance" }}>
            Inside an LSTM Cell
          </h1>
          <p style={{ color: C.ink, fontSize: 15, margin: "10px 0 0", lineHeight: 1.6, maxWidth: "64ch" }}>
            A plain recurrent net rewrites its whole memory at every step, so a gradient flowing back through
            time gets multiplied by a Jacobian over and over until it vanishes. An LSTM keeps a separate cell
            state and learns three gates that decide what to erase, what to write, and what to read. Step one
            cell through a short sequence below. Every gate, every cell update, and the gradient path through
            time are the real computation, not stand-ins.
          </p>
        </header>

        <Card style={{ marginBottom: 16, padding: "14px 18px" }}>
          <Segmented
            ariaLabel="Choose the LSTM cell walkthrough or the comparison against a vanilla recurrent cell"
            value={view}
            onChange={setView}
            options={[
              { value: LSTM, label: "The LSTM cell" },
              { value: COMPARE, label: "Why gradients survive" },
            ]}
          />
        </Card>

        {view === LSTM && (
          <>
            <Card style={{ marginBottom: 16 }}>
              <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 4 }}>Task preset</div>
              <Segmented
                ariaLabel="Pick a task for the cell to run"
                value={presetKey}
                onChange={applyPreset}
                options={Object.entries(PRESETS).map(([k, v]) => ({ value: k, label: v.label }))}
              />
              <p style={{ fontSize: 13, color: C.ink, margin: "10px 0 0", lineHeight: 1.6, maxWidth: "64ch" }}>{preset.blurb}</p>
            </Card>

            <Card style={{ marginBottom: 16 }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", flexWrap: "wrap", gap: 8, marginBottom: 4 }}>
                <span style={{ fontSize: 13, fontWeight: 700 }}>The cell at step {cur + 1}</span>
                <span style={{ fontSize: 12, color: C.muted }}>1 hidden unit, 1 input per step</span>
              </div>
              <p style={{ fontSize: 11, color: C.muted, margin: "0 0 6px", lineHeight: 1.5 }}>
                The thick line is the cell state, the memory highway. The forget gate (x) pinches it; i times g is
                added in. The line thickens as the forget gate opens toward 1.
              </p>
              <CellDiagram step={step} reduced={reduced} />

              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginTop: 4 }}>
                {GATE_KEYS.map((g) => (
                  <GateBar key={g} gkey={g} value={step[g]} reduced={reduced} />
                ))}
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(120px, 1fr))", gap: 10, marginTop: 14 }}>
                <Stat label="input x(t)" value={fmt(step.x)} color={C.ink} />
                <Stat label="kept  f * c(t-1)" value={fmt(step.kept)} color={C.forget} />
                <Stat label="written  i * g" value={fmt(step.add)} color={C.input} />
                <Stat label="cell state c(t)" value={fmt(step.c)} color={C.cell} hint="= kept + written" />
                <Stat label="output h(t)" value={fmt(step.h)} color={C.output} hint="o * tanh(c)" />
              </div>

              <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginTop: 14, alignItems: "center" }}>
                <Btn onClick={() => setCur((c) => Math.max(0, c - 1))} variant="ghost" disabled={cur === 0} ariaLabel="Previous step">
                  Prev
                </Btn>
                <Btn
                  onClick={() => {
                    if (cur >= xs.length - 1) {
                      setCur(0);
                      setPlaying(true);
                    } else {
                      setPlaying((p) => !p);
                    }
                  }}
                  variant="dark"
                  ariaLabel={playing ? "Pause playback" : "Play through the sequence"}
                >
                  {playing ? "Pause" : cur >= xs.length - 1 ? "Replay" : "Play"}
                </Btn>
                <Btn onClick={() => setCur((c) => Math.min(xs.length - 1, c + 1))} variant="ghost" disabled={cur >= xs.length - 1} ariaLabel="Next step">
                  Next
                </Btn>
                <Btn onClick={() => { setCur(0); setPlaying(false); }} variant="ghost" ariaLabel="Reset to the first step">
                  Reset
                </Btn>
                <span style={{ fontSize: 12, color: C.muted, fontFamily: "ui-monospace, monospace" }}>
                  step {cur + 1} / {xs.length}
                </span>
              </div>
            </Card>

            <Card style={{ marginBottom: 16 }}>
              <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 6 }}>Cell state and output across the sequence</div>
              <SequenceChart steps={steps} current={cur} reduced={reduced} />
              <div style={{ display: "flex", gap: 16, marginTop: 8, fontSize: 12, flexWrap: "wrap" }}>
                <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
                  <span style={{ width: 16, height: 3, background: C.cell, borderRadius: 2 }} /> cell state c(t)
                </span>
                <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
                  <span style={{ width: 16, height: 3, background: C.output, borderRadius: 2 }} /> hidden output h(t)
                </span>
              </div>
            </Card>

            <Card style={{ marginBottom: 16 }}>
              <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 8 }}>Input sequence</div>
              <p style={{ fontSize: 11, color: C.muted, margin: "0 0 12px", lineHeight: 1.5 }}>
                One scalar per step. Edit any value, or add and remove steps to make the sequence longer or shorter.
              </p>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(76px, 1fr))", gap: 8 }}>
                {xs.map((v, i) => (
                  <div key={i} style={{ display: "flex", flexDirection: "column", gap: 3 }}>
                    <label htmlFor={`x-${i}`} style={{ fontSize: 10, color: i === cur ? C.accent : C.muted, fontFamily: "ui-monospace, monospace", fontWeight: i === cur ? 700 : 400 }}>
                      x[{i + 1}]
                    </label>
                    <input
                      id={`x-${i}`}
                      type="number"
                      step={0.1}
                      value={v}
                      onChange={(e) => setX(i, +e.target.value)}
                      onFocus={() => setCur(i)}
                      style={{
                        fontFamily: "ui-monospace, monospace",
                        fontSize: 13,
                        padding: "6px 8px",
                        borderRadius: 8,
                        border: `1px solid ${i === cur ? C.accent : C.border}`,
                        background: i === cur ? C.accentSoft : C.bg,
                        color: C.ink,
                        width: "100%",
                        boxSizing: "border-box",
                      }}
                    />
                  </div>
                ))}
              </div>
              <div style={{ display: "flex", gap: 8, marginTop: 12 }}>
                <Btn onClick={addStep} variant="ghost" disabled={xs.length >= 12} ariaLabel="Add a step to the sequence">
                  Add step
                </Btn>
                <Btn onClick={removeStep} variant="ghost" disabled={xs.length <= 2} ariaLabel="Remove the last step">
                  Remove step
                </Btn>
              </div>
            </Card>

            <Card style={{ marginBottom: 16 }}>
              <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 12 }}>Gate weights</div>
              {sliderRow(
                "fb",
                "Forget bias  b_f",
                forgetBias,
                -4,
                5,
                0.1,
                setForgetBias,
                (v) => v.toFixed(1),
                "Raising the forget bias pushes the forget gate toward 1, so the cell holds its memory across steps. This is why LSTMs are often initialized with a positive forget bias.",
              )}
              {sliderRow(
                "ib",
                "Input bias  b_i",
                inputBias,
                -4,
                5,
                0.1,
                setInputBias,
                (v) => v.toFixed(1),
                "Lowering the input bias keeps the input gate near 0, so the cell stops overwriting itself and protects what it already holds.",
              )}
              <div style={{ fontSize: 11, color: C.muted, marginTop: 4, lineHeight: 1.5 }}>
                The remaining weights come from the chosen preset. Each gate computes its pre-activation as a weight
                on the input, a weight on the previous hidden value, and a bias, then a sigmoid (f, i, o) or tanh (g).
              </div>
            </Card>

            <Card>
              <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 8 }}>The four gates, in one place</div>
              <div style={{ display: "grid", gap: 10 }}>
                {GATE_KEYS.map((g) => {
                  const m = GATE_META[g];
                  const eq =
                    g === "f"
                      ? "f = sigmoid(W_f x + U_f h + b_f)"
                      : g === "i"
                        ? "i = sigmoid(W_i x + U_i h + b_i)"
                        : g === "g"
                          ? "g = tanh(W_g x + U_g h + b_g)"
                          : "o = sigmoid(W_o x + U_o h + b_o)";
                  return (
                    <div key={g} style={{ background: m.soft, border: `1px solid ${m.color}33`, borderRadius: 10, padding: "10px 12px" }}>
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 8, flexWrap: "wrap" }}>
                        <span style={{ fontSize: 13, fontWeight: 700, color: m.color }}>
                          {g} &middot; {m.name}
                        </span>
                        <span style={{ fontFamily: "ui-monospace, monospace", fontSize: 11, color: m.color }}>{eq}</span>
                      </div>
                      <div style={{ fontSize: 12.5, color: C.ink, marginTop: 4, lineHeight: 1.5 }}>{m.desc}.</div>
                    </div>
                  );
                })}
              </div>
              <div style={{ background: C.cellSoft, border: `1px solid ${C.cell}33`, borderRadius: 10, padding: "10px 12px", marginTop: 10 }}>
                <div style={{ fontSize: 13, fontWeight: 700, color: C.cell }}>the cell update, protected memory</div>
                <div style={{ fontFamily: "ui-monospace, monospace", fontSize: 12, color: C.cell, margin: "4px 0" }}>
                  c(t) = f &middot; c(t-1) + i &middot; g &nbsp;&nbsp; h(t) = o &middot; tanh(c(t))
                </div>
                <div style={{ fontSize: 12.5, color: C.ink, lineHeight: 1.5 }}>
                  The cell state is updated by addition, not by a full rewrite. The forget gate scales the old memory,
                  the input gate scales the new candidate, and the two are summed. When forget is near 1 and input is
                  near 0, the cell state passes through almost unchanged.
                </div>
              </div>
            </Card>
          </>
        )}

        {view === COMPARE && (
          <>
            <Card style={{ marginBottom: 16 }}>
              <p style={{ fontSize: 14, lineHeight: 1.6, margin: 0, color: C.ink, maxWidth: "64ch" }}>
                Backprop through time sends the gradient at the last step back to the first by multiplying, step by
                step, by the local Jacobian. In a vanilla recurrent net that factor is the recurrent weight times the
                activation derivative, so over T steps the gradient is that factor raised to the T, which decays toward
                zero or blows up. In an LSTM the cell state has an additive update, and the gradient of a late cell
                state with respect to an early one is dominated by the product of the forget gates in between. When
                those gates sit near 1 the product stays near 1, a near-identity path that carries the gradient intact.
              </p>
            </Card>

            <Card style={{ marginBottom: 16 }}>
              <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 10 }}>Gradient path from the last step to the first</div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                <div style={{ background: C.cellSoft, border: `1px solid ${C.cell}44`, borderRadius: 10, padding: "12px 14px" }}>
                  <div style={{ fontSize: 12, fontWeight: 700, color: C.cell, marginBottom: 4 }}>LSTM cell-state path</div>
                  <div style={{ fontSize: 22, fontFamily: "ui-monospace, monospace", fontWeight: 700, color: C.cell }}>{fmtSci(lstmPathToStart)}</div>
                  <div style={{ fontSize: 11, color: C.ink, marginTop: 4, lineHeight: 1.4 }}>product of {steps.length} forget gates</div>
                </div>
                <div style={{ background: C.forgetSoft, border: `1px solid ${C.forget}44`, borderRadius: 10, padding: "12px 14px" }}>
                  <div style={{ fontSize: 12, fontWeight: 700, color: C.forget, marginBottom: 4 }}>Vanilla RNN path</div>
                  <div style={{ fontSize: 22, fontFamily: "ui-monospace, monospace", fontWeight: 700, color: C.forget }}>{fmtSci(vanillaPathToStart)}</div>
                  <div style={{ fontSize: 11, color: C.ink, marginTop: 4, lineHeight: 1.4 }}>product of {vanilla.length} Jacobians W &middot; tanh'(z)</div>
                </div>
              </div>
              <p style={{ fontSize: 11, color: C.muted, margin: "10px 0 0", lineHeight: 1.5 }}>
                Both products run over the same input sequence you set on the cell tab. The LSTM number is the chain of
                forget gates; the RNN number is the chain of recurrent Jacobians for the weight below.
              </p>
            </Card>

            <Card style={{ marginBottom: 16 }}>
              <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 6 }}>Per-step factor along the path</div>
              <FactorChart lstm={steps.map((s) => s.f)} rnn={vanilla.map((s) => s.jac)} reduced={reduced} />
              <div style={{ display: "flex", gap: 16, marginTop: 8, fontSize: 12, flexWrap: "wrap" }}>
                <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
                  <span style={{ width: 16, height: 3, background: C.cell, borderRadius: 2 }} /> forget gate (LSTM)
                </span>
                <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
                  <span style={{ width: 16, height: 3, background: C.forget, borderRadius: 2 }} /> W &middot; tanh'(z) (RNN)
                </span>
                <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
                  <span style={{ width: 16, height: 2, background: C.muted, borderRadius: 2 }} /> the line at 1
                </span>
              </div>
            </Card>

            <Card style={{ marginBottom: 16 }}>
              <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 12 }}>Vanilla RNN recurrent weight</div>
              {sliderRow(
                "vw",
                "Weight W",
                vanillaW,
                0.2,
                2.5,
                0.01,
                setVanillaW,
                (v) => v.toFixed(2),
                "The single recurrent weight applied at every step. Below the point where W times tanh'(z) sits at 1 the gradient vanishes; above it the gradient explodes. There is no value that simply holds, which is the problem the LSTM gate is built to solve.",
              )}
              <p style={{ fontSize: 12.5, color: C.ink, margin: "6px 0 0", lineHeight: 1.6, maxWidth: "64ch" }}>
                The forget gate can sit at any value the data calls for, and a learned bias near a positive value keeps
                it close to 1 by default, so the cell-state path is close to multiply by one over long horizons. The
                vanilla weight has no such freedom: one scalar has to serve both as the memory keeper and as the input
                mixer, and the activation derivative caps it well below 1 across most of its range.
              </p>
            </Card>

            <Card>
              <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 8 }}>The tie to vanishing gradients</div>
              <p style={{ fontSize: 14, lineHeight: 1.6, margin: "0 0 10px", color: C.ink, maxWidth: "64ch" }}>
                The vanishing and exploding gradient artifact showed that a deep or recurrent chain multiplies the
                gradient by act'(z) times the weight at every layer, and that this product is exponential in depth: a
                factor below 1 collapses, a factor above 1 explodes, and only a narrow band stays usable. A vanilla
                recurrent net is exactly that chain unrolled through time, with one shared weight, so long sequences
                are precisely where it fails.
              </p>
              <p style={{ fontSize: 14, lineHeight: 1.6, margin: 0, color: C.ink, maxWidth: "64ch" }}>
                The LSTM does not remove the multiplication, it changes what gets multiplied. By carrying memory in a
                separate cell state that is updated additively, the gradient of a late cell state with respect to an
                early one travels mainly through the product of forget gates, and a forget gate near 1 makes that path
                a near identity. This is the same idea as a residual connection adding a multiply-by-one route through
                a deep net: give the gradient a path it can survive, then let the gates decide when to open and close
                it.
              </p>
            </Card>
          </>
        )}
      </div>
    </div>
  );
}

// Per-step factor on a log axis, LSTM forget gate against the vanilla RNN Jacobian.
// The line at 1 is the watershed: factors below it shrink the running product as
// you walk left toward the first step, factors at it preserve it.
function FactorChart({ lstm, rnn, reduced }) {
  const W = 640;
  const H = 200;
  const padL = 46;
  const padR = 16;
  const padT = 14;
  const padB = 28;
  const n = Math.max(lstm.length, rnn.length);
  const all = lstm.concat(rnn).map((v) => Math.abs(v));
  const logs = all.map((m) => Math.log10(Math.max(m, 1e-6)));
  let lo = Math.min(...logs, 0);
  let hi = Math.max(...logs, 0);
  if (hi - lo < 1) {
    lo -= 0.5;
    hi += 0.5;
  }
  const span = hi - lo;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;
  const xAt = (i) => padL + (n === 1 ? plotW / 2 : (i / (n - 1)) * plotW);
  const yAt = (lg) => padT + (1 - (lg - lo) / span) * plotH;
  const lineFor = (arr) => arr.map((v, i) => `${i === 0 ? "M" : "L"}${xAt(i).toFixed(1)},${yAt(Math.log10(Math.max(Math.abs(v), 1e-6))).toFixed(1)}`).join("");
  const oneY = yAt(0);

  const ticks = [];
  for (let d = Math.ceil(lo); d <= Math.floor(hi); d++) ticks.push(d);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label="Per-step gradient factor on a log scale: the LSTM forget gate stays near one while the vanilla recurrent factor sits below it." style={{ display: "block" }}>
      {ticks.map((d) => (
        <g key={d}>
          <line x1={padL} y1={yAt(d)} x2={W - padR} y2={yAt(d)} stroke={C.grid} strokeWidth={1} />
          <text x={padL - 6} y={yAt(d) + 3} textAnchor="end" fontSize={9} fill={C.muted} fontFamily="ui-monospace, monospace">
            {`1e${d}`}
          </text>
        </g>
      ))}
      <line x1={padL} y1={oneY} x2={W - padR} y2={oneY} stroke={C.muted} strokeWidth={1.4} strokeDasharray="4 3" opacity={0.7} />
      <path d={lineFor(rnn)} fill="none" stroke={C.forget} strokeWidth={2.2} strokeLinejoin="round" strokeLinecap="round" />
      <path d={lineFor(lstm)} fill="none" stroke={C.cell} strokeWidth={2.4} strokeLinejoin="round" strokeLinecap="round" style={{ transition: reduced ? "none" : "stroke 200ms ease" }} />
      {lstm.map((v, i) => (
        <circle key={`l${i}`} cx={xAt(i)} cy={yAt(Math.log10(Math.max(Math.abs(v), 1e-6)))} r={2.6} fill={C.cell} />
      ))}
      {rnn.map((v, i) => (
        <circle key={`r${i}`} cx={xAt(i)} cy={yAt(Math.log10(Math.max(Math.abs(v), 1e-6)))} r={2.6} fill={C.forget} />
      ))}
      {Array.from({ length: n }).map((_, i) => (
        <text key={`x${i}`} x={xAt(i)} y={H - padB + 16} textAnchor="middle" fontSize={9} fill={C.muted} fontFamily="ui-monospace, monospace">
          {i + 1}
        </text>
      ))}
    </svg>
  );
}
