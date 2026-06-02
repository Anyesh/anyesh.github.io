import { useState, useEffect, useMemo, useRef } from "react";

export const meta = {
  title: "Activation Functions, Curve by Curve",
  category: "Deep Learning",
  description:
    "Plot seven activation functions against their exact analytic derivatives, drag an input point across the saturation regions, and stack derivative magnitudes to watch sigmoid and tanh vanish while the ReLU family keeps the gradient alive.",
  date: "2026-05-22",
  tags: ["activation-functions", "gradients", "vanishing-gradients", "deep-learning", "relu"],
};

const C = {
  bg: "#f7f5f2",
  card: "#ffffff",
  border: "#e7e1d8",
  ink: "#1b1916",
  muted: "#6f675e",
  faint: "#9b938a",
  accent: "#c0561f",
  accentSoft: "#f6e9e0",
  deriv: "#2f6f8f",
  derivSoft: "#e3eef3",
  grid: "#efebe4",
  axis: "#cfc7ba",
  warn: "#b23b5e",
  warnSoft: "#f7e6ec",
  good: "#2f7d54",
  goodSoft: "#e6f1ea",
};

const EASE = "cubic-bezier(0.23, 1, 0.32, 1)";

const SQRT_2 = Math.sqrt(2);

const sigmoidScalar = (x) => 1 / (1 + Math.exp(-x));

// Abramowitz and Stegun 7.1.26 rational approximation, max abs error 1.5e-7,
// good enough for the exact erf-based GELU at plotting resolution.
function erf(x) {
  const sign = x < 0 ? -1 : 1;
  const ax = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * ax);
  const y =
    1 -
    ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t +
      0.254829592) *
      t *
      Math.exp(-ax * ax);
  return sign * y;
}

function makeActivations(p) {
  const { leak, alpha, beta } = p;
  return {
    sigmoid: {
      key: "sigmoid",
      label: "Sigmoid",
      color: "#c0561f",
      f: (x) => sigmoidScalar(x),
      df: (x) => {
        const s = sigmoidScalar(x);
        return s * (1 - s);
      },
      range: "(0, 1)",
      zeroCentered: false,
      derivRange: "(0, 0.25]",
      saturates: "Both tails flatten; derivative to 0",
      dead: "No hard dead zone, but tails barely learn",
      use: "Final layer for a single probability.",
    },
    tanh: {
      key: "tanh",
      label: "Tanh",
      color: "#8a5a2b",
      f: (x) => Math.tanh(x),
      df: (x) => {
        const t = Math.tanh(x);
        return 1 - t * t;
      },
      range: "(-1, 1)",
      zeroCentered: true,
      derivRange: "(0, 1]",
      saturates: "Both tails flatten; derivative to 0",
      dead: "No hard dead zone, but tails barely learn",
      use: "Zero-centered hidden units, recurrent cells.",
    },
    relu: {
      key: "relu",
      label: "ReLU",
      color: "#2f7d54",
      f: (x) => (x > 0 ? x : 0),
      df: (x) => (x > 0 ? 1 : 0),
      range: "[0, infinity)",
      zeroCentered: false,
      derivRange: "{0, 1}",
      saturates: "Negative half is exactly flat",
      dead: "Yes: a unit pushed below 0 gets zero gradient",
      use: "Default hidden activation in most deep nets.",
    },
    leaky: {
      key: "leaky",
      label: "Leaky ReLU",
      color: "#3f8fb0",
      f: (x) => (x > 0 ? x : leak * x),
      df: (x) => (x > 0 ? 1 : leak),
      range: "(-infinity, infinity)",
      zeroCentered: false,
      derivRange: `{${leak}, 1}`,
      saturates: "Never flat; negative slope stays nonzero",
      dead: `No: negative side passes ${leak} of the gradient`,
      use: "ReLU drop-in when units keep dying.",
    },
    elu: {
      key: "elu",
      label: "ELU",
      color: "#7a4fb5",
      f: (x) => (x > 0 ? x : alpha * (Math.exp(x) - 1)),
      df: (x) => (x > 0 ? 1 : alpha * Math.exp(x)),
      range: `(-${alpha}, infinity)`,
      zeroCentered: false,
      derivRange: "(0, 1]",
      saturates: "Saturates softly to -alpha on the left",
      dead: "No: negative side keeps a smooth nonzero slope",
      use: "Smoother negatives, mean activations near zero.",
    },
    gelu: {
      key: "gelu",
      label: "GELU",
      color: "#b08300",
      f: (x) => 0.5 * x * (1 + erf(x / SQRT_2)),
      df: (x) => {
        const cdf = 0.5 * (1 + erf(x / SQRT_2));
        const pdf = Math.exp(-(x * x) / 2) / Math.sqrt(2 * Math.PI);
        return cdf + x * pdf;
      },
      range: "approx (-0.17, infinity)",
      zeroCentered: false,
      derivRange: "approx (-0.13, 1.08)",
      saturates: "Left tail to 0; right tail to 1",
      dead: "No: small negative values still pass gradient",
      use: "Transformers and modern language models.",
    },
    silu: {
      key: "silu",
      label: "SiLU / Swish",
      color: "#c0394f",
      f: (x) => x * sigmoidScalar(beta * x),
      df: (x) => {
        const s = sigmoidScalar(beta * x);
        return s + beta * x * s * (1 - s);
      },
      range: "approx (-0.28, infinity)",
      zeroCentered: false,
      derivRange: "approx (-0.10, 1.10)",
      saturates: "Left tail to 0; right tail to 1",
      dead: "No: smooth nonzero gradient near and below 0",
      use: "EfficientNet, many vision and LM backbones.",
    },
  };
}

const ORDER = ["sigmoid", "tanh", "relu", "leaky", "elu", "gelu", "silu"];

function niceTicks(lo, hi, count) {
  const span = hi - lo;
  const raw = span / count;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
  const ticks = [];
  const start = Math.ceil(lo / step) * step;
  for (let v = start; v <= hi + 1e-9; v += step) {
    ticks.push(Math.abs(v) < 1e-9 ? 0 : v);
  }
  return ticks;
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

function Card({ children, style }) {
  return (
    <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 14, padding: 18, ...style }}>
      {children}
    </div>
  );
}

const PLOT_W = 640;
const PLOT_H = 360;
const PAD = { l: 44, r: 16, t: 14, b: 30 };

function Plot({ acts, selected, xRange, point, onPointChange, showDeriv, reduce }) {
  const ref = useRef(null);
  const [xlo, xhi] = xRange;
  const innerW = PLOT_W - PAD.l - PAD.r;
  const innerH = PLOT_H - PAD.t - PAD.b;

  const yExtent = useMemo(() => {
    let lo = 0;
    let hi = 0;
    const steps = 240;
    for (const k of selected) {
      const a = acts[k];
      for (let i = 0; i <= steps; i++) {
        const x = xlo + (i / steps) * (xhi - xlo);
        const fv = a.f(x);
        if (Number.isFinite(fv)) {
          lo = Math.min(lo, fv);
          hi = Math.max(hi, fv);
        }
        if (showDeriv) {
          const dv = a.df(x);
          if (Number.isFinite(dv)) {
            lo = Math.min(lo, dv);
            hi = Math.max(hi, dv);
          }
        }
      }
    }
    const padded = (hi - lo) * 0.08 || 0.5;
    return [lo - padded, hi + padded];
  }, [acts, selected, xlo, xhi, showDeriv]);

  const [ylo, yhi] = yExtent;
  const sx = (x) => PAD.l + ((x - xlo) / (xhi - xlo)) * innerW;
  const sy = (y) => PAD.t + (1 - (y - ylo) / (yhi - ylo)) * innerH;

  const pathFor = (fn) => {
    const steps = 320;
    let d = "";
    let prevFinite = false;
    for (let i = 0; i <= steps; i++) {
      const x = xlo + (i / steps) * (xhi - xlo);
      const y = fn(x);
      if (!Number.isFinite(y)) {
        prevFinite = false;
        continue;
      }
      const px = sx(x).toFixed(2);
      const py = sy(y).toFixed(2);
      d += `${prevFinite ? "L" : "M"}${px} ${py} `;
      prevFinite = true;
    }
    return d.trim();
  };

  const xTicks = niceTicks(xlo, xhi, 8);
  const yTicks = niceTicks(ylo, yhi, 6);

  const xFromClient = (clientX) => {
    const rect = ref.current.getBoundingClientRect();
    const scale = PLOT_W / rect.width;
    const local = (clientX - rect.left) * scale;
    const frac = (local - PAD.l) / innerW;
    return xlo + Math.max(0, Math.min(1, frac)) * (xhi - xlo);
  };

  const dragging = useRef(false);
  const onDown = (e) => {
    dragging.current = true;
    e.currentTarget.setPointerCapture(e.pointerId);
    onPointChange(xFromClient(e.clientX));
  };
  const onMove = (e) => {
    if (!dragging.current) return;
    onPointChange(xFromClient(e.clientX));
  };
  const onUp = () => {
    dragging.current = false;
  };

  const px = sx(point);
  const zeroY = ylo <= 0 && yhi >= 0 ? sy(0) : null;

  return (
    <svg
      ref={ref}
      viewBox={`0 0 ${PLOT_W} ${PLOT_H}`}
      width="100%"
      role="img"
      aria-label={`Activation functions and ${
        showDeriv ? "their derivatives " : ""
      }plotted over x from ${xlo} to ${xhi}. Input point at x equals ${point.toFixed(2)}.`}
      style={{ display: "block", touchAction: "none", cursor: "ew-resize" }}
      onPointerDown={onDown}
      onPointerMove={onMove}
      onPointerUp={onUp}
      onPointerLeave={onUp}
    >
      <rect x={PAD.l} y={PAD.t} width={innerW} height={innerH} fill={C.bg} rx={6} />

      {yTicks.map((t, i) => (
        <g key={`y${i}`}>
          <line x1={PAD.l} y1={sy(t)} x2={PAD.l + innerW} y2={sy(t)} stroke={C.grid} strokeWidth={1} />
          <text x={PAD.l - 6} y={sy(t) + 3} textAnchor="end" fontSize={9} fill={C.faint} fontFamily="ui-monospace, monospace">
            {t}
          </text>
        </g>
      ))}
      {xTicks.map((t, i) => (
        <g key={`x${i}`}>
          <line x1={sx(t)} y1={PAD.t} x2={sx(t)} y2={PAD.t + innerH} stroke={C.grid} strokeWidth={1} />
          <text x={sx(t)} y={PAD.t + innerH + 14} textAnchor="middle" fontSize={9} fill={C.faint} fontFamily="ui-monospace, monospace">
            {t}
          </text>
        </g>
      ))}

      {zeroY !== null && <line x1={PAD.l} y1={zeroY} x2={PAD.l + innerW} y2={zeroY} stroke={C.axis} strokeWidth={1.4} />}
      {xlo <= 0 && xhi >= 0 && <line x1={sx(0)} y1={PAD.t} x2={sx(0)} y2={PAD.t + innerH} stroke={C.axis} strokeWidth={1.4} />}

      <line
        x1={px}
        y1={PAD.t}
        x2={px}
        y2={PAD.t + innerH}
        stroke={C.ink}
        strokeWidth={1.4}
        strokeDasharray="3 3"
        opacity={0.55}
      />

      {showDeriv &&
        selected.map((k) => (
          <path
            key={`d-${k}`}
            d={pathFor(acts[k].df)}
            fill="none"
            stroke={acts[k].color}
            strokeWidth={1.6}
            strokeDasharray="5 4"
            opacity={0.85}
            strokeLinejoin="round"
            strokeLinecap="round"
            style={{ transition: reduce ? "none" : `d 220ms ${EASE}` }}
          />
        ))}

      {selected.map((k) => (
        <path
          key={`f-${k}`}
          d={pathFor(acts[k].f)}
          fill="none"
          stroke={acts[k].color}
          strokeWidth={2.4}
          strokeLinejoin="round"
          strokeLinecap="round"
          style={{ transition: reduce ? "none" : `d 220ms ${EASE}` }}
        />
      ))}

      {selected.map((k) => {
        const y = acts[k].f(point);
        if (!Number.isFinite(y)) return null;
        return <circle key={`pt-${k}`} cx={px} cy={sy(y)} r={4} fill={acts[k].color} stroke={C.card} strokeWidth={1.5} />;
      })}

      <text x={PAD.l + innerW} y={PAD.t + 12} textAnchor="end" fontSize={10} fill={C.muted} fontFamily="ui-monospace, monospace">
        x = {point.toFixed(2)}
      </text>
    </svg>
  );
}

function Toggle({ on, onClick, color, label, sub }) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={on}
      style={{
        display: "flex",
        alignItems: "center",
        gap: 8,
        padding: "7px 11px",
        borderRadius: 9,
        border: `1.5px solid ${on ? color : C.border}`,
        background: on ? color + "14" : C.card,
        color: on ? C.ink : C.muted,
        cursor: "pointer",
        fontFamily: "inherit",
        fontSize: 12.5,
        fontWeight: on ? 700 : 500,
        transition: `transform 140ms ${EASE}, border-color 160ms ease, background 160ms ease`,
      }}
      onMouseDown={(e) => (e.currentTarget.style.transform = "scale(0.97)")}
      onMouseUp={(e) => (e.currentTarget.style.transform = "scale(1)")}
      onMouseLeave={(e) => (e.currentTarget.style.transform = "scale(1)")}
    >
      <span style={{ width: 14, height: 3, borderRadius: 2, background: color, flexShrink: 0 }} />
      <span style={{ display: "flex", flexDirection: "column", lineHeight: 1.2, textAlign: "left" }}>
        <span>{label}</span>
        {sub && <span style={{ fontSize: 10, color: C.faint, fontWeight: 400 }}>{sub}</span>}
      </span>
    </button>
  );
}

function Slider({ id, label, min, max, step, value, onChange, display }) {
  return (
    <div>
      <label htmlFor={id} style={{ fontSize: 12.5, color: C.muted, display: "flex", justifyContent: "space-between", marginBottom: 5 }}>
        <span>{label}</span>
        <span style={{ fontFamily: "ui-monospace, monospace", color: C.accent, fontWeight: 700 }}>{display}</span>
      </label>
      <input id={id} type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(+e.target.value)} aria-label={label} />
    </div>
  );
}

function ReadoutRow({ a, point }) {
  const fv = a.f(point);
  const dv = a.df(point);
  const dead = Math.abs(dv) < 0.02;
  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "minmax(96px, 1.3fr) 1fr 1fr",
        gap: 6,
        alignItems: "center",
        padding: "7px 9px",
        borderRadius: 8,
        background: dead ? C.warnSoft : C.bg,
        border: `1px solid ${dead ? C.warn + "44" : C.border}`,
      }}
    >
      <span style={{ display: "flex", alignItems: "center", gap: 7, fontSize: 12.5, fontWeight: 700, color: C.ink }}>
        <span style={{ width: 12, height: 3, borderRadius: 2, background: a.color }} />
        {a.label}
      </span>
      <span style={{ fontFamily: "ui-monospace, monospace", fontSize: 12.5, color: C.ink }}>f = {fv.toFixed(3)}</span>
      <span style={{ fontFamily: "ui-monospace, monospace", fontSize: 12.5, color: dead ? C.warn : C.deriv, fontWeight: dead ? 700 : 400 }}>
        f&prime; = {dv.toFixed(3)}
        {dead && <span style={{ fontSize: 9, marginLeft: 4 }}>saturated</span>}
      </span>
    </div>
  );
}

const PROP_COLS = [
  { key: "range", label: "Output range" },
  { key: "derivRange", label: "Derivative range" },
  { key: "zeroCentered", label: "Zero-centered" },
  { key: "saturates", label: "Saturation" },
  { key: "dead", label: "Dead units" },
];

export default function App() {
  const reduce = usePrefersReducedMotion();
  const [selected, setSelected] = useState(["sigmoid", "relu", "gelu"]);
  const [leak, setLeak] = useState(0.1);
  const [alpha, setAlpha] = useState(1);
  const [beta, setBeta] = useState(1);
  const [span, setSpan] = useState(6);
  const [point, setPoint] = useState(-1.4);
  const [showDeriv, setShowDeriv] = useState(true);
  const [depth, setDepth] = useState(8);
  const [productKey, setProductKey] = useState("sigmoid");

  const acts = useMemo(() => makeActivations({ leak, alpha, beta }), [leak, alpha, beta]);
  const xRange = [-span, span];

  const orderedSelected = ORDER.filter((k) => selected.includes(k));

  const toggle = (k) => {
    setSelected((prev) => (prev.includes(k) ? prev.filter((x) => x !== k) : [...prev, k]));
  };

  const productAct = acts[productKey];
  const dAtPoint = productAct.df(point);
  const productMag = Math.abs(dAtPoint);
  const chained = Math.pow(productMag, depth);

  const focusCss = `
    .af-root :focus-visible { outline: 2px solid ${C.accent}; outline-offset: 2px; border-radius: 9px; }
    .af-root input[type=range] { accent-color: ${C.accent}; width: 100%; }
    .af-root input[type=range]:focus-visible { outline: 2px solid ${C.accent}; outline-offset: 3px; }
    @media (prefers-reduced-motion: reduce) {
      .af-root * { transition-duration: 0.001ms !important; animation-duration: 0.001ms !important; }
    }
  `;

  const subFor = (k) => {
    if (k === "leaky") return `slope ${leak}`;
    if (k === "elu") return `alpha ${alpha}`;
    if (k === "silu") return `beta ${beta}`;
    return null;
  };

  return (
    <div
      className="af-root"
      style={{
        fontFamily: "'Iowan Old Style', 'Palatino Linotype', Palatino, Georgia, serif",
        background: C.bg,
        minHeight: "100vh",
        padding: "30px 14px 56px",
        color: C.ink,
      }}
    >
      <style>{focusCss}</style>

      <div style={{ maxWidth: 720, margin: "0 auto" }}>
        <header style={{ marginBottom: 22 }}>
          <div style={{ fontSize: 11, letterSpacing: "0.14em", textTransform: "uppercase", color: C.faint, marginBottom: 6 }}>
            Deep learning, computed live
          </div>
          <h1 style={{ fontSize: 30, fontWeight: 700, margin: 0, lineHeight: 1.1, letterSpacing: "-0.02em", textWrap: "balance" }}>
            Activation Functions, Curve by Curve
          </h1>
          <p style={{ color: C.muted, fontSize: 15, lineHeight: 1.6, margin: "10px 0 0", maxWidth: "64ch" }}>
            An activation is the nonlinearity a neuron applies after its weighted sum. Without it a stack of layers
            collapses into one linear map, so the network could never bend a decision boundary. Its derivative is what
            backprop multiplies as the gradient flows back, so the shape of that derivative quietly decides how well a
            deep net learns. Overlay a few below and watch where each one keeps its slope alive and where it flatlines.
          </p>
        </header>

        <Card style={{ marginBottom: 16 }}>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 7, marginBottom: 14 }}>
            {ORDER.map((k) => (
              <Toggle key={k} on={selected.includes(k)} onClick={() => toggle(k)} color={acts[k].color} label={acts[k].label} sub={subFor(k)} />
            ))}
          </div>

          <Plot
            acts={acts}
            selected={orderedSelected}
            xRange={xRange}
            point={point}
            onPointChange={(x) => setPoint(+x.toFixed(3))}
            showDeriv={showDeriv}
            reduce={reduce}
          />

          <div style={{ display: "flex", flexWrap: "wrap", gap: 14, alignItems: "center", marginTop: 12 }}>
            <Toggle on={showDeriv} onClick={() => setShowDeriv((v) => !v)} color={C.deriv} label="Show derivatives" sub="dashed lines" />
            <span style={{ fontSize: 12, color: C.muted, lineHeight: 1.5, flex: "1 1 200px" }}>
              Solid lines are f(x); dashed lines are the exact f&prime;(x). Drag anywhere on the chart to move the input
              point.
            </span>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: 12, marginTop: 14 }}>
            <Slider id="af-span" label="x range" min={2} max={12} step={1} value={span} onChange={setSpan} display={`+/- ${span}`} />
            <Slider id="af-point" label="input x" min={-span} max={span} step={0.01} value={point} onChange={setPoint} display={point.toFixed(2)} />
            <Slider id="af-leak" label="Leaky slope" min={0} max={0.4} step={0.01} value={leak} onChange={setLeak} display={leak.toFixed(2)} />
            <Slider id="af-alpha" label="ELU alpha" min={0.2} max={2} step={0.1} value={alpha} onChange={setAlpha} display={alpha.toFixed(1)} />
            <Slider id="af-beta" label="Swish beta" min={0.2} max={3} step={0.1} value={beta} onChange={setBeta} display={beta.toFixed(1)} />
          </div>
        </Card>

        <Card style={{ marginBottom: 16 }}>
          <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 4 }}>At x = {point.toFixed(2)}</div>
          <p style={{ fontSize: 13, color: C.muted, margin: "0 0 12px", lineHeight: 1.55, maxWidth: "64ch" }}>
            A derivative near zero is a saturation region: the unit barely responds, so almost no gradient passes through
            it. For ReLU every x below 0 has derivative exactly 0, the dead zone where a unit stuck on the negative side
            stops updating. Leaky ReLU and ELU keep a nonzero slope there, so the gradient survives.
          </p>
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            {orderedSelected.length === 0 && (
              <div style={{ fontSize: 13, color: C.faint, padding: "10px 0" }}>Select at least one activation above.</div>
            )}
            {orderedSelected.map((k) => (
              <ReadoutRow key={k} a={acts[k]} point={point} />
            ))}
          </div>
        </Card>

        <Card style={{ marginBottom: 16 }}>
          <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 4 }}>Gradient through a deep stack</div>
          <p style={{ fontSize: 13, color: C.muted, margin: "0 0 14px", lineHeight: 1.55, maxWidth: "64ch" }}>
            Backprop pushes the gradient back through every layer by multiplying by each unit&apos;s local derivative. Stack
            the chosen activation {depth} layers deep at this operating point and the factor that reaches the first layer
            is the product of {depth} copies of |f&prime;(x)|. When that factor is below 1 it shrinks geometrically, the
            vanishing-gradient problem; the ReLU family holds it at 1 on the active half.
          </p>

          <div style={{ display: "flex", flexWrap: "wrap", gap: 7, marginBottom: 14 }}>
            {ORDER.map((k) => (
              <Toggle key={k} on={productKey === k} onClick={() => setProductKey(k)} color={acts[k].color} label={acts[k].label} sub={subFor(k)} />
            ))}
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: 12, marginBottom: 14 }}>
            <Slider id="af-depth" label="Stack depth" min={1} max={30} step={1} value={depth} onChange={setDepth} display={`${depth}`} />
            <Slider id="af-point2" label="operating x" min={-span} max={span} step={0.01} value={point} onChange={setPoint} display={point.toFixed(2)} />
          </div>

          <div style={{ display: "flex", gap: 8, alignItems: "flex-end", marginBottom: 12, height: 96 }}>
            {Array.from({ length: depth }).map((_, i) => {
              const factor = Math.pow(productMag, i + 1);
              const h = Math.max(2, Math.min(1, factor) * 92);
              const tiny = factor < 0.05;
              return (
                <div
                  key={i}
                  title={`after ${i + 1} layer${i ? "s" : ""}: ${factor.toExponential(2)}`}
                  style={{
                    flex: 1,
                    minWidth: 3,
                    height: h,
                    borderRadius: "3px 3px 0 0",
                    background: tiny ? C.warn : productAct.color,
                    opacity: tiny ? 0.85 : 0.55 + 0.45 * Math.min(1, factor),
                    transition: reduce ? "none" : `height 240ms ${EASE}, background 200ms ease`,
                  }}
                />
              );
            })}
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))", gap: 10 }}>
            <Stat label="|f'(x)| per layer" value={productMag.toFixed(4)} />
            <Stat
              label={`product over ${depth}`}
              value={chained < 1e-4 ? chained.toExponential(2) : chained.toFixed(4)}
              tone={chained < 0.05 ? "warn" : chained > 0.5 ? "good" : "neutral"}
            />
            <Stat
              label="verdict"
              value={productMag >= 0.999 ? "preserved" : productMag <= 0.001 ? "dead" : chained < 0.05 ? "vanishing" : "decaying"}
              tone={productMag >= 0.999 ? "good" : chained < 0.05 ? "warn" : "neutral"}
            />
          </div>
          <p style={{ fontSize: 12, color: C.muted, margin: "12px 0 0", lineHeight: 1.5 }}>
            Sigmoid caps its derivative at 0.25, so even at its steepest point a stack multiplies by at most 0.25 each
            layer and the gradient collapses fast. Tanh peaks at 1 only at the origin and falls off on either side. ReLU
            on its active half keeps each factor at exactly 1, so the product stays at 1 no matter how deep the stack.
          </p>
        </Card>

        <Card style={{ marginBottom: 16 }}>
          <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 12 }}>Properties at a glance</div>
          <div style={{ overflowX: "auto" }}>
            <table style={{ borderCollapse: "collapse", width: "100%", minWidth: 540, fontSize: 12 }}>
              <thead>
                <tr>
                  <th style={thStyle("left")}>Function</th>
                  {PROP_COLS.map((c) => (
                    <th key={c.key} style={thStyle("left")}>
                      {c.label}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {ORDER.map((k) => {
                  const a = acts[k];
                  const on = selected.includes(k);
                  return (
                    <tr key={k} style={{ background: on ? a.color + "0f" : "transparent" }}>
                      <td style={tdStyle()}>
                        <span style={{ display: "flex", alignItems: "center", gap: 7, fontWeight: 700 }}>
                          <span style={{ width: 12, height: 3, borderRadius: 2, background: a.color }} />
                          {a.label}
                        </span>
                      </td>
                      {PROP_COLS.map((c) => (
                        <td key={c.key} style={tdStyle()}>
                          {c.key === "zeroCentered" ? (
                            <span style={{ color: a.zeroCentered ? C.good : C.muted, fontWeight: a.zeroCentered ? 700 : 400 }}>
                              {a.zeroCentered ? "yes" : "no"}
                            </span>
                          ) : c.key === "dead" ? (
                            <span style={{ color: a.dead.startsWith("Yes") ? C.warn : C.ink }}>{a[c.key]}</span>
                          ) : c.key === "range" || c.key === "derivRange" ? (
                            <span style={{ fontFamily: "ui-monospace, monospace", fontSize: 11 }}>{a[c.key]}</span>
                          ) : (
                            a[c.key]
                          )}
                        </td>
                      ))}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <p style={{ fontSize: 11.5, color: C.faint, margin: "12px 0 0", lineHeight: 1.5 }}>
            GELU uses the exact erf form, f(x) = 0.5 x (1 + erf(x / sqrt 2)); SiLU is x times sigmoid(beta x). Their
            derivative ranges dip slightly below zero because each has a shallow dip just left of the origin, which is
            why their stated ranges are approximate.
          </p>
        </Card>

        <Card style={{ background: C.accentSoft, borderColor: `${C.accent}33` }}>
          <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 10 }}>Why the curve shape decides training</div>
          <p style={{ fontSize: 14, lineHeight: 1.65, color: "#5d4226", margin: "0 0 10px", maxWidth: "66ch" }}>
            Zero-centered activations like tanh let a layer push its output both above and below zero, so the next
            layer&apos;s gradients are not all forced to share a sign; sigmoid, stuck in (0, 1), biases every downstream
            update the same way and slows learning. Non-saturating activations matter even more: once a sigmoid or tanh
            unit lands in its flat tail, its derivative is near zero and it stops contributing to the gradient, so depth
            multiplies many small factors into nothing.
          </p>
          <p style={{ fontSize: 14, lineHeight: 1.65, color: "#5d4226", margin: 0, maxWidth: "66ch" }}>
            ReLU avoids saturation on its active half, which is why it unlocked very deep networks, but it pays for that
            with a hard dead zone on the left. Leaky ReLU, ELU, GELU, and SiLU each keep a nonzero slope below zero so a
            unit can recover, trading a little of ReLU&apos;s simplicity for gradients that never fully die. Drag the
            input into the tails above and the readouts make the tradeoff concrete.
          </p>
        </Card>
      </div>
    </div>
  );
}

function thStyle(align) {
  return {
    textAlign: align,
    padding: "6px 10px",
    borderBottom: `1.5px solid ${C.border}`,
    fontSize: 10.5,
    letterSpacing: "0.04em",
    textTransform: "uppercase",
    color: C.muted,
    fontWeight: 700,
    whiteSpace: "nowrap",
  };
}

function tdStyle() {
  return {
    padding: "7px 10px",
    borderBottom: `1px solid ${C.grid}`,
    color: C.ink,
    verticalAlign: "top",
    lineHeight: 1.4,
  };
}

function Stat({ label, value, tone = "neutral" }) {
  const color = tone === "warn" ? C.warn : tone === "good" ? C.good : C.ink;
  const bg = tone === "warn" ? C.warnSoft : tone === "good" ? C.goodSoft : C.bg;
  const bd = tone === "warn" ? C.warn + "33" : tone === "good" ? C.good + "33" : C.border;
  return (
    <div style={{ background: bg, borderRadius: 9, padding: "9px 12px", border: `1px solid ${bd}` }}>
      <div style={{ fontSize: 10, letterSpacing: "0.05em", textTransform: "uppercase", color: C.muted, fontWeight: 700, marginBottom: 3 }}>
        {label}
      </div>
      <div style={{ fontFamily: "ui-monospace, monospace", fontSize: 17, fontWeight: 700, color, lineHeight: 1, fontVariantNumeric: "tabular-nums" }}>
        {value}
      </div>
    </div>
  );
}
