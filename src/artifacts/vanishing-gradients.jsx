import { useState, useEffect, useMemo, useRef } from "react";

export const meta = {
  title: "Vanishing and Exploding Gradients",
  category: "Deep Learning",
  description:
    "Backprop turns a deep chain into a product of per-layer Jacobians. Tune the depth, activation, and weight scale and watch the real per-layer gradient norm decay toward the input or blow up, then see the same thing happen through time in a recurrent net.",
  date: "2026-06-02",
  tags: ["gradients", "deep-learning", "backpropagation", "rnn", "initialization"],
};

const C = {
  bg: "#f7f5f2",
  card: "#ffffff",
  border: "#e6e1d9",
  ink: "#1b1916",
  muted: "#7c7368",
  accent: "#c0561f",
  accentSoft: "#f6ece5",
  vanish: "#2a5298",
  vanishSoft: "#e7eef8",
  explode: "#b23b5e",
  explodeSoft: "#f8e7ee",
  stable: "#2e7d51",
  stableSoft: "#e6f1ea",
  grid: "#eee9e1",
};

const EASE = "cubic-bezier(0.23, 1, 0.32, 1)";

const ACT = {
  sigmoid: {
    name: "sigmoid",
    f: (x) => 1 / (1 + Math.exp(-x)),
    d: (x) => {
      const s = 1 / (1 + Math.exp(-x));
      return s * (1 - s);
    },
    dmax: 0.25,
    note: "derivative peaks at 0.25 and saturates to 0 in the tails",
  },
  tanh: {
    name: "tanh",
    f: (x) => Math.tanh(x),
    d: (x) => {
      const t = Math.tanh(x);
      return 1 - t * t;
    },
    dmax: 1,
    note: "derivative peaks at 1 at the origin and saturates to 0 in the tails",
  },
  relu: {
    name: "ReLU",
    f: (x) => (x > 0 ? x : 0),
    d: (x) => (x > 0 ? 1 : 0),
    dmax: 1,
    note: "derivative is exactly 1 on the active branch, 0 when off",
  },
};

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

// Positive weights near `scale` keep the chain on one branch (so ReLU stays
// active) and keep signs consistent, so the gradient product reads as a clean
// geometric profile rather than sign-cancelling noise. The seed only jitters
// the magnitudes by +/-10 percent so a reader can vary the exact chain.
function buildWeights(L, scale, seed) {
  const r = mulberry32(seed >>> 0);
  const w = new Array(L);
  for (let l = 0; l < L; l++) {
    const jitter = 1 + (r() - 0.5) * 0.2;
    w[l] = scale * jitter;
  }
  return w;
}

// Forward pass of a scalar chain: h0 = x, z_l = w_l * h_{l-1}, h_l = act(z_l).
// Loss = 0.5 * h_L^2, so dL/dh_L = h_L. Backward gives dL/dz_l at every layer,
// which is the gradient that flows into layer l. Each step multiplies by the
// local Jacobian factor act'(z_l) * w_l, the exact thing the chain rule chains.
function runChain(x, w, actKey) {
  const a = ACT[actKey];
  const L = w.length;
  const h = new Array(L + 1);
  const z = new Array(L);
  h[0] = x;
  for (let l = 0; l < L; l++) {
    z[l] = w[l] * h[l];
    h[l + 1] = a.f(z[l]);
  }
  const dz = new Array(L);
  const factor = new Array(L);
  let dh = h[L];
  for (let l = L - 1; l >= 0; l--) {
    const dprime = a.d(z[l]);
    dz[l] = dh * dprime;
    factor[l] = dprime * w[l];
    dh = dz[l] * w[l];
  }
  return { h, z, dz, factor };
}

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

const fmtSci = (v) => {
  if (!isFinite(v)) return "inf";
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a >= 1e4 || a < 1e-3) return v.toExponential(2);
  if (a >= 100) return v.toFixed(1);
  if (a >= 1) return v.toFixed(3);
  return v.toFixed(4);
};

function regimeOf(ratio) {
  // ratio = grad(input layer) / grad(output layer). Far below 1 means the signal
  // collapsed on the way back to the input (vanishing); far above 1 means it grew
  // without bound (exploding). The decade-wide band around 1 reads as roughly stable.
  if (!isFinite(ratio) || ratio > 10) return "explode";
  if (ratio < 0.1) return "vanish";
  return "stable";
}

const REGIME_META = {
  vanish: { label: "Vanishing", color: C.vanish, soft: C.vanishSoft },
  stable: { label: "Roughly stable", color: C.stable, soft: C.stableSoft },
  explode: { label: "Exploding", color: C.explode, soft: C.explodeSoft },
};

// Per-layer gradient norm on a log axis. Index runs from the input layer on the
// left to the output layer on the right, so a downhill line toward the left is
// the vanishing signal and an uphill line toward the left is exploding.
function GradientChart({ dz, revealUpTo, regime, reduced }) {
  const W = 640;
  const H = 240;
  const padL = 52;
  const padR = 16;
  const padT = 16;
  const padB = 34;
  const L = dz.length;
  const mags = dz.map((v) => Math.abs(v));
  const logs = mags.map((m) => Math.log10(Math.max(m, 1e-300)));
  let lo = Math.min(...logs);
  let hi = Math.max(...logs);
  if (!isFinite(lo)) lo = -30;
  if (!isFinite(hi)) hi = 0;
  if (hi - lo < 2) {
    const mid = (hi + lo) / 2;
    lo = mid - 1;
    hi = mid + 1;
  }
  const span = hi - lo;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;
  const xAt = (i) => padL + (L === 1 ? plotW / 2 : (i / (L - 1)) * plotW);
  const yAt = (lg) => padT + (1 - (lg - lo) / span) * plotH;

  const color = REGIME_META[regime].color;

  const tickDecades = [];
  const startDec = Math.ceil(lo);
  const endDec = Math.floor(hi);
  const stepDec = Math.max(1, Math.ceil((endDec - startDec) / 5));
  for (let d = startDec; d <= endDec; d += stepDec) tickDecades.push(d);

  const visible = Math.min(revealUpTo, L);
  const pts = [];
  for (let i = 0; i < L; i++) {
    if (i < L - visible) continue;
    pts.push([xAt(i), yAt(logs[i])]);
  }
  const linePath = pts.map(([x, y], i) => `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`).join("");

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      width="100%"
      role="img"
      aria-label={`Per-layer gradient magnitude on a log scale across ${L} layers. The current regime is ${REGIME_META[regime].label}.`}
      style={{ display: "block" }}
    >
      {tickDecades.map((d) => {
        const y = yAt(d);
        return (
          <g key={d}>
            <line x1={padL} y1={y} x2={W - padR} y2={y} stroke={C.grid} strokeWidth={1} />
            <text x={padL - 8} y={y + 3} textAnchor="end" fontSize={10} fontFamily="ui-monospace, monospace" fill={C.muted}>
              {`1e${d}`}
            </text>
          </g>
        );
      })}
      <line x1={padL} y1={padT} x2={padL} y2={H - padB} stroke={C.border} strokeWidth={1} />
      <text x={padL} y={H - padB + 22} textAnchor="middle" fontSize={10} fill={C.muted} letterSpacing="0.04em">
        input
      </text>
      <text x={W - padR} y={H - padB + 22} textAnchor="end" fontSize={10} fill={C.muted} letterSpacing="0.04em">
        output (loss)
      </text>
      <text
        x={padL - 40}
        y={padT + plotH / 2}
        textAnchor="middle"
        fontSize={10}
        fill={C.muted}
        transform={`rotate(-90 ${padL - 40} ${padT + plotH / 2})`}
      >
        |gradient| at z_l
      </text>
      {pts.length > 1 && (
        <path
          d={linePath}
          fill="none"
          stroke={color}
          strokeWidth={2.2}
          strokeLinejoin="round"
          strokeLinecap="round"
          style={{ transition: reduced ? "none" : `stroke 240ms ease` }}
        />
      )}
      {dz.map((v, i) => {
        if (i < L - visible) return null;
        const cx = xAt(i);
        const cy = yAt(logs[i]);
        const isEnd = i === 0 || i === L - 1;
        return (
          <circle
            key={i}
            cx={cx}
            cy={cy}
            r={isEnd ? 3.6 : L > 22 ? 1.8 : 2.6}
            fill={isEnd ? C.card : color}
            stroke={color}
            strokeWidth={isEnd ? 2 : 0}
          />
        );
      })}
    </svg>
  );
}

const DEEP = "deep";
const RNN = "rnn";

export default function App() {
  const reduced = useReducedMotion();
  const [view, setView] = useState(DEEP);
  const [depth, setDepth] = useState(16);
  const [steps, setSteps] = useState(20);
  const [actKey, setActKey] = useState("sigmoid");
  const [scale, setScale] = useState(0.85);
  const [input, setInput] = useState(0.8);
  const [seed, setSeed] = useState(7);
  const [reveal, setReveal] = useState(999);

  const L = view === DEEP ? depth : steps;

  // RNN shares one recurrent weight across all timesteps, so its "weights" array
  // is the same scale repeated; the deep chain jitters per layer by the seed.
  const weights = useMemo(() => {
    if (view === RNN) return new Array(steps).fill(scale);
    return buildWeights(depth, scale, seed);
  }, [view, depth, steps, scale, seed]);

  const chain = useMemo(() => runChain(input, weights, actKey), [input, weights, actKey]);

  useEffect(() => {
    setReveal(999);
  }, [view, depth, steps, actKey, scale, input, seed]);

  const mags = chain.dz.map((v) => Math.abs(v));
  const firstMag = mags[0];
  const lastMag = mags[L - 1];
  const ratio = lastMag === 0 ? Infinity : firstMag / lastMag;
  const regime = regimeOf(ratio);
  const rm = REGIME_META[regime];

  // The mean per-layer factor reported to the reader: the geometric mean of
  // |act'(z) * w| across layers, which is the base that gets raised to the depth.
  const geoFactor = useMemo(() => {
    const logs = chain.factor.map((f) => Math.log(Math.max(Math.abs(f), 1e-300)));
    const mean = logs.reduce((s, v) => s + v, 0) / logs.length;
    return Math.exp(mean);
  }, [chain]);

  const a = ACT[actKey];
  const reseed = () => setSeed((s) => (((s * 1103515245 + 12345) >>> 0) || 1));

  const stepBack = () => setReveal((r) => Math.min(L, (r > L ? 1 : r + 1)));
  const showAll = () => setReveal(999);

  const focusCss = `
    .vg-root :focus-visible { outline: 2px solid ${C.accent}; outline-offset: 2px; border-radius: 9px; }
    .vg-root input[type=range] { accent-color: ${C.accent}; }
    @media (prefers-reduced-motion: reduce) {
      .vg-root * { transition-duration: 0.001ms !important; animation-duration: 0.001ms !important; }
    }
  `;

  const sliderRow = (id, label, value, min, max, stepv, onChange, display, hint) => (
    <div style={{ marginBottom: 12 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
        <label htmlFor={id} style={{ fontSize: 13, color: C.muted, minWidth: 116 }}>
          {label}
        </label>
        <input
          id={id}
          type="range"
          min={min}
          max={max}
          step={stepv}
          value={value}
          onChange={(e) => onChange(+e.target.value)}
          style={{ flex: 1, minWidth: 120 }}
        />
        <span
          style={{
            fontFamily: "ui-monospace, monospace",
            fontSize: 13,
            fontWeight: 700,
            color: C.accent,
            minWidth: 52,
            textAlign: "right",
          }}
        >
          {display(value)}
        </span>
      </div>
      {hint && <div style={{ fontSize: 11, color: C.muted, margin: "3px 0 0 128px" }}>{hint}</div>}
    </div>
  );

  const layerWord = view === DEEP ? "layer" : "step";
  const depthWord = view === DEEP ? "depth" : "timesteps";

  return (
    <div
      className="vg-root"
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
          <div
            style={{
              fontSize: 11,
              letterSpacing: "0.14em",
              textTransform: "uppercase",
              color: C.muted,
              marginBottom: 6,
            }}
          >
            Deep learning, computed live
          </div>
          <h1
            style={{
              fontSize: 30,
              fontWeight: 700,
              margin: 0,
              lineHeight: 1.1,
              letterSpacing: "-0.02em",
              textWrap: "balance",
            }}
          >
            Vanishing and Exploding Gradients
          </h1>
          <p
            style={{
              color: C.muted,
              fontSize: 15,
              margin: "10px 0 0",
              lineHeight: 1.6,
              maxWidth: "62ch",
            }}
          >
            Backprop sends one gradient backward through a deep chain by multiplying, layer by layer,
            by a local factor: the activation derivative times the weight. Multiply many factors below
            one and the gradient collapses before it reaches the first layer. Multiply many above one
            and it blows up. Every number here is the real forward and backward pass of the chain.
          </p>
        </header>

        <Card style={{ marginBottom: 16, padding: "14px 18px" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 10 }}>
            <Segmented
              ariaLabel="Choose deep feedforward chain or recurrent through time"
              value={view}
              onChange={setView}
              options={[
                { value: DEEP, label: "Deep chain (depth)" },
                { value: RNN, label: "Recurrent (through time)" },
              ]}
            />
            <div
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: 8,
                padding: "5px 12px",
                borderRadius: 999,
                background: rm.soft,
                border: `1px solid ${rm.color}55`,
                transition: reduced ? "none" : "background 240ms ease, border-color 240ms ease",
              }}
            >
              <span style={{ width: 9, height: 9, borderRadius: 999, background: rm.color }} />
              <span style={{ fontSize: 12, fontWeight: 700, color: rm.color }}>{rm.label}</span>
            </div>
          </div>
        </Card>

        <Card style={{ marginBottom: 16 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", flexWrap: "wrap", gap: 8, marginBottom: 6 }}>
            <span style={{ fontSize: 13, fontWeight: 700 }}>
              {view === DEEP ? "Gradient magnitude per layer" : "Gradient magnitude per timestep"}
            </span>
            <span style={{ fontSize: 12, color: C.muted }}>
              {view === DEEP
                ? `${depth} layers of ${a.name}`
                : `${steps} steps of one shared ${a.name} cell`}
            </span>
          </div>
          <GradientChart dz={chain.dz} revealUpTo={reveal} regime={regime} reduced={reduced} />

          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))",
              gap: 10,
              marginTop: 12,
            }}
          >
            <Stat label={`|grad| at output ${layerWord}`} value={fmtSci(lastMag)} color={C.ink} />
            <Stat label={`|grad| at input ${layerWord}`} value={fmtSci(firstMag)} color={rm.color} />
            <Stat
              label="ratio input / output"
              value={fmtSci(ratio)}
              color={rm.color}
              hint={regime === "vanish" ? "collapsed" : regime === "explode" ? "blew up" : "near 1"}
            />
            <Stat label={`mean factor |act' x w|`} value={fmtSci(geoFactor)} color={C.ink} hint={`raised to ${L}`} />
          </div>

          <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginTop: 14 }}>
            <Btn onClick={stepBack} variant="dark" ariaLabel={`Reveal one more ${layerWord} of the backward pass`}>
              {reveal > L ? "Step backward from output" : `Step back (${Math.min(reveal, L)} / ${L})`}
            </Btn>
            <Btn onClick={showAll} variant="ghost" ariaLabel="Show the whole gradient profile">
              Show all {L} {layerWord}s
            </Btn>
            <Btn onClick={reseed} variant="ghost" disabled={view === RNN} ariaLabel="Draw a new random weight chain">
              New weights
            </Btn>
          </div>
          <p style={{ fontSize: 11, color: C.muted, margin: "10px 0 0", lineHeight: 1.5 }}>
            Backprop starts at the output (right) and walks left toward the input. Step it to watch the
            running gradient pick up one factor of {view === DEEP ? "act' x weight" : "act' x the shared weight"} per
            {" "}{layerWord}, the product the chain rule is assembling.
          </p>
        </Card>

        <Card style={{ marginBottom: 16 }}>
          <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 12 }}>Controls</div>

          <div style={{ marginBottom: 14 }}>
            <div style={{ fontSize: 13, color: C.muted, marginBottom: 6 }}>Activation</div>
            <Segmented
              ariaLabel="Activation function"
              value={actKey}
              onChange={setActKey}
              options={[
                { value: "sigmoid", label: "sigmoid" },
                { value: "tanh", label: "tanh" },
                { value: "relu", label: "ReLU" },
              ]}
            />
            <p style={{ fontSize: 11, color: C.muted, margin: "6px 0 0", lineHeight: 1.5 }}>
              {a.name}: {a.note}.
            </p>
          </div>

          {view === DEEP
            ? sliderRow("depth", `Depth (${depthWord})`, depth, 2, 30, 1, setDepth, (v) => String(v), "More layers means more factors multiplied together, so any drift away from one compounds.")
            : sliderRow("steps", "Timesteps T", steps, 2, 40, 1, setSteps, (v) => String(v), "The same recurrent weight is applied T times, so the factor is raised to the T-th power.")}

          {sliderRow(
            "scale",
            "Weight scale",
            scale,
            0.2,
            2.0,
            0.01,
            setScale,
            (v) => v.toFixed(2),
            view === DEEP
              ? "Roughly the per-layer weight magnitude. Small saturates the chain (vanish); large overdrives it (explode)."
              : "Magnitude of the single recurrent weight. Below one decays, above one grows, both exponentially in T.",
          )}

          {sliderRow("input", "Input value x", input, -3, 3, 0.05, setInput, (v) => v.toFixed(2), "The seed for the forward pass. Large magnitudes push sigmoid and tanh into their flat, saturated tails.")}

          <div style={{ fontSize: 11, color: C.muted, marginTop: 4, lineHeight: 1.5 }}>
            {view === DEEP
              ? "Each layer is one real scalar weight near the chosen scale (jittered by the seed) followed by the activation. Loss is one half the squared output."
              : "One shared weight equal to the scale, applied at every step, then the activation. Loss is one half the squared final hidden value."}
          </div>
        </Card>

        <Card style={{ marginBottom: 16, background: rm.soft, border: `1px solid ${rm.color}44`, transition: reduced ? "none" : "background 240ms ease, border-color 240ms ease" }}>
          <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 6, color: rm.color }}>
            {regime === "vanish"
              ? "The signal is vanishing"
              : regime === "explode"
                ? "The signal is exploding"
                : "The signal is holding"}
          </div>
          <p style={{ fontSize: 14, lineHeight: 1.6, margin: 0, color: C.ink, maxWidth: "64ch" }}>
            {regime === "vanish" &&
              `The gradient at the input ${layerWord} is ${fmtSci(ratio)} times the gradient at the output. Each ${layerWord} multiplies by a factor near ${fmtSci(geoFactor)}, below one, so across ${L} ${layerWord}s the signal decays to almost nothing. The early ${layerWord}s barely learn. Saturating activations make this worse because their derivative caps at ${a.dmax}, and a large input drives them into the flat tails where it falls toward zero.`}
            {regime === "explode" &&
              `The gradient at the input ${layerWord} is ${fmtSci(ratio)} times the gradient at the output. Each ${layerWord} multiplies by a factor near ${fmtSci(geoFactor)}, above one, so across ${L} ${layerWord}s the product grows geometrically. Updates overshoot and training diverges. This is what gradient clipping caps and what careful initialization is meant to avoid.`}
            {regime === "stable" &&
              `The per-${layerWord} factor sits near ${fmtSci(geoFactor)}, close enough to one that the product neither collapses nor blows up across ${L} ${layerWord}s. The gradient at the input is within a decade of the gradient at the output, so every ${layerWord} gets a usable signal. This is the narrow band that good initialization and normalization try to keep you in.`}
          </p>
        </Card>

        <Card>
          <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 8 }}>Why a product, and how it is tamed</div>
          <p style={{ fontSize: 14, lineHeight: 1.6, margin: "0 0 10px", color: C.ink, maxWidth: "64ch" }}>
            The chain rule writes the gradient at an early {layerWord} as a product of every Jacobian
            between it and the loss. For this scalar chain that product is the running multiplication of
            act'(z) times the weight, {layerWord} after {layerWord}. If the typical factor is below one,
            the product decays like factor to the power of {depthWord}; if it is above one, it grows the
            same way. Both are exponential in {depthWord}, which is why depth and long time horizons are
            exactly where the problem bites. Sigmoid and tanh make the vanishing side worse: their
            derivatives cap at {ACT.sigmoid.dmax} and {ACT.tanh.dmax} and fall to zero once a unit
            saturates, so most factors start well below one before the weights are even counted.
          </p>
          <p style={{ fontSize: 14, lineHeight: 1.6, margin: 0, color: C.ink, maxWidth: "64ch" }}>
            The fixes all attack the factor. ReLU keeps the derivative at exactly one on its active
            branch, so it does not shrink the signal the way a saturating unit does. Xavier and He
            initialization set the weight variance so the typical factor lands near one at the start of
            training, which is the dynamical-isometry idea: keep the chained Jacobian close to an
            isometry. Batch and layer normalization hold activations in the responsive part of the
            nonlinearity so derivatives stay healthy. Residual connections add an identity path, so the
            gradient has a route that multiplies by one and reaches early layers intact. For recurrent
            nets the same repeated factor across time is what LSTMs and GRUs address: their gating gives
            the cell state a near-identity path through time, letting gradients survive across many
            steps instead of decaying as factor to the power of T.
          </p>
        </Card>
      </div>
    </div>
  );
}

function Stat({ label, value, color, hint }) {
  return (
    <div
      style={{
        background: C.bg,
        border: `1px solid ${C.border}`,
        borderRadius: 10,
        padding: "9px 11px",
      }}
    >
      <div style={{ fontSize: 10.5, color: C.muted, letterSpacing: "0.02em", marginBottom: 3 }}>{label}</div>
      <div style={{ display: "flex", alignItems: "baseline", gap: 6 }}>
        <span style={{ fontFamily: "ui-monospace, monospace", fontSize: 15, fontWeight: 700, color }}>{value}</span>
        {hint && <span style={{ fontSize: 10.5, color: C.muted }}>{hint}</span>}
      </div>
    </div>
  );
}
