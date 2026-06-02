import { useState, useEffect, useMemo, useRef, useCallback, useId } from "react";

export const meta = {
  title: "Recurrent Networks, End to End",
  category: "Deep Learning",
  description:
    "A plain RNN forgets fast. Train one against an LSTM on a long-range task and watch the vanilla net's gradient die through time while the LSTM's cell-state path keeps memory alive.",
  date: "2026-05-30",
  tags: ["rnn", "lstm", "bptt", "gradients", "deep-learning", "training", "memory"],
};

const C = {
  bg: "#f7f5f2",
  card: "#ffffff",
  border: "#e6e1d9",
  ink: "#1b1916",
  muted: "#7c7368",
  accent: "#c0561f",
  accentSoft: "#f6ece5",
  rnn: "#2a5298",
  rnnSoft: "#e7eef8",
  lstm: "#2e7d51",
  lstmSoft: "#e6f1ea",
  cell: "#b8821a",
  cellSoft: "#f6efdc",
  forget: "#2a5298",
  input: "#2e7d51",
  output: "#9a4bb0",
  cand: "#c0561f",
  warm: "#b23b5e",
  cool: "#2a5298",
  grid: "#eee9e1",
  neg: "#b23b5e",
};

const EASE = "cubic-bezier(0.23, 1, 0.32, 1)";

const tanh = Math.tanh;
const dtanh = (y) => 1 - y * y;
const sigmoid = (x) => 1 / (1 + Math.exp(-x));
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

function zeros(n) {
  return new Array(n).fill(0);
}
function zeroMat(r, c) {
  return Array.from({ length: r }, () => zeros(c));
}
function randn(rand) {
  let u = 0;
  let v = 0;
  while (u === 0) u = rand();
  while (v === 0) v = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
function randMat(r, c, scale, rand) {
  return Array.from({ length: r }, () => Array.from({ length: c }, () => randn(rand) * scale));
}
function matVec(M, v) {
  const out = zeros(M.length);
  for (let i = 0; i < M.length; i++) {
    let s = 0;
    const row = M[i];
    for (let j = 0; j < v.length; j++) s += row[j] * v[j];
    out[i] = s;
  }
  return out;
}
function matTVec(M, v) {
  const cols = M[0].length;
  const out = zeros(cols);
  for (let i = 0; i < M.length; i++) {
    const row = M[i];
    const vi = v[i];
    for (let j = 0; j < cols; j++) out[j] += row[j] * vi;
  }
  return out;
}
function addOuter(acc, a, b) {
  for (let i = 0; i < a.length; i++) {
    const ai = a[i];
    const row = acc[i];
    for (let j = 0; j < b.length; j++) row[j] += ai * b[j];
  }
}
function norm2(v) {
  let s = 0;
  for (const x of v) s += x * x;
  return Math.sqrt(s);
}

function powerIterMag(M, iters, rand) {
  const n = M.length;
  let v = Array.from({ length: n }, () => rand() - 0.5);
  let nv = norm2(v) || 1;
  v = v.map((x) => x / nv);
  for (let it = 0; it < iters; it++) {
    const w = matVec(M, v);
    const wn = norm2(w);
    if (wn < 1e-300) return 0;
    v = w.map((x) => x / wn);
  }
  const Mv = matVec(M, v);
  let r = 0;
  for (let i = 0; i < n; i++) r += v[i] * Mv[i];
  return Math.abs(r);
}

function initRNN(H, X, seed) {
  const rand = mulberry32(seed);
  return {
    H,
    X,
    Wx: randMat(H, X, 0.5, rand),
    Wh: randMat(H, H, 1.0 / Math.sqrt(H), rand),
    b: zeros(H),
    Wy: randMat(1, H, 0.5, rand),
    by: zeros(1),
  };
}

function rnnRun(p, xs) {
  const T = xs.length;
  const hs = [zeros(p.H)];
  const zs = [];
  for (let t = 0; t < T; t++) {
    const z = matVec(p.Wx, xs[t]);
    const rec = matVec(p.Wh, hs[t]);
    for (let i = 0; i < p.H; i++) z[i] += rec[i] + p.b[i];
    zs.push(z);
    hs.push(z.map(tanh));
  }
  const yhat = matVec(p.Wy, hs[T])[0] + p.by[0];
  return { hs, zs, yhat };
}

function rnnLossGrad(p, xs, target) {
  const T = xs.length;
  const f = rnnRun(p, xs);
  const err = f.yhat - target;
  const loss = 0.5 * err * err;
  const g = {
    Wx: zeroMat(p.H, p.X),
    Wh: zeroMat(p.H, p.H),
    b: zeros(p.H),
    Wy: zeroMat(1, p.H),
    by: zeros(1),
  };
  const hT = f.hs[T];
  g.by[0] = err;
  for (let i = 0; i < p.H; i++) g.Wy[0][i] = err * hT[i];
  let dh = matTVec(p.Wy, [err]);
  const dhNorms = new Array(T + 1).fill(0);
  dhNorms[T] = norm2(dh);
  let dh0 = zeros(p.H);
  for (let t = T - 1; t >= 0; t--) {
    const h = f.hs[t + 1];
    const dz = zeros(p.H);
    for (let i = 0; i < p.H; i++) dz[i] = dh[i] * dtanh(h[i]);
    addOuter(g.Wx, dz, xs[t]);
    addOuter(g.Wh, dz, f.hs[t]);
    for (let i = 0; i < p.H; i++) g.b[i] += dz[i];
    dh = matTVec(p.Wh, dz);
    dhNorms[t] = norm2(dh);
    if (t === 0) dh0 = dh;
  }
  return { loss, yhat: f.yhat, grads: g, dh0: norm2(dh0), dhNorms };
}

const LSTM_FORGET_BIAS = 1.5;

function initLSTM(H, X, seed) {
  const rand = mulberry32(seed);
  const s = 1.0 / Math.sqrt(H + X);
  const gate = (wxScale) => ({ Wx: randMat(H, X, wxScale, rand), Wh: randMat(H, H, s, rand), b: zeros(H) });
  const p = {
    H,
    X,
    f: gate(0.5),
    i: gate(0.8),
    g: gate(0.8),
    o: gate(0.5),
    Wy: randMat(1, H, 0.5, rand),
    by: zeros(1),
  };
  // A positive forget-bias init keeps the cell-state path near identity at the
  // start of training so gradients can survive long horizons before the gate learns.
  for (let i = 0; i < H; i++) p.f.b[i] = LSTM_FORGET_BIAS;
  return p;
}

function gatePre(gate, x, h) {
  const z = matVec(gate.Wx, x);
  const rec = matVec(gate.Wh, h);
  for (let i = 0; i < z.length; i++) z[i] += rec[i] + gate.b[i];
  return z;
}

function lstmRun(p, xs) {
  const T = xs.length;
  const H = p.H;
  const cache = [];
  let h = zeros(H);
  let c = zeros(H);
  const hs = [h];
  const cs = [c];
  for (let t = 0; t < T; t++) {
    const x = xs[t];
    const fg = gatePre(p.f, x, h).map(sigmoid);
    const ig = gatePre(p.i, x, h).map(sigmoid);
    const gg = gatePre(p.g, x, h).map(tanh);
    const og = gatePre(p.o, x, h).map(sigmoid);
    const cNew = zeros(H);
    const ct = zeros(H);
    const hNew = zeros(H);
    for (let k = 0; k < H; k++) {
      cNew[k] = fg[k] * c[k] + ig[k] * gg[k];
      ct[k] = tanh(cNew[k]);
      hNew[k] = og[k] * ct[k];
    }
    cache.push({ x, hPrev: h, cPrev: c, f: fg, i: ig, g: gg, o: og, c: cNew, ct, h: hNew });
    h = hNew;
    c = cNew;
    hs.push(h);
    cs.push(c);
  }
  const yhat = matVec(p.Wy, hs[T])[0] + p.by[0];
  return { cache, hs, cs, yhat };
}

function lstmLossGrad(p, xs, target) {
  const T = xs.length;
  const H = p.H;
  const f = lstmRun(p, xs);
  const err = f.yhat - target;
  const loss = 0.5 * err * err;
  const mkg = () => ({ Wx: zeroMat(H, p.X), Wh: zeroMat(H, H), b: zeros(H) });
  const g = { f: mkg(), i: mkg(), g: mkg(), o: mkg(), Wy: zeroMat(1, H), by: zeros(1) };
  const hT = f.hs[T];
  g.by[0] = err;
  for (let k = 0; k < H; k++) g.Wy[0][k] = err * hT[k];
  let dh = matTVec(p.Wy, [err]);
  let dc = zeros(H);
  const dhNorms = new Array(T + 1).fill(0);
  dhNorms[T] = norm2(dh);
  let dh0 = zeros(H);
  for (let t = T - 1; t >= 0; t--) {
    const st = f.cache[t];
    const dzf = zeros(H);
    const dzi = zeros(H);
    const dzg = zeros(H);
    const dzo = zeros(H);
    const dcNext = zeros(H);
    for (let k = 0; k < H; k++) {
      const dout = dh[k] * st.ct[k];
      const dct = dh[k] * st.o[k];
      const dcTotal = dc[k] + dct * (1 - st.ct[k] * st.ct[k]);
      dcNext[k] = dcTotal * st.f[k];
      dzf[k] = dcTotal * st.cPrev[k] * st.f[k] * (1 - st.f[k]);
      dzi[k] = dcTotal * st.g[k] * st.i[k] * (1 - st.i[k]);
      dzg[k] = dcTotal * st.i[k] * (1 - st.g[k] * st.g[k]);
      dzo[k] = dout * st.o[k] * (1 - st.o[k]);
    }
    addOuter(g.f.Wx, dzf, st.x);
    addOuter(g.i.Wx, dzi, st.x);
    addOuter(g.g.Wx, dzg, st.x);
    addOuter(g.o.Wx, dzo, st.x);
    addOuter(g.f.Wh, dzf, st.hPrev);
    addOuter(g.i.Wh, dzi, st.hPrev);
    addOuter(g.g.Wh, dzg, st.hPrev);
    addOuter(g.o.Wh, dzo, st.hPrev);
    for (let k = 0; k < H; k++) {
      g.f.b[k] += dzf[k];
      g.i.b[k] += dzi[k];
      g.g.b[k] += dzg[k];
      g.o.b[k] += dzo[k];
    }
    const dhf = matTVec(p.f.Wh, dzf);
    const dhi = matTVec(p.i.Wh, dzi);
    const dhg = matTVec(p.g.Wh, dzg);
    const dho = matTVec(p.o.Wh, dzo);
    const dhPrev = zeros(H);
    for (let k = 0; k < H; k++) dhPrev[k] = dhf[k] + dhi[k] + dhg[k] + dho[k];
    dh = dhPrev;
    dc = dcNext;
    dhNorms[t] = norm2(dh);
    if (t === 0) dh0 = dh;
  }
  return { loss, yhat: f.yhat, grads: g, dh0: norm2(dh0), dhNorms };
}

function makeAdding(gap, seed, n) {
  const rand = mulberry32(seed);
  const T = gap + 1;
  const data = [];
  for (let s = 0; s < n; s++) {
    const xs = [];
    for (let t = 0; t < T; t++) xs.push([rand(), 0]);
    const p2 = 1 + Math.floor(rand() * (T - 1));
    xs[0][1] = 1;
    xs[p2][1] = 1;
    const target = (xs[0][0] + xs[p2][0]) / 2;
    data.push({ xs, target, markB: p2 });
  }
  return data;
}

function zeroLike(grads) {
  const out = Array.isArray(grads) ? [] : {};
  for (const k in grads) {
    const v = grads[k];
    out[k] = Array.isArray(v) ? zeroLike(v) : typeof v === "object" ? zeroLike(v) : 0;
  }
  if (Array.isArray(grads)) return grads.map((v) => (Array.isArray(v) ? zeroLike(v) : 0));
  return out;
}

function walkPair(a, b, fn) {
  for (const k in b) {
    const bv = b[k];
    if (Array.isArray(bv)) walkArr(a[k], bv, fn);
    else if (bv && typeof bv === "object") walkPair(a[k], bv, fn);
  }
}
function walkArr(a, b, fn) {
  for (let i = 0; i < b.length; i++) {
    if (Array.isArray(b[i])) walkArr(a[i], b[i], fn);
    else a[i] = fn(a[i], b[i]);
  }
}

function accumInto(acc, g) {
  walkPair(acc, g, (av, bv) => av + bv);
}
function scaleInPlace(g, f) {
  walkPair(g, g, (av) => av * f);
}
function gradGlobalNorm(g) {
  let s = 0;
  const walk = (o) => {
    for (const k in o) {
      const v = o[k];
      if (Array.isArray(v)) {
        const f = (a) => a.forEach((x) => (Array.isArray(x) ? f(x) : (s += x * x)));
        f(v);
      } else if (v && typeof v === "object") walk(v);
    }
  };
  walk(g);
  return Math.sqrt(s);
}

function makeAdam(params) {
  return { m: zeroLike(params), v: zeroLike(params), t: 0 };
}
function adamStep(params, grads, state, lr) {
  const b1 = 0.9;
  const b2 = 0.999;
  const eps = 1e-8;
  state.t += 1;
  const bc1 = 1 - Math.pow(b1, state.t);
  const bc2 = 1 - Math.pow(b2, state.t);
  const stepArr = (pa, ga, ma, va) => {
    for (let i = 0; i < pa.length; i++) {
      if (Array.isArray(pa[i])) stepArr(pa[i], ga[i], ma[i], va[i]);
      else {
        ma[i] = b1 * ma[i] + (1 - b1) * ga[i];
        va[i] = b2 * va[i] + (1 - b2) * ga[i] * ga[i];
        pa[i] -= (lr * (ma[i] / bc1)) / (Math.sqrt(va[i] / bc2) + eps);
      }
    }
  };
  const walk = (po, go, mo, vo) => {
    for (const k in go) {
      const gv = go[k];
      if (Array.isArray(gv)) stepArr(po[k], gv, mo[k], vo[k]);
      else if (gv && typeof gv === "object") walk(po[k], gv, mo[k], vo[k]);
    }
  };
  walk(params, grads, state.m, state.v);
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

const fmt = (v, d = 2) => {
  if (!isFinite(v)) return "inf";
  return v.toFixed(d);
};
const fmtSci = (v) => {
  if (!isFinite(v)) return "inf";
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a >= 1e4 || a < 1e-3) return v.toExponential(2);
  if (a >= 100) return v.toFixed(1);
  if (a >= 1) return v.toFixed(3);
  return v.toFixed(4);
};

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

function SliderRow({ id, label, value, min, max, step, onChange, display, hint, disabled }) {
  return (
    <div style={{ marginBottom: 12, opacity: disabled ? 0.5 : 1 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
        <label htmlFor={id} style={{ fontSize: 13, color: C.muted, minWidth: 118 }}>
          {label}
        </label>
        <input
          id={id}
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          disabled={disabled}
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
      {hint && <div style={{ fontSize: 11, color: C.muted, margin: "3px 0 0 130px" }}>{hint}</div>}
    </div>
  );
}

function heatFill(v, maxAbs) {
  const m = maxAbs < 1e-9 ? 1 : maxAbs;
  const t = clamp(Math.abs(v) / m, 0, 1);
  if (v >= 0) {
    const r = Math.round(247 - t * (247 - 192));
    const g = Math.round(245 - t * (245 - 86));
    const b = Math.round(242 - t * (242 - 31));
    return `rgb(${r},${g},${b})`;
  }
  const r = Math.round(247 - t * (247 - 42));
  const g = Math.round(245 - t * (245 - 82));
  const b = Math.round(242 - t * (242 - 152));
  return `rgb(${r},${g},${b})`;
}

function HiddenHeatmap({ hs, H, current, reduced }) {
  const T = hs.length - 1;
  const cell = T > 16 ? 17 : T > 10 ? 22 : 28;
  const rowH = H > 6 ? 18 : 22;
  const padL = 30;
  const padT = 18;
  const W = padL + T * cell + 4;
  const Ht = padT + H * rowH + 18;
  let maxAbs = 0;
  for (let t = 1; t <= T; t++) for (let k = 0; k < H; k++) maxAbs = Math.max(maxAbs, Math.abs(hs[t][k]));
  return (
    <div style={{ overflowX: "auto" }}>
      <svg
        viewBox={`0 0 ${W} ${Ht}`}
        width="100%"
        style={{ display: "block", minWidth: Math.min(W, 520) }}
        role="img"
        aria-label={`Hidden state heatmap, ${H} units down, ${T} timesteps across. Warmer means a more positive activation, cooler means more negative.`}
      >
        {Array.from({ length: H }).map((_, k) => (
          <text key={`r${k}`} x={padL - 6} y={padT + k * rowH + rowH / 2 + 3} textAnchor="end" fontSize={9} fill={C.muted} fontFamily="ui-monospace, monospace">
            h{k}
          </text>
        ))}
        {Array.from({ length: T }).map((_, ti) => {
          const t = ti + 1;
          const x = padL + ti * cell;
          const isCur = current === t;
          return (
            <g key={`c${t}`}>
              {Array.from({ length: H }).map((_, k) => (
                <rect
                  key={k}
                  x={x + 1}
                  y={padT + k * rowH + 1}
                  width={cell - 2}
                  height={rowH - 2}
                  rx={2}
                  fill={heatFill(hs[t][k], maxAbs)}
                  stroke={isCur ? C.accent : "#fff"}
                  strokeWidth={isCur ? 1.4 : 0.5}
                  style={{ transition: reduced ? "none" : `fill 220ms ease` }}
                />
              ))}
              <text x={x + cell / 2} y={padT + H * rowH + 12} textAnchor="middle" fontSize={8.5} fill={isCur ? C.accent : C.muted} fontWeight={isCur ? 700 : 400} fontFamily="ui-monospace, monospace">
                {t}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

function MatrixGrid({ M, label, color, maxAbs }) {
  const cols = M[0].length;
  let mx = maxAbs;
  if (mx === undefined) {
    mx = 0;
    for (const row of M) for (const v of row) mx = Math.max(mx, Math.abs(v));
  }
  const cell = cols > 6 ? 19 : 26;
  return (
    <div>
      <div style={{ fontSize: 11, color: C.muted, marginBottom: 5, fontFamily: "ui-monospace, monospace" }}>{label}</div>
      <div style={{ display: "inline-grid", gridTemplateColumns: `repeat(${cols}, ${cell}px)`, gap: 2 }}>
        {M.map((row, i) =>
          row.map((v, j) => (
            <div
              key={`${i}-${j}`}
              title={fmt(v, 3)}
              style={{
                width: cell,
                height: cell,
                borderRadius: 3,
                background: heatFill(v, mx),
                border: `0.5px solid ${C.border}`,
              }}
            />
          )),
        )}
      </div>
    </div>
  );
}

function LineChart({ series, height = 170, yLog, yLabel, xLabel, refLine, reduced, ariaLabel }) {
  const W = 640;
  const H = height;
  const padL = 50;
  const padR = 14;
  const padT = 14;
  const padB = 28;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;
  const allPts = series.flatMap((s) => s.data);
  const transform = yLog ? (v) => Math.log10(Math.max(Math.abs(v), 1e-12)) : (v) => v;
  let lo = Infinity;
  let hi = -Infinity;
  for (const p of allPts) {
    const v = transform(p[1]);
    if (!isFinite(v)) continue;
    lo = Math.min(lo, v);
    hi = Math.max(hi, v);
  }
  if (refLine !== undefined) {
    lo = Math.min(lo, transform(refLine));
    hi = Math.max(hi, transform(refLine));
  }
  if (!isFinite(lo)) lo = 0;
  if (!isFinite(hi)) hi = 1;
  if (hi - lo < (yLog ? 1 : 1e-6)) {
    const mid = (hi + lo) / 2;
    const pad = yLog ? 1 : 0.5;
    lo = mid - pad;
    hi = mid + pad;
  } else if (!yLog) {
    const pad = (hi - lo) * 0.08;
    lo -= pad;
    hi += pad;
  }
  const maxX = Math.max(1, ...series.map((s) => (s.data.length ? s.data[s.data.length - 1][0] : 0)));
  const xAt = (x) => padL + (maxX <= 0 ? 0 : (x / maxX) * plotW);
  const yAt = (v) => padT + (1 - (transform(v) - lo) / (hi - lo)) * plotH;

  const ticks = [];
  if (yLog) {
    for (let d = Math.ceil(lo); d <= Math.floor(hi); d++) ticks.push({ v: d, label: `1e${d}` });
  } else {
    for (let i = 0; i <= 4; i++) {
      const v = lo + ((hi - lo) * i) / 4;
      ticks.push({ v: transform === Math ? v : v, label: v.toFixed(2), raw: v });
    }
  }

  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label={ariaLabel} style={{ display: "block" }}>
      {ticks.map((tk, i) => {
        const y = yLog ? yAt(Math.pow(10, tk.v)) : padT + (1 - (tk.raw - lo) / (hi - lo)) * plotH;
        return (
          <g key={i}>
            <line x1={padL} y1={y} x2={W - padR} y2={y} stroke={C.grid} strokeWidth={1} />
            <text x={padL - 6} y={y + 3} textAnchor="end" fontSize={9} fill={C.muted} fontFamily="ui-monospace, monospace">
              {tk.label}
            </text>
          </g>
        );
      })}
      {refLine !== undefined && (
        <line x1={padL} y1={yAt(refLine)} x2={W - padR} y2={yAt(refLine)} stroke={C.muted} strokeWidth={1.2} strokeDasharray="4 3" opacity={0.7} />
      )}
      <line x1={padL} y1={padT} x2={padL} y2={H - padB} stroke={C.border} strokeWidth={1} />
      {yLabel && (
        <text x={14} y={padT + plotH / 2} textAnchor="middle" fontSize={9.5} fill={C.muted} transform={`rotate(-90 14 ${padT + plotH / 2})`}>
          {yLabel}
        </text>
      )}
      {xLabel && (
        <text x={padL + plotW / 2} y={H - 4} textAnchor="middle" fontSize={9.5} fill={C.muted}>
          {xLabel}
        </text>
      )}
      {series.map((s, si) => {
        if (s.data.length < 1) return null;
        const path = s.data
          .map((p, i) => `${i === 0 ? "M" : "L"}${xAt(p[0]).toFixed(1)},${yAt(p[1]).toFixed(1)}`)
          .join("");
        const last = s.data[s.data.length - 1];
        return (
          <g key={si}>
            {s.data.length > 1 && (
              <path
                d={path}
                fill="none"
                stroke={s.color}
                strokeWidth={2.2}
                strokeLinejoin="round"
                strokeLinecap="round"
                style={{ transition: reduced ? "none" : "stroke 160ms ease" }}
              />
            )}
            <circle cx={xAt(last[0])} cy={yAt(last[1])} r={3} fill={s.color} />
          </g>
        );
      })}
    </svg>
  );
}

function Legend({ items }) {
  return (
    <div style={{ display: "flex", gap: 16, marginTop: 8, fontSize: 12, flexWrap: "wrap" }}>
      {items.map((it, i) => (
        <span key={i} style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
          <span style={{ width: 16, height: it.dash ? 2 : 3, background: it.color, borderRadius: 2, ...(it.dash ? { backgroundImage: `repeating-linear-gradient(90deg, ${it.color} 0 4px, transparent 4px 7px)`, background: "none" } : {}) }} />
          {it.label}
        </span>
      ))}
    </div>
  );
}

function GateBar({ gkey, value, color, soft, name, signed, reduced }) {
  const trackH = 13;
  let fill;
  if (signed) {
    const w = Math.abs(value) * 50;
    const left = value >= 0 ? 50 : 50 - w;
    fill = (
      <div
        style={{
          position: "absolute",
          left: `${left}%`,
          width: `${w}%`,
          top: 0,
          bottom: 0,
          background: value < 0 ? C.neg : color,
          borderRadius: 3,
          transition: reduced ? "none" : `left 300ms ${EASE}, width 300ms ${EASE}`,
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
          background: color,
          borderRadius: 3,
          transition: reduced ? "none" : `width 300ms ${EASE}`,
        }}
      />
    );
  }
  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 3 }}>
        <span style={{ fontSize: 11.5 }}>
          <span style={{ fontWeight: 700, color }}>{gkey}</span>
          <span style={{ color: C.muted, marginLeft: 5 }}>{name}</span>
        </span>
        <span style={{ fontFamily: "ui-monospace, monospace", fontSize: 12, fontWeight: 700, color: value < 0 ? C.neg : color }}>
          {fmt(value)}
        </span>
      </div>
      <div style={{ position: "relative", height: trackH, background: soft, borderRadius: 4, overflow: "hidden" }} role="img" aria-label={`${name} ${fmt(value)}`}>
        {signed && <div style={{ position: "absolute", left: "50%", top: 0, bottom: 0, width: 1, background: `${color}55` }} />}
        {fill}
      </div>
    </div>
  );
}

const SEQ_PRESETS = {
  pulse: { label: "A pulse then silence", xs: [[0.9], [0], [0], [0], [0], [0], [0], [0]] },
  steps: { label: "A rising staircase", xs: [[0.2], [0.4], [0.6], [0.8], [0.6], [0.4], [0.2], [0]] },
  alt: { label: "Alternating signs", xs: [[0.8], [-0.8], [0.8], [-0.8], [0.8], [-0.8], [0.8], [-0.8]] },
};

function HiddenStateTab({ reduced }) {
  const [seqKey, setSeqKey] = useState("pulse");
  const [xs, setXs] = useState(SEQ_PRESETS.pulse.xs);
  const [seed, setSeed] = useState(7);
  const [whScale, setWhScale] = useState(1.0);
  const [current, setCurrent] = useState(0);
  const H = 6;

  const baseP = useMemo(() => initRNN(H, 1, seed), [seed]);
  const p = useMemo(() => {
    const q = { ...baseP, Wh: baseP.Wh.map((row) => row.map((v) => v * whScale)) };
    return q;
  }, [baseP, whScale]);

  const run = useMemo(() => rnnRun(p, xs), [p, xs]);
  const spectral = useMemo(() => powerIterMag(p.Wh, 120, mulberry32(13)), [p.Wh]);

  useEffect(() => {
    if (current > xs.length) setCurrent(xs.length);
  }, [xs, current]);

  const finalNorm = norm2(run.hs[run.hs.length - 1]);
  const regime = spectral < 0.85 ? "decay" : spectral > 1.15 ? "grow" : "hold";
  const regimeColor = regime === "decay" ? C.rnn : regime === "grow" ? C.warm : C.lstm;

  const applyPreset = (k) => {
    setSeqKey(k);
    setXs(SEQ_PRESETS[k].xs.map((r) => [...r]));
    setCurrent(0);
  };
  const setX = (idx, val) => setXs((arr) => arr.map((r, i) => (i === idx ? [val] : r)));
  const reseed = () => setSeed((s) => ((s * 1103515245 + 12345) >>> 0) || 1);

  return (
    <>
      <Card style={{ marginBottom: 16 }}>
        <p style={{ fontSize: 14, lineHeight: 1.6, margin: 0, color: C.ink, maxWidth: "64ch" }}>
          A hidden state is not an abstraction, it is a vector of numbers the network keeps and rewrites at every
          step. This is a real {H}-unit vanilla RNN, h_t = tanh(Wx x_t + Wh h_t-1 + b), with genuine weight
          matrices. Feed it a short sequence and read the hidden vector down each column of the heatmap: warm is
          positive, cool is negative. The recurrent matrix Wh decides whether that memory fades, holds, or
          saturates, and its largest eigenvalue magnitude is the dial.
        </p>
      </Card>

      <Card style={{ marginBottom: 16 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", flexWrap: "wrap", gap: 8, marginBottom: 8 }}>
          <span style={{ fontSize: 13, fontWeight: 700 }}>Hidden state across time</span>
          <span style={{ fontSize: 12, color: C.muted }}>{H} units, {xs.length} steps</span>
        </div>
        <HiddenHeatmap hs={run.hs} H={H} current={current} reduced={reduced} />
        <div style={{ display: "flex", gap: 14, marginTop: 4, fontSize: 11, color: C.muted, flexWrap: "wrap", alignItems: "center" }}>
          <span style={{ display: "inline-flex", alignItems: "center", gap: 5 }}>
            <span style={{ width: 12, height: 12, borderRadius: 2, background: heatFill(-1, 1) }} /> negative
          </span>
          <span style={{ display: "inline-flex", alignItems: "center", gap: 5 }}>
            <span style={{ width: 12, height: 12, borderRadius: 2, background: heatFill(0, 1) }} /> zero
          </span>
          <span style={{ display: "inline-flex", alignItems: "center", gap: 5 }}>
            <span style={{ width: 12, height: 12, borderRadius: 2, background: heatFill(1, 1) }} /> positive
          </span>
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: 10, marginTop: 14 }}>
          <Stat label="largest |eigenvalue| of Wh" value={fmt(spectral, 3)} color={regimeColor} hint={regime === "decay" ? "< 1" : regime === "grow" ? "> 1" : "near 1"} />
          <Stat label="|h| at final step" value={fmtSci(finalNorm)} color={C.ink} />
          <Stat label="|h| at first step" value={fmtSci(norm2(run.hs[1]))} color={C.muted} />
          <Stat label="output head y" value={fmt(run.yhat, 3)} color={C.accent} />
        </div>
      </Card>

      <Card style={{ marginBottom: 16, background: `${regimeColor}10`, border: `1px solid ${regimeColor}44`, transition: reduced ? "none" : "background 240ms ease, border-color 240ms ease" }}>
        <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 6, color: regimeColor }}>
          {regime === "decay" ? "Memory decays" : regime === "grow" ? "Memory saturates" : "Memory roughly holds"}
        </div>
        <p style={{ fontSize: 13.5, lineHeight: 1.6, margin: 0, color: C.ink, maxWidth: "64ch" }}>
          {regime === "decay" &&
            `The largest eigenvalue magnitude of Wh is ${fmt(spectral, 3)}, below one. Each step shrinks the carried-over part of the hidden state toward zero, so an early pulse fades within a few columns and the network forgets what it saw. The same factor below one is what makes gradients vanish on the way back.`}
          {regime === "grow" &&
            `The largest eigenvalue magnitude of Wh is ${fmt(spectral, 3)}, above one. The recurrent map amplifies the hidden state every step until tanh clamps the units near plus or minus one. The columns saturate and detail is lost: the running memory is pinned at the edges, and gradients on the way back tend to explode.`}
          {regime === "hold" &&
            `The largest eigenvalue magnitude of Wh is ${fmt(spectral, 3)}, near one. The recurrent map neither shrinks nor blows up the carried state, so an early signal can persist across several steps without fading or saturating. This narrow band is the only place a plain RNN holds memory well, and a single shared matrix rarely stays in it across a whole task.`}
        </p>
      </Card>

      <Card style={{ marginBottom: 16 }}>
        <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 12 }}>Controls</div>
        <SliderRow
          id="hs-wh"
          label="Scale Wh by"
          value={whScale}
          min={0.2}
          max={1.8}
          step={0.01}
          onChange={(v) => { setWhScale(v); }}
          display={(v) => v.toFixed(2)}
          hint="Scaling the recurrent matrix scales its eigenvalues. Drag below one to watch the hidden state die out, above one to watch it saturate."
        />
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginTop: 4, marginBottom: 14 }}>
          <Btn onClick={() => setWhScale(0.4)} ariaLabel="Set Wh scale to a decaying regime">Decay (0.4)</Btn>
          <Btn onClick={() => setWhScale(1.0)} ariaLabel="Set Wh scale to one">Hold (1.0)</Btn>
          <Btn onClick={() => setWhScale(1.6)} ariaLabel="Set Wh scale to a growing regime">Saturate (1.6)</Btn>
          <Btn onClick={reseed} variant="ghost" ariaLabel="Draw new random weight matrices">New weights</Btn>
        </div>

        <div style={{ fontSize: 13, color: C.muted, marginBottom: 6 }}>Input sequence</div>
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginBottom: 10 }}>
          {Object.entries(SEQ_PRESETS).map(([k, v]) => {
            const active = k === seqKey;
            return (
              <button
                key={k}
                onClick={() => applyPreset(k)}
                aria-pressed={active}
                style={{
                  fontFamily: "inherit",
                  fontSize: 12,
                  fontWeight: active ? 700 : 500,
                  padding: "6px 11px",
                  borderRadius: 8,
                  cursor: "pointer",
                  border: `1px solid ${active ? C.accent : C.border}`,
                  background: active ? C.accentSoft : "transparent",
                  color: active ? C.accent : C.muted,
                  transition: `all 150ms ${EASE}`,
                }}
              >
                {v.label}
              </button>
            );
          })}
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(72px, 1fr))", gap: 8 }}>
          {xs.map((v, i) => (
            <div key={i} style={{ display: "flex", flexDirection: "column", gap: 3 }}>
              <label htmlFor={`hsx-${i}`} style={{ fontSize: 10, color: C.muted, fontFamily: "ui-monospace, monospace" }}>
                x[{i + 1}]
              </label>
              <input
                id={`hsx-${i}`}
                type="number"
                step={0.1}
                value={v[0]}
                onChange={(e) => setX(i, +e.target.value)}
                style={{
                  fontFamily: "ui-monospace, monospace",
                  fontSize: 12,
                  padding: "5px 7px",
                  borderRadius: 7,
                  border: `1px solid ${C.border}`,
                  background: C.bg,
                  color: C.ink,
                  width: "100%",
                  boxSizing: "border-box",
                }}
              />
            </div>
          ))}
        </div>
        <div style={{ display: "flex", gap: 8, marginTop: 12 }}>
          <Btn onClick={() => xs.length < 14 && setXs((a) => [...a, [0]])} disabled={xs.length >= 14} ariaLabel="Add a step">Add step</Btn>
          <Btn onClick={() => xs.length > 3 && setXs((a) => a.slice(0, -1))} disabled={xs.length <= 3} ariaLabel="Remove the last step">Remove step</Btn>
        </div>
      </Card>

      <Card>
        <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 10 }}>The weight matrices behind the memory</div>
        <div style={{ display: "flex", gap: 22, flexWrap: "wrap" }}>
          <MatrixGrid M={p.Wh} label={`Wh (recurrent, ${H} x ${H})`} color={C.accent} />
          <MatrixGrid M={p.Wx} label={`Wx (input, ${H} x 1)`} color={C.accent} />
        </div>
        <p style={{ fontSize: 12.5, color: C.ink, lineHeight: 1.6, margin: "12px 0 0", maxWidth: "64ch" }}>
          Wh maps the old hidden vector into the new one, then the input contribution Wx x_t is added and tanh
          squashes the result. Because the same Wh is applied at every step, repeated multiplication by it is what
          drives the carried memory. The largest eigenvalue magnitude (the spectral radius) is the asymptotic
          growth factor of that repeated map: below one it contracts, above one it expands. This is the honest
          matrix version of the single shared weight in the scalar vanishing-gradients story.
        </p>
      </Card>
    </>
  );
}

const TRAIN_H = 8;
const TRAIN_X = 2;
const TRAIN_BATCH = 12;
const STEPS_PER_FRAME = 3;
const MAX_STEPS = 500;
const GRAD_CLIP = 1.0;
const LR = 0.01;

function makeTrainer(kind, gap, seed) {
  const params = kind === "rnn" ? initRNN(TRAIN_H, TRAIN_X, seed) : initLSTM(TRAIN_H, TRAIN_X, seed);
  const lossGrad = kind === "rnn" ? rnnLossGrad : lstmLossGrad;
  const train = makeAdding(gap, seed + 1, 200);
  const test = makeAdding(gap, seed + 8888, 80);
  const adam = makeAdam(params);
  let cursor = 0;
  function stepOnce() {
    let acc = null;
    let loss = 0;
    let dh0 = 0;
    for (let b = 0; b < TRAIN_BATCH; b++) {
      const d = train[cursor % train.length];
      cursor++;
      const r = lossGrad(params, d.xs, d.target);
      loss += r.loss;
      dh0 += r.dh0;
      if (!acc) acc = r.grads;
      else accumInto(acc, r.grads);
    }
    scaleInPlace(acc, 1 / TRAIN_BATCH);
    const gn = gradGlobalNorm(acc);
    if (gn > GRAD_CLIP) scaleInPlace(acc, GRAD_CLIP / gn);
    adamStep(params, acc, adam, LR);
    return { loss: loss / TRAIN_BATCH, dh0: dh0 / TRAIN_BATCH };
  }
  function evaluate() {
    let mse = 0;
    let within = 0;
    for (const d of test) {
      const r = kind === "rnn" ? rnnRun(params, d.xs) : lstmRun(params, d.xs);
      mse += (r.yhat - d.target) ** 2;
      if (Math.abs(r.yhat - d.target) < 0.04) within++;
    }
    return { mse: mse / test.length, acc: within / test.length };
  }
  return { params, stepOnce, evaluate };
}

function TrainingTab({ reduced }) {
  const [gap, setGap] = useState(25);
  const [seed, setSeed] = useState(3);
  const [running, setRunning] = useState(false);
  const [stepCount, setStepCount] = useState(0);
  const [lossHist, setLossHist] = useState({ rnn: [], lstm: [] });
  const [dh0Hist, setDh0Hist] = useState({ rnn: [], lstm: [] });
  const [evalState, setEvalState] = useState({ rnn: { mse: NaN, acc: 0 }, lstm: { mse: NaN, acc: 0 } });
  const trainers = useRef(null);
  const raf = useRef(null);

  const buildTrainers = useCallback(() => {
    trainers.current = {
      rnn: makeTrainer("rnn", gap, seed),
      lstm: makeTrainer("lstm", gap, seed),
    };
  }, [gap, seed]);

  const reset = useCallback(() => {
    setRunning(false);
    buildTrainers();
    setStepCount(0);
    setLossHist({ rnn: [], lstm: [] });
    setDh0Hist({ rnn: [], lstm: [] });
    setEvalState({ rnn: trainers.current.rnn.evaluate(), lstm: trainers.current.lstm.evaluate() });
  }, [buildTrainers]);

  useEffect(() => {
    reset();
  }, [reset]);

  useEffect(() => {
    if (!running) return;
    let stop = false;
    const tick = () => {
      if (stop || !trainers.current) return;
      let lastLossR = 0;
      let lastLossL = 0;
      let lastDh0R = 0;
      let lastDh0L = 0;
      let count = 0;
      setStepCount((prev) => {
        count = prev;
        for (let k = 0; k < STEPS_PER_FRAME && count < MAX_STEPS; k++) {
          const r = trainers.current.rnn.stepOnce();
          const l = trainers.current.lstm.stepOnce();
          lastLossR = r.loss;
          lastLossL = l.loss;
          lastDh0R = r.dh0;
          lastDh0L = l.dh0;
          count++;
        }
        return count;
      });
      setLossHist((h) => ({
        rnn: [...h.rnn, [count, lastLossR]].slice(-260),
        lstm: [...h.lstm, [count, lastLossL]].slice(-260),
      }));
      setDh0Hist((h) => ({
        rnn: [...h.rnn, [count, lastDh0R]].slice(-260),
        lstm: [...h.lstm, [count, lastDh0L]].slice(-260),
      }));
      setEvalState({ rnn: trainers.current.rnn.evaluate(), lstm: trainers.current.lstm.evaluate() });
      if (count >= MAX_STEPS) {
        setRunning(false);
        return;
      }
      raf.current = requestAnimationFrame(tick);
    };
    raf.current = requestAnimationFrame(tick);
    return () => {
      stop = true;
      cancelAnimationFrame(raf.current);
    };
  }, [running]);

  const done = stepCount >= MAX_STEPS;
  const meanBaseline = 1 / 12; // variance of (u1+u2)/2 with u uniform on [0,1]; predicting the mean lands here

  return (
    <>
      <Card style={{ marginBottom: 16 }}>
        <p style={{ fontSize: 14, lineHeight: 1.6, margin: 0, color: C.ink, maxWidth: "64ch" }}>
          Everyone is told a plain RNN forgets long-range dependencies and an LSTM fixes it. Here you train both,
          for real, and watch it happen. The task is the adding problem: each sequence is a stream of random
          numbers in [0, 1], two of them are flagged (the first step and one later step), and the network must
          output their average. The answer depends on a value seen at step one, so the gradient for that step has
          to travel all the way back through the unrolled cells. Both nets are real {TRAIN_H}-unit cells trained
          with backprop through time and Adam, in your browser, a few steps per animation frame.
        </p>
      </Card>

      <Card style={{ marginBottom: 16 }}>
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center", marginBottom: 14 }}>
          <Btn onClick={() => setRunning((r) => !r)} variant="primary" disabled={done} ariaLabel={running ? "Pause training" : "Train both networks"}>
            {running ? "Pause" : stepCount > 0 ? "Resume" : "Train both"}
          </Btn>
          <Btn onClick={reset} variant="ghost" ariaLabel="Reset both networks and start over">Reset</Btn>
          <span style={{ fontSize: 12, color: C.muted, fontFamily: "ui-monospace, monospace" }}>
            step {stepCount} / {MAX_STEPS}{done ? " (budget spent)" : ""}
          </span>
        </div>
        <SliderRow
          id="tr-gap"
          label="Gap length T"
          value={gap}
          min={8}
          max={55}
          step={1}
          onChange={setGap}
          display={(v) => String(v)}
          hint="Longer gap means the flagged value sits further back in time. Watch the RNN fail harder as you grow it. Changing the gap resets training."
          disabled={running}
        />
        <div style={{ fontSize: 11, color: C.muted, lineHeight: 1.5 }}>
          Each sequence is {gap + 1} steps long. Reset and retrain after moving the slider to compare on the new horizon.
        </div>
      </Card>

      <Card style={{ marginBottom: 16 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", flexWrap: "wrap", gap: 8, marginBottom: 6 }}>
          <span style={{ fontSize: 13, fontWeight: 700 }}>Loss over training</span>
          <span style={{ fontSize: 11, color: C.muted }}>mean squared error, dashed line is predict-the-mean</span>
        </div>
        <LineChart
          series={[
            { data: lossHist.rnn, color: C.rnn },
            { data: lossHist.lstm, color: C.lstm },
          ]}
          refLine={meanBaseline}
          yLabel="loss"
          xLabel="training step"
          reduced={reduced}
          ariaLabel={`Training loss for both networks. RNN loss ${fmtSci(evalState.rnn.mse)}, LSTM loss ${fmtSci(evalState.lstm.mse)}.`}
        />
        <Legend
          items={[
            { color: C.rnn, label: "vanilla RNN" },
            { color: C.lstm, label: "LSTM" },
            { color: C.muted, label: "predict-the-mean baseline", dash: true },
          ]}
        />
      </Card>

      <Card style={{ marginBottom: 16 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", flexWrap: "wrap", gap: 8, marginBottom: 6 }}>
          <span style={{ fontSize: 13, fontWeight: 700 }}>Gradient norm at the first timestep</span>
          <span style={{ fontSize: 11, color: C.muted }}>log scale</span>
        </div>
        <LineChart
          series={[
            { data: dh0Hist.rnn, color: C.rnn },
            { data: dh0Hist.lstm, color: C.lstm },
          ]}
          yLog
          yLabel="|dL / dh_1|"
          xLabel="training step"
          reduced={reduced}
          ariaLabel="Gradient norm of the loss with respect to the first hidden state, on a log scale, for both networks across training."
        />
        <Legend items={[{ color: C.rnn, label: "vanilla RNN" }, { color: C.lstm, label: "LSTM" }]} />
        <p style={{ fontSize: 12, color: C.muted, margin: "10px 0 0", lineHeight: 1.5 }}>
          This is the gradient that has to reach step one for the network to learn the long-range link. For the
          vanilla RNN it lives orders of magnitude lower and sinks further as the gap grows: that is the vanishing
          gradient, measured on real multi-unit cells, as training runs.
        </p>
      </Card>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginBottom: 16 }}>
        <Card style={{ background: C.rnnSoft, border: `1px solid ${C.rnn}44` }}>
          <div style={{ fontSize: 12, fontWeight: 700, color: C.rnn, marginBottom: 8 }}>Vanilla RNN</div>
          <div style={{ display: "grid", gap: 8 }}>
            <Stat label="test loss" value={fmtSci(evalState.rnn.mse)} color={C.rnn} />
            <Stat label="within tolerance" value={`${Math.round(evalState.rnn.acc * 100)}%`} color={C.rnn} hint="|error| < 0.04" />
          </div>
        </Card>
        <Card style={{ background: C.lstmSoft, border: `1px solid ${C.lstm}44` }}>
          <div style={{ fontSize: 12, fontWeight: 700, color: C.lstm, marginBottom: 8 }}>LSTM</div>
          <div style={{ display: "grid", gap: 8 }}>
            <Stat label="test loss" value={fmtSci(evalState.lstm.mse)} color={C.lstm} />
            <Stat label="within tolerance" value={`${Math.round(evalState.lstm.acc * 100)}%`} color={C.lstm} hint="|error| < 0.04" />
          </div>
        </Card>
      </div>

      <Card>
        <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 8 }}>What you are seeing</div>
        <p style={{ fontSize: 14, lineHeight: 1.6, margin: "0 0 10px", color: C.ink, maxWidth: "64ch" }}>
          The LSTM drives its loss well below the predict-the-mean baseline and reaches the flagged value through
          the whole gap. The vanilla RNN tends to stall near that baseline: it cannot route enough gradient back to
          the first step, so it gives up on the long-range part and just predicts the average. Push the gap slider
          up and retrain: the RNN's first-step gradient sinks further and its loss flattens earlier, while the
          LSTM holds on longer.
        </p>
        <p style={{ fontSize: 12.5, lineHeight: 1.6, margin: 0, color: C.muted, maxWidth: "64ch" }}>
          Honest caveats: this is a tiny model on a synthetic task with a fixed training budget, and a random seed
          can occasionally let the RNN partly latch a short gap or stall the LSTM. The robust, repeatable signal is
          the first-step gradient norm, which separates the two architectures by orders of magnitude and widens
          with the gap regardless of the run.
        </p>
      </Card>
    </>
  );
}

function BpttTab({ reduced }) {
  const H = 4;
  const seed = 21;
  const xs = useMemo(() => [[0.8, 1], [0.3, 0], [0.6, 0], [0.2, 0], [0.5, 0], [0.7, 1]], []);
  const target = (xs[0][0] + xs[5][0]) / 2;
  const p = useMemo(() => initRNN(H, 2, seed), []);

  const trace = useMemo(() => {
    const f = rnnRun(p, xs);
    const T = xs.length;
    const err = f.yhat - target;
    let dh = matTVec(p.Wy, [err]);
    const gWhAcc = zeroMat(H, H);
    const gWhStep = [];
    for (let t = T - 1; t >= 0; t--) {
      const h = f.hs[t + 1];
      const dz = zeros(H);
      for (let i = 0; i < H; i++) dz[i] = dh[i] * dtanh(h[i]);
      const contrib = zeroMat(H, H);
      addOuter(contrib, dz, f.hs[t]);
      addOuter(gWhAcc, dz, f.hs[t]);
      gWhStep.push({ t, contrib: contrib.map((r) => [...r]), running: gWhAcc.map((r) => [...r]) });
      dh = matTVec(p.Wh, dz);
    }
    gWhStep.reverse();
    return { f, err, gWhStep, T };
  }, [p, xs, target]);

  const [revealed, setRevealed] = useState(0);

  const visibleSteps = trace.gWhStep.slice(trace.T - revealed).reverse();
  const running = revealed === 0 ? zeroMat(H, H) : visibleSteps[0].running;
  const contribNorms = trace.gWhStep.map((s) => norm2(s.contrib.flat()));
  const maxContrib = Math.max(...contribNorms, 1e-9);

  return (
    <>
      <Card style={{ marginBottom: 16 }}>
        <p style={{ fontSize: 14, lineHeight: 1.6, margin: 0, color: C.ink, maxWidth: "64ch" }}>
          The forward pass is clear, but backprop through time feels like magic, and the phrase one shared weight
          is confusing. Here is the resolution. Because the same recurrent matrix Wh is reused at every step, its
          gradient is a sum: each timestep hands Wh one contribution, and the full gradient is all of them added
          together. Step the backward pass and watch the running sum fill in, one timestep at a time, from the
          output back toward step one. The contribution from distant steps is exactly what the per-step Jacobian
          shrinks or grows.
        </p>
      </Card>

      <Card style={{ marginBottom: 16 }}>
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center", marginBottom: 12 }}>
          <Btn onClick={() => setRevealed((r) => Math.min(trace.T, r + 1))} variant="dark" disabled={revealed >= trace.T} ariaLabel="Step the gradient back one more timestep">
            Step backward
          </Btn>
          <Btn onClick={() => setRevealed(0)} variant="ghost" disabled={revealed === 0} ariaLabel="Reset the backward pass">Reset</Btn>
          <Btn onClick={() => setRevealed(trace.T)} variant="ghost" disabled={revealed >= trace.T} ariaLabel="Reveal all timestep contributions">Show all</Btn>
          <span style={{ fontSize: 12, color: C.muted, fontFamily: "ui-monospace, monospace" }}>
            {revealed} / {trace.T} steps summed
          </span>
        </div>

        <div style={{ fontSize: 12, color: C.muted, marginBottom: 8 }}>
          Per-timestep contribution to the recurrent gradient dL/dWh
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(78px, 1fr))", gap: 8 }}>
          {trace.gWhStep.map((s, idx) => {
            const isRevealed = idx >= trace.T - revealed;
            const justAdded = revealed > 0 && idx === trace.T - revealed;
            const mag = contribNorms[idx];
            const t = (mag / maxContrib);
            return (
              <div
                key={idx}
                style={{
                  borderRadius: 9,
                  border: `1px solid ${justAdded ? C.accent : isRevealed ? `${C.accent}55` : C.border}`,
                  background: isRevealed ? `${C.accent}${Math.round(t * 40 + 8).toString(16).padStart(2, "0")}` : C.bg,
                  padding: "8px 9px",
                  opacity: isRevealed ? 1 : 0.4,
                  transition: reduced ? "none" : `all 240ms ${EASE}`,
                }}
              >
                <div style={{ fontSize: 10, color: C.muted, fontFamily: "ui-monospace, monospace" }}>step {s.t + 1}</div>
                <div style={{ fontFamily: "ui-monospace, monospace", fontSize: 13, fontWeight: 700, color: isRevealed ? C.accent : C.muted }}>
                  {fmtSci(mag)}
                </div>
                <div style={{ fontSize: 9, color: C.muted }}>{s.t === 0 ? "first step" : ""}</div>
              </div>
            );
          })}
        </div>
        <p style={{ fontSize: 11.5, color: C.muted, margin: "10px 0 0", lineHeight: 1.5 }}>
          Backprop walks from the last step (right) toward the first (left). Each card is the gradient Wh receives
          at that step, an outer product of the local error with the previous hidden state. The contributions from
          early steps are smaller because the gradient has been multiplied by Wh and tanh's derivative many times
          on the way there.
        </p>
      </Card>

      <div style={{ display: "grid", gridTemplateColumns: "1fr", gap: 12, marginBottom: 16 }}>
        <Card>
          <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 10 }}>The accumulating gradient for the one shared matrix</div>
          <div style={{ display: "flex", gap: 22, flexWrap: "wrap", alignItems: "flex-start" }}>
            <MatrixGrid M={running} label={`running sum of dL/dWh (${revealed} of ${trace.T} steps)`} color={C.accent} maxAbs={Math.max(...trace.gWhStep[0].running.flat().map(Math.abs), ...running.flat().map(Math.abs), 1e-6)} />
            {revealed > 0 && revealed < trace.T && (
              <MatrixGrid M={visibleSteps[0].contrib} label={`just added: step ${trace.gWhStep[trace.T - revealed].t + 1}`} color={C.accent} maxAbs={Math.max(...trace.gWhStep[0].running.flat().map(Math.abs), 1e-6)} />
            )}
          </div>
          <p style={{ fontSize: 12.5, color: C.ink, lineHeight: 1.6, margin: "12px 0 0", maxWidth: "64ch" }}>
            There is only one Wh in the network, so there is only one gradient matrix for it. Every timestep adds
            into the same matrix. That is what one shared weight means under the hood: the update is the sum of the
            pressures from all timesteps at once, which is why a single step of gradient descent moves the
            recurrent matrix in a direction that accounts for the whole sequence.
          </p>
        </Card>
      </div>

      <Card>
        <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 8 }}>Why distant steps fade</div>
        <p style={{ fontSize: 14, lineHeight: 1.6, margin: 0, color: C.ink, maxWidth: "64ch" }}>
          The contribution from step t is built from the gradient that arrived there, and that gradient reached
          step t by being multiplied, once per step, by Wh times the diagonal of tanh'. That repeated product is
          the per-step Jacobian. When its magnitude sits below one the early contributions shrink toward zero, so
          the shared matrix barely feels the distant steps and the long-range dependency goes unlearned. When it
          sits above one they blow up instead. Either way the sum is dominated by the recent steps. The next tab
          shows how the LSTM gives that gradient a second route that does not collapse.
        </p>
      </Card>
    </>
  );
}

function GatesTab({ reduced }) {
  const [seqKey, setSeqKey] = useState("pulse");
  const [seed, setSeed] = useState(11);
  const [vanillaScale, setVanillaScale] = useState(1.0);
  const H = 6;
  const xs = useMemo(() => SEQ_PRESETS[seqKey].xs.map((r) => [...r]), [seqKey]);

  const lstmP = useMemo(() => initLSTM(H, 1, seed), [seed]);
  const rnnP = useMemo(() => {
    const base = initRNN(H, 1, seed);
    return { ...base, Wh: base.Wh.map((row) => row.map((v) => v * vanillaScale)) };
  }, [seed, vanillaScale]);

  const lstmRunRes = useMemo(() => lstmRun(lstmP, xs), [lstmP, xs]);
  const rnnRunRes = useMemo(() => rnnRun(rnnP, xs), [rnnP, xs]);

  const [cur, setCur] = useState(0);
  useEffect(() => {
    if (cur > xs.length - 1) setCur(xs.length - 1);
  }, [xs, cur]);

  const T = xs.length;

  // LSTM cell-state path is dominated by the product of forget gates; report it as
  // the geometric-mean forget magnitude per unit so the comparison stays honest.
  const lstmProfile = useMemo(() => {
    const prof = new Array(T).fill(1);
    const acc = new Array(H).fill(1);
    for (let t = T - 1; t >= 0; t--) {
      let s = 0;
      for (let k = 0; k < H; k++) s += acc[k] * acc[k];
      prof[t] = Math.sqrt(s / H);
      const st = lstmRunRes.cache[t];
      for (let k = 0; k < H; k++) acc[k] *= st.f[k];
    }
    return prof;
  }, [lstmRunRes, T]);

  const rnnProfile = useMemo(() => {
    const prof = new Array(T).fill(1);
    let M = Array.from({ length: H }, (_, i) => Array.from({ length: H }, (_, j) => (i === j ? 1 : 0)));
    for (let t = T - 1; t >= 0; t--) {
      let fro = 0;
      for (const row of M) for (const v of row) fro += v * v;
      prof[t] = Math.sqrt(fro / H);
      const z = rnnRunRes.zs[t];
      const J = rnnP.Wh.map((row, i) => row.map((w) => w * dtanh(tanh(z[i]))));
      const next = zeroMat(H, H);
      for (let i = 0; i < H; i++) for (let j = 0; j < H; j++) { let s = 0; for (let k = 0; k < H; k++) s += M[i][k] * J[k][j]; next[i][j] = s; }
      M = next;
    }
    return prof;
  }, [rnnRunRes, rnnP, T]);

  const step = lstmRunRes.cache[cur];
  const reseed = () => setSeed((s) => ((s * 1103515245 + 12345) >>> 0) || 1);

  return (
    <>
      <Card style={{ marginBottom: 16 }}>
        <p style={{ fontSize: 14, lineHeight: 1.6, margin: 0, color: C.ink, maxWidth: "64ch" }}>
          It is easy to memorize the LSTM gate equations without feeling why they help. Here is the why, side by
          side on the same input. The vanilla RNN sends its gradient back through the product of per-step Jacobians
          Wh times diag(tanh'), which decays or explodes. The LSTM keeps a separate cell state updated by addition,
          c_t = f * c_t-1 + i * g, and the gradient along that cell-state highway is dominated by the product of
          the forget gates. When the forget gates sit near one, that product stays near one: a near-identity path
          the gradient can survive.
        </p>
      </Card>

      <Card style={{ marginBottom: 16 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", flexWrap: "wrap", gap: 8, marginBottom: 6 }}>
          <span style={{ fontSize: 13, fontWeight: 700 }}>Gradient through time, both paths</span>
          <span style={{ fontSize: 11, color: C.muted }}>log scale, line at one is the watershed</span>
        </div>
        <LineChart
          series={[
            { data: rnnProfile.map((v, i) => [i + 1, v]), color: C.rnn },
            { data: lstmProfile.map((v, i) => [i + 1, v]), color: C.lstm },
          ]}
          yLog
          refLine={1}
          yLabel="path magnitude"
          xLabel="step (1 = first, then back to last)"
          reduced={reduced}
          ariaLabel="Gradient path magnitude through time on a log scale. The RNN Jacobian product collapses while the LSTM forget-gate product stays near one."
        />
        <Legend
          items={[
            { color: C.rnn, label: "RNN: Jacobian product Wh diag(tanh')" },
            { color: C.lstm, label: "LSTM: forget-gate product" },
            { color: C.muted, label: "the line at one", dash: true },
          ]}
        />
        <p style={{ fontSize: 11.5, color: C.muted, margin: "10px 0 0", lineHeight: 1.5 }}>
          Read right to left, the way the gradient travels: step T on the right, the first step on the left. The
          RNN curve drifts away from one and keeps going, so the gradient that reaches the first step is tiny. The
          LSTM curve hugs one because the forget gates are near one, so the cell-state gradient arrives roughly
          intact.
        </p>
      </Card>

      <Card style={{ marginBottom: 16 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", flexWrap: "wrap", gap: 8, marginBottom: 8 }}>
          <span style={{ fontSize: 13, fontWeight: 700 }}>The LSTM gates at step {cur + 1}</span>
          <span style={{ fontSize: 12, color: C.muted }}>{H} units</span>
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
          <GateBar gkey="f" name="forget (mean)" value={mean(step.f)} color={C.forget} soft={C.rnnSoft} reduced={reduced} />
          <GateBar gkey="i" name="input (mean)" value={mean(step.i)} color={C.input} soft={C.lstmSoft} reduced={reduced} />
          <GateBar gkey="g" name="candidate (mean)" value={mean(step.g)} color={C.cand} soft={C.accentSoft} signed reduced={reduced} />
          <GateBar gkey="o" name="output (mean)" value={mean(step.o)} color={C.output} soft="#f1e7f5" reduced={reduced} />
        </div>
        <div style={{ marginTop: 14 }}>
          <div style={{ fontSize: 12, color: C.muted, marginBottom: 6 }}>Cell-state highway across the sequence (per unit)</div>
          <HiddenHeatmap hs={lstmRunRes.cs} H={H} current={cur + 1} reduced={reduced} />
        </div>
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginTop: 12, alignItems: "center" }}>
          <Btn onClick={() => setCur((c) => Math.max(0, c - 1))} disabled={cur === 0} ariaLabel="Previous step">Prev</Btn>
          <Btn onClick={() => setCur((c) => Math.min(T - 1, c + 1))} disabled={cur >= T - 1} ariaLabel="Next step">Next</Btn>
          <span style={{ fontSize: 12, color: C.muted, fontFamily: "ui-monospace, monospace" }}>step {cur + 1} / {T}</span>
        </div>
      </Card>

      <Card style={{ marginBottom: 16 }}>
        <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 10 }}>Controls</div>
        <div style={{ fontSize: 13, color: C.muted, marginBottom: 6 }}>Input sequence</div>
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginBottom: 14 }}>
          {Object.entries(SEQ_PRESETS).map(([k, v]) => {
            const active = k === seqKey;
            return (
              <button
                key={k}
                onClick={() => { setSeqKey(k); setCur(0); }}
                aria-pressed={active}
                style={{
                  fontFamily: "inherit",
                  fontSize: 12,
                  fontWeight: active ? 700 : 500,
                  padding: "6px 11px",
                  borderRadius: 8,
                  cursor: "pointer",
                  border: `1px solid ${active ? C.accent : C.border}`,
                  background: active ? C.accentSoft : "transparent",
                  color: active ? C.accent : C.muted,
                  transition: `all 150ms ${EASE}`,
                }}
              >
                {v.label}
              </button>
            );
          })}
        </div>
        <SliderRow
          id="g-vw"
          label="Scale RNN Wh"
          value={vanillaScale}
          min={0.3}
          max={1.7}
          step={0.01}
          onChange={setVanillaScale}
          display={(v) => v.toFixed(2)}
          hint="The vanilla RNN has one recurrent matrix doing both memory and mixing. Below one the Jacobian product vanishes, above one it explodes. There is no value that simply holds across the sequence."
        />
        <div style={{ marginTop: 4 }}>
          <Btn onClick={reseed} variant="ghost" ariaLabel="Draw new random weights for both cells">New weights</Btn>
        </div>
      </Card>

      <Card>
        <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 8 }}>The honest version of the claim</div>
        <p style={{ fontSize: 14, lineHeight: 1.6, margin: "0 0 10px", color: C.ink, maxWidth: "64ch" }}>
          For a single unit the cell-state gradient really is just the product of forget gates. For these
          multi-unit cells it is the dominant term, not the whole story: the exact Jacobian of c_t with respect to
          c_t-1 is a diagonal matrix of forget gates plus smaller off-diagonal corrections from the gates that
          depend on the hidden state. The forget-gate product is what carries the gradient over long horizons, and
          the corrections are second order while the gates stay near saturation. That is why the LSTM curve above
          tracks the forget-gate product so closely.
        </p>
        <p style={{ fontSize: 14, lineHeight: 1.6, margin: 0, color: C.ink, maxWidth: "64ch" }}>
          So the gates do not remove the multiplication, they change what gets multiplied. The vanilla RNN is stuck
          multiplying by Wh times tanh' every step, a factor it cannot hold near one across a whole task with a
          single shared matrix. The LSTM multiplies the cell-state gradient by forget gates the data can set near
          one wherever memory must persist. Same idea as a residual connection in a deep net: add a route that
          multiplies by one, then let the network decide when to open it.
        </p>
      </Card>
    </>
  );
}

function mean(arr) {
  let s = 0;
  for (const v of arr) s += v;
  return s / arr.length;
}

const TABS = [
  { id: "hidden", label: "What a hidden state actually is", short: "Hidden state", Comp: HiddenStateTab },
  { id: "train", label: "Why a plain RNN can't remember", short: "Can't remember", Comp: TrainingTab },
  { id: "bptt", label: "How training flows backward", short: "Backward (BPTT)", Comp: BpttTab },
  { id: "gates", label: "Why gates beat a plain RNN", short: "Why gates win", Comp: GatesTab },
];

export default function App() {
  const reduced = useReducedMotion();
  const [tab, setTab] = useState(0);
  const tabRefs = useRef([]);
  const baseId = useId();

  const onTabKey = (e) => {
    if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
      e.preventDefault();
      const dir = e.key === "ArrowRight" ? 1 : -1;
      const next = (tab + dir + TABS.length) % TABS.length;
      setTab(next);
      tabRefs.current[next]?.focus();
    } else if (e.key === "Home") {
      e.preventDefault();
      setTab(0);
      tabRefs.current[0]?.focus();
    } else if (e.key === "End") {
      e.preventDefault();
      setTab(TABS.length - 1);
      tabRefs.current[TABS.length - 1]?.focus();
    }
  };

  const ActiveComp = TABS[tab].Comp;

  const focusCss = `
    .rn-root :focus-visible { outline: 2px solid ${C.accent}; outline-offset: 2px; border-radius: 9px; }
    .rn-root input[type=range] { accent-color: ${C.accent}; }
    .rn-root input[type=number] { accent-color: ${C.accent}; }
    @media (prefers-reduced-motion: reduce) {
      .rn-root * { transition-duration: 0.001ms !important; animation-duration: 0.001ms !important; }
    }
  `;

  return (
    <div
      className="rn-root"
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
        <header style={{ marginBottom: 20 }}>
          <div style={{ fontSize: 11, letterSpacing: "0.14em", textTransform: "uppercase", color: C.muted, marginBottom: 6 }}>
            Deep learning, computed live
          </div>
          <h1 style={{ fontSize: 30, fontWeight: 700, margin: 0, lineHeight: 1.1, letterSpacing: "-0.02em", textWrap: "balance" }}>
            Recurrent Networks, End to End
          </h1>
          <p style={{ color: C.muted, fontSize: 15, margin: "10px 0 0", lineHeight: 1.6, maxWidth: "64ch" }}>
            A plain recurrent network forgets fast. Send a signal twenty steps down a sequence and it has usually
            faded to nothing by the time it matters, which is what made early RNNs so hard to train. LSTMs fixed it
            with three small gates: train one below, and watch its memory hold where a vanilla RNN's slips away.
          </p>
        </header>

        <div
          role="tablist"
          aria-label="Recurrent network confusions"
          onKeyDown={onTabKey}
          style={{ display: "flex", gap: 6, marginBottom: 18, flexWrap: "wrap" }}
        >
          {TABS.map((t, i) => {
            const active = i === tab;
            return (
              <button
                key={t.id}
                ref={(el) => (tabRefs.current[i] = el)}
                role="tab"
                id={`${baseId}-tab-${t.id}`}
                aria-selected={active}
                aria-controls={`${baseId}-panel-${t.id}`}
                tabIndex={active ? 0 : -1}
                onClick={() => setTab(i)}
                style={{
                  fontFamily: "inherit",
                  fontSize: 12.5,
                  fontWeight: active ? 700 : 500,
                  padding: "7px 13px",
                  borderRadius: 9,
                  cursor: "pointer",
                  border: `1px solid ${active ? C.accent : C.border}`,
                  background: active ? C.accent : C.card,
                  color: active ? "#fff" : C.muted,
                  transition: `all 150ms ${EASE}`,
                  whiteSpace: "nowrap",
                }}
              >
                <span style={{ display: "inline-block" }}>
                  <span style={{ opacity: 0.7, marginRight: 6, fontWeight: 700 }}>{i + 1}</span>
                  {t.short}
                </span>
              </button>
            );
          })}
        </div>

        <div
          role="tabpanel"
          id={`${baseId}-panel-${TABS[tab].id}`}
          aria-labelledby={`${baseId}-tab-${TABS[tab].id}`}
        >
          <div style={{ fontSize: 18, fontWeight: 700, margin: "0 0 14px", letterSpacing: "-0.01em", color: C.ink }}>
            {TABS[tab].label}
          </div>
          <ActiveComp reduced={reduced} />
        </div>
      </div>
    </div>
  );
}
