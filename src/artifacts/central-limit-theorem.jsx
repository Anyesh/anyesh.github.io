import { useState, useMemo, useEffect, useRef, useCallback } from "react";

export const meta = {
  title: "The Central Limit Theorem",
  category: "Statistics",
  description:
    "Start from a lopsided source, a hard skew or a U-shape, draw samples and average each batch, then watch the histogram of those averages climb into a bell no matter how lumpy the source was.",
  date: "2026-04-06",
  tags: ["central-limit-theorem", "sampling", "gaussian", "probability"],
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
  source: "#9a5b2c",
  sourceSoft: "#f5ece2",
  means: "#2a6f8e",
  meansSoft: "#e8f1f5",
  normal: "#2f7d53",
  good: "#2f7d53",
  goodSoft: "#e9f2ec",
  warn: "#a8453f",
  warnSoft: "#fbecea",
};

const EASE = "cubic-bezier(0.23, 1, 0.32, 1)";

function mulberry32(seed) {
  let a = seed >>> 0;
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
    const update = () => setReduce(mq.matches);
    update();
    mq.addEventListener("change", update);
    return () => mq.removeEventListener("change", update);
  }, []);
  return reduce;
}

function normalPDF(x, mu, sigma) {
  if (sigma <= 0) return 0;
  const z = (x - mu) / sigma;
  return Math.exp(-0.5 * z * z) / (sigma * Math.sqrt(2 * Math.PI));
}

// Every source is defined on the support [0, 1] so the axes stay shared, and reports
// its analytic mean and variance so the CLT overlay never depends on the empirical draws.
const SOURCES = {
  uniform: {
    label: "Uniform",
    blurb: "Flat slab: every value in the range equally likely. No skew, but nothing like a bell either.",
    draw: (rng) => rng(),
    mean: 0.5,
    variance: 1 / 12,
    shape: () => 1,
  },
  exponential: {
    label: "Strong right skew",
    blurb:
      "A long tail to the right, the classic awkward case. Most draws are small, a few are large. The averages still go Gaussian, the skew just takes a bigger n to wash out.",
    draw: (rng) => {
      const lam = 4;
      let u = rng();
      if (u <= 0) u = 1e-12;
      // Truncated exponential on [0, 1] keeps the shared axis while staying genuinely skewed.
      const x = -Math.log(1 - u * (1 - Math.exp(-lam))) / lam;
      return Math.min(x, 1);
    },
    mean: (() => {
      const lam = 4;
      const Z = 1 - Math.exp(-lam);
      return (1 - (1 + lam) * Math.exp(-lam)) / (lam * Z);
    })(),
    variance: (() => {
      const lam = 4;
      const Z = 1 - Math.exp(-lam);
      const m1 = (1 - (1 + lam) * Math.exp(-lam)) / (lam * Z);
      const m2 = (2 - (lam * lam + 2 * lam + 2) * Math.exp(-lam)) / (lam * lam * Z);
      return m2 - m1 * m1;
    })(),
    shape: (x) => 4 * Math.exp(-4 * x),
  },
  bimodal: {
    label: "U-shaped",
    blurb:
      "Mass piled at both ends with a hole in the middle, an arcsine law. The least bell-shaped thing here, yet the averages still converge.",
    draw: (rng) => {
      const u = rng();
      return 0.5 * (1 - Math.cos(Math.PI * u));
    },
    mean: 0.5,
    variance: 0.125,
    shape: (x) => 1 / (Math.PI * Math.sqrt(Math.max(x * (1 - x), 1e-6))),
  },
  dice: {
    label: "Six-sided die",
    blurb: "A discrete source: six bars, each equally likely. Averaging discrete things still smooths into a continuous bell.",
    discrete: true,
    levels: 6,
    draw: (rng) => {
      const f = Math.floor(rng() * 6);
      return (f + 0.5) / 6;
    },
    mean: 0.5,
    variance: (() => {
      let m = 0, m2 = 0;
      for (let k = 0; k < 6; k++) {
        const v = (k + 0.5) / 6;
        m += v / 6;
        m2 += (v * v) / 6;
      }
      return m2 - m * m;
    })(),
  },
  custom: {
    label: "Draw your own",
    blurb: "Drag the bars to sculpt any source you like. Make it spiky, lopsided, whatever. The averages will still go Gaussian.",
    custom: true,
  },
};

const CUSTOM_BINS = 12;
const DEFAULT_CUSTOM = [0.2, 0.9, 0.3, 0.1, 0.05, 0.05, 0.05, 0.1, 0.4, 0.8, 0.6, 0.15];

function customStats(weights) {
  const total = weights.reduce((a, b) => a + b, 0) || 1;
  const p = weights.map((w) => w / total);
  let mean = 0;
  for (let i = 0; i < CUSTOM_BINS; i++) mean += p[i] * ((i + 0.5) / CUSTOM_BINS);
  let variance = 0;
  for (let i = 0; i < CUSTOM_BINS; i++) {
    const c = (i + 0.5) / CUSTOM_BINS;
    variance += p[i] * (c - mean) * (c - mean);
  }
  const cum = [];
  let acc = 0;
  for (let i = 0; i < CUSTOM_BINS; i++) {
    acc += p[i];
    cum.push(acc);
  }
  return { p, mean, variance, cum };
}

function makeSampler(key, customWeights) {
  if (key === "custom") {
    const { cum } = customStats(customWeights);
    return (rng) => {
      const u = rng();
      let bin = 0;
      while (bin < CUSTOM_BINS - 1 && u > cum[bin]) bin++;
      return (bin + rng()) / CUSTOM_BINS;
    };
  }
  return SOURCES[key].draw;
}

function sourceMoments(key, customWeights) {
  if (key === "custom") {
    const s = customStats(customWeights);
    return { mean: s.mean, variance: s.variance };
  }
  return { mean: SOURCES[key].mean, variance: SOURCES[key].variance };
}

const N_OPTIONS = [1, 2, 5, 10, 30, 100];
const MEAN_BINS = 48;

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

function Eyebrow({ children, color }) {
  return (
    <div
      style={{
        fontSize: 10,
        letterSpacing: "0.14em",
        textTransform: "uppercase",
        color: color || C.muted,
        fontWeight: 700,
      }}
    >
      {children}
    </div>
  );
}

function Btn({ children, onClick, primary, disabled, ariaLabel, type }) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      type={type || "button"}
      aria-label={ariaLabel}
      className="clt-btn"
      style={{
        padding: primary ? "8px 18px" : "7px 14px",
        borderRadius: 9,
        border: primary ? "none" : `1px solid ${C.border}`,
        background: primary ? C.accent : C.card,
        color: primary ? "#fff" : C.ink,
        fontSize: 13,
        fontWeight: primary ? 600 : 500,
        fontFamily: "inherit",
        cursor: disabled ? "not-allowed" : "pointer",
        opacity: disabled ? 0.45 : 1,
        transition: `transform 140ms ${EASE}, background 160ms ease`,
      }}
    >
      {children}
    </button>
  );
}

function Stat({ label, value, sub, color, bg }) {
  return (
    <div style={{ background: bg || C.faint, borderRadius: 10, padding: "11px 14px", flex: 1, minWidth: 104 }}>
      <div
        style={{
          fontSize: 10.5,
          color: color || C.muted,
          fontWeight: 700,
          letterSpacing: "0.05em",
          textTransform: "uppercase",
          marginBottom: 4,
        }}
      >
        {label}
      </div>
      <div style={{ fontSize: 19, fontWeight: 700, color: color || C.ink, lineHeight: 1.05, fontFamily: "'IBM Plex Mono', ui-monospace, monospace" }}>
        {value}
      </div>
      {sub && <div style={{ fontSize: 11, color: C.muted, marginTop: 4 }}>{sub}</div>}
    </div>
  );
}

const SRC_W = 680;
const SRC_H = 150;

function SourcePlot({ sourceKey, customWeights, onPaint, reduce }) {
  const padL = 8, padR = 8, padT = 10, padB = 24;
  const inW = SRC_W - padL - padR;
  const inH = SRC_H - padT - padB;
  const svgRef = useRef(null);
  const paintingRef = useRef(false);

  const src = SOURCES[sourceKey];
  const isCustom = sourceKey === "custom";

  const bars = useMemo(() => {
    if (isCustom) {
      const total = customWeights.reduce((a, b) => a + b, 0) || 1;
      return customWeights.map((w) => (w / total) * CUSTOM_BINS);
    }
    if (src.discrete) {
      const out = [];
      for (let k = 0; k < src.levels; k++) out.push(1);
      return out;
    }
    const N = 60;
    const out = [];
    for (let i = 0; i < N; i++) {
      const x = (i + 0.5) / N;
      out.push(src.shape ? Math.max(0, src.shape(x)) : 1);
    }
    return out;
  }, [sourceKey, customWeights, isCustom, src]);

  const yMax = useMemo(() => {
    const m = Math.max(...bars, 0.001);
    return isCustom ? Math.max(m, 1.2) * 1.05 : m * 1.12;
  }, [bars, isCustom]);

  const nBars = bars.length;
  const barW = inW / nBars;

  const paintAt = useCallback(
    (clientX, clientY) => {
      const svg = svgRef.current;
      if (!svg) return;
      const rect = svg.getBoundingClientRect();
      const fx = (clientX - rect.left) / rect.width;
      const fy = (clientY - rect.top) / rect.height;
      const px = fx * SRC_W;
      const py = fy * SRC_H;
      let bin = Math.floor(((px - padL) / inW) * CUSTOM_BINS);
      bin = Math.max(0, Math.min(CUSTOM_BINS - 1, bin));
      let h = 1 - (py - padT) / inH;
      h = Math.max(0, Math.min(1, h));
      onPaint(bin, h);
    },
    [inW, onPaint]
  );

  const handleDown = (e) => {
    if (!isCustom) return;
    paintingRef.current = true;
    const p = e.touches ? e.touches[0] : e;
    paintAt(p.clientX, p.clientY);
  };
  const handleMove = (e) => {
    if (!isCustom || !paintingRef.current) return;
    e.preventDefault();
    const p = e.touches ? e.touches[0] : e;
    paintAt(p.clientX, p.clientY);
  };
  const handleUp = () => {
    paintingRef.current = false;
  };

  return (
    <svg
      ref={svgRef}
      viewBox={`0 0 ${SRC_W} ${SRC_H}`}
      width="100%"
      role="img"
      aria-label={`Shape of the source distribution: ${isCustom ? "custom, drag the bars to reshape" : src.label}.`}
      onMouseDown={handleDown}
      onMouseMove={handleMove}
      onMouseUp={handleUp}
      onMouseLeave={handleUp}
      onTouchStart={handleDown}
      onTouchMove={handleMove}
      onTouchEnd={handleUp}
      style={{
        display: "block",
        background: C.bg,
        borderRadius: 10,
        border: `1px solid ${C.border}`,
        cursor: isCustom ? "ns-resize" : "default",
        touchAction: isCustom ? "none" : "auto",
      }}
    >
      <line x1={padL} x2={padL + inW} y1={padT + inH} y2={padT + inH} stroke={C.border} strokeWidth={1} />
      {bars.map((b, i) => {
        const h = (b / yMax) * inH;
        const gap = nBars > 20 ? 0 : Math.min(2, barW * 0.12);
        const x = padL + i * barW + gap / 2;
        return (
          <rect
            key={i}
            x={x}
            y={padT + inH - h}
            width={Math.max(0, barW - gap)}
            height={Math.max(0, h)}
            fill={C.source}
            opacity={0.82}
            rx={nBars > 20 ? 0 : 2}
            style={{ transition: reduce ? "none" : `height 220ms ${EASE}, y 220ms ${EASE}` }}
          />
        );
      })}
      {[0, 0.25, 0.5, 0.75, 1].map((t) => (
        <text key={t} x={padL + t * inW} y={SRC_H - 6} textAnchor="middle" fontSize={11} fill={C.muted} fontFamily="'IBM Plex Mono', monospace">
          {t.toFixed(2)}
        </text>
      ))}
    </svg>
  );
}

const HIST_W = 680;
const HIST_H = 260;

function MeansPlot({ counts, total, mu, se, showNormal, n, reduce }) {
  const padL = 8, padR = 8, padT = 14, padB = 26;
  const inW = HIST_W - padL - padR;
  const inH = HIST_H - padT - padB;
  const xToPx = (x) => padL + x * inW;
  const binW = 1 / MEAN_BINS;

  const { densities, yMax } = useMemo(() => {
    const dens = counts.map((c) => (total > 0 ? c / total / binW : 0));
    let m = Math.max(...dens, 0.001);
    if (showNormal && se > 0) m = Math.max(m, normalPDF(mu, mu, se));
    return { densities: dens, yMax: m * 1.1 };
  }, [counts, total, showNormal, se, mu, binW]);

  const yToPx = (y) => padT + inH - (y / yMax) * inH;

  const normalPath = useMemo(() => {
    if (!showNormal || se <= 0) return null;
    const pts = [];
    const steps = 220;
    for (let i = 0; i <= steps; i++) {
      const x = i / steps;
      const d = normalPDF(x, mu, se);
      pts.push(`${i === 0 ? "M" : "L"}${xToPx(x).toFixed(2)},${yToPx(Math.min(d, yMax)).toFixed(2)}`);
    }
    return pts.join(" ");
  }, [showNormal, se, mu, yMax]);

  const ticks = [0, 0.25, 0.5, 0.75, 1];

  return (
    <svg
      viewBox={`0 0 ${HIST_W} ${HIST_H}`}
      width="100%"
      role="img"
      aria-label={`Histogram of ${total} sample means, each the average of ${n} draws. The green curve is the central limit theorem prediction, a normal centered at ${mu.toFixed(3)} with standard error ${se.toFixed(4)}.`}
      style={{ display: "block", background: C.bg, borderRadius: 10, border: `1px solid ${C.border}` }}
    >
      {ticks.map((t) => (
        <line key={t} x1={xToPx(t)} x2={xToPx(t)} y1={padT} y2={padT + inH} stroke={C.faint} strokeWidth={1} />
      ))}
      <line x1={padL} x2={padL + inW} y1={padT + inH} y2={padT + inH} stroke={C.border} strokeWidth={1} />

      {densities.map((d, i) => {
        const h = (d / yMax) * inH;
        if (h <= 0) return null;
        const x = padL + (i / MEAN_BINS) * inW;
        const w = inW / MEAN_BINS;
        return (
          <rect
            key={i}
            x={x}
            y={padT + inH - h}
            width={Math.max(0, w - 0.5)}
            height={h}
            fill={C.means}
            opacity={0.7}
            style={{ transition: reduce ? "none" : `height 180ms ${EASE}, y 180ms ${EASE}` }}
          />
        );
      })}

      {normalPath && (
        <path d={normalPath} fill="none" stroke={C.normal} strokeWidth={2.4} style={{ transition: reduce ? "none" : `d 300ms ${EASE}` }} />
      )}

      <line x1={xToPx(mu)} x2={xToPx(mu)} y1={padT} y2={padT + inH} stroke={C.normal} strokeWidth={1.4} strokeDasharray="3 3" />

      {showNormal && se > 0 && (
        <g>
          <line x1={xToPx(Math.max(0, mu - se))} x2={xToPx(Math.max(0, mu - se))} y1={padT + inH - 8} y2={padT + inH} stroke={C.normal} strokeWidth={1.4} />
          <line x1={xToPx(Math.min(1, mu + se))} x2={xToPx(Math.min(1, mu + se))} y1={padT + inH - 8} y2={padT + inH} stroke={C.normal} strokeWidth={1.4} />
        </g>
      )}

      {ticks.map((t) => (
        <text key={`l${t}`} x={xToPx(t)} y={HIST_H - 7} textAnchor="middle" fontSize={11} fill={C.muted} fontFamily="'IBM Plex Mono', monospace">
          {t.toFixed(2)}
        </text>
      ))}
    </svg>
  );
}

function LegendSwatch({ color, label, dash }) {
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
      <span
        style={{
          width: 18,
          height: dash ? 0 : 8,
          borderTop: dash ? `2px dashed ${color}` : "none",
          background: dash ? "none" : color,
          opacity: dash ? 1 : 0.7,
          borderRadius: 2,
          display: "inline-block",
        }}
      />
      {label}
    </span>
  );
}

const SEED = 0x5eed1234;

function empiricalMoments(sum, sumSq, sumCube, count) {
  if (count === 0) return { mean: 0, sd: 0, skew: 0 };
  const mean = sum / count;
  const variance = Math.max(0, sumSq / count - mean * mean);
  const sd = Math.sqrt(variance);
  if (sd <= 0 || count < 3) return { mean, sd, skew: 0 };
  const m3 = sumCube / count - 3 * mean * (sumSq / count) + 2 * mean * mean * mean;
  return { mean, sd, skew: m3 / (sd * sd * sd) };
}

export default function App() {
  const reduce = usePrefersReducedMotion();

  const [sourceKey, setSourceKey] = useState("exponential");
  const [customWeights, setCustomWeights] = useState(DEFAULT_CUSTOM);
  const [n, setN] = useState(10);
  const [showNormal, setShowNormal] = useState(true);
  const [autoRun, setAutoRun] = useState(false);

  const [counts, setCounts] = useState(() => new Array(MEAN_BINS).fill(0));
  const [agg, setAgg] = useState({ sum: 0, sumSq: 0, sumCube: 0, count: 0 });

  const rngRef = useRef(mulberry32(SEED));
  const rafRef = useRef(null);

  const samplerKeyDeps = sourceKey === "custom" ? customWeights.join(",") : sourceKey;
  const sampler = useMemo(() => makeSampler(sourceKey, customWeights), [samplerKeyDeps]);
  const { mean: srcMean, variance: srcVar } = useMemo(
    () => sourceMoments(sourceKey, customWeights),
    [samplerKeyDeps]
  );
  const srcSd = Math.sqrt(srcVar);
  const se = srcSd / Math.sqrt(n);

  const resetExperiments = useCallback(() => {
    rngRef.current = mulberry32(SEED);
    setCounts(new Array(MEAN_BINS).fill(0));
    setAgg({ sum: 0, sumSq: 0, sumCube: 0, count: 0 });
  }, []);

  useEffect(() => {
    resetExperiments();
    setAutoRun(false);
  }, [sourceKey, n, samplerKeyDeps, resetExperiments]);

  const runBatch = useCallback(
    (k) => {
      const rng = rngRef.current;
      const newCounts = new Array(MEAN_BINS).fill(0);
      let dSum = 0, dSumSq = 0, dSumCube = 0;
      for (let e = 0; e < k; e++) {
        let acc = 0;
        for (let i = 0; i < n; i++) acc += sampler(rng);
        const mean = acc / n;
        const bin = Math.min(MEAN_BINS - 1, Math.max(0, Math.floor(mean * MEAN_BINS)));
        newCounts[bin] += 1;
        dSum += mean;
        dSumSq += mean * mean;
        dSumCube += mean * mean * mean;
      }
      setCounts((prev) => {
        const out = prev.slice();
        for (let b = 0; b < MEAN_BINS; b++) if (newCounts[b]) out[b] += newCounts[b];
        return out;
      });
      setAgg((prev) => ({
        sum: prev.sum + dSum,
        sumSq: prev.sumSq + dSumSq,
        sumCube: prev.sumCube + dSumCube,
        count: prev.count + k,
      }));
    },
    [n, sampler]
  );

  useEffect(() => {
    if (!autoRun) {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      return;
    }
    let stopped = false;
    const tick = () => {
      if (stopped) return;
      runBatch(reduce ? 120 : 30);
      setAgg((a) => {
        if (a.count >= 20000) {
          setAutoRun(false);
          return a;
        }
        return a;
      });
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => {
      stopped = true;
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [autoRun, runBatch, reduce]);

  const emp = useMemo(
    () => empiricalMoments(agg.sum, agg.sumSq, agg.sumCube, agg.count),
    [agg]
  );

  const paintCustom = useCallback((bin, height) => {
    setCustomWeights((prev) => {
      const out = prev.slice();
      out[bin] = Math.max(0.02, height);
      return out;
    });
  }, []);

  const src = SOURCES[sourceKey];
  const seRatio = se > 0 ? Math.abs(emp.sd / se) : 0;
  const skewMag = Math.abs(emp.skew);
  const bellLabel =
    agg.count < 80 ? "collecting" : skewMag < 0.25 ? "bell-shaped" : skewMag < 0.6 ? "getting there" : "still lumpy";
  const bellColor = agg.count < 80 ? C.muted : skewMag < 0.25 ? C.good : skewMag < 0.6 ? C.accent : C.warn;

  return (
    <div
      style={{
        fontFamily: "'Iowan Old Style', 'Palatino Linotype', Palatino, Georgia, serif",
        background: C.bg,
        minHeight: "100vh",
        padding: "30px 14px 56px",
        color: C.ink,
      }}
    >
      <style>{`
        .clt-btn:active { transform: scale(0.97); }
        button:focus-visible, input:focus-visible, select:focus-visible {
          outline: 2px solid ${C.accent}; outline-offset: 2px;
        }
        input[type=range] { accent-color: ${C.accent}; }
        @media (prefers-reduced-motion: reduce) {
          * { transition-duration: 1ms !important; animation-duration: 1ms !important; }
        }
      `}</style>

      <div style={{ maxWidth: 720, margin: "0 auto" }}>
        <header style={{ marginBottom: 22 }}>
          <Eyebrow>Statistics · Why Averages Go Gaussian</Eyebrow>
          <h1 style={{ fontSize: 30, fontWeight: 700, margin: "6px 0 0", lineHeight: 1.1, textWrap: "balance" }}>
            The Central Limit Theorem
          </h1>
          <p style={{ color: C.ink, fontSize: 15, margin: "10px 0 0", lineHeight: 1.6, maxWidth: "64ch", opacity: 0.9 }}>
            Pick a source that looks nothing like a bell: a hard right skew, a U-shape, a die. Then draw a sample
            of n values from it, take their average, and drop that one average into a histogram. Repeat thousands of
            times. The histogram of averages climbs into a bell anyway, and the bigger n gets the tighter and more
            Gaussian it becomes, no matter how lumpy the source you started from.
          </p>
        </header>

        <Card style={{ marginBottom: 18, background: C.accentSoft, borderColor: `${C.accent}33` }}>
          <Eyebrow color={C.accent}>The three words you need</Eyebrow>
          <div style={{ fontSize: 13.5, color: C.ink, lineHeight: 1.65, marginTop: 10, display: "grid", gap: 7 }}>
            <div>
              <b>Population.</b> The source itself, the thing you draw from. It can be any shape with a finite spread.
            </div>
            <div>
              <b>A sample.</b> n values drawn from the population. Its average is one number, written x&#772;.
            </div>
            <div>
              <b>Sampling distribution of the mean.</b> The histogram you build by collecting many of those averages.
              The claim of the central limit theorem is that this third thing approaches a normal curve as n grows,
              no matter what the population looked like.
            </div>
          </div>
        </Card>

        <Card style={{ marginBottom: 18 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 10, flexWrap: "wrap", marginBottom: 10 }}>
            <h2 style={{ fontSize: 19, fontWeight: 700, margin: 0 }}>1. Choose a deliberately non-normal source</h2>
          </div>
          <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginBottom: 12 }}>
            {Object.entries(SOURCES).map(([key, s]) => (
              <button
                key={key}
                onClick={() => setSourceKey(key)}
                aria-pressed={sourceKey === key}
                className="clt-btn"
                style={{
                  padding: "6px 12px",
                  borderRadius: 8,
                  border: `1.5px solid ${sourceKey === key ? C.source : C.border}`,
                  background: sourceKey === key ? C.sourceSoft : C.card,
                  color: sourceKey === key ? C.source : C.muted,
                  fontSize: 12.5,
                  fontWeight: sourceKey === key ? 700 : 500,
                  fontFamily: "inherit",
                  cursor: "pointer",
                  transition: `transform 140ms ${EASE}`,
                }}
              >
                {s.label}
              </button>
            ))}
          </div>
          <p style={{ fontSize: 13, color: C.ink, opacity: 0.82, margin: "0 0 12px", lineHeight: 1.6, maxWidth: "66ch" }}>
            {src.blurb}
          </p>
          <SourcePlot sourceKey={sourceKey} customWeights={customWeights} onPaint={paintCustom} reduce={reduce} />
          {sourceKey === "custom" && (
            <div style={{ fontSize: 12, color: C.muted, marginTop: 8, fontStyle: "italic" }}>
              Drag across the bars to set their heights. The averages will still find their way to a bell.
            </div>
          )}
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginTop: 14 }}>
            <Stat label="Population mean μ" value={srcMean.toFixed(3)} color={C.source} bg={C.sourceSoft} />
            <Stat label="Population sd σ" value={srcSd.toFixed(3)} color={C.source} bg={C.sourceSoft} />
            <Stat label="Variance σ²" value={srcVar.toFixed(3)} color={C.muted} bg={C.faint} />
          </div>
        </Card>

        <Card style={{ marginBottom: 18 }}>
          <h2 style={{ fontSize: 19, fontWeight: 700, margin: "0 0 6px" }}>2. Set the sample size n</h2>
          <p style={{ fontSize: 13, color: C.ink, opacity: 0.82, margin: "0 0 14px", lineHeight: 1.6, maxWidth: "66ch" }}>
            n is how many draws go into each average. At n = 1 the average is just a single draw, so the histogram
            of means is literally the source itself, the honest base case with no bell in sight. As n climbs, each
            average pools more draws and the histogram tightens.
          </p>
          <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
            {N_OPTIONS.map((opt) => (
              <button
                key={opt}
                onClick={() => setN(opt)}
                aria-pressed={n === opt}
                aria-label={`Sample size ${opt}`}
                className="clt-btn"
                style={{
                  padding: "8px 16px",
                  borderRadius: 8,
                  border: `1.5px solid ${n === opt ? C.means : C.border}`,
                  background: n === opt ? C.meansSoft : C.card,
                  color: n === opt ? C.means : C.muted,
                  fontSize: 14,
                  fontWeight: n === opt ? 700 : 500,
                  fontFamily: "'IBM Plex Mono', monospace",
                  cursor: "pointer",
                  transition: `transform 140ms ${EASE}`,
                }}
              >
                {opt}
              </button>
            ))}
          </div>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginTop: 16 }}>
            <Stat label="Predicted center" value={srcMean.toFixed(3)} sub="same as μ" color={C.normal} bg={C.goodSoft} />
            <Stat label="Predicted spread (SE)" value={se.toFixed(4)} sub={`σ / √n = ${srcSd.toFixed(3)} / ${Math.sqrt(n).toFixed(2)}`} color={C.normal} bg={C.goodSoft} />
          </div>
        </Card>

        <Card style={{ marginBottom: 18 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 10, flexWrap: "wrap", marginBottom: 6 }}>
            <h2 style={{ fontSize: 19, fontWeight: 700, margin: 0 }}>3. Sample for real and watch it build</h2>
            <span style={{ fontSize: 12, color: C.muted }}>x-axis: the value of one sample mean</span>
          </div>
          <p style={{ fontSize: 13, color: C.ink, opacity: 0.82, margin: "0 0 14px", lineHeight: 1.6, maxWidth: "66ch" }}>
            Each draw of "many" experiments takes thousands of fresh samples, averages each, and stacks the result.
            The green curve is the central limit theorem's prediction: a normal centered at μ with standard deviation
            σ / √n. Watch the blue histogram chase that curve.
          </p>
          <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap", marginBottom: 14 }}>
            <Btn primary onClick={() => runBatch(1)} ariaLabel="Draw one sample mean">Draw one</Btn>
            <Btn onClick={() => runBatch(500)} ariaLabel="Draw five hundred sample means">+500</Btn>
            <Btn onClick={() => runBatch(5000)} ariaLabel="Draw five thousand sample means">+5000</Btn>
            <Btn onClick={() => setAutoRun((v) => !v)} ariaLabel={autoRun ? "Pause the auto run" : "Auto run the sampler"}>
              {autoRun ? "Pause" : "Auto-run"}
            </Btn>
            <Btn onClick={resetExperiments} ariaLabel="Reset all experiments">Reset</Btn>
            <label style={{ display: "inline-flex", alignItems: "center", gap: 6, fontSize: 12.5, color: C.muted, marginLeft: "auto", cursor: "pointer" }}>
              <input
                type="checkbox"
                checked={showNormal}
                onChange={(e) => setShowNormal(e.target.checked)}
                aria-label="Toggle the theoretical normal overlay"
                style={{ accentColor: C.normal, width: 15, height: 15 }}
              />
              Normal overlay
            </label>
          </div>

          <MeansPlot counts={counts} total={agg.count} mu={srcMean} se={se} showNormal={showNormal} n={n} reduce={reduce} />

          <div style={{ display: "flex", gap: 16, flexWrap: "wrap", marginTop: 12, fontSize: 12, color: C.muted }}>
            <LegendSwatch color={C.means} label="histogram of sample means" />
            <LegendSwatch color={C.normal} label="CLT normal N(μ, σ²/n)" />
            <LegendSwatch color={C.normal} label="true mean μ" dash />
          </div>

          {agg.count === 0 && (
            <div style={{ marginTop: 12, fontSize: 12.5, color: C.muted, fontStyle: "italic" }}>
              No experiments yet. Draw some sample means to start building the histogram.
            </div>
          )}
        </Card>

        <Card style={{ marginBottom: 18 }}>
          <Eyebrow color={C.means}>Live readouts: prediction vs reality</Eyebrow>
          <h2 style={{ fontSize: 19, fontWeight: 700, margin: "5px 0 4px" }}>Does the histogram actually match σ / √n?</h2>
          <p style={{ fontSize: 13, color: C.ink, opacity: 0.82, margin: "0 0 14px", lineHeight: 1.6, maxWidth: "66ch" }}>
            The empirical numbers come straight from the means you collected. If the central limit theorem holds, the
            empirical mean lands on μ, the empirical spread lands on σ / √n, and the skew of the collected means
            falls toward zero as n grows.
          </p>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
            <Stat label="Experiments" value={agg.count.toLocaleString()} color={C.ink} bg={C.faint} />
            <Stat label="Empirical mean" value={agg.count ? emp.mean.toFixed(4) : "—"} sub={`predicted ${srcMean.toFixed(4)}`} color={C.means} bg={C.meansSoft} />
            <Stat label="Empirical spread" value={agg.count ? emp.sd.toFixed(4) : "—"} sub={`predicted SE ${se.toFixed(4)}`} color={C.means} bg={C.meansSoft} />
          </div>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginTop: 8 }}>
            <Stat
              label="Spread ÷ predicted SE"
              value={agg.count ? seRatio.toFixed(3) : "—"}
              sub="should approach 1.000"
              color={agg.count && Math.abs(seRatio - 1) < 0.06 ? C.good : C.ink}
              bg={agg.count && Math.abs(seRatio - 1) < 0.06 ? C.goodSoft : C.faint}
            />
            <Stat
              label="Skew of the means"
              value={agg.count >= 3 ? emp.skew.toFixed(3) : "—"}
              sub="0 = symmetric bell"
              color={bellColor}
              bg={agg.count < 80 ? C.faint : skewMag < 0.25 ? C.goodSoft : C.warnSoft}
            />
            <Stat label="Verdict" value={bellLabel} sub={`n = ${n}`} color={bellColor} bg={C.faint} />
          </div>
        </Card>

        <Card style={{ background: C.bg, border: `1px solid ${C.border}` }}>
          <Eyebrow color={C.accent}>What you just watched, and the fine print</Eyebrow>
          <div style={{ fontSize: 13, color: C.ink, lineHeight: 1.7, display: "grid", gap: 10, marginTop: 10 }}>
            <p style={{ margin: 0 }}>
              <b>The source shape stops mattering.</b> Skew, U-shape, dice: it makes no difference. Averaging mixes
              many independent draws, and the lumps in any one draw cancel against the lumps in the others. What
              survives is the smooth, symmetric pile in the middle. That is the bell.
            </p>
            <p style={{ margin: 0 }}>
              <b>Bigger n means narrower and more Gaussian.</b> The spread of the means is σ / √n, so going from n = 1
              to n = 100 cuts it tenfold. It shrinks like 1 / √n rather than 1 / n because variances add when you sum
              independent draws: summing n of them multiplies the variance by n, dividing by n to average divides it
              by n², and √(σ² / n) = σ / √n. That square root is why quadrupling the sample only halves the error.
            </p>
            <p style={{ margin: 0 }}>
              <b>The limit is N(μ, σ² / n), not just any bell.</b> It is centered exactly on the population mean and
              its width is fixed by the population spread. The green curve was drawn from μ and σ alone, never from
              the histogram, yet the histogram converges onto it.
            </p>
            <p style={{ margin: 0 }}>
              <b>The caveat is real.</b> The theorem needs the population to have finite variance, and n = 30 is only
              a rule of thumb. Switch to the strong right skew and you can see it: at small n the histogram is still
              visibly lopsided and its skew readout stays well above zero, taking a larger n to settle than the tidy
              uniform does. Heavier skew, slower convergence. Infinite-variance sources like a Cauchy never converge
              at all, which is exactly why the finite-variance condition is not optional.
            </p>
          </div>
        </Card>

        <div style={{ marginTop: 18, textAlign: "center", fontSize: 11, color: C.muted }}>
          Seeded mulberry32 draws · analytic μ and σ per source · empirical moments from the collected means · N(μ, σ²/n) overlay
        </div>
      </div>
    </div>
  );
}
