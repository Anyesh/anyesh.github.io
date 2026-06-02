import { useState, useMemo, useEffect, useRef, useCallback } from "react";

export const meta = {
  title: "Bayesian vs Frequentist",
  category: "Statistics",
  description:
    "One coin, one question: what is its bias? Estimate it two ways on the same data. A frequentist confidence interval that you watch earn its 95% over repeated samples, and a Bayesian Beta posterior that says outright how probable each value of p is.",
  date: "2026-05-28",
  tags: ["bayesian", "frequentist", "confidence-interval", "credible-interval"],
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
  freq: "#2a6f8e",
  freqSoft: "#e8f1f5",
  bayes: "#9a5b2c",
  bayesSoft: "#f5ece2",
  prior: "#b9a98f",
  good: "#2f7d53",
  goodSoft: "#e9f2ec",
  bad: "#a8453f",
  badSoft: "#fbecea",
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

function logGamma(x) {
  const g = 7;
  const c = [
    0.99999999999980993, 676.5203681218851, -1259.1392167224028,
    771.32342877765313, -176.61502916214059, 12.507343278686905,
    -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
  ];
  if (x < 0.5) {
    return Math.log(Math.PI / Math.sin(Math.PI * x)) - logGamma(1 - x);
  }
  x -= 1;
  let a = c[0];
  const t = x + g + 0.5;
  for (let i = 1; i < g + 2; i++) a += c[i] / (x + i);
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}

function logBeta(a, b) {
  return logGamma(a) + logGamma(b) - logGamma(a + b);
}

// Regularized incomplete beta I_x(a,b) via Lentz continued fraction (Numerical Recipes betacf).
function betacf(x, a, b) {
  const FPMIN = 1e-300;
  const qab = a + b;
  const qap = a + 1;
  const qam = a - 1;
  let c = 1;
  let d = 1 - (qab * x) / qap;
  if (Math.abs(d) < FPMIN) d = FPMIN;
  d = 1 / d;
  let h = d;
  for (let m = 1; m <= 300; m++) {
    const m2 = 2 * m;
    let aa = (m * (b - m) * x) / ((qam + m2) * (a + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c;
    if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d;
    h *= d * c;
    aa = (-(a + m) * (qab + m) * x) / ((a + m2) * (qap + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c;
    if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1) < 1e-12) break;
  }
  return h;
}

function betaCDF(x, a, b) {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const lbeta = logBeta(a, b);
  const front = Math.exp(a * Math.log(x) + b * Math.log(1 - x) - lbeta);
  if (x < (a + 1) / (a + b + 2)) {
    return (front * betacf(x, a, b)) / a;
  }
  return 1 - (front * betacf(1 - x, b, a)) / b;
}

function betaQuantile(p, a, b) {
  if (p <= 0) return 0;
  if (p >= 1) return 1;
  let lo = 0;
  let hi = 1;
  let mid = a / (a + b);
  for (let i = 0; i < 80; i++) {
    mid = 0.5 * (lo + hi);
    const c = betaCDF(mid, a, b);
    if (c < p) lo = mid;
    else hi = mid;
    if (hi - lo < 1e-10) break;
  }
  return 0.5 * (lo + hi);
}

function betaLogPDF(x, a, b) {
  if (x <= 0 || x >= 1) return -Infinity;
  return (a - 1) * Math.log(x) + (b - 1) * Math.log(1 - x) - logBeta(a, b);
}

function betaPDF(x, a, b) {
  const lp = betaLogPDF(x, a, b);
  return lp === -Infinity ? 0 : Math.exp(lp);
}

function normalQuantile(p) {
  // Acklam's rational approximation to the standard normal inverse CDF.
  const a = [
    -3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2,
    1.38357751867269e2, -3.066479806614716e1, 2.506628277459239,
  ];
  const b = [
    -5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2,
    6.680131188771972e1, -1.328068155288572e1,
  ];
  const c = [
    -7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838,
    -2.549732539343734, 4.374664141464968, 2.938163982698783,
  ];
  const d = [
    7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996,
    3.754408661907416,
  ];
  const plow = 0.02425;
  const phigh = 1 - plow;
  if (p < plow) {
    const q = Math.sqrt(-2 * Math.log(p));
    return (
      (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    );
  }
  if (p <= phigh) {
    const q = p - 0.5;
    const r = q * q;
    return (
      ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) *
        q) /
      (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    );
  }
  const q = Math.sqrt(-2 * Math.log(1 - p));
  return (
    -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
    ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
  );
}

function waldInterval(h, n, conf) {
  if (n === 0) return [0, 1];
  const phat = h / n;
  const z = normalQuantile(0.5 + conf / 2);
  const se = Math.sqrt((phat * (1 - phat)) / n);
  return [Math.max(0, phat - z * se), Math.min(1, phat + z * se)];
}

function wilsonInterval(h, n, conf) {
  if (n === 0) return [0, 1];
  const phat = h / n;
  const z = normalQuantile(0.5 + conf / 2);
  const z2 = z * z;
  const denom = 1 + z2 / n;
  const center = (phat + z2 / (2 * n)) / denom;
  const half =
    (z * Math.sqrt((phat * (1 - phat)) / n + z2 / (4 * n * n))) / denom;
  return [Math.max(0, center - half), Math.min(1, center + half)];
}

function drawBinomial(n, p, rng) {
  let h = 0;
  for (let i = 0; i < n; i++) if (rng() < p) h++;
  return h;
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

const PRIORS = {
  uniform: { label: "Uniform Beta(1,1)", a: 1, b: 1, note: "flat: every bias equally plausible before data" },
  jeffreys: { label: "Jeffreys Beta(1/2,1/2)", a: 0.5, b: 0.5, note: "the reference prior, mild weight toward the edges" },
  fair: { label: "Believes-fair Beta(20,20)", a: 20, b: 20, note: "informative: starts convinced the coin is near fair" },
};

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
      className="bvf-btn"
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
    <div style={{ background: bg || C.faint, borderRadius: 10, padding: "11px 14px", flex: 1, minWidth: 112 }}>
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
      <div style={{ fontSize: 20, fontWeight: 700, color: color || C.ink, lineHeight: 1.05, fontFamily: "'IBM Plex Mono', ui-monospace, monospace" }}>
        {value}
      </div>
      {sub && <div style={{ fontSize: 11, color: C.muted, marginTop: 4 }}>{sub}</div>}
    </div>
  );
}

const PLOT_W = 680;
const PLOT_H = 220;
const PAD_L = 8;
const PAD_R = 8;
const PAD_T = 14;
const PAD_B = 26;

function DensityPlot({ posteriorA, posteriorB, priorA, priorB, freqCI, credCI, truP, phat, reduce }) {
  const samples = 220;
  const inW = PLOT_W - PAD_L - PAD_R;
  const inH = PLOT_H - PAD_T - PAD_B;
  const xToPx = (x) => PAD_L + x * inW;

  const { postPts, priorPts, yMax } = useMemo(() => {
    const post = [];
    const pri = [];
    let m = 0;
    for (let i = 0; i <= samples; i++) {
      const x = i / samples;
      const dp = betaPDF(x, posteriorA, posteriorB);
      const dpr = betaPDF(x, priorA, priorB);
      post.push([x, dp]);
      pri.push([x, dpr]);
      if (isFinite(dp) && dp > m) m = dp;
      if (isFinite(dpr) && dpr > m) m = dpr;
    }
    return { postPts: post, priorPts: pri, yMax: m * 1.08 || 1 };
  }, [posteriorA, posteriorB, priorA, priorB]);

  const yToPx = (y) => PAD_T + inH - (y / yMax) * inH;

  const pathFrom = (pts) =>
    pts
      .map(([x, y], i) => `${i === 0 ? "M" : "L"}${xToPx(x).toFixed(2)},${yToPx(Math.min(y, yMax)).toFixed(2)}`)
      .join(" ");

  const areaFrom = (pts) => {
    const top = pathFrom(pts);
    return `${top} L${xToPx(1).toFixed(2)},${yToPx(0).toFixed(2)} L${xToPx(0).toFixed(2)},${yToPx(0).toFixed(2)} Z`;
  };

  const ticks = [0, 0.25, 0.5, 0.75, 1];

  return (
    <svg
      viewBox={`0 0 ${PLOT_W} ${PLOT_H}`}
      width="100%"
      role="img"
      aria-label={`Probability density over the coin bias p. Posterior Beta(${posteriorA.toFixed(1)}, ${posteriorB.toFixed(1)}).`}
      style={{ display: "block", background: C.bg, borderRadius: 10, border: `1px solid ${C.border}` }}
    >
      {ticks.map((t) => (
        <line key={t} x1={xToPx(t)} x2={xToPx(t)} y1={PAD_T} y2={PAD_T + inH} stroke={C.faint} strokeWidth={1} />
      ))}
      <line x1={PAD_L} x2={PAD_L + inW} y1={PAD_T + inH} y2={PAD_T + inH} stroke={C.border} strokeWidth={1} />

      {freqCI && (
        <g>
          <rect
            x={xToPx(freqCI[0])}
            y={PAD_T + inH + 6}
            width={Math.max(0, xToPx(freqCI[1]) - xToPx(freqCI[0]))}
            height={5}
            rx={2.5}
            fill={C.freq}
            opacity={0.85}
          />
        </g>
      )}

      <path d={areaFrom(priorPts)} fill={C.prior} opacity={0.16} />
      <path
        d={pathFrom(priorPts)}
        fill="none"
        stroke={C.prior}
        strokeWidth={1.5}
        strokeDasharray="5 4"
        style={{ transition: reduce ? "none" : `d 360ms ${EASE}` }}
      />

      {credCI && (
        <path
          d={areaFrom(postPts.filter(([x]) => x >= credCI[0] && x <= credCI[1]))}
          fill={C.bayes}
          opacity={0.16}
        />
      )}
      <path
        d={pathFrom(postPts)}
        fill="none"
        stroke={C.bayes}
        strokeWidth={2.4}
        style={{ transition: reduce ? "none" : `d 360ms ${EASE}` }}
      />

      {typeof truP === "number" && (
        <line x1={xToPx(truP)} x2={xToPx(truP)} y1={PAD_T} y2={PAD_T + inH} stroke={C.good} strokeWidth={1.6} strokeDasharray="3 3" />
      )}
      {typeof phat === "number" && (
        <line x1={xToPx(phat)} x2={xToPx(phat)} y1={PAD_T} y2={PAD_T + inH} stroke={C.freq} strokeWidth={1.4} />
      )}

      {ticks.map((t) => (
        <text key={`l${t}`} x={xToPx(t)} y={PLOT_H - 6} textAnchor="middle" fontSize={11} fill={C.muted} fontFamily="'IBM Plex Mono', monospace">
          {t}
        </text>
      ))}
    </svg>
  );
}

const COV_W = 680;

function CoveragePlot({ intervals, truP }) {
  if (!intervals || intervals.length === 0) return null;
  const shown = intervals.slice(0, 60);
  const rowH = 4;
  const gap = 1.6;
  const padX = 8;
  const padTop = 8;
  const padBottom = 22;
  const inW = COV_W - padX * 2;
  const h = padTop + shown.length * (rowH + gap) + padBottom;
  const xToPx = (x) => padX + x * inW;
  const ticks = [0, 0.25, 0.5, 0.75, 1];

  return (
    <svg
      viewBox={`0 0 ${COV_W} ${h}`}
      width="100%"
      role="img"
      aria-label={`Stack of ${shown.length} confidence intervals from repeated samples. Intervals that miss the true bias are highlighted.`}
      style={{ display: "block", background: C.bg, borderRadius: 10, border: `1px solid ${C.border}` }}
    >
      {ticks.map((t) => (
        <line key={t} x1={xToPx(t)} x2={xToPx(t)} y1={padTop} y2={h - padBottom} stroke={C.faint} strokeWidth={1} />
      ))}
      <line x1={xToPx(truP)} x2={xToPx(truP)} y1={padTop - 2} y2={h - padBottom + 2} stroke={C.good} strokeWidth={1.6} strokeDasharray="3 3" />
      {shown.map((iv, i) => {
        const y = padTop + i * (rowH + gap);
        const miss = truP < iv[0] || truP > iv[1];
        return (
          <line
            key={i}
            x1={xToPx(iv[0])}
            x2={xToPx(iv[1])}
            y1={y + rowH / 2}
            y2={y + rowH / 2}
            stroke={miss ? C.bad : C.freq}
            strokeWidth={miss ? 3 : 2}
            strokeLinecap="round"
            opacity={miss ? 1 : 0.5}
          />
        );
      })}
      {ticks.map((t) => (
        <text key={`l${t}`} x={xToPx(t)} y={h - 6} textAnchor="middle" fontSize={11} fill={C.muted} fontFamily="'IBM Plex Mono', monospace">
          {t}
        </text>
      ))}
    </svg>
  );
}

const SEED = 0x1234abcd;

export default function App() {
  const reduce = usePrefersReducedMotion();

  const [truP, setTruP] = useState(0.7);
  const [conf, setConf] = useState(0.95);
  const [priorKey, setPriorKey] = useState("uniform");
  const [heads, setHeads] = useState(0);
  const [tails, setTails] = useState(0);
  const [method, setMethod] = useState("wilson");

  const rngRef = useRef(mulberry32(SEED));

  const flip = useCallback(
    (count) => {
      const rng = rngRef.current;
      let h = 0;
      for (let i = 0; i < count; i++) if (rng() < truP) h++;
      setHeads((v) => v + h);
      setTails((v) => v + (count - h));
    },
    [truP]
  );

  const reset = useCallback(() => {
    rngRef.current = mulberry32(SEED);
    setHeads(0);
    setTails(0);
  }, []);

  useEffect(() => {
    reset();
  }, [truP, reset]);

  const n = heads + tails;
  const phat = n > 0 ? heads / n : null;

  const prior = PRIORS[priorKey];
  const postA = prior.a + heads;
  const postB = prior.b + tails;
  const postMean = postA / (postA + postB);
  const postMode = postA > 1 && postB > 1 ? (postA - 1) / (postA + postB - 2) : null;

  const credCI = useMemo(() => {
    const lo = (1 - conf) / 2;
    const hi = 1 - lo;
    return [betaQuantile(lo, postA, postB), betaQuantile(hi, postA, postB)];
  }, [postA, postB, conf]);

  const freqCI = useMemo(() => {
    if (n === 0) return null;
    return method === "wald"
      ? waldInterval(heads, n, conf)
      : wilsonInterval(heads, n, conf);
  }, [heads, n, conf, method]);

  const [coverage, setCoverage] = useState(null);
  const [covRunning, setCovRunning] = useState(false);
  const covN = n > 0 ? n : 20;

  const runCoverage = useCallback(
    (experiments) => {
      setCovRunning(true);
      const rng = mulberry32((SEED ^ (covN * 2654435761)) >>> 0);
      const intervals = [];
      let hit = 0;
      let hitWald = 0;
      for (let e = 0; e < experiments; e++) {
        const h = drawBinomial(covN, truP, rng);
        const wil = wilsonInterval(h, covN, conf);
        const wal = waldInterval(h, covN, conf);
        if (truP >= wil[0] && truP <= wil[1]) hit++;
        if (truP >= wal[0] && truP <= wal[1]) hitWald++;
        intervals.push(method === "wald" ? wal : wil);
      }
      setCoverage({
        experiments,
        n: covN,
        empirical: hit / experiments,
        empiricalWald: hitWald / experiments,
        intervals,
      });
      setCovRunning(false);
    },
    [covN, truP, conf, method]
  );

  useEffect(() => {
    setCoverage(null);
  }, [truP, conf, covN, method]);

  const confPct = Math.round(conf * 100);

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
        .bvf-btn:active { transform: scale(0.97); }
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
          <Eyebrow>Statistics · Two Philosophies of Inference</Eyebrow>
          <h1 style={{ fontSize: 30, fontWeight: 700, margin: "6px 0 0", lineHeight: 1.1, textWrap: "balance" }}>
            Bayesian vs Frequentist
          </h1>
          <p style={{ color: C.ink, fontSize: 15, margin: "10px 0 0", lineHeight: 1.6, maxWidth: "64ch", opacity: 0.9 }}>
            A coin has some unknown bias p toward heads. You flip it and watch two schools answer the same
            question from the same data. The frequentist treats p as one fixed number and reports an interval
            whose 95% is a property of the procedure. The Bayesian treats p as uncertain, carries a distribution,
            and reports an interval you can read as a direct probability. Every number below is computed for real
            from seeded draws, no fakery.
          </p>
        </header>

        <Card style={{ marginBottom: 18, background: C.accentSoft, borderColor: `${C.accent}33` }}>
          <Eyebrow color={C.accent}>The coin and your data</Eyebrow>
          <div style={{ display: "flex", gap: 22, flexWrap: "wrap", alignItems: "flex-start", margin: "14px 0 4px" }}>
            <div style={{ flex: "1 1 240px", minWidth: 220 }}>
              <label htmlFor="truep" style={{ fontSize: 13, color: C.muted, display: "block", marginBottom: 6 }}>
                True bias p (hidden from both methods, changing it reseeds the draws)
              </label>
              <input
                id="truep"
                type="range"
                min={0.02}
                max={0.98}
                step={0.01}
                value={truP}
                onChange={(e) => setTruP(+e.target.value)}
                aria-label="True coin bias toward heads"
                style={{ width: "100%" }}
              />
              <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 22, fontWeight: 700, color: C.accent, marginTop: 4 }}>
                p = {truP.toFixed(2)}
              </div>
            </div>
            <div style={{ flex: "1 1 200px", minWidth: 200 }}>
              <div style={{ fontSize: 13, color: C.muted, marginBottom: 8 }}>Flip the coin</div>
              <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                <Btn primary onClick={() => flip(1)} ariaLabel="Flip the coin once">Flip once</Btn>
                <Btn onClick={() => flip(10)} ariaLabel="Flip the coin ten times">+10</Btn>
                <Btn onClick={() => flip(100)} ariaLabel="Flip the coin one hundred times">+100</Btn>
                <Btn onClick={reset} ariaLabel="Reset all flips">Reset</Btn>
              </div>
              <div style={{ display: "flex", gap: 8, marginTop: 12 }}>
                <Stat label="Heads" value={heads} color={C.accent} bg={C.card} />
                <Stat label="Tails" value={tails} color={C.muted} bg={C.card} />
                <Stat label="Flips n" value={n} color={C.ink} bg={C.card} />
              </div>
            </div>
          </div>
        </Card>

        <Card style={{ marginBottom: 18 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 10, flexWrap: "wrap", marginBottom: 4 }}>
            <h2 style={{ fontSize: 19, fontWeight: 700, margin: 0 }}>Same data, both intervals on one axis</h2>
            <span style={{ fontSize: 12, color: C.muted }}>p axis: 0 (always tails) to 1 (always heads)</span>
          </div>
          <p style={{ fontSize: 13, color: C.ink, opacity: 0.82, margin: "8px 0 14px", lineHeight: 1.6, maxWidth: "66ch" }}>
            The curve is the Bayesian posterior density over p. The dashed curve is the prior you started from.
            The shaded band under the curve is the {confPct}% credible interval. The flat bar just below the axis
            is the frequentist {confPct}% interval. The green dashed line is the true p; the solid blue line is the
            point estimate p&#770; = heads / n.
          </p>
          <DensityPlot
            posteriorA={postA}
            posteriorB={postB}
            priorA={prior.a}
            priorB={prior.b}
            freqCI={freqCI}
            credCI={credCI}
            truP={truP}
            phat={phat}
            reduce={reduce}
          />
          <div style={{ display: "flex", gap: 16, flexWrap: "wrap", marginTop: 12, fontSize: 12, color: C.muted }}>
            <LegendSwatch color={C.bayes} label="posterior density" />
            <LegendSwatch color={C.prior} label="prior (dashed)" dash />
            <LegendSwatch color={C.freq} label={`${confPct}% confidence interval`} />
            <LegendSwatch color={C.bayes} label={`${confPct}% credible interval`} faded />
            <LegendSwatch color={C.good} label="true p" dash />
          </div>
          {n === 0 && (
            <div style={{ marginTop: 12, fontSize: 12.5, color: C.muted, fontStyle: "italic" }}>
              No flips yet, so the posterior is just the prior and the frequentist interval is undefined. Flip the coin to begin.
            </div>
          )}
        </Card>

        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: 16, marginBottom: 18 }}>
          <Card style={{ borderColor: `${C.freq}33` }}>
            <Eyebrow color={C.freq}>Frequentist</Eyebrow>
            <h3 style={{ fontSize: 17, fontWeight: 700, margin: "5px 0 10px" }}>p is fixed, the interval is random</h3>
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 12 }}>
              <Stat label="MLE p-hat" value={phat === null ? "—" : phat.toFixed(3)} color={C.freq} bg={C.freqSoft} />
              <Stat
                label={`${confPct}% ${method === "wald" ? "Wald" : "Wilson"} CI`}
                value={freqCI ? `[${freqCI[0].toFixed(3)}, ${freqCI[1].toFixed(3)}]` : "—"}
                color={C.freq}
                bg={C.freqSoft}
              />
            </div>
            <div style={{ display: "flex", gap: 8, marginBottom: 12 }}>
              {["wilson", "wald"].map((mtd) => (
                <button
                  key={mtd}
                  onClick={() => setMethod(mtd)}
                  className="bvf-btn"
                  aria-pressed={method === mtd}
                  style={{
                    padding: "6px 12px",
                    borderRadius: 8,
                    border: `1.5px solid ${method === mtd ? C.freq : C.border}`,
                    background: method === mtd ? C.freqSoft : C.card,
                    color: method === mtd ? C.freq : C.muted,
                    fontSize: 12,
                    fontWeight: method === mtd ? 700 : 500,
                    fontFamily: "inherit",
                    cursor: "pointer",
                    transition: `transform 140ms ${EASE}`,
                  }}
                >
                  {mtd === "wilson" ? "Wilson" : "Wald"}
                </button>
              ))}
            </div>
            <p style={{ fontSize: 12.5, color: C.ink, opacity: 0.82, lineHeight: 1.6, margin: 0 }}>
              The bias p is one fixed unknown constant; it has no probability distribution. Probability lives in
              the long-run frequency of the procedure. The {confPct}% confidence interval is a recipe: if you
              repeated the whole experiment forever, about {confPct}% of the intervals it produces would cover the
              true p.{" "}
              {method === "wald"
                ? "Wald uses the normal approximation p̂ ± z·√(p̂(1−p̂)/n). It is the textbook formula and it under-covers badly at small n or extreme p."
                : "Wilson inverts the score test and keeps coverage near nominal even at small n, which is why it is preferred."}
            </p>
          </Card>

          <Card style={{ borderColor: `${C.bayes}33` }}>
            <Eyebrow color={C.bayes}>Bayesian</Eyebrow>
            <h3 style={{ fontSize: 17, fontWeight: 700, margin: "5px 0 10px" }}>p is uncertain, described by a distribution</h3>
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 12 }}>
              <Stat label="Posterior mean" value={postMean.toFixed(3)} color={C.bayes} bg={C.bayesSoft} />
              <Stat label="MAP (mode)" value={postMode === null ? "—" : postMode.toFixed(3)} color={C.bayes} bg={C.bayesSoft} />
            </div>
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 12 }}>
              <Stat
                label={`${confPct}% credible interval`}
                value={`[${credCI[0].toFixed(3)}, ${credCI[1].toFixed(3)}]`}
                color={C.bayes}
                bg={C.bayesSoft}
              />
              <Stat label="Posterior" value={`Beta(${postA % 1 === 0 ? postA : postA.toFixed(1)}, ${postB % 1 === 0 ? postB : postB.toFixed(1)})`} color={C.bayes} bg={C.bayesSoft} />
            </div>
            <div style={{ marginBottom: 12 }}>
              <label htmlFor="prior" style={{ fontSize: 12, color: C.muted, display: "block", marginBottom: 6 }}>Prior over p</label>
              <select
                id="prior"
                value={priorKey}
                onChange={(e) => setPriorKey(e.target.value)}
                style={{
                  width: "100%",
                  padding: "8px 10px",
                  borderRadius: 8,
                  border: `1px solid ${C.border}`,
                  background: C.card,
                  color: C.ink,
                  fontSize: 13,
                  fontFamily: "inherit",
                }}
              >
                {Object.entries(PRIORS).map(([key, p]) => (
                  <option key={key} value={key}>{p.label}</option>
                ))}
              </select>
              <div style={{ fontSize: 11.5, color: C.muted, marginTop: 6 }}>{prior.note}</div>
            </div>
            <p style={{ fontSize: 12.5, color: C.ink, opacity: 0.82, lineHeight: 1.6, margin: 0 }}>
              Start with a Beta(&alpha;, &beta;) prior. Because the Beta is conjugate to the binomial likelihood,
              the update is exact and free: heads add to &alpha;, tails add to &beta;, giving the posterior
              Beta(&alpha;+{heads}, &beta;+{tails}). The {confPct}% credible interval holds {confPct}% of the
              posterior probability, so here you may say plainly: given this prior and this data, there is a
              {" "}{confPct}% probability that p lies inside it.
            </p>
          </Card>
        </div>

        <Card style={{ marginBottom: 18 }}>
          <Eyebrow color={C.freq}>What 95% confidence actually means</Eyebrow>
          <h2 style={{ fontSize: 19, fontWeight: 700, margin: "5px 0 8px" }}>Repeated sampling, the honest demonstration</h2>
          <p style={{ fontSize: 13, color: C.ink, opacity: 0.82, margin: "0 0 14px", lineHeight: 1.6, maxWidth: "66ch" }}>
            Fix the true p, then draw many fresh independent experiments of size n = {covN} and build a {confPct}%
            interval for each. The confidence statement is not about any single interval; it is about this stack.
            Roughly {confPct}% of the intervals should cover the true p (the green line). The ones in red miss it.
            That coverage is the entire content of the word confidence.
          </p>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 14, alignItems: "center" }}>
            <Btn primary onClick={() => runCoverage(200)} disabled={covRunning} ariaLabel="Run two hundred experiments">Run 200 experiments</Btn>
            <Btn onClick={() => runCoverage(2000)} disabled={covRunning} ariaLabel="Run two thousand experiments">Run 2000</Btn>
            <span style={{ fontSize: 12, color: C.muted }}>each draws n = {covN} flips at true p = {truP.toFixed(2)}</span>
          </div>

          {coverage ? (
            <>
              <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginBottom: 14 }}>
                <Stat
                  label={`${method === "wald" ? "Wald" : "Wilson"} empirical coverage`}
                  value={`${((method === "wald" ? coverage.empiricalWald : coverage.empirical) * 100).toFixed(1)}%`}
                  sub={`${coverage.experiments} experiments, nominal ${confPct}%`}
                  color={C.freq}
                  bg={C.freqSoft}
                />
                <Stat
                  label="Wald coverage"
                  value={`${(coverage.empiricalWald * 100).toFixed(1)}%`}
                  sub="normal approximation"
                  color={Math.abs(coverage.empiricalWald - conf) > 0.03 ? C.bad : C.ink}
                  bg={Math.abs(coverage.empiricalWald - conf) > 0.03 ? C.badSoft : C.faint}
                />
                <Stat
                  label="Wilson coverage"
                  value={`${(coverage.empirical * 100).toFixed(1)}%`}
                  sub="score interval"
                  color={C.good}
                  bg={C.goodSoft}
                />
              </div>
              <CoveragePlot intervals={coverage.intervals} truP={truP} />
              <p style={{ fontSize: 12, color: C.muted, marginTop: 10, lineHeight: 1.55 }}>
                Showing the first {Math.min(60, coverage.intervals.length)} of {coverage.intervals.length} intervals.
                Notice Wald typically lands under {confPct}% at small n while Wilson stays close. This is exactly
                why the choice of interval matters and why the textbook Wald formula is a trap for small samples.
              </p>
            </>
          ) : (
            <div style={{ padding: "26px 14px", textAlign: "center", color: C.muted, fontSize: 13, background: C.bg, borderRadius: 10, border: `1px dashed ${C.border}` }}>
              Run the simulation to watch coverage approach {confPct}%.
            </div>
          )}
        </Card>

        <Card style={{ marginBottom: 18 }}>
          <Eyebrow>Confidence level</Eyebrow>
          <div style={{ display: "flex", gap: 12, flexWrap: "wrap", alignItems: "center", marginTop: 10 }}>
            <input
              type="range"
              min={0.5}
              max={0.99}
              step={0.01}
              value={conf}
              onChange={(e) => setConf(+e.target.value)}
              aria-label="Confidence and credible level"
              style={{ flex: "1 1 220px", minWidth: 180 }}
            />
            <span style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 18, fontWeight: 700, color: C.accent, minWidth: 56 }}>
              {confPct}%
            </span>
            <span style={{ fontSize: 12.5, color: C.muted, flex: "1 1 200px" }}>
              Both intervals widen together as you raise the level. Wider intervals are the price of being more
              often right.
            </span>
          </div>
        </Card>

        <Card style={{ background: C.bg, border: `1px solid ${C.border}` }}>
          <Eyebrow color={C.accent}>The punchline: two answers, two meanings</Eyebrow>
          <div style={{ fontSize: 13, color: C.ink, lineHeight: 1.7, display: "grid", gap: 10, marginTop: 10 }}>
            <p style={{ margin: 0 }}>
              <b>They often nearly coincide.</b> With a flat Beta(1,1) prior and plenty of data, the Bayesian
              credible interval and the frequentist Wilson interval land almost on top of each other. The numbers
              agree because the likelihood dominates and the prior has washed out.
            </p>
            <p style={{ margin: 0 }}>
              <b>They diverge with small n or a strong prior.</b> Early on, or under the believes-fair Beta(20,20)
              prior, the posterior is pulled toward 0.5 and its interval separates from the frequentist one. Add
              flips and watch the prior's pull fade as the data takes over.
            </p>
            <p style={{ margin: 0 }}>
              <b>They never mean the same thing.</b> The credible interval is a statement about p given your data:
              {confPct}% of the posterior probability sits inside it. The confidence interval is a statement about
              the procedure: over endless repeated samples, {confPct}% of the intervals it builds would cover the
              fixed true p. You cannot say a single confidence interval has a {confPct}% chance of containing p,
              that probability is already either 0 or 1, you just do not know which.
            </p>
          </div>
        </Card>

        <div style={{ marginTop: 18, textAlign: "center", fontSize: 11, color: C.muted }}>
          Seeded mulberry32 draws · conjugate Beta-Binomial update · Wilson and Wald intervals · incomplete-beta quantiles
        </div>
      </div>
    </div>
  );
}

function LegendSwatch({ color, label, dash, faded }) {
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
      <span
        style={{
          width: 18,
          height: dash ? 0 : 3,
          borderTop: dash ? `2px dashed ${color}` : "none",
          background: dash ? "none" : color,
          opacity: faded ? 0.3 : 1,
          borderRadius: 2,
          display: "inline-block",
        }}
      />
      {label}
    </span>
  );
}
