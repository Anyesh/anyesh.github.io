import { useState, useEffect, useRef } from "react";
import { ScatterChart, Scatter, XAxis, YAxis, Tooltip, ResponsiveContainer, LineChart, Line, ReferenceLine, BarChart, Bar, Legend } from "recharts";

export const meta = {
  title: "House Prices with MCMC",
  category: "Bayesian Inference",
  description:
    "Forty synthetic Sydney houses, four unknown weights, and a Markov chain that explores the whole posterior at once. Watch it converge, then predict a price as a full distribution.",
  date: "2026-05-31",
  tags: ["mcmc", "bayesian", "regression", "uncertainty"],
};

// ── palette ──────────────────────────────────────────────────────────────────
const C = {
  bg: "#f7f5f2",
  card: "#ffffff",
  border: "#e4dfd8",
  text: "#1c1a17",
  muted: "#8c8278",
  accent: "#b85c2c",
  accentL: "#f5ece6",
  green: "#2e7d51",
  greenL: "#e4f2eb",
  blue: "#2a5298",
  blueL: "#e5ecf8",
  gold: "#9a7020",
  goldL: "#f5edd8",
};

// ── maths helpers ─────────────────────────────────────────────────────────────
function randn() {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function gaussianLogPDF(x, mu, sigma) {
  return -0.5 * Math.log(2 * Math.PI * sigma * sigma) - ((x - mu) ** 2) / (2 * sigma * sigma);
}

// ── synthetic Sydney data ─────────────────────────────────────────────────────
// price = 300000 + 7500*size + 40000*beds - 8000*dist + noise
const TRUE_THETA = { intercept: 300000, size: 7500, beds: 40000, dist: -8000 };
const SIGMA_NOISE = 60000;

function generateData(n = 40, seed = 42) {
  // seeded-ish deterministic
  const houses = [];
  for (let i = 0; i < n; i++) {
    const size = 60 + ((i * 137 + 11) % 80);           // 60-140 sqm
    const beds = 2 + ((i * 31 + 3) % 3);               // 2-4
    const dist = 3 + ((i * 53 + 7) % 22);              // 3-25 km from CBD
    const noise = randn() * SIGMA_NOISE;
    const price = TRUE_THETA.intercept + TRUE_THETA.size * size + TRUE_THETA.beds * beds + TRUE_THETA.dist * dist + noise;
    houses.push({ size, beds, dist, price: Math.round(price) });
  }
  return houses;
}

const DATA = generateData(40);

// ── log posterior (unnormalized) ──────────────────────────────────────────────
function logPosterior(theta, data) {
  // priors
  let lp = 0;
  lp += gaussianLogPDF(theta.intercept, 300000, 150000);
  lp += gaussianLogPDF(theta.size, 8000, 4000);
  lp += gaussianLogPDF(theta.beds, 35000, 20000);
  lp += gaussianLogPDF(theta.dist, -7000, 5000);

  // likelihood
  for (const h of data) {
    const pred = theta.intercept + theta.size * h.size + theta.beds * h.beds + theta.dist * h.dist;
    lp += gaussianLogPDF(h.price, pred, SIGMA_NOISE);
  }
  return lp;
}

// ── MCMC ──────────────────────────────────────────────────────────────────────
function runMCMC(steps = 3000) {
  const stepSizes = { intercept: 15000, size: 400, beds: 3000, dist: 400 };
  let current = { intercept: 300000, size: 8000, beds: 35000, dist: -7000 };
  let currentLP = logPosterior(current, DATA);
  const samples = [];
  let accepted = 0;

  for (let i = 0; i < steps; i++) {
    const proposal = {
      intercept: current.intercept + randn() * stepSizes.intercept,
      size: current.size + randn() * stepSizes.size,
      beds: current.beds + randn() * stepSizes.beds,
      dist: current.dist + randn() * stepSizes.dist,
    };
    const proposalLP = logPosterior(proposal, DATA);
    const logRatio = proposalLP - currentLP;

    if (Math.log(Math.random()) < logRatio) {
      current = proposal;
      currentLP = proposalLP;
      accepted++;
    }
    samples.push({ ...current, step: i });
  }
  return { samples, acceptRate: accepted / steps };
}

// ── UI helpers ────────────────────────────────────────────────────────────────
function Card({ children, style }) {
  return <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 14, padding: 20, ...style }}>{children}</div>;
}

function Tag({ color, bg, children }) {
  return <span style={{ background: bg, color, borderRadius: 4, padding: "2px 8px", fontSize: 11, fontWeight: 700, letterSpacing: "0.05em", textTransform: "uppercase" }}>{children}</span>;
}

function StatBox({ label, value, sub, color, bg }) {
  return (
    <div style={{ background: bg, borderRadius: 10, padding: "12px 16px", flex: 1, minWidth: 120 }}>
      <div style={{ fontSize: 11, color, fontWeight: 700, letterSpacing: "0.06em", textTransform: "uppercase", marginBottom: 4 }}>{label}</div>
      <div style={{ fontSize: 22, fontWeight: 700, color, lineHeight: 1 }}>{value}</div>
      {sub && <div style={{ fontSize: 11, color, opacity: 0.7, marginTop: 3 }}>{sub}</div>}
    </div>
  );
}

const BURN_IN = 500;
const PARAM_META = {
  size:      { label: "Size coeff ($/sqm)",   true: TRUE_THETA.size,      color: C.accent, fmt: v => `$${Math.round(v).toLocaleString()}` },
  beds:      { label: "Bedrooms coeff ($)",    true: TRUE_THETA.beds,      color: C.green,  fmt: v => `$${Math.round(v).toLocaleString()}` },
  dist:      { label: "Distance coeff ($/km)", true: TRUE_THETA.dist,      color: C.blue,   fmt: v => `$${Math.round(v).toLocaleString()}` },
  intercept: { label: "Base price ($)",        true: TRUE_THETA.intercept, color: C.gold,   fmt: v => `$${Math.round(v / 1000)}k` },
};

// ── main ──────────────────────────────────────────────────────────────────────
export default function App() {
  const [mcmc, setMcmc] = useState(null);
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [tab, setTab] = useState("data");
  const [param, setParam] = useState("size");
  const [predictSize, setPredictSize] = useState(90);
  const [predictBeds, setPredictBeds] = useState(3);
  const [predictDist, setPredictDist] = useState(10);
  const ref = useRef();

  useEffect(() => {
    const result = runMCMC(3000);
    setMcmc(result);
    setStep(0);
  }, []);

  useEffect(() => {
    if (playing && mcmc) {
      ref.current = setInterval(() => {
        setStep(s => {
          if (s >= mcmc.samples.length - 1) { setPlaying(false); return s; }
          return Math.min(s + 15, mcmc.samples.length - 1);
        });
      }, 40);
    } else clearInterval(ref.current);
    return () => clearInterval(ref.current);
  }, [playing, mcmc]);

  const burnedSamples = mcmc ? mcmc.samples.slice(BURN_IN, step + 1) : [];

  // posterior stats per param
  function posteriorStats(key) {
    if (burnedSamples.length < 10) return { mean: "—", lo: "—", hi: "—" };
    const vals = burnedSamples.map(s => s[key]);
    const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
    const sorted = [...vals].sort((a, b) => a - b);
    const lo = sorted[Math.floor(sorted.length * 0.05)];
    const hi = sorted[Math.floor(sorted.length * 0.95)];
    return { mean, lo, hi };
  }

  // histogram for selected param
  function buildHist(key, bins = 35) {
    if (burnedSamples.length < 20) return [];
    const vals = burnedSamples.map(s => s[key]);
    const min = Math.min(...vals), max = Math.max(...vals);
    const w = (max - min) / bins;
    const counts = new Array(bins).fill(0);
    vals.forEach(v => { const i = Math.min(Math.floor((v - min) / w), bins - 1); counts[i]++; });
    return counts.map((c, i) => ({ val: Math.round(min + (i + 0.5) * w), density: c / vals.length / w * 1000 }));
  }

  // chain trace
  const traceData = mcmc ? mcmc.samples.slice(0, step + 1).filter((_, i) => i % 3 === 0).map((s, i) => ({ i: i * 3, val: s[param] })) : [];

  // predictions
  function getPredictions() {
    if (burnedSamples.length < 10) return null;
    const preds = burnedSamples.map(s =>
      s.intercept + s.size * predictSize + s.beds * predictBeds + s.dist * predictDist
    );
    const sorted = [...preds].sort((a, b) => a - b);
    return {
      mean: preds.reduce((a, b) => a + b, 0) / preds.length,
      lo: sorted[Math.floor(sorted.length * 0.05)],
      hi: sorted[Math.floor(sorted.length * 0.95)],
      hist: (() => {
        const min = sorted[0], max = sorted[sorted.length - 1];
        const bins = 30, w = (max - min) / bins;
        const counts = new Array(bins).fill(0);
        sorted.forEach(v => { const i = Math.min(Math.floor((v - min) / w), bins - 1); counts[i]++; });
        return counts.map((c, i) => ({ price: Math.round((min + (i + 0.5) * w) / 1000), density: c / sorted.length }));
      })()
    };
  }

  const pred = getPredictions();
  const tabs = ["data", "mcmc", "posterior", "predict"];
  const tabLabels = { data: "The Data", mcmc: "MCMC Chain", posterior: "Posterior", predict: "Predict" };

  return (
    <div style={{ fontFamily: "'Palatino Linotype', 'Book Antiqua', Palatino, serif", background: C.bg, minHeight: "100vh", padding: "24px 14px", color: C.text }}>
      <div style={{ maxWidth: 860, margin: "0 auto" }}>

        {/* header */}
        <div style={{ marginBottom: 24 }}>
          <div style={{ fontSize: 10, letterSpacing: "0.15em", textTransform: "uppercase", color: C.muted, marginBottom: 5 }}>Bayesian Inference · Sydney Property</div>
          <h1 style={{ fontSize: 24, fontWeight: 700, margin: 0 }}>House Price Prediction with MCMC</h1>
          <p style={{ color: C.muted, fontSize: 13, margin: "5px 0 0", lineHeight: 1.5 }}>
            40 synthetic Sydney houses. Four unknown weights. MCMC explores the posterior over all of them simultaneously.
          </p>
        </div>

        {/* model equation */}
        <Card style={{ marginBottom: 18, background: C.accentL, border: `1px solid ${C.accent}22` }}>
          <div style={{ fontSize: 11, fontWeight: 700, letterSpacing: "0.08em", textTransform: "uppercase", color: C.accent, marginBottom: 8 }}>The Model</div>
          <div style={{ fontFamily: "monospace", fontSize: 13, lineHeight: 2, color: C.text }}>
            price = <b style={{ color: C.gold }}>θ₀</b> + <b style={{ color: C.accent }}>θ₁</b>×size + <b style={{ color: C.green }}>θ₂</b>×bedrooms + <b style={{ color: C.blue }}>θ₃</b>×distance + Gaussian noise
          </div>
          <div style={{ fontSize: 12, color: C.muted, marginTop: 4 }}>
            We don't know θ₀–θ₃. MCMC explores the 4D posterior surface to learn a distribution over each weight.
          </div>
        </Card>

        {/* tabs */}
        <div style={{ display: "flex", gap: 4, marginBottom: 16, flexWrap: "wrap" }}>
          {tabs.map(t => (
            <button key={t} onClick={() => setTab(t)}
              style={{ padding: "7px 15px", borderRadius: 8, border: `1.5px solid ${tab === t ? C.accent : C.border}`, background: tab === t ? C.accentL : "transparent", color: tab === t ? C.accent : C.muted, fontSize: 12, fontWeight: tab === t ? 700 : 400, cursor: "pointer", fontFamily: "inherit" }}>
              {tabLabels[t]}
            </button>
          ))}
        </div>

        {/* ── TAB: DATA ── */}
        {tab === "data" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
            <Card>
              <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 4 }}>Price vs Size (sqm)</div>
              <div style={{ fontSize: 12, color: C.muted, marginBottom: 12 }}>Each dot is a house. True coefficient: $7,500/sqm.</div>
              <ResponsiveContainer width="100%" height={200}>
                <ScatterChart margin={{ left: 10, right: 10 }}>
                  <XAxis dataKey="size" name="Size (sqm)" tick={{ fontSize: 11 }} label={{ value: "Size (sqm)", position: "insideBottom", offset: -2, fontSize: 11 }} />
                  <YAxis dataKey="price" name="Price" tickFormatter={v => `$${(v / 1000).toFixed(0)}k`} tick={{ fontSize: 11 }} />
                  <Tooltip formatter={(v, n) => n === "price" ? [`$${v.toLocaleString()}`, "Price"] : [v, "Size (sqm)"]} />
                  <Scatter data={DATA} fill={C.accent} opacity={0.7} />
                </ScatterChart>
              </ResponsiveContainer>
            </Card>
            <Card>
              <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 4 }}>Price vs Distance from CBD (km)</div>
              <div style={{ fontSize: 12, color: C.muted, marginBottom: 12 }}>Negative relationship. True coefficient: -$8,000/km.</div>
              <ResponsiveContainer width="100%" height={200}>
                <ScatterChart margin={{ left: 10, right: 10 }}>
                  <XAxis dataKey="dist" name="Distance (km)" tick={{ fontSize: 11 }} label={{ value: "Distance from CBD (km)", position: "insideBottom", offset: -2, fontSize: 11 }} />
                  <YAxis dataKey="price" name="Price" tickFormatter={v => `$${(v / 1000).toFixed(0)}k`} tick={{ fontSize: 11 }} />
                  <Tooltip formatter={(v, n) => n === "price" ? [`$${v.toLocaleString()}`, "Price"] : [v, "Distance (km)"]} />
                  <Scatter data={DATA} fill={C.blue} opacity={0.7} />
                </ScatterChart>
              </ResponsiveContainer>
            </Card>
            <Card>
              <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 10 }}>True weights (hidden from MCMC)</div>
              <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
                <StatBox label="Base price" value="$300k" color={C.gold} bg={C.goldL} />
                <StatBox label="Per sqm" value="$7,500" color={C.accent} bg={C.accentL} />
                <StatBox label="Per bedroom" value="$40,000" color={C.green} bg={C.greenL} />
                <StatBox label="Per km CBD" value="-$8,000" color={C.blue} bg={C.blueL} />
              </div>
            </Card>
          </div>
        )}

        {/* ── TAB: MCMC ── */}
        {tab === "mcmc" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
            <Card>
              <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 4 }}>Watch the chain explore</div>
              <div style={{ fontSize: 12, color: C.muted, marginBottom: 14 }}>
                MCMC walks through 4D parameter space simultaneously. Each step proposes a new (θ₀, θ₁, θ₂, θ₃) and accepts/rejects based on the ratio. Pick a parameter to watch.
              </div>
              <div style={{ display: "flex", gap: 6, marginBottom: 14, flexWrap: "wrap" }}>
                {Object.entries(PARAM_META).map(([k, m]) => (
                  <button key={k} onClick={() => setParam(k)}
                    style={{ padding: "5px 12px", borderRadius: 6, border: `1.5px solid ${param === k ? m.color : C.border}`, background: param === k ? m.color + "18" : "transparent", color: param === k ? m.color : C.muted, fontSize: 12, fontWeight: param === k ? 700 : 400, cursor: "pointer", fontFamily: "inherit" }}>
                    {k}
                  </button>
                ))}
              </div>
              <div style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 14, flexWrap: "wrap" }}>
                <button onClick={() => { setStep(0); setPlaying(false); }} style={{ padding: "6px 12px", borderRadius: 6, border: `1px solid ${C.border}`, background: "transparent", cursor: "pointer", fontSize: 12, fontFamily: "inherit" }}>Reset</button>
                <button onClick={() => setPlaying(p => !p)} style={{ padding: "6px 18px", borderRadius: 6, border: "none", background: C.accent, color: "#fff", cursor: "pointer", fontSize: 13, fontWeight: 600, fontFamily: "inherit" }}>
                  {playing ? "Pause" : step === 0 ? "Run Chain" : "Resume"}
                </button>
                <button onClick={() => setStep((mcmc?.samples.length ?? 1) - 1)} style={{ padding: "6px 12px", borderRadius: 6, border: `1px solid ${C.border}`, background: "transparent", cursor: "pointer", fontSize: 12, fontFamily: "inherit" }}>Skip to end</button>
                <div style={{ fontSize: 12, color: C.muted, marginLeft: "auto" }}>
                  Step <b>{step}</b> · Accept rate <b style={{ color: C.green }}>{mcmc ? (mcmc.acceptRate * 100).toFixed(1) : "—"}%</b>
                </div>
              </div>
              <div style={{ fontSize: 11, color: C.muted, marginBottom: 6 }}>Chain trace — {PARAM_META[param].label}</div>
              <ResponsiveContainer width="100%" height={160}>
                <LineChart data={traceData.slice(-600)} margin={{ left: -5, right: 10 }}>
                  <XAxis dataKey="i" tick={{ fontSize: 10 }} />
                  <YAxis tickFormatter={v => PARAM_META[param].fmt(v)} tick={{ fontSize: 9 }} width={70} />
                  <ReferenceLine y={PARAM_META[param].true} stroke={PARAM_META[param].color} strokeDasharray="4 2"
                    label={{ value: "True", fontSize: 9, fill: PARAM_META[param].color }} />
                  <Line type="monotone" dataKey="val" stroke={PARAM_META[param].color} dot={false} strokeWidth={1} isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
              <div style={{ fontSize: 11, color: C.muted, marginTop: 8 }}>
                The dashed line is the true value. Watch the chain converge toward it from the starting guess.
                First {BURN_IN} steps are burn-in — discarded before computing the posterior.
              </div>
            </Card>
          </div>
        )}

        {/* ── TAB: POSTERIOR ── */}
        {tab === "posterior" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
            <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
              {Object.entries(PARAM_META).map(([k, m]) => (
                <button key={k} onClick={() => setParam(k)}
                  style={{ padding: "5px 12px", borderRadius: 6, border: `1.5px solid ${param === k ? m.color : C.border}`, background: param === k ? m.color + "18" : "transparent", color: param === k ? m.color : C.muted, fontSize: 12, fontWeight: param === k ? 700 : 400, cursor: "pointer", fontFamily: "inherit" }}>
                  {k}
                </button>
              ))}
            </div>

            {(() => {
              const stats = posteriorStats(param);
              const hist = buildHist(param);
              const meta = PARAM_META[param];
              return (
                <Card>
                  <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 4 }}>Posterior distribution — {meta.label}</div>
                  <div style={{ fontSize: 12, color: C.muted, marginBottom: 14 }}>
                    This is what MCMC recovered. Width = uncertainty. Center = best estimate.
                  </div>
                  <div style={{ display: "flex", gap: 10, marginBottom: 16, flexWrap: "wrap" }}>
                    <StatBox label="Posterior mean" value={typeof stats.mean === "number" ? meta.fmt(stats.mean) : "—"} sub="MCMC estimate" color={meta.color} bg={meta.color + "15"} />
                    <StatBox label="90% interval" value={typeof stats.lo === "number" ? `${meta.fmt(stats.lo)} – ${meta.fmt(stats.hi)}` : "—"} sub="Range of plausible values" color={C.muted} bg={C.bg} />
                    <StatBox label="True value" value={meta.fmt(meta.true)} sub="Hidden ground truth" color={C.green} bg={C.greenL} />
                  </div>
                  {hist.length > 0 ? (
                    <ResponsiveContainer width="100%" height={180}>
                      <BarChart data={hist} margin={{ left: -5, right: 10 }}>
                        <XAxis dataKey="val" tickFormatter={v => meta.fmt(v)} tick={{ fontSize: 9 }} interval={Math.floor(hist.length / 5)} />
                        <YAxis tick={{ fontSize: 10 }} label={{ value: "Density", angle: -90, position: "insideLeft", fontSize: 10 }} />
                        <Bar dataKey="density" fill={meta.color} opacity={0.75} isAnimationActive={false} />
                        <ReferenceLine x={meta.true} stroke={C.green} strokeWidth={2} label={{ value: "True", fontSize: 9, fill: C.green }} />
                      </BarChart>
                    </ResponsiveContainer>
                  ) : (
                    <div style={{ height: 180, display: "flex", alignItems: "center", justifyContent: "center", color: C.muted, fontSize: 13 }}>
                      Run the MCMC chain first (go to MCMC Chain tab and run it)
                    </div>
                  )}
                  <div style={{ fontSize: 12, color: C.muted, marginTop: 10 }}>
                    Unlike MLE which gives one number, this distribution tells you <i>how confident</i> to be. Wider = more uncertain.
                    With 40 data points the posterior is reasonably tight but not perfectly certain.
                  </div>
                </Card>
              );
            })()}

            <Card>
              <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 10 }}>All parameters at a glance</div>
              <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                {Object.entries(PARAM_META).map(([k, m]) => {
                  const s = posteriorStats(k);
                  const hasData = typeof s.mean === "number";
                  return (
                    <div key={k} style={{ display: "flex", alignItems: "center", gap: 12, padding: "10px 14px", background: C.bg, borderRadius: 8 }}>
                      <Tag color={m.color} bg={m.color + "18"}>{k}</Tag>
                      <div style={{ flex: 1, fontSize: 12, color: C.muted }}>{m.label}</div>
                      <div style={{ fontSize: 13, fontWeight: 700, color: m.color, minWidth: 80, textAlign: "right" }}>{hasData ? m.fmt(s.mean) : "—"}</div>
                      <div style={{ fontSize: 11, color: C.muted, minWidth: 80, textAlign: "right" }}>true: {m.fmt(m.true)}</div>
                    </div>
                  );
                })}
              </div>
            </Card>
          </div>
        )}

        {/* ── TAB: PREDICT ── */}
        {tab === "predict" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
            <Card>
              <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 4 }}>Design your house</div>
              <div style={{ fontSize: 12, color: C.muted, marginBottom: 16 }}>
                Adjust the features. MCMC gives you a full price distribution, not just one number.
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 16, marginBottom: 16 }}>
                {[
                  { label: "Size (sqm)", val: predictSize, set: setPredictSize, min: 40, max: 200, format: v => `${v} sqm` },
                  { label: "Bedrooms", val: predictBeds, set: setPredictBeds, min: 1, max: 6, format: v => `${v} bed` },
                  { label: "Distance from CBD (km)", val: predictDist, set: setPredictDist, min: 1, max: 40, format: v => `${v} km` },
                ].map(({ label, val, set, min, max, format }) => (
                  <div key={label}>
                    <div style={{ fontSize: 12, color: C.muted, marginBottom: 6 }}>{label}</div>
                    <input type="range" min={min} max={max} value={val} onChange={e => set(+e.target.value)} style={{ width: "100%", accentColor: C.accent }} />
                    <div style={{ fontSize: 20, fontWeight: 700, textAlign: "center", marginTop: 4, color: C.accent }}>{format(val)}</div>
                  </div>
                ))}
              </div>
            </Card>

            <Card>
              <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 4 }}>Predicted price distribution</div>
              <div style={{ fontSize: 12, color: C.muted, marginBottom: 14 }}>
                Each posterior sample gives one predicted price. This is what Bayesian prediction looks like -- full uncertainty, not a point estimate.
              </div>
              {pred ? (
                <>
                  <div style={{ display: "flex", gap: 10, marginBottom: 16, flexWrap: "wrap" }}>
                    <StatBox label="Expected price" value={`$${Math.round(pred.mean / 1000)}k`} sub="Posterior mean prediction" color={C.accent} bg={C.accentL} />
                    <StatBox label="90% interval" value={`$${Math.round(pred.lo / 1000)}k – $${Math.round(pred.hi / 1000)}k`} sub="Plausible price range" color={C.blue} bg={C.blueL} />
                  </div>
                  <ResponsiveContainer width="100%" height={180}>
                    <BarChart data={pred.hist} margin={{ left: -5, right: 10 }}>
                      <XAxis dataKey="price" tickFormatter={v => `$${v}k`} tick={{ fontSize: 10 }} interval={4} />
                      <YAxis tick={{ fontSize: 10 }} />
                      <Bar dataKey="density" fill={C.accent} opacity={0.75} isAnimationActive={false} />
                      <ReferenceLine x={Math.round(pred.mean / 1000)} stroke={C.green} strokeWidth={2} label={{ value: "Mean", fontSize: 9, fill: C.green }} />
                    </BarChart>
                  </ResponsiveContainer>
                  <div style={{ fontSize: 12, color: C.muted, marginTop: 10 }}>
                    A standard regression would give you one number. Bayesian gives you this whole curve.
                    You can answer "what's the probability this house costs more than $1M?" by reading the tail.
                  </div>
                </>
              ) : (
                <div style={{ height: 200, display: "flex", alignItems: "center", justifyContent: "center", color: C.muted, fontSize: 13 }}>
                  Run the MCMC chain first (go to MCMC Chain tab)
                </div>
              )}
            </Card>
          </div>
        )}

        <div style={{ marginTop: 20, textAlign: "center", fontSize: 11, color: C.muted }}>
          Synthetic data · true weights are θ₀=$300k, θ₁=$7.5k/sqm, θ₂=$40k/bed, θ₃=-$8k/km
        </div>
      </div>
    </div>
  );
}
