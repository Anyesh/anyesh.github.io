import { useEffect, useRef, useState } from "react";

export const meta = {
  title: "Rotary Position Embedding: The Clock Hands of Attention",
  category: "Transformers",
  description:
    "Attention compares tokens with a dot product, a similarity score that has no idea which token came first. So how does a model tell \"dog bites man\" from \"man bites dog\"? RoPE's answer isn't to stamp a position number onto each vector, it's to spin the vector by an angle proportional to its position. Spin two clock hands to a fixed distance apart and watch their dot product hold dead still, split one hand into four speeds to see a single position encoded at every resolution at once, then push a sequence past the length a model trained on and watch the fix that keeps the angles in familiar territory.",
  date: "2026-08-26",
  tags: ["rope", "positional-encoding", "transformers", "attention", "context-length"],
};

const MONO = "'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, monospace";
const SERIF = "Georgia, 'Iowan Old Style', serif";
const EASE = "cubic-bezier(0.23, 1, 0.32, 1)";

const C = {
  bg: "#f7f5f2",
  card: "#ffffff",
  border: "#e4dfd8",
  ink: "#1c1a17",
  muted: "#857c72",
  faint: "#efeae3",
  accent: "#c0561f",
  accentSoft: "#f6ece5",
  blue: "#1e6fa0",
  blueSoft: "#e8f0f5",
  green: "#3f7d52",
  greenSoft: "#e7f0e9",
  danger: "#a8453f",
  dangerSoft: "#f7e8e6",
};

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

function rotate2D([x, y], angle) {
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  return [x * c - y * s, x * s + y * c];
}

function dot(a, b) {
  return a.reduce((sum, x, i) => sum + x * b[i], 0);
}

// theta_i = base^(-2i/d): the RoPE frequency schedule (Su et al. 2021). It is a
// geometric sequence, so low-index pairs spin fast (they resolve short distances)
// and high-index pairs spin slow (they resolve long ones), like a clock's second
// and hour hands sharing one face.
function ropeFrequencies(dim, base) {
  const half = dim / 2;
  return Array.from({ length: half }, (_, i) => Math.pow(base, (-2 * i) / dim));
}

function ropeRotateFull(vec, pos, freqs) {
  const out = new Array(vec.length);
  freqs.forEach((theta, i) => {
    const [x, y] = rotate2D([vec[2 * i], vec[2 * i + 1]], pos * theta);
    out[2 * i] = x;
    out[2 * i + 1] = y;
  });
  return out;
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

// A per-word hash rather than a shipped embedding table, so the toy sentence's
// query/key vectors are deterministic and reproducible without any real data file.
function hashSeed(str, salt) {
  let h = salt >>> 0;
  for (let i = 0; i < str.length; i++) h = (Math.imul(h, 31) + str.charCodeAt(i)) >>> 0;
  return h;
}

function seededVector(seed, dim) {
  const rng = mulberry32(seed);
  return Array.from({ length: dim }, () => rng() * 2 - 1);
}

const EMB_DIM = 8;
const BASE = 10000;
const FREQS = ropeFrequencies(EMB_DIM, BASE);
const CORE_THETA = Math.PI / 6; // 30 degrees: 12 slider steps sweep one full turn, like a clock face
const Q0 = [1.0, 0.5];
const K0 = [0.5, -0.9];
const SENTENCE = ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog", "runs", "past", "gate"];
const TOKEN_A_IDX = 3; // "fox"
const TOKEN_B_IDX = 1; // "quick"
const L_TRAIN = 128;
const SLOW_IDX = EMB_DIM / 2 - 1;

function qVecFor(word) {
  return seededVector(hashSeed(word, 1000), EMB_DIM);
}
function kVecFor(word) {
  return seededVector(hashSeed(word, 7000), EMB_DIM);
}

// Rescale the frequency base so the slowest pair's angle at the new max length L
// lands exactly where it topped out at L_train, while theta_0 = base^0 = 1 stays
// untouched. Fast pairs already wrap around many times during training, so
// extrapolating them as-is is safe; only the slow ones need interpolating. The
// exponent d/(d-2) is what makes that boundary condition exact for the last pair.
function ntkRescaledBase(base, dim, lTrain, lActual) {
  const scale = lActual / lTrain;
  return base * Math.pow(scale, dim / (dim - 2));
}

function fmt(x, places = 3) {
  const v = Math.abs(x) < 5e-9 ? 0 : x;
  return v.toFixed(places);
}

function degFmt(rad) {
  return `${fmt((rad * 180) / Math.PI, 1)}°`;
}

function clamp(x, lo, hi) {
  return Math.max(lo, Math.min(hi, x));
}

function Card({ children, style }) {
  return (
    <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 14, padding: 20, ...style }}>
      {children}
    </div>
  );
}

function SectionTitle({ children }) {
  return (
    <h2 style={{ fontSize: 19, fontWeight: 700, margin: "34px 0 8px", lineHeight: 1.25, textWrap: "balance" }}>
      {children}
    </h2>
  );
}

function Prose({ children, style }) {
  return (
    <p style={{ fontSize: 14, lineHeight: 1.65, color: C.ink, margin: "0 0 14px", maxWidth: "68ch", ...style }}>
      {children}
    </p>
  );
}

function Caption({ children, style }) {
  return <p style={{ fontSize: 12, lineHeight: 1.6, color: C.muted, margin: "10px 0 0", ...style }}>{children}</p>;
}

function ToggleButton({ active, onClick, children, ariaLabel }) {
  return (
    <button
      type="button"
      className="rope-press"
      onClick={onClick}
      aria-pressed={active}
      aria-label={ariaLabel}
      style={{
        padding: "8px 16px",
        borderRadius: 9,
        border: `1.5px solid ${active ? C.accent : C.border}`,
        background: active ? C.accentSoft : C.card,
        color: active ? C.accent : C.muted,
        fontWeight: 700,
        fontSize: 13,
        cursor: "pointer",
        fontFamily: "inherit",
        transition: `background 160ms ${EASE}, border-color 160ms ${EASE}, color 160ms ${EASE}`,
      }}
    >
      {children}
    </button>
  );
}

function Btn({ children, onClick, disabled, ariaLabel }) {
  return (
    <button
      type="button"
      className="rope-press"
      onClick={onClick}
      disabled={disabled}
      aria-label={ariaLabel}
      style={{
        padding: "7px 14px",
        borderRadius: 8,
        border: `1px solid ${C.border}`,
        background: "transparent",
        color: C.ink,
        fontSize: 13,
        fontWeight: 500,
        fontFamily: "inherit",
        cursor: disabled ? "not-allowed" : "pointer",
        opacity: disabled ? 0.4 : 1,
        transition: `transform 140ms ${EASE}, background 160ms ease`,
      }}
    >
      {children}
    </button>
  );
}

function SliderRow({ id, label, value, min, max, step = 1, onChange, format, hint, color = C.accent }) {
  return (
    <div style={{ marginBottom: 14 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
        <label htmlFor={id} style={{ fontSize: 13, color: C.muted, minWidth: 128 }}>
          {label}
        </label>
        <input
          id={id}
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => onChange(+e.target.value)}
          aria-valuetext={format ? format(value) : String(value)}
          style={{ flex: 1, minWidth: 120, accentColor: color }}
        />
        <span
          style={{
            fontFamily: MONO,
            fontSize: 13,
            fontWeight: 700,
            color,
            minWidth: 58,
            textAlign: "right",
            fontVariantNumeric: "tabular-nums",
          }}
        >
          {format ? format(value) : value}
        </span>
      </div>
      {hint && <div style={{ fontSize: 11.5, color: C.muted, marginTop: 3, marginLeft: 140 }}>{hint}</div>}
    </div>
  );
}

function FlashNumber({ value, format, color, reduce, size = 15 }) {
  const display = format(value);
  return (
    <span
      key={display}
      style={{
        display: "inline-block",
        color,
        fontFamily: MONO,
        fontWeight: 700,
        fontSize: size,
        fontVariantNumeric: "tabular-nums",
        animation: reduce ? "none" : "rope-flash 420ms ease-out",
      }}
    >
      {display}
    </span>
  );
}

function StatBox({ label, value, sub, color = C.ink, bg = C.faint }) {
  return (
    <div style={{ background: bg, borderRadius: 10, padding: "10px 14px", flex: "1 1 130px" }}>
      <div style={{ fontSize: 10.5, color, fontWeight: 700, letterSpacing: "0.05em", textTransform: "uppercase", marginBottom: 4 }}>
        {label}
      </div>
      <div style={{ fontSize: 18, fontWeight: 700, color, lineHeight: 1.15, fontFamily: MONO }}>{value}</div>
      {sub && <div style={{ fontSize: 10.5, color, opacity: 0.75, marginTop: 3 }}>{sub}</div>}
    </div>
  );
}

// Clock hands are drawn as SVG <g> elements rotated with the `transform` attribute
// rather than plain x/y coordinates, so a CSS transition on `transform` animates
// the sweep smoothly (attribute-only changes don't tween without it).
function Clock({ size = 200, hands = [], wedges = [], showTicks = false, ticks = 12, ariaLabel, reduce }) {
  const r = size / 2 - 18;
  const cx = size / 2;
  const cy = size / 2;
  const circumference = 2 * Math.PI * r;
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} role="img" aria-label={ariaLabel} style={{ display: "block", maxWidth: "100%", margin: "0 auto" }}>
      <circle cx={cx} cy={cy} r={r} fill="none" stroke={C.border} strokeWidth={2} />
      {wedges.map((w, i) => (
        <circle
          key={i}
          cx={cx}
          cy={cy}
          r={r}
          fill="none"
          stroke={w.color}
          strokeWidth={7}
          strokeDasharray={`${clamp(w.fraction, 0, 1) * circumference} ${circumference}`}
          strokeLinecap="butt"
          transform={`rotate(-90 ${cx} ${cy})`}
          opacity={0.55}
        />
      ))}
      {showTicks &&
        Array.from({ length: ticks }, (_, i) => {
          const a = (i / ticks) * 2 * Math.PI - Math.PI / 2;
          const x1 = cx + Math.cos(a) * (r - 6);
          const y1 = cy + Math.sin(a) * (r - 6);
          const x2 = cx + Math.cos(a) * r;
          const y2 = cy + Math.sin(a) * r;
          return <line key={i} x1={x1} y1={y1} x2={x2} y2={y2} stroke={C.faint} strokeWidth={2} />;
        })}
      <circle cx={cx} cy={cy} r={3.5} fill={C.ink} />
      {hands.map((h, i) => {
        const deg = (h.angle * 180) / Math.PI;
        const len = r * (h.length ?? 0.85);
        return (
          <g key={i} transform={`translate(${cx} ${cy}) rotate(${deg})`} style={{ transition: reduce ? "none" : `transform 360ms ${EASE}` }}>
            <line x1={0} y1={0} x2={0} y2={-len} stroke={h.color} strokeWidth={h.width ?? 3.5} strokeLinecap="round" />
            <circle cx={0} cy={-len} r={5} fill={h.color} />
          </g>
        );
      })}
    </svg>
  );
}

function vecToHand(vec, color, maxLen, width) {
  const mag = Math.hypot(vec[0], vec[1]);
  return { angle: Math.atan2(vec[1], vec[0]), length: clamp(mag / maxLen, 0.15, 1), color, width };
}

function AdditiveVsRotationDiagram() {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: 14 }}>
      <Card style={{ background: C.faint }}>
        <div style={{ fontSize: 12.5, fontWeight: 700, marginBottom: 10, color: C.muted, textTransform: "uppercase", letterSpacing: "0.04em" }}>
          The older fix: add a position vector
        </div>
        <svg width="100%" height="70" viewBox="0 0 220 70" role="img" aria-label="Content vector plus position vector equals a shifted vector">
          <rect x="6" y="20" width="60" height="30" rx="6" fill={C.blueSoft} stroke={C.blue} />
          <text x="36" y="39" textAnchor="middle" fontSize="10" fill={C.blue} fontFamily={MONO}>content</text>
          <text x="80" y="39" textAnchor="middle" fontSize="18" fill={C.muted}>+</text>
          <rect x="96" y="20" width="60" height="30" rx="6" fill={C.accentSoft} stroke={C.accent} />
          <text x="126" y="39" textAnchor="middle" fontSize="10" fill={C.accent} fontFamily={MONO}>position</text>
          <text x="170" y="39" textAnchor="middle" fontSize="18" fill={C.muted}>=</text>
          <rect x="186" y="10" width="30" height="50" rx="6" fill="none" stroke={C.ink} strokeDasharray="3 2" />
        </svg>
        <div style={{ fontSize: 12, color: C.muted, marginTop: 8, lineHeight: 1.55 }}>
          Bolts a position number onto the meaning. Works, but the combined vector now depends on the absolute
          index, not on how far apart two tokens are.
        </div>
      </Card>
      <Card style={{ background: C.faint }}>
        <div style={{ fontSize: 12.5, fontWeight: 700, marginBottom: 10, color: C.muted, textTransform: "uppercase", letterSpacing: "0.04em" }}>
          RoPE: spin the vector instead
        </div>
        <svg width="100%" height="70" viewBox="0 0 220 70" role="img" aria-label="Content vector rotated by an angle proportional to its position">
          <circle cx="45" cy="35" r="24" fill="none" stroke={C.border} />
          <line x1="45" y1="35" x2="45" y2="14" stroke={C.faint} strokeWidth="2.5" />
          <line x1="45" y1="35" x2="63" y2="20" stroke={C.accent} strokeWidth="3" strokeLinecap="round" />
          <path d="M 45 22 A 13 13 0 0 1 56 26" fill="none" stroke={C.muted} strokeWidth="1.5" markerEnd="url(#rope-arrow)" />
          <defs>
            <marker id="rope-arrow" markerWidth="6" markerHeight="6" refX="3" refY="3" orient="auto">
              <path d="M0,0 L6,3 L0,6 Z" fill={C.muted} />
            </marker>
          </defs>
          <text x="110" y="39" textAnchor="middle" fontSize="18" fill={C.muted}>→</text>
          <rect x="140" y="11" width="30" height="30" rx="15" fill="none" stroke={C.ink} strokeDasharray="3 2" transform="rotate(35 155 26)" />
          <circle cx="155" cy="26" r="14" fill="none" stroke={C.ink} strokeDasharray="3 2" />
        </svg>
        <div style={{ fontSize: 12, color: C.muted, marginTop: 8, lineHeight: 1.55 }}>
          Turns the vector by an angle proportional to position. Nothing is added, so two rotated vectors can
          only ever disagree by the angle between their positions, not by where either one sits.
        </div>
      </Card>
    </div>
  );
}

export default function App() {
  const reduce = usePrefersReducedMotion();

  const [pos1, setPos1] = useState(3);

  const [m, setM] = useState(7);
  const [n, setN] = useState(4);

  const [pos4, setPos4] = useState(20);
  const [playing4, setPlaying4] = useState(false);
  const raf4 = useRef(0);
  const last4 = useRef(0);

  const [shift5, setShift5] = useState(0);
  const [breakOffset5, setBreakOffset5] = useState(0);

  const [seqLen, setSeqLen] = useState(384);
  const [ntk, setNtk] = useState(false);

  useEffect(() => {
    if (!playing4 || reduce) {
      cancelAnimationFrame(raf4.current);
      last4.current = 0;
      return;
    }
    const loop = (ts) => {
      if (!last4.current) last4.current = ts;
      const dt = (ts - last4.current) / 1000;
      last4.current = ts;
      setPos4((p) => {
        const next = p + dt * 6;
        return next > 64 ? 0 : next;
      });
      raf4.current = requestAnimationFrame(loop);
    };
    raf4.current = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf4.current);
  }, [playing4, reduce]);

  const focusCss = `
    .rope-root :focus-visible { outline: 2px solid ${C.accent}; outline-offset: 2px; border-radius: 6px; }
    .rope-root input[type=range] { accent-color: ${C.accent}; }
    .rope-press { transition: transform 140ms ${EASE}; }
    .rope-press:active { transform: scale(0.96); }
    @keyframes rope-flash {
      from { opacity: 0.3; transform: translateY(2px) scale(0.96); }
      to { opacity: 1; transform: translateY(0) scale(1); }
    }
    @media (prefers-reduced-motion: reduce) {
      .rope-root * { animation-duration: 0.001ms !important; }
    }
  `;

  // ---- Section 3: single-vector rotation demo ----
  const vRot1 = rotate2D(Q0, pos1 * CORE_THETA);
  const angle1 = pos1 * CORE_THETA;

  // ---- Section 4: core 2D invariance demo ----
  const qRot = rotate2D(Q0, m * CORE_THETA);
  const kRot = rotate2D(K0, n * CORE_THETA);
  const coreDot = dot(qRot, kRot);
  const coreMaxLen = 1.6;
  const shiftBoth = (delta) => {
    setM((v) => clamp(v + delta, 0, 12));
    setN((v) => clamp(v + delta, 0, 12));
  };

  // ---- Section 5: multi-frequency demo ----
  const freqHands = FREQS.map((theta, i) => {
    const v = rotate2D([1, 0], pos4 * theta);
    const palette = [C.accent, C.blue, C.green, "#8a5a00"];
    return { ...vecToHand(v, palette[i % palette.length], 1), width: 4.5 - i * 0.7 };
  });

  // ---- Section 6: real toy attention scores ----
  const wordA = SENTENCE[TOKEN_A_IDX];
  const wordB = SENTENCE[TOKEN_B_IDX];
  const qVec = qVecFor(wordA);
  const kVec = kVecFor(wordB);
  const posA = TOKEN_A_IDX + shift5;
  const posB = TOKEN_B_IDX + shift5 + breakOffset5;
  const scaleRoot = Math.sqrt(EMB_DIM);
  const scoreRaw = dot(qVec, kVec) / scaleRoot;
  const scoreRope = dot(ropeRotateFull(qVec, posA, FREQS), ropeRotateFull(kVec, posB, FREQS)) / scaleRoot;
  const distance5 = posA - posB;

  // ---- Section 7: context extension ----
  const thetaSlowNaive = FREQS[SLOW_IDX];
  const trainedMaxAngle = L_TRAIN * thetaSlowNaive;
  const naiveAngleAtL = seqLen * thetaSlowNaive;
  const newBase = ntkRescaledBase(BASE, EMB_DIM, L_TRAIN, seqLen);
  const rescaledFreqs = ropeFrequencies(EMB_DIM, newBase);
  const rescaledAngleAtL = seqLen * rescaledFreqs[SLOW_IDX];
  const activeAngleSlow = ntk ? rescaledAngleAtL : naiveAngleAtL;
  const slowInRange = activeAngleSlow <= trainedMaxAngle + 1e-9;
  const fastWedgeFraction = Math.min((L_TRAIN * FREQS[0]) / (2 * Math.PI), 1);
  const slowWedgeFraction = Math.min(trainedMaxAngle / (2 * Math.PI), 1);
  const slowHandVec = rotate2D([1, 0], activeAngleSlow);
  const fastHandVec = rotate2D([1, 0], seqLen * FREQS[0]);

  return (
    <div className="rope-root" style={{ fontFamily: SERIF, background: C.bg, minHeight: "100vh", padding: "26px 14px 56px", color: C.ink }}>
      <style>{focusCss}</style>
      <div style={{ maxWidth: 720, margin: "0 auto" }}>
        <div style={{ fontSize: 10, letterSpacing: "0.14em", textTransform: "uppercase", color: C.muted, marginBottom: 6, fontFamily: MONO }}>
          Transformers · Positional Encoding
        </div>
        <h1 style={{ fontSize: 25, fontWeight: 700, margin: 0, lineHeight: 1.2, textWrap: "balance" }}>
          Rotary Position Embedding
        </h1>

        <Prose style={{ marginTop: 14, fontSize: 15 }}>
          Feed a transformer &ldquo;dog bites man&rdquo; and &ldquo;man bites dog&rdquo; and, on their own,
          attention can&apos;t tell them apart: it scores every pair of tokens with a dot product (a similarity
          score, the sum of each vector&apos;s matching components multiplied together), and a dot product only
          measures how alike two vectors are, never which one came first. So the position of every token has to
          be baked into its vector somehow before that comparison happens.
        </Prose>
        <Prose style={{ fontSize: 15 }}>
          RoPE&apos;s move is to rotate each vector by an angle proportional to its position, rather than
          tagging a position number onto it. Below, drag two clock hands to a fixed distance apart and watch the
          dot product between them hold still no matter where on the clock they stand, then break the match and
          watch it move.
        </Prose>

        <SectionTitle>Why not just add a position number?</SectionTitle>
        <Prose>
          The obvious fix is the older one: compute a position vector for slot 0, 1, 2, &hellip; and add it to
          each token&apos;s embedding (its content vector) before attention ever runs. It works, and it is what
          the original Transformer paper did. But now the vector fed into the dot product carries the token&apos;s
          absolute index baked in&mdash;shift a whole sentence later in a document and every vector changes, even
          though nothing about the sentence itself did.
        </Prose>
        <AdditiveVsRotationDiagram />

        <SectionTitle>Start with one vector</SectionTitle>
        <Prose>
          Forget queries and keys for a second. Take a single toy vector, <code>[1, 0.5]</code>, plotted as a
          point on the circle below&mdash;the same vector this page calls the query in the next section. RoPE
          encodes its position by rotating that point: <code>position &times; &theta;</code>, a small fixed
          angle repeated once per step. The vector&apos;s length never changes, only its direction, and a
          larger position just means more turns around the clock face.
        </Prose>
        <Card style={{ marginBottom: 8 }}>
          <div style={{ display: "flex", gap: 24, flexWrap: "wrap", alignItems: "center", justifyContent: "center" }}>
            <Clock
              size={190}
              showTicks
              ticks={12}
              reduce={reduce}
              ariaLabel={`Vector at position ${pos1}, rotated ${degFmt(angle1)}`}
              hands={[vecToHand(vRot1, C.blue, 1.6, 4)]}
            />
            <div style={{ minWidth: 200 }}>
              <SliderRow id="rope-pos1" label="Position" value={pos1} min={0} max={12} onChange={setPos1} color={C.blue} />
            </div>
          </div>
          <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginTop: 18 }}>
            <StatBox label="Angle turned" value={degFmt(angle1)} bg={C.faint} />
            <StatBox
              label="Rotated vector"
              value={<FlashNumber value={vRot1} format={(v) => `[${fmt(v[0])}, ${fmt(v[1])}]`} color={C.blue} reduce={reduce} size={16} />}
              bg={C.blueSoft}
              color={C.blue}
            />
          </div>
          <Caption>
            Rotation step here is a friendly 30° so 12 slider clicks make one full turn, matching the clock
            below. Real RoPE uses a much smaller angle per position (see &ldquo;One angle isn&apos;t
            enough&rdquo; further down); the mechanism&mdash;turn the vector by position times a fixed
            angle&mdash;is identical either way.
          </Caption>
        </Card>

        <SectionTitle>The core trick: rotate, don&apos;t add</SectionTitle>
        <Prose>
          Take one query vector (what a token is looking for) and one key vector (what a token offers). Rotate
          the query by its position <code>m</code> times a fixed angle <code>&theta;</code>, and the key by its
          position <code>n</code> times the same angle. Because a rotation is a rigid turn that preserves length
          and angle, rotating both vectors and then comparing them is mathematically identical to leaving the
          query alone and rotating the key by <code>(n&minus;m)&theta;</code>&mdash;the dot product literally
          cannot see <code>m</code> or <code>n</code> individually, only their difference.
        </Prose>

        <Card style={{ marginBottom: 8 }}>
          <div style={{ display: "flex", gap: 24, flexWrap: "wrap", alignItems: "center", justifyContent: "center" }}>
            <Clock
              size={210}
              showTicks
              ticks={12}
              reduce={reduce}
              ariaLabel={`Query hand at position ${m}, key hand at position ${n}, on a 12-step clock face`}
              hands={[vecToHand(qRot, C.blue, coreMaxLen, 4), vecToHand(kRot, C.green, coreMaxLen, 4)]}
            />
            <div style={{ minWidth: 200 }}>
              <SliderRow
                id="rope-m"
                label={<span>Query pos <span style={{ color: C.blue, fontWeight: 700 }}>m</span></span>}
                value={m}
                min={0}
                max={12}
                onChange={setM}
                color={C.blue}
              />
              <SliderRow
                id="rope-n"
                label={<span>Key pos <span style={{ color: C.green, fontWeight: 700 }}>n</span></span>}
                value={n}
                min={0}
                max={12}
                onChange={setN}
                color={C.green}
              />
              <div style={{ display: "flex", gap: 8, marginTop: 10 }}>
                <Btn onClick={() => shiftBoth(-1)} disabled={m <= 0 || n <= 0} ariaLabel="Shift both positions back by one, keeping their distance fixed">
                  Shift both −1
                </Btn>
                <Btn onClick={() => shiftBoth(1)} disabled={m >= 12 || n >= 12} ariaLabel="Shift both positions forward by one, keeping their distance fixed">
                  Shift both +1
                </Btn>
              </div>
            </div>
          </div>
          <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginTop: 18 }}>
            <StatBox
              label="Distance (m − n)"
              value={<FlashNumber value={m - n} format={(v) => String(v)} color={C.ink} reduce={reduce} size={18} />}
              bg={C.faint}
            />
            <StatBox
              label="Dot product"
              value={<FlashNumber value={coreDot} format={(v) => fmt(v, 4)} color={C.accent} reduce={reduce} size={18} />}
              bg={C.accentSoft}
              color={C.accent}
              sub="rotate(q,mθ) · rotate(k,nθ)"
            />
          </div>
          <Caption>
            Rotation step here is a friendly 30° so 12 slider clicks make one full turn, like an ordinary clock.
            Real RoPE uses a much smaller angle per position (see below); the identity being demonstrated is the
            same either way. Click &ldquo;shift both&rdquo; repeatedly: the distance box never changes, and
            neither does the dot product. Drag <code>m</code> or <code>n</code> alone and both move.
          </Caption>
        </Card>

        <SectionTitle>One angle isn&apos;t enough</SectionTitle>
        <Prose>
          A single rotation angle can only tell two tokens apart up to how many degrees fit in a circle before
          it repeats. Real RoPE splits each embedding into pairs of numbers (dimensions), and spins every pair
          at its own frequency: <code>&theta;<sub>i</sub> = base<sup>&minus;2i/d</sup></code>, a geometric
          sequence, so pair 0 spins fast and later pairs spin ever slower. It&apos;s a clock with several hands
          sharing one face: the fast hand distinguishes tokens one or two apart, the slow ones distinguish
          tokens dozens or hundreds apart, and together they cover both scales from a single position number.
        </Prose>
        <Card>
          <div style={{ display: "flex", gap: 24, flexWrap: "wrap", alignItems: "center", justifyContent: "center" }}>
            <Clock
              size={210}
              reduce={reduce}
              ariaLabel={`Four dimension pairs at position ${pos4.toFixed(1)}, spinning at different frequencies`}
              hands={freqHands}
            />
            <div style={{ minWidth: 200 }}>
              <SliderRow
                id="rope-pos4"
                label="Position"
                value={Math.round(pos4)}
                min={0}
                max={64}
                onChange={(v) => {
                  setPlaying4(false);
                  setPos4(v);
                }}
                format={(v) => String(v)}
              />
              {!reduce && (
                <ToggleButton active={playing4} onClick={() => setPlaying4((p) => !p)} ariaLabel={playing4 ? "Pause the animation" : "Play the animation"}>
                  {playing4 ? "Pause" : "Play"}
                </ToggleButton>
              )}
              {reduce && <Caption style={{ margin: 0 }}>Motion reduced: drag the slider to advance position.</Caption>}
            </div>
          </div>
          <div style={{ overflowX: "auto", marginTop: 16 }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12, minWidth: 420 }}>
              <thead>
                <tr style={{ color: C.muted, textAlign: "left" }}>
                  <th style={{ padding: "4px 8px", fontWeight: 600 }}>Pair i</th>
                  <th style={{ padding: "4px 8px", fontWeight: 600 }}>θᵢ (rad/step)</th>
                  <th style={{ padding: "4px 8px", fontWeight: 600 }}>Degrees/step</th>
                  <th style={{ padding: "4px 8px", fontWeight: 600 }}>Full turn every</th>
                </tr>
              </thead>
              <tbody>
                {FREQS.map((theta, i) => (
                  <tr key={i} style={{ borderTop: `1px solid ${C.faint}` }}>
                    <td style={{ padding: "4px 8px", fontFamily: MONO, color: freqHands[i].color, fontWeight: 700 }}>{i}</td>
                    <td style={{ padding: "4px 8px", fontFamily: MONO }}>{fmt(theta, 4)}</td>
                    <td style={{ padding: "4px 8px", fontFamily: MONO }}>{degFmt(theta)}</td>
                    <td style={{ padding: "4px 8px", fontFamily: MONO }}>{fmt((2 * Math.PI) / theta, 1)} steps</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <Caption>
            Base 10000 and 4 pairs (an 8-number embedding slice), matching the formula real models use. The
            slowest hand here barely creeps across 64 steps&mdash;that&apos;s deliberate: it&apos;s the one
            reserved for telling apart tokens hundreds of steps apart.
          </Caption>
        </Card>

        <SectionTitle>The same invariance, on real query/key vectors</SectionTitle>
        <Prose>
          The single clock hand above was a toy 2D vector. Real attention runs on 8, 64, or 128 numbers per
          token, rotated in pairs. Here are two fixed toy words, &ldquo;{wordB}&rdquo; (key) and &ldquo;{wordA}
          &rdquo; (query), with deterministic seeded vectors standing in for learned embeddings. Slide both of
          their positions together through a longer stretch of text and the RoPE-adjusted score won&apos;t
          budge; a raw dot product with no positional information at all never moved to begin with&mdash;proof
          that without something like RoPE, attention genuinely cannot see order.
        </Prose>
        <Card>
          <SliderRow
            id="rope-shift5"
            label="Shift both positions"
            value={shift5}
            min={0}
            max={40}
            onChange={setShift5}
            hint={`"${wordB}" sits at position ${TOKEN_B_IDX + shift5}, "${wordA}" at position ${posA}, as if this pair appeared later in a longer document.`}
          />
          <SliderRow
            id="rope-break5"
            label="Break the match"
            value={breakOffset5}
            min={-4}
            max={4}
            onChange={setBreakOffset5}
            color={C.danger}
            hint="Nudges only the key's position, changing the relative distance between the pair."
          />
          <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginTop: 4 }}>
            <StatBox label="Relative distance" value={<FlashNumber value={distance5} format={(v) => String(v)} color={C.ink} reduce={reduce} size={17} />} bg={C.faint} />
            <StatBox
              label="Score, no position info"
              value={<FlashNumber value={scoreRaw} format={(v) => fmt(v, 4)} color={C.muted} reduce={reduce} size={17} />}
              bg={C.faint}
              color={C.muted}
              sub="never changes"
            />
            <StatBox
              label="Score with RoPE"
              value={<FlashNumber value={scoreRope} format={(v) => fmt(v, 4)} color={C.accent} reduce={reduce} size={17} />}
              bg={C.accentSoft}
              color={C.accent}
              sub="tracks the distance"
            />
          </div>
          <Caption>
            The raw score is identical at every shift value because it never looks at position at all&mdash;the
            problem the intro opened with. The RoPE score is exactly the same at every shift too, as long as
            &ldquo;break the match&rdquo; stays at zero; move it and the score changes, because now the relative
            distance changed.
          </Caption>
        </Card>

        <SectionTitle>Past the trained length</SectionTitle>
        <Prose>
          Say a model only ever saw sequences up to <strong>{L_TRAIN}</strong> tokens during training. Every
          dimension pair only ever swept through the angle range that fits inside those {L_TRAIN} steps. The
          fast pairs wrap around the circle many times in that span, so any further rotation just lands on
          angles the model has already seen&mdash;wrapping is harmless for them. The slow pairs are the risk:
          over only {L_TRAIN} steps they barely move past a sliver of the circle, so pushing the sequence longer
          walks them into angles that are numerically valid but were never part of training data.
        </Prose>
        <Card>
          <div style={{ display: "flex", gap: 20, flexWrap: "wrap", alignItems: "flex-start", justifyContent: "center" }}>
            <div style={{ textAlign: "center" }}>
              <Clock
                size={150}
                reduce={reduce}
                wedges={[{ fraction: fastWedgeFraction, color: C.green }]}
                ariaLabel="Fastest dimension pair, always within the training-seen range because it wraps many times"
                hands={[vecToHand(fastHandVec, C.green, 1)]}
              />
              <div style={{ fontSize: 11.5, color: C.muted, marginTop: 6 }}>
                Fastest pair (θ₀) &mdash; always familiar
              </div>
            </div>
            <div style={{ textAlign: "center" }}>
              <Clock
                size={150}
                reduce={reduce}
                wedges={[{ fraction: slowWedgeFraction, color: slowInRange ? C.green : C.danger }]}
                ariaLabel={`Slowest dimension pair, ${slowInRange ? "within" : "outside"} the training-seen range`}
                hands={[vecToHand(slowHandVec, slowInRange ? C.green : C.danger, 1)]}
              />
              <div style={{ fontSize: 11.5, color: C.muted, marginTop: 6 }}>
                Slowest pair (θ<sub>{SLOW_IDX}</sub>) &mdash; {slowInRange ? "still familiar" : "unseen territory"}
              </div>
            </div>
          </div>

          <SliderRow
            id="rope-seqlen"
            label="Actual sequence length"
            value={seqLen}
            min={L_TRAIN}
            max={1024}
            step={16}
            onChange={setSeqLen}
            hint={`Trained on sequences up to ${L_TRAIN} tokens; shaded arc is the angle range seen during training.`}
          />
          <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap", marginBottom: 12 }}>
            <ToggleButton active={ntk} onClick={() => setNtk((v) => !v)} ariaLabel="Toggle NTK-aware frequency rescaling">
              {ntk ? "NTK-aware rescaling: on" : "NTK-aware rescaling: off"}
            </ToggleButton>
            <span style={{ fontSize: 12.5, color: slowInRange ? C.green : C.danger, fontWeight: 700 }}>
              {slowInRange ? "Slow pair's angle is within trained range" : "Slow pair's angle exceeds anything seen in training"}
            </span>
          </div>

          <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
            <StatBox label="Trained max angle (slow pair)" value={fmt(trainedMaxAngle, 4)} sub="radians, at position 0..L_train" bg={C.faint} />
            <StatBox
              label="Angle at extended length"
              value={<FlashNumber value={activeAngleSlow} format={(v) => fmt(v, 4)} color={slowInRange ? C.green : C.danger} reduce={reduce} size={17} />}
              bg={slowInRange ? C.greenSoft : C.dangerSoft}
              color={slowInRange ? C.green : C.danger}
            />
            <StatBox label="Frequency base in use" value={fmt(ntk ? newBase : BASE, 0)} sub={ntk ? "rescaled" : "original 10000"} bg={C.faint} />
          </div>
          <Caption>
            Turning rescaling on changes the frequency base to <code>base &middot; (L/L_train)^(d/(d&minus;2))</code>,
            which stretches the slow pairs back down so their angle at the new length matches what they topped
            out at during training, while leaving the fastest pair (θ₀ = base⁰ = 1, unaffected by any base) exactly
            as it was. This is the idea behind &ldquo;NTK-aware&rdquo; RoPE scaling; YaRN refines it further by
            blending old and new frequencies per dimension instead of one global rescale.
          </Caption>
        </Card>

        <footer style={{ marginTop: 30, paddingTop: 16, borderTop: `1px solid ${C.border}` }}>
          <p style={{ fontSize: 12.5, color: C.muted, lineHeight: 1.7, margin: 0, maxWidth: "68ch" }}>
            Every rotation, dot product, and rescaled frequency on this page is computed live from the formulas
            above&mdash;real 2×2 rotation matrices and real <code>θᵢ = base<sup>&minus;2i/d</sup></code> values,
            not stand-in numbers. The only choices made for you are round ones: a base of 10000 and an 8-number
            toy embedding, and a seeded random generator for the toy sentence so its vectors are stable across
            renders without shipping a real embedding table.
          </p>
        </footer>
      </div>
    </div>
  );
}
