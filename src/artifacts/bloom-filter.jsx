import { useState, useMemo, useEffect, useRef, useCallback } from "react";
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine } from "recharts";

export const meta = {
  title: "Bloom Filters",
  category: "Data Structures",
  description:
    "A bloom filter can say a key is present when it never was. Insert keys, watch their bits light up across k hash probes, and catch the false positive the moment it happens.",
  date: "2026-03-05",
  tags: ["bloom-filter", "probabilistic", "hashing", "data-structures"],
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
  set: "#c0561f",
  probe: "#2e6f8e",
  good: "#2f7d53",
  bad: "#a8453f",
};

const EASE = "cubic-bezier(0.23, 1, 0.32, 1)";
const MAX_RENDERED_CELLS = 320;

function fnv1a(str, seed) {
  let h = (0x811c9dc5 ^ seed) >>> 0;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

function hashIndices(key, k, m) {
  const h1 = fnv1a(key, 0x9e3779b1);
  const h2 = (fnv1a(key, 0x85ebca77) | 1) >>> 0;
  const out = [];
  for (let i = 0; i < k; i++) {
    out.push((((h1 + Math.imul(i, h2)) >>> 0) % m));
  }
  return out;
}

function theoreticalFP(k, n, m) {
  if (m === 0 || n === 0) return 0;
  return Math.pow(1 - Math.exp((-k * n) / m), k);
}

function optimalK(m, n) {
  if (n === 0) return 1;
  return Math.max(1, Math.round((m / n) * Math.LN2));
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

function Btn({ children, onClick, disabled, primary, ariaLabel, type }) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      type={type || "button"}
      aria-label={ariaLabel}
      style={{
        padding: primary ? "8px 18px" : "7px 14px",
        borderRadius: 8,
        border: primary ? "none" : `1px solid ${C.border}`,
        background: primary ? C.accent : "transparent",
        color: primary ? "#fff" : C.ink,
        fontSize: 13,
        fontWeight: primary ? 600 : 500,
        fontFamily: "inherit",
        cursor: disabled ? "not-allowed" : "pointer",
        opacity: disabled ? 0.4 : 1,
        transition: `transform 140ms ${EASE}, background 160ms ease`,
      }}
      onPointerDown={(e) => {
        if (!disabled) e.currentTarget.style.transform = "scale(0.97)";
      }}
      onPointerUp={(e) => {
        e.currentTarget.style.transform = "scale(1)";
      }}
      onPointerLeave={(e) => {
        e.currentTarget.style.transform = "scale(1)";
      }}
    >
      {children}
    </button>
  );
}

function Slider({ label, value, min, max, onChange, id, suffix }) {
  return (
    <div>
      <label
        htmlFor={id}
        style={{
          display: "flex",
          justifyContent: "space-between",
          fontSize: 12,
          color: C.muted,
          marginBottom: 6,
        }}
      >
        <span>{label}</span>
        <span style={{ fontWeight: 700, color: C.accent }}>
          {value}
          {suffix}
        </span>
      </label>
      <input
        id={id}
        type="range"
        min={min}
        max={max}
        value={value}
        onChange={(e) => onChange(+e.target.value)}
        style={{ width: "100%", accentColor: C.accent }}
      />
    </div>
  );
}

function Stat({ label, value, sub, color, bg }) {
  return (
    <div style={{ background: bg, borderRadius: 10, padding: "12px 14px", flex: 1, minWidth: 120 }}>
      <div
        style={{
          fontSize: 10.5,
          color,
          fontWeight: 700,
          letterSpacing: "0.06em",
          textTransform: "uppercase",
          marginBottom: 4,
        }}
      >
        {label}
      </div>
      <div style={{ fontSize: 22, fontWeight: 700, color, lineHeight: 1.05 }}>{value}</div>
      {sub && <div style={{ fontSize: 11, color: C.muted, marginTop: 4 }}>{sub}</div>}
    </div>
  );
}

function BitGrid({ bits, m, probed, lastInserted, reduce }) {
  const rendered = Math.min(m, MAX_RENDERED_CELLS);
  const cell = m > 200 ? 12 : m > 96 ? 16 : 22;
  const gap = m > 200 ? 2 : 3;

  return (
    <div>
      <div
        role="img"
        aria-label={`Bit array of ${m} cells, ${bits.reduce((a, b) => a + b, 0)} set to one`}
        style={{
          display: "grid",
          gridTemplateColumns: `repeat(auto-fill, ${cell}px)`,
          gap,
          justifyContent: "center",
        }}
      >
        {Array.from({ length: rendered }, (_, i) => {
          const on = bits[i] === 1;
          const isProbed = probed && probed.has(i);
          const justSet = lastInserted && lastInserted.has(i);
          let background = on ? C.set : C.faint;
          if (isProbed) background = on ? C.set : "#fff";
          return (
            <div
              key={i}
              title={`bit ${i} = ${bits[i]}`}
              style={{
                width: cell,
                height: cell,
                borderRadius: cell > 16 ? 4 : 3,
                background,
                border: isProbed ? `2px solid ${C.probe}` : `1px solid ${on ? C.set : C.border}`,
                boxShadow: justSet && !reduce ? `0 0 0 3px ${C.accent}33` : "none",
                transform: isProbed && !reduce ? "scale(1.18)" : "scale(1)",
                transition: reduce
                  ? "none"
                  : `transform 180ms ${EASE}, background 180ms ${EASE}, box-shadow 220ms ${EASE}`,
              }}
            />
          );
        })}
      </div>
      {m > rendered && (
        <div style={{ fontSize: 11, color: C.muted, textAlign: "center", marginTop: 10 }}>
          Showing the first {rendered} of {m} bits to keep the grid readable. The filter still uses
          all {m}.
        </div>
      )}
    </div>
  );
}

function Legend() {
  const items = [
    { c: C.set, label: "bit set to 1", border: C.set },
    { c: C.faint, label: "bit still 0", border: C.border },
    { c: "#fff", label: "probed this step", border: C.probe },
  ];
  return (
    <div style={{ display: "flex", gap: 16, justifyContent: "center", flexWrap: "wrap" }}>
      {items.map((it) => (
        <div key={it.label} style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span
            style={{
              width: 14,
              height: 14,
              borderRadius: 3,
              background: it.c,
              border: `2px solid ${it.border}`,
              display: "inline-block",
            }}
          />
          <span style={{ fontSize: 11.5, color: C.muted }}>{it.label}</span>
        </div>
      ))}
    </div>
  );
}

const SAMPLE_QUERIES = ["apple", "mango", "grape", "kiwi", "guava", "lychee"];
const SEED_KEYS = ["apple", "banana", "cherry"];

export default function App() {
  const reduce = usePrefersReducedMotion();
  const [m, setM] = useState(48);
  const [k, setK] = useState(3);
  const [bits, setBits] = useState(() => new Uint8Array(48));
  const [inserted, setInserted] = useState(() => new Set());
  const [insertField, setInsertField] = useState("");
  const [queryField, setQueryField] = useState("");
  const [probed, setProbed] = useState(null);
  const [lastInserted, setLastInserted] = useState(null);
  const [result, setResult] = useState(null);
  const [negTested, setNegTested] = useState(0);
  const [falsePos, setFalsePos] = useState(0);
  const animRef = useRef(null);

  const n = inserted.size;
  const fillCount = useMemo(() => bits.reduce((a, b) => a + b, 0), [bits]);
  const fillRatio = m > 0 ? fillCount / m : 0;
  const theoryFP = theoreticalFP(k, n, m);
  const measuredFP = negTested > 0 ? falsePos / negTested : null;
  const bestK = optimalK(m, n);

  const clearAnim = useCallback(() => {
    if (animRef.current) {
      clearTimeout(animRef.current);
      animRef.current = null;
    }
  }, []);

  useEffect(() => clearAnim, [clearAnim]);

  function animateProbes(indices, after) {
    clearAnim();
    if (reduce) {
      setProbed(new Set(indices));
      after();
      animRef.current = setTimeout(() => setProbed(null), 900);
      return;
    }
    const acc = new Set();
    let i = 0;
    const tick = () => {
      acc.add(indices[i]);
      setProbed(new Set(acc));
      i++;
      if (i < indices.length) {
        animRef.current = setTimeout(tick, 150);
      } else {
        after();
        animRef.current = setTimeout(() => setProbed(null), 1100);
      }
    };
    tick();
  }

  function handleInsert(rawKey) {
    const key = rawKey.trim();
    if (!key) return;
    const indices = hashIndices(key, k, m);
    setLastInserted(new Set(indices));
    animateProbes(indices, () => {
      setBits((prev) => {
        const next = new Uint8Array(prev);
        for (const idx of indices) next[idx] = 1;
        return next;
      });
      setInserted((prev) => new Set(prev).add(key));
      setResult({
        kind: "inserted",
        key,
        indices,
        text: `Inserted "${key}". Its ${k} bits are now 1.`,
      });
    });
    setInsertField("");
  }

  function handleQuery(rawKey) {
    const key = rawKey.trim();
    if (!key) return;
    const indices = hashIndices(key, k, m);
    const present = indices.every((idx) => bits[idx] === 1);
    const trulyIn = inserted.has(key);
    setLastInserted(null);
    animateProbes(indices, () => {
      if (present && !trulyIn) {
        setFalsePos((c) => c + 1);
        setNegTested((c) => c + 1);
        setResult({
          kind: "falsepos",
          key,
          indices,
          text: `"${key}" was never inserted, yet all ${k} of its bits are already 1. Bloom filter says possibly present. This is a false positive: other keys happened to set every bit it checks.`,
        });
      } else if (present && trulyIn) {
        setResult({
          kind: "present",
          key,
          indices,
          text: `"${key}" is in the set and all ${k} bits are 1. Reported possibly present, and this time it truly is.`,
        });
      } else {
        if (!trulyIn) setNegTested((c) => c + 1);
        const missing = indices.find((idx) => bits[idx] === 0);
        setResult({
          kind: "absent",
          key,
          indices,
          text: `"${key}" is definitely not present. Bit ${missing} is still 0, and an inserted key would have set every one of its bits. No false negatives, ever.`,
        });
      }
    });
    setQueryField("");
  }

  function changeM(nextM) {
    clearAnim();
    setProbed(null);
    setLastInserted(null);
    setM(nextM);
    setBits(rebuildWith(nextM, k, inserted));
    setResult(null);
    setNegTested(0);
    setFalsePos(0);
  }

  function changeK(nextK) {
    clearAnim();
    setProbed(null);
    setLastInserted(null);
    setK(nextK);
    setBits(rebuildWith(m, nextK, inserted));
    setResult(null);
    setNegTested(0);
    setFalsePos(0);
  }

  function rebuildWith(useM, useK, keys) {
    const arr = new Uint8Array(useM);
    for (const key of keys) {
      for (const idx of hashIndices(key, useK, useM)) arr[idx] = 1;
    }
    return arr;
  }

  function bulkInsert(count) {
    clearAnim();
    setProbed(null);
    const next = new Set(inserted);
    for (let i = 0; i < count; i++) {
      next.add(`rand-${Math.random().toString(36).slice(2, 9)}`);
    }
    setInserted(next);
    setBits(rebuildWith(m, k, next));
    setLastInserted(null);
    setResult({
      kind: "bulk",
      text: `Added ${count} random keys. Fill ratio climbs, and so does the false-positive rate.`,
    });
  }

  function reset() {
    clearAnim();
    const seeded = new Set(SEED_KEYS);
    setM(48);
    setK(3);
    setInserted(seeded);
    setBits(rebuildWith(48, 3, seeded));
    setInsertField("");
    setQueryField("");
    setProbed(null);
    setLastInserted(null);
    setResult(null);
    setNegTested(0);
    setFalsePos(0);
  }

  useEffect(() => {
    const seeded = new Set(SEED_KEYS);
    setInserted(seeded);
    setBits(rebuildWith(48, 3, seeded));
    setResult({
      kind: "intro",
      text: "Three keys are already in: apple, banana, cherry. Query one of them, then query something you never inserted.",
    });
  }, []);

  const curve = useMemo(() => {
    const pts = [];
    for (let nn = 0; nn <= m; nn += Math.max(1, Math.round(m / 60))) {
      pts.push({ n: nn, fp: theoreticalFP(k, nn, m) });
    }
    if (pts[pts.length - 1].n !== m) pts.push({ n: m, fp: theoreticalFP(k, m, m) });
    return pts;
  }, [k, m]);

  const resultStyle = {
    inserted: { bg: C.accentSoft, color: C.accent, border: `${C.accent}33` },
    present: { bg: "#e9f2ec", color: C.good, border: `${C.good}33` },
    absent: { bg: "#eef3f5", color: C.probe, border: `${C.probe}33` },
    falsepos: { bg: "#fbecea", color: C.bad, border: `${C.bad}44` },
    bulk: { bg: C.faint, color: C.muted, border: C.border },
    intro: { bg: C.faint, color: C.muted, border: C.border },
  };
  const rs = result ? resultStyle[result.kind] : resultStyle.intro;

  return (
    <div
      style={{
        fontFamily: "'Palatino Linotype', 'Book Antiqua', Palatino, serif",
        background: C.bg,
        minHeight: "100vh",
        padding: "24px 14px",
        color: C.ink,
      }}
    >
      <style>{`
        input[type="range"]:focus-visible { outline: 2px solid ${C.accent}; outline-offset: 3px; border-radius: 4px; }
        input[type="text"]:focus-visible { outline: 2px solid ${C.accent}; outline-offset: 1px; }
        button:focus-visible { outline: 2px solid ${C.accent}; outline-offset: 2px; }
        @media (prefers-reduced-motion: reduce) {
          * { transition: none !important; animation: none !important; }
        }
      `}</style>

      <div style={{ maxWidth: 720, margin: "0 auto" }}>
        <div style={{ marginBottom: 22 }}>
          <div
            style={{
              fontSize: 10,
              letterSpacing: "0.15em",
              textTransform: "uppercase",
              color: C.muted,
              marginBottom: 5,
            }}
          >
            Data Structures · Probabilistic
          </div>
          <h1 style={{ fontSize: 25, fontWeight: 700, margin: 0 }}>Bloom Filters</h1>
          <p
            style={{
              color: C.ink,
              fontSize: 13.5,
              margin: "8px 0 0",
              lineHeight: 1.65,
              maxWidth: "64ch",
            }}
          >
            You have a membership check that is expensive: a database round trip, a disk
            seek, a call to another service to ask "have we seen this before?" A Bloom filter
            is a tiny in-memory thing you put <i>in front</i> of that check. You ask it first.
            If it says no, you can skip the expensive lookup entirely and trust that answer. If
            it says maybe, you fall through and do the real check. Being wrong only ever costs
            you one wasted lookup; it never gives you a wrong answer.
          </p>
          <p
            style={{
              color: C.ink,
              fontSize: 13.5,
              margin: "10px 0 0",
              lineHeight: 1.65,
              maxWidth: "64ch",
            }}
          >
            Under the hood it is one bit array shared by every key. Adding a key hashes it to{" "}
            <i>k</i> positions and flips those bits to 1. Checking a key tests the same{" "}
            <i>k</i> positions. Any 0 means it was definitely never added, because adding it
            would have set that bit. All 1s mean it was <i>probably</i> added: other keys may
            have flipped those same bits, and the filter cannot tell the difference. That is
            the whole trade, and the playground below lets you trigger it yourself.
          </p>
        </div>

        <Card style={{ marginBottom: 16 }}>
          <form
            onSubmit={(e) => {
              e.preventDefault();
              handleInsert(insertField);
            }}
            style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 12 }}
          >
            <input
              type="text"
              value={insertField}
              onChange={(e) => setInsertField(e.target.value)}
              placeholder="key to insert"
              aria-label="Key to insert into the filter"
              style={{
                flex: "1 1 140px",
                minWidth: 0,
                padding: "8px 12px",
                borderRadius: 8,
                border: `1px solid ${C.border}`,
                fontSize: 13,
                fontFamily: "inherit",
                color: C.ink,
                background: "#fff",
              }}
            />
            <Btn primary type="submit" ariaLabel="Insert the typed key">
              Insert
            </Btn>
          </form>

          <form
            onSubmit={(e) => {
              e.preventDefault();
              handleQuery(queryField);
            }}
            style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 14 }}
          >
            <input
              type="text"
              value={queryField}
              onChange={(e) => setQueryField(e.target.value)}
              placeholder="key to query"
              aria-label="Key to query against the filter"
              style={{
                flex: "1 1 140px",
                minWidth: 0,
                padding: "8px 12px",
                borderRadius: 8,
                border: `1px solid ${C.border}`,
                fontSize: 13,
                fontFamily: "inherit",
                color: C.ink,
                background: "#fff",
              }}
            />
            <Btn type="submit" ariaLabel="Query the typed key">
              Query
            </Btn>
          </form>

          <div style={{ fontSize: 11.5, color: C.muted, marginBottom: 8 }}>
            Quick queries (none of these were inserted unless you added them):
          </div>
          <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginBottom: 16 }}>
            {SAMPLE_QUERIES.map((q) => (
              <button
                key={q}
                onClick={() => handleQuery(q)}
                style={{
                  padding: "5px 12px",
                  borderRadius: 999,
                  border: `1px solid ${C.border}`,
                  background: "#fff",
                  color: C.ink,
                  fontSize: 12,
                  fontFamily: "inherit",
                  cursor: "pointer",
                  transition: `transform 140ms ${EASE}`,
                }}
                onPointerDown={(e) => (e.currentTarget.style.transform = "scale(0.96)")}
                onPointerUp={(e) => (e.currentTarget.style.transform = "scale(1)")}
                onPointerLeave={(e) => (e.currentTarget.style.transform = "scale(1)")}
              >
                {q}
              </button>
            ))}
          </div>

          <BitGrid
            bits={bits}
            m={m}
            probed={probed}
            lastInserted={lastInserted}
            reduce={reduce}
          />

          <div style={{ marginTop: 14 }}>
            <Legend />
          </div>

          <div
            aria-live="polite"
            style={{
              marginTop: 14,
              minHeight: 52,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            {result && (
              <div
                style={{
                  background: rs.bg,
                  border: `1px solid ${rs.border}`,
                  color: rs.color,
                  borderRadius: 10,
                  padding: "10px 16px",
                  fontSize: 12.5,
                  lineHeight: 1.55,
                  maxWidth: "62ch",
                  textAlign: "center",
                  transition: reduce ? "none" : `opacity 200ms ${EASE}`,
                }}
              >
                {result.kind === "falsepos" && (
                  <b style={{ display: "block", marginBottom: 2 }}>False positive</b>
                )}
                {result.text}
              </div>
            )}
          </div>
        </Card>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))",
            gap: 14,
            marginBottom: 16,
          }}
        >
          <Card>
            <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 14 }}>Tune the filter</div>
            <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
              <Slider
                id="m-slider"
                label="m — bits in the array"
                value={m}
                min={16}
                max={512}
                onChange={changeM}
              />
              <Slider
                id="k-slider"
                label="k — hash functions per key"
                value={k}
                min={1}
                max={10}
                onChange={changeK}
              />
            </div>
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginTop: 16 }}>
              <Btn onClick={() => bulkInsert(10)} ariaLabel="Insert ten random keys">
                +10 random
              </Btn>
              <Btn onClick={() => bulkInsert(40)} ariaLabel="Insert forty random keys">
                +40 random
              </Btn>
              <Btn onClick={reset} ariaLabel="Reset the filter to its starting state">
                Reset
              </Btn>
            </div>
            <p style={{ fontSize: 12, color: C.muted, lineHeight: 1.6, margin: "14px 0 0" }}>
              Optimal k for the current load is{" "}
              <b style={{ color: C.accent }}>{bestK}</b> (from k = (m/n)·ln2). Below it you waste
              bits; above it you flood the array and collisions rise.
            </p>
          </Card>

          <Card>
            <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 12 }}>Live state</div>
            <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginBottom: 10 }}>
              <Stat
                label="Fill ratio"
                value={`${Math.round(fillRatio * 100)}%`}
                sub={`${fillCount} of ${m} bits are 1`}
                color={C.accent}
                bg={C.accentSoft}
              />
              <Stat
                label="Keys inserted (n)"
                value={n}
                sub="the true set"
                color={C.probe}
                bg="#eef3f5"
              />
            </div>
            <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
              <Stat
                label="Predicted FP rate"
                value={`${(theoryFP * 100).toFixed(1)}%`}
                sub="(1 − e^(−kn/m))^k"
                color={C.ink}
                bg={C.faint}
              />
              <Stat
                label="Measured FP rate"
                value={
                  measuredFP === null ? "—" : `${(measuredFP * 100).toFixed(1)}%`
                }
                sub={
                  negTested === 0
                    ? "query absent keys to measure"
                    : `${falsePos} of ${negTested} absent queries`
                }
                color={measuredFP && measuredFP > 0 ? C.bad : C.good}
                bg={measuredFP && measuredFP > 0 ? "#fbecea" : "#e9f2ec"}
              />
            </div>
          </Card>
        </div>

        <Card style={{ marginBottom: 16 }}>
          <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 4 }}>
            False positives climb as the array fills
          </div>
          <div style={{ fontSize: 12.5, color: C.muted, lineHeight: 1.6, marginBottom: 12 }}>
            With m = {m} bits and k = {k} hashes, this is the predicted false-positive rate as you
            add keys. The marker sits at your current n = {n}.
          </div>
          <ResponsiveContainer width="100%" height={190}>
            <AreaChart data={curve} margin={{ left: -12, right: 10, top: 6 }}>
              <defs>
                <linearGradient id="fpFill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={C.accent} stopOpacity={0.32} />
                  <stop offset="100%" stopColor={C.accent} stopOpacity={0.02} />
                </linearGradient>
              </defs>
              <XAxis
                dataKey="n"
                tick={{ fontSize: 11, fill: C.muted }}
                label={{ value: "keys inserted (n)", position: "insideBottom", offset: -2, fontSize: 11, fill: C.muted }}
              />
              <YAxis
                tickFormatter={(v) => `${Math.round(v * 100)}%`}
                tick={{ fontSize: 10, fill: C.muted }}
                width={42}
              />
              <Tooltip
                cursor={{ stroke: C.border }}
                contentStyle={{ borderRadius: 8, border: `1px solid ${C.border}`, fontSize: 12 }}
                formatter={(v) => [`${(v * 100).toFixed(1)}%`, "predicted FP"]}
                labelFormatter={(l) => `n = ${l}`}
              />
              <Area
                type="monotone"
                dataKey="fp"
                stroke={C.accent}
                strokeWidth={2}
                fill="url(#fpFill)"
                isAnimationActive={!reduce}
                animationDuration={300}
              />
              <ReferenceLine x={n} stroke={C.probe} strokeDasharray="4 3" />
            </AreaChart>
          </ResponsiveContainer>
        </Card>

        <Card style={{ background: C.accentSoft, border: `1px solid ${C.accent}22` }}>
          <div
            style={{
              fontSize: 11,
              fontWeight: 700,
              letterSpacing: "0.08em",
              textTransform: "uppercase",
              color: C.accent,
              marginBottom: 10,
            }}
          >
            Why it behaves this way
          </div>
          <div style={{ fontSize: 13, color: C.ink, lineHeight: 1.7, display: "grid", gap: 10 }}>
            <p style={{ margin: 0 }}>
              <b>No false negatives, ever.</b> Inserting a key sets every one of its k bits to 1,
              and bits are never cleared. So when you query a key that really was inserted, all k
              bits it checks are guaranteed to still be 1. A "definitely not present" answer is
              therefore always trustworthy.
            </p>
            <p style={{ margin: 0 }}>
              <b>Why false positives happen.</b> Different keys share the same array, so one key's
              bits can be set by a mix of other keys. If every bit a never-inserted key probes was
              already flipped on by someone else, the filter cannot tell the difference and answers
              possibly present. The denser the array, the likelier this collision.
            </p>
            <p style={{ margin: 0 }}>
              <b>The m, k, n trade-off.</b> More bits (m) spreads keys out and lowers collisions.
              More keys (n) fills the array and raises them. More hashes (k) sets more bits per key,
              which sharpens the test up to a point, then past that it just saturates the array. The
              sweet spot is k = (m/n)·ln2, which keeps the array about half full and minimizes the
              false-positive rate (1 − e^(−kn/m))^k.
            </p>
          </div>
        </Card>

        <div style={{ marginTop: 18, textAlign: "center", fontSize: 11, color: C.muted }}>
          Bits are set by real FNV-1a hashes combined by double hashing: index_i = (h1 + i·h2) mod m
        </div>
      </div>
    </div>
  );
}
