import { useState, useEffect, useRef } from "react";

const TOKENS = ["The", "cat", "sat", "on", "mat"];
const COLORS = ["#e8d5b7", "#c9e4ca", "#c6d8f5", "#f5c6d0", "#e8c6f5"];
const TOKEN_COLORS = ["#8B6914", "#2D7A35", "#1A4D8C", "#8C2D3A", "#5A2D8C"];

const stages = [
  {
    id: "tokens",
    title: "Step 1: Tokens",
    subtitle: "Words become vectors",
    description: "Each word is converted into a list of numbers — a vector. These numbers capture rough meaning. This is the starting point.",
  },
  {
    id: "qkv",
    title: "Step 2: Q, K, V",
    subtitle: "Three roles per token",
    description: "Each token projects itself into three different vectors. Q asks a question. K advertises what it contains. V holds the actual information to share.",
  },
  {
    id: "scores",
    title: "Step 3: Attention Scores",
    subtitle: "Who pays attention to whom?",
    description: 'Pick a token below. Its Query vector is dotted against every other token\'s Key vector. Higher dot product = more "relevant". These raw scores are then softmaxed into weights that sum to 1.',
  },
  {
    id: "output",
    title: "Step 4: Weighted Sum",
    subtitle: "Blend the Values",
    description: "The attention weights are used to blend the Value vectors. Each token\'s final representation is a weighted mix of everyone else\'s content — shaped by who it paid attention to.",
  },
  {
    id: "kvcache",
    title: "Step 5: KV Cache",
    subtitle: "Don't recompute what you already know",
    description: "During generation, you produce one token at a time. K and V for old tokens don't change — so we cache them. Only the new token needs fresh computation.",
  },
];

// Seeded random scores for each token pair
const ATTENTION_SCORES = [
  [1.0, 0.15, 0.08, 0.45, 0.32],
  [0.22, 1.0, 0.61, 0.18, 0.29],
  [0.09, 0.72, 1.0, 0.55, 0.14],
  [0.41, 0.19, 0.48, 1.0, 0.88],
  [0.30, 0.25, 0.12, 0.91, 1.0],
];

function softmax(arr) {
  const exp = arr.map((x) => Math.exp(x * 3));
  const sum = exp.reduce((a, b) => a + b, 0);
  return exp.map((x) => x / sum);
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

// Token chip
function TokenChip({ token, color, textColor, size = "md", glow, dim, onClick, label }) {
  const sizes = { sm: "px-2 py-1 text-xs", md: "px-3 py-1.5 text-sm", lg: "px-4 py-2 text-base" };
  return (
    <div
      onClick={onClick}
      style={{
        background: color,
        color: textColor || "#1a1a1a",
        borderRadius: 8,
        fontFamily: "'IBM Plex Mono', monospace",
        fontWeight: 600,
        cursor: onClick ? "pointer" : "default",
        opacity: dim ? 0.3 : 1,
        boxShadow: glow ? `0 0 16px ${color}99, 0 2px 8px ${color}55` : "0 1px 3px rgba(0,0,0,0.1)",
        transition: "all 0.25s ease",
        transform: glow ? "scale(1.06)" : "scale(1)",
        display: "inline-flex",
        flexDirection: "column",
        alignItems: "center",
        gap: 2,
        userSelect: "none",
      }}
      className={sizes[size]}
    >
      <span>{token}</span>
      {label && <span style={{ fontSize: 9, opacity: 0.7, fontWeight: 400 }}>{label}</span>}
    </div>
  );
}

// Animated number vector
function VectorBar({ value, color, label, animate }) {
  const [displayed, setDisplayed] = useState(0);
  useEffect(() => {
    if (!animate) { setDisplayed(value); return; }
    let start = null;
    const dur = 600;
    const step = (ts) => {
      if (!start) start = ts;
      const t = Math.min((ts - start) / dur, 1);
      setDisplayed(lerp(0, value, t));
      if (t < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }, [value, animate]);

  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
      <span style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 10, color: "#888", width: 12 }}>{label}</span>
      <div style={{ flex: 1, background: "#f0f0f0", borderRadius: 4, height: 10, overflow: "hidden" }}>
        <div style={{
          width: `${displayed * 100}%`, height: "100%",
          background: color, borderRadius: 4,
          transition: "width 0.05s linear",
        }} />
      </div>
      <span style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 10, color: "#555", width: 28, textAlign: "right" }}>
        {displayed.toFixed(2)}
      </span>
    </div>
  );
}

// Stage 1: Tokens
function StageTokens() {
  const [revealed, setRevealed] = useState([]);
  const vecs = [
    [0.82, 0.14, 0.55, 0.31],
    [0.23, 0.91, 0.44, 0.67],
    [0.11, 0.76, 0.88, 0.22],
    [0.58, 0.33, 0.19, 0.74],
    [0.40, 0.62, 0.71, 0.48],
  ];
  return (
    <div style={{ display: "flex", gap: 16, flexWrap: "wrap", justifyContent: "center" }}>
      {TOKENS.map((t, i) => (
        <div key={t} style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 10 }}>
          <TokenChip token={t} color={COLORS[i]} textColor={TOKEN_COLORS[i]} size="lg"
            onClick={() => setRevealed(r => r.includes(i) ? r.filter(x => x !== i) : [...r, i])}
            glow={revealed.includes(i)}
          />
          {revealed.includes(i) && (
            <div style={{
              background: "#fff", border: `2px solid ${COLORS[i]}`,
              borderRadius: 10, padding: "10px 14px", width: 180,
              boxShadow: "0 4px 16px rgba(0,0,0,0.08)",
            }}>
              <div style={{ fontSize: 10, color: "#aaa", fontFamily: "'IBM Plex Mono', monospace", marginBottom: 6 }}>
                embedding vector
              </div>
              {vecs[i].map((v, j) => (
                <VectorBar key={j} value={v} color={COLORS[i]} label={`d${j}`} animate />
              ))}
            </div>
          )}
        </div>
      ))}
      <div style={{ width: "100%", textAlign: "center", fontSize: 12, color: "#aaa", fontFamily: "Georgia, serif", marginTop: 8 }}>
        tap a token to see its embedding vector
      </div>
    </div>
  );
}

// Stage 2: QKV
function StageQKV() {
  const [selected, setSelected] = useState(null);
  const roles = [
    { label: "Q", desc: "What am I looking for?", color: "#dbeafe", border: "#3b82f6" },
    { label: "K", desc: "What do I contain?", color: "#dcfce7", border: "#22c55e" },
    { label: "V", desc: "What will I share?", color: "#fef9c3", border: "#eab308" },
  ];
  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 20 }}>
      <div style={{ display: "flex", gap: 12, justifyContent: "center", flexWrap: "wrap" }}>
        {TOKENS.map((t, i) => (
          <TokenChip key={t} token={t} color={COLORS[i]} textColor={TOKEN_COLORS[i]} size="md"
            onClick={() => setSelected(selected === i ? null : i)}
            glow={selected === i}
          />
        ))}
      </div>
      {selected !== null && (
        <div style={{ display: "flex", gap: 12, flexWrap: "wrap", justifyContent: "center" }}>
          {roles.map((r) => (
            <div key={r.label} style={{
              background: r.color, border: `2px solid ${r.border}`,
              borderRadius: 12, padding: "12px 18px", minWidth: 130,
              boxShadow: "0 2px 12px rgba(0,0,0,0.06)",
            }}>
              <div style={{ fontSize: 22, fontFamily: "'IBM Plex Mono', monospace", fontWeight: 700, color: r.border }}>{r.label}</div>
              <div style={{ fontSize: 11, color: "#555", marginTop: 4 }}>{r.desc}</div>
              <div style={{ marginTop: 10 }}>
                {[0.6, 0.3, 0.8, 0.4].map((v, j) => (
                  <VectorBar key={j} value={v * (0.7 + selected * 0.1)} color={r.border} label={`${j}`} animate />
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
      {selected === null && (
        <div style={{ fontSize: 12, color: "#aaa", fontFamily: "Georgia, serif" }}>
          tap a token to see its Q, K, V projections
        </div>
      )}
    </div>
  );
}

// Stage 3: Scores
function StageScores() {
  const [focus, setFocus] = useState(0);
  const rawScores = ATTENTION_SCORES[focus];
  const weights = softmax(rawScores);

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 20 }}>
      <div>
        <div style={{ fontSize: 11, color: "#aaa", textAlign: "center", marginBottom: 8, fontFamily: "Georgia, serif" }}>
          whose perspective?
        </div>
        <div style={{ display: "flex", gap: 8, justifyContent: "center" }}>
          {TOKENS.map((t, i) => (
            <TokenChip key={t} token={t} color={COLORS[i]} textColor={TOKEN_COLORS[i]} size="md"
              onClick={() => setFocus(i)} glow={focus === i}
            />
          ))}
        </div>
      </div>

      <div style={{ width: "100%", maxWidth: 480 }}>
        <div style={{ fontSize: 11, color: "#aaa", marginBottom: 10, fontFamily: "Georgia, serif", textAlign: "center" }}>
          attention weights — how much <strong style={{ color: TOKEN_COLORS[focus] }}>"{TOKENS[focus]}"</strong> attends to each token
        </div>
        {TOKENS.map((t, i) => (
          <div key={t} style={{
            display: "flex", alignItems: "center", gap: 10,
            marginBottom: 8, opacity: weights[i] < 0.05 ? 0.4 : 1,
            transition: "opacity 0.3s",
          }}>
            <TokenChip token={t} color={COLORS[i]} textColor={TOKEN_COLORS[i]} size="sm" />
            <div style={{ flex: 1, background: "#f0f0f0", borderRadius: 6, height: 18, overflow: "hidden" }}>
              <div style={{
                width: `${weights[i] * 100}%`, height: "100%",
                background: `linear-gradient(90deg, ${COLORS[focus]}, ${COLORS[i]})`,
                borderRadius: 6, transition: "width 0.5s cubic-bezier(0.4,0,0.2,1)",
              }} />
            </div>
            <span style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 11, color: "#555", width: 36, textAlign: "right" }}>
              {(weights[i] * 100).toFixed(0)}%
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

// Stage 4: Output
function StageOutput() {
  const [focus, setFocus] = useState(2);
  const weights = softmax(ATTENTION_SCORES[focus]);
  const baseVecs = [
    [0.82, 0.14, 0.55, 0.31],
    [0.23, 0.91, 0.44, 0.67],
    [0.11, 0.76, 0.88, 0.22],
    [0.58, 0.33, 0.19, 0.74],
    [0.40, 0.62, 0.71, 0.48],
  ];
  const output = [0, 1, 2, 3].map(d =>
    baseVecs.reduce((sum, vec, i) => sum + vec[d] * weights[i], 0)
  );
  const maxOut = Math.max(...output);
  const normOutput = output.map(v => v / maxOut);

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 20 }}>
      <div>
        <div style={{ fontSize: 11, color: "#aaa", textAlign: "center", marginBottom: 8, fontFamily: "Georgia, serif" }}>select token</div>
        <div style={{ display: "flex", gap: 8, justifyContent: "center" }}>
          {TOKENS.map((t, i) => (
            <TokenChip key={t} token={t} color={COLORS[i]} textColor={TOKEN_COLORS[i]} size="md"
              onClick={() => setFocus(i)} glow={focus === i}
            />
          ))}
        </div>
      </div>

      <div style={{ display: "flex", gap: 24, alignItems: "flex-start", flexWrap: "wrap", justifyContent: "center" }}>
        <div style={{ minWidth: 180 }}>
          <div style={{ fontSize: 10, color: "#aaa", marginBottom: 6, fontFamily: "Georgia, serif", textAlign: "center" }}>
            attention weights
          </div>
          {TOKENS.map((t, i) => (
            <div key={t} style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 5 }}>
              <TokenChip token={t} color={COLORS[i]} textColor={TOKEN_COLORS[i]} size="sm" />
              <span style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 11, color: "#888" }}>
                ×{weights[i].toFixed(2)}
              </span>
            </div>
          ))}
        </div>

        <div style={{ fontSize: 28, color: "#ccc", alignSelf: "center" }}>→</div>

        <div style={{ minWidth: 180 }}>
          <div style={{ fontSize: 10, color: "#aaa", marginBottom: 6, fontFamily: "Georgia, serif", textAlign: "center" }}>
            new representation of "{TOKENS[focus]}"
          </div>
          <div style={{
            background: COLORS[focus], border: `2px solid ${TOKEN_COLORS[focus]}33`,
            borderRadius: 10, padding: "12px 14px",
          }}>
            {normOutput.map((v, j) => (
              <VectorBar key={j} value={v} color={TOKEN_COLORS[focus]} label={`d${j}`} animate />
            ))}
          </div>
          <div style={{ fontSize: 10, color: "#aaa", marginTop: 6, fontFamily: "Georgia, serif", textAlign: "center" }}>
            a blend of all values, shaped by attention
          </div>
        </div>
      </div>
    </div>
  );
}

// Stage 5: KV Cache
function StageKVCache() {
  const [step, setStep] = useState(0);
  const generated = ["The", "cat", "sat", "on", "mat"];
  const visible = generated.slice(0, step + 1);
  const isNew = (i) => i === step;

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 20 }}>
      <div style={{ fontSize: 12, color: "#888", fontFamily: "Georgia, serif", textAlign: "center" }}>
        generating token by token — step {step + 1} of {generated.length}
      </div>

      <div style={{ display: "flex", gap: 8, justifyContent: "center" }}>
        {visible.map((t, i) => (
          <TokenChip key={i} token={t} color={isNew(i) ? "#fff3cd" : COLORS[i]}
            textColor={isNew(i) ? "#856404" : TOKEN_COLORS[i]}
            size="md" glow={isNew(i)}
            label={isNew(i) ? "new" : "cached"}
          />
        ))}
      </div>

      <div style={{ width: "100%", maxWidth: 480 }}>
        <div style={{ fontSize: 11, color: "#aaa", marginBottom: 8, fontFamily: "Georgia, serif", textAlign: "center" }}>
          KV cache — what's stored in memory
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
          {["K (keys)", "V (values)"].map(label => (
            <div key={label} style={{ background: "#f8f8f8", borderRadius: 10, padding: 12, border: "1px solid #eee" }}>
              <div style={{ fontSize: 10, color: "#aaa", marginBottom: 8, fontFamily: "'IBM Plex Mono', monospace" }}>{label}</div>
              {visible.map((t, i) => (
                <div key={i} style={{
                  display: "flex", alignItems: "center", gap: 6, marginBottom: 4,
                  background: isNew(i) ? "#fff3cd" : COLORS[i] + "55",
                  borderRadius: 6, padding: "4px 8px",
                  border: isNew(i) ? "1px solid #ffc107" : "none",
                  transition: "all 0.3s",
                }}>
                  <span style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 10, color: TOKEN_COLORS[i], fontWeight: 600 }}>{t}</span>
                  <div style={{ flex: 1, display: "flex", gap: 2 }}>
                    {[0, 1, 2].map(j => (
                      <div key={j} style={{
                        flex: 1, height: 6, borderRadius: 3,
                        background: isNew(i) ? "#ffc107" : TOKEN_COLORS[i] + "66",
                      }} />
                    ))}
                  </div>
                  {isNew(i) && <span style={{ fontSize: 8, color: "#856404", fontFamily: "'IBM Plex Mono', monospace" }}>NEW</span>}
                  {!isNew(i) && <span style={{ fontSize: 8, color: "#aaa", fontFamily: "'IBM Plex Mono', monospace" }}>HIT</span>}
                </div>
              ))}
            </div>
          ))}
        </div>
        <div style={{ marginTop: 10, background: "#f0fdf4", border: "1px solid #86efac", borderRadius: 8, padding: "8px 12px", fontSize: 11, color: "#166534" }}>
          <strong>Work saved:</strong> only computing K+V for 1 token, not {visible.length}.
          Cache grows by 1 row per step.
        </div>
      </div>

      <div style={{ display: "flex", gap: 10 }}>
        <button onClick={() => setStep(s => Math.max(0, s - 1))} disabled={step === 0}
          style={{
            background: "#fff", border: "1px solid #ddd", borderRadius: 8,
            padding: "8px 18px", cursor: step === 0 ? "not-allowed" : "pointer",
            fontFamily: "'IBM Plex Mono', monospace", fontSize: 12, opacity: step === 0 ? 0.4 : 1,
          }}>← prev</button>
        <button onClick={() => setStep(s => Math.min(generated.length - 1, s + 1))} disabled={step === generated.length - 1}
          style={{
            background: "#1a1a1a", color: "#fff", border: "none", borderRadius: 8,
            padding: "8px 18px", cursor: step === generated.length - 1 ? "not-allowed" : "pointer",
            fontFamily: "'IBM Plex Mono', monospace", fontSize: 12, opacity: step === generated.length - 1 ? 0.4 : 1,
          }}>generate next →</button>
      </div>
    </div>
  );
}

const stageComponents = [StageTokens, StageQKV, StageScores, StageOutput, StageKVCache];

export default function App() {
  const [stage, setStage] = useState(0);
  const StageComp = stageComponents[stage];

  return (
    <div style={{
      minHeight: "100vh",
      background: "#fafaf8",
      fontFamily: "Georgia, serif",
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      padding: "32px 16px",
    }}>
      {/* Import fonts */}
      <style>{`@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&display=swap');`}</style>

      {/* Header */}
      <div style={{ textAlign: "center", marginBottom: 28 }}>
        <div style={{ fontSize: 11, letterSpacing: 3, color: "#aaa", textTransform: "uppercase", marginBottom: 6 }}>
          visual explainer
        </div>
        <h1 style={{ fontSize: 24, fontWeight: 400, color: "#1a1a1a", margin: 0 }}>
          Attention & KV Cache
        </h1>
      </div>

      {/* Stage nav */}
      <div style={{ display: "flex", gap: 6, marginBottom: 28, flexWrap: "wrap", justifyContent: "center" }}>
        {stages.map((s, i) => (
          <button key={s.id} onClick={() => setStage(i)} style={{
            background: stage === i ? "#1a1a1a" : "#fff",
            color: stage === i ? "#fff" : "#888",
            border: `1px solid ${stage === i ? "#1a1a1a" : "#ddd"}`,
            borderRadius: 20, padding: "5px 14px",
            fontFamily: "'IBM Plex Mono', monospace", fontSize: 11,
            cursor: "pointer", transition: "all 0.2s",
          }}>
            {i + 1}. {s.id}
          </button>
        ))}
      </div>

      {/* Main card */}
      <div style={{
        width: "100%", maxWidth: 600,
        background: "#fff",
        border: "1px solid #ebebeb",
        borderRadius: 16,
        padding: "24px 20px",
        boxShadow: "0 2px 24px rgba(0,0,0,0.05)",
      }}>
        <div style={{ marginBottom: 20 }}>
          <div style={{ fontSize: 11, color: "#aaa", letterSpacing: 2, textTransform: "uppercase", marginBottom: 4 }}>
            {stages[stage].subtitle}
          </div>
          <h2 style={{ fontSize: 18, margin: 0, marginBottom: 8, fontWeight: 500, color: "#1a1a1a" }}>
            {stages[stage].title}
          </h2>
          <p style={{ fontSize: 13, color: "#666", lineHeight: 1.6, margin: 0 }}>
            {stages[stage].description}
          </p>
        </div>

        <div style={{ borderTop: "1px solid #f0f0f0", paddingTop: 20 }}>
          <StageComp />
        </div>
      </div>

      {/* Navigation */}
      <div style={{ display: "flex", gap: 12, marginTop: 20 }}>
        <button onClick={() => setStage(s => Math.max(0, s - 1))} disabled={stage === 0}
          style={{
            background: "#fff", border: "1px solid #ddd", borderRadius: 10,
            padding: "10px 20px", fontFamily: "Georgia, serif", fontSize: 13,
            cursor: stage === 0 ? "not-allowed" : "pointer", color: "#666",
            opacity: stage === 0 ? 0.4 : 1, transition: "opacity 0.2s",
          }}>← back</button>
        <button onClick={() => setStage(s => Math.min(stages.length - 1, s + 1))} disabled={stage === stages.length - 1}
          style={{
            background: "#1a1a1a", color: "#fff", border: "none", borderRadius: 10,
            padding: "10px 24px", fontFamily: "Georgia, serif", fontSize: 13,
            cursor: stage === stages.length - 1 ? "not-allowed" : "pointer",
            opacity: stage === stages.length - 1 ? 0.4 : 1, transition: "opacity 0.2s",
          }}>next →</button>
      </div>
    </div>
  );
}
