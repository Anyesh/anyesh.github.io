import { useState, useEffect } from "react";

export const meta = {
  title: "Attention & the KV Cache",
  category: "Transformers",
  description:
    "One attention head, five tokens, worked by hand. Step from embeddings through Q/K/V and masked softmax to the output, then see why a KV cache makes generation cheap.",
  date: "2026-01-26",
  tags: ["attention", "transformers", "kv-cache", "softmax"],
};

// ── tiny linear algebra helpers ──────────────────────────────────────────────
const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
const matVec = (M, v) => M.map(row => dot(row, v));
const vecAdd = (a, b) => a.map((v, i) => v + b[i]);
const scalarMul = (s, v) => v.map(x => x * s);

function softmax(arr) {
  const m = Math.max(...arr);
  const e = arr.map(x => Math.exp(x - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map(x => x / s);
}

// ── fixed small embeddings (d_model = 4, d_k = d_v = 4) ──────────────────────
// These are hand-crafted so relationships are visible
const D_MODEL = 4;
const D_K = 4;

const TOKENS = ["The", "cat", "sat", "on", "mat"];
const TOKEN_COLORS = ["#e8d5b7", "#c9e4ca", "#c6d8f5", "#f5c6d0", "#e8c6f5"];
const TOKEN_TEXT  = ["#7a5c1e", "#1f6b35", "#1a3d8c", "#8c1a2e", "#5a1a8c"];

// Token embeddings: shape [5, 4]
const EMBEDDINGS = [
  [0.9, 0.1, 0.2, 0.05],  // The   — article, low content
  [0.2, 0.8, 0.6, 0.3 ],  // cat   — subject, animate
  [0.1, 0.3, 0.9, 0.7 ],  // sat   — verb, action
  [0.5, 0.1, 0.1, 0.4 ],  // on    — preposition
  [0.3, 0.7, 0.4, 0.2 ],  // mat   — object, inanimate
];

// Weight matrices: shape [D_K, D_MODEL]
const W_Q = [
  [ 0.6, -0.2,  0.1,  0.3],
  [-0.1,  0.7,  0.2, -0.1],
  [ 0.2,  0.1, -0.5,  0.6],
  [ 0.1, -0.3,  0.4,  0.2],
];
const W_K = [
  [ 0.5,  0.1, -0.2,  0.4],
  [ 0.2,  0.6,  0.1, -0.2],
  [-0.1,  0.2,  0.7,  0.1],
  [ 0.3, -0.1,  0.2,  0.5],
];
const W_V = [
  [ 0.4,  0.2,  0.1,  0.3],
  [-0.1,  0.5,  0.3,  0.1],
  [ 0.2, -0.1,  0.6,  0.2],
  [ 0.1,  0.3, -0.2,  0.7],
];

// Pre-compute Q, K, V for all tokens
const Q = EMBEDDINGS.map(e => matVec(W_Q, e));
const K = EMBEDDINGS.map(e => matVec(W_K, e));
const V = EMBEDDINGS.map(e => matVec(W_V, e));

// Raw attention scores: shape [5, 5]
const RAW_SCORES = Q.map(q => K.map(k => dot(q, k) / Math.sqrt(D_K)));

// Softmax attention weights (no masking): shape [5, 5]
const WEIGHTS_FULL = RAW_SCORES.map(row => softmax(row));

// Softmax attention weights (causal masking): shape [5, 5]
const WEIGHTS_CAUSAL = RAW_SCORES.map((row, i) =>
  softmax(row.map((s, j) => (j > i ? -Infinity : s)))
);

// Output = weighted sum of V
const OUTPUT_FULL   = WEIGHTS_FULL.map(w => w.reduce((acc, wi, i) => vecAdd(acc, scalarMul(wi, V[i])), [0,0,0,0]));
const OUTPUT_CAUSAL = WEIGHTS_CAUSAL.map(w => w.reduce((acc, wi, i) => vecAdd(acc, scalarMul(wi, V[i])), [0,0,0,0]));

// ── colour utils ──────────────────────────────────────────────────────────────
function heatColor(v) {
  // 0 → white, 1 → deep indigo
  const r = Math.round(255 - v * 180);
  const g = Math.round(255 - v * 160);
  const b = Math.round(255 - v * 60);
  return `rgb(${r},${g},${b})`;
}
function textOnHeat(v) { return v > 0.45 ? "#fff" : "#1a1a1a"; }

// ── shared components ─────────────────────────────────────────────────────────
function Chip({ token, i, onClick, glow, dim, label }) {
  return (
    <div onClick={onClick} style={{
      background: TOKEN_COLORS[i], color: TOKEN_TEXT[i],
      borderRadius: 8, padding: "6px 12px",
      fontFamily: "'IBM Plex Mono', monospace", fontWeight: 700, fontSize: 13,
      cursor: onClick ? "pointer" : "default",
      opacity: dim ? 0.25 : 1,
      boxShadow: glow ? `0 0 0 2px ${TOKEN_TEXT[i]}, 0 4px 16px ${TOKEN_COLORS[i]}cc` : "0 1px 4px rgba(0,0,0,0.1)",
      transform: glow ? "scale(1.07)" : "scale(1)",
      transition: "all 0.2s",
      userSelect: "none",
      display: "flex", flexDirection: "column", alignItems: "center", gap: 2,
      minWidth: 48, textAlign: "center",
    }}>
      {token}
      {label && <span style={{ fontSize: 8, fontWeight: 400, opacity: 0.65 }}>{label}</span>}
    </div>
  );
}

function VecDisplay({ vec, color, label, maxVal }) {
  const max = maxVal ?? Math.max(...vec.map(Math.abs), 0.01);
  return (
    <div style={{ background: "#fafaf8", borderRadius: 8, padding: "10px 12px", border: `1.5px solid ${color}44` }}>
      {label && <div style={{ fontSize: 9, color: "#aaa", fontFamily: "'IBM Plex Mono', monospace", marginBottom: 6 }}>{label}</div>}
      {vec.map((v, j) => {
        const pct = Math.abs(v) / max;
        return (
          <div key={j} style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 4 }}>
            <span style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 9, color: "#aaa", width: 10 }}>{j}</span>
            <div style={{ flex: 1, background: "#ececec", borderRadius: 4, height: 8, overflow: "hidden" }}>
              <div style={{
                width: `${pct * 100}%`, height: "100%", borderRadius: 4,
                background: v < 0 ? "#f87171" : color,
                transition: "width 0.4s cubic-bezier(0.4,0,0.2,1)",
              }} />
            </div>
            <span style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 9, color: "#555", width: 36, textAlign: "right" }}>
              {v.toFixed(3)}
            </span>
          </div>
        );
      })}
    </div>
  );
}

function SectionLabel({ children }) {
  return (
    <div style={{ fontSize: 10, letterSpacing: 2, textTransform: "uppercase", color: "#bbb", fontFamily: "'IBM Plex Mono', monospace", marginBottom: 6 }}>
      {children}
    </div>
  );
}

function Formula({ children }) {
  return (
    <div style={{
      background: "#f5f5f0", borderRadius: 8, padding: "8px 14px",
      fontFamily: "'IBM Plex Mono', monospace", fontSize: 12, color: "#444",
      border: "1px solid #e8e8e0", marginBottom: 12, textAlign: "center",
    }}>{children}</div>
  );
}

function InfoBox({ children, color = "#e0f0ff", border = "#93c5fd" }) {
  return (
    <div style={{
      background: color, border: `1px solid ${border}`, borderRadius: 8,
      padding: "10px 14px", fontSize: 12, color: "#1e3a5f", lineHeight: 1.6,
    }}>{children}</div>
  );
}

// ── Stage 1: Embeddings ───────────────────────────────────────────────────────
function StageEmbeddings() {
  const [sel, setSel] = useState(null);
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      <Formula>token → embedding vector ∈ ℝ⁴</Formula>
      <div style={{ display: "flex", gap: 8, justifyContent: "center", flexWrap: "wrap" }}>
        {TOKENS.map((t, i) => (
          <Chip key={t} token={t} i={i} onClick={() => setSel(sel === i ? null : i)} glow={sel === i} />
        ))}
      </div>
      {sel !== null ? (
        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
          <SectionLabel>embedding for "{TOKENS[sel]}"</SectionLabel>
          <VecDisplay vec={EMBEDDINGS[sel]} color={TOKEN_TEXT[sel]} />
          <InfoBox>
            Each dimension encodes a learned feature. Higher values = stronger signal on that feature.
            "cat" scores high on dim 1 (animate?), "sat" high on dim 2 (action?).
          </InfoBox>
        </div>
      ) : (
        <div style={{ textAlign: "center", fontSize: 12, color: "#bbb", fontFamily: "Georgia, serif" }}>
          tap a token to inspect its embedding
        </div>
      )}
    </div>
  );
}

// ── Stage 2: Q K V projection ─────────────────────────────────────────────────
function StageQKV() {
  const [sel, setSel] = useState(null);
  const roles = [
    { key: "Q", label: "Query", desc: "What am I looking for?", vec: sel !== null ? Q[sel] : null, color: "#3b82f6" },
    { key: "K", label: "Key",   desc: "What do I contain?",    vec: sel !== null ? K[sel] : null, color: "#22c55e" },
    { key: "V", label: "Value", desc: "What will I share?",    vec: sel !== null ? V[sel] : null, color: "#f59e0b" },
  ];
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      <Formula>Q = W_Q · e,  K = W_K · e,  V = W_V · e</Formula>
      <InfoBox color="#f0fdf4" border="#86efac">
        Same embedding, three different linear projections. Each weight matrix is <em>learned</em> during training — these are fixed here for illustration.
      </InfoBox>
      <div style={{ display: "flex", gap: 8, justifyContent: "center", flexWrap: "wrap" }}>
        {TOKENS.map((t, i) => (
          <Chip key={t} token={t} i={i} onClick={() => setSel(sel === i ? null : i)} glow={sel === i} />
        ))}
      </div>
      {sel !== null ? (
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 8 }}>
          {roles.map(r => (
            <div key={r.key}>
              <div style={{ textAlign: "center", marginBottom: 4 }}>
                <span style={{ fontFamily: "'IBM Plex Mono', monospace", fontWeight: 700, fontSize: 16, color: r.color }}>{r.key}</span>
                <div style={{ fontSize: 10, color: "#aaa" }}>{r.desc}</div>
              </div>
              <VecDisplay vec={r.vec} color={r.color} />
            </div>
          ))}
        </div>
      ) : (
        <div style={{ textAlign: "center", fontSize: 12, color: "#bbb", fontFamily: "Georgia, serif" }}>
          tap a token to see its Q, K, V
        </div>
      )}
    </div>
  );
}

// ── Stage 3: Dot products & raw scores ────────────────────────────────────────
function StageRawScores() {
  const [focus, setFocus] = useState(2); // "sat"
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      <Formula>score(i, j) = Q_i · K_j  /  √d_k</Formula>
      <InfoBox>
        We divide by √{D_K} = {Math.sqrt(D_K).toFixed(2)} to stop dot products from growing large and pushing softmax into flat regions. This is the "scaled" in "scaled dot-product attention."
      </InfoBox>
      <div>
        <SectionLabel>whose query?</SectionLabel>
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
          {TOKENS.map((t, i) => (
            <Chip key={t} token={t} i={i} onClick={() => setFocus(i)} glow={focus === i} />
          ))}
        </div>
      </div>
      <div>
        <SectionLabel>raw score = Q[{TOKENS[focus]}] · K[token] / √{D_K}</SectionLabel>
        {TOKENS.map((t, j) => {
          const raw = RAW_SCORES[focus][j];
          const absMax = Math.max(...RAW_SCORES[focus].map(Math.abs), 0.01);
          const pct = Math.abs(raw) / absMax;
          return (
            <div key={t} style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
              <Chip token={t} i={j} />
              <div style={{ flex: 1 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                  <div style={{ flex: 1, background: "#f0f0f0", borderRadius: 4, height: 12, overflow: "hidden" }}>
                    <div style={{
                      width: `${pct * 100}%`, height: "100%", borderRadius: 4,
                      background: raw < 0 ? "#f87171" : TOKEN_COLORS[focus],
                      transition: "width 0.4s ease",
                    }} />
                  </div>
                  <span style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 11, color: "#555", width: 48 }}>
                    {raw.toFixed(3)}
                  </span>
                </div>
                <div style={{ fontSize: 9, color: "#bbb", fontFamily: "'IBM Plex Mono', monospace", marginTop: 2 }}>
                  [{Q[focus].map(x => x.toFixed(2)).join(", ")}] · [{K[j].map(x => x.toFixed(2)).join(", ")}]
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ── Stage 4: Softmax + causal masking ─────────────────────────────────────────
function StageSoftmax() {
  const [focus, setFocus] = useState(2);
  const [causal, setCausal] = useState(false);
  const weights = causal ? WEIGHTS_CAUSAL[focus] : WEIGHTS_FULL[focus];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      <Formula>a = softmax(scores)    weights sum to 1.0</Formula>

      <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
        <button onClick={() => setCausal(false)} style={{
          flex: 1, padding: "8px", borderRadius: 8, border: "1.5px solid",
          borderColor: !causal ? "#1a1a1a" : "#ddd",
          background: !causal ? "#1a1a1a" : "#fff",
          color: !causal ? "#fff" : "#888",
          fontFamily: "'IBM Plex Mono', monospace", fontSize: 11, cursor: "pointer",
        }}>bidirectional</button>
        <button onClick={() => setCausal(true)} style={{
          flex: 1, padding: "8px", borderRadius: 8, border: "1.5px solid",
          borderColor: causal ? "#1a1a1a" : "#ddd",
          background: causal ? "#1a1a1a" : "#fff",
          color: causal ? "#fff" : "#888",
          fontFamily: "'IBM Plex Mono', monospace", fontSize: 11, cursor: "pointer",
        }}>causal mask ◣</button>
      </div>

      {causal && (
        <InfoBox color="#fef9c3" border="#fbbf24">
          During generation, token <em>i</em> can only attend to tokens ≤ <em>i</em>. Future tokens are masked to −∞ before softmax → probability becomes 0. This is what prevents the model from "cheating."
        </InfoBox>
      )}

      <div>
        <SectionLabel>token: </SectionLabel>
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
          {TOKENS.map((t, i) => (
            <Chip key={t} token={t} i={i} onClick={() => setFocus(i)} glow={focus === i}
              dim={causal && i > focus}
            />
          ))}
        </div>
      </div>

      <div>
        <SectionLabel>attention weights (sum = {weights.reduce((a,b)=>a+b,0).toFixed(3)})</SectionLabel>
        {TOKENS.map((t, j) => {
          const w = weights[j];
          const masked = causal && j > focus;
          return (
            <div key={t} style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8, opacity: masked ? 0.25 : 1, transition: "opacity 0.3s" }}>
              <Chip token={t} i={j} />
              <div style={{ flex: 1, background: "#f0f0f0", borderRadius: 6, height: 16, overflow: "hidden" }}>
                <div style={{
                  width: `${w * 100}%`, height: "100%", borderRadius: 6,
                  background: `linear-gradient(90deg, ${TOKEN_COLORS[focus]}, ${TOKEN_COLORS[j]})`,
                  transition: "width 0.5s cubic-bezier(0.4,0,0.2,1)",
                }} />
              </div>
              <span style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 11, color: masked ? "#ccc" : "#555", width: 44, textAlign: "right" }}>
                {masked ? "masked" : `${(w * 100).toFixed(1)}%`}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ── Stage 5: Weighted sum → output ────────────────────────────────────────────
function StageOutput() {
  const [focus, setFocus] = useState(2);
  const [causal, setCausal] = useState(false);
  const weights = causal ? WEIGHTS_CAUSAL[focus] : WEIGHTS_FULL[focus];
  const output  = causal ? OUTPUT_CAUSAL[focus]  : OUTPUT_FULL[focus];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      <Formula>output_i = Σ_j  a_ij · V_j</Formula>
      <InfoBox color="#f0fdf4" border="#86efac">
        The output is a <em>real weighted average</em> of Value vectors. Tokens you attend to heavily contribute more to your new representation.
      </InfoBox>

      <div style={{ display: "flex", gap: 8 }}>
        <button onClick={() => setCausal(false)} style={{
          flex: 1, padding: "7px", borderRadius: 8, border: "1.5px solid",
          borderColor: !causal ? "#1a1a1a" : "#ddd", background: !causal ? "#1a1a1a" : "#fff",
          color: !causal ? "#fff" : "#888", fontFamily: "'IBM Plex Mono', monospace", fontSize: 11, cursor: "pointer",
        }}>bidirectional</button>
        <button onClick={() => setCausal(true)} style={{
          flex: 1, padding: "7px", borderRadius: 8, border: "1.5px solid",
          borderColor: causal ? "#1a1a1a" : "#ddd", background: causal ? "#1a1a1a" : "#fff",
          color: causal ? "#fff" : "#888", fontFamily: "'IBM Plex Mono', monospace", fontSize: 11, cursor: "pointer",
        }}>causal ◣</button>
      </div>

      <div>
        <SectionLabel>token</SectionLabel>
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
          {TOKENS.map((t, i) => (
            <Chip key={t} token={t} i={i} onClick={() => setFocus(i)} glow={focus === i}
              dim={causal && i > focus} />
          ))}
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr auto 1fr", gap: 12, alignItems: "center" }}>
        <div>
          <SectionLabel>weighted values</SectionLabel>
          {TOKENS.map((t, j) => {
            const w = causal && j > focus ? 0 : weights[j];
            return (
              <div key={t} style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 6 }}>
                <Chip token={t} i={j} label={`×${w.toFixed(2)}`} dim={w < 0.001} />
              </div>
            );
          })}
        </div>
        <div style={{ fontSize: 28, color: "#ccc", textAlign: "center" }}>→</div>
        <div>
          <SectionLabel>output for "{TOKENS[focus]}"</SectionLabel>
          <VecDisplay vec={output} color={TOKEN_TEXT[focus]} />
          <div style={{ fontSize: 10, color: "#aaa", marginTop: 6, fontFamily: "Georgia, serif" }}>
            Compare to embedding: context has changed this token's representation.
          </div>
          <div style={{ marginTop: 8 }}>
            <SectionLabel>original embedding</SectionLabel>
            <VecDisplay vec={EMBEDDINGS[focus]} color="#bbb" />
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Stage 6: Attention heatmap (full matrix) ──────────────────────────────────
function StageHeatmap() {
  const [causal, setCausal] = useState(false);
  const weights = causal ? WEIGHTS_CAUSAL : WEIGHTS_FULL;
  const [hover, setHover] = useState(null);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      <Formula>A[i, j] = attention weight from token i → token j</Formula>
      <div style={{ display: "flex", gap: 8 }}>
        <button onClick={() => setCausal(false)} style={{
          flex: 1, padding: "7px", borderRadius: 8, border: "1.5px solid",
          borderColor: !causal ? "#1a1a1a" : "#ddd", background: !causal ? "#1a1a1a" : "#fff",
          color: !causal ? "#fff" : "#888", fontFamily: "'IBM Plex Mono', monospace", fontSize: 11, cursor: "pointer",
        }}>bidirectional</button>
        <button onClick={() => setCausal(true)} style={{
          flex: 1, padding: "7px", borderRadius: 8, border: "1.5px solid",
          borderColor: causal ? "#1a1a1a" : "#ddd", background: causal ? "#1a1a1a" : "#fff",
          color: causal ? "#fff" : "#888", fontFamily: "'IBM Plex Mono', monospace", fontSize: 11, cursor: "pointer",
        }}>causal ◣</button>
      </div>

      <div style={{ overflowX: "auto" }}>
        <div style={{ display: "inline-flex", flexDirection: "column", gap: 0 }}>
          {/* column header */}
          <div style={{ display: "flex", marginLeft: 52 }}>
            {TOKENS.map((t, j) => (
              <div key={j} style={{
                width: 52, textAlign: "center", fontSize: 10, color: "#aaa",
                fontFamily: "'IBM Plex Mono', monospace", padding: "0 2px 4px",
              }}>{t}</div>
            ))}
            <div style={{ width: 80, fontSize: 9, color: "#bbb", fontFamily: "Georgia, serif", paddingLeft: 8, display: "flex", alignItems: "flex-end" }}>
              ← attends to
            </div>
          </div>
          {weights.map((row, i) => (
            <div key={i} style={{ display: "flex", alignItems: "center" }}>
              <div style={{
                width: 52, fontSize: 10, color: TOKEN_TEXT[i], fontFamily: "'IBM Plex Mono', monospace",
                fontWeight: 700, textAlign: "right", paddingRight: 8,
              }}>{TOKENS[i]}</div>
              {row.map((w, j) => {
                const masked = causal && j > i;
                return (
                  <div key={j}
                    onMouseEnter={() => setHover([i, j])}
                    onMouseLeave={() => setHover(null)}
                    style={{
                      width: 52, height: 44, display: "flex", alignItems: "center", justifyContent: "center",
                      background: masked ? "#f5f5f5" : heatColor(w),
                      color: masked ? "#ccc" : textOnHeat(w),
                      fontFamily: "'IBM Plex Mono', monospace", fontSize: 10,
                      cursor: "default", border: "1px solid #fff",
                      transition: "background 0.4s",
                      position: "relative",
                    }}>
                    {masked ? "—" : (w * 100).toFixed(0) + "%"}
                    {hover && hover[0] === i && hover[1] === j && !masked && (
                      <div style={{
                        position: "absolute", top: -28, left: "50%", transform: "translateX(-50%)",
                        background: "#1a1a1a", color: "#fff", borderRadius: 4, padding: "3px 7px",
                        fontSize: 9, whiteSpace: "nowrap", zIndex: 10,
                        fontFamily: "'IBM Plex Mono', monospace",
                      }}>
                        "{TOKENS[i]}" → "{TOKENS[j]}" = {(w * 100).toFixed(1)}%
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          ))}
        </div>
      </div>

      <InfoBox>
        Rows = query token (who's asking). Columns = key token (who's answering). Each row sums to 100%.
        {causal && " The ◣ triangle pattern is the causal mask — future tokens are zeroed out."}
      </InfoBox>
    </div>
  );
}

// ── Stage 7: KV cache mechanics ──────────────────────────────────────────────
function StageKVCache() {
  const [step, setStep] = useState(0);

  const cacheSize = (n) => n * D_K * 2; // K + V, each D_K floats
  const recomputeWithout = (n) => n * n; // naive: n² per step
  const recomputeWith = (n) => n;        // cached: 1 new token per step

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      <Formula>cache K_j, V_j for j &lt; i — reuse on every new token</Formula>

      <div style={{ display: "flex", gap: 8, justifyContent: "center" }}>
        {TOKENS.map((t, i) => (
          <Chip key={i} token={t} i={i}
            glow={i === step}
            label={i < step ? "cached" : i === step ? "new" : ""}
            dim={i > step}
          />
        ))}
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
        <div style={{ background: "#f8f8f8", borderRadius: 10, padding: 12, border: "1px solid #eee" }}>
          <SectionLabel>K cache (keys)</SectionLabel>
          {TOKENS.slice(0, step + 1).map((t, i) => (
            <div key={i} style={{
              display: "flex", alignItems: "center", gap: 6, marginBottom: 4,
              background: i === step ? "#fff3cd" : TOKEN_COLORS[i] + "55",
              borderRadius: 6, padding: "4px 8px",
              border: i === step ? "1px solid #fbbf24" : "none",
              transition: "all 0.3s",
            }}>
              <span style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 10, color: TOKEN_TEXT[i], fontWeight: 700, width: 24 }}>{t}</span>
              {K[i].map((v, j) => (
                <div key={j} title={v.toFixed(3)} style={{
                  flex: 1, height: 8, borderRadius: 2,
                  background: i === step ? "#fbbf24" : TOKEN_TEXT[i] + "66",
                  opacity: 0.5 + Math.abs(v) * 0.5,
                }} />
              ))}
              <span style={{ fontSize: 8, color: i === step ? "#92400e" : "#aaa", fontFamily: "'IBM Plex Mono', monospace", width: 28 }}>
                {i === step ? "NEW" : "HIT"}
              </span>
            </div>
          ))}
        </div>
        <div style={{ background: "#f8f8f8", borderRadius: 10, padding: 12, border: "1px solid #eee" }}>
          <SectionLabel>V cache (values)</SectionLabel>
          {TOKENS.slice(0, step + 1).map((t, i) => (
            <div key={i} style={{
              display: "flex", alignItems: "center", gap: 6, marginBottom: 4,
              background: i === step ? "#fff3cd" : TOKEN_COLORS[i] + "55",
              borderRadius: 6, padding: "4px 8px",
              border: i === step ? "1px solid #fbbf24" : "none",
              transition: "all 0.3s",
            }}>
              <span style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 10, color: TOKEN_TEXT[i], fontWeight: 700, width: 24 }}>{t}</span>
              {V[i].map((v, j) => (
                <div key={j} title={v.toFixed(3)} style={{
                  flex: 1, height: 8, borderRadius: 2,
                  background: i === step ? "#fbbf24" : TOKEN_TEXT[i] + "66",
                  opacity: 0.5 + Math.abs(v) * 0.5,
                }} />
              ))}
              <span style={{ fontSize: 8, color: i === step ? "#92400e" : "#aaa", fontFamily: "'IBM Plex Mono', monospace", width: 28 }}>
                {i === step ? "NEW" : "HIT"}
              </span>
            </div>
          ))}
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
        <div style={{ background: "#f0fdf4", border: "1px solid #86efac", borderRadius: 8, padding: "10px 12px" }}>
          <div style={{ fontSize: 10, color: "#166534", fontFamily: "'IBM Plex Mono', monospace", marginBottom: 4 }}>with cache</div>
          <div style={{ fontSize: 20, fontFamily: "'IBM Plex Mono', monospace", fontWeight: 700, color: "#15803d" }}>
            {recomputeWith(step + 1)} op{recomputeWith(step + 1) !== 1 ? "s" : ""}
          </div>
          <div style={{ fontSize: 10, color: "#166534" }}>compute K,V for 1 new token only</div>
        </div>
        <div style={{ background: "#fef2f2", border: "1px solid #fca5a5", borderRadius: 8, padding: "10px 12px" }}>
          <div style={{ fontSize: 10, color: "#991b1b", fontFamily: "'IBM Plex Mono', monospace", marginBottom: 4 }}>without cache</div>
          <div style={{ fontSize: 20, fontFamily: "'IBM Plex Mono', monospace", fontWeight: 700, color: "#dc2626" }}>
            {recomputeWithout(step + 1)} ops
          </div>
          <div style={{ fontSize: 10, color: "#991b1b" }}>recompute all {step + 1} tokens every step</div>
        </div>
      </div>

      <InfoBox color="#fef9c3" border="#fbbf24">
        Cache grows by 2 × d_k floats per token per layer per head. At 32k context, 32 layers, 32 heads, d_k=128 in fp16 → ~8GB just for KV. This is why EVOKE's eviction and recompute-via-RoPE matters.
      </InfoBox>

      <div style={{ display: "flex", gap: 10, justifyContent: "center" }}>
        <button onClick={() => setStep(s => Math.max(0, s - 1))} disabled={step === 0}
          style={{
            background: "#fff", border: "1px solid #ddd", borderRadius: 8,
            padding: "8px 20px", cursor: step === 0 ? "not-allowed" : "pointer",
            fontFamily: "'IBM Plex Mono', monospace", fontSize: 12, opacity: step === 0 ? 0.4 : 1,
          }}>← prev token</button>
        <button onClick={() => setStep(s => Math.min(TOKENS.length - 1, s + 1))} disabled={step === TOKENS.length - 1}
          style={{
            background: "#1a1a1a", color: "#fff", border: "none", borderRadius: 8,
            padding: "8px 20px", cursor: step === TOKENS.length - 1 ? "not-allowed" : "pointer",
            fontFamily: "'IBM Plex Mono', monospace", fontSize: 12, opacity: step === TOKENS.length - 1 ? 0.4 : 1,
          }}>generate next →</button>
      </div>
    </div>
  );
}

// ── App shell ─────────────────────────────────────────────────────────────────
const STAGES = [
  { id: "embed",    title: "1. Embeddings",     subtitle: "words → vectors",             Comp: StageEmbeddings },
  { id: "qkv",     title: "2. Q K V",           subtitle: "three projections per token", Comp: StageQKV        },
  { id: "scores",  title: "3. Dot Products",    subtitle: "Q·K / √d_k",                 Comp: StageRawScores  },
  { id: "softmax", title: "4. Softmax + Mask",  subtitle: "weights that sum to 1",       Comp: StageSoftmax    },
  { id: "output",  title: "5. Output",          subtitle: "weighted sum of V",           Comp: StageOutput     },
  { id: "heatmap", title: "6. Full Matrix",     subtitle: "all attention at once",       Comp: StageHeatmap    },
  { id: "kvcache", title: "7. KV Cache",        subtitle: "don't recompute old tokens",  Comp: StageKVCache    },
];

export default function App() {
  const [stage, setStage] = useState(0);
  const { Comp, title, subtitle } = STAGES[stage];

  return (
    <div style={{ minHeight: "100vh", background: "#fafaf8", fontFamily: "Georgia, serif", display: "flex", flexDirection: "column", alignItems: "center", padding: "28px 12px 48px" }}>
      <style>{`@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&display=swap');`}</style>

      <div style={{ textAlign: "center", marginBottom: 24 }}>
        <div style={{ fontSize: 10, letterSpacing: 3, color: "#bbb", textTransform: "uppercase", marginBottom: 4 }}>real math, real computation</div>
        <h1 style={{ fontSize: 22, fontWeight: 400, color: "#1a1a1a", margin: 0 }}>Attention & KV Cache</h1>
        <div style={{ fontSize: 12, color: "#aaa", marginTop: 4 }}>d_model = {D_MODEL}, d_k = {D_K}, 5 tokens, 1 head</div>
      </div>

      {/* Stage tabs — scrollable on mobile */}
      <div style={{ display: "flex", gap: 5, marginBottom: 20, overflowX: "auto", maxWidth: "100%", paddingBottom: 4 }}>
        {STAGES.map((s, i) => (
          <button key={s.id} onClick={() => setStage(i)} style={{
            background: stage === i ? "#1a1a1a" : "#fff",
            color: stage === i ? "#fff" : "#888",
            border: `1px solid ${stage === i ? "#1a1a1a" : "#ddd"}`,
            borderRadius: 20, padding: "5px 12px",
            fontFamily: "'IBM Plex Mono', monospace", fontSize: 10,
            cursor: "pointer", whiteSpace: "nowrap", transition: "all 0.2s",
          }}>{s.title}</button>
        ))}
      </div>

      {/* Card */}
      <div style={{ width: "100%", maxWidth: 580, background: "#fff", border: "1px solid #ebebeb", borderRadius: 16, padding: "22px 18px", boxShadow: "0 2px 24px rgba(0,0,0,0.05)" }}>
        <div style={{ marginBottom: 18 }}>
          <div style={{ fontSize: 10, color: "#aaa", letterSpacing: 2, textTransform: "uppercase", marginBottom: 3, fontFamily: "'IBM Plex Mono', monospace" }}>{subtitle}</div>
          <h2 style={{ fontSize: 17, margin: 0, fontWeight: 500, color: "#1a1a1a" }}>{title}</h2>
        </div>
        <div style={{ borderTop: "1px solid #f0f0f0", paddingTop: 18 }}>
          <Comp />
        </div>
      </div>

      {/* Nav */}
      <div style={{ display: "flex", gap: 10, marginTop: 18 }}>
        <button onClick={() => setStage(s => Math.max(0, s - 1))} disabled={stage === 0}
          style={{ background: "#fff", border: "1px solid #ddd", borderRadius: 10, padding: "9px 20px", fontFamily: "Georgia, serif", fontSize: 13, cursor: stage === 0 ? "not-allowed" : "pointer", color: "#666", opacity: stage === 0 ? 0.4 : 1 }}>← back</button>
        <button onClick={() => setStage(s => Math.min(STAGES.length - 1, s + 1))} disabled={stage === STAGES.length - 1}
          style={{ background: "#1a1a1a", color: "#fff", border: "none", borderRadius: 10, padding: "9px 24px", fontFamily: "Georgia, serif", fontSize: 13, cursor: stage === STAGES.length - 1 ? "not-allowed" : "pointer", opacity: stage === STAGES.length - 1 ? 0.4 : 1 }}>next →</button>
      </div>
    </div>
  );
}
