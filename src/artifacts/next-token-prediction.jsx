import { useState, useEffect, useMemo, useRef, useCallback } from "react";

export const meta = {
  title: "Next-Token Prediction & Sampling",
  category: "Machine Learning",
  description:
    "A language model writes one token at a time, drawing each from a probability distribution and feeding its own choice back in. Turn the temperature, top-k, and top-p knobs and watch the same distribution reshape under your hands.",
  date: "2026-03-08",
  tags: ["language-models", "sampling", "softmax", "autoregressive"],
};

const C = {
  bg: "#f7f5f2",
  card: "#ffffff",
  border: "#e4dfd8",
  ink: "#1c1a17",
  muted: "#8c8278",
  accent: "#c0561f",
  accentL: "#f6ece5",
  green: "#2e7d51",
  greenL: "#e4f2eb",
  blue: "#2a5298",
  blueL: "#e5ecf8",
  gold: "#9a7020",
  goldL: "#f5edd8",
  faint: "#efeae3",
};

const MONO = "'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, monospace";
const SERIF = "'Palatino Linotype', 'Book Antiqua', Palatino, Georgia, serif";

// mulberry32 so that a given seed reproduces the same draws without touching Math.random.
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

const CORPORA = {
  weather: {
    label: "weather log",
    text: `the sky was grey and the wind was cold this morning. by noon the sun broke through the clouds and the air grew warm. a light rain fell in the afternoon and the streets turned wet and dark. the wind picked up again at dusk and the clouds rolled in from the west. the night was clear and cold and the stars came out over the quiet town. in the morning a thick fog covered the river and the hills beyond. the sun rose slow and pale behind the fog and the day stayed cool. clouds gathered through the afternoon and a storm came in from the sea. rain fell hard against the windows and the wind shook the trees. by night the storm passed and the sky cleared and the moon rose bright and full.`,
  },
  fable: {
    label: "short fable",
    text: `a fox saw a crow with a piece of cheese in its beak. the fox wanted the cheese and so the fox spoke kind words to the crow. your feathers shine and your eyes are bright said the fox to the crow. if your voice is as fine as your form you are the queen of all the birds. the crow was proud and opened its beak to sing a song. the cheese fell from the beak and the fox caught it on the ground. the fox ate the cheese and looked up at the crow with a grin. do not trust the words of those who only want what you hold said the fox. the crow flew off in shame and the fox went home well fed and pleased.`,
  },
  recipe: {
    label: "plain recipe",
    text: `take a cup of flour and a pinch of salt and mix them in a wide bowl. add a spoon of oil and a little water and stir until the dough comes together. knead the dough on a clean board for a few minutes until it turns smooth. cover the dough with a cloth and let it rest in a warm place for an hour. roll the dough into a thin round sheet and cut it into even strips. heat a pan over a medium flame and add a thin layer of oil to the pan. cook each strip until it turns golden brown on both sides and lift it out. sprinkle a little salt over the warm strips and serve them on a plain plate.`,
  },
};

function tokenize(text) {
  return text
    .toLowerCase()
    .replace(/[^a-z. ]/g, " ")
    .replace(/\./g, " . ")
    .split(/\s+/)
    .filter(Boolean);
}

const START = "▁start";

function trainTrigram(text) {
  const tokens = tokenize(text);
  const vocabSet = new Set(tokens);
  const vocab = [...vocabSet].sort();
  const vocabIndex = new Map(vocab.map((w, i) => [w, i]));

  const padded = [START, START, ...tokens];
  const counts = new Map();
  for (let i = 2; i < padded.length; i++) {
    const key = padded[i - 2] + "" + padded[i - 1];
    let row = counts.get(key);
    if (!row) {
      row = new Map();
      counts.set(key, row);
    }
    const w = padded[i];
    row.set(w, (row.get(w) || 0) + 1);
  }

  return { vocab, vocabIndex, counts, tokens };
}

// add-alpha (Laplace) smoothing so that words unseen after this context keep a small nonzero mass.
function distributionFor(model, context, alpha = 0.05) {
  const { vocab, counts } = model;
  const a = context.length >= 1 ? context[context.length - 1] : START;
  const b = context.length >= 2 ? context[context.length - 2] : START;
  const key = b + "" + a;
  const row = counts.get(key) || new Map();

  let total = 0;
  for (const v of row.values()) total += v;
  const denom = total + alpha * vocab.length;

  const probs = vocab.map((w) => ((row.get(w) || 0) + alpha) / denom);
  return { tokens: vocab, probs };
}

// Treating each log-prob as a logit and dividing by T before softmax is the standard temperature
// rescaling: T below 1 sharpens toward the argmax, T above 1 flattens toward uniform.
function softmaxFromProbs(probs, temperature) {
  const t = Math.max(temperature, 1e-3);
  const logits = probs.map((p) => Math.log(Math.max(p, 1e-12)) / t);
  const m = Math.max(...logits);
  const exps = logits.map((l) => Math.exp(l - m));
  const s = exps.reduce((acc, v) => acc + v, 0);
  return exps.map((v) => v / s);
}

function applyTopK(items, k) {
  if (k <= 0 || k >= items.length) return items.map((it) => ({ ...it, kept: true }));
  const ranked = [...items].sort((x, y) => y.p - x.p);
  const keep = new Set(ranked.slice(0, k).map((it) => it.token));
  return items.map((it) => ({ ...it, kept: keep.has(it.token) }));
}

function applyTopP(items, p) {
  if (p >= 1) return items.map((it) => ({ ...it, kept: true }));
  const ranked = [...items].sort((x, y) => y.p - x.p);
  const keep = new Set();
  let cum = 0;
  for (const it of ranked) {
    keep.add(it.token);
    cum += it.p;
    if (cum >= p) break;
  }
  return items.map((it) => ({ ...it, kept: keep.has(it.token) }));
}

// invariant: the kept tokens are renormalized so that final probabilities sum to one before sampling.
function buildCandidates(model, context, { temperature, topK, topP }) {
  const dist = distributionFor(model, context);
  const base = dist.tokens.map((token, i) => ({ token, base: dist.probs[i] }));
  const tempP = softmaxFromProbs(
    base.map((b) => b.base),
    temperature
  );
  let items = base.map((b, i) => ({ token: b.token, base: b.base, p: tempP[i] }));

  items = applyTopK(items, topK);
  items = items.map((it) => ({ ...it, keptK: it.kept }));
  items = applyTopP(items, topP);
  items = items.map((it) => ({ ...it, kept: it.kept && it.keptK }));

  let keptSum = 0;
  for (const it of items) if (it.kept) keptSum += it.p;
  items = items.map((it) => ({
    ...it,
    final: it.kept && keptSum > 0 ? it.p / keptSum : 0,
  }));

  return items.sort((x, y) => y.base - x.base);
}

function sampleFrom(items, rand) {
  const r = rand();
  let cum = 0;
  for (const it of items) {
    if (!it.kept) continue;
    cum += it.final;
    if (r <= cum) return it.token;
  }
  for (let i = items.length - 1; i >= 0; i--) if (items[i].kept) return items[i].token;
  return items[0].token;
}

function fmtToken(t) {
  if (t === ".") return ".";
  return t;
}

function joinTokens(tokens) {
  let out = "";
  for (const t of tokens) {
    if (t === ".") out += ".";
    else out += (out.length ? " " : "") + t;
  }
  return out;
}

function Label({ children, color = C.muted }) {
  return (
    <div
      style={{
        fontSize: 10,
        letterSpacing: "0.14em",
        textTransform: "uppercase",
        color,
        fontFamily: MONO,
        marginBottom: 8,
      }}
    >
      {children}
    </div>
  );
}

function Slider({ label, value, min, max, step, onChange, format, color, hint }) {
  const id = useMemo(() => "sl-" + Math.random().toString(36).slice(2, 8), []);
  return (
    <div>
      <label
        htmlFor={id}
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "baseline",
          fontSize: 12,
          color: C.ink,
          marginBottom: 6,
          fontFamily: SERIF,
        }}
      >
        <span>{label}</span>
        <span style={{ fontFamily: MONO, fontWeight: 700, color }}>{format(value)}</span>
      </label>
      <input
        id={id}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(+e.target.value)}
        style={{ width: "100%", accentColor: color, cursor: "pointer" }}
      />
      {hint && (
        <div style={{ fontSize: 10.5, color: C.muted, marginTop: 3, fontFamily: SERIF }}>
          {hint}
        </div>
      )}
    </div>
  );
}

function Button({ children, onClick, variant = "ghost", disabled, title }) {
  const base = {
    fontFamily: MONO,
    fontSize: 12,
    borderRadius: 9,
    padding: "9px 16px",
    cursor: disabled ? "not-allowed" : "pointer",
    transition:
      "transform 140ms cubic-bezier(0.23,1,0.32,1), background 140ms ease, border-color 140ms ease",
    opacity: disabled ? 0.45 : 1,
    border: "1.5px solid",
    userSelect: "none",
  };
  const styles =
    variant === "solid"
      ? { ...base, background: C.accent, color: "#fff", borderColor: C.accent }
      : variant === "dark"
      ? { ...base, background: C.ink, color: "#fff", borderColor: C.ink }
      : { ...base, background: C.card, color: C.ink, borderColor: C.border };
  const ref = useRef(null);
  return (
    <button
      ref={ref}
      className="ntp-focus"
      title={title}
      disabled={disabled}
      onClick={onClick}
      onPointerDown={() => {
        if (!disabled && ref.current) ref.current.style.transform = "scale(0.97)";
      }}
      onPointerUp={() => {
        if (ref.current) ref.current.style.transform = "scale(1)";
      }}
      onPointerLeave={() => {
        if (ref.current) ref.current.style.transform = "scale(1)";
      }}
      style={styles}
    >
      {children}
    </button>
  );
}

function DistributionBars({ candidates, picked, mode, onPick, reducedMotion }) {
  const top = candidates.slice(0, 12);
  const maxBase = Math.max(...top.map((c) => c.base), 1e-6);
  const trans = reducedMotion
    ? "background 120ms ease"
    : "width 360ms cubic-bezier(0.23,1,0.32,1), background 160ms ease, opacity 200ms ease";
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 7 }}>
      {top.map((c) => {
        const dropped = !c.kept;
        const isPicked = c.token === picked;
        const basePct = (c.base / maxBase) * 100;
        const finalPct = c.kept ? c.final * 100 : 0;
        const pickable = mode === "hand" && c.kept;
        return (
          <div
            key={c.token}
            className={pickable ? "ntp-focus" : undefined}
            onClick={pickable ? () => onPick(c.token) : undefined}
            role={pickable ? "button" : undefined}
            tabIndex={pickable ? 0 : undefined}
            onKeyDown={
              pickable
                ? (e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      onPick(c.token);
                    }
                  }
                : undefined
            }
            aria-label={
              pickable
                ? `Pick token ${fmtToken(c.token)}, probability ${(c.final * 100).toFixed(
                    1
                  )} percent`
                : undefined
            }
            style={{
              display: "grid",
              gridTemplateColumns: "minmax(58px, 84px) 1fr auto",
              alignItems: "center",
              gap: 10,
              padding: "3px 6px",
              borderRadius: 8,
              cursor: pickable ? "pointer" : "default",
              background: isPicked ? C.accentL : "transparent",
              outline: isPicked ? `1.5px solid ${C.accent}` : "none",
              opacity: dropped ? 0.4 : 1,
              transition: trans,
            }}
          >
            <span
              style={{
                fontFamily: MONO,
                fontSize: 12.5,
                fontWeight: isPicked ? 700 : 500,
                color: dropped ? C.muted : C.ink,
                whiteSpace: "nowrap",
                overflow: "hidden",
                textOverflow: "ellipsis",
                textDecoration: dropped ? "line-through" : "none",
              }}
            >
              {fmtToken(c.token)}
            </span>
            <div
              style={{
                position: "relative",
                height: 16,
                background: C.faint,
                borderRadius: 5,
                overflow: "hidden",
              }}
            >
              <div
                style={{
                  position: "absolute",
                  inset: 0,
                  width: `${basePct}%`,
                  background: C.border,
                  borderRadius: 5,
                  transition: trans,
                }}
                title="base probability"
              />
              <div
                style={{
                  position: "absolute",
                  inset: 0,
                  width: `${finalPct}%`,
                  background: isPicked ? C.accent : dropped ? "transparent" : C.blue,
                  borderRadius: 5,
                  transition: trans,
                }}
                title="probability after the knobs"
              />
            </div>
            <span
              style={{
                fontFamily: MONO,
                fontSize: 11,
                color: dropped ? C.muted : C.ink,
                width: 54,
                textAlign: "right",
              }}
            >
              {dropped ? "cut" : `${(c.final * 100).toFixed(1)}%`}
            </span>
          </div>
        );
      })}
    </div>
  );
}

export default function App() {
  const [corpusKey, setCorpusKey] = useState("weather");
  const [temperature, setTemperature] = useState(0.9);
  const [topK, setTopK] = useState(0);
  const [topP, setTopP] = useState(1);
  const [mode, setMode] = useState("sample");
  const [seed, setSeed] = useState(7);
  const [context, setContext] = useState([]);
  const [picked, setPicked] = useState(null);
  const [auto, setAuto] = useState(false);

  const reducedMotion =
    typeof window !== "undefined" &&
    window.matchMedia &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const model = useMemo(() => trainTrigram(CORPORA[corpusKey].text), [corpusKey]);

  const drawRef = useRef(mulberry32(seed));
  useEffect(() => {
    drawRef.current = mulberry32(seed);
  }, [seed, corpusKey]);

  const candidates = useMemo(
    () => buildCandidates(model, context, { temperature, topK, topP }),
    [model, context, temperature, topK, topP]
  );

  const keptCount = candidates.filter((c) => c.kept).length;
  const topToken = candidates[0];

  const reset = useCallback(() => {
    setContext([]);
    setPicked(null);
    setAuto(false);
    drawRef.current = mulberry32(seed);
  }, [seed]);

  const stepOnce = useCallback(() => {
    const cands = buildCandidates(model, context, { temperature, topK, topP });
    const tok = sampleFrom(cands, drawRef.current);
    setPicked(tok);
    setContext((c) => [...c, tok]);
  }, [model, context, temperature, topK, topP]);

  const pickByHand = useCallback((tok) => {
    setPicked(tok);
    setContext((c) => [...c, tok]);
  }, []);

  useEffect(() => {
    if (!auto) return;
    if (context.length >= 60) {
      setAuto(false);
      return;
    }
    const delay = reducedMotion ? 220 : 380;
    const id = setTimeout(() => {
      const cands = buildCandidates(model, context, { temperature, topK, topP });
      const tok = sampleFrom(cands, drawRef.current);
      setPicked(tok);
      setContext((c) => [...c, tok]);
    }, delay);
    return () => clearTimeout(id);
  }, [auto, context, model, temperature, topK, topP, reducedMotion]);

  const tail = context.slice(-2);
  const contextLabel =
    tail.length === 0
      ? `${START} ${START}`
      : tail.length === 1
      ? `${START} ${fmtToken(tail[0])}`
      : `${fmtToken(tail[0])} ${fmtToken(tail[1])}`;

  return (
    <div
      style={{
        fontFamily: SERIF,
        background: C.bg,
        minHeight: "100vh",
        color: C.ink,
        padding: "26px 14px 56px",
      }}
    >
      <style>{`@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&display=swap');
        .ntp-focus:focus-visible { outline: 2px solid ${C.accent}; outline-offset: 2px; border-radius: 8px; }
      `}</style>

      <div style={{ maxWidth: 720, margin: "0 auto" }}>
        <header style={{ marginBottom: 22 }}>
          <div
            style={{
              fontSize: 10,
              letterSpacing: "0.16em",
              textTransform: "uppercase",
              color: C.muted,
              fontFamily: MONO,
              marginBottom: 7,
            }}
          >
            Machine Learning
          </div>
          <h1
            style={{
              fontSize: 27,
              fontWeight: 700,
              margin: 0,
              letterSpacing: "-0.01em",
              textWrap: "balance",
            }}
          >
            Next-Token Prediction & Sampling
          </h1>
          <p
            style={{
              color: C.ink,
              fontSize: 14.5,
              lineHeight: 1.62,
              margin: "12px 0 0",
              maxWidth: "64ch",
              textWrap: "pretty",
            }}
          >
            A language model never plans a sentence. It looks at the running text, produces a
            probability over every word that could come next, picks one, appends it, and then reads
            its own output back in to predict again. The knobs people argue about, temperature,
            top-k, and top-p, are just different ways of choosing from that one distribution. Run the
            loop by hand below and feel each knob bend the odds.
          </p>
        </header>

        <section
          style={{
            background: C.accentL,
            border: `1px solid ${C.accent}33`,
            borderRadius: 12,
            padding: "14px 16px",
            marginBottom: 18,
            fontSize: 12.5,
            lineHeight: 1.6,
            color: C.ink,
          }}
        >
          <strong style={{ color: C.accent }}>What the model is here.</strong> A real language model
          is a large neural network, far too heavy to ship into a browser. So this page trains a
          small trigram model on the corpus below: it counts how often each word follows the two
          words before it, smooths the counts, and turns them into next-token probabilities. The
          neural net and the trigram differ enormously in how good those probabilities are, but the
          loop is identical. Predict a distribution, pick a token, append, repeat. The trigram is a
          tiny stand-in for the part you cannot see; the generation mechanism is the real thing.
        </section>

        <section
          style={{
            background: C.card,
            border: `1px solid ${C.border}`,
            borderRadius: 14,
            padding: "18px 16px",
            marginBottom: 16,
          }}
        >
          <Label>corpus</Label>
          <div style={{ display: "flex", gap: 7, flexWrap: "wrap", marginBottom: 8 }}>
            {Object.entries(CORPORA).map(([k, c]) => {
              const active = corpusKey === k;
              return (
                <button
                  key={k}
                  className="ntp-focus"
                  onClick={() => {
                    setCorpusKey(k);
                    setContext([]);
                    setPicked(null);
                    setAuto(false);
                  }}
                  style={{
                    fontFamily: MONO,
                    fontSize: 11.5,
                    padding: "6px 13px",
                    borderRadius: 20,
                    border: `1.5px solid ${active ? C.accent : C.border}`,
                    background: active ? C.accentL : C.card,
                    color: active ? C.accent : C.muted,
                    fontWeight: active ? 700 : 400,
                    cursor: "pointer",
                    transition: "all 140ms ease",
                  }}
                >
                  {c.label}
                </button>
              );
            })}
          </div>
          <div style={{ fontSize: 11.5, color: C.muted, lineHeight: 1.55, fontFamily: SERIF }}>
            {model.tokens.length} tokens trained, {model.vocab.length} distinct words. The model only
            knows what appears in this text, which is why short corpora go in circles.
          </div>
        </section>

        <section
          style={{
            background: C.card,
            border: `1px solid ${C.border}`,
            borderRadius: 14,
            padding: "18px 16px",
            marginBottom: 16,
          }}
        >
          <Label>the running text</Label>
          <div
            aria-live="polite"
            style={{
              minHeight: 76,
              background: C.bg,
              border: `1px solid ${C.border}`,
              borderRadius: 10,
              padding: "13px 15px",
              fontSize: 15.5,
              lineHeight: 1.7,
              fontFamily: SERIF,
            }}
          >
            {context.length === 0 ? (
              <span style={{ color: C.muted, fontStyle: "italic" }}>
                Empty. The context is two start markers. Sample or pick the first token to begin.
              </span>
            ) : (
              context.map((t, i) => {
                const isLast = i === context.length - 1;
                return (
                  <span
                    key={i}
                    style={{
                      background: isLast ? C.accentL : "transparent",
                      color: isLast ? C.accent : C.ink,
                      borderRadius: 4,
                      padding: isLast ? "1px 3px" : 0,
                      transition: reducedMotion ? "none" : "background 300ms ease",
                    }}
                  >
                    {t === "." ? "." : (i ? " " : "") + t}
                  </span>
                );
              })
            )}
          </div>
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              marginTop: 8,
              fontSize: 11,
              color: C.muted,
              fontFamily: MONO,
              gap: 10,
              flexWrap: "wrap",
            }}
          >
            <span>{context.length} tokens generated</span>
            <span>
              context window:{" "}
              <span style={{ color: C.ink, fontWeight: 600 }}>{contextLabel}</span> &rarr; ?
            </span>
          </div>
        </section>

        <section
          style={{
            background: C.card,
            border: `1px solid ${C.border}`,
            borderRadius: 14,
            padding: "18px 16px",
            marginBottom: 16,
          }}
        >
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "baseline",
              flexWrap: "wrap",
              gap: 6,
              marginBottom: 12,
            }}
          >
            <Label color={C.ink}>distribution over the next token</Label>
            <span style={{ fontSize: 11, color: C.muted, fontFamily: MONO }}>
              {keptCount} of {candidates.length} words kept
            </span>
          </div>

          <div style={{ display: "flex", gap: 14, marginBottom: 12, flexWrap: "wrap" }}>
            <LegendSwatch color={C.border} label="base probability (from counts)" />
            <LegendSwatch color={C.blue} label="after temperature / top-k / top-p" />
          </div>

          <DistributionBars
            candidates={candidates}
            picked={picked}
            mode={mode}
            onPick={pickByHand}
            reducedMotion={reducedMotion}
          />

          {mode === "hand" && (
            <div style={{ fontSize: 11.5, color: C.accent, marginTop: 10, fontFamily: SERIF }}>
              Pick-by-hand is on. Click any surviving word to append it yourself.
            </div>
          )}

          <div
            style={{
              marginTop: 14,
              paddingTop: 12,
              borderTop: `1px solid ${C.faint}`,
              fontSize: 11.5,
              color: C.muted,
              fontFamily: MONO,
              display: "flex",
              gap: 16,
              flexWrap: "wrap",
            }}
          >
            <span>
              top word:{" "}
              <span style={{ color: C.ink, fontWeight: 600 }}>
                {topToken ? fmtToken(topToken.token) : "—"}
              </span>{" "}
              at{" "}
              <span style={{ color: C.ink, fontWeight: 600 }}>
                {topToken ? (topToken.final * 100).toFixed(1) : "0"}%
              </span>
            </span>
            <span>
              survivors sum to{" "}
              <span style={{ color: C.ink, fontWeight: 600 }}>
                {candidates.reduce((s, c) => s + (c.kept ? c.final : 0), 0).toFixed(2)}
              </span>
            </span>
          </div>
        </section>

        <section
          style={{
            background: C.card,
            border: `1px solid ${C.border}`,
            borderRadius: 14,
            padding: "18px 16px",
            marginBottom: 16,
          }}
        >
          <Label>the knobs</Label>
          <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
            <Slider
              label="temperature"
              value={temperature}
              min={0.1}
              max={2}
              step={0.05}
              onChange={setTemperature}
              format={(v) => v.toFixed(2)}
              color={C.accent}
              hint="Divide the log-probs by T, then softmax. Below 1 sharpens toward the top word; above 1 flattens toward uniform."
            />
            <Slider
              label="top-k"
              value={topK}
              min={0}
              max={Math.min(40, model.vocab.length)}
              step={1}
              onChange={setTopK}
              format={(v) => (v === 0 ? "off" : v)}
              color={C.green}
              hint="Keep only the k most probable words, drop the rest, renormalize. Zero means keep all."
            />
            <Slider
              label="top-p (nucleus)"
              value={topP}
              min={0.1}
              max={1}
              step={0.01}
              onChange={setTopP}
              format={(v) => (v >= 1 ? "off" : v.toFixed(2))}
              color={C.blue}
              hint="Keep the smallest set of words whose probabilities add up past p, drop the long tail, renormalize."
            />
          </div>
        </section>

        <section
          style={{
            background: C.card,
            border: `1px solid ${C.border}`,
            borderRadius: 14,
            padding: "18px 16px",
            marginBottom: 16,
          }}
        >
          <Label>run the loop</Label>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 14 }}>
            <Button
              variant="solid"
              onClick={stepOnce}
              disabled={mode === "hand" || auto}
              title="Draw one token from the distribution and append it"
            >
              sample one &rarr;
            </Button>
            <Button variant="dark" onClick={() => setAuto((a) => !a)} disabled={mode === "hand"}>
              {auto ? "pause" : "auto-generate"}
            </Button>
            <Button onClick={reset}>reset</Button>
          </div>

          <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 14 }}>
            <ModeToggle mode={mode} setMode={setMode} setAuto={setAuto} />
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
            <span style={{ fontSize: 12, color: C.muted, fontFamily: MONO }}>seed</span>
            <span
              style={{
                fontFamily: MONO,
                fontSize: 13,
                fontWeight: 700,
                color: C.ink,
                minWidth: 36,
              }}
            >
              {seed}
            </span>
            <Button
              onClick={() => {
                const next = (seed * 1664525 + 1013904223) % 100000;
                setSeed(next);
              }}
              title="Reseed the sampler so the run is reproducible from a new starting point"
            >
              reseed
            </Button>
            <span style={{ fontSize: 11, color: C.muted, fontFamily: SERIF, lineHeight: 1.5 }}>
              Same seed plus same knobs plus same corpus reproduces the same text exactly.
            </span>
          </div>
        </section>

        <section
          style={{
            background: C.goldL,
            border: `1px solid ${C.gold}33`,
            borderRadius: 12,
            padding: "15px 17px",
            fontSize: 12.5,
            lineHeight: 1.65,
            color: C.ink,
          }}
        >
          <Label color={C.gold}>what the knobs trade</Label>
          <p style={{ margin: "0 0 9px" }}>
            <strong>Temperature</strong> rescales confidence. The raw scores are logits, here the log
            of each count-based probability. Dividing them by a small T before the softmax stretches
            the gaps, so the top word dominates and the text turns rigid and repetitive. A large T
            squashes the gaps toward uniform, so rare words slip in and the text drifts into nonsense.
          </p>
          <p style={{ margin: "0 0 9px" }}>
            <strong>Top-k</strong> and <strong>top-p</strong> both prune the tail before sampling.
            Top-k keeps a fixed number of candidates; top-p keeps however many it takes to cover a
            fixed share of the probability mass, so it adapts: a confident step keeps few words, an
            uncertain step keeps many. After either prune, the survivors are renormalized so they sum
            back to one.
          </p>
          <p style={{ margin: 0 }}>
            The whole game is a trade between coherence and diversity. Low temperature with tight
            pruning is safe and dull; high temperature with no pruning is surprising and often broken.
            Useful sampling lives in the middle.
          </p>
        </section>

        <footer
          style={{
            marginTop: 22,
            textAlign: "center",
            fontSize: 11,
            color: C.muted,
            fontFamily: MONO,
            lineHeight: 1.6,
          }}
        >
          trigram counts with add-0.05 smoothing &middot; mulberry32 seeded sampler &middot; the loop,
          not the model, is the point
        </footer>
      </div>
    </div>
  );
}

function LegendSwatch({ color, label }) {
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
      <span
        style={{
          width: 14,
          height: 11,
          borderRadius: 3,
          background: color,
          display: "inline-block",
        }}
      />
      <span style={{ fontSize: 10.5, color: C.muted, fontFamily: MONO }}>{label}</span>
    </span>
  );
}

function ModeToggle({ mode, setMode, setAuto }) {
  const opts = [
    { id: "sample", label: "sample mode" },
    { id: "hand", label: "pick by hand" },
  ];
  return (
    <div
      role="group"
      aria-label="generation mode"
      style={{
        display: "inline-flex",
        background: C.bg,
        border: `1px solid ${C.border}`,
        borderRadius: 10,
        padding: 3,
        gap: 3,
      }}
    >
      {opts.map((o) => {
        const active = mode === o.id;
        return (
          <button
            key={o.id}
            className="ntp-focus"
            aria-pressed={active}
            onClick={() => {
              setMode(o.id);
              setAuto(false);
            }}
            style={{
              fontFamily: MONO,
              fontSize: 11.5,
              padding: "6px 13px",
              borderRadius: 8,
              border: "none",
              background: active ? C.card : "transparent",
              color: active ? C.ink : C.muted,
              fontWeight: active ? 700 : 400,
              boxShadow: active ? "0 1px 3px rgba(0,0,0,0.08)" : "none",
              cursor: "pointer",
              transition: "all 140ms ease",
            }}
          >
            {o.label}
          </button>
        );
      })}
    </div>
  );
}
