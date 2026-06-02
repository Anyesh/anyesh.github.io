import { useState, useMemo, useEffect } from "react";

export const meta = {
  title: "Semantic Search: Meaning, Not Keywords",
  category: "Machine Learning",
  description:
    "You search for will it rain and the best answer says clear skies tomorrow, never once using your words. Watch meaning win where matching letters cannot.",
  date: "2026-02-13",
  tags: ["embeddings", "semantic-search", "cosine-similarity", "retrieval", "rag"],
};

const C = {
  bg: "#f7f5f2",
  card: "#ffffff",
  border: "#e6e1d9",
  ink: "#1b1916",
  muted: "#6f675e",
  faint: "#9b938a",
  accent: "#c0561f",
  accentSoft: "#f6ece5",
  grid: "#efebe4",
  good: "#3f7d52",
  goodSoft: "#eaf2ec",
  cool: "#2f6f9e",
  coolSoft: "#eaf1f6",
  bad: "#a8453f",
};

const EASE = "cubic-bezier(0.23, 1, 0.32, 1)";
const MONO = "'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, monospace";

const DIMS = [
  { key: "weather", label: "weather" },
  { key: "cooking", label: "cooking" },
  { key: "finance", label: "finance" },
  { key: "space", label: "space" },
  { key: "pets", label: "pets" },
  { key: "travel", label: "travel" },
  { key: "music", label: "music" },
];

function vec(obj) {
  return DIMS.map((d) => obj[d.key] || 0);
}

const DOCS = [
  { id: "forecast", text: "Tomorrow stays dry with light winds and clear skies.", e: vec({ weather: 1.0 }) },
  { id: "umbrella", text: "Carry an umbrella; showers are likely before noon.", e: vec({ weather: 1.0, travel: 0.15 }) },
  { id: "risotto", text: "Stir the rice slowly, adding warm broth one ladle at a time.", e: vec({ cooking: 1.0 }) },
  { id: "knife", text: "A sharp chef's knife makes dicing onions far safer.", e: vec({ cooking: 0.9 }) },
  { id: "index", text: "Low-cost index funds spread your savings across the whole market.", e: vec({ finance: 1.0 }) },
  { id: "paycheck", text: "Set aside part of each paycheck before you spend the rest.", e: vec({ finance: 0.95 }) },
  { id: "orbit", text: "A satellite stays up by falling around the planet forever.", e: vec({ space: 1.0 }) },
  { id: "mars", text: "The rover drilled a rock sample on the dusty Martian plain.", e: vec({ space: 0.95, travel: 0.1 }) },
  { id: "puppy", text: "A new puppy needs patient house training and short daily walks.", e: vec({ pets: 1.0 }) },
  { id: "cat", text: "Cats groom themselves and like a quiet, warm place to nap.", e: vec({ pets: 1.0 }) },
  { id: "flight", text: "Book the connecting flight with a long enough layover.", e: vec({ travel: 1.0 }) },
  { id: "passport", text: "Pack light and keep your passport in a zipped inner pocket.", e: vec({ travel: 0.95 }) },
  { id: "guitar", text: "Tune the guitar to standard pitch before the first chord.", e: vec({ music: 1.0 }) },
  { id: "songrain", text: "Their old song about purple rain still fills arenas.", e: vec({ music: 1.0, weather: 0.05 }) },
];

const WORDS = {
  rain: { label: "rain", e: vec({ weather: 1.0 }) },
  forecast: { label: "forecast", e: vec({ weather: 1.0 }) },
  umbrella: { label: "umbrella", e: vec({ weather: 0.85, travel: 0.15 }) },
  cook: { label: "cook", e: vec({ cooking: 1.0 }) },
  recipe: { label: "recipe", e: vec({ cooking: 1.0 }) },
  invest: { label: "invest", e: vec({ finance: 1.0 }) },
  money: { label: "money", e: vec({ finance: 0.9 }) },
  stocks: { label: "stocks", e: vec({ finance: 1.0 }) },
  satellite: { label: "satellite", e: vec({ space: 1.0 }) },
  planet: { label: "planet", e: vec({ space: 0.9 }) },
  rocket: { label: "rocket", e: vec({ space: 1.0 }) },
  dog: { label: "dog", e: vec({ pets: 1.0 }) },
  pet: { label: "pet", e: vec({ pets: 1.0 }) },
  trip: { label: "trip", e: vec({ travel: 1.0 }) },
  song: { label: "song", e: vec({ music: 1.0 }) },
};

const PRESETS = [
  {
    id: "rain",
    label: "will it rain?",
    words: ["rain", "forecast"],
    note: "The match that wins, dry with clear skies, shares not one word with your query. Keyword search ranks a rock song called purple rain first because the letters line up.",
  },
  {
    id: "money",
    label: "where to put my money",
    words: ["invest", "money", "stocks"],
    note: "Here the words mostly line up, so both methods find the finance notes. Easy cases hide the difference; the paraphrase cases expose it.",
  },
  {
    id: "orbit",
    label: "thing that circles a planet",
    words: ["satellite", "planet"],
    note: "Both space notes rank at the top by meaning. Keyword search only catches the one that happens to print the word satellite.",
  },
  {
    id: "dog",
    label: "getting a dog",
    words: ["dog", "pet"],
    note: "The best answer is about a puppy and never says dog or pet. Keyword overlap is zero, so string matching cannot surface it at all.",
  },
];

function add(a, b) {
  return a.map((x, i) => x + b[i]);
}
function scaleVec(a, s) {
  return a.map((x) => x * s);
}
function dot(a, b) {
  return a.reduce((s, x, i) => s + x * b[i], 0);
}
function mag(a) {
  return Math.hypot(...a);
}
function cosine(a, b) {
  const m = mag(a) * mag(b);
  if (m < 1e-9) return 0;
  return Math.max(-1, Math.min(1, dot(a, b) / m));
}

function queryVector(words) {
  let acc = DIMS.map(() => 0);
  let n = 0;
  for (const w of words) {
    if (WORDS[w]) {
      acc = add(acc, WORDS[w].e);
      n++;
    }
  }
  return n ? scaleVec(acc, 1 / n) : acc;
}

const STOP = new Set([
  "a", "an", "the", "to", "of", "for", "and", "with", "in", "on", "is", "are",
  "your", "you", "my", "it", "that", "this", "at", "by", "one", "still",
]);

function tokenize(s) {
  return s
    .toLowerCase()
    .replace(/[^a-z\s]/g, " ")
    .split(/\s+/)
    .filter((t) => t && !STOP.has(t));
}

const DOC_TOKENS = DOCS.map((d) => new Set(tokenize(d.text)));

function keywordScore(words, docIndex) {
  const toks = DOC_TOKENS[docIndex];
  let overlap = 0;
  for (const w of words) {
    if (toks.has(w)) overlap += 1;
  }
  return overlap;
}

function fmt(x, places = 2) {
  if (!Number.isFinite(x)) return "n/a";
  const v = Math.abs(x) < 1e-9 ? 0 : x;
  return v.toFixed(places);
}

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

function Card({ children, style }) {
  return (
    <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 14, padding: 18, ...style }}>
      {children}
    </div>
  );
}

function ConceptBar({ value, color }) {
  const w = Math.min(1, Math.abs(value)) * 100;
  return (
    <span
      aria-hidden="true"
      style={{ position: "relative", display: "block", height: 7, borderRadius: 4, background: C.grid, overflow: "hidden" }}
    >
      <span style={{ position: "absolute", inset: 0, width: `${w}%`, background: color, borderRadius: 4 }} />
    </span>
  );
}

function rankList(items, reduce, tone) {
  const accentColor = tone === "keyword" ? C.cool : C.good;
  const topBg = tone === "keyword" ? C.coolSoft : C.goodSoft;
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
      {items.map((item, i) => {
        const top = i === 0 && item.shown > 0;
        const dead = item.shown <= 1e-9;
        return (
          <div
            key={item.id}
            style={{
              display: "flex",
              alignItems: "center",
              gap: 9,
              padding: "7px 10px",
              borderRadius: 9,
              background: top ? topBg : C.bg,
              border: `1px solid ${top ? accentColor + "55" : C.border}`,
              opacity: dead ? 0.5 : 1,
              transition: reduce ? "none" : `background 200ms ${EASE}, opacity 200ms ${EASE}`,
            }}
          >
            <span style={{ fontFamily: MONO, fontSize: 10.5, color: C.faint, width: 14, textAlign: "right" }}>
              {i + 1}
            </span>
            <span style={{ flex: 1, minWidth: 0 }}>
              <span
                style={{
                  display: "block",
                  fontSize: 12.5,
                  lineHeight: 1.35,
                  color: top ? C.ink : C.muted,
                  fontWeight: top ? 600 : 400,
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  whiteSpace: "nowrap",
                }}
                title={item.text}
              >
                {item.text}
              </span>
              <span style={{ display: "block", marginTop: 4 }}>
                <ConceptBar value={item.barFrac} color={top ? accentColor : C.faint} />
              </span>
            </span>
            <span
              style={{
                fontFamily: MONO,
                fontSize: 12,
                fontWeight: 700,
                color: top ? accentColor : dead ? C.faint : C.muted,
                minWidth: 34,
                textAlign: "right",
                fontVariantNumeric: "tabular-nums",
              }}
            >
              {item.scoreText}
            </span>
          </div>
        );
      })}
    </div>
  );
}

function wordChipStyle(active) {
  return {
    padding: "5px 11px",
    borderRadius: 999,
    border: `1.5px solid ${active ? C.accent : C.border}`,
    background: active ? C.accent : C.card,
    color: active ? "#fff" : C.muted,
    fontSize: 12,
    fontWeight: active ? 700 : 500,
    fontFamily: MONO,
    cursor: "pointer",
    transition: `background 150ms ${EASE}, border-color 150ms ${EASE}, color 150ms ${EASE}, transform 130ms ${EASE}`,
  };
}

function presetStyle(active) {
  return {
    padding: "8px 13px",
    borderRadius: 10,
    border: `1.5px solid ${active ? C.accent : C.border}`,
    background: active ? C.accentSoft : C.card,
    color: active ? C.accent : C.muted,
    fontSize: 13,
    fontWeight: active ? 700 : 500,
    fontFamily: "inherit",
    cursor: "pointer",
    textAlign: "left",
    transition: `background 150ms ${EASE}, border-color 150ms ${EASE}, color 150ms ${EASE}, transform 130ms ${EASE}`,
  };
}

export default function App() {
  const reduce = usePrefersReducedMotion();
  const [presetId, setPresetId] = useState("rain");
  const [active, setActive] = useState(() => new Set(PRESETS[0].words));
  const [showVectors, setShowVectors] = useState(false);

  const activeWords = useMemo(() => [...active], [active]);
  const preset = PRESETS.find((p) => p.id === presetId) || null;

  const qVec = useMemo(() => queryVector(activeWords), [activeWords]);
  const qHasMeaning = mag(qVec) > 1e-9;

  const semantic = useMemo(() => {
    const scored = DOCS.map((d, i) => ({
      id: d.id,
      text: d.text,
      e: d.e,
      score: cosine(qVec, d.e),
      idx: i,
    }));
    scored.sort((a, b) => b.score - a.score || a.idx - b.idx);
    return scored.map((d) => ({
      ...d,
      shown: Math.max(0, d.score),
      barFrac: Math.max(0, d.score),
      scoreText: fmt(d.score),
    }));
  }, [qVec]);

  const keyword = useMemo(() => {
    const scored = DOCS.map((d, i) => ({
      id: d.id,
      text: d.text,
      score: keywordScore(activeWords, i),
      idx: i,
    }));
    const maxScore = scored.reduce((m, d) => Math.max(m, d.score), 0);
    scored.sort((a, b) => b.score - a.score || a.idx - b.idx);
    return scored.map((d) => ({
      ...d,
      shown: d.score,
      barFrac: maxScore > 0 ? d.score / maxScore : 0,
      scoreText: String(d.score),
    }));
  }, [activeWords]);

  const semTop = semantic[0];
  const kwTopHasHit = keyword[0] && keyword[0].score > 0;
  const semTopInKeyword = keyword.findIndex((d) => d.id === semTop.id);
  const semTopKeywordScore = semTopInKeyword >= 0 ? keyword[semTopInKeyword].score : 0;
  const agree = kwTopHasHit && keyword[0].id === semTop.id;

  const applyPreset = (p) => {
    setPresetId(p.id);
    setActive(new Set(p.words));
  };

  const toggleWord = (w) => {
    setActive((prev) => {
      const next = new Set(prev);
      if (next.has(w)) next.delete(w);
      else next.add(w);
      return next;
    });
    setPresetId(null);
  };

  const css = `
    .ss-root :focus-visible { outline: 2px solid ${C.accent}; outline-offset: 2px; border-radius: 9px; }
    .ss-tap:active { transform: scale(0.97); }
    @media (prefers-reduced-motion: reduce) {
      .ss-root * { transition-duration: 0.001ms !important; animation-duration: 0.001ms !important; }
    }
  `;

  const showVecDims = DIMS;

  return (
    <div
      className="ss-root"
      style={{
        fontFamily: "'Iowan Old Style', 'Palatino Linotype', Palatino, Georgia, serif",
        background: C.bg,
        minHeight: "100vh",
        padding: "30px 14px 56px",
        color: C.ink,
      }}
    >
      <style>{css}</style>
      <div style={{ maxWidth: 720, margin: "0 auto" }}>
        <header style={{ marginBottom: 22 }}>
          <div style={{ fontSize: 11, letterSpacing: "0.14em", textTransform: "uppercase", color: C.faint, marginBottom: 6 }}>
            Machine learning / Searching by meaning
          </div>
          <h1 style={{ fontSize: 30, fontWeight: 700, margin: 0, lineHeight: 1.1, letterSpacing: "-0.02em", textWrap: "balance" }}>
            Semantic Search: Meaning, Not Keywords
          </h1>
          <p style={{ color: C.ink, fontSize: 15, lineHeight: 1.6, margin: "12px 0 0", maxWidth: "66ch" }}>
            Type <em>will it rain</em> and the answer you want, <em>dry with clear skies tomorrow</em>, shares none of
            your words. Keyword search reads letters, so it cannot find that line, and it will happily rank a rock song
            called <em>purple rain</em> instead. Semantic search reads direction in a space of meaning. Pick a query
            below and watch the two rankings split apart.
          </p>
        </header>

        <Card style={{ marginBottom: 18 }}>
          <div style={{ fontSize: 11, letterSpacing: "0.06em", textTransform: "uppercase", color: C.muted, fontWeight: 700, marginBottom: 8 }}>
            Pick a query
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: 8, marginBottom: 16 }}>
            {PRESETS.map((p) => (
              <button
                key={p.id}
                type="button"
                className="ss-tap"
                onClick={() => applyPreset(p)}
                style={presetStyle(presetId === p.id)}
                aria-pressed={presetId === p.id}
              >
                {p.label}
              </button>
            ))}
          </div>

          <div style={{ fontSize: 11, letterSpacing: "0.06em", textTransform: "uppercase", color: C.muted, fontWeight: 700, marginBottom: 8 }}>
            Or shape the query word by word
          </div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 7, marginBottom: 12 }}>
            {Object.entries(WORDS).map(([w, info]) => (
              <button
                key={w}
                type="button"
                className="ss-tap"
                onClick={() => toggleWord(w)}
                style={wordChipStyle(active.has(w))}
                aria-pressed={active.has(w)}
              >
                {info.label}
              </button>
            ))}
          </div>

          <div
            aria-live="polite"
            style={{
              background: C.bg,
              border: `1px solid ${C.border}`,
              borderRadius: 10,
              padding: "10px 12px",
              fontSize: 12.5,
              color: C.muted,
              lineHeight: 1.55,
              minHeight: 20,
            }}
          >
            {qHasMeaning ? (
              <>
                Query meaning is the average of the chosen word vectors:{" "}
                <span style={{ fontFamily: MONO, color: C.ink }}>
                  ({showVecDims.map((d, i) => fmt(qVec[i], 2)).join(", ")})
                </span>{" "}
                over {showVecDims.map((d) => d.label).join(", ")}.
              </>
            ) : (
              <>Pick at least one query word to give the query a direction in concept space.</>
            )}
          </div>
        </Card>

        <Card style={{ marginBottom: 18 }}>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
              gap: 16,
            }}
          >
            <div>
              <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", marginBottom: 4 }}>
                <span style={{ fontSize: 13.5, fontWeight: 700, color: C.good }}>Semantic ranking</span>
                <span style={{ fontSize: 11, color: C.faint }}>cosine, meaning</span>
              </div>
              <p style={{ fontSize: 11.5, color: C.muted, margin: "0 0 10px", lineHeight: 1.5 }}>
                Each document is a direction in concept space. Rank by the cosine of the angle to the query.
              </p>
              {qHasMeaning ? (
                rankList(semantic, reduce, "semantic")
              ) : (
                <div style={{ fontSize: 12.5, color: C.faint, padding: "12px 0" }}>No query direction yet.</div>
              )}
            </div>

            <div>
              <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", marginBottom: 4 }}>
                <span style={{ fontSize: 13.5, fontWeight: 700, color: C.cool }}>Keyword ranking</span>
                <span style={{ fontSize: 11, color: C.faint }}>shared words</span>
              </div>
              <p style={{ fontSize: 11.5, color: C.muted, margin: "0 0 10px", lineHeight: 1.5 }}>
                Count how many query words appear verbatim in each document. No word in common means a score of zero.
              </p>
              {rankList(keyword, reduce, "keyword")}
            </div>
          </div>

          <div
            aria-live="polite"
            style={{
              marginTop: 16,
              padding: "12px 14px",
              borderRadius: 10,
              background: agree ? C.bg : C.accentSoft,
              border: `1px solid ${agree ? C.border : C.accent + "33"}`,
              fontSize: 13,
              lineHeight: 1.6,
              color: agree ? C.muted : "#5d4226",
            }}
          >
            {!qHasMeaning ? (
              "Choose query words to compare the two rankings."
            ) : agree ? (
              <>
                This time the words mostly line up, so both methods agree on{" "}
                <strong style={{ color: C.ink }}>{semTop.text}</strong>. The gap only opens when your words and the
                document's words drift apart.
              </>
            ) : !kwTopHasHit ? (
              <>
                Keyword search returns <strong>nothing useful</strong>: not one query word appears in any document, so
                every keyword score is zero. Semantic search still ranks{" "}
                <strong style={{ color: C.good }}>{semTop.text}</strong> first, at cosine{" "}
                <strong style={{ color: C.good }}>{fmt(semTop.score)}</strong>, because its meaning points the same way.
              </>
            ) : (
              <>
                Semantic search puts <strong style={{ color: C.good }}>{semTop.text}</strong> first at cosine{" "}
                <strong style={{ color: C.good }}>{fmt(semTop.score)}</strong>, yet keyword search ranks that same line{" "}
                <strong>#{semTopInKeyword + 1}</strong> with a score of {semTopKeywordScore}. It instead promotes{" "}
                <strong style={{ color: C.cool }}>{keyword[0].text}</strong> only because the words happen to match.
              </>
            )}
          </div>

          {preset && (
            <p style={{ fontSize: 12.5, color: C.muted, margin: "12px 0 0", lineHeight: 1.6 }}>
              {preset.note}
            </p>
          )}
        </Card>

        <Card style={{ marginBottom: 18 }}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 10, flexWrap: "wrap", marginBottom: 4 }}>
            <div style={{ fontSize: 13, fontWeight: 700 }}>Meaning lives in the concept vectors</div>
            <button
              type="button"
              className="ss-tap"
              onClick={() => setShowVectors((v) => !v)}
              style={{
                padding: "6px 12px",
                borderRadius: 8,
                border: `1px solid ${showVectors ? C.accent : C.border}`,
                background: showVectors ? C.accentSoft : "transparent",
                color: showVectors ? C.accent : C.muted,
                fontSize: 12,
                fontWeight: showVectors ? 700 : 500,
                fontFamily: "inherit",
                cursor: "pointer",
                transition: `background 150ms ${EASE}, border-color 150ms ${EASE}, color 150ms ${EASE}`,
              }}
              aria-pressed={showVectors}
            >
              {showVectors ? "Hide vectors" : "Show vectors"}
            </button>
          </div>
          <p style={{ fontSize: 13.5, color: C.muted, lineHeight: 1.55, margin: "0 0 12px", maxWidth: "64ch" }}>
            Every document and every query word is placed along the same handful of concept axes. Two lines about
            money point the same way even with no shared words, so the cosine between them is near one. A weather query
            and a cooking note point in different directions, so their cosine is near zero. Direction is the meaning.
          </p>

          {showVectors && (
            <div style={{ overflowX: "auto", margin: "0 -4px" }}>
              <table style={{ borderCollapse: "collapse", width: "100%", minWidth: 520, fontSize: 11.5 }}>
                <thead>
                  <tr>
                    <th style={{ textAlign: "left", padding: "6px 8px", color: C.muted, fontWeight: 600, borderBottom: `1px solid ${C.border}` }}>
                      document
                    </th>
                    {DIMS.map((d) => (
                      <th
                        key={d.key}
                        style={{
                          padding: "6px 6px",
                          color: C.muted,
                          fontWeight: 600,
                          borderBottom: `1px solid ${C.border}`,
                          fontFamily: MONO,
                          fontSize: 10.5,
                        }}
                      >
                        {d.label}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {DOCS.map((doc) => (
                    <tr key={doc.id}>
                      <td
                        style={{
                          padding: "6px 8px",
                          color: C.ink,
                          borderBottom: `1px solid ${C.grid}`,
                          maxWidth: 220,
                          overflow: "hidden",
                          textOverflow: "ellipsis",
                          whiteSpace: "nowrap",
                        }}
                        title={doc.text}
                      >
                        {doc.text}
                      </td>
                      {doc.e.map((v, i) => (
                        <td
                          key={i}
                          style={{
                            padding: "6px 6px",
                            textAlign: "center",
                            fontFamily: MONO,
                            color: v > 0 ? C.ink : C.faint,
                            background: v > 0 ? `rgba(192, 86, 31, ${0.06 + v * 0.16})` : "transparent",
                            borderBottom: `1px solid ${C.grid}`,
                          }}
                        >
                          {v > 0 ? fmt(v, 1) : "0"}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Card>

        <Card style={{ background: C.accentSoft, borderColor: `${C.accent}33`, marginBottom: 18 }}>
          <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 10, color: "#5d4226" }}>Why meaning beats matching</div>
          <p style={{ fontSize: 14, lineHeight: 1.65, color: "#5d4226", margin: "0 0 10px", maxWidth: "66ch" }}>
            Embeddings put meaning into geometry: similar ideas become nearby directions, unrelated ideas become
            perpendicular ones. Cosine similarity reads only that direction, ignoring how long a vector is, so it asks
            <em> do these mean the same thing</em> rather than <em>do they share the same letters</em>. That is why a
            paraphrase or a synonym still lands on the right document, and why a coincidental word match gets pushed
            down where it belongs.
          </p>
          <p style={{ fontSize: 14, lineHeight: 1.65, color: "#5d4226", margin: 0, maxWidth: "66ch" }}>
            Keyword search has the opposite failure: it can only rank what it can spell. When your words and the answer's
            words diverge, it returns the wrong line or nothing at all. Meaning fills exactly that gap.
          </p>
        </Card>

        <Card>
          <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 8 }}>What the toy leaves out</div>
          <p style={{ fontSize: 13, color: C.muted, lineHeight: 1.6, margin: 0, maxWidth: "66ch" }}>
            These vectors are hand placed along seven named topics so you can read the geometry directly. Real systems
            learn the embedding from text with a neural model, and the space has hundreds or thousands of dimensions
            with no human-readable labels. The cosine ranking math is exactly the same; only the source and size of the
            vectors change. At scale the query embedding is compared against millions of document embeddings in a vector
            database, and feeding the top matches to a language model is what retrieval augmented generation, RAG,
            actually does.
          </p>
        </Card>

        <p style={{ textAlign: "center", fontSize: 11.5, color: C.faint, marginTop: 22, lineHeight: 1.6 }}>
          The query vector is the average of the selected word vectors; each semantic score is the cosine between it and a
          document vector, and each keyword score counts shared words. Both rankings update live from those numbers.
        </p>
      </div>
    </div>
  );
}
