import { useState, useEffect, useRef, useMemo } from "react";

export const meta = {
  title: "Word Embeddings and Analogy Arithmetic",
  category: "Machine Learning",
  description:
    "King minus man plus woman lands on queen, and it sounds like a trick. Place words in a space where meaning is direction, then watch the arithmetic fall out of the geometry.",
  date: "2026-05-15",
  tags: ["word-embeddings", "vectors", "cosine-similarity", "analogies", "nlp"],
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
  axis: "#cfc7ba",
  a: "#2f6f9e",
  b: "#9a6b1f",
  c: "#3f7d52",
  result: "#c0561f",
  hit: "#3f7d52",
};

const EASE = "cubic-bezier(0.23, 1, 0.32, 1)";
const MONO = "'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, monospace";

const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));

const DIMS = ["royalty", "gender", "age", "human", "place"];
const DIM_HINT = [
  "royal vs ordinary",
  "female vs male",
  "adult vs young",
  "person vs animal vs thing",
  "place vs concept",
];

// An illustrative hand-built space. Each word is five numbers, one per labeled axis,
// placed by hand so the classic analogies close by construction. Learned embeddings
// have hundreds of axes found from text and none of them are named like this; the
// geometry is the same idea, the readable labels are the simplification.
const VOCAB = {
  king: [0.95, -0.85, 0.7, 0.9, 0.0],
  queen: [0.95, 0.85, 0.7, 0.9, 0.0],
  prince: [0.85, -0.8, -0.55, 0.9, 0.0],
  princess: [0.85, 0.8, -0.55, 0.9, 0.0],
  man: [-0.1, -0.85, 0.7, 0.9, 0.0],
  woman: [-0.1, 0.85, 0.7, 0.9, 0.0],
  boy: [-0.1, -0.8, -0.7, 0.9, 0.0],
  girl: [-0.1, 0.8, -0.7, 0.9, 0.0],
  uncle: [0.05, -0.8, 0.65, 0.9, 0.0],
  aunt: [0.05, 0.8, 0.65, 0.9, 0.0],
  nephew: [0.05, -0.8, -0.6, 0.9, 0.0],
  niece: [0.05, 0.8, -0.6, 0.9, 0.0],
  actor: [0.1, -0.78, 0.55, 0.92, 0.2],
  actress: [0.1, 0.78, 0.55, 0.92, 0.2],
  father: [0.0, -0.85, 0.75, 0.9, 0.0],
  mother: [0.0, 0.85, 0.75, 0.9, 0.0],
  son: [0.0, -0.8, -0.65, 0.9, 0.0],
  daughter: [0.0, 0.8, -0.65, 0.9, 0.0],
  brother: [0.0, -0.82, 0.3, 0.9, 0.0],
  sister: [0.0, 0.82, 0.3, 0.9, 0.0],
  dog: [-0.2, 0.0, 0.2, -0.85, -0.1],
  cat: [-0.2, 0.05, 0.2, -0.85, -0.1],
  puppy: [-0.2, 0.0, -0.75, -0.85, -0.1],
  kitten: [-0.2, 0.05, -0.75, -0.85, -0.1],
  horse: [-0.15, 0.0, 0.3, -0.8, -0.1],
  lion: [0.3, -0.1, 0.3, -0.8, -0.05],
  apple: [0.0, 0.0, 0.0, -0.95, -0.4],
  bread: [0.0, 0.0, 0.1, -0.95, -0.35],
  car: [0.0, 0.0, 0.2, -0.9, -0.3],
  book: [0.0, 0.0, 0.1, -0.9, 0.4],
  Paris: [0.05, 0.0, 0.3, -0.3, 0.82],
  France: [0.05, 0.0, 0.3, -0.3, 0.98],
  Tokyo: [0.2, 0.05, 0.55, -0.3, 0.82],
  Japan: [0.2, 0.05, 0.55, -0.3, 0.98],
  London: [0.35, 0.0, 0.1, -0.3, 0.82],
  England: [0.35, 0.0, 0.1, -0.3, 0.98],
  Rome: [-0.1, 0.0, 0.7, -0.3, 0.82],
  Italy: [-0.1, 0.0, 0.7, -0.3, 0.98],
  freedom: [0.0, 0.0, 0.0, -0.2, 0.6],
  music: [0.0, 0.0, 0.0, -0.3, 0.55],
};

const WORDS = Object.keys(VOCAB);

// Fixed linear projection to a plane: x leans on gender (and a touch of place), y on
// royalty plus humanity. Chosen so the visible parallelogram tracks the real offsets.
const PROJ_X = [0.15, 1.0, 0.1, 0.0, 0.35];
const PROJ_Y = [0.9, 0.0, 0.15, 0.55, -0.2];

const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
const mag = (v) => Math.hypot(...v);
const cosine = (a, b) => {
  const m = mag(a) * mag(b);
  return m < 1e-9 ? 0 : clamp(dot(a, b) / m, -1, 1);
};
const sub = (a, b) => a.map((v, i) => v - b[i]);
const add = (a, b) => a.map((v, i) => v + b[i]);
const project = (v) => [dot(v, PROJ_X), dot(v, PROJ_Y)];

function nearest(vec, exclude, limit) {
  return WORDS.filter((w) => !exclude.includes(w))
    .map((w) => ({ word: w, cos: cosine(vec, VOCAB[w]) }))
    .sort((p, q) => q.cos - p.cos)
    .slice(0, limit);
}

function fmt(x, places = 3) {
  if (!Number.isFinite(x)) return "n/a";
  const v = Math.abs(x) < 1e-9 ? 0 : x;
  return v.toFixed(places);
}

function Card({ children, style }) {
  return (
    <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 14, padding: 18, ...style }}>
      {children}
    </div>
  );
}

function chipStyle(active, tone) {
  const color = tone || C.accent;
  return {
    padding: "5px 11px",
    borderRadius: 8,
    border: `1.5px solid ${active ? color : C.border}`,
    background: active ? color : C.card,
    color: active ? "#fff" : C.muted,
    fontFamily: MONO,
    fontSize: 12.5,
    fontWeight: 600,
    cursor: "pointer",
    transition: `transform 140ms ${EASE}, background 150ms ${EASE}, border-color 150ms ${EASE}, color 150ms ${EASE}`,
  };
}

function toggleStyle(active) {
  return {
    padding: "6px 12px",
    borderRadius: 8,
    border: `1px solid ${active ? C.accent : C.border}`,
    background: active ? C.accentSoft : "transparent",
    color: active ? C.accent : C.muted,
    fontSize: 12,
    fontWeight: active ? 700 : 500,
    cursor: "pointer",
    fontFamily: "inherit",
    transition: `background 150ms ${EASE}, border-color 150ms ${EASE}, color 150ms ${EASE}`,
  };
}

const VIEW_W = 440;
const VIEW_H = 360;

function Map2D({ analogy, result, resultWord, showOffsets, showAxes, onPick, selected }) {
  const ref = useRef(null);

  const pts = useMemo(() => {
    const raw = WORDS.map((w) => ({ word: w, p: project(VOCAB[w]) }));
    let minX = Infinity,
      maxX = -Infinity,
      minY = Infinity,
      maxY = -Infinity;
    for (const r of raw) {
      minX = Math.min(minX, r.p[0]);
      maxX = Math.max(maxX, r.p[0]);
      minY = Math.min(minY, r.p[1]);
      maxY = Math.max(maxY, r.p[1]);
    }
    const pad = 42;
    const sx = (VIEW_W - pad * 2) / (maxX - minX || 1);
    const sy = (VIEW_H - pad * 2) / (maxY - minY || 1);
    const toPx = (p) => [pad + (p[0] - minX) * sx, VIEW_H - pad - (p[1] - minY) * sy];
    const map = {};
    for (const r of raw) map[r.word] = toPx(r.p);
    return { map, toPx };
  }, []);

  const aWord = analogy.a;
  const bWord = analogy.b;
  const cWord = analogy.c;

  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    cv.width = VIEW_W * dpr;
    cv.height = VIEW_H * dpr;
    const ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, VIEW_W, VIEW_H);

    if (showAxes) {
      ctx.strokeStyle = C.grid;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(20, VIEW_H / 2);
      ctx.lineTo(VIEW_W - 20, VIEW_H / 2);
      ctx.moveTo(VIEW_W / 2, 24);
      ctx.lineTo(VIEW_W / 2, VIEW_H - 18);
      ctx.stroke();
      ctx.font = `600 10px ${MONO}`;
      ctx.fillStyle = C.faint;
      ctx.textAlign = "left";
      ctx.fillText("male", 8, VIEW_H / 2 - 6);
      ctx.textAlign = "right";
      ctx.fillText("female", VIEW_W - 8, VIEW_H / 2 - 6);
      ctx.textAlign = "center";
      ctx.fillText("royal / person", VIEW_W / 2, 14);
      ctx.fillText("animal / thing", VIEW_W / 2, VIEW_H - 6);
    }

    const resultPx = result ? pts.toPx(project(result)) : null;

    if (showOffsets && pts.map[aWord] && pts.map[bWord] && pts.map[cWord] && resultPx) {
      const [ax, ay] = pts.map[aWord];
      const [bx, by] = pts.map[bWord];
      const [cx, cy] = pts.map[cWord];
      const [rx, ry] = resultPx;
      ctx.setLineDash([5, 5]);
      ctx.strokeStyle = C.b + "55";
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      ctx.moveTo(bx, by);
      ctx.lineTo(cx, cy);
      ctx.moveTo(ax, ay);
      ctx.lineTo(rx, ry);
      ctx.stroke();
      ctx.setLineDash([]);
      const arrow = (x0, y0, x1, y1, color) => {
        ctx.strokeStyle = color;
        ctx.lineWidth = 2.6;
        ctx.beginPath();
        ctx.moveTo(x0, y0);
        ctx.lineTo(x1, y1);
        ctx.stroke();
        const ang = Math.atan2(y1 - y0, x1 - x0);
        const h = 9;
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x1 - h * Math.cos(ang - 0.4), y1 - h * Math.sin(ang - 0.4));
        ctx.lineTo(x1 - h * Math.cos(ang + 0.4), y1 - h * Math.sin(ang + 0.4));
        ctx.closePath();
        ctx.fill();
      };
      arrow(bx, by, ax, ay, C.a);
      arrow(cx, cy, rx, ry, C.result);
    }

    for (const w of WORDS) {
      const [x, y] = pts.map[w];
      const role =
        w === aWord ? "a" : w === bWord ? "b" : w === cWord ? "c" : w === resultWord ? "hit" : null;
      const isSel = selected === w;
      const color =
        role === "a"
          ? C.a
          : role === "b"
            ? C.b
            : role === "c"
              ? C.c
              : role === "hit"
                ? C.hit
                : C.faint;
      const big = role || isSel;
      ctx.fillStyle = role ? color : isSel ? C.accent : "#bdb4a8";
      ctx.beginPath();
      ctx.arc(x, y, big ? 5 : 3.2, 0, Math.PI * 2);
      ctx.fill();
      if (big) {
        ctx.strokeStyle = C.card;
        ctx.lineWidth = 1.6;
        ctx.stroke();
      }
      ctx.font = `${big ? "700" : "500"} ${big ? 12 : 10.5}px ${MONO}`;
      ctx.fillStyle = role ? color : isSel ? C.accent : C.muted;
      ctx.textAlign = "left";
      ctx.fillText(w, x + 7, y - 5);
    }

    if (resultPx && resultWord !== aWord) {
      const [rx, ry] = resultPx;
      ctx.strokeStyle = C.result;
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      ctx.arc(rx, ry, 9, 0, Math.PI * 2);
      ctx.stroke();
    }
  }, [analogy, result, resultWord, showOffsets, showAxes, pts, selected, aWord, bWord, cWord]);

  const handleClick = (e) => {
    if (!onPick) return;
    const rect = ref.current.getBoundingClientRect();
    const sx = VIEW_W / rect.width;
    const sy = VIEW_H / rect.height;
    const mx = (e.clientX - rect.left) * sx;
    const my = (e.clientY - rect.top) * sy;
    let best = null;
    let bestD = 22;
    for (const w of WORDS) {
      const [x, y] = pts.map[w];
      const d = Math.hypot(mx - x, my - y);
      if (d < bestD) {
        bestD = d;
        best = w;
      }
    }
    if (best) onPick(best);
  };

  return (
    <canvas
      ref={ref}
      onClick={handleClick}
      style={{
        width: VIEW_W,
        height: VIEW_H,
        maxWidth: "100%",
        borderRadius: 12,
        background: C.bg,
        border: `1px solid ${C.border}`,
        display: "block",
        cursor: onPick ? "pointer" : "default",
        touchAction: "manipulation",
      }}
      role="img"
      aria-label={`A plane of ${WORDS.length} words projected from the five labeled dimensions. The analogy ${aWord} is to ${bWord} as ${cWord} is to ${resultWord}. Two parallel arrows show the offset from ${bWord} to ${aWord} and from ${cWord} to the computed point, which lands nearest ${resultWord}.`}
    />
  );
}

function WordSelect({ value, tone, onChange }) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      style={{
        fontFamily: MONO,
        fontSize: 13,
        fontWeight: 600,
        color: C.ink,
        padding: "7px 9px",
        borderRadius: 9,
        border: `1.5px solid ${tone ? tone + "88" : C.border}`,
        background: C.card,
        cursor: "pointer",
      }}
    >
      {WORDS.map((w) => (
        <option key={w} value={w}>
          {w}
        </option>
      ))}
    </select>
  );
}

function WordPicker({ label, value, tone, onChange }) {
  return (
    <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      <span style={{ fontSize: 10, letterSpacing: "0.05em", textTransform: "uppercase", color: tone, fontWeight: 700 }}>
        {label}
      </span>
      <WordSelect value={value} tone={tone} onChange={onChange} />
    </label>
  );
}

function RankList({ ranked, expected }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
      {ranked.map((item, i) => {
        const top = i === 0;
        const isExpected = expected && item.word === expected;
        return (
          <div
            key={item.word}
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              gap: 10,
              padding: "7px 11px",
              borderRadius: 9,
              background: top ? "rgba(63, 125, 82, 0.1)" : C.bg,
              border: `1px solid ${top ? C.hit + "55" : C.border}`,
            }}
          >
            <span style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <span style={{ fontFamily: MONO, fontSize: 11, color: C.faint, width: 14 }}>{i + 1}</span>
              <span
                style={{
                  fontFamily: MONO,
                  fontSize: 13.5,
                  fontWeight: top || isExpected ? 700 : 500,
                  color: top ? C.hit : isExpected ? C.accent : C.ink,
                }}
              >
                {item.word}
              </span>
            </span>
            <span style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <span
                style={{ width: 56, height: 6, borderRadius: 4, background: C.grid, overflow: "hidden", position: "relative" }}
                aria-hidden="true"
              >
                <span
                  style={{
                    position: "absolute",
                    inset: 0,
                    width: `${clamp(item.cos, 0, 1) * 100}%`,
                    background: top ? C.hit : C.faint,
                  }}
                />
              </span>
              <span
                style={{
                  fontFamily: MONO,
                  fontSize: 12,
                  fontWeight: 700,
                  color: top ? C.hit : C.muted,
                  minWidth: 44,
                  textAlign: "right",
                }}
              >
                {fmt(item.cos)}
              </span>
            </span>
          </div>
        );
      })}
    </div>
  );
}

const ANALOGY_PRESETS = [
  { a: "king", b: "man", c: "woman", expect: "queen" },
  { a: "uncle", b: "man", c: "woman", expect: "aunt" },
  { a: "prince", b: "boy", c: "girl", expect: "princess" },
  { a: "Paris", b: "France", c: "Japan", expect: "Tokyo" },
  { a: "actor", b: "man", c: "woman", expect: "actress" },
];

export default function App() {
  const [analogy, setAnalogy] = useState({ a: "king", b: "man", c: "woman", expect: "queen" });
  const [showOffsets, setShowOffsets] = useState(true);
  const [showAxes, setShowAxes] = useState(true);

  const [builder, setBuilder] = useState([
    { word: "king", sign: 1 },
    { word: "man", sign: -1 },
    { word: "woman", sign: 1 },
  ]);

  const [lookup, setLookup] = useState("queen");

  const resultVec = useMemo(
    () => add(sub(VOCAB[analogy.a], VOCAB[analogy.b]), VOCAB[analogy.c]),
    [analogy]
  );

  const ranked = useMemo(
    () => nearest(resultVec, [analogy.a, analogy.b, analogy.c], 5),
    [resultVec, analogy]
  );
  const resultWord = ranked[0]?.word;

  const builderVec = useMemo(() => {
    let v = [0, 0, 0, 0, 0];
    for (const term of builder) {
      const tv = VOCAB[term.word];
      v = v.map((x, i) => x + term.sign * tv[i]);
    }
    return v;
  }, [builder]);

  const builderRanked = useMemo(
    () =>
      nearest(
        builderVec,
        builder.map((t) => t.word),
        4
      ),
    [builderVec, builder]
  );

  const lookupRanked = useMemo(() => nearest(VOCAB[lookup], [lookup], 5), [lookup]);

  const setTerm = (key, word) => setAnalogy((prev) => ({ ...prev, [key]: word, expect: null }));

  const setBuilderWord = (idx, word) =>
    setBuilder((prev) => prev.map((t, i) => (i === idx ? { ...t, word } : t)));
  const toggleSign = (idx) =>
    setBuilder((prev) => prev.map((t, i) => (i === idx ? { ...t, sign: t.sign * -1 } : t)));
  const addTerm = () => setBuilder((prev) => [...prev, { word: "boy", sign: 1 }]);
  const removeTerm = (idx) => setBuilder((prev) => (prev.length > 2 ? prev.filter((_, i) => i !== idx) : prev));

  const css = `
    .we-root :focus-visible { outline: 2px solid ${C.accent}; outline-offset: 2px; border-radius: 9px; }
    .we-chip:active { transform: scale(0.97); }
    @media (prefers-reduced-motion: reduce) {
      .we-root * { transition-duration: 0.001ms !important; animation-duration: 0.001ms !important; }
    }
  `;

  return (
    <div
      className="we-root"
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
            Machine learning / Meaning as direction
          </div>
          <h1 style={{ fontSize: 30, fontWeight: 700, margin: 0, lineHeight: 1.1, letterSpacing: "-0.02em", textWrap: "balance" }}>
            Word Embeddings and Analogy Arithmetic
          </h1>
          <p style={{ color: C.ink, fontSize: 15, lineHeight: 1.6, margin: "12px 0 0", maxWidth: "66ch" }}>
            King minus man plus woman lands on queen, and the first time you see it the result feels like sleight of
            hand. It stops being magic once each word is a point in space. Give every word a few coordinates so that
            similar words sit close together, and a relationship like male to female becomes a fixed step in one
            direction. Take that same step from king and you arrive next to queen.
          </p>
        </header>

        <Card style={{ marginBottom: 18 }}>
          <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 4 }}>The space these words live in</div>
          <p style={{ fontSize: 13.5, color: C.muted, lineHeight: 1.55, margin: "0 0 12px", maxWidth: "64ch" }}>
            Every word here is five numbers, one per labeled axis. A word is a vector, and two words mean similar things
            when their vectors point nearly the same way. The map further down is a flat snapshot of those five
            dimensions, so left to right tracks male to female and bottom to top tracks animal-or-thing up through
            person and royalty.
          </p>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(118px, 1fr))", gap: 7 }}>
            {DIMS.map((d, i) => (
              <div
                key={d}
                style={{
                  background: C.bg,
                  border: `1px solid ${C.border}`,
                  borderRadius: 9,
                  padding: "7px 10px",
                }}
              >
                <div style={{ fontFamily: MONO, fontSize: 11.5, fontWeight: 700, color: C.accent }}>
                  {i}. {d}
                </div>
                <div style={{ fontSize: 11, color: C.muted, marginTop: 2, lineHeight: 1.35 }}>{DIM_HINT[i]}</div>
              </div>
            ))}
          </div>
        </Card>

        <Card style={{ marginBottom: 18 }}>
          <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 4 }}>The analogy as a parallelogram</div>
          <p style={{ fontSize: 13.5, color: C.muted, lineHeight: 1.55, margin: "0 0 14px", maxWidth: "64ch" }}>
            Read it as <em>a</em> is to <em>b</em> as <em>c</em> is to what. The step from <strong>b</strong> to{" "}
            <strong>a</strong> is one arrow; take the same arrow starting at <strong>c</strong> and its tip lands on the
            answer. When the two arrows come out parallel and the same length, the four points form a parallelogram,
            which is the picture behind the arithmetic <span style={{ fontFamily: MONO }}>a - b + c</span>.
          </p>

          <div style={{ display: "flex", gap: 18, flexWrap: "wrap", alignItems: "flex-start" }}>
            <Map2D
              analogy={analogy}
              result={resultVec}
              resultWord={resultWord}
              showOffsets={showOffsets}
              showAxes={showAxes}
            />

            <div style={{ flex: "1 1 230px", minWidth: 220 }}>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 9, marginBottom: 12 }}>
                <WordPicker label="a" tone={C.a} value={analogy.a} onChange={(w) => setTerm("a", w)} />
                <WordPicker label="b" tone={C.b} value={analogy.b} onChange={(w) => setTerm("b", w)} />
                <WordPicker label="c" tone={C.c} value={analogy.c} onChange={(w) => setTerm("c", w)} />
                <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                  <span style={{ fontSize: 10, letterSpacing: "0.05em", textTransform: "uppercase", color: C.result, fontWeight: 700 }}>
                    result
                  </span>
                  <div
                    style={{
                      fontFamily: MONO,
                      fontSize: 13,
                      fontWeight: 700,
                      color: C.result,
                      padding: "7px 9px",
                      borderRadius: 9,
                      border: `1.5px solid ${C.result}`,
                      background: C.accentSoft,
                    }}
                  >
                    {resultWord}
                  </div>
                </div>
              </div>

              <div
                style={{
                  fontFamily: MONO,
                  fontSize: 12.5,
                  color: "#5d4226",
                  background: C.accentSoft,
                  border: `1px solid ${C.accent}33`,
                  borderRadius: 9,
                  padding: "9px 11px",
                  marginBottom: 12,
                  lineHeight: 1.5,
                }}
              >
                <span style={{ color: C.a, fontWeight: 700 }}>{analogy.a}</span> -{" "}
                <span style={{ color: C.b, fontWeight: 700 }}>{analogy.b}</span> +{" "}
                <span style={{ color: C.c, fontWeight: 700 }}>{analogy.c}</span> ={" "}
                <strong style={{ color: C.result }}>{resultWord}</strong>
              </div>

              <div style={{ display: "flex", gap: 7, flexWrap: "wrap", marginBottom: 10 }}>
                <button type="button" className="we-chip" onClick={() => setShowOffsets((s) => !s)} style={toggleStyle(showOffsets)} aria-pressed={showOffsets}>
                  Offset vectors
                </button>
                <button type="button" className="we-chip" onClick={() => setShowAxes((s) => !s)} style={toggleStyle(showAxes)} aria-pressed={showAxes}>
                  Axis labels
                </button>
              </div>

              <div style={{ fontSize: 10, letterSpacing: "0.05em", textTransform: "uppercase", color: C.muted, fontWeight: 700, marginBottom: 6 }}>
                try an analogy
              </div>
              <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                {ANALOGY_PRESETS.map((p) => {
                  const active = analogy.a === p.a && analogy.b === p.b && analogy.c === p.c;
                  return (
                    <button
                      key={`${p.a}-${p.b}-${p.c}`}
                      type="button"
                      className="we-chip"
                      onClick={() => setAnalogy(p)}
                      style={chipStyle(active)}
                    >
                      {p.a}-{p.b}+{p.c}
                    </button>
                  );
                })}
              </div>
            </div>
          </div>

          <div style={{ marginTop: 16 }}>
            <div style={{ fontSize: 11, letterSpacing: "0.05em", textTransform: "uppercase", color: C.muted, fontWeight: 700, marginBottom: 6 }}>
              nearest words to the result vector, by cosine
            </div>
            <RankList ranked={ranked} expected={analogy.expect} />
            <p style={{ fontSize: 11.5, color: C.faint, margin: "10px 0 0", lineHeight: 1.5 }}>
              The three input words are held out of the ranking, since the answer should be a fourth word, not one you
              already named.
            </p>
          </div>
        </Card>

        <Card style={{ marginBottom: 18 }}>
          <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 4 }}>Free vector arithmetic</div>
          <p style={{ fontSize: 13.5, color: C.muted, lineHeight: 1.55, margin: "0 0 14px", maxWidth: "64ch" }}>
            Add and subtract any words you like, then read off the nearest neighbor of the sum. Flip a term between plus
            and minus to feel how each one pushes the result toward or away from a region of the space.
          </p>

          <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: 8, marginBottom: 14 }}>
            {builder.map((term, idx) => (
              <span key={idx} style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
                {idx > 0 || term.sign < 0 ? (
                  <button
                    type="button"
                    className="we-chip"
                    onClick={() => toggleSign(idx)}
                    aria-label={`Term ${idx + 1} sign, currently ${term.sign > 0 ? "plus" : "minus"}, click to flip`}
                    style={{
                      width: 28,
                      height: 28,
                      borderRadius: 8,
                      border: `1.5px solid ${term.sign > 0 ? C.hit : C.result}`,
                      background: term.sign > 0 ? "rgba(63,125,82,0.12)" : C.accentSoft,
                      color: term.sign > 0 ? C.hit : C.result,
                      fontFamily: MONO,
                      fontWeight: 800,
                      fontSize: 16,
                      cursor: "pointer",
                      lineHeight: 1,
                    }}
                  >
                    {term.sign > 0 ? "+" : "−"}
                  </button>
                ) : null}
                <WordSelect value={term.word} onChange={(w) => setBuilderWord(idx, w)} />
                {builder.length > 2 && (
                  <button
                    type="button"
                    className="we-chip"
                    onClick={() => removeTerm(idx)}
                    aria-label={`Remove term ${idx + 1}`}
                    style={{
                      width: 22,
                      height: 22,
                      borderRadius: 6,
                      border: `1px solid ${C.border}`,
                      background: "transparent",
                      color: C.faint,
                      fontSize: 13,
                      cursor: "pointer",
                      lineHeight: 1,
                    }}
                  >
                    {"×"}
                  </button>
                )}
              </span>
            ))}
            <button type="button" className="we-chip" onClick={addTerm} style={chipStyle(false)}>
              + add word
            </button>
            <span style={{ fontFamily: MONO, fontSize: 15, color: C.muted, fontWeight: 700 }}>=</span>
            <span
              style={{
                fontFamily: MONO,
                fontSize: 15,
                fontWeight: 800,
                color: C.result,
                padding: "6px 12px",
                borderRadius: 9,
                background: C.accentSoft,
                border: `1.5px solid ${C.result}`,
              }}
            >
              {builderRanked[0]?.word}
            </span>
          </div>

          <RankList ranked={builderRanked} expected={null} />
        </Card>

        <Card style={{ marginBottom: 18 }}>
          <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 4 }}>Nearest neighbors of one word</div>
          <p style={{ fontSize: 13.5, color: C.muted, lineHeight: 1.55, margin: "0 0 14px", maxWidth: "64ch" }}>
            No arithmetic here, just closeness. Pick a word and see which others point most nearly the same way. This is
            the raw similarity the analogies are built on top of.
          </p>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 14, alignItems: "center" }}>
            <span style={{ fontSize: 12, color: C.muted, fontWeight: 600 }}>closest to</span>
            <select
              value={lookup}
              onChange={(e) => setLookup(e.target.value)}
              style={{
                fontFamily: MONO,
                fontSize: 13.5,
                fontWeight: 700,
                color: C.accent,
                padding: "7px 11px",
                borderRadius: 9,
                border: `1.5px solid ${C.accent}66`,
                background: C.accentSoft,
                cursor: "pointer",
              }}
            >
              {WORDS.map((w) => (
                <option key={w} value={w}>
                  {w}
                </option>
              ))}
            </select>
          </div>
          <RankList ranked={lookupRanked} expected={null} />
        </Card>

        <Card style={{ background: C.accentSoft, borderColor: `${C.accent}33` }}>
          <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 10, color: "#5d4226" }}>
            What is hand-built here, and what is not
          </div>
          <p style={{ fontSize: 14, lineHeight: 1.65, color: "#5d4226", margin: "0 0 10px", maxWidth: "66ch" }}>
            This is an illustrative space with five named axes, and the {WORDS.length} words were placed by hand so the
            textbook analogies close. The cosine similarities and the{" "}
            <span style={{ fontFamily: MONO }}>a - b + c</span> arithmetic are computed straight from those
            coordinates, and the nearest word is found by ranking the whole vocabulary, never by special-casing the
            expected answer.
          </p>
          <p style={{ fontSize: 14, lineHeight: 1.65, color: "#5d4226", margin: 0, maxWidth: "66ch" }}>
            A learned embedding works the same way geometrically, with one honest difference: its dimensions are not
            assigned by a person and carry no labels. A model reads enormous amounts of text and discovers a few hundred
            directions that happen to separate words usefully. No single axis says gender, yet the gender step still
            exists as some consistent direction through the cloud, which is why the same arithmetic carries over.
          </p>
        </Card>

        <p style={{ textAlign: "center", fontSize: 11.5, color: C.faint, marginTop: 22, lineHeight: 1.6 }}>
          Each result is the vector a - b + c over the five coordinates, and each ranking is the cosine between that
          vector and every word in the vocabulary, sorted high to low.
        </p>
      </div>
    </div>
  );
}
