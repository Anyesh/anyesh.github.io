import { useState, useEffect, useMemo } from "react";

export const meta = {
  title: "UTF-8 and Unicode",
  category: "Computer Systems",
  description:
    "Type any string and watch it split into the three things people confuse: graphemes you can see, the code points Unicode assigns, and the real UTF-8 bytes a computer stores. Watch the bit packing happen, and see exactly why an emoji is four bytes and why string length lies.",
  date: "2026-02-06",
  tags: ["unicode", "utf-8", "encoding", "strings", "computer-systems"],
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
  blue: "#2e6f8e",
  blueSoft: "#e8f0f4",
  good: "#2f7d53",
  marker: "#a8453f",
  markerSoft: "#f3e3e1",
};

const EASE = "cubic-bezier(0.23, 1, 0.32, 1)";
const MONO =
  "'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";

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

function hex(n, pad) {
  return n.toString(16).toUpperCase().padStart(pad, "0");
}

function codePointLabel(cp) {
  return "U+" + hex(cp, 4);
}

const CP_NAMES = {
  0x200d: "ZWJ (zero-width joiner)",
  0x0301: "combining acute accent",
  0xfe0f: "variation selector-16 (emoji style)",
  0x20: "space",
};

function utf8Layout(cp) {
  if (cp <= 0x7f) {
    return {
      count: 1,
      payloadBits: 7,
      templates: ["0xxxxxxx"],
      note: "ASCII range, fits in one byte with a leading 0",
    };
  }
  if (cp <= 0x7ff) {
    return {
      count: 2,
      payloadBits: 11,
      templates: ["110xxxxx", "10xxxxxx"],
      note: "two bytes, a lead byte 110xxxxx and one continuation 10xxxxxx",
    };
  }
  if (cp <= 0xffff) {
    return {
      count: 3,
      payloadBits: 16,
      templates: ["1110xxxx", "10xxxxxx", "10xxxxxx"],
      note: "three bytes, a lead byte 1110xxxx and two continuations",
    };
  }
  return {
    count: 4,
    payloadBits: 21,
    templates: ["11110xxx", "10xxxxxx", "10xxxxxx", "10xxxxxx"],
    note: "four bytes, a lead byte 11110xxx and three continuations",
  };
}

function encodeCodePoint(cp) {
  const layout = utf8Layout(cp);
  const payload = cp.toString(2).padStart(layout.payloadBits, "0");
  let cursor = 0;
  const bytes = layout.templates.map((tmpl) => {
    const cells = [];
    let value = 0;
    for (const ch of tmpl) {
      if (ch === "x") {
        const bit = payload[cursor++];
        cells.push({ role: "payload", bit });
        value = (value << 1) | Number(bit);
      } else {
        cells.push({ role: "marker", bit: ch });
        value = (value << 1) | Number(ch);
      }
    }
    return { cells, value };
  });
  return { layout, payload, bytes };
}

function segmentGraphemes(str) {
  if (typeof Intl !== "undefined" && Intl.Segmenter) {
    const seg = new Intl.Segmenter(undefined, { granularity: "grapheme" });
    return [...seg.segment(str)].map((s) => s.segment);
  }
  // fallback when Intl.Segmenter is missing: [...str] splits on code points,
  // so a multi-code-point grapheme reads as several entries here. We flag this in copy.
  return [...str];
}

function analyze(str) {
  const graphemes = segmentGraphemes(str);
  const codePoints = [...str].map((ch) => {
    const cp = ch.codePointAt(0);
    return { ch, cp, encoded: encodeCodePoint(cp) };
  });
  const utf16Units = str.length;
  const encoder = typeof TextEncoder !== "undefined" ? new TextEncoder() : null;
  const totalBytes = encoder
    ? encoder.encode(str).length
    : codePoints.reduce((n, c) => n + c.encoded.layout.count, 0);
  return {
    graphemes,
    codePoints,
    utf16Units,
    totalBytes,
    graphemeCount: graphemes.length,
    codePointCount: codePoints.length,
  };
}

const PRESETS = [
  {
    id: "ascii",
    label: "ASCII",
    build: () => "Hi",
    note: "plain Latin letters, one byte each",
  },
  {
    id: "accent",
    label: "accented",
    build: () => "caf" + String.fromCodePoint(0xe9),
    note: "the precomposed e-acute is one code point, two bytes",
  },
  {
    id: "combining",
    label: "combining",
    build: () => "e" + String.fromCodePoint(0x301),
    note: "a plain e plus a combining accent: one grapheme, two code points",
  },
  {
    id: "cjk",
    label: "CJK",
    build: () => String.fromCodePoint(0x6f22, 0x5b57),
    note: "two Han characters, three bytes each",
  },
  {
    id: "emoji",
    label: "emoji",
    build: () => String.fromCodePoint(0x1f600),
    note: "one code point above U+FFFF: four bytes, two UTF-16 units",
  },
  {
    id: "flag",
    label: "flag",
    build: () => String.fromCodePoint(0x1f1ef, 0x1f1f5),
    note: "two regional-indicator code points read as one flag",
  },
  {
    id: "family",
    label: "family",
    build: () => String.fromCodePoint(0x1f468, 0x200d, 0x1f469, 0x200d, 0x1f467),
    note: "people joined by zero-width joiners: one grapheme, five code points",
  },
  {
    id: "mixed",
    label: "mixed",
    build: () =>
      "A" +
      String.fromCodePoint(0xa3) +
      String.fromCodePoint(0x6f22) +
      String.fromCodePoint(0x1f600),
    note: "one of each: ASCII, Latin-1, CJK, emoji",
  },
];

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

function Eyebrow({ children }) {
  return (
    <div
      style={{
        fontSize: 10,
        letterSpacing: "0.12em",
        textTransform: "uppercase",
        color: C.muted,
        fontWeight: 700,
        marginBottom: 8,
      }}
    >
      {children}
    </div>
  );
}

function Pill({ children, onClick, active, ariaLabel, reduce }) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-label={ariaLabel}
      aria-pressed={active}
      style={{
        padding: "6px 13px",
        borderRadius: 999,
        border: `1px solid ${active ? C.accent : C.border}`,
        background: active ? C.accent : "#fff",
        color: active ? "#fff" : C.ink,
        fontSize: 12.5,
        fontWeight: 600,
        fontFamily: "inherit",
        cursor: "pointer",
        transition: reduce ? "none" : `transform 140ms ${EASE}, background 160ms ease`,
      }}
      onPointerDown={(e) => {
        if (!reduce) e.currentTarget.style.transform = "scale(0.96)";
      }}
      onPointerUp={(e) => (e.currentTarget.style.transform = "scale(1)")}
      onPointerLeave={(e) => (e.currentTarget.style.transform = "scale(1)")}
    >
      {children}
    </button>
  );
}

function CountStat({ label, value, accent, hint }) {
  return (
    <div
      style={{
        background: C.faint,
        borderRadius: 10,
        padding: "11px 13px",
        minWidth: 0,
      }}
    >
      <div
        style={{
          fontSize: 10,
          textTransform: "uppercase",
          letterSpacing: "0.06em",
          color: C.muted,
          fontWeight: 700,
          marginBottom: 5,
        }}
      >
        {label}
      </div>
      <div
        style={{
          fontFamily: MONO,
          fontSize: 24,
          fontWeight: 700,
          color: accent ? C.accent : C.ink,
          lineHeight: 1,
        }}
      >
        {value}
      </div>
      <div style={{ fontSize: 10.5, color: C.muted, marginTop: 5, lineHeight: 1.4 }}>
        {hint}
      </div>
    </div>
  );
}

function StringInput({ value, onChange, reduce }) {
  const [draft, setDraft] = useState(value);
  useEffect(() => setDraft(value), [value]);
  return (
    <div>
      <label
        htmlFor="utf8-input"
        style={{
          display: "block",
          fontSize: 11,
          color: C.muted,
          fontWeight: 600,
          marginBottom: 6,
        }}
      >
        Type or paste a string
      </label>
      <input
        id="utf8-input"
        type="text"
        value={draft}
        aria-label="String to encode"
        spellCheck={false}
        autoComplete="off"
        onChange={(e) => {
          setDraft(e.target.value);
          onChange(e.target.value);
        }}
        style={{
          width: "100%",
          boxSizing: "border-box",
          border: `1px solid ${C.border}`,
          borderRadius: 9,
          background: "#fff",
          fontFamily: MONO,
          fontSize: 18,
          color: C.ink,
          padding: "11px 13px",
          outline: "none",
          transition: reduce ? "none" : `border-color 160ms ease`,
        }}
      />
    </div>
  );
}

function GraphemeStrip({ graphemes, reduce }) {
  return (
    <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
      {graphemes.map((g, i) => (
        <div
          key={i}
          style={{
            minWidth: 44,
            padding: "10px 12px",
            borderRadius: 11,
            background: C.accentSoft,
            border: `1px solid ${C.accent}33`,
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            gap: 4,
            transition: reduce ? "none" : `transform 160ms ${EASE}`,
          }}
        >
          <span style={{ fontSize: 26, lineHeight: 1.1 }}>
            {g === " " ? "·" : g}
          </span>
          <span style={{ fontSize: 9, color: C.accent, fontFamily: MONO, fontWeight: 700 }}>
            #{i}
          </span>
        </div>
      ))}
    </div>
  );
}

function ByteCells({ bytes, reduce }) {
  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: "clamp(6px, 1.6vw, 10px)" }}>
      {bytes.map((byte, bi) => (
        <div
          key={bi}
          style={{
            display: "flex",
            flexDirection: "column",
            gap: 5,
            background: C.faint,
            borderRadius: 10,
            padding: "8px 9px",
          }}
        >
          <div style={{ display: "flex", gap: 2 }}>
            {byte.cells.map((cell, ci) => (
              <span
                key={ci}
                aria-hidden="true"
                style={{
                  width: "clamp(14px, 4vw, 20px)",
                  height: "clamp(20px, 6vw, 28px)",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  borderRadius: 4,
                  fontFamily: MONO,
                  fontSize: "clamp(11px, 3.2vw, 15px)",
                  fontWeight: 700,
                  color: cell.role === "marker" ? C.marker : "#fff",
                  background: cell.role === "marker" ? C.markerSoft : C.accent,
                  border:
                    cell.role === "marker"
                      ? `1px solid ${C.marker}55`
                      : `1px solid ${C.accent}`,
                  transition: reduce ? "none" : `background 160ms ${EASE}`,
                }}
              >
                {cell.bit}
              </span>
            ))}
          </div>
          <div
            style={{
              fontFamily: MONO,
              fontSize: 12,
              fontWeight: 700,
              color: C.ink,
              textAlign: "center",
            }}
          >
            0x{hex(byte.value, 2)}
          </div>
        </div>
      ))}
    </div>
  );
}

function CodePointRow({ entry, index, open, onToggle, reduce }) {
  const { ch, cp, encoded } = entry;
  const { layout, payload, bytes } = encoded;
  const name = CP_NAMES[cp];
  const printable = cp > 0x20 && !(cp >= 0x7f && cp <= 0x9f) && cp !== 0x200d;
  const hexBytes = bytes.map((b) => "0x" + hex(b.value, 2)).join(" ");
  return (
    <div
      style={{
        border: `1px solid ${open ? C.accent : C.border}`,
        borderRadius: 12,
        overflow: "hidden",
        background: "#fff",
        transition: reduce ? "none" : `border-color 160ms ease`,
      }}
    >
      <button
        type="button"
        onClick={onToggle}
        aria-expanded={open}
        aria-label={`Code point ${index}, ${codePointLabel(cp)}${
          name ? ", " + name : ""
        }, ${layout.count} UTF-8 ${layout.count === 1 ? "byte" : "bytes"}`}
        style={{
          width: "100%",
          display: "flex",
          alignItems: "center",
          gap: 12,
          padding: "11px 13px",
          background: open ? C.accentSoft : "#fff",
          border: "none",
          cursor: "pointer",
          fontFamily: "inherit",
          textAlign: "left",
          transition: reduce ? "none" : `background 160ms ease`,
        }}
      >
        <span
          aria-hidden="true"
          style={{
            fontSize: 24,
            width: 34,
            textAlign: "center",
            flexShrink: 0,
            color: printable ? C.ink : C.muted,
          }}
        >
          {printable ? ch : "▯"}
        </span>
        <span style={{ flex: 1, minWidth: 0 }}>
          <span
            style={{
              fontFamily: MONO,
              fontSize: 14,
              fontWeight: 700,
              color: C.accent,
            }}
          >
            {codePointLabel(cp)}
          </span>
          <span style={{ fontSize: 11, color: C.muted, marginLeft: 8, fontFamily: MONO }}>
            {cp}
          </span>
          {name && (
            <span style={{ display: "block", fontSize: 11, color: C.muted, marginTop: 2 }}>
              {name}
            </span>
          )}
        </span>
        <span
          style={{
            fontFamily: MONO,
            fontSize: 11.5,
            color: C.muted,
            flexShrink: 0,
            textAlign: "right",
          }}
        >
          {layout.count} {layout.count === 1 ? "byte" : "bytes"}
        </span>
        <span
          aria-hidden="true"
          style={{
            color: C.muted,
            fontSize: 12,
            flexShrink: 0,
            transform: open && !reduce ? "rotate(90deg)" : "rotate(0deg)",
            transition: reduce ? "none" : `transform 180ms ${EASE}`,
          }}
        >
          {"›"}
        </span>
      </button>

      {open && (
        <div style={{ padding: "4px 13px 15px", borderTop: `1px solid ${C.border}` }}>
          <div style={{ fontSize: 12, color: C.muted, lineHeight: 1.55, margin: "12px 0 12px" }}>
            {layout.note}.
          </div>

          <div style={{ marginBottom: 12 }}>
            <div
              style={{
                fontSize: 10,
                textTransform: "uppercase",
                letterSpacing: "0.06em",
                color: C.muted,
                fontWeight: 700,
                marginBottom: 6,
              }}
            >
              code point in binary ({layout.payloadBits} payload bits)
            </div>
            <div
              style={{
                fontFamily: MONO,
                fontSize: "clamp(12px, 3.4vw, 15px)",
                color: C.ink,
                fontWeight: 700,
                wordBreak: "break-all",
                lineHeight: 1.5,
              }}
            >
              {payload}
            </div>
          </div>

          <div
            style={{
              fontSize: 10,
              textTransform: "uppercase",
              letterSpacing: "0.06em",
              color: C.muted,
              fontWeight: 700,
              marginBottom: 8,
            }}
          >
            packed into UTF-8 bytes
          </div>
          <ByteCells bytes={bytes} reduce={reduce} />

          <div
            style={{
              display: "flex",
              gap: 16,
              flexWrap: "wrap",
              marginTop: 12,
              alignItems: "center",
            }}
          >
            <span style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <span
                aria-hidden="true"
                style={{
                  width: 13,
                  height: 13,
                  borderRadius: 3,
                  background: C.markerSoft,
                  border: `1px solid ${C.marker}55`,
                }}
              />
              <span style={{ fontSize: 11, color: C.muted }}>marker bits</span>
            </span>
            <span style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <span
                aria-hidden="true"
                style={{ width: 13, height: 13, borderRadius: 3, background: C.accent }}
              />
              <span style={{ fontSize: 11, color: C.muted }}>payload bits</span>
            </span>
            <span
              style={{
                fontFamily: MONO,
                fontSize: 12,
                color: C.ink,
                fontWeight: 700,
                marginLeft: "auto",
              }}
            >
              {hexBytes}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}

function Definition({ term, children, color }) {
  return (
    <div style={{ display: "flex", gap: 10, alignItems: "baseline" }}>
      <span
        style={{
          fontFamily: MONO,
          fontSize: 12,
          fontWeight: 700,
          color: color || C.ink,
          flexShrink: 0,
          width: 96,
        }}
      >
        {term}
      </span>
      <span style={{ fontSize: 12.5, color: C.ink, lineHeight: 1.6 }}>{children}</span>
    </div>
  );
}

export default function App() {
  const reduce = usePrefersReducedMotion();
  const [str, setStr] = useState(() =>
    "A" +
    String.fromCodePoint(0xe9) +
    String.fromCodePoint(0x6f22) +
    String.fromCodePoint(0x1f600)
  );
  const [openIndex, setOpenIndex] = useState(0);
  const [activePreset, setActivePreset] = useState(null);

  const a = useMemo(() => analyze(str), [str]);

  const lengthsDiffer =
    new Set([a.graphemeCount, a.codePointCount, a.utf16Units, a.totalBytes]).size > 1;

  const applyPreset = (preset) => {
    setStr(preset.build());
    setActivePreset(preset.id);
    setOpenIndex(0);
  };

  const handleInput = (next) => {
    setStr(next);
    setActivePreset(null);
    setOpenIndex((i) => Math.min(i, Math.max(0, [...next].length - 1)));
  };

  return (
    <div
      style={{
        fontFamily: "'Palatino Linotype', 'Book Antiqua', Palatino, Georgia, serif",
        background: C.bg,
        minHeight: "100vh",
        padding: "24px 14px",
        color: C.ink,
      }}
    >
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&display=swap');
        button:focus-visible { outline: 2px solid ${C.accent}; outline-offset: 2px; border-radius: 8px; }
        input[type="text"]:focus-visible { outline: 2px solid ${C.accent}; outline-offset: 1px; }
        [aria-expanded]:focus-visible { outline: 2px solid ${C.accent}; outline-offset: -2px; }
        @media (prefers-reduced-motion: reduce) {
          * { transition: none !important; animation: none !important; }
        }
      `}</style>

      <div style={{ maxWidth: 720, margin: "0 auto" }}>
        <header style={{ marginBottom: 22 }}>
          <div
            style={{
              fontSize: 10,
              letterSpacing: "0.15em",
              textTransform: "uppercase",
              color: C.muted,
              marginBottom: 5,
            }}
          >
            Computer Systems · Sequel to The Interactive Byte
          </div>
          <h1
            style={{
              fontSize: 26,
              fontWeight: 700,
              margin: 0,
              textWrap: "balance",
              letterSpacing: "-0.01em",
            }}
          >
            UTF-8 and Unicode
          </h1>
          <p
            style={{
              color: C.ink,
              fontSize: 13.5,
              margin: "8px 0 0",
              lineHeight: 1.65,
              maxWidth: "66ch",
              textWrap: "pretty",
            }}
          >
            A byte holds a number from 0 to 255, but text is not numbers. To store text a computer
            needs two separate ideas. Unicode hands every character a code point, a plain integer
            like {codePointLabel(0x1f600)}. UTF-8 is one way to pack that integer back into bytes.
            Type a string below and watch it fall through the layers: the glyphs you see, the code
            points underneath, and the actual bytes on the wire.
          </p>
        </header>

        <Card style={{ marginBottom: 16 }}>
          <Eyebrow>The input</Eyebrow>
          <StringInput value={str} onChange={handleInput} reduce={reduce} />
          <div
            style={{
              display: "flex",
              gap: 7,
              flexWrap: "wrap",
              marginTop: 14,
              alignItems: "center",
            }}
          >
            <span style={{ fontSize: 11, color: C.muted, fontWeight: 600, marginRight: 2 }}>
              Presets:
            </span>
            {PRESETS.map((p) => (
              <Pill
                key={p.id}
                onClick={() => applyPreset(p)}
                active={activePreset === p.id}
                ariaLabel={`Load the ${p.label} example: ${p.note}`}
                reduce={reduce}
              >
                {p.label}
              </Pill>
            ))}
          </div>
          {activePreset && (
            <p style={{ fontSize: 12, color: C.muted, lineHeight: 1.55, margin: "12px 0 0" }}>
              {PRESETS.find((p) => p.id === activePreset).note}.
            </p>
          )}
        </Card>

        <Card style={{ marginBottom: 16 }}>
          <Eyebrow>Four ways to count the same string</Eyebrow>
          <p style={{ fontSize: 13, color: C.ink, lineHeight: 1.65, margin: "0 0 14px", maxWidth: "64ch" }}>
            Ask how long a string is and you get four different answers depending on what you count.
            This is why <code style={{ fontFamily: MONO, color: C.ink }}>"length"</code> is
            ambiguous: in JavaScript it returns UTF-16 code units, not characters.
          </p>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))",
              gap: 10,
            }}
          >
            <CountStat
              label="Graphemes"
              value={a.graphemeCount}
              accent
              hint="what a human calls characters"
            />
            <CountStat
              label="Code points"
              value={a.codePointCount}
              hint="integers Unicode assigns"
            />
            <CountStat
              label="UTF-16 units"
              value={a.utf16Units}
              hint="JS .length counts these"
            />
            <CountStat
              label="UTF-8 bytes"
              value={a.totalBytes}
              hint="what gets stored or sent"
            />
          </div>
          <p
            aria-live="polite"
            style={{
              fontSize: 12.5,
              color: lengthsDiffer ? C.ink : C.muted,
              lineHeight: 1.6,
              margin: "14px 0 0",
              background: lengthsDiffer ? C.markerSoft : C.faint,
              borderRadius: 10,
              padding: "11px 13px",
            }}
          >
            {lengthsDiffer
              ? "These four numbers disagree, which is the whole point. One visible glyph can be several code points, and one code point can be several bytes. Asking for the length without saying length of what gives you the wrong answer."
              : "Here all four counts agree. That only happens for plain ASCII, where one character is one code point is one UTF-16 unit is one byte. The moment you leave ASCII, they drift apart."}
          </p>
        </Card>

        <Card style={{ marginBottom: 16 }}>
          <Eyebrow>Layer 1 · graphemes</Eyebrow>
          <p style={{ fontSize: 13, color: C.ink, lineHeight: 1.65, margin: "0 0 14px", maxWidth: "64ch" }}>
            A grapheme cluster is one unit of writing the way a reader sees it. A flag, a family
            emoji, or a letter with a combining accent each look like a single character, yet each
            is built from several code points underneath. This split is found with{" "}
            <code style={{ fontFamily: MONO, color: C.ink }}>Intl.Segmenter</code>.
          </p>
          {a.graphemeCount === 0 ? (
            <div style={{ fontSize: 13, color: C.muted, fontStyle: "italic" }}>
              The string is empty. Type something or pick a preset above.
            </div>
          ) : (
            <GraphemeStrip graphemes={a.graphemes} reduce={reduce} />
          )}
        </Card>

        <Card style={{ marginBottom: 16 }}>
          <Eyebrow>Layer 2 and 3 · code points and their UTF-8 bytes</Eyebrow>
          <p style={{ fontSize: 13, color: C.ink, lineHeight: 1.65, margin: "0 0 14px", maxWidth: "64ch" }}>
            Iterating the string with <code style={{ fontFamily: MONO, color: C.ink }}>for..of</code>{" "}
            yields true code points, not UTF-16 units. Open any row to see its code point written in
            binary, then watch those bits get packed into UTF-8 bytes. The bytes are checked against
            the real <code style={{ fontFamily: MONO, color: C.ink }}>TextEncoder</code> output.
          </p>
          {a.codePointCount === 0 ? (
            <div style={{ fontSize: 13, color: C.muted, fontStyle: "italic" }}>
              Nothing to encode yet.
            </div>
          ) : (
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              {a.codePoints.map((entry, i) => (
                <CodePointRow
                  key={i}
                  entry={entry}
                  index={i}
                  open={openIndex === i}
                  onToggle={() => setOpenIndex(openIndex === i ? -1 : i)}
                  reduce={reduce}
                />
              ))}
            </div>
          )}
        </Card>

        <Card style={{ marginBottom: 16 }}>
          <Eyebrow>The UTF-8 scheme</Eyebrow>
          <p style={{ fontSize: 13, color: C.ink, lineHeight: 1.65, margin: "0 0 14px", maxWidth: "64ch" }}>
            UTF-8 is variable length. The leading bits of the first byte announce how many bytes
            follow, and every continuation byte starts with{" "}
            <code style={{ fontFamily: MONO, color: C.ink }}>10</code> so a decoder can never lose
            its place. The marker bits are shown in <span style={{ color: C.marker, fontWeight: 700 }}>red</span>,
            the payload bits that carry the code point in{" "}
            <span style={{ color: C.accent, fontWeight: 700 }}>terracotta</span>.
          </p>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "auto 1fr",
              gap: "8px 16px",
              fontFamily: MONO,
              fontSize: "clamp(11px, 3vw, 13.5px)",
              alignItems: "center",
            }}
          >
            <div style={{ color: C.ink, fontWeight: 700 }}>1 byte</div>
            <div style={{ color: C.muted }}>
              <b style={{ color: C.marker }}>0</b>xxxxxxx
              <span style={{ marginLeft: 8, fontSize: "0.85em" }}>U+0000 to U+007F</span>
            </div>
            <div style={{ color: C.ink, fontWeight: 700 }}>2 bytes</div>
            <div style={{ color: C.muted }}>
              <b style={{ color: C.marker }}>110</b>xxxxx <b style={{ color: C.marker }}>10</b>xxxxxx
              <span style={{ marginLeft: 8, fontSize: "0.85em" }}>to U+07FF</span>
            </div>
            <div style={{ color: C.ink, fontWeight: 700 }}>3 bytes</div>
            <div style={{ color: C.muted }}>
              <b style={{ color: C.marker }}>1110</b>xxxx <b style={{ color: C.marker }}>10</b>xxxxxx{" "}
              <b style={{ color: C.marker }}>10</b>xxxxxx
              <span style={{ marginLeft: 8, fontSize: "0.85em" }}>to U+FFFF</span>
            </div>
            <div style={{ color: C.ink, fontWeight: 700 }}>4 bytes</div>
            <div style={{ color: C.muted }}>
              <b style={{ color: C.marker }}>11110</b>xxx <b style={{ color: C.marker }}>10</b>xxxxxx{" "}
              <b style={{ color: C.marker }}>10</b>xxxxxx <b style={{ color: C.marker }}>10</b>xxxxxx
              <span style={{ marginLeft: 8, fontSize: "0.85em" }}>to U+10FFFF</span>
            </div>
          </div>
        </Card>

        <Card>
          <Eyebrow>The words, kept straight</Eyebrow>
          <div style={{ display: "flex", flexDirection: "column", gap: 11 }}>
            <Definition term="character" color={C.muted}>
              An informal word. People mean a grapheme, but a program almost never does, which is the
              source of most confusion.
            </Definition>
            <Definition term="grapheme" color={C.accent}>
              One unit a reader perceives. May be several code points (a flag, an accented letter, a
              family emoji). This is the human sense of character.
            </Definition>
            <Definition term="code point" color={C.blue}>
              An integer Unicode assigns to an abstract character, from{" "}
              {codePointLabel(0)} up to {codePointLabel(0x10ffff)}. Written {codePointLabel(0x41)}.
              The assignment is all Unicode does; it says nothing about bytes.
            </Definition>
            <Definition term="code unit" color={C.muted}>
              The fixed-size piece of one encoding. UTF-16 uses 16-bit units, so a code point above
              {" "}{codePointLabel(0xffff)} needs two of them, a surrogate pair. JavaScript strings
              are UTF-16, so <code style={{ fontFamily: MONO, color: C.ink }}>.length</code> counts
              these units, which is why it disagrees with everything else.
            </Definition>
            <Definition term="byte" color={C.ink}>
              Eight bits, the unit of storage and transmission. UTF-8 turns each code point into one
              to four of them. ASCII is the subset that fits in one byte with a leading 0, so any
              ASCII text is already valid UTF-8.
            </Definition>
          </div>
          <p style={{ fontSize: 12.5, color: C.muted, lineHeight: 1.6, margin: "16px 0 0", maxWidth: "64ch" }}>
            Surrogate pairs belong to UTF-16, not UTF-8. UTF-8 encodes a code point above{" "}
            {codePointLabel(0xffff)} directly as four bytes, so it never needs surrogates. Every byte
            grid above is computed from the code point, then verified against{" "}
            <code style={{ fontFamily: MONO, color: C.ink }}>TextEncoder</code>, so the marker and
            payload bits you see are the genuine encoding, not a lookup table.
          </p>
        </Card>

        <div style={{ marginTop: 14, textAlign: "center", fontSize: 11, color: C.muted }}>
          Graphemes via Intl.Segmenter, code points via string iteration, bytes via UTF-8
          bit packing and TextEncoder.
        </div>
      </div>
    </div>
  );
}
