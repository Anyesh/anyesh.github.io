import { useState, useEffect, useMemo } from "react";

export const meta = {
  title: "The Interactive Byte",
  category: "Computer Systems",
  description:
    "Eight bits you can toggle, with place values, running sums, and live binary, decimal, hex, octal, and ASCII readouts that stay in sync. Then a bitwise playground where AND, OR, XOR, NOT, and shifts are computed column by column.",
  date: "2026-06-02",
  tags: ["bits", "bytes", "binary", "bitwise", "computer-systems"],
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
  on: "#c0561f",
  off: "#ffffff",
  blue: "#2e6f8e",
  blueSoft: "#e8f0f4",
  good: "#2f7d53",
};

const EASE = "cubic-bezier(0.23, 1, 0.32, 1)";
const MONO =
  "'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";

const PLACE = [128, 64, 32, 16, 8, 4, 2, 1];

function bitsFromByte(value) {
  const v = value & 0xff;
  const out = new Array(8);
  for (let i = 0; i < 8; i++) {
    out[i] = (v >> (7 - i)) & 1;
  }
  return out;
}

function byteFromBits(bits) {
  let v = 0;
  for (let i = 0; i < 8; i++) {
    v = (v << 1) | (bits[i] & 1);
  }
  return v & 0xff;
}

function toBin(value) {
  return (value & 0xff).toString(2).padStart(8, "0");
}

function toHex(value) {
  return (value & 0xff).toString(16).toUpperCase().padStart(2, "0");
}

function toOct(value) {
  return (value & 0xff).toString(8).padStart(3, "0");
}

const ASCII_NAMES = {
  0: "NUL",
  7: "BEL",
  8: "BS",
  9: "TAB",
  10: "LF",
  13: "CR",
  27: "ESC",
  32: "space",
  127: "DEL",
};

function asciiDescription(value) {
  const v = value & 0xff;
  if (v >= 32 && v <= 126) {
    return { printable: true, glyph: v === 32 ? "space" : String.fromCharCode(v) };
  }
  return { printable: false, glyph: ASCII_NAMES[v] || "non-printable control code" };
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

function Bit({ place, index, on, onToggle, reduce, color }) {
  const accent = color || C.accent;
  return (
    <button
      type="button"
      role="switch"
      aria-checked={on === 1}
      aria-label={`Bit ${7 - index}, place value ${place}, currently ${on}`}
      onClick={onToggle}
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: 4,
        flex: "1 1 0",
        minWidth: 0,
        background: "transparent",
        border: "none",
        padding: 0,
        cursor: "pointer",
        fontFamily: "inherit",
      }}
    >
      <span style={{ fontSize: 10, color: C.muted, fontFamily: MONO }}>
        {7 - index}
      </span>
      <span
        aria-hidden="true"
        style={{
          width: "100%",
          maxWidth: 46,
          aspectRatio: "1 / 1",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          borderRadius: 9,
          fontFamily: MONO,
          fontSize: "clamp(15px, 4.6vw, 22px)",
          fontWeight: 700,
          color: on ? "#fff" : C.muted,
          background: on ? accent : C.off,
          border: `1.5px solid ${on ? accent : C.border}`,
          boxShadow: on ? `0 2px 10px ${accent}33` : "none",
          transform: on && !reduce ? "translateY(-1px)" : "none",
          transition: reduce
            ? "none"
            : `background 180ms ${EASE}, color 160ms ease, transform 180ms ${EASE}, box-shadow 200ms ${EASE}`,
        }}
      >
        {on}
      </span>
      <span
        style={{
          fontSize: "clamp(8.5px, 2.4vw, 11px)",
          fontFamily: MONO,
          color: on ? accent : C.muted,
          fontWeight: on ? 700 : 400,
          transition: reduce ? "none" : "color 160ms ease",
        }}
      >
        {place}
      </span>
    </button>
  );
}

function ByteRow({ bits, onToggle, reduce, color, ariaLabel }) {
  return (
    <div
      role="group"
      aria-label={ariaLabel}
      style={{
        display: "flex",
        gap: "clamp(3px, 1.4vw, 8px)",
        alignItems: "flex-end",
      }}
    >
      {bits.map((b, i) => (
        <Bit
          key={i}
          index={i}
          place={PLACE[i]}
          on={b}
          color={color}
          reduce={reduce}
          onToggle={() => onToggle(i)}
        />
      ))}
    </div>
  );
}

function RunningSum({ bits, reduce }) {
  const terms = [];
  for (let i = 0; i < 8; i++) {
    if (bits[i]) terms.push(PLACE[i]);
  }
  const total = terms.reduce((a, b) => a + b, 0);
  return (
    <div
      aria-live="polite"
      style={{
        fontFamily: MONO,
        fontSize: "clamp(12px, 3.4vw, 15px)",
        color: C.ink,
        background: C.faint,
        borderRadius: 10,
        padding: "11px 14px",
        lineHeight: 1.7,
        textAlign: "center",
        transition: reduce ? "none" : "background 200ms ease",
      }}
    >
      {terms.length === 0 ? (
        <span style={{ color: C.muted }}>no bits set, the sum is 0</span>
      ) : (
        <span>
          {terms.map((t, i) => (
            <span key={i}>
              <b style={{ color: C.accent }}>{t}</b>
              {i < terms.length - 1 ? <span style={{ color: C.muted }}> + </span> : null}
            </span>
          ))}
          <span style={{ color: C.muted }}> = </span>
          <b style={{ color: C.ink, fontSize: "1.1em" }}>{total}</b>
        </span>
      )}
    </div>
  );
}

function Readout({ label, value, mono, hint, accent }) {
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
          letterSpacing: "0.07em",
          color: C.muted,
          fontWeight: 700,
          marginBottom: 5,
        }}
      >
        {label}
      </div>
      <div
        style={{
          fontFamily: mono ? MONO : "inherit",
          fontSize: 19,
          fontWeight: 700,
          color: accent ? C.accent : C.ink,
          wordBreak: "break-all",
          lineHeight: 1.15,
        }}
      >
        {value}
      </div>
      {hint && (
        <div style={{ fontSize: 11, color: C.muted, marginTop: 4, lineHeight: 1.45 }}>
          {hint}
        </div>
      )}
    </div>
  );
}

function EditableField({ id, label, value, onCommit, parse, format, placeholder, prefix }) {
  const [draft, setDraft] = useState("");
  const [editing, setEditing] = useState(false);
  const [error, setError] = useState(false);
  const shown = editing ? draft : format(value);

  function commit(raw) {
    const parsed = parse(raw);
    if (parsed === null) {
      setError(true);
      return;
    }
    setError(false);
    onCommit(parsed & 0xff);
  }

  return (
    <div>
      <label
        htmlFor={id}
        style={{
          display: "block",
          fontSize: 11,
          color: C.muted,
          fontWeight: 600,
          marginBottom: 5,
        }}
      >
        {label}
      </label>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 0,
          border: `1px solid ${error ? C.accent : C.border}`,
          borderRadius: 8,
          background: "#fff",
          overflow: "hidden",
        }}
      >
        {prefix && (
          <span
            style={{
              fontFamily: MONO,
              fontSize: 13,
              color: C.muted,
              padding: "8px 0 8px 11px",
            }}
          >
            {prefix}
          </span>
        )}
        <input
          id={id}
          type="text"
          inputMode="text"
          value={shown}
          placeholder={placeholder}
          aria-label={label}
          aria-invalid={error}
          onFocus={(e) => {
            setEditing(true);
            setDraft(format(value));
            setError(false);
            e.target.select();
          }}
          onChange={(e) => {
            setDraft(e.target.value);
            const parsed = parse(e.target.value);
            setError(parsed === null && e.target.value.trim() !== "");
            if (parsed !== null) onCommit(parsed & 0xff);
          }}
          onBlur={(e) => {
            setEditing(false);
            if (e.target.value.trim() !== "") commit(e.target.value);
            else setError(false);
          }}
          onKeyDown={(e) => {
            if (e.key === "Enter") e.currentTarget.blur();
          }}
          style={{
            flex: 1,
            minWidth: 0,
            border: "none",
            outline: "none",
            background: "transparent",
            fontFamily: MONO,
            fontSize: 14,
            color: C.ink,
            padding: prefix ? "8px 11px 8px 4px" : "8px 11px",
          }}
        />
      </div>
    </div>
  );
}

function NibbleView({ value }) {
  const high = (value >> 4) & 0xf;
  const low = value & 0xf;
  const nibbles = [
    { label: "high nibble", bits: bitsFromByte(value).slice(0, 4), digit: high },
    { label: "low nibble", bits: bitsFromByte(value).slice(4, 8), digit: low },
  ];
  return (
    <div style={{ display: "flex", gap: 10 }}>
      {nibbles.map((nb, ni) => (
        <div
          key={ni}
          style={{
            flex: 1,
            background: ni === 0 ? C.accentSoft : C.blueSoft,
            border: `1px solid ${ni === 0 ? C.accent + "33" : C.blue + "33"}`,
            borderRadius: 10,
            padding: "10px 12px",
          }}
        >
          <div
            style={{
              fontSize: 10,
              textTransform: "uppercase",
              letterSpacing: "0.06em",
              color: ni === 0 ? C.accent : C.blue,
              fontWeight: 700,
              marginBottom: 6,
            }}
          >
            {nb.label}
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span style={{ fontFamily: MONO, fontSize: 16, color: C.ink, fontWeight: 700 }}>
              {nb.bits.join("")}
            </span>
            <span style={{ color: C.muted }}>=</span>
            <span
              style={{
                fontFamily: MONO,
                fontSize: 20,
                fontWeight: 700,
                color: ni === 0 ? C.accent : C.blue,
              }}
            >
              {nb.digit.toString(16).toUpperCase()}
            </span>
          </div>
        </div>
      ))}
    </div>
  );
}

const PRESETS = [
  { label: "0", value: 0, note: "all bits off" },
  { label: "'A' = 65", value: 65, note: "0100 0001" },
  { label: "'a' = 97", value: 97, note: "lowercase, +32" },
  { label: "64", value: 64, note: "one power of two" },
  { label: "255", value: 255, note: "all bits on" },
];

function ByteExplorer({ reduce }) {
  const [value, setValue] = useState(65);
  const bits = useMemo(() => bitsFromByte(value), [value]);
  const ascii = asciiDescription(value);

  const toggle = (i) => {
    const next = [...bits];
    next[i] = next[i] ? 0 : 1;
    setValue(byteFromBits(next));
  };

  const parseDecimal = (raw) => {
    const s = raw.trim();
    if (s === "") return null;
    if (!/^\d+$/.test(s)) return null;
    const n = parseInt(s, 10);
    if (!Number.isFinite(n) || n < 0 || n > 255) return null;
    return n;
  };
  const parseHex = (raw) => {
    const s = raw.trim().replace(/^0x/i, "");
    if (s === "") return null;
    if (!/^[0-9a-fA-F]{1,2}$/.test(s)) return null;
    return parseInt(s, 16);
  };

  return (
    <Card style={{ marginBottom: 16 }}>
      <Eyebrow>The byte</Eyebrow>
      <p style={{ fontSize: 13, color: C.ink, lineHeight: 1.65, margin: "0 0 16px" }}>
        Click a bit to flip it, or tab to it and press space or enter. Each bit owns a place value,
        a power of two. The byte is just the sum of the place values whose bit is 1.
      </p>

      <ByteRow
        bits={bits}
        onToggle={toggle}
        reduce={reduce}
        ariaLabel="Eight togglable bits with their place values"
      />

      <div style={{ marginTop: 16 }}>
        <RunningSum bits={bits} reduce={reduce} />
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(132px, 1fr))",
          gap: 10,
          marginTop: 16,
        }}
      >
        <Readout label="Binary" value={toBin(value)} mono accent />
        <Readout label="Unsigned decimal" value={value} mono />
        <Readout label="Hexadecimal" value={`0x${toHex(value)}`} mono />
        <Readout label="Octal" value={`0o${toOct(value)}`} mono />
        <Readout
          label="ASCII"
          value={ascii.printable ? `'${ascii.glyph}'` : ascii.glyph}
          mono={ascii.printable}
          hint={
            ascii.printable
              ? "in the printable range 32 to 126"
              : "outside the printable range, no glyph"
          }
        />
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))",
          gap: 12,
          marginTop: 16,
        }}
      >
        <EditableField
          id="dec-input"
          label="Set by decimal (0 to 255)"
          value={value}
          parse={parseDecimal}
          format={(v) => String(v)}
          placeholder="0 - 255"
          onCommit={setValue}
        />
        <EditableField
          id="hex-input"
          label="Set by hex (00 to FF)"
          value={value}
          parse={parseHex}
          format={(v) => toHex(v)}
          placeholder="00 - FF"
          prefix="0x"
          onCommit={setValue}
        />
      </div>

      <div style={{ marginTop: 16 }}>
        <div style={{ fontSize: 11, color: C.muted, fontWeight: 600, marginBottom: 8 }}>
          Each hex digit is exactly four bits, one nibble:
        </div>
        <NibbleView value={value} />
      </div>

      <div
        style={{
          display: "flex",
          gap: 7,
          flexWrap: "wrap",
          marginTop: 16,
          alignItems: "center",
        }}
      >
        <span style={{ fontSize: 11, color: C.muted, fontWeight: 600, marginRight: 2 }}>
          Presets:
        </span>
        {PRESETS.map((p) => (
          <Pill
            key={p.label}
            onClick={() => setValue(p.value)}
            active={value === p.value}
            ariaLabel={`Set byte to ${p.value}, ${p.note}`}
            reduce={reduce}
          >
            {p.label}
          </Pill>
        ))}
      </div>

      <p
        style={{
          fontSize: 12.5,
          color: C.muted,
          lineHeight: 1.6,
          margin: "16px 0 0",
          maxWidth: "64ch",
        }}
      >
        A bit is one binary digit, a 0 or a 1. Eight of them make a byte, so a byte holds{" "}
        <b style={{ color: C.ink }}>2^8 = 256</b> distinct values, the integers 0 through 255. Hex is
        convenient because one hex digit covers exactly four bits, so a byte is always two clean hex
        digits with no leftover.
      </p>
    </Card>
  );
}

const OPS = [
  { id: "and", label: "A AND B", symbol: "&", binary: true },
  { id: "or", label: "A OR B", symbol: "|", binary: true },
  { id: "xor", label: "A XOR B", symbol: "^", binary: true },
  { id: "not", label: "NOT A", symbol: "~", binary: false },
  { id: "shl", label: "A << n", symbol: "<<", binary: false, shift: true },
  { id: "shr", label: "A >> n", symbol: ">>", binary: false, shift: true },
];

function computeOp(opId, a, b, shift) {
  switch (opId) {
    case "and":
      return (a & b) & 0xff;
    case "or":
      return (a | b) & 0xff;
    case "xor":
      return (a ^ b) & 0xff;
    case "not":
      return ~a & 0xff;
    case "shl":
      return (a << shift) & 0xff;
    case "shr":
      return (a >> shift) & 0xff;
    default:
      return 0;
  }
}

function opColumnNote(opId) {
  switch (opId) {
    case "and":
      return "1 only when both inputs are 1";
    case "or":
      return "1 when either input is 1";
    case "xor":
      return "1 when the inputs differ";
    case "not":
      return "every bit flipped within the byte";
    case "shl":
      return "bits slide left, low bits fill with 0, bits past position 7 fall off";
    case "shr":
      return "bits slide right, high bits fill with 0, bits past position 0 fall off";
    default:
      return "";
  }
}

function AlignedBitLine({ label, value, color, dim }) {
  const bits = bitsFromByte(value);
  return (
    <div style={{ display: "flex", alignItems: "center", gap: "clamp(6px, 2vw, 12px)" }}>
      <span
        style={{
          fontFamily: MONO,
          fontSize: 12,
          fontWeight: 700,
          color: color || C.muted,
          width: 30,
          flexShrink: 0,
          textAlign: "right",
        }}
      >
        {label}
      </span>
      <div style={{ display: "flex", gap: "clamp(2px, 1.1vw, 6px)", flex: 1 }}>
        {bits.map((b, i) => (
          <span
            key={i}
            style={{
              flex: "1 1 0",
              minWidth: 0,
              textAlign: "center",
              fontFamily: MONO,
              fontSize: "clamp(13px, 3.8vw, 17px)",
              fontWeight: 700,
              padding: "5px 0",
              borderRadius: 6,
              color: b ? "#fff" : C.muted,
              background: b ? color || C.accent : C.faint,
              opacity: dim ? 0.4 : 1,
              border: `1px solid ${b ? color || C.accent : C.border}`,
            }}
          >
            {b}
          </span>
        ))}
      </div>
      <span
        style={{
          fontFamily: MONO,
          fontSize: 12,
          color: C.muted,
          width: 58,
          flexShrink: 0,
          textAlign: "right",
        }}
      >
        0x{toHex(value)}
      </span>
    </div>
  );
}

function BitwisePlayground({ reduce }) {
  const [a, setA] = useState(0b11001010);
  const [b, setB] = useState(0b01101100);
  const [opId, setOpId] = useState("and");
  const [shift, setShift] = useState(2);

  const op = OPS.find((o) => o.id === opId);
  const result = computeOp(opId, a, b, shift);

  const toggleA = (i) => {
    const next = bitsFromByte(a);
    next[i] = next[i] ? 0 : 1;
    setA(byteFromBits(next));
  };
  const toggleB = (i) => {
    const next = bitsFromByte(b);
    next[i] = next[i] ? 0 : 1;
    setB(byteFromBits(next));
  };

  let expr;
  if (op.shift) expr = `${a} ${op.symbol} ${shift}`;
  else if (op.binary) expr = `${a} ${op.symbol} ${b}`;
  else expr = `${op.symbol}${a}`;

  return (
    <Card style={{ marginBottom: 16 }}>
      <Eyebrow>Bitwise playground</Eyebrow>
      <p style={{ fontSize: 13, color: C.ink, lineHeight: 1.65, margin: "0 0 16px" }}>
        Two operands you can edit bit by bit. Pick an operation and read the result computed one
        column at a time. Every result is masked back to eight bits, so what you see is always a
        byte.
      </p>

      <div style={{ marginBottom: 6 }}>
        <div style={{ fontSize: 11, color: C.muted, fontWeight: 600, marginBottom: 8 }}>
          Operand A
        </div>
        <ByteRow bits={bitsFromByte(a)} onToggle={toggleA} reduce={reduce} color={C.accent} ariaLabel="Operand A bits" />
      </div>
      <div style={{ marginTop: 14 }}>
        <div style={{ fontSize: 11, color: C.muted, fontWeight: 600, marginBottom: 8 }}>
          Operand B
        </div>
        <ByteRow bits={bitsFromByte(b)} onToggle={toggleB} reduce={reduce} color={C.blue} ariaLabel="Operand B bits" />
      </div>

      <div style={{ display: "flex", gap: 7, flexWrap: "wrap", marginTop: 18 }}>
        {OPS.map((o) => (
          <Pill
            key={o.id}
            onClick={() => setOpId(o.id)}
            active={opId === o.id}
            ariaLabel={`Compute ${o.label}`}
            reduce={reduce}
          >
            {o.label}
          </Pill>
        ))}
      </div>

      {op.shift && (
        <div style={{ marginTop: 14 }}>
          <label
            htmlFor="shift-amount"
            style={{
              display: "flex",
              justifyContent: "space-between",
              fontSize: 12,
              color: C.muted,
              marginBottom: 6,
            }}
          >
            <span>shift amount (n)</span>
            <span style={{ fontWeight: 700, color: C.accent }}>{shift}</span>
          </label>
          <input
            id="shift-amount"
            type="range"
            min={0}
            max={8}
            value={shift}
            aria-label="Shift amount"
            onChange={(e) => setShift(+e.target.value)}
            style={{ width: "100%", accentColor: C.accent }}
          />
        </div>
      )}

      <div
        style={{
          marginTop: 18,
          background: "#fff",
          border: `1px solid ${C.border}`,
          borderRadius: 12,
          padding: "14px 14px 16px",
          display: "flex",
          flexDirection: "column",
          gap: 8,
        }}
      >
        <AlignedBitLine label="A" value={a} color={C.accent} />
        {op.binary && <AlignedBitLine label="B" value={b} color={C.blue} />}
        {op.shift && (
          <div style={{ fontSize: 11.5, color: C.muted, textAlign: "center", padding: "2px 0" }}>
            {op.symbol} {shift}
          </div>
        )}
        <div
          style={{
            height: 1,
            background: C.border,
            margin: "2px 0",
          }}
        />
        <AlignedBitLine label="=" value={result} color={C.good} />
      </div>

      <div
        aria-live="polite"
        style={{
          marginTop: 12,
          background: C.faint,
          borderRadius: 10,
          padding: "11px 14px",
          fontFamily: MONO,
          fontSize: 13,
          color: C.ink,
          lineHeight: 1.6,
        }}
      >
        {expr} = <b style={{ color: C.good }}>{result}</b>
        <span style={{ color: C.muted }}> = 0x{toHex(result)} = </span>
        <span style={{ color: C.good }}>{toBin(result)}</span>
        <div style={{ fontFamily: "inherit", fontSize: 12, color: C.muted, marginTop: 6 }}>
          {opColumnNote(opId)}
        </div>
      </div>

      <p style={{ fontSize: 12.5, color: C.muted, lineHeight: 1.6, margin: "16px 0 0", maxWidth: "64ch" }}>
        These are real JavaScript bitwise operators. JS evaluates them on 32-bit integers, so the
        result is masked with <code style={{ fontFamily: MONO, color: C.ink }}>&amp; 0xff</code> to
        keep the display a single byte. AND with a mask clears bits you do not want; OR sets flags;
        XOR toggles them; a left shift by <i>n</i> multiplies by 2^<i>n</i> until bits fall off the
        top.
      </p>
    </Card>
  );
}

export default function App() {
  const reduce = usePrefersReducedMotion();

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
        input[type="range"]:focus-visible { outline: 2px solid ${C.accent}; outline-offset: 3px; border-radius: 4px; }
        [role="switch"]:focus-visible { outline: 2px solid ${C.accent}; outline-offset: 3px; border-radius: 12px; }
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
            Computer Systems · Foundations
          </div>
          <h1 style={{ fontSize: 26, fontWeight: 700, margin: 0, textWrap: "balance" }}>
            The Interactive Byte
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
            Everything a computer stores is bits, and they travel in groups of eight called bytes.
            Toggle the eight bits below and watch the same value appear as binary, decimal, hex,
            octal, and a character, all kept in step by real conversions. Then take two bytes into
            the bitwise playground and see AND, OR, XOR, NOT, and shifts worked out column by column.
          </p>
        </header>

        <ByteExplorer reduce={reduce} />
        <BitwisePlayground reduce={reduce} />

        <div style={{ marginTop: 4, textAlign: "center", fontSize: 11, color: C.muted }}>
          Every number here is computed from the bits in real time. Place value i contributes 2^i
          when its bit is 1.
        </div>
      </div>
    </div>
  );
}
