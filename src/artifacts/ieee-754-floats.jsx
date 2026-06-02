import { useState, useMemo, useEffect, useCallback } from "react";

export const meta = {
  title: "IEEE 754 Floating Point",
  category: "Computer Systems",
  description:
    "Toggle the real 32 bits a float is made of, type a decimal and watch the nearest representable value snap into place, and see honestly why 0.1 plus 0.2 misses 0.3. Every bit pattern is the exact one your hardware stores.",
  date: "2026-05-11",
  tags: ["ieee-754", "floating-point", "binary", "computer-systems"],
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
  sign: "#a8453f",
  signSoft: "#f7e7e5",
  exp: "#2e6f8e",
  expSoft: "#e4eef2",
  mant: "#5b7a3a",
  mantSoft: "#eaf0e1",
  good: "#2f7d53",
  warn: "#9a6a1a",
};

const EASE = "cubic-bezier(0.23, 1, 0.32, 1)";
const MONO = "'IBM Plex Mono', 'SF Mono', ui-monospace, Menlo, monospace";

const FORMATS = {
  32: { exponentBits: 8, mantissaBits: 23, bias: 127, totalBits: 32 },
  64: { exponentBits: 11, mantissaBits: 52, bias: 1023, totalBits: 64 },
};

function f32ToBits(value) {
  const buf = new ArrayBuffer(4);
  new Float32Array(buf)[0] = value;
  return new Uint32Array(buf)[0] >>> 0;
}

function bitsToF32(bits) {
  const buf = new ArrayBuffer(4);
  new Uint32Array(buf)[0] = bits >>> 0;
  return new Float32Array(buf)[0];
}

function f64ToBits(value) {
  const buf = new ArrayBuffer(8);
  new Float64Array(buf)[0] = value;
  const u = new Uint32Array(buf);
  return (BigInt(u[1] >>> 0) << 32n) | BigInt(u[0] >>> 0);
}

function bitsToF64(bits) {
  const buf = new ArrayBuffer(8);
  const u = new Uint32Array(buf);
  u[0] = Number(bits & 0xffffffffn);
  u[1] = Number((bits >> 32n) & 0xffffffffn);
  return new Float64Array(buf)[0];
}

function valueToBits(value, fmt) {
  return fmt.totalBits === 32 ? BigInt(f32ToBits(value)) : f64ToBits(value);
}

function bitsToValue(bits, fmt) {
  return fmt.totalBits === 32 ? bitsToF32(Number(bits)) : bitsToF64(bits);
}

function getBit(bits, index, fmt) {
  const pos = BigInt(fmt.totalBits - 1 - index);
  return Number((bits >> pos) & 1n);
}

function toggleBit(bits, index, fmt) {
  const pos = BigInt(fmt.totalBits - 1 - index);
  return bits ^ (1n << pos);
}

function fields(bits, fmt) {
  const sign = Number((bits >> BigInt(fmt.totalBits - 1)) & 1n);
  const expMask = (1n << BigInt(fmt.exponentBits)) - 1n;
  const exponent = Number((bits >> BigInt(fmt.mantissaBits)) & expMask);
  const mantMask = (1n << BigInt(fmt.mantissaBits)) - 1n;
  const mantissa = bits & mantMask;
  return { sign, exponent, mantissa };
}

function classify(bits, fmt) {
  const { exponent, mantissa } = fields(bits, fmt);
  const allOnes = (1 << fmt.exponentBits) - 1;
  if (exponent === allOnes) return mantissa === 0n ? "infinity" : "nan";
  if (exponent === 0) return mantissa === 0n ? "zero" : "subnormal";
  return "normal";
}

// Exact decimal expansion of value = mantissaInt * 2^scale, computed in BigInt so
// the displayed digits are the true rational the bit pattern names, not a rounded double.
function exactDecimal(mantissaInt, scale, negative, maxFrac = 60) {
  if (mantissaInt === 0n) return negative ? "-0" : "0";
  const sign = negative ? "-" : "";
  if (scale >= 0n) {
    const intPart = mantissaInt << scale;
    return sign + intPart.toString();
  }
  const shift = -scale;
  const denom = 1n << shift;
  let intPart = mantissaInt / denom;
  let rem = mantissaInt % denom;
  let frac = "";
  let guard = 0;
  while (rem !== 0n && guard < maxFrac) {
    rem *= 10n;
    const digit = rem / denom;
    frac += digit.toString();
    rem %= denom;
    guard++;
  }
  const truncated = rem !== 0n;
  let out = sign + intPart.toString();
  if (frac.length > 0) out += "." + frac + (truncated ? "…" : "");
  return out;
}

function decodeExact(bits, fmt) {
  const { sign, exponent, mantissa } = fields(bits, fmt);
  const kind = classify(bits, fmt);
  const negative = sign === 1;
  if (kind === "zero") return { kind, text: negative ? "-0" : "0" };
  if (kind === "infinity") return { kind, text: negative ? "-Infinity" : "Infinity" };
  if (kind === "nan") return { kind, text: "NaN" };
  const mantBits = BigInt(fmt.mantissaBits);
  if (kind === "subnormal") {
    const scale = BigInt(1 - fmt.bias) - mantBits;
    return { kind, text: exactDecimal(mantissa, scale, negative) };
  }
  const significand = (1n << mantBits) | mantissa;
  const scale = BigInt(exponent - fmt.bias) - mantBits;
  return { kind, text: exactDecimal(significand, scale, negative) };
}

function ulpExact(bits, fmt) {
  const kind = classify(bits, fmt);
  if (kind === "infinity" || kind === "nan") return null;
  const { exponent } = fields(bits, fmt);
  const mantBits = BigInt(fmt.mantissaBits);
  const e = exponent === 0 ? 1 : exponent;
  const scale = BigInt(e - fmt.bias) - mantBits;
  return exactDecimal(1n, scale, false);
}

function nextBits(bits, fmt, dir) {
  const max = (1n << BigInt(fmt.totalBits)) - 1n;
  if (dir > 0) return bits >= max ? bits : bits + 1n;
  return bits <= 0n ? bits : bits - 1n;
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

function Eyebrow({ children, color = C.muted }) {
  return (
    <div
      style={{
        fontSize: 10.5,
        fontWeight: 700,
        letterSpacing: "0.08em",
        textTransform: "uppercase",
        color,
        marginBottom: 6,
      }}
    >
      {children}
    </div>
  );
}

function Bit({ value, index, field, onToggle, reduce, recent }) {
  const palette = {
    sign: { on: C.sign, soft: C.signSoft },
    exp: { on: C.exp, soft: C.expSoft },
    mant: { on: C.mant, soft: C.mantSoft },
  }[field];
  const on = value === 1;
  return (
    <button
      type="button"
      onClick={() => onToggle(index)}
      aria-label={`${field} bit ${index}, currently ${value}. Activate to flip.`}
      style={{
        width: 19,
        height: 26,
        padding: 0,
        borderRadius: 4,
        border: `1px solid ${on ? palette.on : C.border}`,
        background: on ? palette.on : palette.soft,
        color: on ? "#fff" : palette.on,
        fontFamily: MONO,
        fontSize: 12,
        fontWeight: 600,
        cursor: "pointer",
        boxShadow: recent && !reduce ? `0 0 0 3px ${palette.on}44` : "none",
        transition: reduce
          ? "none"
          : `background 160ms ${EASE}, border-color 160ms ${EASE}, box-shadow 220ms ${EASE}, transform 120ms ${EASE}`,
      }}
      onPointerDown={(e) => {
        if (!reduce) e.currentTarget.style.transform = "scale(0.9)";
      }}
      onPointerUp={(e) => {
        e.currentTarget.style.transform = "scale(1)";
      }}
      onPointerLeave={(e) => {
        e.currentTarget.style.transform = "scale(1)";
      }}
    >
      {value}
    </button>
  );
}

function BitField({ label, color, soft, bits, range, fmt, onToggle, reduce, recent, note }) {
  return (
    <div style={{ flex: range.length, minWidth: 0 }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "baseline",
          marginBottom: 6,
        }}
      >
        <span style={{ fontSize: 11, fontWeight: 700, color }}>{label}</span>
        <span style={{ fontSize: 10.5, color: C.muted }}>{note}</span>
      </div>
      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          gap: 3,
          padding: 6,
          borderRadius: 8,
          background: soft,
          border: `1px solid ${color}33`,
        }}
      >
        {range.map((idx) => (
          <Bit
            key={idx}
            index={idx}
            value={getBit(bits, idx, fmt)}
            field={label === "sign" ? "sign" : label === "exponent" ? "exp" : "mant"}
            onToggle={onToggle}
            reduce={reduce}
            recent={recent === idx}
          />
        ))}
      </div>
    </div>
  );
}

function Readout({ label, value, color, bg, mono = true, title }) {
  return (
    <div
      title={title}
      style={{
        background: bg,
        borderRadius: 10,
        padding: "10px 12px",
        minWidth: 0,
        overflow: "hidden",
      }}
    >
      <div
        style={{
          fontSize: 10,
          fontWeight: 700,
          letterSpacing: "0.05em",
          textTransform: "uppercase",
          color,
          marginBottom: 4,
        }}
      >
        {label}
      </div>
      <div
        style={{
          fontFamily: mono ? MONO : "inherit",
          fontSize: 13,
          color: C.ink,
          fontWeight: 600,
          wordBreak: "break-all",
        }}
      >
        {value}
      </div>
    </div>
  );
}

function Chip({ children, onClick, active }) {
  return (
    <button
      type="button"
      onClick={onClick}
      style={{
        padding: "5px 12px",
        borderRadius: 999,
        border: `1px solid ${active ? C.accent : C.border}`,
        background: active ? C.accentSoft : "#fff",
        color: active ? C.accent : C.ink,
        fontSize: 12,
        fontFamily: "inherit",
        fontWeight: active ? 600 : 500,
        cursor: "pointer",
        transition: `transform 140ms ${EASE}, border-color 140ms ease`,
      }}
      onPointerDown={(e) => (e.currentTarget.style.transform = "scale(0.96)")}
      onPointerUp={(e) => (e.currentTarget.style.transform = "scale(1)")}
      onPointerLeave={(e) => (e.currentTarget.style.transform = "scale(1)")}
    >
      {children}
    </button>
  );
}

const KIND_LABEL = {
  normal: { text: "Normal", color: C.good, bg: "#e9f2ec" },
  subnormal: { text: "Subnormal (denormal)", color: C.warn, bg: "#f5edd8" },
  zero: { text: "Zero", color: C.exp, bg: C.expSoft },
  infinity: { text: "Infinity", color: C.accent, bg: C.accentSoft },
  nan: { text: "NaN", color: C.sign, bg: C.signSoft },
};

function presets(fmt) {
  const allOnes = (1n << BigInt(fmt.exponentBits)) - 1n;
  const infBits = allOnes << BigInt(fmt.mantissaBits);
  const nanBits = infBits | (1n << BigInt(fmt.mantissaBits - 1));
  const smallestNormalBits = 1n << BigInt(fmt.mantissaBits);
  const denormBits = 1n;
  return [
    { label: "0.1", bits: valueToBits(0.1, fmt) },
    { label: "0.2", bits: valueToBits(0.2, fmt) },
    { label: "1.0", bits: valueToBits(1, fmt) },
    { label: "-0.0", bits: 1n << BigInt(fmt.totalBits - 1) },
    { label: "smallest normal", bits: smallestNormalBits },
    { label: "smallest denormal", bits: denormBits },
    { label: "infinity", bits: infBits },
    { label: "NaN", bits: nanBits },
  ];
}

export default function App() {
  const reduce = usePrefersReducedMotion();
  const [precision, setPrecision] = useState(32);
  const fmt = FORMATS[precision];
  const [bits, setBits] = useState(() => valueToBits(0.15625, FORMATS[32]));
  const [recent, setRecent] = useState(null);
  const [decimalField, setDecimalField] = useState("0.15625");
  const [decimalError, setDecimalError] = useState(null);

  const decoded = useMemo(() => decodeExact(bits, fmt), [bits, fmt]);
  const fld = useMemo(() => fields(bits, fmt), [bits, fmt]);
  const kind = decoded.kind;
  const ulp = useMemo(() => ulpExact(bits, fmt), [bits, fmt]);
  const hex = bits.toString(16).padStart(fmt.totalBits / 4, "0");

  const signRange = [0];
  const expRange = useMemo(
    () => Array.from({ length: fmt.exponentBits }, (_, i) => 1 + i),
    [fmt],
  );
  const mantRange = useMemo(
    () =>
      Array.from(
        { length: fmt.mantissaBits },
        (_, i) => 1 + fmt.exponentBits + i,
      ),
    [fmt],
  );

  const handleToggle = useCallback(
    (index) => {
      setBits((prev) => toggleBit(prev, index, fmt));
      setRecent(index);
      setDecimalError(null);
    },
    [fmt],
  );

  const setFromValue = useCallback(
    (value) => {
      const next = valueToBits(value, fmt);
      setBits(next);
      setRecent(null);
    },
    [fmt],
  );

  function applyDecimal(raw) {
    const trimmed = raw.trim();
    if (trimmed === "") {
      setDecimalError("Type a number, for example 0.1 or 6.022e23.");
      return;
    }
    if (/^-?inf(inity)?$/i.test(trimmed)) {
      setFromValue(trimmed[0] === "-" ? -Infinity : Infinity);
      setDecimalError(null);
      return;
    }
    if (/^nan$/i.test(trimmed)) {
      setFromValue(NaN);
      setDecimalError(null);
      return;
    }
    const parsed = Number(trimmed);
    if (Number.isNaN(parsed)) {
      setDecimalError("That is not a number I can parse.");
      return;
    }
    setFromValue(parsed);
    setDecimalError(null);
  }

  function changePrecision(next) {
    const value = bitsToValue(bits, fmt);
    setPrecision(next);
    const nextFmt = FORMATS[next];
    if (Number.isNaN(value)) {
      const allOnes = (1n << BigInt(nextFmt.exponentBits)) - 1n;
      setBits((allOnes << BigInt(nextFmt.mantissaBits)) | 1n);
    } else {
      setBits(valueToBits(value, nextFmt));
    }
    setRecent(null);
  }

  function step(dir) {
    setBits((prev) => nextBits(prev, fmt, dir));
    setRecent(null);
    setDecimalError(null);
  }

  const k = KIND_LABEL[kind];
  const presetList = useMemo(() => presets(fmt), [fmt]);

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
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&display=swap');
        .f754-input:focus-visible { outline: 2px solid ${C.accent}; outline-offset: 1px; }
        .f754 button:focus-visible { outline: 2px solid ${C.accent}; outline-offset: 2px; border-radius: 6px; }
        @media (prefers-reduced-motion: reduce) {
          .f754 * { transition: none !important; animation: none !important; }
        }
      `}</style>

      <div className="f754" style={{ maxWidth: 720, margin: "0 auto" }}>
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
            Computer Systems
          </div>
          <h1 style={{ fontSize: 26, fontWeight: 700, margin: 0, textWrap: "balance" }}>
            IEEE 754 Floating Point
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
            A float is one sign bit, a few exponent bits, and a fraction, packed into 32 or 64 bits.
            The value is {"(-1)"}
            <sup>sign</sup> times 1.fraction times 2 raised to (exponent minus bias). Flip any bit
            below and the decoded number changes, or type a decimal and watch the nearest value the
            hardware can actually hold snap into place.
          </p>
        </header>

        <Card style={{ marginBottom: 16 }}>
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              flexWrap: "wrap",
              gap: 10,
              marginBottom: 14,
            }}
          >
            <Eyebrow>The 32 bits, live</Eyebrow>
            <div style={{ display: "flex", gap: 6 }} role="group" aria-label="Precision">
              {[32, 64].map((p) => (
                <button
                  key={p}
                  type="button"
                  onClick={() => changePrecision(p)}
                  aria-pressed={precision === p}
                  style={{
                    padding: "5px 12px",
                    borderRadius: 8,
                    border: `1px solid ${precision === p ? C.accent : C.border}`,
                    background: precision === p ? C.accent : "#fff",
                    color: precision === p ? "#fff" : C.ink,
                    fontSize: 12,
                    fontWeight: 600,
                    fontFamily: "inherit",
                    cursor: "pointer",
                  }}
                >
                  {p}-bit
                </button>
              ))}
            </div>
          </div>

          <div style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "flex-start" }}>
            <BitField
              label="sign"
              color={C.sign}
              soft={C.signSoft}
              bits={bits}
              range={signRange}
              fmt={fmt}
              onToggle={handleToggle}
              reduce={reduce}
              recent={recent}
              note="1 bit"
            />
            <BitField
              label="exponent"
              color={C.exp}
              soft={C.expSoft}
              bits={bits}
              range={expRange}
              fmt={fmt}
              onToggle={handleToggle}
              reduce={reduce}
              recent={recent}
              note={`${fmt.exponentBits} bits`}
            />
            <BitField
              label="mantissa"
              color={C.mant}
              soft={C.mantSoft}
              bits={bits}
              range={mantRange}
              fmt={fmt}
              onToggle={handleToggle}
              reduce={reduce}
              recent={recent}
              note={`${fmt.mantissaBits} bits`}
            />
          </div>

          <div
            style={{
              marginTop: 14,
              display: "flex",
              alignItems: "center",
              gap: 10,
              flexWrap: "wrap",
            }}
          >
            <span
              style={{
                background: k.bg,
                color: k.color,
                borderRadius: 6,
                padding: "3px 10px",
                fontSize: 11,
                fontWeight: 700,
                letterSpacing: "0.04em",
                textTransform: "uppercase",
              }}
            >
              {k.text}
            </span>
            <span style={{ fontFamily: MONO, fontSize: 12, color: C.muted }}>
              0x{hex}
            </span>
          </div>

          <div
            style={{
              marginTop: 14,
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))",
              gap: 10,
            }}
          >
            <Readout
              label="sign"
              value={`${fld.sign} (${fld.sign ? "negative" : "positive"})`}
              color={C.sign}
              bg={C.signSoft}
            />
            <Readout
              label="exponent field"
              value={`${fld.exponent} - ${fmt.bias} = ${
                kind === "subnormal"
                  ? 1 - fmt.bias
                  : fld.exponent - fmt.bias
              }`}
              color={C.exp}
              bg={C.expSoft}
              title="Stored exponent minus the bias gives the real power of two."
            />
            <Readout
              label="mantissa fraction"
              value={
                kind === "subnormal"
                  ? `0.${fld.mantissa.toString(2).padStart(fmt.mantissaBits, "0")}`
                  : `1.${fld.mantissa.toString(2).padStart(fmt.mantissaBits, "0").replace(/0+$/, "") || "0"}`
              }
              color={C.mant}
              bg={C.mantSoft}
              title="The implicit leading 1 (or 0 for subnormals) followed by the stored fraction bits."
            />
          </div>

          <div
            aria-live="polite"
            style={{
              marginTop: 12,
              background: C.faint,
              borderRadius: 10,
              padding: "12px 14px",
            }}
          >
            <div
              style={{
                fontSize: 10,
                fontWeight: 700,
                letterSpacing: "0.05em",
                textTransform: "uppercase",
                color: C.muted,
                marginBottom: 5,
              }}
            >
              Exact value of this bit pattern
            </div>
            <div
              style={{
                fontFamily: MONO,
                fontSize: 14,
                fontWeight: 600,
                color: C.ink,
                wordBreak: "break-all",
                lineHeight: 1.5,
              }}
            >
              {decoded.text}
            </div>
            {ulp && kind !== "zero" && (
              <div style={{ fontSize: 12, color: C.muted, marginTop: 8, lineHeight: 1.5 }}>
                Gap to the next float (one ULP) at this magnitude:{" "}
                <span style={{ fontFamily: MONO, color: C.ink }}>{ulp}</span>
              </div>
            )}
          </div>
        </Card>

        <Card style={{ marginBottom: 16 }}>
          <Eyebrow>Type a decimal, get the real float</Eyebrow>
          <form
            onSubmit={(e) => {
              e.preventDefault();
              applyDecimal(decimalField);
            }}
            style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 10 }}
          >
            <input
              className="f754-input"
              type="text"
              inputMode="decimal"
              value={decimalField}
              onChange={(e) => setDecimalField(e.target.value)}
              aria-label="Decimal value to convert to its nearest float"
              placeholder="0.1, -3.14, 6.022e23, inf, nan"
              style={{
                flex: "1 1 160px",
                minWidth: 0,
                padding: "8px 12px",
                borderRadius: 8,
                border: `1px solid ${decimalError ? C.sign : C.border}`,
                fontSize: 13,
                fontFamily: MONO,
                color: C.ink,
                background: "#fff",
              }}
            />
            <button
              type="submit"
              style={{
                padding: "8px 18px",
                borderRadius: 8,
                border: "none",
                background: C.accent,
                color: "#fff",
                fontSize: 13,
                fontWeight: 600,
                fontFamily: "inherit",
                cursor: "pointer",
              }}
            >
              Convert
            </button>
          </form>
          {decimalError && (
            <div style={{ fontSize: 12, color: C.sign, marginBottom: 10 }}>{decimalError}</div>
          )}
          <div style={{ fontSize: 12, color: C.muted, lineHeight: 1.6, marginBottom: 12 }}>
            Your input is rounded to the nearest representable {precision}-bit float, then decoded
            back. What you typed and what gets stored often differ, and the exact stored value
            appears above.
          </div>

          <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginBottom: 14 }}>
            {presetList.map((p) => (
              <Chip
                key={p.label}
                onClick={() => {
                  setBits(p.bits);
                  setRecent(null);
                  setDecimalError(null);
                  setDecimalField(p.label);
                }}
              >
                {p.label}
              </Chip>
            ))}
          </div>

          <div style={{ display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center" }}>
            <span style={{ fontSize: 12, color: C.muted }}>Walk one float at a time:</span>
            <button
              type="button"
              onClick={() => step(-1)}
              aria-label="Previous representable float"
              style={stepBtnStyle}
            >
              previous
            </button>
            <button
              type="button"
              onClick={() => step(1)}
              aria-label="Next representable float"
              style={stepBtnStyle}
            >
              next
            </button>
            <span style={{ fontSize: 11.5, color: C.muted }}>
              (this is +/-1 on the integer bit pattern: nextafter)
            </span>
          </div>
        </Card>

        <PointZeroOneDemo
          precision={precision}
          setBits={setBits}
          setDecimalField={setDecimalField}
          fmt={fmt}
        />

        <Spacing fmt={fmt} bits={bits} />

        <Card style={{ background: C.accentSoft, border: `1px solid ${C.accent}22` }}>
          <Eyebrow color={C.accent}>What the fields mean</Eyebrow>
          <div style={{ fontSize: 13, color: C.ink, lineHeight: 1.7, display: "grid", gap: 10 }}>
            <p style={{ margin: 0 }}>
              <b>Bias on the exponent.</b> The {fmt.exponentBits}-bit exponent field stores an
              unsigned number, so to reach negative powers of two it carries a fixed offset. Subtract
              the bias ({fmt.bias}) from the stored field to get the real exponent. A stored field of{" "}
              {fmt.bias} means 2 to the power 0.
            </p>
            <p style={{ margin: 0 }}>
              <b>The implicit leading 1.</b> Normal numbers are written 1.fraction in binary, and
              that leading 1 is always there, so it is not stored. You get an extra bit of precision
              for free. Subnormals are the exception: when the exponent field is all zeros the
              leading bit is 0 instead, which lets values shrink gradually down to zero.
            </p>
            <p style={{ margin: 0 }}>
              <b>The four special cases.</b> Exponent field all zeros with a zero fraction is signed
              zero; all zeros with a nonzero fraction is a subnormal. Exponent field all ones with a
              zero fraction is infinity; all ones with a nonzero fraction is NaN. Flip the bits above
              into those shapes and the label updates.
            </p>
            <p style={{ margin: 0 }}>
              <b>Why decimal fractions break.</b> Binary fractions are sums of 1/2, 1/4, 1/8, and so
              on. A value like 0.1 is 1/10, and 10 has the factor 5, which no power of two divides,
              so 0.1 has no finite binary expansion. The hardware stores the nearest value it can,
              and that tiny rounding gap is what you see below.
            </p>
          </div>
        </Card>

        <div style={{ marginTop: 18, textAlign: "center", fontSize: 11, color: C.muted }}>
          Every bit pattern here is produced by Float32Array / Float64Array round-trips, so it is
          exactly what your CPU stores.
        </div>
      </div>
    </div>
  );
}

const stepBtnStyle = {
  padding: "6px 14px",
  borderRadius: 8,
  border: `1px solid ${C.border}`,
  background: "#fff",
  color: C.ink,
  fontSize: 12.5,
  fontFamily: "inherit",
  fontWeight: 600,
  cursor: "pointer",
};

function PointZeroOneDemo({ precision, setBits, setDecimalField, fmt }) {
  const a = valueToBits(0.1, fmt);
  const b = valueToBits(0.2, fmt);
  const target = valueToBits(0.3, fmt);
  const sum = valueToBits(0.1 + 0.2, fmt);

  const aExact = decodeExact(a, fmt).text;
  const bExact = decodeExact(b, fmt).text;
  const targetExact = decodeExact(target, fmt).text;
  const sumExact = decodeExact(sum, fmt).text;
  const equal = 0.1 + 0.2 === 0.3;

  const rows = [
    { label: "0.1 stored as", value: aExact, jump: 0.1 },
    { label: "0.2 stored as", value: bExact, jump: 0.2 },
    { label: "0.1 + 0.2 computes to", value: sumExact, jump: 0.1 + 0.2 },
    { label: "0.3 stored as", value: targetExact, jump: 0.3 },
  ];

  return (
    <Card style={{ marginBottom: 16 }}>
      <Eyebrow color={C.accent}>Why 0.1 + 0.2 is not 0.3</Eyebrow>
      <div style={{ fontSize: 13, color: C.ink, lineHeight: 1.6, marginBottom: 14, maxWidth: "66ch" }}>
        None of 0.1, 0.2, or 0.3 land exactly on a {precision}-bit float, so each is rounded the
        moment it is stored. Adding the rounded 0.1 and 0.2 lands on a different float than the
        rounded 0.3. Here are the real stored values, decoded exactly.
      </div>
      <div style={{ display: "grid", gap: 8 }}>
        {rows.map((r) => (
          <button
            key={r.label}
            type="button"
            onClick={() => {
              setBits(valueToBits(r.jump, fmt));
              setDecimalField(String(r.jump));
            }}
            style={{
              textAlign: "left",
              display: "flex",
              flexDirection: "column",
              gap: 3,
              background: C.faint,
              border: `1px solid ${C.border}`,
              borderRadius: 10,
              padding: "10px 12px",
              cursor: "pointer",
              fontFamily: "inherit",
            }}
          >
            <span style={{ fontSize: 11.5, color: C.muted, fontWeight: 600 }}>{r.label}</span>
            <span style={{ fontFamily: MONO, fontSize: 12.5, color: C.ink, wordBreak: "break-all" }}>
              {r.value}
            </span>
          </button>
        ))}
      </div>
      <div
        style={{
          marginTop: 12,
          background: equal ? "#e9f2ec" : C.signSoft,
          border: `1px solid ${equal ? C.good : C.sign}33`,
          borderRadius: 10,
          padding: "10px 14px",
          fontSize: 13,
          color: equal ? C.good : C.sign,
          lineHeight: 1.55,
        }}
      >
        {precision === 64 ? (
          <>
            In 64-bit, 0.1 + 0.2 stores as 0.30000000000000004, while 0.3 stores as
            0.299999999999999988..., so <b>0.1 + 0.2 === 0.3 is {String(equal)}</b>. The famous
            extra digit is real, and it is right there in the stored sum.
          </>
        ) : (
          <>
            At 32-bit the two rounding paths happen to land on the same float, so here{" "}
            <b>0.1 + 0.2 === 0.3 is {String(equal)}</b>. Switch to 64-bit above to see the famous
            0.30000000000000004 that JavaScript and most languages actually report.
          </>
        )}
      </div>
    </Card>
  );
}

function Spacing({ fmt, bits }) {
  const exponents = useMemo(() => {
    const out = [];
    const maxNormalExp = (1 << fmt.exponentBits) - 2;
    for (let stored = 1; stored <= maxNormalExp; stored += Math.max(1, Math.floor(maxNormalExp / 10))) {
      const realExp = stored - fmt.bias;
      const ulpBits = (BigInt(stored) << BigInt(fmt.mantissaBits));
      const ulp = ulpExact(ulpBits, fmt);
      out.push({ realExp, ulp, magnitude: Math.pow(2, realExp) });
    }
    return out;
  }, [fmt]);

  const current = useMemo(() => {
    const { exponent } = fields(bits, fmt);
    return exponent === 0 ? 1 - fmt.bias : exponent - fmt.bias;
  }, [bits, fmt]);

  const maxLog = Math.max(...exponents.map((e) => e.realExp));
  const minLog = Math.min(...exponents.map((e) => e.realExp));

  return (
    <Card style={{ marginBottom: 16 }}>
      <Eyebrow>Floats thin out as they grow</Eyebrow>
      <div style={{ fontSize: 13, color: C.ink, lineHeight: 1.6, marginBottom: 14, maxWidth: "66ch" }}>
        Between consecutive powers of two there are always the same number of floats, so the gap
        between neighbours (one ULP) doubles every time the exponent grows by one. Near zero floats
        are dense; out past large magnitudes whole integers start falling between them.
      </div>
      <div style={{ display: "grid", gap: 6 }}>
        {exponents.map((e) => {
          const frac = (e.realExp - minLog) / (maxLog - minLog || 1);
          const isCurrent = e.realExp === current;
          return (
            <div key={e.realExp} style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <span
                style={{
                  fontFamily: MONO,
                  fontSize: 11,
                  color: isCurrent ? C.accent : C.muted,
                  width: 58,
                  textAlign: "right",
                  fontWeight: isCurrent ? 700 : 500,
                }}
              >
                2^{e.realExp}
              </span>
              <div
                style={{
                  flex: 1,
                  height: 14,
                  borderRadius: 7,
                  background: C.faint,
                  overflow: "hidden",
                }}
              >
                <div
                  style={{
                    width: `${10 + frac * 90}%`,
                    height: "100%",
                    borderRadius: 7,
                    background: isCurrent ? C.accent : C.exp,
                    opacity: isCurrent ? 1 : 0.5,
                  }}
                />
              </div>
              <span
                style={{
                  fontFamily: MONO,
                  fontSize: 10.5,
                  color: C.muted,
                  width: 132,
                  whiteSpace: "nowrap",
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                }}
                title={`ULP near 2^${e.realExp} is ${e.ulp}`}
              >
                ulp {e.ulp}
              </span>
            </div>
          );
        })}
      </div>
      <div style={{ fontSize: 11.5, color: C.muted, marginTop: 12, lineHeight: 1.5 }}>
        The bar marked in terracotta is the magnitude band of the value you are editing above.
      </div>
    </Card>
  );
}
