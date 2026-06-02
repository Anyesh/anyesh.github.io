import { useState, useMemo, useEffect, useRef, useCallback } from "react";

export const meta = {
  title: "Hash Tables and Collisions",
  category: "Data Structures",
  description:
    "A dictionary lookup feels instant, yet two different keys can land in the same bucket. Insert keys, watch the hash pick a slot, and see what the table does when slots collide and when it runs out of room.",
  date: "2026-03-26",
  tags: ["hash-table", "hashing", "collisions", "data-structures"],
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
  probe: "#2e6f8e",
  probeSoft: "#eef3f5",
  good: "#2f7d53",
  goodSoft: "#e9f2ec",
  bad: "#a8453f",
  badSoft: "#fbecea",
};

const EASE = "cubic-bezier(0.23, 1, 0.32, 1)";
const EMPTY = Symbol("empty");
const TOMBSTONE = Symbol("tombstone");

function fnv1a(str) {
  let h = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

function badHash(str) {
  let sum = 0;
  for (let i = 0; i < str.length; i++) sum += str.charCodeAt(i);
  return sum >>> 0;
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

function emptyChaining(m) {
  return Array.from({ length: m }, () => []);
}

function emptyOpen(m) {
  return Array.from({ length: m }, () => EMPTY);
}

function makeTable(m, mode) {
  return {
    m,
    mode,
    slots: mode === "chaining" ? emptyChaining(m) : emptyOpen(m),
    count: 0,
  };
}

function hashOf(key, hashFn) {
  return hashFn === "bad" ? badHash(key) : fnv1a(key);
}

function homeBucket(key, m, hashFn) {
  return hashOf(key, hashFn) % m;
}

function chainHas(table, key, hashFn) {
  const home = homeBucket(key, table.m, hashFn);
  const chain = table.slots[home];
  for (let i = 0; i < chain.length; i++) {
    if (chain[i].key === key) return { found: true, home, comparisons: i + 1, index: i };
  }
  return { found: false, home, comparisons: chain.length, index: -1 };
}

function probeSequence(table, key, hashFn) {
  const home = homeBucket(key, table.m, hashFn);
  const seq = [];
  let firstTomb = -1;
  for (let step = 0; step < table.m; step++) {
    const slot = (home + step) % table.m;
    const cell = table.slots[slot];
    seq.push(slot);
    if (cell === EMPTY) {
      return { home, seq, terminal: "empty", slot, found: false, firstTomb };
    }
    if (cell === TOMBSTONE) {
      if (firstTomb < 0) firstTomb = slot;
      continue;
    }
    if (cell.key === key) {
      return { home, seq, terminal: "match", slot, found: true, firstTomb };
    }
  }
  return { home, seq, terminal: "full", slot: -1, found: false, firstTomb };
}

function liveCount(table) {
  if (table.mode === "chaining") {
    let n = 0;
    for (const chain of table.slots) n += chain.length;
    return n;
  }
  let n = 0;
  for (const cell of table.slots) {
    if (cell !== EMPTY && cell !== TOMBSTONE) n++;
  }
  return n;
}

function rebuildAt(newM, mode, hashFn, entries) {
  const next = makeTable(newM, mode);
  for (const key of entries) {
    if (mode === "chaining") {
      const home = homeBucket(key, newM, hashFn);
      next.slots[home].push({ key });
    } else {
      const home = homeBucket(key, newM, hashFn);
      for (let step = 0; step < newM; step++) {
        const slot = (home + step) % newM;
        if (next.slots[slot] === EMPTY) {
          next.slots[slot] = { key };
          break;
        }
      }
    }
  }
  next.count = liveCount(next);
  return next;
}

function allKeys(table) {
  const keys = [];
  if (table.mode === "chaining") {
    for (const chain of table.slots) for (const e of chain) keys.push(e.key);
  } else {
    for (const cell of table.slots) {
      if (cell !== EMPTY && cell !== TOMBSTONE) keys.push(cell.key);
    }
  }
  return keys;
}

function probeStats(table, hashFn) {
  const keys = allKeys(table);
  if (keys.length === 0) return { avg: 0, worst: 0 };
  let total = 0;
  let worst = 0;
  for (const key of keys) {
    let cost;
    if (table.mode === "chaining") {
      cost = chainHas(table, key, hashFn).comparisons;
    } else {
      cost = probeSequence(table, key, hashFn).seq.length;
    }
    total += cost;
    if (cost > worst) worst = cost;
  }
  return { avg: total / keys.length, worst };
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

function Btn({ children, onClick, disabled, primary, ariaLabel, type, tone }) {
  const bg = primary ? (tone === "probe" ? C.probe : C.accent) : "transparent";
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
        background: bg,
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

function Slider({ label, value, min, max, onChange, id, suffix, step }) {
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
        step={step || 1}
        value={value}
        onChange={(e) => onChange(+e.target.value)}
        style={{ width: "100%", accentColor: C.accent }}
      />
    </div>
  );
}

function Stat({ label, value, sub, color, bg }) {
  return (
    <div style={{ background: bg, borderRadius: 10, padding: "11px 13px", flex: 1, minWidth: 110 }}>
      <div
        style={{
          fontSize: 10.5,
          color,
          fontWeight: 700,
          letterSpacing: "0.05em",
          textTransform: "uppercase",
          marginBottom: 4,
        }}
      >
        {label}
      </div>
      <div style={{ fontSize: 21, fontWeight: 700, color, lineHeight: 1.05 }}>{value}</div>
      {sub && <div style={{ fontSize: 10.5, color: C.muted, marginTop: 3 }}>{sub}</div>}
    </div>
  );
}

function Toggle({ options, value, onChange, label }) {
  return (
    <div role="radiogroup" aria-label={label}>
      <div
        style={{
          display: "inline-flex",
          background: C.faint,
          borderRadius: 9,
          padding: 3,
          gap: 3,
        }}
      >
        {options.map((opt) => {
          const active = opt.value === value;
          return (
            <button
              key={opt.value}
              role="radio"
              aria-checked={active}
              onClick={() => onChange(opt.value)}
              style={{
                padding: "6px 13px",
                borderRadius: 7,
                border: "none",
                background: active ? C.card : "transparent",
                color: active ? C.ink : C.muted,
                fontWeight: active ? 700 : 500,
                fontSize: 12.5,
                fontFamily: "inherit",
                cursor: "pointer",
                boxShadow: active ? "0 1px 2px rgba(0,0,0,0.08)" : "none",
                transition: `background 160ms ease, color 160ms ease`,
              }}
            >
              {opt.label}
            </button>
          );
        })}
      </div>
    </div>
  );
}

function BucketRow({ index, mode, cell, isHome, onProbe, justResized, reduce }) {
  const probeState = onProbe ? onProbe.get(index) : undefined;
  const occupied =
    mode === "chaining"
      ? cell.length > 0
      : cell !== EMPTY && cell !== TOMBSTONE;
  const tomb = mode === "open" && cell === TOMBSTONE;

  let ring = "transparent";
  let fillTint = occupied ? C.accentSoft : "#fff";
  if (probeState === "scan") ring = C.probe;
  if (probeState === "hit") {
    ring = C.good;
    fillTint = C.goodSoft;
  }
  if (probeState === "miss") {
    ring = C.bad;
    fillTint = C.badSoft;
  }
  if (probeState === "land") {
    ring = C.accent;
    fillTint = C.accentSoft;
  }

  return (
    <div
      style={{
        display: "flex",
        alignItems: "stretch",
        gap: 8,
        opacity: justResized && !reduce ? 0 : 1,
        animation:
          justResized && !reduce ? `htReveal 420ms ${EASE} forwards` : "none",
        animationDelay: justResized && !reduce ? `${Math.min(index, 24) * 14}ms` : "0ms",
      }}
    >
      <div
        style={{
          width: 30,
          flexShrink: 0,
          fontSize: 11,
          fontVariantNumeric: "tabular-nums",
          color: isHome ? C.accent : C.muted,
          fontWeight: isHome ? 700 : 500,
          textAlign: "right",
          alignSelf: "center",
        }}
      >
        {index}
      </div>
      <div
        style={{
          flex: 1,
          minWidth: 0,
          minHeight: 30,
          borderRadius: 8,
          border: `2px solid ${ring === "transparent" ? C.border : ring}`,
          background: fillTint,
          padding: "4px 6px",
          display: "flex",
          alignItems: "center",
          gap: 5,
          flexWrap: "wrap",
          transform: probeState && probeState !== "scan" && !reduce ? "scale(1.02)" : "scale(1)",
          transition: reduce
            ? "none"
            : `border-color 180ms ${EASE}, background 180ms ${EASE}, transform 180ms ${EASE}`,
        }}
      >
        {mode === "chaining" ? (
          cell.length === 0 ? (
            <span style={{ fontSize: 11, color: C.muted, fontStyle: "italic" }}>empty</span>
          ) : (
            cell.map((e, i) => (
              <span key={e.key} style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
                <span
                  style={{
                    background: "#fff",
                    border: `1px solid ${C.accent}55`,
                    color: C.ink,
                    borderRadius: 6,
                    padding: "2px 8px",
                    fontSize: 12,
                    fontWeight: 600,
                  }}
                >
                  {e.key}
                </span>
                {i < cell.length - 1 && (
                  <span style={{ color: C.muted, fontSize: 12 }} aria-hidden="true">
                    {"→"}
                  </span>
                )}
              </span>
            ))
          )
        ) : tomb ? (
          <span style={{ fontSize: 11, color: C.bad, fontStyle: "italic" }}>deleted</span>
        ) : occupied ? (
          <span
            style={{
              background: "#fff",
              border: `1px solid ${C.accent}55`,
              color: C.ink,
              borderRadius: 6,
              padding: "2px 8px",
              fontSize: 12,
              fontWeight: 600,
            }}
          >
            {cell.key}
          </span>
        ) : (
          <span style={{ fontSize: 11, color: C.muted, fontStyle: "italic" }}>empty</span>
        )}
      </div>
    </div>
  );
}

const PRESETS = ["apple", "banana", "cherry", "date", "fig", "grape", "kiwi", "lemon"];
const COLLIDERS = ["abc", "acb", "bac", "bca", "cab", "cba"];

export default function App() {
  const reduce = usePrefersReducedMotion();
  const [mode, setMode] = useState("chaining");
  const [hashFn, setHashFn] = useState("fnv");
  const [threshold, setThreshold] = useState(0.75);
  const [showProbes, setShowProbes] = useState(true);
  const [table, setTable] = useState(() => makeTable(8, "chaining"));
  const [insertField, setInsertField] = useState("");
  const [queryField, setQueryField] = useState("");
  const [probeMap, setProbeMap] = useState(null);
  const [homeIndex, setHomeIndex] = useState(null);
  const [justResized, setJustResized] = useState(false);
  const [result, setResult] = useState(null);
  const [presetIdx, setPresetIdx] = useState(0);
  const animRef = useRef(null);
  const resizeRef = useRef(null);

  const n = table.count;
  const load = table.m > 0 ? n / table.m : 0;
  const stats = useMemo(() => probeStats(table, hashFn), [table, hashFn]);

  const clearAnim = useCallback(() => {
    if (animRef.current) {
      clearTimeout(animRef.current);
      animRef.current = null;
    }
    if (resizeRef.current) {
      clearTimeout(resizeRef.current);
      resizeRef.current = null;
    }
  }, []);

  useEffect(() => clearAnim, [clearAnim]);

  const resetTo = useCallback(
    (nextMode, nextHash) => {
      clearAnim();
      const seeded = makeTable(8, nextMode);
      for (let i = 0; i < 3; i++) {
        const key = PRESETS[i];
        if (nextMode === "chaining") {
          seeded.slots[homeBucket(key, 8, nextHash)].push({ key });
        } else {
          const home = homeBucket(key, 8, nextHash);
          for (let s = 0; s < 8; s++) {
            const slot = (home + s) % 8;
            if (seeded.slots[slot] === EMPTY) {
              seeded.slots[slot] = { key };
              break;
            }
          }
        }
      }
      seeded.count = liveCount(seeded);
      setTable(seeded);
      setProbeMap(null);
      setHomeIndex(null);
      setJustResized(false);
      setPresetIdx(3);
      setResult({
        kind: "intro",
        text: "Three keys are already in: apple, banana, cherry. Insert another, or look one up and watch the probe path.",
      });
    },
    [clearAnim]
  );

  useEffect(() => {
    resetTo("chaining", "fnv");
  }, [resetTo]);

  function flashProbe(map, home, hold) {
    setProbeMap(map);
    setHomeIndex(home);
    animRef.current = setTimeout(() => {
      setProbeMap(null);
      setHomeIndex(null);
    }, hold);
  }

  function maybeResize(base) {
    const factor = base.m > 0 ? base.count / base.m : 0;
    if (factor <= threshold) return { table: base, resized: false };
    const grown = rebuildAt(base.m * 2, base.mode, hashFn, allKeys(base));
    return { table: grown, resized: true };
  }

  function doInsert(rawKey) {
    const key = rawKey.trim();
    if (!key) return;
    clearAnim();
    setInsertField("");

    if (table.mode === "chaining") {
      const probe = chainHas(table, key, hashFn);
      if (probe.found) {
        flashProbe(new Map([[probe.home, "hit"]]), probe.home, 1600);
        setResult({
          kind: "dup",
          text: `"${key}" is already in bucket ${probe.home}. Hash maps it to the same slot every time, so a duplicate insert just finds it.`,
        });
        return;
      }
      const next = { ...table, slots: table.slots.map((c) => c.slice()) };
      next.slots[probe.home] = [...next.slots[probe.home], { key }];
      next.count = table.count + 1;
      const collided = table.slots[probe.home].length > 0;
      const { table: settled, resized } = maybeResize(next);
      if (resized) {
        settleResize(settled, key, probe.home, collided, "chain");
      } else {
        setTable(settled);
        flashProbe(new Map([[probe.home, "land"]]), probe.home, 1600);
        setResult({
          kind: collided ? "collision" : "insert",
          text: collided
            ? `hash("${key}") mod ${table.m} = ${probe.home}, already taken. Chaining appends "${key}" to that bucket's list, so both keys live there.`
            : `hash("${key}") mod ${table.m} = ${probe.home}, an empty bucket. "${key}" drops straight in.`,
        });
      }
      return;
    }

    const probe = probeSequence(table, key, hashFn);
    if (probe.found) {
      flashProbe(new Map([[probe.slot, "hit"]]), probe.home, 1600);
      setResult({
        kind: "dup",
        text: `"${key}" is already stored at slot ${probe.slot}. The probe walked ${probe.seq.length} slot${probe.seq.length === 1 ? "" : "s"} from home ${probe.home} to find it.`,
      });
      return;
    }
    if (probe.terminal === "full") {
      setResult({
        kind: "full",
        text: `Every slot is occupied, so linear probing has nowhere to put "${key}". Open addressing must resize before the table fills. Raise the table size or lower the threshold.`,
      });
      return;
    }
    const landing = probe.firstTomb >= 0 ? probe.firstTomb : probe.slot;
    const next = { ...table, slots: table.slots.slice() };
    next.slots[landing] = { key };
    next.count = table.count + 1;
    const steps = probe.seq.length;
    const map = new Map();
    for (const s of probe.seq) map.set(s, "scan");
    map.set(landing, "land");
    const { table: settled, resized } = maybeResize(next);
    if (resized) {
      settleResize(settled, key, probe.home, steps > 1, "open");
    } else {
      setTable(settled);
      flashProbe(map, probe.home, 1900);
      setResult({
        kind: steps > 1 ? "collision" : "insert",
        text:
          steps > 1
            ? `hash("${key}") mod ${table.m} = ${probe.home} was taken. Linear probing stepped ${steps - 1} slot${steps - 1 === 1 ? "" : "s"} forward to the first opening at ${landing}.`
            : `hash("${key}") mod ${table.m} = ${probe.home}, empty. "${key}" lands in one probe.`,
      });
    }
  }

  function settleResize(grown, key, home, collided, kind) {
    setTable(grown);
    setProbeMap(null);
    setHomeIndex(null);
    setJustResized(true);
    setResult({
      kind: "resize",
      text: `Load factor crossed ${threshold.toFixed(2)} after adding "${key}", so the table doubled to ${grown.m} buckets and rehashed every key. Each key's bucket is hash(key) mod ${grown.m} now, so the slots change but every key is still found.`,
    });
    resizeRef.current = setTimeout(() => setJustResized(false), 900);
  }

  function doLookup(rawKey) {
    const key = rawKey.trim();
    if (!key) return;
    clearAnim();
    setQueryField("");

    if (table.mode === "chaining") {
      const probe = chainHas(table, key, hashFn);
      flashProbe(new Map([[probe.home, probe.found ? "hit" : "miss"]]), probe.home, 1900);
      setResult({
        kind: probe.found ? "found" : "absent",
        text: probe.found
          ? `Found "${key}". Hash sent us straight to bucket ${probe.home}, then we scanned the chain: ${probe.comparisons} comparison${probe.comparisons === 1 ? "" : "s"} to reach it.`
          : `"${key}" is not here. Bucket ${probe.home} is its only possible home, and ${probe.comparisons === 0 ? "that bucket is empty" : `${probe.comparisons} comparison${probe.comparisons === 1 ? "" : "s"} found no match`}.`,
      });
      return;
    }

    const probe = probeSequence(table, key, hashFn);
    const map = new Map();
    for (const s of probe.seq) map.set(s, "scan");
    if (probe.found) map.set(probe.slot, "hit");
    else if (probe.seq.length) map.set(probe.seq[probe.seq.length - 1], "miss");
    flashProbe(map, probe.home, 2100);
    setResult({
      kind: probe.found ? "found" : "absent",
      text: probe.found
        ? `Found "${key}" at slot ${probe.slot}. The probe walked ${probe.seq.length} slot${probe.seq.length === 1 ? "" : "s"} from home ${probe.home}.`
        : `"${key}" is not here. Probing from home ${probe.home} hit ${probe.terminal === "empty" ? `an empty slot after ${probe.seq.length} step${probe.seq.length === 1 ? "" : "s"}, which proves it was never inserted` : "the end of the table"}.`,
    });
  }

  function doDelete(rawKey) {
    const key = rawKey.trim();
    if (!key) return;
    clearAnim();
    setQueryField("");

    if (table.mode === "chaining") {
      const probe = chainHas(table, key, hashFn);
      if (!probe.found) {
        flashProbe(new Map([[probe.home, "miss"]]), probe.home, 1700);
        setResult({ kind: "absent", text: `"${key}" is not in bucket ${probe.home}, nothing to delete.` });
        return;
      }
      const next = { ...table, slots: table.slots.map((c) => c.slice()) };
      next.slots[probe.home] = next.slots[probe.home].filter((e) => e.key !== key);
      next.count = table.count - 1;
      setTable(next);
      flashProbe(new Map([[probe.home, "miss"]]), probe.home, 1500);
      setResult({
        kind: "delete",
        text: `Removed "${key}" from bucket ${probe.home}'s chain. The rest of the chain stays linked.`,
      });
      return;
    }

    const probe = probeSequence(table, key, hashFn);
    if (!probe.found) {
      const map = new Map();
      for (const s of probe.seq) map.set(s, "scan");
      if (probe.seq.length) map.set(probe.seq[probe.seq.length - 1], "miss");
      flashProbe(map, probe.home, 1700);
      setResult({ kind: "absent", text: `"${key}" is not in the table, nothing to delete.` });
      return;
    }
    const next = { ...table, slots: table.slots.slice() };
    next.slots[probe.slot] = TOMBSTONE;
    next.count = table.count - 1;
    setTable(next);
    flashProbe(new Map([[probe.slot, "miss"]]), probe.home, 1700);
    setResult({
      kind: "delete",
      text: `Removed "${key}" from slot ${probe.slot}, but the slot becomes a tombstone, not empty. A real empty here would cut off probes for keys that hopped over this slot, so a deleted marker keeps later lookups correct.`,
    });
  }

  function insertPreset() {
    if (presetIdx >= PRESETS.length) {
      setResult({ kind: "intro", text: "All sample fruit keys are in. Type your own, or load the colliders." });
      return;
    }
    doInsert(PRESETS[presetIdx]);
    setPresetIdx((i) => i + 1);
  }

  function loadColliders() {
    clearAnim();
    const seeded = makeTable(table.m, table.mode);
    for (const key of COLLIDERS) {
      if (table.mode === "chaining") {
        seeded.slots[homeBucket(key, table.m, "bad")].push({ key });
      } else {
        const home = homeBucket(key, table.m, "bad");
        for (let s = 0; s < table.m; s++) {
          const slot = (home + s) % table.m;
          if (seeded.slots[slot] === EMPTY) {
            seeded.slots[slot] = { key };
            break;
          }
        }
      }
    }
    seeded.count = liveCount(seeded);
    setHashFn("bad");
    setTable(seeded);
    setProbeMap(null);
    setHomeIndex(null);
    setJustResized(false);
    setPresetIdx(PRESETS.length);
    setResult({
      kind: "collision",
      text:
        table.mode === "chaining"
          ? "These six keys are anagrams. The weak sum-of-characters hash gives them all the same value, so they pile into one bucket. A lookup now scans that whole chain: this is the O(n) worst case."
          : "These six anagrams share one hash under the weak sum-of-characters function. Linear probing packs them into a contiguous run, so a lookup probes through the whole cluster: the O(n) worst case.",
    });
  }

  function changeMode(next) {
    if (next === table.mode) return;
    clearAnim();
    const keys = allKeys(table);
    const rebuilt = rebuildAt(table.m, next, hashFn, keys);
    setMode(next);
    setTable(rebuilt);
    setProbeMap(null);
    setHomeIndex(null);
    setJustResized(false);
    setResult({
      kind: "intro",
      text:
        next === "chaining"
          ? "Separate chaining: each bucket holds a list, so colliding keys share a bucket and stay linked."
          : "Open addressing: every key lives in the array itself, and a collision probes forward to the next open slot.",
    });
  }

  function changeHash(next) {
    clearAnim();
    const keys = allKeys(table);
    const rebuilt = rebuildAt(table.m, table.mode, next, keys);
    setHashFn(next);
    setTable(rebuilt);
    setProbeMap(null);
    setHomeIndex(null);
    setResult({
      kind: "intro",
      text:
        next === "bad"
          ? "Now hashing by sum of character codes. Anagrams and same-letter keys collide, so the table clumps and lookups slow down."
          : "Back to FNV-1a, which spreads keys evenly across the buckets.",
    });
  }

  function changeSize(nextM) {
    clearAnim();
    const keys = allKeys(table);
    const rebuilt = rebuildAt(nextM, table.mode, hashFn, keys);
    setTable(rebuilt);
    setProbeMap(null);
    setHomeIndex(null);
    setResult({
      kind: "resize",
      text: `Resized to ${nextM} buckets and rehashed all ${keys.length} key${keys.length === 1 ? "" : "s"} into hash(key) mod ${nextM}.`,
    });
  }

  const resultStyle = {
    insert: { bg: C.accentSoft, color: C.accent, border: `${C.accent}33` },
    collision: { bg: C.badSoft, color: C.bad, border: `${C.bad}44` },
    found: { bg: C.goodSoft, color: C.good, border: `${C.good}33` },
    absent: { bg: C.probeSoft, color: C.probe, border: `${C.probe}33` },
    delete: { bg: C.faint, color: C.muted, border: C.border },
    dup: { bg: C.faint, color: C.muted, border: C.border },
    resize: { bg: C.probeSoft, color: C.probe, border: `${C.probe}33` },
    full: { bg: C.badSoft, color: C.bad, border: `${C.bad}44` },
    intro: { bg: C.faint, color: C.muted, border: C.border },
  };
  const rs = result ? resultStyle[result.kind] || resultStyle.intro : resultStyle.intro;

  const overThreshold = load > threshold;

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
        @keyframes htReveal { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }
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
            Data Structures
          </div>
          <h1 style={{ fontSize: 25, fontWeight: 700, margin: 0 }}>Hash Tables and Collisions</h1>
          <p
            style={{
              color: C.ink,
              fontSize: 13.5,
              margin: "8px 0 0",
              lineHeight: 1.65,
              maxWidth: "64ch",
            }}
          >
            Looking a key up in a dictionary feels instant because a hash function turns the key into
            a number and that number points straight at a bucket, no scanning required. The catch is
            that many keys map to few buckets, so two different keys can pick the same one. What the
            table does at that moment, and what it does when the buckets get crowded, is the whole
            story behind average constant time.
          </p>
        </div>

        <Card style={{ marginBottom: 16 }}>
          <div style={{ display: "flex", gap: 14, flexWrap: "wrap", marginBottom: 16 }}>
            <Toggle
              label="Collision strategy"
              value={mode}
              onChange={changeMode}
              options={[
                { value: "chaining", label: "Separate chaining" },
                { value: "open", label: "Open addressing" },
              ]}
            />
            <Toggle
              label="Hash function"
              value={hashFn}
              onChange={changeHash}
              options={[
                { value: "fnv", label: "FNV-1a (good)" },
                { value: "bad", label: "Sum of chars (weak)" },
              ]}
            />
          </div>

          <form
            onSubmit={(e) => {
              e.preventDefault();
              doInsert(insertField);
            }}
            style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 10 }}
          >
            <input
              type="text"
              value={insertField}
              onChange={(e) => setInsertField(e.target.value)}
              placeholder="key to insert"
              aria-label="Key to insert into the table"
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
              doLookup(queryField);
            }}
            style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 14 }}
          >
            <input
              type="text"
              value={queryField}
              onChange={(e) => setQueryField(e.target.value)}
              placeholder="key to look up or delete"
              aria-label="Key to look up or delete"
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
            <Btn type="submit" tone="probe" primary ariaLabel="Look up the typed key">
              Look up
            </Btn>
            <Btn onClick={() => doDelete(queryField)} ariaLabel="Delete the typed key">
              Delete
            </Btn>
          </form>

          <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginBottom: 16 }}>
            <Btn onClick={insertPreset} ariaLabel="Insert the next sample key">
              Insert sample key
            </Btn>
            <Btn onClick={loadColliders} ariaLabel="Load six keys that all collide">
              Load colliders
            </Btn>
            <Btn onClick={() => resetTo(table.mode, hashFn)} ariaLabel="Reset the table">
              Reset
            </Btn>
          </div>

          {showProbes && (homeIndex !== null || probeMap) && (
            <div
              aria-hidden="true"
              style={{ fontSize: 11.5, color: C.muted, marginBottom: 10, textAlign: "center" }}
            >
              {homeIndex !== null && (
                <>
                  Home bucket{" "}
                  <b style={{ color: C.accent }}>hash mod {table.m} = {homeIndex}</b>
                </>
              )}
            </div>
          )}

          <div
            style={{
              display: "flex",
              flexDirection: "column",
              gap: 5,
              maxHeight: 360,
              overflowY: "auto",
              paddingRight: 4,
            }}
          >
            {table.slots.map((cell, i) => (
              <BucketRow
                key={i}
                index={i}
                mode={table.mode}
                cell={cell}
                isHome={homeIndex === i}
                onProbe={showProbes ? probeMap : null}
                justResized={justResized}
                reduce={reduce}
              />
            ))}
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
                {result.kind === "collision" && (
                  <b style={{ display: "block", marginBottom: 2 }}>Collision</b>
                )}
                {result.kind === "resize" && (
                  <b style={{ display: "block", marginBottom: 2 }}>Resize</b>
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
            <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 14 }}>Tune the table</div>
            <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
              <Slider
                id="size-slider"
                label="m, number of buckets"
                value={table.m}
                min={4}
                max={64}
                onChange={changeSize}
              />
              <Slider
                id="threshold-slider"
                label="resize at load factor"
                value={threshold}
                min={0.5}
                max={1}
                step={0.05}
                onChange={(v) => setThreshold(+v.toFixed(2))}
              />
            </div>
            <label
              style={{
                display: "flex",
                alignItems: "center",
                gap: 8,
                marginTop: 16,
                fontSize: 12.5,
                color: C.ink,
                cursor: "pointer",
              }}
            >
              <input
                type="checkbox"
                checked={showProbes}
                onChange={(e) => setShowProbes(e.target.checked)}
                style={{ accentColor: C.probe, width: 15, height: 15 }}
              />
              Show probe steps on the buckets
            </label>
            <p style={{ fontSize: 12, color: C.muted, lineHeight: 1.6, margin: "14px 0 0" }}>
              Load factor is{" "}
              <b style={{ color: overThreshold ? C.bad : C.accent }}>{load.toFixed(2)}</b> (n/m). The
              next insert that pushes it past <b>{threshold.toFixed(2)}</b> doubles the table and
              rehashes everything.
            </p>
          </Card>

          <Card>
            <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 12 }}>Live state</div>
            <div style={{ display: "flex", gap: 9, flexWrap: "wrap", marginBottom: 9 }}>
              <Stat label="Keys (n)" value={n} color={C.accent} bg={C.accentSoft} />
              <Stat label="Buckets (m)" value={table.m} color={C.probe} bg={C.probeSoft} />
              <Stat
                label="Load factor"
                value={load.toFixed(2)}
                color={overThreshold ? C.bad : C.ink}
                bg={overThreshold ? C.badSoft : C.faint}
              />
            </div>
            <div style={{ display: "flex", gap: 9, flexWrap: "wrap" }}>
              <Stat
                label="Avg probes"
                value={stats.avg.toFixed(2)}
                sub="comparisons per found key"
                color={C.good}
                bg={C.goodSoft}
              />
              <Stat
                label="Worst probes"
                value={stats.worst}
                sub="slowest key in the table"
                color={stats.worst > 3 ? C.bad : C.ink}
                bg={stats.worst > 3 ? C.badSoft : C.faint}
              />
            </div>
          </Card>
        </div>

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
              <b>Why lookup is usually O(1).</b> The hash turns a key into a bucket index in one
              step, so the table jumps to a slot instead of scanning. With a good hash and a load
              factor kept low, each bucket holds only a key or two, so the scan after the jump is
              tiny and constant on average.
            </p>
            <p style={{ margin: 0 }}>
              <b>Why collisions are unavoidable.</b> There are far more possible keys than buckets,
              so by the pigeonhole principle some keys must share a bucket. Even well short of full,
              the birthday paradox makes the first collision arrive surprisingly early. The question
              was never whether collisions happen, only how the table absorbs them.
            </p>
            <p style={{ margin: 0 }}>
              <b>Chaining versus probing.</b> Chaining hangs a list off each bucket, so colliding
              keys cost a short list walk and the table never runs out of room. Open addressing keeps
              every key in the array and probes forward to the next opening, which is cache-friendly
              but forms clusters and cannot exceed m keys, so it leans harder on resizing.
            </p>
            <p style={{ margin: 0 }}>
              <b>Why resize, and why it is still O(1) amortized.</b> As the load factor climbs,
              chains lengthen and probe clusters grow, so the table doubles m and rehashes every key
              to spread them out again. That rehash is O(n), but it happens rarely enough that
              spread across all the cheap inserts the average stays constant.
            </p>
            <p style={{ margin: 0 }}>
              <b>How it degrades to O(n).</b> Switch to the weak hash or load the colliders and watch
              every key crowd into one bucket. Now a lookup walks the entire chain or probes the
              whole cluster, and the table behaves like an unsorted list. A bad hash or a load factor
              left too high erases the constant-time guarantee.
            </p>
          </div>
        </Card>

        <div style={{ marginTop: 18, textAlign: "center", fontSize: 11, color: C.muted }}>
          Buckets are chosen by a real FNV-1a hash reduced mod m; the weak option sums character
          codes so anagrams collide
        </div>
      </div>
    </div>
  );
}
