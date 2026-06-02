import { useState, useEffect, useRef, useMemo, useCallback } from "react";

export const meta = {
  title: "Wave Function Collapse",
  category: "Procedural Generation",
  description:
    "A grid where every cell starts as a superposition of every terrain tile, then collapses one cell at a time. Each collapse ripples through the neighbours as a constraint wave until the whole map agrees. Step it, watch the entropy drain, and force a contradiction to see it backtrack.",
  date: "2026-04-08",
  tags: ["wfc", "constraint-propagation", "procedural-generation", "entropy"],
};

const C = {
  bg: "#f7f5f2",
  card: "#ffffff",
  border: "#e4dfd8",
  ink: "#1c1a17",
  muted: "#6f675e",
  faint: "#9b938a",
  accent: "#c0561f",
  accentSoft: "#f5e7df",
  good: "#2f6d4f",
  goodSoft: "#dceae3",
  warn: "#b3341b",
  warnSoft: "#f5ddd6",
};

const EASE = "cubic-bezier(0.23, 1, 0.32, 1)";

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

const TILES = [
  { id: "sea", name: "Sea", edge: "W", color: "#4a78a8" },
  { id: "coast", name: "Coast", edge: "WL", color: "#cdb787" },
  { id: "land", name: "Land", edge: "L", color: "#7faa5e" },
  { id: "forest", name: "Forest", edge: "L", color: "#3f7a43" },
  { id: "mountain", name: "Mountain", edge: "M", color: "#8a8079" },
];

const N = TILES.length;
const FULL_MASK = (1 << N) - 1;

const WEIGHTS = [3, 2, 3, 2, 1];

const EDGE_COMPAT = {
  W: ["W", "WL"],
  WL: ["W", "WL", "L"],
  L: ["WL", "L", "M"],
  M: ["L", "M"],
};

function edgesAgree(a, b) {
  return EDGE_COMPAT[TILES[a].edge].includes(TILES[b].edge);
}

const COMPAT = TILES.map((_, a) => {
  let mask = 0;
  for (let b = 0; b < N; b++) {
    if (edgesAgree(a, b)) mask |= 1 << b;
  }
  return mask;
});

function maskTiles(mask) {
  const out = [];
  for (let t = 0; t < N; t++) if (mask & (1 << t)) out.push(t);
  return out;
}

function popcount(mask) {
  let c = 0;
  while (mask) {
    mask &= mask - 1;
    c++;
  }
  return c;
}

const DIRS = [
  [-1, 0],
  [1, 0],
  [0, -1],
  [0, 1],
];

function allowedFromNeighbour(neighbourMask) {
  let allowed = 0;
  const tiles = maskTiles(neighbourMask);
  for (const t of tiles) allowed |= COMPAT[t];
  return allowed;
}

function freshGrid(size) {
  return new Array(size * size).fill(FULL_MASK);
}

function pickCollapseTarget(grid, size, rng) {
  let best = -1;
  let bestCount = Infinity;
  const ties = [];
  for (let i = 0; i < grid.length; i++) {
    const count = popcount(grid[i]);
    if (count <= 1) continue;
    if (count < bestCount) {
      bestCount = count;
      ties.length = 0;
      ties.push(i);
    } else if (count === bestCount) {
      ties.push(i);
    }
  }
  if (ties.length === 0) return -1;
  best = ties[Math.floor(rng() * ties.length)];
  return best;
}

function chooseTile(mask, rng) {
  const tiles = maskTiles(mask);
  let total = 0;
  for (const t of tiles) total += WEIGHTS[t];
  let r = rng() * total;
  for (const t of tiles) {
    r -= WEIGHTS[t];
    if (r <= 0) return t;
  }
  return tiles[tiles.length - 1];
}

function propagate(grid, size, seeds) {
  const changed = new Set();
  const queue = [...seeds];
  let contradiction = -1;
  while (queue.length > 0) {
    const idx = queue.shift();
    const r = Math.floor(idx / size);
    const c = idx % size;
    for (const [dr, dc] of DIRS) {
      const nr = r + dr;
      const nc = c + dc;
      if (nr < 0 || nr >= size || nc < 0 || nc >= size) continue;
      const nIdx = nr * size + nc;
      const allowed = allowedFromNeighbour(grid[idx]);
      const before = grid[nIdx];
      const after = before & allowed;
      if (after !== before) {
        grid[nIdx] = after;
        changed.add(nIdx);
        if (after === 0) {
          contradiction = nIdx;
        } else {
          queue.push(nIdx);
        }
      }
    }
    if (contradiction !== -1) break;
  }
  return { changed, contradiction };
}

function buildSolveLog(size, seed, maxRetries) {
  const log = [];
  let attempts = 0;
  let seedUsed = seed;
  for (let retry = 0; retry <= maxRetries; retry++) {
    seedUsed = (seed + retry * 0x9e3779b1) >>> 0;
    const rng = mulberry32(seedUsed);
    const grid = freshGrid(size);
    const steps = [];
    let failed = false;
    let guard = 0;
    const maxGuard = size * size * 4 + 50;
    while (guard++ < maxGuard) {
      const target = pickCollapseTarget(grid, size, rng);
      if (target === -1) break;
      const tile = chooseTile(grid[target], rng);
      grid[target] = 1 << tile;
      const { changed, contradiction } = propagate(grid, size, [target]);
      steps.push({
        collapsed: target,
        tile,
        rippled: [...changed],
        contradiction,
        snapshot: grid.slice(),
      });
      if (contradiction !== -1) {
        failed = true;
        break;
      }
    }
    attempts++;
    if (!failed) {
      return { steps, seedUsed, attempts, solved: true };
    }
    log.push(...steps);
  }
  return { steps: log, seedUsed, attempts, solved: false };
}

function StatBox({ label, value, sub, color, bg }) {
  return (
    <div style={{ background: bg, borderRadius: 10, padding: "12px 14px", flex: 1, minWidth: 104 }}>
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
      <div style={{ fontSize: 24, fontWeight: 700, color, lineHeight: 1, fontVariantNumeric: "tabular-nums" }}>
        {value}
      </div>
      {sub && <div style={{ fontSize: 11, color, opacity: 0.75, marginTop: 4 }}>{sub}</div>}
    </div>
  );
}

function ControlButton({ onClick, children, primary, disabled, ariaLabel }) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      aria-label={ariaLabel}
      style={{
        padding: primary ? "9px 18px" : "9px 14px",
        borderRadius: 9,
        border: primary ? "none" : `1.5px solid ${C.border}`,
        background: primary ? C.accent : C.card,
        color: primary ? "#fff" : C.ink,
        fontSize: 13,
        fontWeight: primary ? 600 : 500,
        fontFamily: "inherit",
        cursor: disabled ? "not-allowed" : "pointer",
        opacity: disabled ? 0.4 : 1,
        transition: `transform 140ms ${EASE}, background 160ms ease`,
        boxShadow: primary ? "0 1px 3px rgba(192,86,31,0.3)" : "none",
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

function blendMask(mask) {
  const tiles = maskTiles(mask);
  if (tiles.length === 0) return "#2a2622";
  let total = 0;
  let r = 0;
  let g = 0;
  let b = 0;
  for (const t of tiles) {
    const w = WEIGHTS[t];
    const hex = TILES[t].color;
    r += parseInt(hex.slice(1, 3), 16) * w;
    g += parseInt(hex.slice(3, 5), 16) * w;
    b += parseInt(hex.slice(5, 7), 16) * w;
    total += w;
  }
  return `rgb(${Math.round(r / total)}, ${Math.round(g / total)}, ${Math.round(b / total)})`;
}

export default function App() {
  const [size, setSize] = useState(14);
  const [seed, setSeed] = useState(7);
  const [stepIdx, setStepIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [reduceMotion, setReduceMotion] = useState(false);
  const [forceTile, setForceTile] = useState(null);
  const [hovered, setHovered] = useState(null);
  const playRef = useRef(null);

  useEffect(() => {
    if (typeof window === "undefined" || !window.matchMedia) return;
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    const apply = () => setReduceMotion(mq.matches);
    apply();
    mq.addEventListener("change", apply);
    return () => mq.removeEventListener("change", apply);
  }, []);

  const run = useMemo(() => buildSolveLog(size, seed, 12), [size, seed]);
  const totalSteps = run.steps.length;
  const clampedIdx = Math.min(stepIdx, totalSteps);

  useEffect(() => {
    setStepIdx(0);
    setPlaying(false);
  }, [size, seed]);

  useEffect(() => {
    if (!playing) {
      clearInterval(playRef.current);
      return;
    }
    playRef.current = setInterval(
      () => {
        setStepIdx((s) => {
          if (s >= totalSteps) {
            setPlaying(false);
            return s;
          }
          return s + 1;
        });
      },
      reduceMotion ? 24 : 90
    );
    return () => clearInterval(playRef.current);
  }, [playing, totalSteps, reduceMotion]);

  const view = useMemo(() => {
    if (clampedIdx === 0) {
      return {
        grid: freshGrid(size),
        rippled: new Set(),
        head: null,
        contradiction: -1,
      };
    }
    const ev = run.steps[clampedIdx - 1];
    return {
      grid: ev.snapshot,
      rippled: new Set(ev.rippled),
      head: ev.collapsed,
      contradiction: ev.contradiction,
    };
  }, [run, clampedIdx, size]);

  const stats = useMemo(() => {
    let collapsed = 0;
    let entropy = 0;
    let contradictions = 0;
    for (const m of view.grid) {
      const count = popcount(m);
      if (count === 1) collapsed += 1;
      else entropy += count - 1;
    }
    for (let i = 0; i < clampedIdx; i++) {
      if (run.steps[i].contradiction !== -1) contradictions += 1;
    }
    return { collapsed, entropy, contradictions, total: size * size };
  }, [view, clampedIdx, run, size]);

  const cellSize = `min(${Math.floor(560 / size)}px, calc((100vw - 60px) / ${size}))`;

  const forceCollapse = useCallback(
    (idx) => {
      if (forceTile === null) return;
      const baseSnapshot = view.grid.slice();
      if (!(baseSnapshot[idx] & (1 << forceTile))) return;
      const newSteps = run.steps.slice(0, clampedIdx);
      baseSnapshot[idx] = 1 << forceTile;
      const { changed, contradiction } = propagate(baseSnapshot, size, [idx]);
      newSteps.push({
        collapsed: idx,
        tile: forceTile,
        rippled: [...changed],
        contradiction,
        snapshot: baseSnapshot.slice(),
        forced: true,
      });
      run.steps.length = 0;
      run.steps.push(...newSteps);
      setStepIdx(newSteps.length);
      setPlaying(false);
    },
    [forceTile, view, run, clampedIdx, size]
  );

  const stage = useMemo(() => {
    if (clampedIdx === 0) {
      return {
        title: "Every cell holds every tile",
        body:
          "Before a single observation, each cell is in superposition: all five terrain tiles are still possible. Entropy is at its maximum. The algorithm now repeats two moves, observe then propagate, until the grid either fully agrees or hits a cell with no options left.",
      };
    }
    const ev = run.steps[clampedIdx - 1];
    if (ev.contradiction !== -1) {
      const cr = Math.floor(ev.contradiction / size);
      const cc = ev.contradiction % size;
      return {
        title: `Contradiction at (${cr}, ${cc})`,
        body:
          "Propagation drove a cell down to zero possible tiles: no terrain can sit there and satisfy all its neighbours at once. The solver discards this attempt and restarts from a fresh seed offset. With these adjacency rules a clean run is common, but greedy collapses can still paint themselves into a corner.",
      };
    }
    const cr = Math.floor(ev.collapsed / size);
    const cc = ev.collapsed % size;
    const tileName = TILES[ev.tile].name;
    return {
      title: ev.forced
        ? `You forced (${cr}, ${cc}) to ${tileName}`
        : `Collapsed (${cr}, ${cc}) to ${tileName}`,
      body: ev.forced
        ? `You pinned this cell yourself. The same propagation runs: ${ev.rippled.length} neighbouring cells lost options that no longer fit beside ${tileName}, and the wave cascaded outward until the grid was stable again.`
        : `The cell with the fewest remaining options was observed and collapsed to ${tileName}, weighted by tile frequency. The constraint wave then removed incompatible tiles from ${ev.rippled.length} cell${ev.rippled.length === 1 ? "" : "s"} before settling.`,
    };
  }, [clampedIdx, run, size]);

  const goalK = view.contradiction;
  const headK = view.head;

  return (
    <div
      style={{
        fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif",
        background: C.bg,
        minHeight: "100vh",
        padding: "26px 16px 48px",
        color: C.ink,
      }}
    >
      <style>{`
        .wfc-cell:focus-visible {
          outline: 3px solid ${C.accent};
          outline-offset: -3px;
          z-index: 4;
        }
        .wfc-grid button { -webkit-tap-highlight-color: transparent; }
        @keyframes wfc-pulse {
          0% { box-shadow: 0 0 0 0 ${C.accent}88; }
          100% { box-shadow: 0 0 0 6px ${C.accent}00; }
        }
        @media (prefers-reduced-motion: reduce) {
          .wfc-cell { transition: none !important; animation: none !important; }
        }
      `}</style>

      <div style={{ maxWidth: 720, margin: "0 auto" }}>
        <header style={{ marginBottom: 22 }}>
          <div
            style={{
              fontSize: 11,
              letterSpacing: "0.14em",
              textTransform: "uppercase",
              color: C.faint,
              marginBottom: 6,
            }}
          >
            Procedural generation / Constraint solving
          </div>
          <h1 style={{ fontSize: 28, fontWeight: 700, margin: 0, letterSpacing: "-0.02em", textWrap: "balance" }}>
            Wave Function Collapse
          </h1>
          <p
            style={{
              color: C.muted,
              fontSize: 14.5,
              lineHeight: 1.6,
              margin: "10px 0 0",
              maxWidth: "64ch",
            }}
          >
            Every cell starts as a blur of all five terrain tiles. The solver finds the most certain cell, collapses
            it to one tile, then propagates that choice outward so neighbours drop whatever no longer fits. Sea only
            touches sea or coast, mountains only touch land or mountains, and so the map stays coherent.
          </p>
        </header>

        <div style={{ display: "flex", gap: 10, marginBottom: 18, flexWrap: "wrap" }}>
          <StatBox
            label="Collapsed"
            value={`${stats.collapsed}`}
            sub={`of ${stats.total} cells`}
            color={C.good}
            bg={C.goodSoft}
          />
          <StatBox
            label="Entropy left"
            value={stats.entropy}
            sub="surplus options"
            color="#8a6f3f"
            bg={C.accentSoft}
          />
          <StatBox
            label="Contradictions"
            value={stats.contradictions}
            sub={run.solved ? "restarts on this seed" : "still searching"}
            color={C.warn}
            bg={C.warnSoft}
          />
        </div>

        <div
          className="wfc-grid"
          role="grid"
          aria-label={`Wave function collapse grid, ${size} by ${size}. Each cell shows its current terrain or its remaining uncertainty.`}
          style={{
            display: "grid",
            gridTemplateColumns: `repeat(${size}, ${cellSize})`,
            gap: 1.5,
            background: C.border,
            padding: 1.5,
            borderRadius: 12,
            width: "fit-content",
            margin: "0 auto",
            touchAction: "manipulation",
            userSelect: "none",
          }}
        >
          {view.grid.map((mask, idx) => {
            const count = popcount(mask);
            const r = Math.floor(idx / size);
            const c = idx % size;
            const collapsedTile = count === 1 ? maskTiles(mask)[0] : -1;
            const isContradiction = idx === goalK;
            const isHead = idx === headK;
            const canForce = forceTile !== null && count > 1 && (mask & (1 << forceTile));

            let bg;
            let label;
            if (isContradiction) {
              bg = C.warn;
              label = `Cell ${r}, ${c}: contradiction, no tiles left`;
            } else if (collapsedTile !== -1) {
              bg = TILES[collapsedTile].color;
              label = `Cell ${r}, ${c}: ${TILES[collapsedTile].name}`;
            } else {
              bg = blendMask(mask);
              label = `Cell ${r}, ${c}: ${count} tiles still possible`;
            }

            const uncertain = collapsedTile === -1 && !isContradiction;

            return (
              <button
                key={idx}
                type="button"
                role="gridcell"
                aria-label={label}
                className="wfc-cell"
                onClick={() => forceCollapse(idx)}
                onPointerEnter={() => setHovered(idx)}
                onPointerLeave={() => setHovered((h) => (h === idx ? null : h))}
                style={{
                  width: cellSize,
                  height: cellSize,
                  padding: 0,
                  border: isHead
                    ? `2px solid ${C.ink}`
                    : isContradiction
                    ? `2px solid #fff`
                    : "none",
                  borderRadius: 3,
                  background: bg,
                  opacity: uncertain ? 0.4 + (1 - (count - 1) / (N - 1)) * 0.5 : 1,
                  cursor: canForce ? "pointer" : "default",
                  position: "relative",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  fontFamily: "ui-monospace, 'SF Mono', Menlo, monospace",
                  color: "#fff",
                  fontSize: Math.max(7, Math.min(11, 132 / size)),
                  fontWeight: 700,
                  transition: reduceMotion ? "none" : `background 240ms ${EASE}, opacity 240ms ${EASE}, transform 160ms ${EASE}`,
                  transform: isHead && !reduceMotion ? "scale(1.08)" : "scale(1)",
                  zIndex: isHead ? 3 : isContradiction ? 2 : 1,
                  animation: isHead && !reduceMotion ? "wfc-pulse 520ms ease-out" : "none",
                  outline: canForce && hovered === idx ? `2px solid ${C.accent}` : "none",
                  outlineOffset: -2,
                }}
              >
                {uncertain && count <= N && (
                  <span style={{ textShadow: "0 1px 2px rgba(0,0,0,0.5)", opacity: 0.92 }}>{count}</span>
                )}
                {isContradiction && <span style={{ fontSize: Math.max(8, 150 / size) }}>x</span>}
              </button>
            );
          })}
        </div>

        <p
          style={{
            textAlign: "center",
            fontSize: 12,
            color: C.faint,
            margin: "12px 0 0",
            lineHeight: 1.6,
          }}
        >
          A solid cell is collapsed. A faded cell still holds several tiles, blended by colour, with a count of how
          many remain. The pulsing outline marks the cell just observed.
        </p>

        <div
          style={{
            display: "flex",
            gap: 9,
            flexWrap: "wrap",
            margin: "20px 0 14px",
            justifyContent: "center",
          }}
        >
          <ControlButton
            onClick={() => setStepIdx((s) => Math.min(s + 1, totalSteps))}
            disabled={clampedIdx >= totalSteps}
            ariaLabel="Observe one cell and propagate"
          >
            Step
          </ControlButton>
          <ControlButton
            primary
            onClick={() => {
              if (clampedIdx >= totalSteps) setStepIdx(0);
              setPlaying((p) => !p);
            }}
            disabled={totalSteps === 0}
            ariaLabel={playing ? "Pause auto run" : "Play auto run"}
          >
            {playing ? "Pause" : clampedIdx >= totalSteps && totalSteps > 0 ? "Replay" : "Play"}
          </ControlButton>
          <ControlButton
            onClick={() => {
              setStepIdx(0);
              setPlaying(false);
            }}
            ariaLabel="Reset to the initial superposition"
          >
            Reset
          </ControlButton>
          <ControlButton
            onClick={() => {
              setSeed((s) => (((s * 1664525 + 1013904223) >>> 0) % 100000) || 1);
            }}
            ariaLabel="Reseed and regenerate the map"
          >
            Reseed
          </ControlButton>
        </div>

        <div
          style={{
            display: "flex",
            gap: 16,
            flexWrap: "wrap",
            alignItems: "center",
            justifyContent: "center",
            marginBottom: 18,
          }}
        >
          <label style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 13, color: C.muted }}>
            Grid size
            <select
              value={size}
              onChange={(e) => setSize(Number(e.target.value))}
              aria-label="Grid size"
              style={{
                fontFamily: "inherit",
                fontSize: 13,
                padding: "6px 10px",
                borderRadius: 8,
                border: `1.5px solid ${C.border}`,
                background: C.card,
                color: C.ink,
                cursor: "pointer",
              }}
            >
              {[10, 12, 14, 16, 18, 20].map((s) => (
                <option key={s} value={s}>
                  {s} x {s}
                </option>
              ))}
            </select>
          </label>
        </div>

        <section
          style={{
            background: C.card,
            border: `1px solid ${C.border}`,
            borderRadius: 14,
            padding: "16px 18px",
            marginBottom: 14,
          }}
        >
          <div
            style={{
              fontSize: 11,
              letterSpacing: "0.08em",
              textTransform: "uppercase",
              color: C.accent,
              fontWeight: 700,
              marginBottom: 9,
            }}
          >
            Force a cell
          </div>
          <p style={{ fontSize: 13.5, lineHeight: 1.55, color: C.muted, margin: "0 0 12px", maxWidth: "62ch" }}>
            Pick a tile, then click any uncertain cell to pin it and watch the constraint wave ripple from your
            choice. The solver respects your pin and continues from there.
          </p>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
            {TILES.map((tile, t) => {
              const active = forceTile === t;
              return (
                <button
                  key={tile.id}
                  type="button"
                  onClick={() => setForceTile(active ? null : t)}
                  aria-pressed={active}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 7,
                    padding: "7px 12px",
                    borderRadius: 8,
                    border: `1.5px solid ${active ? C.ink : C.border}`,
                    background: active ? C.ink : C.card,
                    color: active ? "#fff" : C.ink,
                    fontSize: 12.5,
                    fontWeight: active ? 700 : 500,
                    fontFamily: "inherit",
                    cursor: "pointer",
                    transition: `background 160ms ease, border-color 160ms ease`,
                  }}
                >
                  <span
                    aria-hidden="true"
                    style={{
                      width: 14,
                      height: 14,
                      borderRadius: 4,
                      background: tile.color,
                      border: "1px solid rgba(0,0,0,0.15)",
                      flexShrink: 0,
                    }}
                  />
                  {tile.name}
                </button>
              );
            })}
          </div>
          {forceTile !== null && (
            <div style={{ fontSize: 12.5, color: C.accent, marginTop: 10, fontWeight: 600 }}>
              {TILES[forceTile].name} armed. Click an uncertain cell to pin it.
            </div>
          )}
        </section>

        <section
          aria-live="polite"
          style={{
            background: view.contradiction !== -1 ? C.warnSoft : C.card,
            border: `1px solid ${view.contradiction !== -1 ? C.warn + "55" : C.border}`,
            borderRadius: 14,
            padding: "18px 20px",
            marginBottom: 14,
          }}
        >
          <div
            style={{
              fontSize: 11,
              letterSpacing: "0.08em",
              textTransform: "uppercase",
              color: view.contradiction !== -1 ? C.warn : C.accent,
              fontWeight: 700,
              marginBottom: 7,
            }}
          >
            Observe then propagate
            {totalSteps > 0 && ` · step ${clampedIdx} of ${totalSteps}`}
          </div>
          <h2 style={{ fontSize: 17, fontWeight: 700, margin: "0 0 8px", letterSpacing: "-0.01em", textWrap: "balance" }}>
            {stage.title}
          </h2>
          <p style={{ fontSize: 14, lineHeight: 1.65, color: C.muted, margin: 0, maxWidth: "66ch" }}>{stage.body}</p>
        </section>

        <section
          style={{
            background: C.accentSoft,
            border: `1px solid ${C.accent}33`,
            borderRadius: 14,
            padding: "18px 20px",
          }}
        >
          <h2 style={{ fontSize: 15, fontWeight: 700, margin: "0 0 10px" }}>The adjacency rules</h2>
          <p style={{ fontSize: 14, lineHeight: 1.65, color: "#6b4b32", margin: "0 0 12px", maxWidth: "66ch" }}>
            Each tile carries an edge type, and two tiles may sit side by side only when their edges are compatible.
            That single table is the entire grammar of the map. Minimum entropy first matters because collapsing the
            most constrained cell keeps later choices open: a cell with two options is far more likely to survive than
            one chosen at random. A contradiction means propagation cornered a cell into zero options, which is why
            the solver keeps a retry budget.
          </p>
          <div style={{ display: "flex", flexDirection: "column", gap: 7 }}>
            {TILES.map((tile, t) => {
              const compatible = maskTiles(COMPAT[t]);
              return (
                <div
                  key={tile.id}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 10,
                    fontSize: 13,
                    color: "#6b4b32",
                  }}
                >
                  <span
                    aria-hidden="true"
                    style={{
                      width: 18,
                      height: 18,
                      borderRadius: 5,
                      background: tile.color,
                      border: "1px solid rgba(0,0,0,0.15)",
                      flexShrink: 0,
                    }}
                  />
                  <span style={{ fontWeight: 700, minWidth: 76 }}>{tile.name}</span>
                  <span style={{ color: "#8a6f50" }}>touches</span>
                  <span style={{ fontWeight: 600 }}>
                    {compatible.map((b) => TILES[b].name).join(", ")}
                  </span>
                </div>
              );
            })}
          </div>
        </section>

        <p style={{ textAlign: "center", fontSize: 11.5, color: C.faint, marginTop: 22, lineHeight: 1.6 }}>
          Five terrain tiles, edge-compatibility adjacency, seeded mulberry32 PRNG. Observe picks the
          minimum-entropy cell with seeded tie breaks; propagation runs a worklist until the grid is stable.
        </p>
      </div>
    </div>
  );
}
