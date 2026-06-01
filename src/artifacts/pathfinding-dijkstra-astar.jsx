import { useState, useEffect, useRef, useMemo, useCallback } from "react";

export const meta = {
  title: "Pathfinding: Dijkstra & A*",
  category: "Algorithms",
  description:
    "Watch Dijkstra flood a grid evenly while A* aims straight at the goal, then read why an admissible heuristic lets A* visit far fewer cells for the same shortest path.",
  date: "2026-06-01",
  tags: ["pathfinding", "dijkstra", "a-star", "graphs"],
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
  open: "#5a8f7b",
  openSoft: "#dceae3",
  closed: "#c9b89c",
  closedSoft: "#efe6d6",
  path: "#c0561f",
  wall: "#3a352f",
  start: "#2f6d4f",
  goal: "#b3341b",
};

const COLS = 16;
const ROWS = 12;

const EASE = "cubic-bezier(0.23, 1, 0.32, 1)";

const key = (r, c) => r * COLS + c;
const inBounds = (r, c) => r >= 0 && r < ROWS && c >= 0 && c < COLS;

const NEIGHBORS = [
  [-1, 0],
  [1, 0],
  [0, -1],
  [0, 1],
];

function manhattan(a, b) {
  return Math.abs(a.r - b.r) + Math.abs(a.c - b.c);
}

class MinHeap {
  constructor() {
    this.items = [];
  }
  get size() {
    return this.items.length;
  }
  push(node) {
    this.items.push(node);
    this._up(this.items.length - 1);
  }
  pop() {
    const items = this.items;
    const top = items[0];
    const last = items.pop();
    if (items.length > 0) {
      items[0] = last;
      this._down(0);
    }
    return top;
  }
  _up(i) {
    const items = this.items;
    while (i > 0) {
      const parent = (i - 1) >> 1;
      if (items[parent].priority <= items[i].priority) break;
      [items[parent], items[i]] = [items[i], items[parent]];
      i = parent;
    }
  }
  _down(i) {
    const items = this.items;
    const n = items.length;
    for (;;) {
      const l = 2 * i + 1;
      const r = 2 * i + 2;
      let smallest = i;
      if (l < n && items[l].priority < items[smallest].priority) smallest = l;
      if (r < n && items[r].priority < items[smallest].priority) smallest = r;
      if (smallest === i) break;
      [items[smallest], items[i]] = [items[i], items[smallest]];
      i = smallest;
    }
  }
}

function planSearch({ start, goal, walls, useHeuristic }) {
  const dist = new Map();
  const cameFrom = new Map();
  const heuristicOf = new Map();
  const frontier = new MinHeap();
  const events = [];

  const startK = key(start.r, start.c);
  const goalK = key(goal.r, goal.c);
  dist.set(startK, 0);
  const startH = useHeuristic ? manhattan(start, goal) : 0;
  heuristicOf.set(startK, startH);
  frontier.push({ r: start.r, c: start.c, priority: startH, g: 0 });

  const expandedOrder = [];

  while (frontier.size > 0) {
    const node = frontier.pop();
    const k = key(node.r, node.c);
    if (node.g > (dist.get(k) ?? Infinity)) continue;

    expandedOrder.push(k);
    const discovered = [];
    if (k !== goalK) {
      for (const [dr, dc] of NEIGHBORS) {
        const nr = node.r + dr;
        const nc = node.c + dc;
        if (!inBounds(nr, nc)) continue;
        const nk = key(nr, nc);
        if (walls.has(nk)) continue;
        const tentative = node.g + 1;
        if (tentative < (dist.get(nk) ?? Infinity)) {
          dist.set(nk, tentative);
          cameFrom.set(nk, k);
          const h = useHeuristic ? manhattan({ r: nr, c: nc }, goal) : 0;
          heuristicOf.set(nk, h);
          frontier.push({ r: nr, c: nc, priority: tentative + h, g: tentative });
          discovered.push(nk);
        }
      }
    }

    events.push({
      expanded: k,
      g: node.g,
      f: node.priority,
      discovered,
      reachedGoal: k === goalK,
    });

    if (k === goalK) break;
  }

  let path = null;
  if (dist.has(goalK)) {
    path = [];
    let cur = goalK;
    while (cur !== undefined) {
      path.push(cur);
      cur = cameFrom.get(cur);
    }
    path.reverse();
  }

  return {
    events,
    path,
    dist,
    heuristicOf,
    found: dist.has(goalK),
    expandedCount: expandedOrder.length,
  };
}

function defaultWalls() {
  const w = new Set();
  for (let r = 2; r <= 8; r++) w.add(key(r, 6));
  for (let c = 6; c <= 11; c++) w.add(key(8, c));
  for (let r = 3; r <= 9; r++) w.add(key(r, 11));
  return w;
}

const START = { r: 6, c: 2 };
const GOAL = { r: 5, c: 13 };

function Legend() {
  const items = [
    { label: "Start", swatch: C.start, text: "#fff" },
    { label: "Goal", swatch: C.goal, text: "#fff" },
    { label: "Wall", swatch: C.wall, text: "#fff" },
    { label: "Frontier (open)", swatch: C.open, text: "#fff" },
    { label: "Visited (closed)", swatch: C.closed, text: C.ink },
    { label: "Shortest path", swatch: C.path, text: "#fff" },
  ];
  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: 10 }}>
      {items.map((it) => (
        <div key={it.label} style={{ display: "flex", alignItems: "center", gap: 7 }}>
          <span
            aria-hidden="true"
            style={{
              width: 16,
              height: 16,
              borderRadius: 4,
              background: it.swatch,
              border: `1px solid ${C.border}`,
              flexShrink: 0,
            }}
          />
          <span style={{ fontSize: 12, color: C.muted }}>{it.label}</span>
        </div>
      ))}
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
        transition: `transform 140ms ${EASE}, background 160ms ease, box-shadow 160ms ease`,
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

function StatBox({ label, value, sub, color, bg }) {
  return (
    <div style={{ background: bg, borderRadius: 10, padding: "12px 14px", flex: 1, minWidth: 110 }}>
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
      <div style={{ fontSize: 26, fontWeight: 700, color, lineHeight: 1, fontVariantNumeric: "tabular-nums" }}>
        {value}
      </div>
      {sub && <div style={{ fontSize: 11, color, opacity: 0.75, marginTop: 4 }}>{sub}</div>}
    </div>
  );
}

export default function App() {
  const [algo, setAlgo] = useState("astar");
  const [walls, setWalls] = useState(() => defaultWalls());
  const [stepIdx, setStepIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [hovered, setHovered] = useState(null);
  const [reduceMotion, setReduceMotion] = useState(false);
  const [paint, setPaint] = useState(null);
  const playRef = useRef(null);

  useEffect(() => {
    if (typeof window === "undefined" || !window.matchMedia) return;
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    const apply = () => setReduceMotion(mq.matches);
    apply();
    mq.addEventListener("change", apply);
    return () => mq.removeEventListener("change", apply);
  }, []);

  const result = useMemo(
    () => planSearch({ start: START, goal: GOAL, walls, useHeuristic: algo === "astar" }),
    [walls, algo]
  );

  const dijkstraResult = useMemo(
    () => planSearch({ start: START, goal: GOAL, walls, useHeuristic: false }),
    [walls]
  );
  const astarResult = useMemo(
    () => planSearch({ start: START, goal: GOAL, walls, useHeuristic: true }),
    [walls]
  );

  const totalSteps = result.events.length;
  const clampedIdx = Math.min(stepIdx, totalSteps);

  useEffect(() => {
    setStepIdx(0);
    setPlaying(false);
  }, [walls, algo]);

  useEffect(() => {
    if (!playing) {
      clearInterval(playRef.current);
      return;
    }
    playRef.current = setInterval(() => {
      setStepIdx((s) => {
        if (s >= totalSteps) {
          setPlaying(false);
          return s;
        }
        return s + 1;
      });
    }, reduceMotion ? 18 : 45);
    return () => clearInterval(playRef.current);
  }, [playing, totalSteps, reduceMotion]);

  const state = useMemo(() => {
    const closed = new Set();
    const openSet = new Set();
    const gShown = new Map();
    let frontierEvent = null;
    const sliced = result.events.slice(0, clampedIdx);
    for (const ev of sliced) {
      closed.add(ev.expanded);
      openSet.delete(ev.expanded);
      gShown.set(ev.expanded, ev.g);
      for (const nk of ev.discovered) {
        if (!closed.has(nk)) openSet.add(nk);
      }
    }
    if (clampedIdx > 0) frontierEvent = result.events[clampedIdx - 1];
    const done = clampedIdx >= totalSteps && totalSteps > 0;
    const lastReachedGoal = sliced.length > 0 && sliced[sliced.length - 1].reachedGoal;
    return { closed, openSet, gShown, frontierEvent, done, lastReachedGoal };
  }, [result, clampedIdx, totalSteps]);

  const showPath = state.lastReachedGoal && result.path;
  const pathSet = useMemo(() => new Set(showPath ? result.path : []), [showPath, result.path]);

  const goalK = key(GOAL.r, GOAL.c);
  const startK = key(START.r, START.c);

  const cellSize = `min(${Math.floor(680 / COLS)}px, calc((100vw - 60px) / ${COLS}))`;

  const toggleWall = useCallback(
    (k, mode) => {
      if (k === startK || k === goalK) return;
      setWalls((prev) => {
        const next = new Set(prev);
        if (mode === "add") next.add(k);
        else if (mode === "remove") next.delete(k);
        else if (next.has(k)) next.delete(k);
        else next.add(k);
        return next;
      });
    },
    [startK, goalK]
  );

  const randomize = useCallback(() => {
    const w = new Set();
    for (let r = 0; r < ROWS; r++) {
      for (let c = 0; c < COLS; c++) {
        const k = key(r, c);
        if (k === startK || k === goalK) continue;
        if (Math.random() < 0.28) w.add(k);
      }
    }
    setWalls(w);
  }, [startK, goalK]);

  const onCellDown = (r, c) => {
    const k = key(r, c);
    if (k === startK || k === goalK) return;
    const mode = walls.has(k) ? "remove" : "add";
    setPaint(mode);
    toggleWall(k, mode);
  };

  useEffect(() => {
    if (!paint) return;
    const up = () => setPaint(null);
    window.addEventListener("pointerup", up);
    return () => window.removeEventListener("pointerup", up);
  }, [paint]);

  const stageText = useMemo(() => {
    if (totalSteps === 0) {
      return {
        title: "No path exists",
        body:
          "Every route from start to goal is blocked by walls. The frontier emptied before reaching the goal, so the search returns failure. Clear some walls and try again.",
      };
    }
    if (clampedIdx === 0) {
      return algo === "astar"
        ? {
            title: "A* is ready: priority = g + h",
            body:
              "A* orders its frontier by f = g + h, where g is the true cost from the start and h is the Manhattan estimate to the goal. Because h points toward the goal, A* expands cells that look promising first. Step through to watch the frontier lean toward the goal.",
          }
        : {
            title: "Dijkstra is ready: priority = g",
            body:
              "Dijkstra orders its frontier purely by g, the cheapest known cost from the start. It has no sense of where the goal is, so it expands outward in all directions like a flood. Step through to watch it spread evenly.",
          };
    }
    if (state.done && showPath) {
      return {
        title: "Shortest path locked in",
        body:
          "The goal was popped from the priority queue, which guarantees its cost is final: nothing cheaper can still be in the frontier. Walking the cameFrom pointers backward reconstructs the shortest path. Both algorithms find the same length; they differ only in how many cells they had to visit to be sure.",
      };
    }
    const ev = state.frontierEvent;
    const expandedR = Math.floor(ev.expanded / COLS);
    const expandedC = ev.expanded % COLS;
    return algo === "astar"
      ? {
          title: `Expanded the lowest-f cell (${expandedR}, ${expandedC})`,
          body:
            `The priority queue handed back the open cell with the smallest f = g + h (here f = ${ev.f}, g = ${ev.g}). A* trusts this estimate because Manhattan distance never overestimates the real cost on a 4-neighbour grid, which keeps it admissible and the answer optimal. Its neighbours join the frontier, sorted again by f.`,
        }
      : {
          title: `Expanded the lowest-g cell (${expandedR}, ${expandedC})`,
          body:
            `The priority queue handed back the open cell with the smallest g = ${ev.g}, the cheapest confirmed distance from the start. With no heuristic, every direction looks equally worth exploring, so the visited region grows as a roughly circular blob. Its neighbours join the frontier, sorted again by g.`,
        };
  }, [algo, clampedIdx, state, showPath, totalSteps]);

  const contrast = dijkstraResult.expandedCount - astarResult.expandedCount;
  const contrastPct =
    dijkstraResult.expandedCount > 0
      ? Math.round((contrast / dijkstraResult.expandedCount) * 100)
      : 0;

  return (
    <div
      style={{
        fontFamily:
          "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif",
        background: C.bg,
        minHeight: "100vh",
        padding: "26px 16px 48px",
        color: C.ink,
      }}
    >
      <style>{`
        .pf-cell:focus-visible {
          outline: 3px solid ${C.accent};
          outline-offset: -3px;
          z-index: 3;
        }
        .pf-grid button { -webkit-tap-highlight-color: transparent; }
        @media (prefers-reduced-motion: reduce) {
          .pf-cell, .pf-anim { transition: none !important; animation: none !important; }
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
            Algorithms / Graph search
          </div>
          <h1 style={{ fontSize: 28, fontWeight: 700, margin: 0, letterSpacing: "-0.02em" }}>
            Dijkstra and A*
          </h1>
          <p
            style={{
              color: C.muted,
              fontSize: 14.5,
              lineHeight: 1.6,
              margin: "10px 0 0",
              maxWidth: "62ch",
            }}
          >
            Both algorithms find the same shortest path on this grid. The difference is how much of
            the map they explore on the way. Toggle between them, then step or play the search and
            watch the visited region change shape.
          </p>
        </header>

        <div
          style={{
            display: "inline-flex",
            background: C.card,
            border: `1px solid ${C.border}`,
            borderRadius: 11,
            padding: 4,
            marginBottom: 18,
            gap: 4,
          }}
          role="radiogroup"
          aria-label="Choose pathfinding algorithm"
        >
          {[
            { id: "dijkstra", label: "Dijkstra", sub: "f = g" },
            { id: "astar", label: "A*", sub: "f = g + h" },
          ].map((opt) => {
            const active = algo === opt.id;
            return (
              <button
                key={opt.id}
                type="button"
                role="radio"
                aria-checked={active}
                onClick={() => setAlgo(opt.id)}
                style={{
                  padding: "8px 18px",
                  borderRadius: 8,
                  border: "none",
                  background: active ? C.accent : "transparent",
                  color: active ? "#fff" : C.muted,
                  fontSize: 14,
                  fontWeight: active ? 600 : 500,
                  fontFamily: "inherit",
                  cursor: "pointer",
                  transition: `background 160ms ease, color 160ms ease`,
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  gap: 1,
                  lineHeight: 1.2,
                }}
              >
                {opt.label}
                <span
                  style={{
                    fontSize: 10.5,
                    fontWeight: 500,
                    opacity: active ? 0.85 : 0.65,
                    fontFamily: "ui-monospace, 'SF Mono', Menlo, monospace",
                  }}
                >
                  {opt.sub}
                </span>
              </button>
            );
          })}
        </div>

        <div
          style={{
            display: "flex",
            gap: 10,
            marginBottom: 18,
            flexWrap: "wrap",
          }}
        >
          <StatBox
            label="Cells visited"
            value={clampedIdx}
            sub={`of ${algo === "astar" ? astarResult.expandedCount : dijkstraResult.expandedCount} total`}
            color="#8a6f3f"
            bg={C.closedSoft}
          />
          <StatBox
            label="Frontier size"
            value={state.openSet.size}
            sub="open cells waiting"
            color={C.open}
            bg={C.openSoft}
          />
          <StatBox
            label="Path length"
            value={showPath ? result.path.length - 1 : "—"}
            sub={showPath ? "steps, start to goal" : "not reached yet"}
            color={C.accent}
            bg={C.accentSoft}
          />
        </div>

        <div
          className="pf-grid"
          role="grid"
          aria-label={`Pathfinding grid, ${ROWS} rows by ${COLS} columns. Activate a cell to toggle a wall.`}
          style={{
            display: "grid",
            gridTemplateColumns: `repeat(${COLS}, ${cellSize})`,
            gap: 2,
            background: C.border,
            padding: 2,
            borderRadius: 12,
            width: "fit-content",
            margin: "0 auto",
            touchAction: "none",
            userSelect: "none",
          }}
        >
          {Array.from({ length: ROWS * COLS }, (_, k) => {
            const r = Math.floor(k / COLS);
            const c = k % COLS;
            const isStart = k === startK;
            const isGoal = k === goalK;
            const isWall = walls.has(k);
            const isPath = pathSet.has(k) && !isStart && !isGoal;
            const isClosed = state.closed.has(k) && !isStart && !isGoal && !isPath;
            const isOpen = state.openSet.has(k) && !isStart && !isGoal && !isPath;
            const isFrontierHead = state.frontierEvent && state.frontierEvent.expanded === k;

            let bg = C.card;
            let fg = C.faint;
            if (isWall) {
              bg = C.wall;
              fg = "#fff";
            } else if (isStart) {
              bg = C.start;
              fg = "#fff";
            } else if (isGoal) {
              bg = C.goal;
              fg = "#fff";
            } else if (isPath) {
              bg = C.path;
              fg = "#fff";
            } else if (isOpen) {
              bg = C.openSoft;
              fg = "#3c5a4c";
            } else if (isClosed) {
              bg = C.closedSoft;
              fg = "#8a6f3f";
            }

            const g = state.gShown.get(k);
            const h = algo === "astar" ? result.heuristicOf.get(k) : undefined;
            const showLabels =
              !isWall && (isStart || isOpen || isClosed || isPath) && g !== undefined;

            const label = isStart
              ? "Start"
              : isGoal
              ? "Goal"
              : isWall
              ? "Wall"
              : `Cell ${r}, ${c}`;

            return (
              <button
                key={k}
                type="button"
                role="gridcell"
                aria-label={label}
                className="pf-cell pf-anim"
                onPointerDown={() => onCellDown(r, c)}
                onPointerEnter={() => {
                  if (paint) toggleWall(k, paint);
                  setHovered(k);
                }}
                onPointerLeave={() => setHovered((h2) => (h2 === k ? null : h2))}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    toggleWall(k, "toggle");
                  }
                }}
                style={{
                  width: cellSize,
                  height: cellSize,
                  border: isFrontierHead ? `2px solid ${C.accent}` : "none",
                  padding: 0,
                  borderRadius: 3,
                  background: bg,
                  color: fg,
                  cursor: isStart || isGoal ? "default" : "pointer",
                  position: "relative",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  fontFamily: "ui-monospace, 'SF Mono', Menlo, monospace",
                  transition: reduceMotion
                    ? "none"
                    : `background 220ms ${EASE}, transform 160ms ${EASE}`,
                  transform: isFrontierHead && !reduceMotion ? "scale(1.06)" : "scale(1)",
                  zIndex: isFrontierHead ? 2 : 1,
                  overflow: "hidden",
                }}
              >
                {isStart && <span style={{ fontSize: 11, fontWeight: 700 }}>S</span>}
                {isGoal && <span style={{ fontSize: 11, fontWeight: 700 }}>G</span>}
                {showLabels && !isStart && (
                  <span style={{ fontSize: 8.5, lineHeight: 1.05, fontWeight: 600, textAlign: "center" }}>
                    {algo === "astar" && h !== undefined ? (
                      <>
                        {g + h}
                        <br />
                        <span style={{ fontWeight: 400, opacity: 0.7, fontSize: 7.5 }}>
                          {g}+{h}
                        </span>
                      </>
                    ) : (
                      g
                    )}
                  </span>
                )}
              </button>
            );
          })}
        </div>

        {hovered !== null && (() => {
          const r = Math.floor(hovered / COLS);
          const c = hovered % COLS;
          const g = state.gShown.get(hovered);
          const h = algo === "astar" ? result.heuristicOf.get(hovered) : manhattan({ r, c }, GOAL);
          const visited = state.closed.has(hovered) || state.openSet.has(hovered);
          return (
            <div
              style={{
                textAlign: "center",
                marginTop: 10,
                fontSize: 12.5,
                color: C.muted,
                fontFamily: "ui-monospace, 'SF Mono', Menlo, monospace",
              }}
            >
              ({r}, {c}) &nbsp;
              {visited ? (
                <>
                  g = <b style={{ color: C.ink }}>{g}</b>
                  {algo === "astar" && (
                    <>
                      {" "}&middot; h = <b style={{ color: C.ink }}>{h}</b> &middot; f ={" "}
                      <b style={{ color: C.accent }}>{g + h}</b>
                    </>
                  )}
                </>
              ) : (
                <span style={{ color: C.faint }}>
                  not visited &middot; h to goal = {h}
                </span>
              )}
            </div>
          );
        })()}

        <div
          style={{
            display: "flex",
            gap: 9,
            flexWrap: "wrap",
            margin: "20px 0 16px",
            justifyContent: "center",
          }}
        >
          <ControlButton
            onClick={() => setStepIdx((s) => Math.min(s + 1, totalSteps))}
            disabled={clampedIdx >= totalSteps}
            ariaLabel="Expand one node"
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
            ariaLabel={playing ? "Pause animation" : "Play animation"}
          >
            {playing ? "Pause" : clampedIdx >= totalSteps && totalSteps > 0 ? "Replay" : "Play"}
          </ControlButton>
          <ControlButton
            onClick={() => {
              setStepIdx(0);
              setPlaying(false);
            }}
            ariaLabel="Reset the search to the beginning"
          >
            Reset
          </ControlButton>
          <ControlButton onClick={() => setWalls(new Set())} ariaLabel="Clear all walls">
            Clear walls
          </ControlButton>
          <ControlButton onClick={randomize} ariaLabel="Randomize walls">
            Randomize
          </ControlButton>
        </div>

        <Legend />

        <section
          className="pf-anim"
          aria-live="polite"
          style={{
            background: C.card,
            border: `1px solid ${C.border}`,
            borderRadius: 14,
            padding: "18px 20px",
            marginTop: 18,
            transition: reduceMotion ? "none" : `box-shadow 200ms ease`,
          }}
        >
          <div
            style={{
              fontSize: 11,
              letterSpacing: "0.08em",
              textTransform: "uppercase",
              color: C.accent,
              fontWeight: 700,
              marginBottom: 7,
            }}
          >
            {algo === "astar" ? "A* search" : "Dijkstra"}
            {totalSteps > 0 && ` · step ${clampedIdx} of ${totalSteps}`}
          </div>
          <h2 style={{ fontSize: 17, fontWeight: 700, margin: "0 0 8px", letterSpacing: "-0.01em" }}>
            {stageText.title}
          </h2>
          <p style={{ fontSize: 14, lineHeight: 1.65, color: C.muted, margin: 0, maxWidth: "66ch" }}>
            {stageText.body}
          </p>
        </section>

        <section
          style={{
            background: C.accentSoft,
            border: `1px solid ${C.accent}33`,
            borderRadius: 14,
            padding: "18px 20px",
            marginTop: 14,
          }}
        >
          <h2 style={{ fontSize: 15, fontWeight: 700, margin: "0 0 10px" }}>
            Why A* explores less
          </h2>
          <div style={{ display: "flex", gap: 16, flexWrap: "wrap", marginBottom: 12 }}>
            <div style={{ flex: 1, minWidth: 130 }}>
              <div style={{ fontSize: 12, color: C.muted, marginBottom: 3 }}>Dijkstra visits</div>
              <div style={{ fontSize: 30, fontWeight: 700, color: "#8a6f3f", lineHeight: 1, fontVariantNumeric: "tabular-nums" }}>
                {dijkstraResult.expandedCount}
              </div>
              <div style={{ fontSize: 12, color: C.faint }}>cells</div>
            </div>
            <div style={{ flex: 1, minWidth: 130 }}>
              <div style={{ fontSize: 12, color: C.muted, marginBottom: 3 }}>A* visits</div>
              <div style={{ fontSize: 30, fontWeight: 700, color: C.accent, lineHeight: 1, fontVariantNumeric: "tabular-nums" }}>
                {astarResult.expandedCount}
              </div>
              <div style={{ fontSize: 12, color: C.faint }}>cells</div>
            </div>
            <div style={{ flex: 1, minWidth: 130 }}>
              <div style={{ fontSize: 12, color: C.muted, marginBottom: 3 }}>Path length</div>
              <div style={{ fontSize: 30, fontWeight: 700, color: C.start, lineHeight: 1, fontVariantNumeric: "tabular-nums" }}>
                {astarResult.found ? astarResult.path.length - 1 : "—"}
              </div>
              <div style={{ fontSize: 12, color: C.faint }}>
                {astarResult.found ? "identical for both" : "no path"}
              </div>
            </div>
          </div>
          <p style={{ fontSize: 14, lineHeight: 1.65, color: "#6b4b32", margin: 0, maxWidth: "66ch" }}>
            {dijkstraResult.found && astarResult.found && contrast > 0 ? (
              <>
                On the current walls A* touches {contrastPct}% fewer cells than Dijkstra yet returns a
                path of the same length. The heuristic h is the whole reason: it adds an estimate of
                the remaining distance to each cell&rsquo;s priority, so the queue keeps handing back
                cells that head toward the goal instead of cells that merely sit close to the start.
                Because Manhattan distance never overestimates the true cost (it is{" "}
                <i>admissible</i>), A* can prune those detours without ever risking the optimal
                answer. Drop walls in front of the goal and the gap widens; in an open room with no
                walls the two look nearly identical because there are no detours to skip.
              </>
            ) : !astarResult.found ? (
              <>
                There is no path right now, so both searches drain their frontier and report failure.
                Clear a channel between start and goal to compare them.
              </>
            ) : (
              <>
                On these particular walls the heuristic has little to prune, so the visited counts are
                close. Add a barrier that forces a detour and A* pulls ahead.
              </>
            )}
          </p>
        </section>

        <p
          style={{
            textAlign: "center",
            fontSize: 11.5,
            color: C.faint,
            marginTop: 22,
            lineHeight: 1.6,
          }}
        >
          4-neighbour movement, uniform step cost of 1, Manhattan-distance heuristic. Click or drag
          to draw and erase walls; the search recomputes live.
        </p>
      </div>
    </div>
  );
}
