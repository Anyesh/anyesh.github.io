import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import {
  Bar,
  BarChart,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { DIFFUSION_LM } from "./data/diffusion-lm-weights.js";
import { createModel, createSampler, mulberry32 } from "./data/diffusion-lm-core.js";

export const meta = {
  title: "Diffusion Language Models: All the Words at Once",
  category: "LLM Systems",
  description:
    "A normal LLM writes one token per forward pass because each word has to wait for the last. A diffusion language model drafts the whole line and firms it up over a few passes. Race both decode orders on the same tiny model trained for this page, dissolve a sentence into masks, give the denoiser fewer steps and read the damage, then pin both ends of a story and let the middle fill itself in.",
  date: "2026-07-30",
  tags: ["diffusion", "llm", "text-generation", "parallel-decoding"],
};

const C = {
  bg: "#f7f5f2",
  card: "#ffffff",
  border: "#e4dfd8",
  ink: "#1c1a17",
  muted: "#857c72",
  faint: "#efeae3",
  accent: "#c0561f",
  accentSoft: "#f6ece5",
  blue: "#1e6fa0",
  blueSoft: "#e8f0f5",
  green: "#5a7d3c",
  maskBlock: "#ddd6ca",
  danger: "#a8453f",
};

const MONO = "'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, monospace";
const EASE = "cubic-bezier(0.23, 1, 0.32, 1)";
const RACE_STEPS = 8;

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

function SectionTitle({ children }) {
  return (
    <h2 style={{ fontSize: 19, fontWeight: 700, margin: "34px 0 8px", lineHeight: 1.25, textWrap: "balance" }}>
      {children}
    </h2>
  );
}

function Prose({ children, style }) {
  return (
    <p style={{ fontSize: 14, lineHeight: 1.65, color: C.ink, margin: "0 0 14px", maxWidth: "68ch", ...style }}>
      {children}
    </p>
  );
}

function Caption({ children, style }) {
  return (
    <p style={{ fontSize: 12, lineHeight: 1.6, color: C.muted, margin: "10px 0 0", ...style }}>{children}</p>
  );
}

function Btn({ children, onClick, disabled, primary, ariaLabel, ariaPressed }) {
  return (
    <button
      type="button"
      className="dlm-press"
      onClick={onClick}
      disabled={disabled}
      aria-label={ariaLabel}
      aria-pressed={ariaPressed}
      style={{
        padding: primary ? "8px 18px" : "7px 14px",
        borderRadius: 8,
        border: primary ? "none" : `1px solid ${C.border}`,
        background: primary ? C.accent : "transparent",
        color: primary ? "#fff" : C.ink,
        fontSize: 13,
        fontWeight: primary ? 600 : 500,
        fontFamily: "inherit",
        cursor: disabled ? "not-allowed" : "pointer",
        opacity: disabled ? 0.4 : 1,
        transition: `transform 140ms ${EASE}, background 160ms ease`,
      }}
    >
      {children}
    </button>
  );
}

function Pill({ children, active, onClick, ariaLabel }) {
  return (
    <button
      type="button"
      className="dlm-press"
      onClick={onClick}
      aria-pressed={active}
      aria-label={ariaLabel}
      style={{
        padding: "6px 12px",
        borderRadius: 999,
        border: `1.5px solid ${active ? C.accent : C.border}`,
        background: active ? C.accentSoft : "transparent",
        color: active ? C.accent : C.muted,
        fontWeight: active ? 700 : 500,
        fontSize: 12.5,
        cursor: "pointer",
        fontFamily: "inherit",
        transition: `background 160ms ease, border-color 160ms ease, color 160ms ease`,
      }}
    >
      {children}
    </button>
  );
}

function Stat({ label, value, unit, tone }) {
  return (
    <div
      style={{
        flex: "1 1 140px",
        background: C.faint,
        borderRadius: 10,
        padding: "12px 14px",
      }}
    >
      <div style={{ fontSize: 20, fontWeight: 700, color: tone || C.ink, lineHeight: 1.1 }}>
        {value}
        {unit && <span style={{ fontSize: 11, fontWeight: 500, color: C.muted, marginLeft: 4 }}>{unit}</span>}
      </div>
      <div style={{ fontSize: 11, color: C.muted, marginTop: 4, lineHeight: 1.4 }}>{label}</div>
    </div>
  );
}

function TokenCell({ cell, reduce }) {
  const base = {
    fontFamily: MONO,
    fontSize: 12,
    lineHeight: "20px",
    borderRadius: 6,
    padding: "1px 6px",
    minWidth: 26,
    textAlign: "center",
    height: 22,
    boxSizing: "border-box",
  };
  if (cell.kind === "mask") {
    return (
      <span
        style={{
          ...base,
          background: C.maskBlock,
          color: "transparent",
          userSelect: "none",
          transition: reduce ? "none" : `background 240ms ${EASE}`,
        }}
      >
        &middot;&middot;
      </span>
    );
  }
  if (cell.kind === "ghost") {
    const op = 0.2 + 0.75 * Math.sqrt(Math.min(1, cell.prob));
    return (
      <span
        title={`${cell.word}: ${(cell.prob * 100).toFixed(1)}% confident`}
        style={{
          ...base,
          background: C.blueSoft,
          border: `1px dashed ${C.blue}55`,
          padding: "0 5px",
          color: C.blue,
          opacity: op,
        }}
      >
        {cell.word}
      </span>
    );
  }
  return (
    <span
      style={{
        ...base,
        background: cell.kind === "pinned" ? "#fff" : cell.kind === "fresh" ? C.accentSoft : "#fff",
        border: cell.kind === "pinned" ? `1.5px solid ${C.accent}` : `1px solid ${C.border}`,
        padding: cell.kind === "pinned" ? "0 5px" : "1px 6px",
        color: C.ink,
        fontWeight: cell.kind === "pinned" ? 700 : 500,
        animation: cell.kind === "fresh" && !reduce ? `dlm-settle 300ms ${EASE}` : "none",
      }}
    >
      {cell.word}
    </span>
  );
}

function TokenCanvas({ cells, reduce, ariaLabel, style }) {
  return (
    <div
      role="img"
      aria-label={ariaLabel}
      style={{
        display: "flex",
        flexWrap: "wrap",
        gap: 4,
        padding: 10,
        borderRadius: 10,
        background: C.faint,
        alignContent: "flex-start",
        ...style,
      }}
    >
      {cells.map((c, i) => (
        <TokenCell key={i} cell={c} reduce={reduce} />
      ))}
    </div>
  );
}

function canvasCells(vocab, ids, canvas, freshSet, ghosts) {
  const ghostAt = new Map((ghosts || []).map((g) => [g.pos, g]));
  return ids.map((t, i) => {
    if (t === 1) {
      const g = ghostAt.get(i);
      if (g) return { kind: "ghost", word: vocab[g.tok], prob: g.prob };
      return { kind: "mask" };
    }
    if (canvas[i] !== null && canvas[i] >= 0) return { kind: "pinned", word: vocab[t] };
    if (freshSet && freshSet.has(i)) return { kind: "fresh", word: vocab[t] };
    return { kind: "word", word: vocab[t] };
  });
}

function describeCanvas(vocab, ids) {
  return ids.map((t) => (t === 1 ? "blank" : vocab[t])).join(" ");
}

function useTicker(active, intervalMs, onTick) {
  const cb = useRef(onTick);
  cb.current = onTick;
  useEffect(() => {
    if (!active) return;
    const iv = setInterval(() => cb.current(), intervalMs);
    return () => clearInterval(iv);
  }, [active, intervalMs]);
}

const RACE_PROMPT = ["once", "upon", "a", "time", ","];

function RaceDemo({ model, reduce }) {
  const ctx = model.cfg.ctx;
  const vocab = model.vocab;
  const buildCanvas = useCallback(() => {
    const c = new Array(ctx).fill(null);
    RACE_PROMPT.forEach((w, i) => {
      c[i] = model.stoi.get(w);
    });
    return c;
  }, [ctx, model]);

  const samplers = useRef(null);
  const canvasRef = useRef(null);
  const [panels, setPanels] = useState(null);
  const [running, setRunning] = useState(false);
  const [started, setStarted] = useState(false);

  const reset = useCallback(() => {
    const canvas = buildCanvas();
    canvasRef.current = canvas;
    samplers.current = {
      l: createSampler(model, canvas, { order: "ltr", temperature: 0 }),
      r: createSampler(model, canvas, { order: "confidence", steps: RACE_STEPS, temperature: 0 }),
    };
    setPanels({
      l: { ids: samplers.current.l.ids, fresh: new Set(), forwards: 0, done: false },
      r: { ids: samplers.current.r.ids, fresh: new Set(), forwards: 0, done: false },
    });
    setRunning(false);
    setStarted(false);
  }, [buildCanvas, model]);

  useEffect(() => {
    reset();
  }, [reset]);

  useTicker(running, reduce ? 560 : 420, () => {
    const s = samplers.current;
    // Samplers mutate outside the state updater because React may invoke
    // updaters twice under StrictMode, which would double-step the race.
    const lr = s.l.maskedLeft > 0 ? s.l.step() : null;
    const rr = s.r.maskedLeft > 0 ? s.r.step() : null;
    setPanels((prev) => ({
      l: lr
        ? { ids: lr.ids, fresh: new Set(lr.placed.map((p) => p.pos)), forwards: lr.forwards, done: lr.done }
        : { ...prev.l, fresh: new Set(), done: true },
      r: rr
        ? { ids: rr.ids, fresh: new Set(rr.placed.map((p) => p.pos)), forwards: rr.forwards, done: rr.done }
        : { ...prev.r, fresh: new Set(), done: true },
    }));
    if (s.l.maskedLeft === 0 && s.r.maskedLeft === 0) setRunning(false);
  });

  if (!panels) return null;
  const total = samplers.current.l.totalMasked;
  const bothDone = panels.l.done && panels.r.done;

  const panel = (side, title, sub) => {
    const p = panels[side];
    return (
      <div style={{ flex: "1 1 300px", minWidth: 0 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 8, marginBottom: 6 }}>
          <div style={{ fontSize: 12.5, fontWeight: 700, color: C.ink }}>{title}</div>
          <div style={{ fontSize: 11, color: C.muted }}>{sub}</div>
        </div>
        <TokenCanvas
          cells={canvasCells(vocab, p.ids, canvasRef.current, p.fresh)}
          reduce={reduce}
          ariaLabel={`${title}. ${describeCanvas(vocab, p.ids)}`}
          style={{ minHeight: 128 }}
        />
        <div style={{ display: "flex", gap: 8, marginTop: 8 }}>
          <Stat label="forward passes" value={p.forwards} tone={side === "r" ? C.accent : C.blue} />
          <Stat label="tokens placed" value={`${total - (p.ids.filter((t) => t === 1).length)} / ${total}`} />
        </div>
      </div>
    );
  };

  return (
    <Card>
      <div style={{ display: "flex", gap: 18, flexWrap: "wrap" }}>
        {panel("l", "One token per pass", "autoregressive order")}
        {panel("r", `Whole canvas, ${RACE_STEPS} passes`, "parallel confidence order")}
      </div>
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginTop: 14, alignItems: "center" }}>
        {!started || bothDone ? (
          <Btn
            primary
            onClick={() => {
              if (bothDone) reset();
              setStarted(true);
              setRunning(true);
            }}
            ariaLabel={bothDone ? "Replay the race" : "Start the race"}
          >
            {bothDone ? "Replay" : "Run the race"}
          </Btn>
        ) : (
          <Btn primary onClick={() => setRunning((r) => !r)} ariaLabel={running ? "Pause the race" : "Resume the race"}>
            {running ? "Pause" : "Resume"}
          </Btn>
        )}
        <Btn onClick={reset} ariaLabel="Reset both panels">
          Reset
        </Btn>
        {bothDone && (
          <span style={{ fontSize: 12, color: C.green, fontWeight: 600 }}>
            {panels.r.forwards} passes beat {panels.l.forwards}
          </span>
        )}
      </div>
      <Caption>
        Same network, same weights, same starting prompt. The only thing that changed is the order in which
        blanks get committed. Both finish a 32 slot canvas; the right panel commits its most confident blanks
        in batches, so it needs {RACE_STEPS} forward passes where the left needs one per token.
      </Caption>
    </Card>
  );
}

const CORRUPT_WORDS =
  "once upon a time , there was a little girl named lily . she liked to play outside with her friends . they had fun in the sun .".split(" ");

function CorruptionDemo({ reduce }) {
  const [t, setT] = useState(0.35);
  const thresholds = useMemo(() => {
    const rng = mulberry32(20260730);
    return CORRUPT_WORDS.map(() => rng());
  }, []);
  const cells = CORRUPT_WORDS.map((w, i) =>
    thresholds[i] < t ? { kind: "mask" } : { kind: "word", word: w }
  );
  const maskedCount = cells.filter((c) => c.kind === "mask").length;

  return (
    <Card>
      <TokenCanvas
        cells={cells}
        reduce={reduce}
        ariaLabel={`Sentence at corruption level ${Math.round(t * 100)} percent: ${cells
          .map((c) => (c.kind === "mask" ? "blank" : c.word))
          .join(" ")}`}
      />
      <div style={{ marginTop: 14 }}>
        <label
          htmlFor="dlm-corrupt"
          style={{ display: "flex", justifyContent: "space-between", fontSize: 12, color: C.muted, marginBottom: 6 }}
        >
          <span>corruption level t</span>
          <span style={{ fontWeight: 700, color: C.accent }}>
            t = {t.toFixed(2)} &middot; {maskedCount}/{CORRUPT_WORDS.length} masked
          </span>
        </label>
        <input
          id="dlm-corrupt"
          type="range"
          min={0}
          max={100}
          value={Math.round(t * 100)}
          onChange={(e) => setT(+e.target.value / 100)}
          style={{ width: "100%", accentColor: C.accent }}
        />
      </div>
      <Caption>
        Each position holds a fixed lottery ticket, so a word that disappears at t = 0.4 is still gone at every
        higher t. That is the absorbing state trick from D3PM: masking only ever accumulates, which keeps the
        training story consistent across corruption levels. Slide to t = 1 and the canvas is all mask, and that
        is exactly the input generation starts from.
      </Caption>
    </Card>
  );
}

const PRESETS = [
  { name: "once upon a time ,", words: ["once", "upon", "a", "time", ","] },
  { name: "one day , a little girl", words: ["one", "day", ",", "a", "little", "girl"] },
  { name: "tom and his mom", words: ["tom", "and", "his", "mom"] },
  { name: "no prompt", words: [] },
];
const STEP_CHOICES = [1, 2, 4, 8, 16, 32];
const ORDERS = [
  { id: "confidence", label: "parallel" },
  { id: "ltr", label: "left to right" },
  { id: "random", label: "random" },
];

function Playground({ model, reduce }) {
  const ctx = model.cfg.ctx;
  const vocab = model.vocab;
  const [preset, setPreset] = useState(0);
  const [steps, setSteps] = useState(8);
  const [order, setOrder] = useState("confidence");
  const [temp, setTemp] = useState(0);
  const [seed, setSeed] = useState(7);
  const [view, setView] = useState(null);
  const [playing, setPlaying] = useState(false);
  const [lastMs, setLastMs] = useState(null);
  const samplerRef = useRef(null);
  const canvasRef = useRef(null);

  const rebuild = useCallback(() => {
    const canvas = new Array(ctx).fill(null);
    PRESETS[preset].words.forEach((w, i) => {
      canvas[i] = model.stoi.get(w);
    });
    canvasRef.current = canvas;
    samplerRef.current = createSampler(model, canvas, { steps, order, temperature: temp, seed });
    setView({ ids: samplerRef.current.ids, fresh: new Set(), ghosts: [], forwards: 0, done: false });
    setPlaying(false);
  }, [ctx, model, preset, steps, order, temp, seed]);

  useEffect(() => {
    rebuild();
  }, [rebuild]);

  const stepOnce = useCallback(() => {
    const s = samplerRef.current;
    if (!s || s.maskedLeft === 0) {
      setPlaying(false);
      return;
    }
    const t0 = performance.now();
    const r = s.step();
    setLastMs(performance.now() - t0);
    const placedSet = new Set(r.placed.map((p) => p.pos));
    setView({
      ids: r.ids,
      fresh: placedSet,
      ghosts: r.candidates.filter((c) => !placedSet.has(c.pos)),
      forwards: r.forwards,
      done: r.done,
    });
    if (r.done) setPlaying(false);
  }, []);

  useTicker(playing, reduce ? 620 : 460, stepOnce);

  if (!view) return null;
  const totalMasked = samplerRef.current.totalMasked;
  const expected = order === "ltr" ? totalMasked : Math.min(steps, totalMasked);

  return (
    <Card>
      <div style={{ display: "flex", gap: 14, flexWrap: "wrap", marginBottom: 14 }}>
        <div>
          <div style={{ fontSize: 11, fontWeight: 700, color: C.muted, textTransform: "uppercase", letterSpacing: 0.8, marginBottom: 6 }}>
            Prompt
          </div>
          <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
            {PRESETS.map((p, i) => (
              <Pill key={p.name} active={preset === i} onClick={() => setPreset(i)}>
                {p.name}
              </Pill>
            ))}
          </div>
        </div>
        <div>
          <div style={{ fontSize: 11, fontWeight: 700, color: C.muted, textTransform: "uppercase", letterSpacing: 0.8, marginBottom: 6 }}>
            Decode order
          </div>
          <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
            {ORDERS.map((o) => (
              <Pill key={o.id} active={order === o.id} onClick={() => setOrder(o.id)}>
                {o.label}
              </Pill>
            ))}
          </div>
        </div>
      </div>

      <div style={{ display: "flex", gap: 18, flexWrap: "wrap", marginBottom: 14 }}>
        <div>
          <div style={{ fontSize: 11, fontWeight: 700, color: C.muted, textTransform: "uppercase", letterSpacing: 0.8, marginBottom: 6 }}>
            Denoising steps {order === "ltr" && <span style={{ fontWeight: 400, textTransform: "none" }}>(ignored: one token per pass)</span>}
          </div>
          <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
            {STEP_CHOICES.map((s) => (
              <Pill key={s} active={steps === s} onClick={() => setSteps(s)} ariaLabel={`${s} denoising steps`}>
                {s}
              </Pill>
            ))}
          </div>
        </div>
        <div style={{ flex: "1 1 180px", maxWidth: 260 }}>
          <label
            htmlFor="dlm-temp"
            style={{ display: "flex", justifyContent: "space-between", fontSize: 11, fontWeight: 700, color: C.muted, textTransform: "uppercase", letterSpacing: 0.8, marginBottom: 6 }}
          >
            <span>Temperature</span>
            <span style={{ color: C.accent }}>{temp === 0 ? "greedy" : temp.toFixed(1)}</span>
          </label>
          <input
            id="dlm-temp"
            type="range"
            min={0}
            max={12}
            value={Math.round(temp * 10)}
            onChange={(e) => setTemp(+e.target.value / 10)}
            style={{ width: "100%", accentColor: C.accent }}
          />
        </div>
      </div>

      <TokenCanvas
        cells={canvasCells(vocab, view.ids, canvasRef.current, view.fresh, view.ghosts)}
        reduce={reduce}
        ariaLabel={`Canvas after ${view.forwards} forward passes: ${describeCanvas(vocab, view.ids)}`}
        style={{ minHeight: 128 }}
      />

      <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginTop: 12, alignItems: "center" }}>
        <Btn primary onClick={stepOnce} disabled={view.done} ariaLabel="Advance one forward pass">
          Step
        </Btn>
        <Btn onClick={() => setPlaying((p) => !p)} disabled={view.done} ariaLabel={playing ? "Pause autoplay" : "Autoplay all steps"}>
          {playing ? "Pause" : "Play"}
        </Btn>
        <Btn
          onClick={() => {
            setSeed((s) => s + 1);
          }}
          ariaLabel="Reset the canvas with a new random seed"
        >
          Reset
        </Btn>
        <span style={{ marginLeft: "auto", fontSize: 12, color: C.muted }}>
          pass <b style={{ color: C.ink }}>{view.forwards}</b> / {expected}
          {lastMs !== null && <span> &middot; last forward {lastMs.toFixed(1)} ms</span>}
        </span>
      </div>

      <Caption>
        Blue ghost words are the model&apos;s current best guess for each still open slot, drawn at an opacity
        that tracks its confidence; a Step locks in the strongest of them. At 32 steps this model is as careful
        as its left to right twin. At 4 it must commit eight words per pass and the seams show: repeated words,
        broken agreement. SEDD measured this dial directly and matched GPT-2 quality with roughly 32 times fewer
        network evaluations.
      </Caption>
    </Card>
  );
}

const INFILL_WORDS =
  "one day , tom went to the park with his mom . he saw a big dog and a little cat . they all played together and were very happy .".split(" ");
const DEFAULT_PINS = [0, 1, 2, 3, 4, 26, 27, 28, 29, 30];

function InfillDemo({ model, reduce }) {
  const vocab = model.vocab;
  const [pinned, setPinned] = useState(() => new Set(DEFAULT_PINS));
  const [view, setView] = useState(null);
  const [playing, setPlaying] = useState(false);
  const samplerRef = useRef(null);
  const canvasRef = useRef(null);

  const start = useCallback(() => {
    const canvas = INFILL_WORDS.map((w, i) => (pinned.has(i) ? model.stoi.get(w) : null));
    canvasRef.current = canvas;
    samplerRef.current = createSampler(model, canvas, { steps: 8, order: "confidence", temperature: 0 });
    setView({ ids: samplerRef.current.ids, fresh: new Set(), forwards: 0, done: false });
    setPlaying(true);
  }, [model, pinned]);

  useTicker(playing, reduce ? 560 : 420, () => {
    const s = samplerRef.current;
    if (!s || s.maskedLeft === 0) {
      setPlaying(false);
      return;
    }
    const r = s.step();
    setView({
      ids: r.ids,
      fresh: new Set(r.placed.map((p) => p.pos)),
      forwards: r.forwards,
      done: r.done,
    });
    if (r.done) setPlaying(false);
  });

  const allPinned = pinned.size === INFILL_WORDS.length;

  return (
    <Card>
      <div style={{ fontSize: 11, fontWeight: 700, color: C.muted, textTransform: "uppercase", letterSpacing: 0.8, marginBottom: 6 }}>
        Tap words to pin or free them
      </div>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 4, padding: 10, borderRadius: 10, background: C.faint }}>
        {INFILL_WORDS.map((w, i) => {
          const isPinned = pinned.has(i);
          return (
            <button
              key={i}
              type="button"
              className="dlm-press"
              aria-pressed={isPinned}
              aria-label={`${w}, ${isPinned ? "pinned, tap to free" : "free, tap to pin"}`}
              onClick={() => {
                setPinned((prev) => {
                  const next = new Set(prev);
                  if (next.has(i)) next.delete(i);
                  else next.add(i);
                  return next;
                });
                setView(null);
                setPlaying(false);
              }}
              style={{
                fontFamily: MONO,
                fontSize: 12,
                lineHeight: "20px",
                borderRadius: 6,
                padding: "0 5px",
                height: 22,
                boxSizing: "border-box",
                cursor: "pointer",
                background: isPinned ? "#fff" : C.maskBlock,
                border: isPinned ? `1.5px solid ${C.accent}` : `1px solid transparent`,
                color: isPinned ? C.ink : "#8d857a",
                fontWeight: isPinned ? 700 : 500,
                transition: `background 180ms ease, border-color 180ms ease, color 180ms ease`,
              }}
            >
              {w}
            </button>
          );
        })}
      </div>

      {view && (
        <div style={{ marginTop: 12 }}>
          <div style={{ fontSize: 11, fontWeight: 700, color: C.muted, textTransform: "uppercase", letterSpacing: 0.8, marginBottom: 6 }}>
            The model&apos;s fill
          </div>
          <TokenCanvas
            cells={canvasCells(vocab, view.ids, canvasRef.current, view.fresh)}
            reduce={reduce}
            ariaLabel={`Model fill after ${view.forwards} passes: ${describeCanvas(vocab, view.ids)}`}
          />
        </div>
      )}

      <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginTop: 12, alignItems: "center" }}>
        <Btn primary onClick={start} disabled={allPinned || playing} ariaLabel="Regenerate the freed words">
          {view ? "Fill again" : "Fill the gaps"}
        </Btn>
        <Btn
          onClick={() => {
            setPinned(new Set(DEFAULT_PINS));
            setView(null);
            setPlaying(false);
          }}
          ariaLabel="Restore the default pinned words"
        >
          Reset pins
        </Btn>
        {allPinned && <span style={{ fontSize: 12, color: C.muted }}>free at least one word to fill</span>}
        {view?.done && (
          <span style={{ fontSize: 12, color: C.green, fontWeight: 600 }}>
            filled in {view.forwards} passes
          </span>
        )}
      </div>
      <Caption>
        The pinned words never move; everything gray is regenerated to agree with both sides at once. Pin only
        the ending and watch the model write toward it, something a left to right decoder cannot do without
        tricks, because a causal model has no way to let early words depend on later ones.
      </Caption>
    </Card>
  );
}

const SPEED_DIFFUSION = [
  { label: "Mercury Coder Mini (Inception Labs)", value: "1,109" },
  { label: "Gemini Diffusion, sampling speed (DeepMind)", value: "1,479" },
  { label: "DiffusionGemma on one H100 (Google)", value: "1,000+" },
];
const SPEED_AR = [
  { label: "Gemini 2.0 Flash-Lite", value: "201" },
  { label: "GPT-4o Mini", value: "59" },
  { label: "Claude 3.5 Haiku", value: "61" },
];

const BENCH = [
  { name: "HumanEval", gloss: "write small Python functions from a docstring", diffusion: 89.6, ar: 90.2 },
  { name: "MBPP", gloss: "solve short everyday Python problems", diffusion: 76.0, ar: 75.8 },
  { name: "LiveCode v6", gloss: "recent competitive programming problems (LiveCodeBench v6)", diffusion: 30.9, ar: 28.5 },
  { name: "GPQA", gloss: "graduate level science questions (GPQA Diamond)", diffusion: 40.4, ar: 56.5 },
  { name: "MMLU-Lite", gloss: "broad knowledge quiz across many languages (Global MMLU-Lite)", diffusion: 69.1, ar: 79.0 },
];

function BenchChart() {
  return (
    <div>
      <ResponsiveContainer width="100%" height={240}>
        <BarChart data={BENCH} margin={{ top: 8, right: 8, left: -16, bottom: 0 }} barGap={2}>
          <XAxis dataKey="name" tick={{ fontSize: 10, fill: C.muted }} stroke={C.border} interval={0} />
          <YAxis domain={[0, 100]} tick={{ fontSize: 10, fill: C.muted }} stroke={C.border} />
          <Tooltip
            cursor={{ fill: C.faint }}
            contentStyle={{ fontSize: 12, borderRadius: 8, border: `1px solid ${C.border}` }}
            formatter={(v, key) => [v, key === "diffusion" ? "Gemini Diffusion" : "Gemini 2.0 Flash-Lite"]}
            labelFormatter={(label) => {
              const b = BENCH.find((x) => x.name === label);
              return b ? `${b.name}: ${b.gloss}` : label;
            }}
          />
          <Legend
            wrapperStyle={{ fontSize: 12 }}
            formatter={(v) => (v === "diffusion" ? "Gemini Diffusion" : "Gemini 2.0 Flash-Lite (autoregressive)")}
          />
          <Bar dataKey="diffusion" fill={C.accent} barSize={16} radius={[4, 4, 0, 0]} isAnimationActive={false} />
          <Bar dataKey="ar" fill={C.blue} barSize={16} radius={[4, 4, 0, 0]} isAnimationActive={false} />
        </BarChart>
      </ResponsiveContainer>
      <details style={{ marginTop: 6 }}>
        <summary style={{ fontSize: 12, color: C.muted, cursor: "pointer" }}>Scores as a table</summary>
        <table style={{ fontSize: 12, borderCollapse: "collapse", marginTop: 8 }}>
          <thead>
            <tr>
              <th style={{ textAlign: "left", padding: "3px 10px 3px 0", color: C.muted, fontWeight: 600 }}>Benchmark</th>
              <th style={{ textAlign: "right", padding: "3px 10px", color: C.muted, fontWeight: 600 }}>Gemini Diffusion</th>
              <th style={{ textAlign: "right", padding: "3px 0 3px 10px", color: C.muted, fontWeight: 600 }}>Flash-Lite</th>
            </tr>
          </thead>
          <tbody>
            {BENCH.map((b) => (
              <tr key={b.name}>
                <td style={{ padding: "3px 10px 3px 0" }} title={b.gloss}>
                  {b.name}
                </td>
                <td style={{ textAlign: "right", padding: "3px 10px", fontVariantNumeric: "tabular-nums" }}>{b.diffusion}</td>
                <td style={{ textAlign: "right", padding: "3px 0 3px 10px", fontVariantNumeric: "tabular-nums" }}>{b.ar}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </div>
  );
}

const TIMELINE = [
  {
    year: "2021",
    name: "D3PM",
    who: "Austin et al., Google",
    link: "https://arxiv.org/abs/2107.03006",
    note: "Discrete diffusion over tokens, including the absorbing [MASK] state this page's slider shows. The recipe that everything below refines.",
  },
  {
    year: "2022",
    name: "Diffusion-LM",
    who: "Li et al., Stanford",
    link: "https://arxiv.org/abs/2205.14217",
    note: "Runs diffusion on continuous word embeddings instead of tokens, and shows the payoff is fine grained control over what gets generated.",
  },
  {
    year: "2023",
    name: "SEDD",
    who: "Lou et al., Stanford",
    link: "https://arxiv.org/abs/2310.16834",
    note: "Score entropy training makes discrete diffusion competitive: it beat GPT-2 while using far fewer network evaluations per sample.",
  },
  {
    year: "2024",
    name: "MDLM",
    who: "Sahoo et al., Cornell",
    link: "https://arxiv.org/abs/2406.07524",
    note: "Strips masked diffusion to a weighted average of masked language modeling losses, plus semi autoregressive sampling past the context window.",
  },
  {
    year: "Feb 2025",
    name: "LLaDA",
    who: "Nie et al., Renmin U. and Ant Group",
    link: "https://arxiv.org/abs/2502.09992",
    note: "An 8B diffusion LM trained from scratch that holds its own against LLaMA3 8B. The training objective of the toy model on this page.",
  },
  {
    year: "Feb 2025",
    name: "Mercury",
    who: "Inception Labs",
    link: "https://arxiv.org/abs/2506.17298",
    note: "The first commercial scale diffusion LLM, sold on exactly one number: tokens per second on code.",
  },
  {
    year: "May 2025",
    name: "Gemini Diffusion",
    who: "Google DeepMind",
    link: "https://deepmind.google/models/gemini-diffusion/",
    note: "DeepMind's experimental demo at I/O, source of the benchmark chart above.",
  },
  {
    year: "Aug 2025",
    name: "Dream 7B",
    who: "HKU and Huawei",
    link: "https://arxiv.org/abs/2508.15487",
    note: "Shows you can warm start a diffusion LM from an autoregressive one instead of training from scratch.",
  },
  {
    year: "2026",
    name: "DiffusionGemma",
    who: "Google",
    link: "https://blog.google/innovation-and-ai/technology/developers-tools/diffusion-gemma-faster-text-generation/",
    note: "Open weights, 26B mixture of experts (only a slice of the network runs per token, so capacity is cheap), generating in 256 token blocks.",
  },
];

export default function App() {
  const reduce = usePrefersReducedMotion();
  const model = useMemo(() => createModel(DIFFUSION_LM), []);

  return (
    <div style={{ background: C.bg, minHeight: "100vh", color: C.ink, fontFamily: "system-ui, -apple-system, sans-serif" }}>
      <style>{`
        @keyframes dlm-settle {
          from { opacity: 0; transform: translateY(4px) scale(0.94); }
          to { opacity: 1; transform: none; }
        }
        .dlm-press:active { transform: scale(0.97); }
        .dlm-press:focus-visible, button:focus-visible, input[type=range]:focus-visible, summary:focus-visible, a:focus-visible {
          outline: 2px solid ${C.accent};
          outline-offset: 2px;
        }
        @media (prefers-reduced-motion: reduce) {
          * { transition-duration: 1ms !important; animation-duration: 1ms !important; }
        }
      `}</style>
      <div style={{ maxWidth: 720, margin: "0 auto", padding: "28px 16px 60px" }}>
        <header style={{ marginBottom: 18 }}>
          <div style={{ fontSize: 11, fontWeight: 700, letterSpacing: 1.2, textTransform: "uppercase", color: C.accent, marginBottom: 5 }}>
            LLM Systems &middot; Text Generation
          </div>
          <h1 style={{ fontSize: 26, fontWeight: 700, margin: 0, lineHeight: 1.2, textWrap: "balance" }}>
            Diffusion Language Models: All the Words at Once
          </h1>
          <Prose style={{ color: C.muted, margin: "10px 0 0", fontSize: 13.5 }}>
            Every chatbot you have used writes like a typewriter: one token at a time (a token is the word or
            word piece an LLM reads and writes in), each waiting on the one before, because the network is only
            ever asked what comes next. Google&apos;s newest Gemma variant ignores that rule. It lays out 256
            blank slots, drafts all of them in a single forward pass (one full run of the network), and spends a
            few more passes firming up whatever it was unsure about. The same trick, at toy scale, is running
            live in this page.
          </Prose>
        </header>

        <SectionTitle>The race</SectionTitle>
        <Prose>
          Both panels below share one model, a small transformer trained on children&apos;s stories and embedded
          in this page. The left panel decodes autoregressively: predict a token, append it, run the whole
          network again, which is the loop inside GPT style models. The right panel treats the line as a canvas
          of masked slots (mask means &quot;hidden, to be filled in&quot;), fills every slot with a draft each
          pass, and locks in only the guesses it is most confident about.
        </Prose>
        <RaceDemo model={model} reduce={reduce} />

        <SectionTitle>Corrupt, then learn to reverse</SectionTitle>
        <Prose>
          Where does a model like this come from? Training never shows it clean text alone. Each example is
          corrupted first: sample a level t between 0 and 1, hide that fraction of the words behind mask tokens,
          then grade the model on restoring the originals. Do this at every level, from nearly intact to fully
          erased, and the model learns to repair any amount of damage. Drag the slider to see what its training
          data looks like.
        </Prose>
        <CorruptionDemo reduce={reduce} />

        <SectionTitle>The denoising playground</SectionTitle>
        <Prose>
          Generation is just the reverse walk: start from all mask, repair, repeat. How many repair passes to
          spend is a dial you choose at inference time, not a property baked into the weights. So is the order.
          The same model will fill slots by confidence, strictly left to right, or at random, and you can trade
          quality for passes right here. Temperature controls how the model picks each word: zero always takes
          its single most likely choice, higher values roll dice weighted by its probabilities.
        </Prose>
        <Playground model={model} reduce={reduce} />

        <SectionTitle>The trick autoregression can&apos;t do</SectionTitle>
        <Prose>
          A causal model reads with blinders on: attention, the machinery that lets each position consult the
          others, is masked so a word can only look left. A diffusion LM drops that mask. Every slot attends in
          both directions, which means you can pin a beginning and an ending and ask for a middle that agrees
          with both. That is why these models are pitched at code editing, where the surrounding file is exactly
          a pinned prefix and suffix, and it is why LLaDA beat GPT-4o at completing a poem backwards from its
          last line: nothing in the architecture privileges left to right.
        </Prose>
        <InfillDemo model={model} reduce={reduce} />

        <SectionTitle>The speed, and the fine print</SectionTitle>
        <Prose>
          The commercial pitch is a single number: tokens per second. Published figures, diffusion first, then
          autoregressive peers measured by the same third party where available:
        </Prose>
        <Card style={{ marginBottom: 14 }}>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 8 }}>
            {SPEED_DIFFUSION.map((s) => (
              <Stat key={s.label} label={s.label} value={s.value} unit="tok/s" tone={C.accent} />
            ))}
          </div>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
            {SPEED_AR.map((s) => (
              <Stat key={s.label} label={s.label} value={s.value} unit="tok/s" tone={C.blue} />
            ))}
          </div>
          <Caption>
            Sources: Inception Labs and Artificial Analysis for Mercury and the autoregressive models, DeepMind
            and Google for the Gemini and Gemma numbers. Gemini Diffusion&apos;s figure is sampling speed,
            excluding some serving overhead.
          </Caption>
        </Card>
        <Card style={{ marginBottom: 14 }}>
          <div style={{ fontSize: 13.5, fontWeight: 700, marginBottom: 8 }}>
            What the speed costs: DeepMind&apos;s own benchmark of Gemini Diffusion against its speed matched
            autoregressive sibling
          </div>
          <BenchChart />
          <Caption>
            Code generation holds up and even edges ahead on LiveCodeBench. Knowledge and reasoning lag clearly.
            Google says the same about DiffusionGemma: it trades quality for speed against Gemma 4 of similar
            size. Source: DeepMind&apos;s Gemini Diffusion model page.
          </Caption>
        </Card>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: 14 }}>
          <Card>
            <div style={{ fontSize: 13.5, fontWeight: 700, marginBottom: 6 }}>The KV cache problem</div>
            <p style={{ fontSize: 12.5, color: C.muted, lineHeight: 1.65, margin: 0 }}>
              An autoregressive model finishes tokens in order, so it can memoize the attention state of
              everything already written (the KV cache) and never touch it again. In pure diffusion nothing is
              final until the last pass, so every pass recomputes attention over the whole block. Fixes exist:
              BD3-LM makes blocks causal with diffusion only inside each block, Fast-dLLM caches approximately
              and refreshes when needed, and DiffusionGemma&apos;s 256 token blocks are the block idea shipped
              at scale.
            </p>
          </Card>
          <Card>
            <div style={{ fontSize: 13.5, fontWeight: 700, marginBottom: 6 }}>Google&apos;s own batching caveat</div>
            <p style={{ fontSize: 12.5, color: C.muted, lineHeight: 1.65, margin: 0 }}>
              The headline speedup assumes the GPU has idle capacity to spend on one user, which is true on a
              laptop or a single stream agent. A busy serving cluster already keeps the GPU saturated by
              batching many users&apos; autoregressive requests together, and there the advantage largely
              evaporates. Google states this limit itself: the win is biggest for local and single user
              inference.
            </p>
          </Card>
        </div>

        <SectionTitle>Where this came from</SectionTitle>
        <Prose>
          Google shipped the biggest version, but the idea has a decade of lineage and none of it started
          there.
        </Prose>
        <Card>
          <ol style={{ listStyle: "none", margin: 0, padding: 0 }}>
            {TIMELINE.map((e, i) => (
              <li
                key={e.name}
                style={{
                  display: "flex",
                  gap: 12,
                  padding: "10px 0",
                  borderBottom: i < TIMELINE.length - 1 ? `1px solid ${C.faint}` : "none",
                }}
              >
                <div style={{ fontFamily: MONO, fontSize: 11, color: C.accent, fontWeight: 700, width: 66, flexShrink: 0, paddingTop: 2 }}>
                  {e.year}
                </div>
                <div style={{ minWidth: 0 }}>
                  <a
                    href={e.link}
                    target="_blank"
                    rel="noreferrer"
                    style={{ fontSize: 13.5, fontWeight: 700, color: C.ink, textDecorationColor: `${C.accent}66` }}
                  >
                    {e.name}
                  </a>
                  <span style={{ fontSize: 12, color: C.muted }}> &middot; {e.who}</span>
                  <div style={{ fontSize: 12.5, color: C.muted, lineHeight: 1.6, marginTop: 2 }}>{e.note}</div>
                </div>
              </li>
            ))}
          </ol>
        </Card>

        <footer style={{ marginTop: 30, paddingTop: 16, borderTop: `1px solid ${C.border}` }}>
          <p style={{ fontSize: 12.5, color: C.muted, lineHeight: 1.7, margin: 0, maxWidth: "68ch" }}>
            Diffusion started with images; the same noise and denoise loop over pixels is taken apart in{" "}
            <Link to="/a/diffusion-denoising" style={{ color: C.accent, textDecorationColor: `${C.accent}66`, fontWeight: 600 }}>
              Diffusion: Noise and Denoise
            </Link>
            . The model running in this page is a 174k parameter word level toy with a 350 word vocabulary,
            trained on TinyStories with the LLaDA masked diffusion objective by the training script committed
            alongside this site; every forward pass and every sample above is computed live in your browser.
          </p>
        </footer>
      </div>
    </div>
  );
}
