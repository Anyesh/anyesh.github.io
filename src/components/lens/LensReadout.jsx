import { useState, useEffect, useRef, useMemo, useCallback } from "react";

// Shared reader for the Jacobian-lens artifacts. Given one baked prompt item
// (real readouts from Qwen3.5-4B, lens neuronpedia/jacobian-lens@qwen-n1000),
// it shows the prompt, a rank-trajectory chart of a few tracked concepts across
// layers (where a "thought" surfaces), and a per-layer scrubber of the raw
// top-k tokens the lens decodes at the read position.

export const LC = {
  bg: "#fafaf8",
  card: "#ffffff",
  border: "#e7e5df",
  ink: "#1a1a1a",
  muted: "#5f5a52",
  faint: "#9b958c",
  grid: "#efece6",
  op: "#7c6f9c",
  inter: "#b45309",
  answer: "#15803d",
  note: "#2563eb",
  hot: "#b91c1c",
};
export const LMONO =
  "'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, monospace";
export const LSERIF = "Georgia, 'Times New Roman', serif";
const VOCAB = 151936;

// Qwen decodes whitespace/control tokens; render them legibly.
function pretty(tok) {
  if (tok === undefined || tok === null) return "";
  if (tok === "") return "·";
  if (/^\s+$/.test(tok)) {
    if (tok.includes("\n")) return "⏎";
    return "␣";
  }
  return tok.replace(/\n/g, "⏎");
}

const norm = (s) => (s || "").trim().toLowerCase();

function strength(rank, vocab) {
  // rank 0 (the model's leading word) -> 1, deep in the tail -> 0. Log so the
  // climb from "thousands" into "top ten" is visible.
  if (rank == null) return 0;
  return Math.max(0, 1 - Math.log(rank + 1) / Math.log(vocab || VOCAB));
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

function PromptBlock({ item }) {
  const toks = item.tokens || [];
  const readIdx = toks.length + (item.read_pos ?? -1);
  return (
    <div
      style={{
        fontFamily: LMONO,
        fontSize: 13,
        lineHeight: 1.9,
        whiteSpace: "pre-wrap",
        wordBreak: "break-word",
        background: "#fbfaf7",
        border: `1px solid ${LC.border}`,
        borderRadius: 10,
        padding: "12px 14px",
        color: LC.muted,
      }}
    >
      {toks.map((t, i) => {
        const isRead = i === readIdx;
        return (
          <span
            key={i}
            style={{
              background: isRead ? "#fde68a" : "transparent",
              color: isRead ? LC.ink : LC.muted,
              borderRadius: 3,
              padding: isRead ? "1px 2px" : 0,
              fontWeight: isRead ? 600 : 400,
            }}
          >
            {t.replace(/\n/g, "↵\n")}
          </span>
        );
      })}
      <div
        style={{
          marginTop: 8,
          fontFamily: LSERIF,
          fontSize: 12,
          color: LC.faint,
          fontStyle: "italic",
        }}
      >
        reading the model's mind at the highlighted position (what it is about to
        say next)
      </div>
    </div>
  );
}

function Timeline({ item, concepts }) {
  const layers = item.layers;
  const W = 640;
  const H = 210;
  const padL = 40;
  const padR = 12;
  const padT = 14;
  const padB = 26;
  const iw = W - padL - padR;
  const ih = H - padT - padB;
  const x = (li) => padL + (iw * li) / (layers.length - 1);
  const y = (s) => padT + ih * (1 - s);

  const vocab = item.wordlike_vocab || VOCAB;
  const series = concepts.map((c) => {
    const pts = item.per_layer.map((pl, li) => {
      const rec = pl.track ? pl.track[c.word] : null;
      // rank among word-like tokens, matching the filtered list below
      const rank = rec ? (rec.rank_word ?? rec.rank) : null;
      return { li, layer: pl.layer, rank };
    });
    // ignition: first layer where the word breaks into the visible top-8 list
    const ign = pts.find((p) => p.rank != null && p.rank <= 8);
    return { ...c, pts, ign };
  });

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      style={{ width: "100%", height: "auto", display: "block" }}
      role="img"
      aria-label="rank of each concept across layers"
    >
      {[0, 0.25, 0.5, 0.75, 1].map((g) => (
        <line
          key={g}
          x1={padL}
          x2={W - padR}
          y1={y(g)}
          y2={y(g)}
          stroke={LC.grid}
          strokeWidth={1}
        />
      ))}
      <text x={4} y={y(1) + 4} fontSize={9} fill={LC.faint} fontFamily={LMONO}>
        top
      </text>
      <text x={4} y={y(0) + 4} fontSize={9} fill={LC.faint} fontFamily={LMONO}>
        buried
      </text>
      {[0, Math.floor((layers.length - 1) / 2), layers.length - 1].map((li) => (
        <text
          key={li}
          x={x(li)}
          y={H - 8}
          fontSize={9}
          fill={LC.faint}
          fontFamily={LMONO}
          textAnchor="middle"
        >
          L{layers[li]}
        </text>
      ))}
      {series.map((s) => {
        const d = s.pts
          .map((p, i) => `${i === 0 ? "M" : "L"}${x(p.li)},${y(strength(p.rank, vocab))}`)
          .join(" ");
        return (
          <g key={s.word}>
            <path d={d} fill="none" stroke={s.color} strokeWidth={2} opacity={0.9} />
            {s.ign && (
              <circle cx={x(s.ign.li)} cy={y(strength(s.ign.rank, vocab))} r={3.5} fill={s.color} />
            )}
          </g>
        );
      })}
    </svg>
  );
}

function TopKBars({ pl, concepts }) {
  const colorFor = (tok) => {
    const n = norm(tok);
    for (const c of concepts) {
      if (norm(c.word) === n || (c.match && c.match.some((m) => norm(m) === n)))
        return c.color;
    }
    return null;
  };
  const rows = pl.top.slice(0, 8);
  const max = Math.max(...rows.map((r) => r.prob), 0.001);
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      {rows.map((r, i) => {
        const col = colorFor(r.tok);
        return (
          <div key={i} style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <div
              style={{
                width: 74,
                textAlign: "right",
                fontFamily: LMONO,
                fontSize: 12,
                color: col || LC.ink,
                fontWeight: col ? 700 : 500,
                whiteSpace: "nowrap",
                overflow: "hidden",
                textOverflow: "ellipsis",
              }}
              title={r.tok}
            >
              {pretty(r.tok)}
            </div>
            <div style={{ flex: 1, height: 12, background: LC.grid, borderRadius: 3 }}>
              <div
                style={{
                  width: `${(r.prob / max) * 100}%`,
                  height: "100%",
                  background: col || "#cfcabf",
                  borderRadius: 3,
                  transition: "width 0.25s ease",
                }}
              />
            </div>
            <div
              style={{
                width: 38,
                fontFamily: LMONO,
                fontSize: 11,
                color: LC.faint,
                textAlign: "right",
              }}
            >
              {(r.prob * 100).toFixed(0)}%
            </div>
          </div>
        );
      })}
    </div>
  );
}

export default function LensReadout({ item, concepts = [], milestones = [] }) {
  const layers = item.layers;
  const reduce = usePrefersReducedMotion();
  const [li, setLi] = useState(0);
  const [playing, setPlaying] = useState(false);
  const raf = useRef(null);
  const last = useRef(0);

  const milestoneAt = useMemo(() => {
    const m = {};
    for (const ms of milestones) {
      const idx = layers.indexOf(ms.layer);
      if (idx >= 0) m[idx] = ms;
    }
    return m;
  }, [milestones, layers]);

  // caption = the most recent milestone at or before the current layer
  const activeMilestone = useMemo(() => {
    let cur = null;
    for (const ms of milestones) {
      const idx = layers.indexOf(ms.layer);
      if (idx >= 0 && idx <= li) cur = ms;
    }
    return cur;
  }, [milestones, layers, li]);

  const step = useCallback(
    (ts) => {
      if (ts - last.current > 260) {
        last.current = ts;
        setLi((v) => {
          if (v >= layers.length - 1) {
            setPlaying(false);
            return v;
          }
          return v + 1;
        });
      }
      raf.current = requestAnimationFrame(step);
    },
    [layers.length]
  );

  useEffect(() => {
    if (!playing) return;
    raf.current = requestAnimationFrame(step);
    return () => raf.current && cancelAnimationFrame(raf.current);
  }, [playing, step]);

  const play = () => {
    if (li >= layers.length - 1) setLi(0);
    setPlaying((p) => !p);
  };

  const pl = item.per_layer[li];
  const kindColor = { op: LC.op, inter: LC.inter, answer: LC.answer, note: LC.note };

  return (
    <div
      style={{
        border: `1px solid ${LC.border}`,
        borderRadius: 14,
        background: LC.card,
        padding: 16,
        boxShadow: "0 1px 3px rgba(0,0,0,0.03)",
      }}
    >
      <PromptBlock item={item} />

      <div style={{ marginTop: 16 }}>
        <div
          style={{
            fontFamily: LMONO,
            fontSize: 11,
            letterSpacing: 0.6,
            textTransform: "uppercase",
            color: LC.faint,
            marginBottom: 4,
          }}
        >
          when each word lights up, layer by layer
        </div>
        <div
          style={{
            fontFamily: LSERIF,
            fontSize: 12.5,
            color: LC.muted,
            marginBottom: 6,
            lineHeight: 1.5,
          }}
        >
          Each line is one word. Reading left to right is going deeper into the
          network. Higher up means the model ranks that word closer to the top of
          what it might say next; a dot marks where it first breaks into the top few.
        </div>
        <Timeline item={item} concepts={concepts} />
        <div
          style={{
            display: "flex",
            flexWrap: "wrap",
            gap: 12,
            marginTop: 4,
            fontFamily: LMONO,
            fontSize: 11,
          }}
        >
          {concepts.map((c) => (
            <span key={c.word} style={{ display: "inline-flex", alignItems: "center", gap: 5 }}>
              <span
                style={{
                  width: 14,
                  height: 3,
                  background: c.color,
                  display: "inline-block",
                  borderRadius: 2,
                }}
              />
              <span style={{ color: LC.muted }}>{c.label || c.word}</span>
            </span>
          ))}
        </div>
      </div>

      <div style={{ marginTop: 18 }}>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            gap: 12,
          }}
        >
          <div style={{ fontFamily: LSERIF, fontSize: 15, color: LC.ink }}>
            Layer <strong>{layers[li]}</strong>{" "}
            <span style={{ color: LC.faint, fontSize: 12 }}>of {item.n_layers - 1}</span>
          </div>
          <button
            onClick={play}
            style={{
              background: playing ? "#fff" : LC.ink,
              color: playing ? LC.ink : "#fff",
              border: `1px solid ${LC.ink}`,
              borderRadius: 8,
              padding: "6px 16px",
              fontFamily: LSERIF,
              fontSize: 13,
              cursor: "pointer",
            }}
          >
            {playing ? "pause" : li >= layers.length - 1 ? "replay ↺" : "play ▶"}
          </button>
        </div>

        <input
          type="range"
          min={0}
          max={layers.length - 1}
          value={li}
          onChange={(e) => {
            setPlaying(false);
            setLi(Number(e.target.value));
          }}
          style={{ width: "100%", margin: "10px 0 6px", accentColor: LC.ink }}
          aria-label="layer"
        />
        <div style={{ position: "relative", height: 14 }}>
          {milestones.map((ms) => {
            const idx = layers.indexOf(ms.layer);
            if (idx < 0) return null;
            const pct = (idx / (layers.length - 1)) * 100;
            return (
              <span
                key={ms.layer}
                title={ms.label}
                style={{
                  position: "absolute",
                  left: `${pct}%`,
                  transform: "translateX(-50%)",
                  width: 8,
                  height: 8,
                  borderRadius: "50%",
                  background: kindColor[ms.kind] || LC.faint,
                }}
              />
            );
          })}
        </div>

        <div
          style={{
            minHeight: 44,
            marginTop: 6,
            marginBottom: 12,
            padding: "8px 12px",
            background: activeMilestone ? "#fbfaf7" : "transparent",
            border: `1px solid ${activeMilestone ? LC.border : "transparent"}`,
            borderRadius: 8,
            fontFamily: LSERIF,
            fontSize: 14,
            lineHeight: 1.5,
            color: activeMilestone ? LC.ink : LC.faint,
          }}
        >
          {activeMilestone ? (
            <>
              <span
                style={{
                  color: kindColor[activeMilestone.kind] || LC.ink,
                  fontWeight: 700,
                }}
              >
                L{activeMilestone.layer}:{" "}
              </span>
              {activeMilestone.label}
            </>
          ) : (
            "drag the slider or press play to sweep from the first layer to the last"
          )}
        </div>

        <TopKBars pl={pl} concepts={concepts} />
        <div
          style={{
            marginTop: 8,
            fontFamily: LSERIF,
            fontSize: 12,
            color: LC.faint,
            lineHeight: 1.5,
          }}
        >
          Showing real words only. The raw top of the list is mostly punctuation and
          word fragments, filtered out here the same way the paper's own viewer does.
          Even filtered, the middle layers are messy before the answer settles.
        </div>
      </div>

      <div
        style={{
          marginTop: 16,
          paddingTop: 12,
          borderTop: `1px solid ${LC.border}`,
          display: "flex",
          alignItems: "center",
          gap: 10,
          flexWrap: "wrap",
        }}
      >
        <span style={{ fontFamily: LSERIF, fontSize: 13, color: LC.muted }}>
          what the model actually said next:
        </span>
        <span
          style={{
            fontFamily: LMONO,
            fontSize: 13,
            fontWeight: 700,
            color: LC.answer,
            background: "#eefaf0",
            border: `1px solid #cdeccf`,
            borderRadius: 6,
            padding: "2px 9px",
          }}
        >
          {pretty(item.generated || (item.model_out && item.model_out[0].tok))}
        </span>
      </div>
    </div>
  );
}
