import { useEffect, useMemo, useRef, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import runData from "./data/moe-stream-jetson/run.json";

export const meta = {
  title: "Streaming a 26B MoE From SSD on a Jetson",
  category: "LLM Systems",
  description:
    "A single request on an 8 GB Jetson Orin Nano, with the model's experts coming off SSD mid-inference. Scrub the run behind the blog post's GIF and watch GPU load spike as each expert read lands, or flip to the bench numbers.",
  date: "2026-08-01",
  tags: ["moe-stream", "llama-cpp", "mixture-of-experts", "jetson-orin", "llm-inference", "edge-ai"],
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
  good: "#5a7d3c",
  goodSoft: "#eef3e6",
  splice: "#2e6f8e",
};

// Baseline vs streaming, llama-bench tg128, Gemma 4 26B-A4B QAT Q4 on an Orin
// Nano Super (8 GB, MAXN, NVMe). Source: moe-stream blog post numbers table.
const BENCH_ROWS = [
  { config: "--cpu-moe mmap baseline", tokS: 1.94 },
  { config: "Streaming, 32 slots x 8 IO threads, overlap on", tokS: 2.95, isTarget: true },
  { config: "Same, overlap off", tokS: 2.23 },
];

const SAMPLES = runData.samples;
const MAX_TOK_S = 8; // clips the token-1 division artifact (elapsed ~0s) from dominating the axis
const POWER_VALUES = SAMPLES.map((s) => s.power_mw);
const POWER_DOMAIN = [Math.floor(Math.min(...POWER_VALUES) / 100) * 100 - 100, Math.ceil(Math.max(...POWER_VALUES) / 100) * 100 + 100];

function usePrefersReducedMotion() {
  const [reduce, setReduce] = useState(false);
  useEffect(() => {
    if (!window.matchMedia) return;
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    setReduce(mq.matches);
    const onChange = (e) => setReduce(e.matches);
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, []);
  return reduce;
}

function Card({ children, style }) {
  return (
    <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 14, padding: 20, ...style }}>
      {children}
    </div>
  );
}

function Btn({ children, onClick, disabled, primary, ariaLabel }) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      aria-label={ariaLabel}
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
      }}
    >
      {children}
    </button>
  );
}

function Stat({ label, value, tone }) {
  return (
    <div style={{ flex: "1 1 90px", background: C.faint, borderRadius: 10, padding: "10px 12px", textAlign: "center" }}>
      <div style={{ fontSize: 19, fontWeight: 700, color: tone || C.ink }}>{value}</div>
      <div style={{ fontSize: 10.5, color: C.muted, marginTop: 2 }}>{label}</div>
    </div>
  );
}

function MiniTooltip({ active, payload, label, unit, valueLabel }) {
  if (!active || !payload?.length) return null;
  return (
    <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 8, padding: "8px 12px", fontSize: 12, boxShadow: "0 2px 8px #0001" }}>
      <div style={{ color: C.muted, marginBottom: 2 }}>token {label}</div>
      <div style={{ fontWeight: 700, color: C.ink }}>
        {payload[0].value}
        {unit} {valueLabel}
      </div>
    </div>
  );
}

function MiniChart({ dataKey, label, unit, valueLabel, color, domain, tickFormatter, playIdx }) {
  return (
    <div style={{ marginBottom: 16 }}>
      <div style={{ fontSize: 11, fontWeight: 700, color: C.muted, textTransform: "uppercase", letterSpacing: 0.6, marginBottom: 4 }}>
        {label}
      </div>
      <ResponsiveContainer width="100%" height={110}>
        <LineChart data={SAMPLES} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
          <XAxis dataKey="token_idx" hide />
          <YAxis
            domain={domain}
            allowDataOverflow
            tick={{ fontSize: 10, fill: C.muted }}
            tickLine={false}
            axisLine={false}
            width={44}
            tickFormatter={tickFormatter}
          />
          <Tooltip content={<MiniTooltip unit={unit} valueLabel={valueLabel} />} />
          <Line dataKey={dataKey} stroke={color} strokeWidth={1.8} dot={false} isAnimationActive={false} />
          <ReferenceLine x={playIdx} stroke={C.accent} strokeWidth={1.5} strokeDasharray="3 2" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

const TABS = [
  { id: "replay", label: "Replay" },
  { id: "bench", label: "Bench numbers" },
];

export default function App() {
  const reduce = usePrefersReducedMotion();
  const [tab, setTab] = useState("replay");
  const [stepIdx, setStepIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const timerRef = useRef(null);

  const sample = SAMPLES[stepIdx];
  const atEnd = stepIdx >= SAMPLES.length - 1;
  const maxBenchTokS = Math.max(...BENCH_ROWS.map((r) => r.tokS));

  // Advances through the real per-sample timestamps captured during the run
  // rather than a fixed tick, so playback reproduces the actual decode
  // cadence instead of a simulated pace.
  useEffect(() => {
    if (!playing || atEnd) {
      if (atEnd) setPlaying(false);
      return;
    }
    const cur = SAMPLES[stepIdx];
    const next = SAMPLES[stepIdx + 1];
    const realDelayMs = (next.t - cur.t) * 1000;
    const delayMs = reduce ? Math.max(realDelayMs, 400) : Math.min(2000, Math.max(30, realDelayMs));
    timerRef.current = setTimeout(() => setStepIdx((i) => i + 1), delayMs);
    return () => clearTimeout(timerRef.current);
  }, [playing, stepIdx, atEnd, reduce]);

  return (
    <div style={{ background: C.bg, minHeight: "100vh", color: C.ink, fontFamily: "system-ui, -apple-system, sans-serif" }}>
      <div style={{ maxWidth: 820, margin: "0 auto", padding: "28px 16px 60px" }}>
        <div style={{ marginBottom: 20 }}>
          <div style={{ fontSize: 11, fontWeight: 700, letterSpacing: 1.2, textTransform: "uppercase", color: C.accent, marginBottom: 5 }}>
            LLM Systems · MoE Streaming
          </div>
          <h1 style={{ fontSize: 24, fontWeight: 700, margin: 0 }}>
            Streaming a 26B MoE From SSD on a Jetson
          </h1>
          <p style={{ color: C.muted, fontSize: 13, margin: "8px 0 0", lineHeight: 1.65, maxWidth: "62ch" }}>
            This is the exact run behind the GIF in the blog post: one streamed completion against
            the production <code>gemma4-26b</code> roster entry on a Jetson Orin Nano, with{" "}
            <code>tegrastats</code> sampling GPU load and power draw alongside every token as it
            arrived. Step or play through it below.
          </p>
        </div>

        <div style={{ display: "flex", gap: 6, marginBottom: 18 }}>
          {TABS.map((t) => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              style={{
                padding: "7px 14px",
                borderRadius: 999,
                border: `1.5px solid ${tab === t.id ? C.accent : C.border}`,
                background: tab === t.id ? C.accentSoft : "transparent",
                color: tab === t.id ? C.accent : C.muted,
                fontWeight: tab === t.id ? 700 : 500,
                fontSize: 12.5,
                cursor: "pointer",
                fontFamily: "inherit",
              }}
            >
              {t.label}
            </button>
          ))}
        </div>

        {tab === "replay" && (
          <>
            <Card style={{ marginBottom: 16 }}>
              <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginBottom: 16 }}>
                <Stat label="elapsed" value={`${sample.t.toFixed(1)}s`} />
                <Stat label="token" value={`${sample.token_idx}/${SAMPLES.length}`} />
                <Stat label="cumulative tok/s" value={Math.min(sample.cumulative_tok_s, MAX_TOK_S).toFixed(2)} tone={C.accent} />
                <Stat label="GPU load" value={`${sample.gpu_pct}%`} tone={C.splice} />
                <Stat label="power draw" value={`${(sample.power_mw / 1000).toFixed(2)} W`} tone={C.good} />
              </div>

              <MiniChart
                dataKey="gpu_pct"
                label="GPU load"
                unit="%"
                valueLabel="GR3D_FREQ"
                color={C.splice}
                domain={[0, 100]}
                tickFormatter={(v) => `${v}%`}
                playIdx={sample.token_idx}
              />
              <MiniChart
                dataKey="power_mw"
                label="Power draw (VDD_IN)"
                unit="mW"
                valueLabel=""
                color={C.good}
                domain={POWER_DOMAIN}
                tickFormatter={(v) => `${(v / 1000).toFixed(1)}W`}
                playIdx={sample.token_idx}
              />
              <MiniChart
                dataKey="cumulative_tok_s"
                label="Cumulative tok/s"
                unit=""
                valueLabel="tok/s"
                color={C.accent}
                domain={[0, MAX_TOK_S]}
                tickFormatter={(v) => v.toFixed(0)}
                playIdx={sample.token_idx}
              />

              <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginTop: 4 }}>
                <Btn primary onClick={() => setStepIdx((i) => Math.min(i + 1, SAMPLES.length - 1))} disabled={atEnd} ariaLabel="Advance one token">
                  Step
                </Btn>
                <Btn onClick={() => setPlaying((p) => !p)} disabled={atEnd} ariaLabel={playing ? "Pause playback" : "Play at the recorded pace"}>
                  {playing ? "Pause" : "Play"}
                </Btn>
                <Btn onClick={() => setStepIdx((i) => Math.max(i - 1, 0))} disabled={stepIdx === 0} ariaLabel="Go back one token">
                  Back
                </Btn>
                <Btn
                  onClick={() => {
                    setStepIdx(0);
                    setPlaying(false);
                  }}
                  ariaLabel="Reset to the first token"
                >
                  Reset
                </Btn>
                <div style={{ marginLeft: "auto", alignSelf: "center", fontSize: 12, color: C.muted }}>
                  token <b style={{ color: C.ink }}>{stepIdx + 1}</b> / {SAMPLES.length}
                </div>
              </div>
            </Card>

            <Card>
              <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 8 }}>Ground truth, from llama.cpp itself</div>
              <p style={{ fontSize: 12, color: C.muted, lineHeight: 1.7, margin: 0 }}>
                The tok/s trace above is derived from token arrival times measured on this side of
                the HTTP connection. llama-server's own final <code>timings</code> block is the
                cross-check: <b>{runData.final_timings.predicted_n}</b> tokens predicted at{" "}
                <b>{runData.final_timings.predicted_per_second.toFixed(2)} tok/s</b>, prompt eval at{" "}
                <b>{runData.final_timings.prompt_per_second.toFixed(2)} tok/s</b> over{" "}
                {runData.final_timings.prompt_n} prompt tokens. That is the same run, measured from
                inside the engine instead of from the client.
              </p>
            </Card>
          </>
        )}

        {tab === "bench" && (
          <Card>
            <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 4 }}>
              Baseline vs. streaming, llama-bench tg128
            </div>
            <p style={{ fontSize: 12, color: C.muted, lineHeight: 1.6, margin: "0 0 16px" }}>
              Gemma 4 26B-A4B QAT Q4 on a Jetson Orin Nano Super (8 GB, MAXN, NVMe). The baseline is
              the best you could previously do in llama.cpp: dense weights on the GPU, experts
              mmap'd on the CPU with <code>--cpu-moe</code>. This is the steady bench number, not the
              single live request in the Replay tab, which is why they differ.
            </p>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13, marginBottom: 16 }}>
              <thead>
                <tr>
                  <th style={{ textAlign: "left", padding: "6px 8px", borderBottom: `1px solid ${C.border}`, color: C.muted, fontSize: 11, textTransform: "uppercase", letterSpacing: 0.5 }}>
                    Configuration
                  </th>
                  <th style={{ textAlign: "right", padding: "6px 8px", borderBottom: `1px solid ${C.border}`, color: C.muted, fontSize: 11, textTransform: "uppercase", letterSpacing: 0.5 }}>
                    Decode tok/s
                  </th>
                </tr>
              </thead>
              <tbody>
                {BENCH_ROWS.map((r) => (
                  <tr key={r.config}>
                    <td style={{ padding: "8px", borderBottom: `1px solid ${C.faint}`, fontWeight: r.isTarget ? 700 : 500, color: r.isTarget ? C.accent : C.ink }}>
                      {r.config}
                    </td>
                    <td style={{ padding: "8px", borderBottom: `1px solid ${C.faint}`, textAlign: "right", fontWeight: 700, color: r.isTarget ? C.accent : C.ink }}>
                      <span
                        style={{
                          display: "inline-block",
                          width: `${(r.tokS / maxBenchTokS) * 70}px`,
                          height: 8,
                          background: r.isTarget ? C.accent : C.muted,
                          opacity: 0.7,
                          borderRadius: 3,
                          marginRight: 8,
                          verticalAlign: "middle",
                        }}
                      />
                      {r.tokS.toFixed(2)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: 10 }}>
              <div style={{ background: C.accentSoft, borderRadius: 10, padding: "12px 14px" }}>
                <div style={{ fontSize: 18, fontWeight: 700, color: C.accent }}>+32%</div>
                <div style={{ fontSize: 11, color: C.muted, marginTop: 2 }}>from the IO/compute overlap alone</div>
              </div>
              <div style={{ background: C.goodSoft, borderRadius: 10, padding: "12px 14px" }}>
                <div style={{ fontSize: 18, fontWeight: 700, color: C.good }}>~5 tok/s</div>
                <div style={{ fontSize: 11, color: C.muted, marginTop: 2 }}>projected IO-bound ceiling (fio, 1.27 GB/s)</div>
              </div>
              <div style={{ background: C.faint, borderRadius: 10, padding: "12px 14px" }}>
                <div style={{ fontSize: 18, fontWeight: 700, color: C.ink }}>6.7 GB</div>
                <div style={{ fontSize: 11, color: C.muted, marginTop: 2 }}>peak RSS, 32 slots x 8 layers</div>
              </div>
              <div style={{ background: C.faint, borderRadius: 10, padding: "12px 14px" }}>
                <div style={{ fontSize: 18, fontWeight: 700, color: C.ink }}>bitwise</div>
                <div style={{ fontSize: 11, color: C.muted, marginTop: 2 }}>identical logits, streaming vs. baseline</div>
              </div>
            </div>
          </Card>
        )}

        <div
          style={{
            marginTop: 20,
            padding: "14px 18px",
            borderRadius: 12,
            border: `1px solid ${C.border}`,
            background: C.faint,
            fontSize: 12,
            color: C.muted,
            lineHeight: 1.6,
          }}
        >
          <b style={{ color: C.ink }}>Where this comes from:</b> the architecture, the llama.cpp
          patch, and the full write-up are on the blog:{" "}
          <a
            href="https://www.anyesh.me/writing/moe-stream-jetson-orin"
            style={{ color: C.accent, textDecorationColor: `${C.accent}66`, fontWeight: 600 }}
          >
            Running a 26B MoE on an 8 GB Jetson by streaming experts from SSD
          </a>
          .
        </div>
      </div>
    </div>
  );
}
