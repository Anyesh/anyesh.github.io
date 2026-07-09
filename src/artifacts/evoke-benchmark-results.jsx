import { useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Legend,
  BarChart,
  Bar,
  Cell,
  ReferenceLine,
} from "recharts";

export const meta = {
  title: "EVOKE: Benchmark Results",
  category: "LLM Systems",
  description:
    "How well can an LLM answer questions about content it had to evict from its KV cache? Measured across three model architectures and three memory budgets, with EVOKE's tensor-save recovery compared to heuristic eviction baselines.",
  date: "2026-06-30",
  tags: ["kv-cache", "llm-inference", "eviction", "benchmark", "memory-hierarchy"],
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
  evoke: "#c0561f",
  baseline: "#857c72",
  infllm: "#2e6f8e",
  good: "#5a7d3c",
  goodSoft: "#eef3e6",
  danger: "#a8453f",
};

// Multifact benchmark: the agent reads a long document and then evicts
// some of it to fit within a KV budget. A probe question arrives whose
// answer lives in one of the evicted passages. Accuracy = fraction of
// probes answered correctly across 15 independent seeds.
//
// Source: mfb_qwen25_7b_n15.json (Qwen2.5-7B-Instruct-Q4_K_M, chihiro)
//         mfb_llama31_8b_n15.json (Llama-3.1-8B-Instruct, chihiro)
//         mfb_qwen35_9b.json      (Qwen3.5-9B-Q4_K_M, chihiro)
//         mfb_qwen36_35ba3b.json  (Qwen3.6-35B-A3B, chihiro)
const BUDGET_LABELS = ["512 blk\n(~33K tok)", "1024 blk\n(~66K tok)", "2048 blk\n(~131K tok)"];
const BUDGET_SWEEP = [
  { budget: "512", evoke: 50.7, infllm: 81.3, snapkv: 0.0, h2o: 0.0, recency: 0.0 },
  { budget: "1024", evoke: 57.3, infllm: 58.7, snapkv: 2.7, h2o: 1.3, recency: 5.3 },
  { budget: "2048", evoke: 62.7, infllm: 56.0, snapkv: 25.3, h2o: 25.3, recency: 22.7 },
];

// Cross-architecture at b=1024 (n=15 seeds). Best non-recovery baseline per arch:
// Qwen2.5-7B best=SnapKV 2.7%, Llama3.1-8B best=SnapKV 16.0%,
// Qwen3.5-9B best=H2O 8.0%, Qwen3.6-35B best=H2O 0.0%.
const CROSS_ARCH = [
  { arch: "Qwen2.5-7B", type: "Dense", evoke: 57.3, baseline: 2.7, baselineLabel: "SnapKV" },
  { arch: "Llama3.1-8B", type: "Dense", evoke: 57.3, baseline: 16.0, baselineLabel: "SnapKV" },
  { arch: "Qwen3.5-9B", type: "Hybrid", evoke: 68.0, baseline: 8.0, baselineLabel: "H2O" },
  { arch: "Qwen3.6-35B", type: "MoE", evoke: 52.0, baseline: 0.0, baselineLabel: "H2O" },
];

// Live agent run: opencode built a notes webapp through an EVOKE server.
// Qwen3-8B, 2048-token KV budget, RTX 4070 Ti SUPER.
// Source: agent_opencode_qwen3_8b.json
const AGENT = {
  evoke: { decoded: 9719, total: 17397, evictions: 124, splices: 59, label: "EVOKE (kv_restore)" },
  discard: { decoded: 17200, total: 17200, evictions: 63, splices: 0, label: "Evict, no recovery" },
  no_eviction: { decoded: 9910, total: 44886, evictions: 0, splices: 0, label: "No eviction (OOM risk)" },
};

const SERIES = [
  { key: "evoke", label: "EVOKE (kv_restore)", color: C.evoke, width: 2.5, dash: "" },
  { key: "infllm", label: "InfLLM (retrieval baseline)", color: C.infllm, width: 1.5, dash: "5 3" },
  { key: "snapkv", label: "SnapKV", color: "#6b6560", width: 1.5, dash: "3 3" },
  { key: "h2o", label: "H2O", color: "#9b928a", width: 1.5, dash: "3 3" },
  { key: "recency", label: "Recency (tail-only)", color: "#c0bab4", width: 1.5, dash: "3 3" },
];

// Workspace-scored eviction (J-lens probe): true probe accuracy after masked
// eviction at three retention budgets, 36 planted-fact episodes per model.
// Solid lines = Qwen2.5-7B, dashed = Qwen3-8B (replication, no hand-tuning).
// Source: eviction_eval_qwen2.5-7b-instruct.json / eviction_eval_qwen3-8b.json
// in the EVOKE repo, produced by github.com/Anyesh/j-space.
const JLENS_SWEEP = [
  { kept: "25%", workspace: 97.2, snapkv: 69.4, recency: 22.2, workspace8b: 97.2, snapkv8b: 38.9, recency8b: 22.2 },
  { kept: "50%", workspace: 100.0, snapkv: 94.4, recency: 33.3, workspace8b: 100.0, snapkv8b: 86.1, recency8b: 33.3 },
  { kept: "75%", workspace: 100.0, snapkv: 100.0, recency: 66.7, workspace8b: 100.0, snapkv8b: 97.2, recency8b: 66.7 },
];

const JLENS_SERIES = [
  { key: "workspace", label: "Workspace probe · 7B", color: C.evoke, width: 2.5, dash: "" },
  { key: "workspace8b", label: "Workspace probe · 8B", color: C.evoke, width: 1.8, dash: "5 3" },
  { key: "snapkv", label: "SnapKV · 7B", color: "#6b6560", width: 1.5, dash: "" },
  { key: "snapkv8b", label: "SnapKV · 8B", color: "#6b6560", width: 1.5, dash: "5 3" },
  { key: "recency", label: "Recency · 7B", color: "#c0bab4", width: 1.5, dash: "" },
  { key: "recency8b", label: "Recency · 8B", color: "#c0bab4", width: 1.5, dash: "5 3" },
];

function Card({ children, style }) {
  return (
    <div
      style={{
        background: C.card,
        border: `1px solid ${C.border}`,
        borderRadius: 14,
        padding: "20px 22px",
        ...style,
      }}
    >
      {children}
    </div>
  );
}

function SectionTitle({ children }) {
  return (
    <div style={{ fontSize: 14, fontWeight: 700, color: C.ink, marginBottom: 4 }}>
      {children}
    </div>
  );
}

function Caption({ children, style }) {
  return (
    <p style={{ fontSize: 12, color: C.muted, lineHeight: 1.65, margin: "8px 0 0", ...style }}>
      {children}
    </p>
  );
}

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div
      style={{
        background: C.card,
        border: `1px solid ${C.border}`,
        borderRadius: 8,
        padding: "10px 14px",
        fontSize: 12,
        boxShadow: "0 2px 8px #0001",
      }}
    >
      <div style={{ fontWeight: 700, marginBottom: 6, color: C.ink }}>Budget: {label} blocks</div>
      {payload
        .slice()
        .sort((a, b) => b.value - a.value)
        .map((p) => (
          <div key={p.dataKey} style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 3 }}>
            <div style={{ width: 10, height: 10, borderRadius: 2, background: p.color, flexShrink: 0 }} />
            <span style={{ color: C.muted }}>{p.name}:</span>
            <span style={{ fontWeight: 600, color: C.ink }}>{p.value.toFixed(1)}%</span>
          </div>
        ))}
    </div>
  );
}

function BudgetSweepChart() {
  return (
    <ResponsiveContainer width="100%" height={260}>
      <LineChart data={BUDGET_SWEEP} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
        <XAxis
          dataKey="budget"
          tick={{ fontSize: 11, fill: C.muted }}
          tickLine={false}
          axisLine={{ stroke: C.border }}
          label={{ value: "KV budget (blocks)", position: "insideBottom", offset: -2, fontSize: 11, fill: C.muted }}
        />
        <YAxis
          domain={[0, 100]}
          tick={{ fontSize: 11, fill: C.muted }}
          tickLine={false}
          axisLine={false}
          tickFormatter={(v) => `${v}%`}
        />
        <Tooltip content={<CustomTooltip />} />
        <ReferenceLine y={50} stroke={C.faint} strokeDasharray="4 2" />
        {SERIES.map((s) => (
          <Line
            key={s.key}
            dataKey={s.key}
            name={s.label}
            stroke={s.color}
            strokeWidth={s.width}
            strokeDasharray={s.dash}
            dot={{ fill: s.color, r: 3, strokeWidth: 0 }}
            activeDot={{ r: 5, strokeWidth: 0 }}
          />
        ))}
        <Legend
          iconType="line"
          wrapperStyle={{ fontSize: 11, paddingTop: 8 }}
          formatter={(val, entry) => (
            <span style={{ color: entry.color === C.evoke ? C.evoke : C.muted, fontWeight: entry.color === C.evoke ? 700 : 400 }}>
              {val}
            </span>
          )}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}


function JlensChart() {
  return (
    <ResponsiveContainer width="100%" height={280}>
      <LineChart data={JLENS_SWEEP} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
        <XAxis
          dataKey="kept"
          tick={{ fontSize: 11, fill: C.muted }}
          tickLine={false}
          axisLine={{ stroke: C.border }}
          label={{ value: "KV blocks retained", position: "insideBottom", offset: -2, fontSize: 11, fill: C.muted }}
        />
        <YAxis
          domain={[0, 100]}
          tick={{ fontSize: 11, fill: C.muted }}
          tickLine={false}
          axisLine={false}
          tickFormatter={(v) => `${v}%`}
        />
        <Tooltip content={<CustomTooltip />} />
        {JLENS_SERIES.map((s) => (
          <Line
            key={s.key}
            dataKey={s.key}
            name={s.label}
            stroke={s.color}
            strokeWidth={s.width}
            strokeDasharray={s.dash}
            dot={{ fill: s.color, r: 3, strokeWidth: 0 }}
            activeDot={{ r: 5, strokeWidth: 0 }}
          />
        ))}
        <Legend
          iconType="line"
          wrapperStyle={{ fontSize: 11, paddingTop: 8 }}
          formatter={(val, entry) => (
            <span style={{ color: entry.color === C.evoke ? C.evoke : C.muted, fontWeight: entry.color === C.evoke ? 700 : 400 }}>
              {val}
            </span>
          )}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}

function CrossArchTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  const entry = CROSS_ARCH.find((r) => r.arch === label);
  return (
    <div
      style={{
        background: C.card,
        border: `1px solid ${C.border}`,
        borderRadius: 8,
        padding: "10px 14px",
        fontSize: 12,
        boxShadow: "0 2px 8px #0001",
      }}
    >
      <div style={{ fontWeight: 700, marginBottom: 4, color: C.ink }}>
        {label} ({entry?.type})
      </div>
      {payload.map((p) => (
        <div key={p.name} style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 3 }}>
          <div style={{ width: 10, height: 10, borderRadius: 2, background: p.fill, flexShrink: 0 }} />
          <span style={{ color: C.muted }}>{p.name}:</span>
          <span style={{ fontWeight: 600, color: C.ink }}>{p.value.toFixed(1)}%</span>
        </div>
      ))}
      {entry && (
        <div style={{ color: C.muted, marginTop: 4, fontSize: 11 }}>
          Best baseline: {entry.baselineLabel}
        </div>
      )}
    </div>
  );
}

function CrossArchChart() {
  return (
    <ResponsiveContainer width="100%" height={200}>
      <BarChart data={CROSS_ARCH} margin={{ top: 8, right: 16, bottom: 8, left: 0 }} barGap={3}>
        <XAxis
          dataKey="arch"
          tick={{ fontSize: 10.5, fill: C.muted }}
          tickLine={false}
          axisLine={{ stroke: C.border }}
        />
        <YAxis
          domain={[0, 100]}
          tick={{ fontSize: 11, fill: C.muted }}
          tickLine={false}
          axisLine={false}
          tickFormatter={(v) => `${v}%`}
        />
        <Tooltip content={<CrossArchTooltip />} />
        <Bar dataKey="evoke" name="EVOKE" radius={[4, 4, 0, 0]} maxBarSize={36}>
          {CROSS_ARCH.map((d) => (
            <Cell key={d.arch} fill={C.evoke} />
          ))}
        </Bar>
        <Bar dataKey="baseline" name="Best baseline" radius={[4, 4, 0, 0]} maxBarSize={36}>
          {CROSS_ARCH.map((d) => (
            <Cell key={d.arch} fill={C.baseline} opacity={0.6} />
          ))}
        </Bar>
        <Legend
          iconType="square"
          wrapperStyle={{ fontSize: 11, paddingTop: 6 }}
          formatter={(val) => (
            <span style={{ color: val === "EVOKE" ? C.evoke : C.muted, fontWeight: val === "EVOKE" ? 700 : 400 }}>
              {val}
            </span>
          )}
        />
      </BarChart>
    </ResponsiveContainer>
  );
}

function AgentBar({ label, decoded, total, splices, color, isTarget }) {
  const pct = Math.round((decoded / AGENT.discard.decoded) * 100);
  return (
    <div style={{ marginBottom: 14 }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, marginBottom: 5 }}>
        <span style={{ fontWeight: isTarget ? 700 : 500, color: isTarget ? C.ink : C.muted }}>{label}</span>
        <span style={{ color: C.muted, fontSize: 11 }}>
          {decoded.toLocaleString()} tokens decoded
          {splices > 0 ? ` · ${splices} splices` : ""}
        </span>
      </div>
      <div style={{ height: 18, borderRadius: 5, background: C.faint, overflow: "hidden" }}>
        <div
          style={{
            height: "100%",
            width: `${pct}%`,
            background: color,
            borderRadius: 5,
            transition: "width 500ms ease",
          }}
        />
      </div>
      <div style={{ fontSize: 10.5, color: C.muted, marginTop: 3 }}>
        {pct}% of discard arm's decode cost
      </div>
    </div>
  );
}

const TABS = [
  { id: "sweep", label: "Budget sweep" },
  { id: "arch", label: "Cross-architecture" },
  { id: "agent", label: "Agent efficiency" },
  { id: "workspace", label: "Workspace signal (new)" },
];

export default function App() {
  const [tab, setTab] = useState("sweep");

  return (
    <div style={{ background: C.bg, minHeight: "100vh", color: C.ink, fontFamily: "system-ui, -apple-system, sans-serif" }}>
      <div style={{ maxWidth: 820, margin: "0 auto", padding: "28px 16px 60px" }}>
        <div style={{ marginBottom: 20 }}>
          <div
            style={{
              fontSize: 11,
              fontWeight: 700,
              letterSpacing: 1.2,
              textTransform: "uppercase",
              color: C.accent,
              marginBottom: 5,
            }}
          >
            LLM Systems · KV Cache
          </div>
          <h1 style={{ fontSize: 24, fontWeight: 700, margin: 0 }}>
            EVOKE: Benchmark Results
          </h1>
          <p style={{ color: C.muted, fontSize: 13, margin: "8px 0 0", lineHeight: 1.65, maxWidth: "62ch" }}>
            When an LLM evicts context from its KV cache to save GPU memory, can it still answer questions
            about what it evicted? EVOKE saves the raw K/V tensors to host RAM and splices them back on
            re-send. These are the measured results against heuristic baselines.
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

        {tab === "sweep" && (
          <Card>
            <SectionTitle>Fact recall vs. KV budget (Qwen2.5-7B, n=15 seeds)</SectionTitle>
            <Caption>
              The agent reads a long document that is then evicted to free KV memory. A probe question
              asks for a detail from the evicted text. Accuracy = share of probes answered correctly.
              EVOKE saves each evicted block's K/V tensors to host RAM; heuristic methods throw them away.
            </Caption>
            <div style={{ marginTop: 20 }}>
              <BudgetSweepChart />
            </div>
            <div
              style={{
                marginTop: 18,
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
                gap: 10,
              }}
            >
              <div style={{ background: C.accentSoft, borderRadius: 10, padding: "12px 14px" }}>
                <div style={{ fontSize: 18, fontWeight: 700, color: C.evoke }}>57%</div>
                <div style={{ fontSize: 11, color: C.muted, marginTop: 2 }}>EVOKE at b=1024</div>
              </div>
              <div style={{ background: C.faint, borderRadius: 10, padding: "12px 14px" }}>
                <div style={{ fontSize: 18, fontWeight: 700, color: C.baseline }}>2.7%</div>
                <div style={{ fontSize: 11, color: C.muted, marginTop: 2 }}>SnapKV at b=1024</div>
              </div>
              <div style={{ background: C.faint, borderRadius: 10, padding: "12px 14px" }}>
                <div style={{ fontSize: 18, fontWeight: 700, color: C.baseline }}>1.3%</div>
                <div style={{ fontSize: 11, color: C.muted, marginTop: 2 }}>H2O at b=1024</div>
              </div>
              <div style={{ background: C.faint, borderRadius: 10, padding: "12px 14px" }}>
                <div style={{ fontSize: 18, fontWeight: 700, color: C.infllm }}>59%</div>
                <div style={{ fontSize: 11, color: C.muted, marginTop: 2 }}>InfLLM at b=1024</div>
              </div>
            </div>
            <Caption style={{ marginTop: 12 }}>
              InfLLM also recovers well at tight budgets because it offloads text to CPU and retrieves
              by similarity. EVOKE recovers by exact identity: same bytes at the same position splice
              back with zero recompute, no retrieval model required. At b=2048, EVOKE overtakes
              InfLLM (63% vs. 56%) while heuristic methods are still catching up.
            </Caption>
          </Card>
        )}

        {tab === "arch" && (
          <Card>
            <SectionTitle>EVOKE vs. best heuristic baseline at b=1024, across architectures</SectionTitle>
            <Caption>
              Same multifact probe benchmark, n=15 seeds, 1024-block KV budget. Each "best baseline"
              bar is the highest-scoring heuristic eviction method for that architecture (SnapKV or H2O).
              The test covers dense attention (Qwen2.5, Llama3.1), hybrid Mamba/attention (Qwen3.5),
              and MoE (Qwen3.6). EVOKE's C++ primitives were added to the llama.cpp fork once; all
              architectures inherit them without model-specific tuning.
            </Caption>
            <div style={{ marginTop: 20 }}>
              <CrossArchChart />
            </div>
            <div
              style={{
                marginTop: 16,
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))",
                gap: 8,
              }}
            >
              {CROSS_ARCH.map((d) => (
                <div key={d.arch} style={{ background: C.faint, borderRadius: 10, padding: "10px 12px" }}>
                  <div style={{ fontSize: 10, fontWeight: 700, color: C.muted, textTransform: "uppercase", letterSpacing: 0.5 }}>
                    {d.arch} ({d.type})
                  </div>
                  <div style={{ marginTop: 6, display: "flex", gap: 10, alignItems: "baseline" }}>
                    <span style={{ fontSize: 17, fontWeight: 700, color: C.evoke }}>{d.evoke}%</span>
                    <span style={{ fontSize: 11, color: C.muted }}>EVOKE</span>
                  </div>
                  <div style={{ display: "flex", gap: 10, alignItems: "baseline", marginTop: 2 }}>
                    <span style={{ fontSize: 13, fontWeight: 600, color: C.baseline }}>{d.baseline}%</span>
                    <span style={{ fontSize: 11, color: C.muted }}>{d.baselineLabel}</span>
                  </div>
                </div>
              ))}
            </div>
            <Caption style={{ marginTop: 12 }}>
              Qwen3.5-9B uses hybrid SSM/attention layers; SSM layers have no KV cache, so EVOKE only
              manages the attention layers. Qwen3.6-35B-A3B is a sparse MoE model where only 3B of 35B
              parameters activate per token. The recovery primitive is architecture-agnostic: it
              operates on the raw K/V tensor layout, not on model internals.
            </Caption>
          </Card>
        )}

        {tab === "agent" && (
          <Card>
            <SectionTitle>Agent session efficiency (Qwen3-8B, RTX 4070 Ti SUPER, b=2048)</SectionTitle>
            <Caption>
              opencode built a notes webapp across two turns through an EVOKE server. The KV budget
              was 2,048 tokens. The test measures how many prompt tokens had to be re-decoded vs.
              recovered from the saved tensor archive. Lower decoded = cheaper inference; the
              "no eviction" arm never evicts but runs out of KV memory on long sessions.
            </Caption>
            <div style={{ marginTop: 24 }}>
              <AgentBar
                label="EVOKE (kv_restore)"
                decoded={AGENT.evoke.decoded}
                total={AGENT.evoke.total}
                splices={AGENT.evoke.splices}
                color={C.evoke}
                isTarget
              />
              <AgentBar
                label="Evict, no recovery"
                decoded={AGENT.discard.decoded}
                total={AGENT.discard.total}
                splices={0}
                color={C.baseline}
              />
              <AgentBar
                label="No eviction (OOM risk on long sessions)"
                decoded={AGENT.no_eviction.decoded}
                total={AGENT.no_eviction.total}
                splices={0}
                color={C.infllm}
              />
            </div>
            <div
              style={{
                marginTop: 18,
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))",
                gap: 10,
              }}
            >
              <div style={{ background: C.accentSoft, borderRadius: 10, padding: "12px 14px" }}>
                <div style={{ fontSize: 18, fontWeight: 700, color: C.evoke }}>
                  {Math.round((1 - AGENT.evoke.decoded / AGENT.discard.decoded) * 100)}%
                </div>
                <div style={{ fontSize: 11, color: C.muted, marginTop: 2 }}>
                  re-decode savings vs. discard
                </div>
              </div>
              <div style={{ background: C.goodSoft, borderRadius: 10, padding: "12px 14px" }}>
                <div style={{ fontSize: 18, fontWeight: 700, color: C.good }}>59</div>
                <div style={{ fontSize: 11, color: C.muted, marginTop: 2 }}>
                  blocks spliced back (0 recomputed)
                </div>
              </div>
              <div style={{ background: C.faint, borderRadius: 10, padding: "12px 14px" }}>
                <div style={{ fontSize: 18, fontWeight: 700, color: C.ink }}>0</div>
                <div style={{ fontSize: 11, color: C.muted, marginTop: 2 }}>
                  identity mismatches across all splices
                </div>
              </div>
              <div style={{ background: C.faint, borderRadius: 10, padding: "12px 14px" }}>
                <div style={{ fontSize: 18, fontWeight: 700, color: C.ink }}>124</div>
                <div style={{ fontSize: 11, color: C.muted, marginTop: 2 }}>
                  total evictions across the session
                </div>
              </div>
            </div>
            <Caption style={{ marginTop: 14 }}>
              The "discard" arm re-decoded every prompt token on each turn (17,200 total), because it
              had no saved tensors to splice back. EVOKE decoded 9,719 tokens (44% fewer), with the
              remaining 59 evicted blocks recovered via tensor splice at zero recompute cost. The
              "no eviction" arm decoded cheaply (intact prefix cache) but left 10,952 tokens resident
              at session end and would OOM on longer sessions or smaller GPUs.
            </Caption>
          </Card>
        )}


        {tab === "workspace" && (
          <Card>
            <SectionTitle>Choosing better eviction victims: the workspace signal (new)</SectionTitle>
            <Caption>
              Every classic eviction selector ranks blocks by attention history (H2O, SnapKV) or
              position (recency), so none can score a block before the model has attended to it.
              The workspace signal is different: a tiny ridge probe distilled from the model's
              Jacobian-lens workspace readout scores each block by <i>content</i> at prefill time,
              one dot product per position. Blocks holding workspace content, the sliver later
              reasoning routes through, tend to be exactly the blocks a future question needs.
            </Caption>
            <div style={{ marginTop: 20 }}>
              <JlensChart />
            </div>
            <div
              style={{
                marginTop: 18,
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
                gap: 10,
              }}
            >
              <div style={{ background: C.accentSoft, borderRadius: 10, padding: "12px 14px" }}>
                <div style={{ fontSize: 18, fontWeight: 700, color: C.evoke }}>0.89 vs 0.62</div>
                <div style={{ fontSize: 11, color: C.muted, marginTop: 2 }}>
                  fact-AUC vs SnapKV (Qwen2.5-7B)
                </div>
              </div>
              <div style={{ background: C.accentSoft, borderRadius: 10, padding: "12px 14px" }}>
                <div style={{ fontSize: 18, fontWeight: 700, color: C.evoke }}>0.87 / 0.94</div>
                <div style={{ fontSize: 11, color: C.muted, marginTop: 2 }}>
                  replicated on Qwen3-8B and Qwen3-4B
                </div>
              </div>
              <div style={{ background: C.goodSoft, borderRadius: 10, padding: "12px 14px" }}>
                <div style={{ fontSize: 18, fontWeight: 700, color: C.good }}>3/3</div>
                <div style={{ fontSize: 11, color: C.muted, marginTop: 2 }}>
                  agent-bench budgets passed with recovery off
                </div>
              </div>
              <div style={{ background: C.faint, borderRadius: 10, padding: "12px 14px" }}>
                <div style={{ fontSize: 18, fontWeight: 700, color: C.ink }}>&minus;0.6%</div>
                <div style={{ fontSize: 11, color: C.muted, marginTop: 2 }}>
                  measured decode overhead (noise)
                </div>
              </div>
            </div>
            <Caption style={{ marginTop: 14 }}>
              At 25% retention the workspace ranking keeps the planted fact answerable in 35-36 of 36
              episodes on all three models tested (Qwen3-4B saturates at 36/36 everywhere), while
              SnapKV drops to 25/36, 14/36, and 13/36. It composes
              with recovery rather than replacing it: the probe avoids evicting what will be needed,
              and the splice brings back whatever still had to go. Pipeline and probes:{" "}
              <a
                href="https://github.com/Anyesh/j-space"
                style={{ color: C.accent, textDecorationColor: `${C.accent}66`, fontWeight: 600 }}
              >
                github.com/Anyesh/j-space
              </a>
              .
            </Caption>
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
          <b style={{ color: C.ink }}>Setup:</b> All multifact results from Qwen2.5-7B-Instruct-Q4_K_M on
          an RTX 4070 Ti SUPER (16 GB), n=15 independent seeds, 5 planted facts per seed, KV block size 64.
          EVOKE uses a custom llama.cpp fork with C++ primitives for tensor save/restore and RoPE
          re-anchoring. Baselines run within the same harness with identical eviction schedules.{" "}
          <a
            href="/a/evoke-kv-eviction"
            style={{ color: C.accent, textDecorationColor: `${C.accent}66`, fontWeight: 600 }}
          >
            Interactive mechanism demo
          </a>
        </div>
      </div>
    </div>
  );
}
