import fs from "node:fs";
import { performance } from "node:perf_hooks";
import { DIFFUSION_LM } from "../src/artifacts/data/diffusion-lm-weights.js";
import { createModel, greedyConfidenceTrajectory } from "../src/artifacts/data/diffusion-lm-core.js";

const outPath = process.argv[2] || "/tmp/claude/diffusion-lm-parity.json";

// The gate is only honest if the artifact really runs this same core, so
// refuse to produce evidence when the jsx bypasses it.
const jsxPath = new URL("../src/artifacts/diffusion-language-models.jsx", import.meta.url);
const jsxSrc = fs.readFileSync(jsxPath, "utf8");
if (!jsxSrc.includes("data/diffusion-lm-core")) {
  console.error("FAIL: diffusion-language-models.jsx does not import data/diffusion-lm-core");
  process.exit(1);
}

const model = createModel(DIFFUSION_LM);
const { cfg, vocab, stoi } = model;

function id(word) {
  if (!stoi.has(word)) throw new Error(`prompt word not in vocab: ${word}`);
  return stoi.get(word);
}

function canvasOf(prefix, suffix = []) {
  const canvas = new Array(cfg.ctx).fill(-1);
  prefix.forEach((w, i) => {
    canvas[i] = id(w);
  });
  suffix.forEach((w, i) => {
    canvas[cfg.ctx - suffix.length + i] = id(w);
  });
  return canvas;
}

const RUNS = [
  { name: "once upon a time ,", canvas: canvasOf(["once", "upon", "a", "time", ","]), steps: 8 },
  { name: "the little girl", canvas: canvasOf(["the", "little", "girl"]), steps: 16 },
  {
    name: "one day , [infill] and they were happy .",
    canvas: canvasOf(["one", "day", ","], ["and", "they", "were", "happy", "."]),
    steps: 4,
  },
];

const runs = RUNS.map((r) => {
  const { snapshots, logitsList } = greedyConfidenceTrajectory(model, r.canvas, r.steps, true);
  const final = snapshots[snapshots.length - 1].map((t) => vocab[t]).join(" ");
  console.log(`js run "${r.name}" (${r.steps} steps): ${final}`);
  return {
    name: r.name,
    canvas: r.canvas,
    steps: r.steps,
    snapshots,
    logitsB64: logitsList.map((f32) =>
      Buffer.from(f32.buffer, f32.byteOffset, f32.byteLength).toString("base64")
    ),
  };
});

const ids = new Array(cfg.ctx).fill(1);
for (let i = 0; i < 5; i++) model.forward(ids);
const N_TIMED = 30;
const t0 = performance.now();
for (let i = 0; i < N_TIMED; i++) model.forward(ids);
const perForward = (performance.now() - t0) / N_TIMED;
console.log(`js forward pass over ${cfg.ctx} positions: ${perForward.toFixed(2)} ms (mean of ${N_TIMED})`);

fs.writeFileSync(outPath, JSON.stringify({ runs, msPerForward: perForward }));
console.log(`wrote ${outPath}`);
