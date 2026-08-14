import fs from "node:fs";
import { fileURLToPath } from "node:url";
import { mulberry32, sampleNextWord, detectWatermark } from "../src/artifacts/data/watermark-lm-core.js";

const dataPath = fileURLToPath(new URL("../src/artifacts/data/watermark-lm.json", import.meta.url));
const model = JSON.parse(fs.readFileSync(dataPath, "utf8"));
const vocabSize = model.vocab.length;
const stoi = new Map(model.vocab.map((w, i) => [w, i]));

const KEY = 20260814;
const GAMMA = 0.25;
const DELTA = 4;
const TEMPERATURE = 1;
const N_WORDS = 200;
const SEED_WORD = "once";

function generate(watermark, seed) {
  const rng = mulberry32(seed);
  const sequence = [stoi.get(SEED_WORD)];
  for (let i = 0; i < N_WORDS; i++) {
    const prev = sequence[sequence.length - 1];
    const { wordIdx } = sampleNextWord(prev, model, {
      watermark,
      key: KEY,
      gamma: GAMMA,
      delta: DELTA,
      temperature: TEMPERATURE,
      rng,
    });
    sequence.push(wordIdx);
  }
  return sequence;
}

const watermarked = generate(true, 1337);
const plain = generate(false, 1337);

const wmResult = detectWatermark(watermarked, KEY, vocabSize, GAMMA);
const plainResult = detectWatermark(plain, KEY, vocabSize, GAMMA);

console.log(`watermarked text (${watermarked.length} words):`);
console.log(watermarked.map((i) => model.vocab[i]).join(" "));
console.log(`\nunwatermarked text (${plain.length} words):`);
console.log(plain.map((i) => model.vocab[i]).join(" "));

console.log(`\nwatermarked: T=${wmResult.T} green=${wmResult.greenCount} z=${wmResult.z.toFixed(3)}`);
console.log(`unwatermarked: T=${plainResult.T} green=${plainResult.greenCount} z=${plainResult.z.toFixed(3)}`);

let ok = true;
if (!(wmResult.z > 4)) {
  console.error(`FAIL: watermarked z-score ${wmResult.z.toFixed(3)} is not > 4`);
  ok = false;
}
if (!(Math.abs(plainResult.z) < 2)) {
  console.error(`FAIL: unwatermarked |z-score| ${Math.abs(plainResult.z).toFixed(3)} is not < 2`);
  ok = false;
}

if (!ok) process.exit(1);
console.log("\nPASS");
