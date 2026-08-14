const BIGRAM_MIN_COUNT = 5;
const SMOOTH_ALPHA = 0.5;

export function mulberry32(seed) {
  let a = seed | 0;
  return function () {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// MurmurHash3 fmix32, combining the previous word's vocab index with the
// secret key so a different key produces an unrelated green/red split at
// every context without either party needing to share state beyond the key.
export function hashSeed(prevWordIdx, key) {
  let h = (prevWordIdx | 0) ^ Math.imul(key | 0, 0x9e3779b1);
  h = Math.imul(h ^ (h >>> 16), 0x85ebca6b);
  h = Math.imul(h ^ (h >>> 13), 0xc2b2ae35);
  h ^= h >>> 16;
  return h >>> 0;
}

export function computeGreenList(prevWordIdx, key, vocabSize, gamma) {
  const rng = mulberry32(hashSeed(prevWordIdx, key));
  const idx = new Array(vocabSize);
  for (let i = 0; i < vocabSize; i++) idx[i] = i;
  for (let i = vocabSize - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    const tmp = idx[i];
    idx[i] = idx[j];
    idx[j] = tmp;
  }
  const greenCount = Math.round(gamma * vocabSize);
  return new Set(idx.slice(0, greenCount));
}

export function getNextWordDistribution(prevWordIdx, model) {
  const { bigram, unigram } = model;
  const entries = bigram[String(prevWordIdx)];
  const bigramTotal = entries ? entries.reduce((s, [, c]) => s + c, 0) : 0;

  if (entries && bigramTotal >= BIGRAM_MIN_COUNT) {
    const denom = bigramTotal + SMOOTH_ALPHA * entries.length;
    return {
      indices: entries.map(([idx]) => idx),
      probs: entries.map(([, c]) => (c + SMOOTH_ALPHA) / denom),
    };
  }

  const unigramTotal = unigram.reduce((s, c) => s + c, 0);
  const denom = unigramTotal + SMOOTH_ALPHA * unigram.length;
  return {
    indices: unigram.map((_, i) => i),
    probs: unigram.map((c) => (c + SMOOTH_ALPHA) / denom),
  };
}

export function sampleNextWord(
  prevWordIdx,
  model,
  { watermark = false, key = 0, gamma = 0.25, delta = 4, temperature = 1, rng }
) {
  const { indices, probs } = getNextWordDistribution(prevWordIdx, model);
  const vocabSize = model.vocab.length;
  // Computed unconditionally: the split is a deterministic function of
  // (context, key) whether or not the caller chooses to bias toward it, so
  // the UI can show which candidates were green even on an unwatermarked run.
  const greenList = computeGreenList(prevWordIdx, key, vocabSize, gamma);

  const logits = probs.map((p) => Math.log(p));
  const biased = logits.map((l, i) => (watermark && greenList.has(indices[i]) ? l + delta : l));
  const t = Math.max(temperature, 1e-6);
  const scaled = biased.map((l) => l / t);
  const maxL = Math.max(...scaled);
  const exps = scaled.map((l) => Math.exp(l - maxL));
  const z = exps.reduce((a, b) => a + b, 0);
  const postProbs = exps.map((e) => e / z);

  let r = rng();
  let chosen = postProbs.length - 1;
  let acc = 0;
  for (let i = 0; i < postProbs.length; i++) {
    acc += postProbs[i];
    if (r <= acc) {
      chosen = i;
      break;
    }
  }

  const wordIdx = indices[chosen];
  const wasGreen = greenList.has(wordIdx);

  const order = indices.map((_, i) => i).sort((a, b) => probs[b] - probs[a]);
  const top = order.slice(0, 10);
  const distributionForDisplay = top.map((i) => ({
    wordIdx: indices[i],
    word: model.vocab[indices[i]],
    preProb: probs[i],
    postProb: postProbs[i],
    isGreen: greenList.has(indices[i]),
  }));

  return { wordIdx, wasGreen, distributionForDisplay };
}

export function detectWatermark(wordIdxSequence, key, vocabSize, gamma) {
  let greenCount = 0;
  let T = 0;
  for (let i = 1; i < wordIdxSequence.length; i++) {
    const greenList = computeGreenList(wordIdxSequence[i - 1], key, vocabSize, gamma);
    if (greenList.has(wordIdxSequence[i])) greenCount++;
    T++;
  }
  const z = (greenCount - gamma * T) / Math.sqrt(gamma * (1 - gamma) * T);
  return { T, greenCount, z };
}
