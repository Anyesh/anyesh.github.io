import { DIFFUSION_LM } from "./diffusion-lm-weights.js";

export const PAD = 0;
export const MASK = 1;
const N_SPECIAL = 2;

function decodeTensor(t) {
  const bin = atob(t.b64);
  const out = new Float32Array(bin.length);
  for (let i = 0; i < bin.length; i++) {
    const v = bin.charCodeAt(i);
    out[i] = (v > 127 ? v - 256 : v) * t.scale;
  }
  return out;
}

export function mulberry32(seed) {
  let a = seed | 0;
  return function () {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Weights dequantize once into Float32Arrays (so every value is the same
// f32-rounded number the numpy reference sees); all arithmetic then runs in
// JS doubles, which is what keeps the parity gate's logit diff near 1e-10
// instead of accumulating f32 error.
export function createModel(blob = DIFFUSION_LM) {
  const cfg = blob.config;
  const vocab = blob.vocab;
  const stoi = new Map(vocab.map((w, i) => [w, i]));
  const w = {};
  for (const name of Object.keys(blob.tensors)) w[name] = decodeTensor(blob.tensors[name]);

  const L = cfg.n_layer;
  const H = cfg.n_head;
  const D = cfg.d_model;
  const CTX = cfg.ctx;
  const V = cfg.vocab_size;
  const HD = D / H;
  const MLPD = cfg.mlp_ratio * D;
  const ATT_SCALE = 1 / Math.sqrt(HD);
  const GELU_C = Math.sqrt(2 / Math.PI);

  const x = new Float64Array(CTX * D);
  const h = new Float64Array(CTX * D);
  const qkv = new Float64Array(CTX * 3 * D);
  const attY = new Float64Array(CTX * D);
  const mlpH = new Float64Array(CTX * MLPD);
  const scores = new Float64Array(CTX);

  function layerNorm(T, wln, bln) {
    for (let t = 0; t < T; t++) {
      const o = t * D;
      let mu = 0;
      for (let j = 0; j < D; j++) mu += x[o + j];
      mu /= D;
      let va = 0;
      for (let j = 0; j < D; j++) {
        const c = x[o + j] - mu;
        va += c * c;
      }
      va /= D;
      const inv = 1 / Math.sqrt(va + 1e-5);
      for (let j = 0; j < D; j++) h[o + j] = (x[o + j] - mu) * inv * wln[j] + bln[j];
    }
  }

  function forward(ids) {
    const T = ids.length;
    const wte = w.wte;
    const wpe = w.wpe;
    for (let t = 0; t < T; t++) {
      const eo = ids[t] * D;
      const po = t * D;
      for (let j = 0; j < D; j++) x[po + j] = wte[eo + j] + wpe[po + j];
    }
    for (let l = 0; l < L; l++) {
      layerNorm(T, w[`h${l}.ln1.w`], w[`h${l}.ln1.b`]);
      const Wqkv = w[`h${l}.attn.qkv.w`];
      const Bqkv = w[`h${l}.attn.qkv.b`];
      for (let t = 0; t < T; t++) {
        const ho = t * D;
        const qo = t * 3 * D;
        for (let o = 0; o < 3 * D; o++) {
          let s = Bqkv[o];
          const ro = o * D;
          for (let j = 0; j < D; j++) s += h[ho + j] * Wqkv[ro + j];
          qkv[qo + o] = s;
        }
      }
      for (let hd = 0; hd < H; hd++) {
        const qoff = hd * HD;
        const koff = D + hd * HD;
        const voff = 2 * D + hd * HD;
        for (let t = 0; t < T; t++) {
          const qb = t * 3 * D + qoff;
          let mx = -Infinity;
          for (let s = 0; s < T; s++) {
            const kb = s * 3 * D + koff;
            let acc = 0;
            for (let j = 0; j < HD; j++) acc += qkv[qb + j] * qkv[kb + j];
            acc *= ATT_SCALE;
            scores[s] = acc;
            if (acc > mx) mx = acc;
          }
          let z = 0;
          for (let s = 0; s < T; s++) {
            const e = Math.exp(scores[s] - mx);
            scores[s] = e;
            z += e;
          }
          const yb = t * D + hd * HD;
          for (let j = 0; j < HD; j++) attY[yb + j] = 0;
          for (let s = 0; s < T; s++) {
            const wgt = scores[s] / z;
            const vb = s * 3 * D + voff;
            for (let j = 0; j < HD; j++) attY[yb + j] += wgt * qkv[vb + j];
          }
        }
      }
      const Wp = w[`h${l}.attn.proj.w`];
      const Bp = w[`h${l}.attn.proj.b`];
      for (let t = 0; t < T; t++) {
        const yo = t * D;
        for (let o = 0; o < D; o++) {
          let s = Bp[o];
          const ro = o * D;
          for (let j = 0; j < D; j++) s += attY[yo + j] * Wp[ro + j];
          x[yo + o] += s;
        }
      }
      layerNorm(T, w[`h${l}.ln2.w`], w[`h${l}.ln2.b`]);
      const Wfc = w[`h${l}.mlp.fc.w`];
      const Bfc = w[`h${l}.mlp.fc.b`];
      for (let t = 0; t < T; t++) {
        const ho = t * D;
        const mo = t * MLPD;
        for (let o = 0; o < MLPD; o++) {
          let s = Bfc[o];
          const ro = o * D;
          for (let j = 0; j < D; j++) s += h[ho + j] * Wfc[ro + j];
          mlpH[mo + o] = 0.5 * s * (1 + Math.tanh(GELU_C * (s + 0.044715 * s * s * s)));
        }
      }
      const Wp2 = w[`h${l}.mlp.proj.w`];
      const Bp2 = w[`h${l}.mlp.proj.b`];
      for (let t = 0; t < T; t++) {
        const mo = t * MLPD;
        const xo = t * D;
        for (let o = 0; o < D; o++) {
          let s = Bp2[o];
          const ro = o * MLPD;
          for (let j = 0; j < MLPD; j++) s += mlpH[mo + j] * Wp2[ro + j];
          x[xo + o] += s;
        }
      }
    }
    layerNorm(T, w["lnf.w"], w["lnf.b"]);
    const logits = new Float64Array(T * V);
    for (let t = 0; t < T; t++) {
      const ho = t * D;
      const lo = t * V;
      for (let v = 0; v < V; v++) {
        let s = 0;
        const eo = v * D;
        for (let j = 0; j < D; j++) s += h[ho + j] * wte[eo + j];
        logits[lo + v] = s;
      }
    }
    return logits;
  }

  return { forward, cfg, vocab, stoi };
}

// Per masked position: the greedy pick and its softmax probability, specials
// masked out exactly as the Python reference does (-inf before softmax).
function greedyCandidate(logits, pos, V) {
  const off = pos * V;
  let best = N_SPECIAL;
  let bestL = -Infinity;
  for (let v = N_SPECIAL; v < V; v++) {
    const l = logits[off + v];
    if (l > bestL) {
      bestL = l;
      best = v;
    }
  }
  let z = 0;
  for (let v = N_SPECIAL; v < V; v++) z += Math.exp(logits[off + v] - bestL);
  return { pos, tok: best, prob: 1 / z };
}

function sampledCandidate(logits, pos, V, temperature, rng) {
  const off = pos * V;
  let mx = -Infinity;
  for (let v = N_SPECIAL; v < V; v++) {
    const l = logits[off + v] / temperature;
    if (l > mx) mx = l;
  }
  let z = 0;
  const e = new Float64Array(V);
  for (let v = N_SPECIAL; v < V; v++) {
    e[v] = Math.exp(logits[off + v] / temperature - mx);
    z += e[v];
  }
  let r = rng() * z;
  let tok = V - 1;
  for (let v = N_SPECIAL; v < V; v++) {
    r -= e[v];
    if (r <= 0) {
      tok = v;
      break;
    }
  }
  return { pos, tok, prob: e[tok] / z };
}

// Mirrors confidence_trajectory() in scripts/train-diffusion-lm.py operation
// for operation; the parity gate compares this function's output against the
// float64 numpy reference, so any edit here must keep the two in lockstep.
export function greedyConfidenceTrajectory(model, canvas, steps, collectLogits = false) {
  const V = model.cfg.vocab_size;
  const ids = canvas.map((c) => (c === null || c < 0 ? MASK : c));
  const masked = [];
  canvas.forEach((c, i) => {
    if (c === null || c < 0) masked.push(i);
  });
  const snapshots = [];
  const logitsList = [];
  for (let s = 0; s < steps; s++) {
    if (!masked.length) {
      snapshots.push([...ids]);
      continue;
    }
    const logits = model.forward(ids);
    if (collectLogits) logitsList.push(Float32Array.from(logits));
    const cands = masked.map((pos) => greedyCandidate(logits, pos, V));
    cands.sort((a, b) => b.prob - a.prob || a.pos - b.pos);
    const k = Math.ceil(masked.length / (steps - s));
    for (let i = 0; i < k; i++) {
      const c = cands[i];
      ids[c.pos] = c.tok;
      masked.splice(masked.indexOf(c.pos), 1);
    }
    snapshots.push([...ids]);
  }
  return { snapshots, logitsList };
}

// The UI stepper. Order is a policy over the same forward pass: "confidence"
// locks the k most confident positions per step, "ltr" locks exactly the
// leftmost masked position each forward (one token per pass, like an
// autoregressive decode), "random" locks k random masked positions.
export function createSampler(model, canvas, { steps = 8, order = "confidence", temperature = 0, seed = 1 } = {}) {
  const V = model.cfg.vocab_size;
  const rng = mulberry32(seed);
  const ids = canvas.map((c) => (c === null || c < 0 ? MASK : c));
  const masked = [];
  canvas.forEach((c, i) => {
    if (c === null || c < 0) masked.push(i);
  });
  const totalMasked = masked.length;
  let stepIndex = 0;
  let forwards = 0;

  function step() {
    if (!masked.length) return null;
    const logits = model.forward(ids);
    forwards += 1;
    const cands = masked.map((pos) =>
      temperature > 0 ? sampledCandidate(logits, pos, V, temperature, rng) : greedyCandidate(logits, pos, V)
    );
    let k;
    if (order === "ltr") {
      cands.sort((a, b) => a.pos - b.pos);
      k = 1;
    } else if (order === "random") {
      for (let i = cands.length - 1; i > 0; i--) {
        const j = Math.floor(rng() * (i + 1));
        [cands[i], cands[j]] = [cands[j], cands[i]];
      }
      k = Math.ceil(masked.length / Math.max(1, steps - stepIndex));
    } else {
      cands.sort((a, b) => b.prob - a.prob || a.pos - b.pos);
      k = Math.ceil(masked.length / Math.max(1, steps - stepIndex));
    }
    const placed = cands.slice(0, k);
    for (const c of placed) {
      ids[c.pos] = c.tok;
      masked.splice(masked.indexOf(c.pos), 1);
    }
    stepIndex += 1;
    return {
      placed,
      candidates: cands,
      ids: [...ids],
      maskedLeft: masked.length,
      forwards,
      done: masked.length === 0,
    };
  }

  return {
    step,
    get ids() {
      return [...ids];
    },
    get maskedLeft() {
      return masked.length;
    },
    get forwards() {
      return forwards;
    },
    totalMasked,
  };
}

export function tokensToWords(vocab, ids) {
  return ids.map((t) => vocab[t]);
}
