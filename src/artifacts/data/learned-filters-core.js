// Tensors are flat Float32Arrays in NHWC order for a single sample:
// index = ((y * W) + x) * C + c, so the innermost convolution loop walks
// contiguous memory.

export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussian(rand) {
  let u = 0;
  let v = 0;
  while (u === 0) u = rand();
  while (v === 0) v = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

class Conv {
  constructor(inH, inW, inC, outC, k, stride, rand) {
    this.inH = inH;
    this.inW = inW;
    this.inC = inC;
    this.outC = outC;
    this.k = k;
    this.stride = stride;
    this.outH = Math.floor((inH - k) / stride) + 1;
    this.outW = Math.floor((inW - k) / stride) + 1;
    this.filterSize = k * k * inC;
    this.w = new Float32Array(outC * this.filterSize);
    const scale = Math.sqrt(2 / this.filterSize);
    for (let i = 0; i < this.w.length; i++) this.w[i] = gaussian(rand) * scale;
    this.b = new Float32Array(outC);
    this.gw = new Float32Array(this.w.length);
    this.gb = new Float32Array(outC);
  }

  forward(x, out) {
    const { inW, inC, outC, outH, outW, k, stride, w, b, filterSize } = this;
    const span = k * inC;
    for (let oy = 0; oy < outH; oy++) {
      for (let ox = 0; ox < outW; ox++) {
        const outBase = (oy * outW + ox) * outC;
        const iy0 = oy * stride;
        const ix0 = ox * stride;
        for (let oc = 0; oc < outC; oc++) {
          let sum = b[oc];
          const wBase = oc * filterSize;
          for (let ky = 0; ky < k; ky++) {
            const rowIn = ((iy0 + ky) * inW + ix0) * inC;
            const rowW = wBase + ky * span;
            for (let t = 0; t < span; t++) sum += x[rowIn + t] * w[rowW + t];
          }
          out[outBase + oc] = sum;
        }
      }
    }
  }

  backward(x, gradOut, gradIn) {
    const { inW, inC, outC, outH, outW, k, stride, w, gw, gb, filterSize } = this;
    const span = k * inC;
    if (gradIn) gradIn.fill(0);
    for (let oy = 0; oy < outH; oy++) {
      for (let ox = 0; ox < outW; ox++) {
        const outBase = (oy * outW + ox) * outC;
        const iy0 = oy * stride;
        const ix0 = ox * stride;
        for (let oc = 0; oc < outC; oc++) {
          const g = gradOut[outBase + oc];
          if (g === 0) continue;
          gb[oc] += g;
          const wBase = oc * filterSize;
          for (let ky = 0; ky < k; ky++) {
            const rowIn = ((iy0 + ky) * inW + ix0) * inC;
            const rowW = wBase + ky * span;
            for (let t = 0; t < span; t++) {
              gw[rowW + t] += g * x[rowIn + t];
              if (gradIn) gradIn[rowIn + t] += g * w[rowW + t];
            }
          }
        }
      }
    }
  }
}

class Dense {
  constructor(inN, outN, rand) {
    this.inN = inN;
    this.outN = outN;
    this.w = new Float32Array(inN * outN);
    const scale = Math.sqrt(2 / inN);
    for (let i = 0; i < this.w.length; i++) this.w[i] = gaussian(rand) * scale;
    this.b = new Float32Array(outN);
    this.gw = new Float32Array(this.w.length);
    this.gb = new Float32Array(outN);
  }

  forward(x, out) {
    const { inN, outN, w, b } = this;
    for (let o = 0; o < outN; o++) {
      let sum = b[o];
      const base = o * inN;
      for (let i = 0; i < inN; i++) sum += x[i] * w[base + i];
      out[o] = sum;
    }
  }

  backward(x, gradOut, gradIn) {
    const { inN, outN, w, gw, gb } = this;
    gradIn.fill(0);
    for (let o = 0; o < outN; o++) {
      const g = gradOut[o];
      gb[o] += g;
      const base = o * inN;
      for (let i = 0; i < inN; i++) {
        gw[base + i] += g * x[i];
        gradIn[i] += g * w[base + i];
      }
    }
  }
}

function maxPoolForward(x, H, W, C, pool, out, argmax) {
  const outH = Math.floor(H / pool);
  const outW = Math.floor(W / pool);
  for (let oy = 0; oy < outH; oy++) {
    for (let ox = 0; ox < outW; ox++) {
      const outBase = (oy * outW + ox) * C;
      for (let c = 0; c < C; c++) {
        let best = -Infinity;
        let bestIdx = -1;
        for (let py = 0; py < pool; py++) {
          for (let px = 0; px < pool; px++) {
            const idx = ((oy * pool + py) * W + ox * pool + px) * C + c;
            if (x[idx] > best) {
              best = x[idx];
              bestIdx = idx;
            }
          }
        }
        out[outBase + c] = best;
        argmax[outBase + c] = bestIdx;
      }
    }
  }
}

function softmaxCrossEntropy(logits, label, grad) {
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) if (logits[i] > max) max = logits[i];
  let sum = 0;
  for (let i = 0; i < logits.length; i++) sum += Math.exp(logits[i] - max);
  const logSum = Math.log(sum) + max;
  for (let i = 0; i < logits.length; i++) grad[i] = Math.exp(logits[i] - logSum);
  grad[label] -= 1;
  return logSum - logits[label];
}

export const DEFAULT_CFG = {
  seed: 5,
  size: 32,
  channels: 1,
  f1: 16,
  k1: 7,
  s1: 2,
  f2: 24,
  k2: 3,
  classes: 10,
  lr: 0.004,
  wd: 0.004,
  batch: 16,
  // A first layer started well below He scale lets the structure gradients find
  // outgrow the random draw within the few seconds a browser tab can spare. The
  // same oriented filters appear from a standard init, several times slower.
  initScale: 0.15,
};

export function createNet(cfg = DEFAULT_CFG) {
  const c = { ...DEFAULT_CFG, ...cfg };
  const rand = mulberry32(c.seed);
  const conv1 = new Conv(c.size, c.size, c.channels, c.f1, c.k1, c.s1, rand);
  if (c.initScale !== 1) for (let i = 0; i < conv1.w.length; i++) conv1.w[i] *= c.initScale;
  const p1H = Math.floor(conv1.outH / 2);
  const p1W = Math.floor(conv1.outW / 2);
  const conv2 = new Conv(p1H, p1W, c.f1, c.f2, c.k2, 1, rand);
  const p2H = Math.floor(conv2.outH / 2);
  const p2W = Math.floor(conv2.outW / 2);
  const flat = p2H * p2W * c.f2;
  const head = new Dense(flat, c.classes, rand);

  const size1 = conv1.outH * conv1.outW * c.f1;
  const pool1 = p1H * p1W * c.f1;
  const size2 = conv2.outH * conv2.outW * c.f2;

  const buf = {
    z1: new Float32Array(size1),
    a1: new Float32Array(size1),
    p1: new Float32Array(pool1),
    p1arg: new Int32Array(pool1),
    z2: new Float32Array(size2),
    a2: new Float32Array(size2),
    p2: new Float32Array(flat),
    p2arg: new Int32Array(flat),
    logits: new Float32Array(c.classes),
    gLogits: new Float32Array(c.classes),
    gp2: new Float32Array(flat),
    ga2: new Float32Array(size2),
    gz2: new Float32Array(size2),
    gp1: new Float32Array(pool1),
    ga1: new Float32Array(size1),
    gz1: new Float32Array(size1),
  };

  const params = [
    { value: conv1.w, grad: conv1.gw },
    { value: conv1.b, grad: conv1.gb },
    { value: conv2.w, grad: conv2.gw },
    { value: conv2.b, grad: conv2.gb },
    { value: head.w, grad: head.gw },
    { value: head.b, grad: head.gb },
  ].map((p) => ({ ...p, m: new Float32Array(p.value.length), v: new Float32Array(p.value.length) }));

  const paramCount = params.reduce((sum, p) => sum + p.value.length, 0);

  return {
    cfg: c,
    conv1,
    conv2,
    head,
    buf,
    params,
    paramCount,
    flat,
    step: 0,
    samplesSeen: 0,
    initialFilters: conv1.w.slice(),
  };
}

export function forward(net, x) {
  const { conv1, conv2, head, buf, cfg } = net;
  conv1.forward(x, buf.z1);
  for (let i = 0; i < buf.z1.length; i++) buf.a1[i] = buf.z1[i] > 0 ? buf.z1[i] : 0;
  maxPoolForward(buf.a1, conv1.outH, conv1.outW, cfg.f1, 2, buf.p1, buf.p1arg);
  conv2.forward(buf.p1, buf.z2);
  for (let i = 0; i < buf.z2.length; i++) buf.a2[i] = buf.z2[i] > 0 ? buf.z2[i] : 0;
  maxPoolForward(buf.a2, conv2.outH, conv2.outW, cfg.f2, 2, buf.p2, buf.p2arg);
  head.forward(buf.p2, buf.logits);
  return buf.logits;
}

function backward(net, x, label) {
  const { conv1, conv2, head, buf } = net;
  const loss = softmaxCrossEntropy(buf.logits, label, buf.gLogits);
  head.backward(buf.p2, buf.gLogits, buf.gp2);
  buf.ga2.fill(0);
  for (let i = 0; i < buf.gp2.length; i++) buf.ga2[buf.p2arg[i]] += buf.gp2[i];
  for (let i = 0; i < buf.z2.length; i++) buf.gz2[i] = buf.z2[i] > 0 ? buf.ga2[i] : 0;
  conv2.backward(buf.p1, buf.gz2, buf.gp1);
  buf.ga1.fill(0);
  for (let i = 0; i < buf.gp1.length; i++) buf.ga1[buf.p1arg[i]] += buf.gp1[i];
  for (let i = 0; i < buf.z1.length; i++) buf.gz1[i] = buf.z1[i] > 0 ? buf.ga1[i] : 0;
  conv1.backward(x, buf.gz1, null);
  return loss;
}

function applyAdam(net) {
  const { cfg, params } = net;
  net.step += 1;
  const b1 = 0.9;
  const b2 = 0.999;
  const c1 = 1 - Math.pow(b1, net.step);
  const c2 = 1 - Math.pow(b2, net.step);
  for (const p of params) {
    const { value, grad, m, v } = p;
    for (let j = 0; j < value.length; j++) {
      const g = grad[j] + cfg.wd * value[j];
      m[j] = b1 * m[j] + (1 - b1) * g;
      v[j] = b2 * v[j] + (1 - b2) * g * g;
      value[j] -= (cfg.lr * (m[j] / c1)) / (Math.sqrt(v[j] / c2) + 1e-8);
      grad[j] = 0;
    }
  }
}

// Walks `order` from `cursor`, reshuffling with `rand` each time it wraps, so a
// pause and resume never replays the same stretch of the training set.
export function trainChunk(net, data, order, cursor, count, rand) {
  let pos = cursor;
  let lossSum = 0;
  for (let i = 0; i < count; i++) {
    if (pos >= order.length) {
      for (let j = order.length - 1; j > 0; j--) {
        const k = Math.floor(rand() * (j + 1));
        const tmp = order[j];
        order[j] = order[k];
        order[k] = tmp;
      }
      pos = 0;
    }
    const idx = order[pos++];
    const x = data.sample(idx);
    forward(net, x);
    lossSum += backward(net, x, data.labels[idx]);
    net.samplesSeen += 1;
    if (net.samplesSeen % net.cfg.batch === 0) applyAdam(net);
  }
  return { cursor: pos, loss: lossSum / count };
}

export function evaluate(net, data, from, to) {
  const confusion = new Int32Array(net.cfg.classes * net.cfg.classes);
  let correct = 0;
  let lossSum = 0;
  for (let i = from; i < to; i++) {
    const logits = forward(net, data.sample(i));
    let best = 0;
    let max = logits[0];
    for (let c = 1; c < logits.length; c++) {
      if (logits[c] > max) {
        max = logits[c];
        best = c;
      }
    }
    const label = data.labels[i];
    confusion[label * net.cfg.classes + best] += 1;
    if (best === label) correct += 1;
    let sum = 0;
    for (let c = 0; c < logits.length; c++) sum += Math.exp(logits[c] - max);
    lossSum += Math.log(sum) + max - logits[label];
  }
  const n = to - from;
  return { acc: correct / n, loss: lossSum / n, confusion };
}

// Tiles stay as bytes (a few megabytes instead of tens) and are standardised
// one at a time into a shared scratch buffer: subtract the tile's own mean,
// divide by its own standard deviation, so overall brightness carries no signal
// and the layer has to work on structure.
export function createSampleSource(bytes, labels, per) {
  const scratch = new Float32Array(per);
  return {
    labels,
    count: labels.length,
    sample(index) {
      const base = index * per;
      let mean = 0;
      for (let j = 0; j < per; j++) mean += bytes[base + j];
      mean /= per;
      let varsum = 0;
      for (let j = 0; j < per; j++) {
        const d = bytes[base + j] - mean;
        varsum += d * d;
      }
      const inv = 1 / (Math.sqrt(varsum / per) + 1e-3);
      for (let j = 0; j < per; j++) scratch[j] = (bytes[base + j] - mean) * inv;
      return scratch;
    },
  };
}

export function filterAt(net, index) {
  const size = net.conv1.filterSize;
  return net.conv1.w.subarray(index * size, (index + 1) * size);
}

// Slides one k x k x C kernel over a full-size image and returns the raw
// response, which is signed: positive where the patch matches the kernel's
// bright side, negative where it matches the dark side.
export function convolveImage(img, H, W, C, kernel, k, stride = 1) {
  const outH = Math.floor((H - k) / stride) + 1;
  const outW = Math.floor((W - k) / stride) + 1;
  const out = new Float32Array(outH * outW);
  const span = k * C;
  for (let oy = 0; oy < outH; oy++) {
    for (let ox = 0; ox < outW; ox++) {
      let sum = 0;
      for (let ky = 0; ky < k; ky++) {
        const row = ((oy * stride + ky) * W + ox * stride) * C;
        const wRow = ky * span;
        for (let t = 0; t < span; t++) sum += img[row + t] * kernel[wRow + t];
      }
      out[oy * outW + ox] = sum;
    }
  }
  return { data: out, H: outH, W: outW };
}

export function grayscaleKernel(net, index) {
  const { k1, channels } = net.cfg;
  const f = filterAt(net, index);
  const out = new Float32Array(k1 * k1);
  for (let i = 0; i < k1 * k1; i++) {
    let sum = 0;
    for (let c = 0; c < channels; c++) sum += f[i * channels + c];
    out[i] = sum / channels;
  }
  return out;
}

function gaborTemplate(k, angle, phase) {
  const out = new Float32Array(k * k);
  const c = (k - 1) / 2;
  const sigma = k / 4;
  const lambda = k / 1.6;
  const ca = Math.cos(angle);
  const sa = Math.sin(angle);
  for (let y = 0; y < k; y++) {
    for (let x = 0; x < k; x++) {
      const dx = x - c;
      const dy = y - c;
      const xr = dx * ca + dy * sa;
      const yr = -dx * sa + dy * ca;
      out[y * k + x] =
        Math.exp(-(xr * xr + yr * yr) / (2 * sigma * sigma)) * Math.cos((2 * Math.PI * xr) / lambda + phase);
    }
  }
  return out;
}

function correlation(a, b) {
  let ma = 0;
  let mb = 0;
  for (let i = 0; i < a.length; i++) {
    ma += a[i];
    mb += b[i];
  }
  ma /= a.length;
  mb /= b.length;
  let num = 0;
  let da = 0;
  let db = 0;
  for (let i = 0; i < a.length; i++) {
    const x = a[i] - ma;
    const y = b[i] - mb;
    num += x * y;
    da += x * x;
    db += y * y;
  }
  const denom = Math.sqrt(da * db);
  return denom < 1e-9 ? 0 : num / denom;
}

// Scores a learned kernel against a bank of ideal oriented-edge templates and
// reports the best match, so "this one is a 45-degree edge detector" is a
// measurement rather than an impression.
export function describeFilter(net, index, angleSteps = 36) {
  const k = net.cfg.k1;
  const gray = grayscaleKernel(net, index);
  let best = { angle: 0, score: 0, phase: 0 };
  for (let a = 0; a < angleSteps; a++) {
    const angle = (a / angleSteps) * Math.PI;
    for (const phase of [0, Math.PI / 2]) {
      const score = correlation(gray, gaborTemplate(k, angle, phase));
      if (Math.abs(score) > Math.abs(best.score)) best = { angle, score, phase };
    }
  }
  const f = filterAt(net, index);
  const channelMeans = [];
  for (let c = 0; c < net.cfg.channels; c++) {
    let sum = 0;
    for (let i = 0; i < k * k; i++) sum += f[i * net.cfg.channels + c];
    channelMeans.push(sum / (k * k));
  }
  const spread = Math.max(...channelMeans) - Math.min(...channelMeans);
  let energy = 0;
  for (let i = 0; i < f.length; i++) energy += f[i] * f[i];
  return {
    angle: best.angle,
    orientationScore: Math.abs(best.score),
    phase: best.phase,
    colorSpread: spread,
    norm: Math.sqrt(energy),
  };
}

function grating(angle, period, size) {
  const out = new Float32Array(size * size);
  const c = (size - 1) / 2;
  const ca = Math.cos(angle);
  const sa = Math.sin(angle);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      out[y * size + x] = Math.cos((2 * Math.PI * ((x - c) * ca + (y - c) * sa)) / period);
    }
  }
  return out;
}

// The orientation tuning curve a visual neuroscientist would measure: show the
// filter a striped patch at each angle in turn and record how hard it answers.
// A filter tuned to one orientation gives a single hump; an untuned one gives
// noise.
export function orientationTuning(weights, k, angles = 24, period = 4, patch = 64) {
  const curve = new Float32Array(angles);
  for (let a = 0; a < angles; a++) {
    const img = grating((a / angles) * Math.PI, period, patch);
    const resp = convolveImage(img, patch, patch, 1, weights, k);
    let energy = 0;
    for (let i = 0; i < resp.data.length; i++) energy += resp.data[i] * resp.data[i];
    curve[a] = Math.sqrt(energy / resp.data.length);
  }
  return curve;
}

export const SOBEL_X = new Float32Array([-1, 0, 1, -2, 0, 2, -1, 0, 1]);
export const SOBEL_Y = new Float32Array([-1, -2, -1, 0, 0, 0, 1, 2, 1]);
