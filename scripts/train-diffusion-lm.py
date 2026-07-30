"""Train the tiny masked-diffusion word LM embedded in the
diffusion-language-models artifact.

Downloads the TinyStories validation split to /tmp/claude (never committed),
builds a word-level vocab of the most frequent words plus [PAD] and [MASK],
trains a small bidirectional transformer encoder with the LLaDA-style masked
diffusion objective (t ~ U(0,1), independent masking, 1/t-weighted CE on
masked positions), quantizes every tensor to int8, and writes the export blob
to src/artifacts/data/diffusion-lm-weights.js. Seeds are fixed so the export
is reproducible.

Usage:
  .venv/bin/python scripts/train-diffusion-lm.py [--steps 15000]
  .venv/bin/python scripts/train-diffusion-lm.py --verify
    Parity gate: runs scripts/parity-diffusion-lm.mjs (which runs the shared
    sampler core module the jsx imports, against the blob as the jsx imports
    it), replays the same greedy confidence-order trajectories with a float64
    numpy reference, and checks token-identical unmasking plus max logit diff.
"""

import argparse
import base64
import json
import math
import os
import re
import subprocess
import sys
import urllib.request
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CORPUS_URL = (
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
    "TinyStoriesV2-GPT4-valid.txt"
)
CORPUS_PATH = "/tmp/claude/tinystories-valid.txt"
CKPT_PATH = "/tmp/claude/diffusion-lm-ckpt.pt"
PARITY_JSON = "/tmp/claude/diffusion-lm-parity.json"

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_JS = os.path.join(
    REPO_ROOT, "src", "artifacts", "data", "diffusion-lm-weights.js"
)
ARTIFACT_JSX = os.path.join(
    REPO_ROOT, "src", "artifacts", "diffusion-language-models.jsx"
)
PARITY_HARNESS = os.path.join(REPO_ROOT, "scripts", "parity-diffusion-lm.mjs")

PAD, MASK = 0, 1
SPECIALS = ["[PAD]", "[MASK]"]

CONFIG = {
    "n_layer": 3,
    "n_head": 4,
    "d_model": 64,
    "ctx": 32,
    "mlp_ratio": 4,
    "vocab_size": 350,
}

WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?|[.,!?\"]")
SENT_END = {".", "!", "?"}


class BiSelfAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.n_head = n_head
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, C = x.shape
        hd = C // self.n_head
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, hd).transpose(1, 2)
        k = k.view(B, T, self.n_head, hd).transpose(1, 2)
        v = v.view(B, T, self.n_head, hd).transpose(1, 2)
        # no causal mask: every position sees the whole canvas, which is what
        # lets the sampler fill positions in any order
        att = F.softmax((q @ k.transpose(-2, -1)) / math.sqrt(hd), dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, d_model, mlp_ratio):
        super().__init__()
        self.fc = nn.Linear(d_model, mlp_ratio * d_model)
        self.proj = nn.Linear(mlp_ratio * d_model, d_model)
        # tanh approximation so the browser forward pass reproduces GELU exactly
        self.act = nn.GELU(approximate="tanh")

    def forward(self, x):
        return self.proj(self.act(self.fc(x)))


class Block(nn.Module):
    def __init__(self, d_model, n_head, mlp_ratio):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = BiSelfAttention(d_model, n_head)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, mlp_ratio)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class DiffusionLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg["d_model"]
        self.wte = nn.Embedding(cfg["vocab_size"], d)
        self.wpe = nn.Embedding(cfg["ctx"], d)
        self.blocks = nn.ModuleList(
            Block(d, cfg["n_head"], cfg["mlp_ratio"]) for _ in range(cfg["n_layer"])
        )
        self.ln_f = nn.LayerNorm(d)
        self.apply(self._init)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg["n_layer"]))

    @staticmethod
    def _init(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        # weight-tied head: logits are dot products against the token embeddings
        return F.linear(x, self.wte.weight)


def load_corpus():
    if not os.path.exists(CORPUS_PATH):
        os.makedirs(os.path.dirname(CORPUS_PATH), exist_ok=True)
        urllib.request.urlretrieve(CORPUS_URL, CORPUS_PATH)
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        return f.read()


def build_dataset():
    text = load_corpus()
    stories = [WORD_RE.findall(s.lower()) for s in text.split("<|endoftext|>")]
    stories = [s for s in stories if len(s) >= CONFIG["ctx"]]

    counts = Counter(tok for s in stories for tok in s)
    keep = [w for w, _ in counts.most_common(CONFIG["vocab_size"] - len(SPECIALS))]
    vocab = SPECIALS + keep
    stoi = {w: i for i, w in enumerate(vocab)}

    total = sum(counts.values())
    covered = sum(counts[w] for w in keep)
    print(
        f"stories {len(stories)}, distinct tokens {len(counts)}, "
        f"vocab {len(vocab)}, token coverage {covered / total:.1%}"
    )

    ctx = CONFIG["ctx"]
    windows = []
    for s in stories:
        in_vocab = [tok in stoi for tok in s]
        for i, tok in enumerate(s):
            at_sentence_start = i == 0 or s[i - 1] in SENT_END
            if not at_sentence_start or i + ctx > len(s):
                continue
            if all(in_vocab[i : i + ctx]):
                windows.append([stoi[t] for t in s[i : i + ctx]])
    data = torch.tensor(windows, dtype=torch.long)
    print(f"windows fully covered by vocab: {len(windows)}")

    perm = torch.randperm(len(data), generator=torch.Generator().manual_seed(7))
    data = data[perm]
    n_val = max(1, len(data) // 20)
    return vocab, data[n_val:], data[:n_val]


def diffusion_loss(model, batch, generator):
    B, T = batch.shape
    t = torch.rand(B, 1, generator=generator).clamp_(0.02, 1.0)
    masked = torch.rand(B, T, generator=generator) < t
    none = ~masked.any(dim=1)
    if none.any():
        force = torch.randint(T, (int(none.sum()),), generator=generator)
        masked[none, force] = True
    x_in = torch.where(masked, torch.full_like(batch, MASK), batch)
    logits = model(x_in)
    ce = F.cross_entropy(
        logits.view(-1, logits.size(-1)), batch.view(-1), reduction="none"
    )
    ce = ce.view(B, T) * masked
    # 1/t weighting makes the objective an ELBO on the data likelihood
    # (LLaDA eq. 5 / MDLM): heavily-masked examples are seen more often at
    # sampling time, lightly-masked ones carry more information per token
    return ((ce.sum(dim=1) / T) / t.squeeze(1)).mean()


@torch.no_grad()
def estimate_loss(model, data, batches=60, seed=1234):
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    total = 0.0
    for _ in range(batches):
        ix = torch.randint(len(data), (128,), generator=gen)
        total += diffusion_loss(model, data[ix], gen).item()
    model.train()
    return total / batches


def export_tensors(model):
    named = {"wte": model.wte.weight, "wpe": model.wpe.weight}
    for i, b in enumerate(model.blocks):
        named[f"h{i}.ln1.w"] = b.ln1.weight
        named[f"h{i}.ln1.b"] = b.ln1.bias
        named[f"h{i}.attn.qkv.w"] = b.attn.qkv.weight
        named[f"h{i}.attn.qkv.b"] = b.attn.qkv.bias
        named[f"h{i}.attn.proj.w"] = b.attn.proj.weight
        named[f"h{i}.attn.proj.b"] = b.attn.proj.bias
        named[f"h{i}.ln2.w"] = b.ln2.weight
        named[f"h{i}.ln2.b"] = b.ln2.bias
        named[f"h{i}.mlp.fc.w"] = b.mlp.fc.weight
        named[f"h{i}.mlp.fc.b"] = b.mlp.fc.bias
        named[f"h{i}.mlp.proj.w"] = b.mlp.proj.weight
        named[f"h{i}.mlp.proj.b"] = b.mlp.proj.bias
    named["lnf.w"] = model.ln_f.weight
    named["lnf.b"] = model.ln_f.bias

    tensors = {}
    for name, tensor in named.items():
        w = tensor.detach().cpu().numpy().astype(np.float64)
        scale = float(np.max(np.abs(w)) / 127.0) or 1.0
        q = np.clip(np.round(w / scale), -127, 127).astype(np.int8)
        tensors[name] = {
            "shape": list(w.shape),
            "scale": scale,
            "b64": base64.b64encode(q.tobytes()).decode("ascii"),
        }
    return tensors


def dequantize(tensors):
    out = {}
    for name, t in tensors.items():
        q = np.frombuffer(base64.b64decode(t["b64"]), dtype=np.int8).astype(np.float64)
        # float64 multiply, one rounding to float32, back to float64: exactly
        # what the browser sees after filling a Float32Array
        out[name] = (
            (q * t["scale"]).astype(np.float32).astype(np.float64).reshape(t["shape"])
        )
    return out


class NumpyModel:
    """Float64 reference forward pass mirroring the JS core operation for
    operation, so parity failures point at real logic differences rather than
    dtype noise."""

    def __init__(self, tensors, cfg):
        self.w = dequantize(tensors)
        self.cfg = cfg

    @staticmethod
    def _ln(x, w, b):
        mu = x.mean(axis=-1, keepdims=True)
        var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)
        return (x - mu) / np.sqrt(var + 1e-5) * w + b

    @staticmethod
    def _softmax(x):
        e = np.exp(x - x.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)

    @staticmethod
    def _gelu(x):
        return (
            0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))
        )

    def forward(self, ids):
        w, cfg = self.w, self.cfg
        T = len(ids)
        d = cfg["d_model"]
        nh = cfg["n_head"]
        hd = d // nh
        x = w["wte"][ids] + w["wpe"][:T]
        for i in range(cfg["n_layer"]):
            h = self._ln(x, w[f"h{i}.ln1.w"], w[f"h{i}.ln1.b"])
            qkv = h @ w[f"h{i}.attn.qkv.w"].T + w[f"h{i}.attn.qkv.b"]
            q, k, v = np.split(qkv, 3, axis=-1)
            q = q.reshape(T, nh, hd).transpose(1, 0, 2)
            k = k.reshape(T, nh, hd).transpose(1, 0, 2)
            v = v.reshape(T, nh, hd).transpose(1, 0, 2)
            att = self._softmax(q @ k.transpose(0, 2, 1) / math.sqrt(hd))
            y = (att @ v).transpose(1, 0, 2).reshape(T, d)
            x = x + y @ w[f"h{i}.attn.proj.w"].T + w[f"h{i}.attn.proj.b"]
            h = self._ln(x, w[f"h{i}.ln2.w"], w[f"h{i}.ln2.b"])
            h = self._gelu(h @ w[f"h{i}.mlp.fc.w"].T + w[f"h{i}.mlp.fc.b"])
            x = x + h @ w[f"h{i}.mlp.proj.w"].T + w[f"h{i}.mlp.proj.b"]
        x = self._ln(x, w["lnf.w"], w["lnf.b"])
        return x @ w["wte"].T


def confidence_trajectory(model, canvas, steps):
    """Greedy confidence-order unmasking, the exact policy the JS core's
    sampleStep implements with order="confidence", temperature 0."""
    ids = [MASK if c is None else c for c in canvas]
    masked = [i for i, c in enumerate(canvas) if c is None]
    snapshots = []
    all_logits = []
    for s in range(steps):
        if not masked:
            snapshots.append(list(ids))
            continue
        logits = model.forward(np.array(ids))
        all_logits.append(logits.copy())
        logits = logits.copy()
        logits[:, : len(SPECIALS)] = -np.inf
        probs = NumpyModel._softmax(logits)
        cands = []
        for pos in masked:
            tok = int(np.argmax(logits[pos]))
            cands.append((-probs[pos, tok], pos, tok))
        cands.sort()
        k = math.ceil(len(masked) / (steps - s))
        for _, pos, tok in cands[:k]:
            ids[pos] = tok
            masked.remove(pos)
        snapshots.append(list(ids))
    return snapshots, all_logits


def verify():
    with open(WEIGHTS_JS, "r", encoding="utf-8") as f:
        js_src = f.read()
    blob = js_src[js_src.index("{") : js_src.rindex("}") + 1]
    export = json.loads(blob)

    with open(ARTIFACT_JSX, "r", encoding="utf-8") as f:
        jsx = f.read()
    if "data/diffusion-lm-weights" not in jsx:
        print("FAIL: artifact jsx does not import data/diffusion-lm-weights.js")
        sys.exit(1)
    print("jsx imports data/diffusion-lm-weights.js: ok")

    print("running node harness (extracts sampler core from the final jsx) ...")
    subprocess.run(["node", PARITY_HARNESS, PARITY_JSON], check=True, cwd=REPO_ROOT)
    with open(PARITY_JSON, "r", encoding="utf-8") as f:
        js_out = json.load(f)

    model = NumpyModel(export["tensors"], export["config"])
    itos = export["vocab"]
    ok = True
    worst = 0.0
    for run in js_out["runs"]:
        canvas = [None if c < 0 else c for c in run["canvas"]]
        py_snaps, py_logits = confidence_trajectory(model, canvas, run["steps"])
        js_snaps = run["snapshots"]
        identical = py_snaps == js_snaps
        ok &= identical
        run_worst = 0.0
        for step_i, b64 in enumerate(run["logitsB64"]):
            js_l = np.frombuffer(base64.b64decode(b64), dtype="<f4").reshape(
                py_logits[step_i].shape
            )
            run_worst = max(run_worst, float(np.abs(js_l - py_logits[step_i]).max()))
        worst = max(worst, run_worst)
        final = " ".join(itos[t] for t in py_snaps[-1])
        print(f"\nprompt: {run['name']} ({run['steps']} steps)")
        print(f"  trajectories token-identical: {identical}")
        print(f"  max |logit_js - logit_py| this run: {run_worst:.3e}")
        print(f"  final canvas: {final}")
        for si, snap in enumerate(py_snaps):
            shown = " ".join("_" if t == MASK else itos[t] for t in snap)
            print(f"    step {si + 1}: {shown}")

    print(f"\nmax logit diff across all runs: {worst:.3e} (gate: < 1e-4)")
    if not ok or worst >= 1e-4:
        print("PARITY FAIL")
        sys.exit(1)
    print("PARITY PASS")


def sample_preview(model_np, vocab, prompt_words, steps=8):
    stoi = {w: i for i, w in enumerate(vocab)}
    ids = [stoi[w] for w in prompt_words]
    canvas = ids + [None] * (CONFIG["ctx"] - len(ids))
    snaps, _ = confidence_trajectory(model_np, canvas, steps)
    return " ".join(vocab[t] for t in snaps[-1])


def train(args):
    torch.manual_seed(1337)
    np.random.seed(1337)
    vocab, train_data, val_data = build_dataset()

    model = DiffusionLM(CONFIG)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params}")

    decay, no_decay = [], []
    for _, p in model.named_parameters():
        (decay if p.dim() >= 2 else no_decay).append(p)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 0.1},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=2e-3,
        betas=(0.9, 0.99),
    )

    max_lr, min_lr, warmup = 2e-3, 2e-4, 200
    gen = torch.Generator().manual_seed(42)
    model.train()
    for step in range(args.steps):
        if step < warmup:
            lr = max_lr * (step + 1) / warmup
        else:
            frac = (step - warmup) / max(args.steps - warmup, 1)
            lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * frac))
        for g in optimizer.param_groups:
            g["lr"] = lr

        ix = torch.randint(len(train_data), (args.batch,), generator=gen)
        loss = diffusion_loss(model, train_data[ix], gen)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 1000 == 0 or step == args.steps - 1:
            print(
                f"step {step:6d}  batch loss {loss.item():.4f}  lr {lr:.2e}", flush=True
            )

    train_loss = estimate_loss(model, train_data)
    val_loss = estimate_loss(model, val_data)
    print(f"float model:      train {train_loss:.4f}  val {val_loss:.4f}")
    torch.save({"model": model.state_dict(), "vocab": vocab}, CKPT_PATH)

    tensors = export_tensors(model)
    export = {"vocab": vocab, "config": CONFIG, "tensors": tensors}
    blob = json.dumps(export, separators=(",", ":"))
    os.makedirs(os.path.dirname(WEIGHTS_JS), exist_ok=True)
    with open(WEIGHTS_JS, "w", encoding="utf-8") as f:
        f.write("export const DIFFUSION_LM = " + blob + ";\n")
    print(f"export written to {WEIGHTS_JS} ({len(blob)} bytes)")

    model_np = NumpyModel(tensors, CONFIG)
    for prompt in [
        ["once", "upon", "a", "time", ","],
        ["the", "little", "girl"],
        ["one", "day", ","],
    ]:
        try:
            out = sample_preview(model_np, vocab, prompt)
            print(f"int8 sample [{' '.join(prompt)}]: {out}")
        except KeyError as err:
            print(f"prompt word not in vocab: {err}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=15000)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verify()
    else:
        train(args)


if __name__ == "__main__":
    main()
