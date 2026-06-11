"""Train the tiny character-level GPT embedded in the inside-an-llm artifact.

Downloads tiny shakespeare to /tmp, trains a 2-layer pre-LN transformer
(~30k params), quantizes every tensor to int8, and writes a JSON blob that
the artifact embeds verbatim. Seeds are fixed so the export is reproducible.

Usage: .venv/bin/python scripts/train-tiny-gpt.py [--steps 8000] [--out /tmp/tiny-gpt-export.json]
"""

import argparse
import base64
import json
import math
import os
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CORPUS_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
CORPUS_PATH = "/tmp/tinyshakespeare.txt"

CONFIG = {
    "n_layer": 2,
    "n_head": 4,
    "d_model": 32,
    "ctx": 64,
    "mlp_ratio": 4,
}


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, ctx):
        super().__init__()
        self.n_head = n_head
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        mask = torch.tril(torch.ones(ctx, ctx, dtype=torch.bool))
        self.register_buffer("mask", mask.view(1, 1, ctx, ctx), persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        hd = C // self.n_head
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, hd).transpose(1, 2)
        k = k.view(B, T, self.n_head, hd).transpose(1, 2)
        v = v.view(B, T, self.n_head, hd).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
        att = att.masked_fill(~self.mask[:, :, :T, :T], float("-inf"))
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, d_model, mlp_ratio):
        super().__init__()
        self.fc = nn.Linear(d_model, mlp_ratio * d_model)
        self.proj = nn.Linear(mlp_ratio * d_model, d_model)
        # tanh approximation so the browser engine can reproduce GELU exactly
        self.act = nn.GELU(approximate="tanh")

    def forward(self, x):
        return self.proj(self.act(self.fc(x)))


class Block(nn.Module):
    def __init__(self, d_model, n_head, ctx, mlp_ratio):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, ctx)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, mlp_ratio)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size, cfg):
        super().__init__()
        d, ctx = cfg["d_model"], cfg["ctx"]
        self.ctx = ctx
        self.wte = nn.Embedding(vocab_size, d)
        self.wpe = nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList(
            Block(d, cfg["n_head"], ctx, cfg["mlp_ratio"])
            for _ in range(cfg["n_layer"])
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

    @torch.no_grad()
    def generate(self, idx, max_new, temperature=0.0, generator=None):
        for _ in range(max_new):
            window = idx[:, -self.ctx :]
            logits = self(window)[:, -1, :]
            if temperature <= 0:
                nxt = logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                nxt = torch.multinomial(probs, 1, generator=generator)
            idx = torch.cat([idx, nxt], dim=1)
        return idx


def load_corpus():
    if not os.path.exists(CORPUS_PATH):
        urllib.request.urlretrieve(CORPUS_URL, CORPUS_PATH)
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        return f.read()


def get_batch(data, ctx, batch_size, generator):
    ix = torch.randint(len(data) - ctx - 1, (batch_size,), generator=generator)
    x = torch.stack([data[i : i + ctx] for i in ix])
    y = torch.stack([data[i + 1 : i + ctx + 1] for i in ix])
    return x, y


@torch.no_grad()
def estimate_loss(model, data, ctx, batches=100, seed=1234):
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    total = 0.0
    for _ in range(batches):
        x, y = get_batch(data, ctx, 64, gen)
        logits = model(x)
        total += F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1)).item()
    model.train()
    return total / batches


def export_tensors(model, cfg):
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
    for name, t in named.items():
        w = t.detach().cpu().numpy().astype(np.float64)
        scale = float(np.max(np.abs(w)) / 127.0)
        if scale == 0.0:
            scale = 1.0
        q = np.clip(np.round(w / scale), -127, 127).astype(np.int8)
        tensors[name] = {
            "shape": list(w.shape),
            "scale": scale,
            "b64": base64.b64encode(q.tobytes()).decode("ascii"),
        }
    return tensors


def dequantize_export(tensors):
    out = {}
    for name, t in tensors.items():
        q = np.frombuffer(base64.b64decode(t["b64"]), dtype=np.int8).astype(np.float64)
        # float64 multiply then one rounding to float32 matches JS Math.fround(q * scale)
        out[name] = torch.from_numpy(
            (q * t["scale"]).astype(np.float32).reshape(t["shape"])
        )
    return out


def load_dequantized(model, tensors):
    deq = dequantize_export(tensors)
    sd = model.state_dict()
    sd["wte.weight"] = deq["wte"]
    sd["wpe.weight"] = deq["wpe"]
    for i in range(len(model.blocks)):
        sd[f"blocks.{i}.ln1.weight"] = deq[f"h{i}.ln1.w"]
        sd[f"blocks.{i}.ln1.bias"] = deq[f"h{i}.ln1.b"]
        sd[f"blocks.{i}.attn.qkv.weight"] = deq[f"h{i}.attn.qkv.w"]
        sd[f"blocks.{i}.attn.qkv.bias"] = deq[f"h{i}.attn.qkv.b"]
        sd[f"blocks.{i}.attn.proj.weight"] = deq[f"h{i}.attn.proj.w"]
        sd[f"blocks.{i}.attn.proj.bias"] = deq[f"h{i}.attn.proj.b"]
        sd[f"blocks.{i}.ln2.weight"] = deq[f"h{i}.ln2.w"]
        sd[f"blocks.{i}.ln2.bias"] = deq[f"h{i}.ln2.b"]
        sd[f"blocks.{i}.mlp.fc.weight"] = deq[f"h{i}.mlp.fc.w"]
        sd[f"blocks.{i}.mlp.fc.bias"] = deq[f"h{i}.mlp.fc.b"]
        sd[f"blocks.{i}.mlp.proj.weight"] = deq[f"h{i}.mlp.proj.w"]
        sd[f"blocks.{i}.mlp.proj.bias"] = deq[f"h{i}.mlp.proj.b"]
    sd["ln_f.weight"] = deq["lnf.w"]
    sd["ln_f.bias"] = deq["lnf.b"]
    model.load_state_dict(sd)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--out", type=str, default="/tmp/tiny-gpt-export.json")
    args = parser.parse_args()

    torch.manual_seed(1337)
    np.random.seed(1337)

    text = load_corpus()
    vocab = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(vocab)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n_train = int(0.9 * len(data))
    train_data, val_data = data[:n_train], data[n_train:]
    print(
        f"corpus {len(text)} chars, vocab {len(vocab)}, train {n_train}, val {len(data) - n_train}"
    )

    model = TinyGPT(len(vocab), CONFIG)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params}")

    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (decay if p.dim() >= 2 else no_decay).append(p)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 0.1},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=2e-3,
        betas=(0.9, 0.99),
    )

    max_lr, min_lr, warmup = 2e-3, 2e-4, 100
    batch_gen = torch.Generator().manual_seed(42)
    ctx = CONFIG["ctx"]
    model.train()
    for step in range(args.steps):
        if step < warmup:
            lr = max_lr * (step + 1) / warmup
        else:
            t = (step - warmup) / max(args.steps - warmup, 1)
            lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * t))
        for g in optimizer.param_groups:
            g["lr"] = lr

        x, y = get_batch(train_data, ctx, 64, batch_gen)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % 1000 == 0 or step == args.steps - 1:
            print(f"step {step:5d}  batch loss {loss.item():.4f}  lr {lr:.2e}")

    train_loss = estimate_loss(model, train_data, ctx)
    val_loss = estimate_loss(model, val_data, ctx)
    print(f"float model:       train {train_loss:.4f}  val {val_loss:.4f}")

    tensors = export_tensors(model, CONFIG)
    export = {"vocab": vocab, "config": CONFIG, "tensors": tensors}
    blob = json.dumps(export, separators=(",", ":"))
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(blob)
    print(f"export written to {args.out} ({len(blob)} bytes)")

    deq_model = load_dequantized(TinyGPT(len(vocab), CONFIG), tensors)
    deq_train = estimate_loss(deq_model, train_data, ctx)
    deq_val = estimate_loss(deq_model, val_data, ctx)
    print(f"dequantized int8:  train {deq_train:.4f}  val {deq_val:.4f}")

    itos = {i: ch for ch, i in stoi.items()}
    for label, temp in [("greedy", 0.0), ("temp 0.8", 0.8)]:
        gen = torch.Generator().manual_seed(7)
        prompt = torch.tensor([[stoi[c] for c in "ROMEO:"]], dtype=torch.long)
        out = deq_model.generate(prompt, 200, temperature=temp, generator=gen)
        sample = "".join(itos[int(i)] for i in out[0].tolist())
        print(f"\n--- dequantized sample ({label}) ---\n{sample}\n")


if __name__ == "__main__":
    main()
