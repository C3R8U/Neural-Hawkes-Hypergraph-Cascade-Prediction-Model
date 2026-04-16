# ============================================================
# Title: Neural Hawkes Hypergraph Cascade Prediction Model
# Author: OpenAI ChatGPT
# Language: English
# ============================================================
# This script is a practical, runnable implementation inspired by
# the paper "Modeling and Analysis of Language Propagation Cascades
# on Social Platforms Using a Neural Hawkes Hypergraph Point Process".
#
# It loads:
#   C:\Users\86156\Desktop\weibo_dataset.txt
#   C:\Users\86156\Desktop\twitter_dataset.txt
#   C:\Users\86156\Desktop\aps_dataset.txt
#
# The script tries to automatically parse multiple possible text formats:
#   1) JSON lines
#   2) tab-separated text
#   3) comma-separated text
#   4) generic delimited text with header
#
# Expected logical fields (best effort, flexible names):
#   cascade_id / cascade / thread_id / root_id
#   user_id / user / uid / author_id
#   timestamp / time / created_at / t
#   text / content / body / message / title
#   depth / level / hop
#
# If a field is missing, this script fills a safe default:
#   - missing text  -> ""
#   - missing depth -> event order within the cascade
#   - missing time  -> order index
#   - missing user  -> anonymous user
#
# Model overview:
#   1) Dynamic hypergraph-style encoder:
#      - Build node-hyperedge incidence matrix from cascade-user relations
#      - Learn user/node latent vectors with a variational hypergraph encoder
#
#   2) Transformer semantic encoder:
#      - Encode event text and cascade depth
#
#   3) Neural Hawkes-style module:
#      - ODE-like hidden state evolution between events
#      - Predict next-event time gap and next user
#      - Predict final cascade size as an auxiliary task
# ============================================================

import os
import re
import json
import math
import time
import random
import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict, Counter


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Configuration
# ============================================================

@dataclass
class Config:
    data_dir: str = r"C:\Users\86156\Desktop"
    dataset_files: Tuple[str, ...] = (
        "weibo_dataset.txt",
        "twitter_dataset.txt",
        "aps_dataset.txt",
    )
    max_vocab_size: int = 20000
    min_token_freq: int = 2
    max_text_len: int = 64
    embed_dim: int = 128
    latent_dim: int = 64
    transformer_heads: int = 4
    transformer_layers: int = 2
    dropout: float = 0.2
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-5
    num_epochs: int = 10
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    hidden_dim: int = 128
    ode_steps: int = 4
    ode_dt_scale: float = 1.0
    time_loss_weight: float = 1.0
    user_loss_weight: float = 1.0
    size_loss_weight: float = 0.5
    kld_weight: float = 1e-4
    print_every: int = 1


CFG = Config()


# ============================================================
# Utility helpers
# ============================================================

def safe_float(x, default=0.0):
    try:
        if x is None:
            return default
        if isinstance(x, (int, float)):
            return float(x)
        x = str(x).strip()
        if x == "":
            return default
        return float(x)
    except Exception:
        return default


def try_parse_time(value, fallback=0.0):
    if value is None:
        return fallback
    if isinstance(value, (int, float)):
        return float(value)

    s = str(value).strip()
    if s == "":
        return fallback

    try:
        return float(s)
    except Exception:
        pass

    fmts = [
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y/%m/%d %H:%M",
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M",
        "%d/%m/%Y",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M",
        "%m/%d/%Y",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
    ]
    for fmt in fmts:
        try:
            dt = datetime.strptime(s, fmt)
            return dt.timestamp()
        except Exception:
            continue

    return fallback


def tokenize(text: str) -> List[str]:
    text = (text or "").lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#", " ", text)
    text = re.sub(r"[^a-zA-Z0-9_\u4e00-\u9fff]+", " ", text)
    tokens = text.strip().split()
    return tokens


def stable_hash(text: str) -> int:
    h = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


# ============================================================
# Data structures
# ============================================================

@dataclass
class Event:
    dataset_name: str
    cascade_id: str
    user_id: str
    timestamp: float
    text: str
    depth: int


@dataclass
class SequenceSample:
    dataset_name: str
    cascade_id: str
    user_ids: List[int]
    times: List[float]
    depths: List[int]
    text_token_ids: List[List[int]]
    final_size: int


# ============================================================
# Flexible TXT data loader
# ============================================================

class FlexibleCascadeLoader:
    def __init__(self, data_dir: str, filenames: Tuple[str, ...]):
        self.data_dir = data_dir
        self.filenames = filenames

        self.cascade_keys = {"cascade_id", "cascade", "thread_id", "root_id", "conversation_id", "topic_id", "paper_id"}
        self.user_keys = {"user_id", "user", "uid", "author_id", "author", "account_id"}
        self.time_keys = {"timestamp", "time", "created_at", "datetime", "date", "event_time", "t"}
        self.text_keys = {"text", "content", "body", "message", "title", "sentence", "post", "tweet", "abstract"}
        self.depth_keys = {"depth", "level", "hop", "layer"}

    def load_all_events(self) -> List[Event]:
        all_events = []
        for fname in self.filenames:
            path = os.path.join(self.data_dir, fname)
            if not os.path.exists(path):
                print(f"[WARN] File not found: {path}")
                continue
            dataset_name = os.path.splitext(fname)[0]
            events = self.load_one_file(path, dataset_name)
            all_events.extend(events)
            print(f"[INFO] Loaded {len(events)} events from {path}")
        return all_events

    def load_one_file(self, path: str, dataset_name: str) -> List[Event]:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = [line.rstrip("\n") for line in f if line.strip()]

        if not lines:
            return []

        json_success = 0
        for line in lines[:50]:
            try:
                row = json.loads(line)
                if isinstance(row, dict):
                    json_success += 1
            except Exception:
                pass

        if json_success >= max(3, len(lines[:50]) // 2):
            rows = []
            for line in lines:
                try:
                    row = json.loads(line)
                    if isinstance(row, dict):
                        rows.append(row)
                except Exception:
                    continue
            return self.rows_to_events(rows, dataset_name)

        for delimiter in ["\t", ",", "|", ";"]:
            maybe = self.try_parse_delimited(lines, delimiter, dataset_name)
            if maybe is not None and len(maybe) > 0:
                return maybe

        return self.parse_heuristic_lines(lines, dataset_name)

    def normalize_key(self, key: str) -> str:
        return str(key).strip().lower().replace(" ", "_")

    def find_value(self, row: Dict, keyset: set):
        for k, v in row.items():
            nk = self.normalize_key(k)
            if nk in keyset:
                return v
        return None

    def rows_to_events(self, rows: List[Dict], dataset_name: str) -> List[Event]:
        events = []
        for idx, row in enumerate(rows):
            cascade_id = self.find_value(row, self.cascade_keys)
            user_id = self.find_value(row, self.user_keys)
            timestamp = self.find_value(row, self.time_keys)
            text = self.find_value(row, self.text_keys)
            depth = self.find_value(row, self.depth_keys)

            if cascade_id is None:
                cascade_id = f"{dataset_name}_cascade_{idx // 20}"
            if user_id is None:
                user_id = f"{dataset_name}_anonymous_{idx % 50}"
            if text is None:
                text = ""
            if depth is None:
                depth = 0

            ev = Event(
                dataset_name=dataset_name,
                cascade_id=str(cascade_id),
                user_id=str(user_id),
                timestamp=try_parse_time(timestamp, fallback=float(idx)),
                text=str(text),
                depth=int(safe_float(depth, default=0)),
            )
            events.append(ev)

        events = self.repair_events(events)
        return events

    def try_parse_delimited(self, lines: List[str], delimiter: str, dataset_name: str):
        if len(lines) < 2:
            return None

        header = [self.normalize_key(x) for x in lines[0].split(delimiter)]
        if len(header) < 2:
            return None

        known = 0
        for h in header:
            if h in self.cascade_keys or h in self.user_keys or h in self.time_keys or h in self.text_keys or h in self.depth_keys:
                known += 1

        if known == 0:
            return None

        rows = []
        for line in lines[1:]:
            parts = line.split(delimiter)
            if len(parts) != len(header):
                if len(parts) > len(header):
                    parts = parts[:len(header)-1] + [delimiter.join(parts[len(header)-1:])]
                else:
                    parts += [""] * (len(header) - len(parts))
            row = dict(zip(header, parts))
            rows.append(row)

        return self.rows_to_events(rows, dataset_name)

    def parse_heuristic_lines(self, lines: List[str], dataset_name: str) -> List[Event]:
        events = []
        current_cascade = None

        for idx, line in enumerate(lines):
            m_cascade = re.search(r"(cascade|thread|root|paper)[=: ]+([A-Za-z0-9_\-]+)", line, re.IGNORECASE)
            if m_cascade:
                current_cascade = m_cascade.group(2)

            parts = re.split(r"\t|,|\|", line)
            parts = [p.strip() for p in parts if p.strip()]

            cascade_id = current_cascade or f"{dataset_name}_cascade_{idx // 20}"
            user_id = f"{dataset_name}_anonymous_{idx % 100}"
            timestamp = float(idx)
            text = line
            depth = idx % 10

            for p in parts:
                if re.match(r"^(user|uid|author)[=: ]", p, re.IGNORECASE):
                    user_id = p.split("=", 1)[-1].split(":", 1)[-1].strip()

            for p in parts:
                if re.match(r"^(time|timestamp|date|t)[=: ]", p, re.IGNORECASE):
                    timestamp = try_parse_time(p.split("=", 1)[-1].split(":", 1)[-1].strip(), fallback=float(idx))

            for p in parts:
                if re.match(r"^(depth|level|hop)[=: ]", p, re.IGNORECASE):
                    depth = int(safe_float(p.split("=", 1)[-1].split(":", 1)[-1].strip(), default=0))

            ev = Event(
                dataset_name=dataset_name,
                cascade_id=str(cascade_id),
                user_id=str(user_id),
                timestamp=float(timestamp),
                text=str(text),
                depth=int(depth),
            )
            events.append(ev)

        events = self.repair_events(events)
        return events

    def repair_events(self, events: List[Event]) -> List[Event]:
        grouped = defaultdict(list)
        for ev in events:
            grouped[(ev.dataset_name, ev.cascade_id)].append(ev)

        repaired = []
        for _, group in grouped.items():
            group = sorted(group, key=lambda x: x.timestamp)
            times = [g.timestamp for g in group]
            if len(set(times)) <= 1:
                base = times[0] if times else 0.0
                for i, ev in enumerate(group):
                    ev.timestamp = base + i

            depths = [g.depth for g in group]
            if sum(depths) == 0:
                for i, ev in enumerate(group):
                    ev.depth = i

            repaired.extend(group)
        return repaired


# ============================================================
# Vocabulary
# ============================================================

class Vocabulary:
    PAD = "<PAD>"
    UNK = "<UNK>"

    def __init__(self, max_size=20000, min_freq=2):
        self.max_size = max_size
        self.min_freq = min_freq
        self.token_to_id = {self.PAD: 0, self.UNK: 1}
        self.id_to_token = {0: self.PAD, 1: self.UNK}

    def build(self, texts: List[str]):
        counter = Counter()
        for text in texts:
            counter.update(tokenize(text))

        sorted_tokens = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        kept = [tok for tok, freq in sorted_tokens if freq >= self.min_freq]
        kept = kept[: max(0, self.max_size - 2)]

        for idx, tok in enumerate(kept, start=2):
            self.token_to_id[tok] = idx
            self.id_to_token[idx] = tok

    def encode(self, text: str, max_len: int = 64) -> List[int]:
        tokens = tokenize(text)[:max_len]
        ids = [self.token_to_id.get(tok, self.token_to_id[self.UNK]) for tok in tokens]
        if not ids:
            ids = [self.token_to_id[self.UNK]]
        return ids


# ============================================================
# Dataset preprocessing
# ============================================================

class CascadePreprocessor:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.vocab = Vocabulary(max_size=cfg.max_vocab_size, min_freq=cfg.min_token_freq)
        self.user_to_id = {}
        self.id_to_user = {}

    def fit_transform(self, events: List[Event]) -> Tuple[List[SequenceSample], Dict]:
        texts = [ev.text for ev in events]
        self.vocab.build(texts)

        users = sorted(list({ev.user_id for ev in events}))
        self.user_to_id = {u: i for i, u in enumerate(users)}
        self.id_to_user = {i: u for u, i in self.user_to_id.items()}

        grouped = defaultdict(list)
        for ev in events:
            grouped[(ev.dataset_name, ev.cascade_id)].append(ev)

        samples = []
        for (dataset_name, cascade_id), group in grouped.items():
            group = sorted(group, key=lambda x: x.timestamp)
            if len(group) < 3:
                continue

            user_ids = [self.user_to_id[g.user_id] for g in group]
            times = [g.timestamp for g in group]
            depths = [g.depth for g in group]
            text_ids = [self.vocab.encode(g.text, max_len=self.cfg.max_text_len) for g in group]
            final_size = len(group)

            samples.append(
                SequenceSample(
                    dataset_name=dataset_name,
                    cascade_id=cascade_id,
                    user_ids=user_ids,
                    times=times,
                    depths=depths,
                    text_token_ids=text_ids,
                    final_size=final_size,
                )
            )

        random.shuffle(samples)
        n = len(samples)
        n_train = int(n * self.cfg.train_ratio)
        n_val = int(n * self.cfg.val_ratio)
        train_samples = samples[:n_train]
        val_samples = samples[n_train:n_train + n_val]
        test_samples = samples[n_train + n_val:]

        meta = {
            "num_users": len(self.user_to_id),
            "vocab_size": len(self.vocab.token_to_id),
            "train_samples": train_samples,
            "val_samples": val_samples,
            "test_samples": test_samples,
            "all_samples": samples,
        }
        return samples, meta


# ============================================================
# Hypergraph builder
# ============================================================

class HypergraphBuilder:
    def __init__(self, num_users: int):
        self.num_users = num_users

    def build_incidence(self, samples: List[SequenceSample]) -> torch.Tensor:
        num_edges = max(1, len(samples))
        H = torch.zeros((self.num_users, num_edges), dtype=torch.float32)

        for e_idx, sample in enumerate(samples):
            unique_users = set(sample.user_ids)
            for u in unique_users:
                H[u, e_idx] = 1.0
        return H


# ============================================================
# Model blocks
# ============================================================

class HypergraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.2):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        device = X.device

        W_e = torch.ones(H.size(1), device=device)
        De = torch.diag(torch.clamp(H.sum(dim=0), min=1.0))
        Dv = torch.diag(torch.clamp(H.sum(dim=1), min=1.0))
        We = torch.diag(W_e)

        Dv_inv_sqrt = torch.linalg.inv(torch.sqrt(Dv + eps * torch.eye(Dv.size(0), device=device)))
        De_inv = torch.linalg.inv(De + eps * torch.eye(De.size(0), device=device))

        propagation = Dv_inv_sqrt @ H @ We @ De_inv @ H.t() @ Dv_inv_sqrt
        out = propagation @ X
        out = self.linear(out)
        out = torch.relu(out)
        out = self.dropout(out)
        return out


class VariationalHypergraphEncoder(nn.Module):
    def __init__(self, num_users: int, hidden_dim: int, latent_dim: int, dropout: float = 0.2):
        super().__init__()
        self.user_features = nn.Parameter(torch.randn(num_users, hidden_dim) * 0.02)

        self.hg1 = HypergraphConv(hidden_dim, hidden_dim, dropout)
        self.mu_layer = HypergraphConv(hidden_dim, latent_dim, dropout)
        self.logvar_layer = HypergraphConv(hidden_dim, latent_dim, dropout)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode_adj(self, Z):
        recon = torch.sigmoid(Z @ Z.t())
        return recon

    def forward(self, H: torch.Tensor):
        X = self.user_features
        hidden = self.hg1(X, H)
        mu = self.mu_layer(hidden, H)
        logvar = self.logvar_layer(hidden, H)
        z = self.reparameterize(mu, logvar)
        recon = self.decode_adj(z)
        return z, mu, logvar, recon


class PositionalDepthEncoding(nn.Module):
    def __init__(self, dim: int, max_depth: int = 512):
        super().__init__()
        pe = torch.zeros(max_depth, dim)
        pos = torch.arange(0, max_depth, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.register_buffer("pe", pe)

    def forward(self, depths: torch.Tensor) -> torch.Tensor:
        depths = torch.clamp(depths, min=0, max=self.pe.size(0) - 1)
        return self.pe[depths]


class SemanticTransformerEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int, num_layers: int, dropout: float = 0.2):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.depth_embed = PositionalDepthEncoding(embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.time_decay = nn.Parameter(torch.tensor(0.1))

    def encode_text_batch(self, token_batch: List[List[int]]) -> torch.Tensor:
        max_len = max(len(x) for x in token_batch)
        ids = []
        mask = []
        for seq in token_batch:
            padded = seq + [0] * (max_len - len(seq))
            m = [1] * len(seq) + [0] * (max_len - len(seq))
            ids.append(padded)
            mask.append(m)

        ids = torch.tensor(ids, dtype=torch.long, device=DEVICE)
        mask = torch.tensor(mask, dtype=torch.bool, device=DEVICE)

        emb = self.token_embed(ids)
        mask_f = mask.unsqueeze(-1).float()
        pooled = (emb * mask_f).sum(dim=1) / torch.clamp(mask_f.sum(dim=1), min=1.0)
        return pooled

    def forward(self, token_ids_per_event: List[List[int]], depths: List[int], event_times: List[float]) -> torch.Tensor:
        event_emb = self.encode_text_batch(token_ids_per_event)

        depth_tensor = torch.tensor(depths, dtype=torch.long, device=DEVICE)
        depth_emb = self.depth_embed(depth_tensor)

        x = event_emb + depth_emb
        x = x.unsqueeze(0)
        h = self.transformer(x).squeeze(0)

        t = torch.tensor(event_times, dtype=torch.float32, device=DEVICE)
        t0 = t[0]
        delta = t - t0
        decay = torch.exp(-torch.relu(self.time_decay) * delta.unsqueeze(-1))
        h = h * decay
        return h


class ODEFunction(nn.Module):
    def __init__(self, hidden_dim: int, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + input_dim + 1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

    def forward(self, h: torch.Tensor, x: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([h, x, dt], dim=-1)
        dh = self.net(inp)
        return dh


class NHHPPModel(nn.Module):
    def __init__(self, cfg: Config, vocab_size: int, num_users: int):
        super().__init__()
        self.cfg = cfg
        self.num_users = num_users

        self.hyper_encoder = VariationalHypergraphEncoder(
            num_users=num_users,
            hidden_dim=cfg.hidden_dim,
            latent_dim=cfg.latent_dim,
            dropout=cfg.dropout,
        )

        self.semantic_encoder = SemanticTransformerEncoder(
            vocab_size=vocab_size,
            embed_dim=cfg.embed_dim,
            num_heads=cfg.transformer_heads,
            num_layers=cfg.transformer_layers,
            dropout=cfg.dropout,
        )

        fused_dim = cfg.latent_dim + cfg.embed_dim

        self.input_proj = nn.Sequential(
            nn.Linear(fused_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
        )

        self.ode_func = ODEFunction(cfg.hidden_dim, cfg.hidden_dim)
        self.gru_cell = nn.GRUCell(cfg.hidden_dim, cfg.hidden_dim)

        self.user_head = nn.Linear(cfg.hidden_dim, num_users)
        self.time_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim // 2, 1),
            nn.Softplus(),
        )
        self.size_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim // 2, 1),
            nn.Softplus(),
        )

        self.intensity_w = nn.Linear(cfg.hidden_dim, 1)
        self.intensity_b = nn.Parameter(torch.tensor(0.1))

    def evolve_hidden(self, h: torch.Tensor, x: torch.Tensor, dt_scalar: float) -> torch.Tensor:
        dt_total = max(float(dt_scalar), 1e-6)
        steps = self.cfg.ode_steps
        dt = dt_total / steps

        for _ in range(steps):
            dt_tensor = torch.tensor([[dt]], dtype=torch.float32, device=DEVICE)
            dh = self.ode_func(h, x, dt_tensor)
            h = h + dt * dh
        return h

    def forward(self, sample: SequenceSample, H: torch.Tensor):
        user_z, mu, logvar, recon_adj = self.hyper_encoder(H)

        semantic_h = self.semantic_encoder(
            token_ids_per_event=sample.text_token_ids,
            depths=sample.depths,
            event_times=sample.times,
        )

        T = len(sample.user_ids)
        fused_inputs = []
        for i in range(T):
            u = sample.user_ids[i]
            z_u = user_z[u]
            s_i = semantic_h[i]
            fused = torch.cat([z_u, s_i], dim=-1)
            fused_inputs.append(fused)
        fused_inputs = torch.stack(fused_inputs, dim=0)

        x_proj = self.input_proj(fused_inputs)
        h = torch.zeros((1, self.cfg.hidden_dim), dtype=torch.float32, device=DEVICE)

        user_logits_list = []
        time_preds_list = []
        intensity_list = []

        for i in range(T - 1):
            current_x = x_proj[i].unsqueeze(0)
            h = self.gru_cell(current_x, h).unsqueeze(0) if h.dim() == 1 else self.gru_cell(current_x, h.squeeze(0)).unsqueeze(0)
            intensity = torch.softplus(self.intensity_w(h) + self.intensity_b)
            intensity_list.append(intensity)
            user_logits = self.user_head(h)
            time_pred = self.time_head(h)
            user_logits_list.append(user_logits)
            time_preds_list.append(time_pred)
            dt = sample.times[i + 1] - sample.times[i]
            h = self.evolve_hidden(h, current_x, dt)

        size_pred = self.size_head(h)

        outputs = {
            "user_logits": torch.cat(user_logits_list, dim=0),
            "time_preds": torch.cat(time_preds_list, dim=0).squeeze(-1),
            "size_pred": size_pred.squeeze(-1).squeeze(-1),
            "mu": mu,
            "logvar": logvar,
            "recon_adj": recon_adj,
            "user_z": user_z,
            "intensity_list": intensity_list,
        }
        return outputs


# ============================================================
# Loss functions
# ============================================================

def kld_gaussian(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def build_user_adj_from_H(H: torch.Tensor) -> torch.Tensor:
    A = (H @ H.t()) > 0
    A = A.float()
    A.fill_diagonal_(1.0)
    return A


def compute_losses(outputs, sample: SequenceSample, H: torch.Tensor, cfg: Config):
    target_next_users = torch.tensor(sample.user_ids[1:], dtype=torch.long, device=DEVICE)
    target_time_gaps = []
    for i in range(len(sample.times) - 1):
        gap = max(sample.times[i + 1] - sample.times[i], 1e-6)
        target_time_gaps.append(gap)
    target_time_gaps = torch.tensor(target_time_gaps, dtype=torch.float32, device=DEVICE)

    target_log_gaps = torch.log1p(target_time_gaps)
    pred_log_gaps = torch.log1p(outputs["time_preds"] + 1e-6)

    user_loss = nn.CrossEntropyLoss()(outputs["user_logits"], target_next_users)
    time_loss = nn.MSELoss()(pred_log_gaps, target_log_gaps)

    target_size = torch.tensor(float(sample.final_size), dtype=torch.float32, device=DEVICE)
    pred_size = outputs["size_pred"]
    size_loss = nn.MSELoss()(torch.log1p(pred_size + 1e-6), torch.log1p(target_size))

    adj_target = build_user_adj_from_H(H).to(DEVICE)
    recon_loss = nn.BCELoss()(outputs["recon_adj"], adj_target)

    kld = kld_gaussian(outputs["mu"], outputs["logvar"])

    total_loss = (
        cfg.user_loss_weight * user_loss
        + cfg.time_loss_weight * time_loss
        + cfg.size_loss_weight * size_loss
        + recon_loss
        + cfg.kld_weight * kld
    )

    metrics = {
        "total_loss": total_loss,
        "user_loss": user_loss.detach().item(),
        "time_loss": time_loss.detach().item(),
        "size_loss": size_loss.detach().item(),
        "recon_loss": recon_loss.detach().item(),
        "kld": kld.detach().item(),
        "target_size": float(target_size.item()),
        "pred_size": float(pred_size.detach().item()),
    }
    return total_loss, metrics


# ============================================================
# Training / evaluation
# ============================================================

def evaluate(model: nn.Module, samples: List[SequenceSample], H: torch.Tensor, cfg: Config):
    model.eval()
    total = 0.0
    n = 0

    mae_time = []
    acc_user = []
    msle_size = []

    with torch.no_grad():
        for sample in samples:
            outputs = model(sample, H.to(DEVICE))

            pred_users = outputs["user_logits"].argmax(dim=-1).cpu().numpy().tolist()
            true_users = sample.user_ids[1:]
            correct = sum(int(p == t) for p, t in zip(pred_users, true_users))
            acc_user.append(correct / max(1, len(true_users)))

            pred_gaps = outputs["time_preds"].cpu().numpy().tolist()
            true_gaps = [max(sample.times[i + 1] - sample.times[i], 1e-6) for i in range(len(sample.times) - 1)]
            abs_err = np.mean([abs(p - t) for p, t in zip(pred_gaps, true_gaps)])
            mae_time.append(abs_err)

            pred_size = max(float(outputs["size_pred"].item()), 0.0)
            true_size = float(sample.final_size)
            msle = (math.log1p(pred_size) - math.log1p(true_size)) ** 2
            msle_size.append(msle)

            loss, _ = compute_losses(outputs, sample, H.to(DEVICE), cfg)
            total += float(loss.item())
            n += 1

    result = {
        "loss": total / max(1, n),
        "user_acc": float(np.mean(acc_user)) if acc_user else 0.0,
        "time_mae": float(np.mean(mae_time)) if mae_time else 0.0,
        "size_msle": float(np.mean(msle_size)) if msle_size else 0.0,
    }
    return result


def train_model(model: nn.Module, train_samples: List[SequenceSample], val_samples: List[SequenceSample], H: torch.Tensor, cfg: Config):
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_val = float("inf")
    best_state = None

    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        random.shuffle(train_samples)

        epoch_losses = []
        start = time.time()

        for sample in train_samples:
            optimizer.zero_grad()
            outputs = model(sample, H.to(DEVICE))
            loss, metrics = compute_losses(outputs, sample, H.to(DEVICE), cfg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        val_result = evaluate(model, val_samples, H, cfg) if val_samples else {"loss": 0.0, "user_acc": 0.0, "time_mae": 0.0, "size_msle": 0.0}

        if val_result["loss"] < best_val:
            best_val = val_result["loss"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % cfg.print_every == 0:
            print(
                f"[Epoch {epoch:03d}] "
                f"train_loss={np.mean(epoch_losses):.4f} "
                f"val_loss={val_result['loss']:.4f} "
                f"val_user_acc={val_result['user_acc']:.4f} "
                f"val_time_mae={val_result['time_mae']:.4f} "
                f"val_size_msle={val_result['size_msle']:.4f} "
                f"time={time.time() - start:.2f}s"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 80)
    print("NHHPP-style local training example")
    print(f"Device: {DEVICE}")
    print("=" * 80)

    loader = FlexibleCascadeLoader(CFG.data_dir, CFG.dataset_files)
    events = loader.load_all_events()

    if len(events) == 0:
        raise FileNotFoundError(
            "No events were loaded. Please check that these files exist:\n"
            f"{os.path.join(CFG.data_dir, 'weibo_dataset.txt')}\n"
            f"{os.path.join(CFG.data_dir, 'twitter_dataset.txt')}\n"
            f"{os.path.join(CFG.data_dir, 'aps_dataset.txt')}"
        )

    print(f"[INFO] Total events loaded: {len(events)}")

    preprocessor = CascadePreprocessor(CFG)
    samples, meta = preprocessor.fit_transform(events)

    train_samples = meta["train_samples"]
    val_samples = meta["val_samples"]
    test_samples = meta["test_samples"]

    num_users = meta["num_users"]
    vocab_size = meta["vocab_size"]

    print(f"[INFO] Number of cascades/sequences: {len(samples)}")
    print(f"[INFO] Train/Val/Test: {len(train_samples)}/{len(val_samples)}/{len(test_samples)}")
    print(f"[INFO] Number of users: {num_users}")
    print(f"[INFO] Vocabulary size: {vocab_size}")

    if len(train_samples) == 0:
        raise RuntimeError("Training split is empty. The dataset parsing may need adjustment for your specific TXT format.")

    hg_builder = HypergraphBuilder(num_users=num_users)
    H = hg_builder.build_incidence(train_samples + val_samples + test_samples)
    print(f"[INFO] Hypergraph incidence matrix shape: {tuple(H.shape)}")

    model = NHHPPModel(
        cfg=CFG,
        vocab_size=vocab_size,
        num_users=num_users,
    ).to(DEVICE)

    model = train_model(model, train_samples, val_samples, H, CFG)

    train_result = evaluate(model, train_samples, H, CFG)
    val_result = evaluate(model, val_samples, H, CFG) if val_samples else {"loss": 0.0, "user_acc": 0.0, "time_mae": 0.0, "size_msle": 0.0}
    test_result = evaluate(model, test_samples, H, CFG) if test_samples else {"loss": 0.0, "user_acc": 0.0, "time_mae": 0.0, "size_msle": 0.0}

    print("\n" + "=" * 80)
    print("Final Results")
    print("=" * 80)
    print(f"Train Loss      : {train_result['loss']:.6f}")
    print(f"Train User Acc  : {train_result['user_acc']:.6f}")
    print(f"Train Time MAE  : {train_result['time_mae']:.6f}")
    print(f"Train Size MSLE : {train_result['size_msle']:.6f}")
    print("-" * 80)
    print(f"Val Loss        : {val_result['loss']:.6f}")
    print(f"Val User Acc    : {val_result['user_acc']:.6f}")
    print(f"Val Time MAE    : {val_result['time_mae']:.6f}")
    print(f"Val Size MSLE   : {val_result['size_msle']:.6f}")
    print("-" * 80)
    print(f"Test Loss       : {test_result['loss']:.6f}")
    print(f"Test User Acc   : {test_result['user_acc']:.6f}")
    print(f"Test Time MAE   : {test_result['time_mae']:.6f}")
    print(f"Test Size MSLE  : {test_result['size_msle']:.6f}")
    print("=" * 80)

    if len(test_samples) > 0:
        sample = test_samples[0]
        model.eval()
        with torch.no_grad():
            outputs = model(sample, H.to(DEVICE))

        pred_users = outputs["user_logits"].argmax(dim=-1).cpu().numpy().tolist()
        pred_gaps = outputs["time_preds"].cpu().numpy().tolist()
        pred_size = float(outputs["size_pred"].item())

        print("\nExample Prediction on First Test Cascade")
        print("=" * 80)
        print(f"Dataset         : {sample.dataset_name}")
        print(f"Cascade ID      : {sample.cascade_id}")
        print(f"Observed length : {len(sample.user_ids)}")
        print(f"True final size : {sample.final_size}")
        print(f"Pred final size : {pred_size:.4f}")
        print("-" * 80)
        print("Next-user predictions (ID space):")
        print(pred_users)
        print("-" * 80)
        print("Next-time-gap predictions:")
        print([round(x, 6) for x in pred_gaps])
        print("=" * 80)


if __name__ == "__main__":
    main()
