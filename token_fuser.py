# imf_pipeline.py
import os
import re
import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# -----------------------------
# Simple config object (editable)
# -----------------------------
class CFG:
    # I/O
    vmd_save_dir = "vmd_results"         # put your *_vmd.npz files here
    feat_save_dir = "features"           # step2 outputs
    feat_npz_dir = "features_npz"
    feat_excel = "features/all_file_features.csv"

    # IMF token fuser params
    fuser = lambda: None
    fuser.d_model = 64
    fuser.nhead = 4
    fuser.num_layers = 1
    fuser.dropout = 0.1
    fuser.se_ratio = 4

    # postprocessing / pruning
    post = lambda: None
    post.use_fbe = False
    post.fbe_top_k = 6
    post.prune_energy_thr = 0.01
    post.prune_corr_thr = 0.9
    post.weight_eta = 1.0
    post.wavelet = None
    post.wavelet_levels = None

    seg = lambda: None
    seg.fs_target = 16000

    phys = lambda: None
    phys.bands = [0, 200, 400, 800, 1600, 3200]

    # training params
    train = lambda: None
    train.batch_size = 32
    train.epochs = 20
    train.lr = 1e-3
    train.weight_decay = 1e-5
    train.contrastive_lambda = 1.0
    train.temperature = 0.07
    train.device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Utility / placeholder functions
# -----------------------------

def ola_reconstruct(frames: np.ndarray, frame_win: np.ndarray, frame_hop: int, frames_are_windowed=True):
    """Simple overlap-add reconstruction (assumes frames are successive)."""
    num_frames, frame_len = frames.shape
    out_len = (num_frames - 1) * frame_hop + frame_len
    out = np.zeros(out_len)
    win = frame_win if frame_win is not None else np.ones(frame_len)
    for i in range(num_frames):
        start = i * frame_hop
        out[start:start+frame_len] += frames[i] * (win if frames_are_windowed else 1.0)
    return out

def select_imfs_by_FBE(U: np.ndarray, fs: int, use_envelope=True, top_k=6):
    """
    Placeholder: select top_k IMFs by band energy (simple heuristic).
    U: shape (K, N)
    returns: U_sel (K_sel, N)
    """
    if U.size == 0:
        return U
    K, N = U.shape
    energies = np.sqrt(np.mean(U**2, axis=1))
    idx = np.argsort(energies)[::-1][:top_k]
    return U[idx, :]

def prune_merge_imfs(U: np.ndarray, energy_thr=0.01, corr_thr=0.9):
    """Placeholder: remove low-energy IMFs and merge highly correlated ones (simple)."""
    if U.size == 0:
        return U
    energies = np.sqrt(np.mean(U**2, axis=1))
    keep = energies > (energies.max() * energy_thr)
    Uk = U[keep]
    # No actual merging implemented here for simplicity
    return Uk

def compute_weights(U_sel: np.ndarray, e: np.ndarray, eta=1.0):
    """simple energy-proportional weights"""
    if U_sel.size == 0:
        return np.zeros((0,))
    energies = np.sqrt(np.mean(U_sel**2, axis=1))
    w = energies ** eta
    w = w + 1e-12
    return w / (w.sum() + 1e-12)

def reconstruct(U_sel: np.ndarray, e: np.ndarray, w: np.ndarray, wavelet=None, levels=None):
    """Weighted sum reconstruction - returns s_hat and U_den (dense IMF aligned)"""
    if U_sel.size == 0:
        return np.zeros((0,)), np.zeros((0, 0))
    s_hat = (w[:, None] * U_sel).sum(axis=0) + e  # include residual
    U_den = (w[:, None] * U_sel)  # (K_sel, N)
    return s_hat, U_den

# A basic imf_features function (returns feature vector per IMF frame)
def imf_features(U_den: np.ndarray, fs: int, bands: List[int]):
    """
    U_den: (K_sel, N) for a segment
    returns: feats_per_imf: (K_sel, D) and feat_names
    We'll compute a compact set of per-imf features: energy, rms, zcr, centroid, mel approx (12),
    mfcc-like pseudo (12 via DCT of log-spectrum), envelope stats.
    This is a simplified but runnable version — replace with your lab's detailed features.
    """
    K, N = U_den.shape
    feats = []
    feat_names = []
    # For reproducibility, define mel-like 12 dims via FFT bin pooling
    for k in range(K):
        x = U_den[k]
        energy = np.sum(x**2)
        rms = np.sqrt(np.mean(x**2))
        zcr = ((x[:-1] * x[1:]) < 0).sum() / max(1, len(x)-1)
        # spectrum
        X = np.abs(np.fft.rfft(x))
        freqs = np.fft.rfftfreq(len(x), d=1.0/fs)
        centroid = np.sum(freqs * X) / (np.sum(X) + 1e-12)
        bandwidth = np.sqrt(np.sum(((freqs - centroid)**2) * X) / (np.sum(X) + 1e-12))
        # mel-like pooling: split freq axis into 12 bins
        bins = np.array_split(X, 12)
        mel_feats = [np.log(np.sum(b) + 1e-12) for b in bins]
        # pseudo-mfcc: dct of log-spectrum, take first 12
        logspec = np.log(X + 1e-12)
        from scipy.fftpack import dct
        mfcc_like = dct(logspec, type=2, norm='ortho')[:12]
        # envelope stats
        env = np.abs(np.imag(np.hilbert(x))) if np.isrealobj(x) else np.abs(x)
        env_mean = np.mean(env)
        env_std = np.std(env)
        # combine
        feat = np.hstack([
            energy, rms, zcr, centroid, bandwidth,
            np.array(mel_feats),
            mfcc_like,
            env_mean, env_std
        ])
        feats.append(feat)
    feats = np.vstack(feats)  # (K_sel, D)
    # construct names
    feat_names = ["energy", "rms", "zcr", "centroid", "bandwidth"] + \
                 [f"mel_{i}" for i in range(12)] + \
                 [f"mfcc_like_{i}" for i in range(12)] + \
                 ["env_mean", "env_std"]
    return feats, feat_names

# -----------------------------
# Model components (IMFTokenFuser, small projection)
# -----------------------------
class SimpleSEBlock(nn.Module):
    def __init__(self, dim, se_ratio=4):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim // se_ratio)
        self.fc2 = nn.Linear(dim // se_ratio, dim)
    def forward(self, x):
        # x: (B, N, D)
        s = x.mean(dim=1)  # (B, D)
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))  # (B, D)
        return x * s.unsqueeze(1)

class IMFTokenFuser(nn.Module):
    """
    Lightweight fuser that (a) projects group tokens to d_model,
    (b) applies multi-head self-attention across tokens (IMFs),
    (c) returns fused token (pooled) and attention weights.
    """
    def __init__(self, d_model=64, nhead=4, num_layers=1, dropout=0.1, se_ratio=4, bands=None):
        super().__init__()
        self.d_model = d_model
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*4, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.se = SimpleSEBlock(d_model, se_ratio=se_ratio)
        # optional conditioning on frequency centers (we assume F: (B, N, 1) added in forward)
        self.freq_proj = nn.Linear(1, d_model) if bands is not None else None

    def forward(self, T: torch.Tensor, F_center: torch.Tensor = None):
        """
        T: (B, N, d_model) tokens per IMF
        F_center: (B, N, 1) optional frequency center or band info
        returns:
          z: (B, d_model) fused representation (pooled)
          h: attention weights approx (B, N) - here we approximate via token norms or transformer's attn not directly accessible
        """
        # optionally incorporate F_center by adding small projection
        if self.freq_proj is not None and F_center is not None:
            T = T + self.freq_proj(F_center)  # broadcast add
        # transformer expects (B, N, D)
        out = self.transformer(T)  # (B, N, D)
        # apply SE block
        out = self.se(out)
        # pooled representation
        # transpose for AdaptiveAvgPool1d: (B, D, N)
        z = self.pool(out.transpose(1,2)).squeeze(-1)  # (B, D)
        # approximate attention scores via token L2 norms (just for debugging & saving)
        h = torch.norm(out, dim=-1, keepdim=False)  # (B,N)
        # normalize into [0,1]
        h = (h - h.min(dim=1, keepdim=True)[0]) / (h.max(dim=1, keepdim=True)[0] - h.min(dim=1, keepdim=True)[0] + 1e-12)
        return z, h  # fused, attn-like

# -----------------------------
# Step2: Feature representation (IMF-token construction)
# -----------------------------
def step2_imf_token_representation(vmd_save_dir: str, feat_npz_dir: str, cfg: CFG = CFG):
    """
    1) 遍历 vmd files
    2) IMF 筛选 & 加权重构
    3) per-IMF 多域特征计算 -> 每个 IMF 得到一个 token (D)
    4) 段级/文件级池化并保存单文件 npz（包含 per-IMF tokens）
    """
    os.makedirs(feat_npz_dir, exist_ok=True)
    vmd_files = [f for f in os.listdir(vmd_save_dir) if f.endswith("_vmd.npz")]
    assert len(vmd_files) > 0, f"未找到 VMD 文件 in {vmd_save_dir}"
    all_feat_rows = []
    for f in tqdm(vmd_files, desc="Step2 VMD files"):
        path = os.path.join(vmd_save_dir, f)
        data = np.load(path, allow_pickle=True)
        seg_vmd_results = data["seg_vmd_results"].tolist()
        frame_win = data.get("frame_win", None)
        frame_hop = int(data.get("frame_hop", 256))
        fs = int(data.get("fs", cfg.seg.fs_target))
        # parse label from filename
        label_m = re.search(r"label(\d+)_vmd\.npz", f)
        label = int(label_m.group(1)) if label_m else -1
        stem = f.replace(f"_label{label}_vmd.npz", "")

        file_token_list = []  # will store fused per-seg tokens for file-level pooling if needed
        imf_token_allsegs = []  # list of lists: seg -> [K tokens x D]
        seg_attn_all = []
        for seg_idx, seg_vmd in enumerate(seg_vmd_results):
            U = seg_vmd.get("U_fix", seg_vmd.get("U", None))
            e = seg_vmd.get("e_fix", np.zeros(U.shape[1]) if U is not None else np.zeros(1))
            if U is None or U.size == 0:
                continue
            # IMF selection
            if cfg.post.use_fbe:
                U_sel = select_imfs_by_FBE(U, fs, use_envelope=True, top_k=cfg.post.fbe_top_k)
            else:
                U_sel = prune_merge_imfs(U, cfg.post.prune_energy_thr, cfg.post.prune_corr_thr)
            if U_sel.size == 0:
                continue
            # weights & reconstruction
            w = compute_weights(U_sel, e, eta=cfg.post.weight_eta)
            s_hat, U_den = reconstruct(U_sel, e, w, cfg.post.wavelet, cfg.post.wavelet_levels)  # U_den shape (K_sel, N)
            # per-IMF features
            feats, feat_names = imf_features(U_den, fs, cfg.phys.bands)
            # pool per-imf features across frames inside IMF (feats is already per-IMF)
            # If feats shape (K_sel, D), we treat each row as token for that IMF
            imf_tokens = feats  # (K_sel, D)
            imf_token_allsegs.append(imf_tokens)
            # compute simple attn proxy: mean energy per IMF
            seg_attn_all.append(np.sqrt(np.mean(U_den**2, axis=1)))
        # File-level aggregation / saving
        if len(imf_token_allsegs) == 0:
            print(f"⚠ {f} no valid imf tokens - skipped")
            continue
        # convert to per-file structure: list of segments, each seg is array (K_seg, D)
        out_path = os.path.join(feat_npz_dir, f"{stem}_label{label}_tokens.npz")
        np.savez(out_path,
                 filename=stem,
                 label=label,
                 seg_imf_tokens=np.array(imf_token_allsegs, dtype=object),
                 seg_attn=np.array(seg_attn_all, dtype=object),
                 feat_names=np.array(feat_names, dtype=object))
        # also collect file-level mean for quick summary table
        # we pool by taking mean of all per-IMF tokens flattened
        all_tokens_flat = np.vstack([t for seg in imf_token_allsegs for t in seg])
        file_feat_mean = np.mean(all_tokens_flat, axis=0)
        row = {"filename": stem, "label": label}
        for k, nm in enumerate(feat_names):
            row[nm] = file_feat_mean[k]
        all_feat_rows.append(row)
        print(f"Saved tokens: {out_path} (segments={len(imf_token_allsegs)})")
    # save CSV summary
    df = pd.DataFrame(all_feat_rows)
    csvp = os.path.join(cfg.feat_save_dir if hasattr(cfg, "feat_save_dir") else ".", "all_file_features.csv")
    os.makedirs(os.path.dirname(csvp), exist_ok=True)
    df.to_csv(csvp, index=False)
    print("Step2 completed, summary CSV saved at:", csvp)
    return df

# -----------------------------
# Helper dataset to load token files
# -----------------------------
class IMFTokenDataset(Dataset):
    def __init__(self, token_npz_files: List[str], cfg: CFG = CFG, max_tokens_per_file=32):
        """
        token_npz_files: list of .npz files created by step2_imf_token_representation
        For each file we will flatten seg_imf_tokens -> a list of IMF-level tokens,
        and treat each file as a bag of tokens (we'll sample up to max_tokens_per_file tokens per file)
        """
        self.files = token_npz_files
        self.cfg = cfg
        self.max_tokens = max_tokens_per_file
        self.data = []  # each entry: dict{tokens: np.array(N, D), label:int, filename:str}
        for p in self.files:
            d = np.load(p, allow_pickle=True)
            segs = d["seg_imf_tokens"]
            label = int(d["label"])
            fname = str(d["filename"])
            # flatten segs (list of arrays)
            all_tokens = np.vstack([seg for seg in segs])
            # optionally shuffle tokens
            if all_tokens.shape[0] == 0:
                continue
            self.data.append({"tokens": all_tokens.astype(np.float32), "label": label, "filename": fname})
        print(f"Dataset built with {len(self.data)} files.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        tokens = entry["tokens"]
        # sample up to max_tokens (pad if needed)
        N, D = tokens.shape
        if N >= self.max_tokens:
            inds = np.random.choice(N, self.max_tokens, replace=False)
            tok = tokens[inds]
            mask = np.ones(self.max_tokens, dtype=np.float32)
        else:
            pad = np.zeros((self.max_tokens - N, D), dtype=np.float32)
            tok = np.vstack([tokens, pad])
            mask = np.concatenate([np.ones(N, dtype=np.float32), np.zeros(self.max_tokens - N, dtype=np.float32)])
        return {"tokens": tok, "mask": mask, "label": entry["label"], "filename": entry["filename"]}

# -----------------------------
# Step3: Attention fusion (batch inference mode)
# -----------------------------
def step3_attention_fusion(token_npz_dir: str, fused_out_dir: str, cfg: CFG = CFG):
    """
    Load token files -> project tokens -> run IMFTokenFuser -> save fused representations per file.
    """
    os.makedirs(fused_out_dir, exist_ok=True)
    files = [os.path.join(token_npz_dir, f) for f in os.listdir(token_npz_dir) if f.endswith("_tokens.npz")]
    if len(files) == 0:
        raise RuntimeError("No token npz files found in " + token_npz_dir)
    device = cfg.train.device
    fuser = IMFTokenFuser(d_model=cfg.fuser.d_model, nhead=cfg.fuser.nhead, num_layers=cfg.fuser.num_layers, dropout=cfg.fuser.dropout, se_ratio=cfg.fuser.se_ratio, bands=cfg.phys.bands)
    fuser.to(device)
    fuser.eval()
    # We'll lazily init a projection based on first file's token D
    proj = None
    for p in tqdm(files, desc="Fusing tokens"):
        d = np.load(p, allow_pickle=True)
        segs = d["seg_imf_tokens"]
        label = int(d["label"])
        fname = str(d["filename"])
        # flatten tokens per file
        all_tokens = np.vstack([seg for seg in segs])
        if all_tokens.size == 0:
            continue
        if proj is None:
            D = all_tokens.shape[1]
            proj = nn.Linear(D, cfg.fuser.d_model).to(device)
            print("Initialized proj:", proj)
        # sample up to a max token count for fuser (or use all)
        max_toks = 32
        N = all_tokens.shape[0]
        if N > max_toks:
            inds = np.linspace(0, N-1, max_toks).astype(int)
            toks = all_tokens[inds]
        else:
            toks = all_tokens
        T = torch.tensor(toks, dtype=torch.float32, device=device).unsqueeze(0)  # (1, N, D)
        Tp = proj(T)  # (1,N,d_model)
        # create dummy F_center (zeros)
        F_center = torch.zeros((1, Tp.shape[1], 1), dtype=torch.float32, device=device)
        with torch.no_grad():
            z, h = fuser(Tp, F_center)  # z: (1,d_model), h: (1,N)
        fused = z.squeeze(0).cpu().numpy()
        attn = h.squeeze(0).cpu().numpy()
        outp = os.path.join(fused_out_dir, f"{fname}_label{label}_fused.npz")
        np.savez(outp, filename=fname, label=label, fused=fused, attn=attn)
    print("Step3 done. Fused representations saved to", fused_out_dir)

# -----------------------------
# Contrastive loss implementation (NT-Xent)
# -----------------------------
def nt_xent_loss(z_i: torch.Tensor, z_j: torch.Tensor, temperature=0.07):
    """
    z_i, z_j: (B, D) representations of positive pairs. We compute contrastive loss for B pairs.
    Returns scalar loss.
    """
    B = z_i.shape[0]
    z = torch.cat([F.normalize(z_i, dim=1), F.normalize(z_j, dim=1)], dim=0)  # (2B, D)
    sim = torch.matmul(z, z.t()) / temperature  # (2B,2B)
    # mask out self-contrast
    labels = torch.arange(B, device=z.device)
    positives = torch.cat([labels + B, labels], dim=0)
    mask = (~torch.eye(2*B, dtype=torch.bool, device=z.device)).float()
    exp_sim = torch.exp(sim) * mask
    denom = exp_sim.sum(dim=1)
    # positive similarities
    pos_sim = torch.exp(torch.sum(F.normalize(z_i, dim=1) * F.normalize(z_j, dim=1), dim=1) / temperature)
    pos = torch.cat([pos_sim, pos_sim], dim=0)
    loss = -torch.log(pos / denom).mean()
    return loss

# -----------------------------
# Step4: Training with contrastive regularization
# -----------------------------
class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim//2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_dim//2, num_classes)
        )
    def forward(self, x):
        return self.net(x)

def train_with_contrastive(token_npz_dir: str, model_save: str, cfg: CFG = CFG):
    """
    Training pipeline:
      - build dataset from token npz files (per-file tokens)
      - dataloader yields batch of files -> sample tokens per file -> fuser -> fused representation z
      - compute classifier loss + contrastive loss (positive pairs produced by augmentation)
    For simplicity we build positive pairs by augmenting tokens within the same file: i) random masking / dropout to produce two views.
    """
    device = cfg.train.device
    files = [os.path.join(token_npz_dir, f) for f in os.listdir(token_npz_dir) if f.endswith("_tokens.npz")]
    ds = IMFTokenDataset(files, cfg, max_tokens_per_file=32)
    # split
    idxs = list(range(len(ds)))
    tr_idx, val_idx = train_test_split(idxs, test_size=0.2, random_state=0, stratify=[ds.data[i]["label"] for i in idxs])
    tr_files = [files[i] for i in tr_idx]
    val_files = [files[i] for i in val_idx]

    tr_ds = IMFTokenDataset(tr_files, cfg, max_tokens_per_file=32)
    val_ds = IMFTokenDataset(val_files, cfg, max_tokens_per_file=32)

    tr_loader = DataLoader(tr_ds, batch_size=cfg.train.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False)

    fuser = IMFTokenFuser(d_model=cfg.fuser.d_model, nhead=cfg.fuser.nhead, num_layers=cfg.fuser.num_layers, dropout=cfg.fuser.dropout, se_ratio=cfg.fuser.se_ratio, bands=cfg.phys.bands).to(device)
    proj = None
    # build classifier after seeing one batch
    num_classes = len(set([d["label"] for d in ds.data]))
    classifier = None

    opt = None
    for epoch in range(cfg.train.epochs):
        fuser.train()
        total_loss = 0.0
        total_ce = 0.0
        total_con = 0.0
        total_samples = 0
        for batch in tr_loader:
            tokens = batch["tokens"].to(device)  # (B, max_tokens, D)
            masks = batch["mask"].to(device)
            labels = batch["label"].to(device)
            B, N, D = tokens.shape
            # lazy init projection & classifier
            if proj is None:
                proj = nn.Linear(D, cfg.fuser.d_model).to(device)
            if classifier is None:
                classifier = SimpleClassifier(cfg.fuser.d_model, num_classes).to(device)
                params = list(fuser.parameters()) + list(proj.parameters()) + list(classifier.parameters())
                opt = torch.optim.Adam(params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

            # create two augmented views: simple dropout mask on tokens (per-sample)
            def augment(x):
                # x: (B, N, D)
                noise = (torch.rand_like(x) > 0.15).float()
                return x * noise
            view1 = augment(tokens)
            view2 = augment(tokens)

            # project
            T1 = proj(view1)  # (B,N,d_model)
            T2 = proj(view2)

            # fuse to per-file representation by averaging fuser output across segments (we use full token list)
            z1, h1 = fuser(T1, None)  # (B, d_model)
            z2, h2 = fuser(T2, None)

            # classification on z1 (use z1 as representation)
            logits = classifier(z1)
            ce_loss = F.cross_entropy(logits, labels)

            # contrastive loss between z1 and z2
            con_loss = nt_xent_loss(z1, z2, temperature=cfg.train.temperature)

            loss = ce_loss + cfg.train.contrastive_lambda * con_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * B
            total_ce += ce_loss.item() * B
            total_con += con_loss.item() * B
            total_samples += B

        avg_loss = total_loss / total_samples
        print(f"Epoch {epoch+1}/{cfg.train.epochs} train loss={avg_loss:.4f} ce={total_ce/total_samples:.4f} con={total_con/total_samples:.4f}")

        # validation
        fuser.eval(); classifier.eval()
        correct = 0; total = 0
        with torch.no_grad():
            for batch in val_loader:
                tokens = batch["tokens"].to(device)
                labels = batch["label"].to(device)
                if proj is None:
                    continue
                T = proj(tokens)
                z, _ = fuser(T, None)
                logits = classifier(z)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.shape[0]
        acc = correct / total if total>0 else 0.0
        print(f" Validation acc={acc:.4f}")

    # save
    os.makedirs(os.path.dirname(model_save), exist_ok=True)
    save_obj = {
        "fuser": fuser.state_dict(),
        "proj": proj.state_dict(),
        "classifier": classifier.state_dict()
    }
    torch.save(save_obj, model_save)
    print("Model saved to", model_save)

# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step2", action="store_true", help="Run step2 token extraction")
    parser.add_argument("--step3", action="store_true", help="Run step3 attention fusion (inference)")
    parser.add_argument("--train", action="store_true", help="Run training with contrastive regularization")
    parser.add_argument("--vmd_dir", type=str, default=CFG.vmd_save_dir)
    parser.add_argument("--token_dir", type=str, default=CFG.feat_npz_dir)
    parser.add_argument("--fused_out", type=str, default="fused")
    parser.add_argument("--model_out", type=str, default="models/model.pth")
    args = parser.parse_args()

    if args.step2:
        step2_imf_token_representation(args.vmd_dir, args.token_dir, CFG)
    if args.step3:
        step3_attention_fusion(args.token_dir, args.fused_out, CFG)
    if args.train:
        train_with_contrastive(args.token_dir, args.model_out, CFG)

if __name__ == "__main__":
    main()
