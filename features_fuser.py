import torch
import os
import torch.nn as nn
import numpy as np
from config import CFG
from models_attn_fuser import IMFTokenFuser

# ======================
# 1. 加载特征文件
# ======================
def load_token_npz(root_dir):
    files = []

    print(f"开始读取 Token NPZ 文件：{root_dir}")
    for cls in sorted(os.listdir(root_dir)):
        cls_path = os.path.join(root_dir, cls)
        if not os.path.isdir(cls_path):
            continue

        for fname in sorted(os.listdir(cls_path)):
            if not fname.endswith("_seqfeat.npz"):
                continue
            fpath = os.path.join(cls_path, fname)
            data = np.load(fpath, allow_pickle=True)

            # 必要的键检查
            if not {"seg_tokens", "seg_fcenters", "label", "filename"}.issubset(set(data.keys())):
                raise KeyError(f"{fpath} 缺少必需的键 (file_token, f_center, label)")

            files.append({
                "path": fpath,
                "seg_tokens": data["seg_tokens"],  # object array
                "seg_fcenters": data["seg_fcenters"],
                "label": int(data["label"]),
                "filename": str(data["filename"])
            })
    if len(files) == 0:
        raise RuntimeError(f"在 {root_dir} 未找到任何 seqfeat .npz 文件")
    print(f"找到 {len(files)} 个序列特征文件")
    return files


# ======================
# 2. 构建fuser
# ======================
def build_fuser(CFG, token_dim: int = 1, state_dict_path=None, device="cpu"):
    """
       构建 IMFTokenFuser，并返回 fuser。注意这里传入 token_dim，以便模型内使用正确的 token projection。
    """
    fuser = IMFTokenFuser(
                        token_dim=token_dim,
                        d_model=CFG.fuser.d_model,
                        nhead=CFG.fuser.nhead,
                        num_layers=CFG.fuser.num_layers,
                        dropout=CFG.fuser.dropout,
                        se_ratio=CFG.fuser.se_ratio,
                        bands=CFG.phys.bands
                        )
    if state_dict_path is not None:
        sd = torch.load(state_dict_path, map_location=device)
        fuser.load_state_dict(sd)
    fuser.to(device)
    fuser.eval()
    return fuser


# ======================
# 3. 单个特征融合
# ======================
def fuse_single_segment(tokens, f_center, fuser, device="cpu"):
    """
    tokens: numpy array (T, D)  -- per-IMF token feature vectors
    f_center: numpy array (T,)    -- per-IMF frequency centers
    proj: nn.Linear(D, d_model) or similar projection module mapping (B,T,D)->(B,T,d_model)
    fuser expects inputs:
        tokens: (B, T, d_model)  (we'll feed T_proj)
        f_center: (B, T, 1)
    返回:
        z_np: numpy array (d_model,)  -- 段级融合特征
        h_att_np: numpy array (T,)    -- token-level attention-like weights (we compute h.mean(dim=-1))
    """
    assert tokens.ndim == 2, f"tokens_np 期望形状 (T, D), 实际 {tokens.shape}"
    T, D = tokens.shape
    # 转为 tensor 并送到 device
    tokens_t = torch.tensor(tokens, dtype=torch.float32, device=device).unsqueeze(0)  # [1, T, D]
    # 投影到 d_model
    with torch.no_grad():
        # T_proj = proj(tokens_t)  # [1, T, d_model]

        F = torch.tensor(f_center, dtype=torch.float32, device=device).view(1, -1, 1)  # [1, T, 1]

        # z, h = fuser(T_proj, F)  # z: (1, d_model), h: (1, T, d_model)
        z, h = fuser(tokens_t, F)

    z_np = z.squeeze(0).cpu().numpy()  # (d_model,)
    # token-level summary: mean over feature dim -> (1, T) -> squeeze -> (T,)
    h_att = h.mean(dim=-1).squeeze(0).cpu().numpy()  # (T,)
    return z_np, h_att

def step3_fuse_tokens(CFG, root_dir, save_path, fuser_state=None, device="cpu"):
    """
        对每个文件（由 step2 产生的 e_segment -> 段级 z
          - 文件级池化：对段级 z 做均值 (可替换为其他策略)
          - 保存 fused_features (N, d_model), attn_weights (object ndarray per-file; each element is list of per-segment arrays), labels

        save_path: 输出 npz 路径
    """
    files = load_token_npz(root_dir)
    os.makedirs(save_path, exist_ok=True)

    token_dim = None
    for f in files:
        segs = f["seg_tokens"]
        if len(segs) > 0:
            first_seg = segs[0]
            if first_seg is not None and first_seg.size != 0:
                # first_seg shape (K, D)
                if first_seg.ndim != 2:
                    raise ValueError(f"第一个 segment token 维度异常: {first_seg.shape}")
                token_dim = first_seg.shape[1]
                break
    if token_dim is None:
        raise RuntimeError("未能从任何文件中推断 token_dim")

    # 构建 fuser 与 proj
    fuser = build_fuser(CFG, token_dim=token_dim, state_dict_path=fuser_state, device=device)
    # --- 新增：如果是第一次运行（没有传入权重），则保存当前的随机权重 ---
    if fuser_state is None:
        # 建议保存到模型统一目录
        auto_save_path = "outputs/models/fuser_weights.pth"
        os.makedirs("outputs/models", exist_ok=True)
        torch.save(fuser.state_dict(), auto_save_path)
        print(f"💾 [重要] 已自动保存初始 Fuser 权重至: {auto_save_path}")
        print("请确保后续测试脚本加载此权重！")
    # proj: Linear(D, d_model) 外部投影
    # proj = torch.nn.Linear(token_dim, CFG.fuser.d_model).to(device)
    # proj.eval()

    fused_list = []
    attn_list = []
    labels = []

    for idx, f in enumerate(files):
        seg_tokens_obj = f["seg_tokens"]  # object array: each element (K_seg, D)
        seg_fcenters_obj = f["seg_fcenters"]  # object array: each element (K_seg,)
        label = f["label"]
        filename = f["filename"]
        stem = filename.replace(f"_label{label}_seqfeat.npz", "")

        # ========== 文件级输出路径（提前构造） ==========
        class_dir = os.path.join(save_path, f"class_{label}")
        os.makedirs(class_dir, exist_ok=True)
        out_file = os.path.join(class_dir, f"{stem}_fuser.npz")

        # ========== 文件级可跳过 ==========
        if os.path.exists(out_file):
            print(f"⚙️ 跳过已存在融合特征: {out_file}")
            continue

        seg_z_list = []
        seg_attn_list = []

        # iterate segments
        for seg_idx in range(len(seg_tokens_obj)):
            toks = seg_tokens_obj[seg_idx]  # (K_seg, D)
            fcs = seg_fcenters_obj[seg_idx]  # (K_seg,)
            if toks is None or toks.size == 0:
                continue
            # Cast to float np arrays to avoid dtype issues
            toks = np.asarray(toks, dtype=float)
            fcs = np.asarray(fcs, dtype=float)

            # Ensure tokens have right feature dim
            if toks.ndim != 2 or toks.shape[1] != token_dim:
                raise ValueError(f"文件 {filename} 段 {seg_idx} token 维度不一致: {toks.shape}, 期望第二维 {token_dim}")

            # z_seg, h_seg = fuse_single_segment(toks, fcs, fuser, device=device)
            # seg_z_list.append(z_seg)  # (d_model,)
            # seg_attn_list.append(h_seg)  # (K_seg,)

            # 临时修改
            # 直接使用 IMF 特征均值，不用 fuser
            seg_feat = toks.mean(axis=0)  # (D,)
            seg_z_list.append(seg_feat)

        if len(seg_z_list) == 0:
            # 若文件没有任何有效段，可跳过或填 0 向量（这里选择跳过并打印警告）
            print(f"⚠ 文件 {filename} 未产生任何段级嵌入，跳过")
            continue

        # 文件级池化：对段级 z 做均值
        file_z = np.mean(np.vstack(seg_z_list), axis=0)  # (d_model,)
        # fused_list.append(file_z)
        # # attn_list 存储每文件的段级 token-attn 列表（不做合并以保留信息）
        # attn_list.append(seg_attn_list)
        # labels.append(label)
        #
        # print(f"[{idx + 1}/{len(files)}] 文件 {filename} 融合完成: segments={len(seg_z_list)}, file_z={file_z.shape}")
        #
        # fused_arr = np.vstack(fused_list)  # (N_files, d_model)
        # # attn_list 是 variable-length per-file; 存为 object ndarray
        # attn_obj = np.array(attn_list, dtype=object)
        # labels_arr = np.array(labels, dtype=int)

        np.savez(out_file, fused_features=file_z.astype(np.float32), labels=np.array(label, dtype=int))

        print(f"📦 saved seq fuser feature: {save_path}")
