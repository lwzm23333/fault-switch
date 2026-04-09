# config.py   调参配置
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class SegmentConfig:
    #fs_target: int = 16000    # 采样率
    fs_target: int = 44100  # 采样率
    seg_len_s: float = 0.5    # 分段窗长
    hop_ratio: float = 0.5    # 重叠
    window: str = "hann"      # 分窗函数
    do_ola: bool = True

@dataclass
class VMDConfig:
    K_max: int = 6
    alpha0: float = 3600
    beta_se: float = 0.4         # α_eff = α0*(1 + beta*(SE - 0.5))
    lambda_c_range: Tuple[float, float] = (0.5, 2.0)
    admm_iters: int = 200
    admm_tol: float = 1e-6
    gamma_bw: float = 4.0        # λ2
    gamma_w_smooth: float = 0.2  # λ3
    gamma_lock: float = 0.5      # λ4
    seed: int = 42

@dataclass
class PostConfig:
    prune_energy_thr: float = 0.001     # 1%
    prune_corr_thr: float = 0.9
    wavelet: str = "db4"
    wavelet_levels: int = 6
    weight_eta: Tuple[float, float, float] = (0.7, 0.2, 0.1)  # (energy, SNR, noise dissimilarity)  原始(0.7,0.2,0.1)


@dataclass
class FeatureConfig:
    mfcc_n: int = 20
    mfcc_use_deltas: bool = True
    wpe_levels: int = 3  # wavelet packet levels
    cyclic_alpha: float = 0.1 # smoothing for pseudo-cyclic estimate

@dataclass
class FuserConfig:
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    se_ratio: int = 8
    contrastive_temp: float = 0.07

@dataclass
class TrainConfig:
    batch_size: int = 32
    epochs: int = 40
    lr: float = 1e-3
    weight_decay: float = 1e-5
    device: str = "cuda"  # or "cpu"

@dataclass
class BOConfig:
    n_calls: int = 25
    cv_folds: int = 5

@dataclass
class PhysConfig:
    # 物理先验频带 (Hz)：基频、啮合频率、谐波邻域
    bands: List[Tuple[float, float]] = ((130, 200),     # 原 (50–70)
        (800, 940),     # 原 (300–340)
        (1650, 1870),   # 原 (600–680)
        (2500, 2800)    # 原 (900–1020)
    )
    # bands: List[Tuple[float, float]] = ((50, 70),  # 原 (50–70)
    #                                     (300, 340),  # 原 (300–340)
    #                                     (600, 680),  # 原 (600–680)
    #                                     (900,1500)  # 原 (900–1020)
    #                                     )

@dataclass
class GlobalConfig:
    seg: SegmentConfig = SegmentConfig()
    vmd: VMDConfig = VMDConfig()
    post: PostConfig = PostConfig()
    feat: FeatureConfig = FeatureConfig()
    fuser: FuserConfig = FuserConfig()
    train: TrainConfig = TrainConfig()
    bayes: BOConfig = BOConfig()
    phys: PhysConfig = PhysConfig()

CFG = GlobalConfig()

class PostConfig:
    prune_energy_thr = 0.01
    prune_corr_thr = 0.9
    wavelet = "db4"
    wavelet_levels = 6
    weight_eta = (0.5, 0.3, 0.2)

    # 新增部分：
    use_fbe = False         # 是否启用 FBE 对比模式
    fbe_top_k = 3           # 保留 FBE 最低的 IMF 数