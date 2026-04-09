各文件/函数要点与 I/O 形状
config.py

改采样率、分段窗长/重叠 → SegmentConfig

改 VMD 正则权重、迭代次数 → VMDConfig

剪枝阈、小波基、权重融合系数 → PostConfig

特征维度、物理频带 → FeatureConfig / PhysConfig

注意力模型大小（d_model、nhead、num_layers）→ FuserConfig

dataio.py

list_wavs_with_labels(root) -> List[(path, label)]
目录名到标签映射在函数内：{"normal":0,"loose":1,"abnormal":2,"jam":3}，可按你数据改。

segmentation.py

frame_signal(x, fs, seg_len_s, hop_ratio)
返回：frames:(S,N), idxs:(S,), win:(N,), hop:int

ola_reconstruct(frames, win, hop)
把分段信号用加权重叠相加还原（若你要段级处理后重建时域用它）。

ul_rmsa_vmd.py

estimate_K_by_spectral_peaks(seg, fs, K_max) → int

alpha_from_SE(seg, fs, alpha0, beta) → float

admm_ul_rmsa_vmd(x, fs, K, alpha, lam1, lam2, lam3, lam4, iters, tol)

入：单段时域 x:(N,)；出：U:(K,N), W:(K,), e:(N,)

可替换点：IMF 更新处当前用“投影+平滑”的稳健近似；若追求与原 VMD 更一致，可在此替换为解析解调-重调制与变分子问题的频域更新，接口不变。

cross_segment_frequency_lock(W_list, n_clusters) → 频率簇中心（若你做“跨段对齐”，在段间迭代里调它对齐 W）。

modal_postprocess.py

prune_merge_imfs(U, energy_thr, corr_thr) → 清理后的 U2

wavelet_denoise_imf(u, wavelet, level) → 去噪 IMF

compute_weights(U, e, eta, noise_template) → w:(K,)

eta=(η1,η2,η3) 分别权衡：能量占比、SNR、噪声“反相似”。

reconstruct(U, e, weights, wavelet, level) → (s_hat, U_denoised)

features.py

imf_features(U, fs, phys_bands) → tokens:(K, D_token)

包含：谱熵、Teager、峭度、谱平坦度、滚降、伪循环指标（对物理频带求能量占比与选择性）。

mfcc_features(x, fs, n, use_deltas) → MFCC 特征向量

models/attn_fuser.py

IMFTokenFuser

入：tokens:(B,L,D)（先经线性层把 D_token -> d_model），f_center:(B,L,1)

物理偏置：PhysBandBias 对中心频率做高斯核并投影到 d_model，与 tokens 相加。

出：z:(B,D_model)（段/样本级向量）、h:(B,L,D_model)（token 隐状态，可视化注意力/权重）。

models/classifiers.py

SVMRBF：.fit(X,y), .predict(X), .predict_proba(X)

LGBM：LightGBM 对照

SmallCNN：如需端到端对时域或时频图训练的 baseline

training/contrastive.py

InfoNCELoss(temperature)

入：z:(B,D)；pos_idx:(B,) 指定每个样本的正对索引（同工况不同噪声）

用于预训练融合器：构造 batch，把同类不同噪声成对。

training/train_classifier.py

crossval_train_eval(X, y, model_name="svm", n_splits=5)

返回 Macro-F1、AURC 的均值。

若要做贝叶斯优化，在外层循环里调它并把得分送进优化器。

utils/metrics.py

macro_f1、aurc 与 bo_objective（F1 − AURC）

eval/explain.py

shap_explain(model, X, feature_names=None)：画 SHAP beeswarm（SVM 使用 predict_proba 解释）

show_attention_heatmap(attn_tokens_matrix)：可把 h.mean(-1) 喂进去

show_phys_mapping(W, bands)：画 IMF 中心频率与物理频带对应