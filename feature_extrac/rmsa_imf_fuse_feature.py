from adaptive_vmd import step1_vmd_decompose
from extract_features_token import step2_imf_token_representation
from features_fuser import step3_fuse_tokens
from config import CFG
from features_extractors.base import load_npz

def extract_imf_token_features(data_dir, out_dir):
    vmd_dir = out_dir.replace("features", "vmd")
    token_dir = out_dir.replace("features", "tokens")

    step1_vmd_decompose(data_dir, vmd_dir)
    step2_imf_token_representation(vmd_dir, token_dir)
    step3_fuse_tokens(CFG, token_dir, out_dir)

    return load_npz(out_dir)
