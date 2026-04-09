# eval/explain.py
import numpy as np
import shap
import matplotlib.pyplot as plt
from utils.plotting import plot_attention_heatmap, plot_phys_mapping

def shap_explain(model, X, feature_names=None, max_display=20):
    explainer = shap.Explainer(model.clf.predict_proba, X)
    shap_values = explainer(X)
    shap.plots.beeswarm(shap_values, max_display=max_display, show=False)
    plt.tight_layout()

def show_attention_heatmap(attn_tokens_matrix):
    # attn_tokens_matrix: (L, D) 或 (L, heads) 预先聚合
    plot_attention_heatmap(attn_tokens_matrix)

def show_phys_mapping(W, bands):
    plot_phys_mapping(W, bands)
