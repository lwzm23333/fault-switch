# models/classifiers.py
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import f1_score, roc_auc_score
from lightgbm import LGBMClassifier
import torch
import torch.nn as nn

class SVMRBF:
    def __init__(self, C=10.0, gamma='scale'):
        self.clf = SVC(C=C, gamma=gamma, probability=True)

    def fit(self, X, y):
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)

class LGBM:
    def __init__(self):
        self.clf = LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=-1)

    def fit(self, X, y):
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)

class SmallCNN(nn.Module):
    def __init__(self, in_ch=1, n_cls=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 16, 9, padding=4), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 9, padding=4), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 9, padding=4), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(64, n_cls)

    def forward(self, x):
        # x: (B,1,T)
        h = self.net(x).squeeze(-1)
        return self.fc(h)
