from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    balanced_accuracy_score,
    recall_score
)

def build_model(model_name: str, random_state=42):
    model_name = model_name.lower()

    if model_name == "svm":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                kernel="rbf",
                C=10,
                gamma="scale",
                probability=True,
                class_weight="balanced",
                random_state=random_state
            ))
        ])


    elif model_name == "rf":
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state
        )

    elif model_name == "mlp":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            max_iter=800,
            early_stopping=True,
            n_iter_no_change=20,
            random_state=42
            ))
        ])

    elif model_name == "knn":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(
                n_neighbors=7,
                weights="distance"
            ))
        ])

    else:
        raise ValueError(f"未知模型类型: {model_name}")



def crossval_train_eval(
    X,
    y,
    model_name="svm",
    n_splits=5,
    return_preds=False
):
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )

    f1_macro_scores = []
    bal_acc_scores = []
    recall_macro_scores = []
    auc_scores = []

    y_true_all, y_pred_all = [], []

    for train_idx, test_idx in skf.split(X, y):
        clf = build_model(model_name)

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # ===== 核心指标 =====
        f1_macro_scores.append(
            f1_score(y_test, y_pred, average="macro", zero_division=0)
        )

        bal_acc_scores.append(
            balanced_accuracy_score(y_test, y_pred)
        )

        recall_macro_scores.append(
            recall_score(y_test, y_pred, average="macro", zero_division=0)
        )

        # ===== AUC（辅助指标）=====
        if hasattr(clf, "predict_proba"):
            try:
                y_prob = clf.predict_proba(X_test)
                auc_scores.append(
                    roc_auc_score(
                        y_test,
                        y_prob,
                        multi_class="ovo",
                        average="macro"
                    )
                )
            except Exception:
                auc_scores.append(np.nan)
        else:
            auc_scores.append(np.nan)

        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)

    results = {
        "f1_macro": np.nanmean(f1_macro_scores),
        "balanced_accuracy": np.nanmean(bal_acc_scores),
        "recall_macro": np.nanmean(recall_macro_scores),
        "auc_macro": np.nanmean(auc_scores)
    }

    if return_preds:
        return (
            results,
            np.array(y_true_all),
            np.array(y_pred_all),
            clf
        )
    else:
        return results


