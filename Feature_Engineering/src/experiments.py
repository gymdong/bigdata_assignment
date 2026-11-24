"""STEP 03~05 + 실험 비교: 필수 비교표, 단일 요인 ablation, 변수 선택 실험.

평가 방식
- Stratified 80/20 hold-out (test는 마지막 평가에만 사용)
- train 안에서 Stratified 5-fold CV → 평균 지표(주 비교 기준)
- 과적합 지표: CV train accuracy - CV validation accuracy
"""
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from . import plotting
from .data import RANDOM_STATE, load_xy
from .pipelines import EXPERIMENTS, FEConfig, build_base_pipeline, build_pipeline, get_models

SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
DROP_NA_SUBSET = ["age", "fare", "embarked"]

# ablation 기준 설정: 한 번에 한 요인만 바꾼다
REFERENCE = FEConfig(impute="median", encoding="onehot", scaling="standard", derived=True, fs_k=None)


def split():
    X, y = load_xy()
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)


def evaluate(pipe, X_tr, y_tr, X_te, y_te) -> dict:
    cv = cross_validate(pipe, X_tr, y_tr, cv=CV, scoring=SCORING, return_train_score=True, n_jobs=1)
    fitted = clone(pipe).fit(X_tr, y_tr)
    proba = fitted.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    res = {f"cv_{m}": cv[f"test_{m}"].mean() for m in SCORING}
    res["cv_auc_std"] = cv["test_roc_auc"].std()
    res["cv_train_acc"] = cv["train_accuracy"].mean()
    res["overfit_gap"] = res["cv_train_acc"] - res["cv_accuracy"]
    res.update({
        "test_accuracy": accuracy_score(y_te, pred),
        "test_precision": precision_score(y_te, pred),
        "test_recall": recall_score(y_te, pred),
        "test_f1": f1_score(y_te, pred),
        "test_roc_auc": roc_auc_score(y_te, proba),
        "n_features": _n_features(fitted),
        "n_train": len(X_tr),
        "n_test": len(X_te),
    })
    return res


def _n_features(fitted) -> int:
    model = fitted.named_steps["model"]
    return int(getattr(model, "n_features_in_", -1))


def run_config(cfg: FEConfig, data, models=None) -> list[dict]:
    X_tr, X_te, y_tr, y_te = data
    if cfg.impute == "drop":
        keep_tr = X_tr[DROP_NA_SUBSET].notna().all(axis=1)
        keep_te = X_te[DROP_NA_SUBSET].notna().all(axis=1)
        X_tr, y_tr, X_te, y_te = X_tr[keep_tr], y_tr[keep_tr], X_te[keep_te], y_te[keep_te]
    rows = []
    for name, model in (models or get_models()).items():
        res = evaluate(build_pipeline(cfg, model), X_tr, y_tr, X_te, y_te)
        rows.append({"model": name, **res})
    return rows


def config_cols(cfg: FEConfig) -> dict:
    return {
        "결측치": cfg.impute, "인코딩": cfg.encoding, "스케일링": cfg.scaling,
        "파생변수": "O" if cfg.derived else "X",
        "FS": f"O (top {cfg.fs_k}, {cfg.fs_method})" if cfg.fs_k else "X",
    }


# ---------------------------------------------------------------------------
def main_comparison(data) -> pd.DataFrame:
    """과제 5장 필수 비교표: Base / Exp-1 / Exp-2 / Exp-3."""
    X_tr, X_te, y_tr, y_te = data
    rows = []
    for name, model in get_models().items():
        res = evaluate(build_base_pipeline(name, model), X_tr, y_tr, X_te, y_te)
        rows.append({"실험": "Base", "결측치": "없음", "인코딩": "없음", "스케일링": "없음",
                     "파생변수": "X", "FS": "X", "model": name, **res})
    for exp, cfg in EXPERIMENTS.items():
        for r in run_config(cfg, data):
            rows.append({"실험": exp, **config_cols(cfg), **r})
    return pd.DataFrame(rows)


def ablations(data) -> pd.DataFrame:
    """REFERENCE에서 한 요인씩만 바꾸는 단일 요인 실험."""
    grid = {
        "결측치 처리": [REFERENCE.with_(impute=v) for v in ["mean", "median", "most_frequent", "drop"]],
        "인코딩": [REFERENCE.with_(encoding=v) for v in ["onehot", "label"]],
        "스케일링": [REFERENCE.with_(scaling=v) for v in ["none", "standard", "minmax", "robust"]],
        "파생변수": [REFERENCE.with_(derived=v) for v in [False, True]],
    }
    rows = []
    for factor, cfgs in grid.items():
        for cfg in cfgs:
            level = {"결측치 처리": cfg.impute, "인코딩": cfg.encoding, "스케일링": cfg.scaling,
                     "파생변수": "사용" if cfg.derived else "미사용"}[factor]
            for r in run_config(cfg, data):
                rows.append({"factor": factor, "level": level, **r})
    return pd.DataFrame(rows)


def feature_selection(data) -> pd.DataFrame:
    """변수 수(k) 곡선 + 선택 방법(RF 중요도 / RFE / SelectKBest) 비교."""
    rows = []
    n_all = 32  # REFERENCE(one-hot + 파생변수) 전처리 후 전체 변수 수
    for k in [3, 5, 8, 10, 12, 15, 20, None]:
        cfg = REFERENCE.with_(fs_k=k, fs_method="rf")
        for r in run_config(cfg, data):
            rows.append({"method": "rf", "k": k or n_all, **r})
    for method in ["rfe", "kbest"]:
        cfg = REFERENCE.with_(fs_k=10, fs_method=method)
        for r in run_config(cfg, data):
            rows.append({"method": method, "k": 10, **r})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def plot_main(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    for ax, metric, title in [(axes[0], "cv_roc_auc", "5-fold CV ROC-AUC"),
                              (axes[1], "test_f1", "Test F1-score")]:
        sns.barplot(data=df, x="실험", y=metric, hue="model", ax=ax)
        lo = df[metric].min()
        ax.set_ylim(max(0, lo - 0.08), min(1, df[metric].max() + 0.03))
        ax.set_title(f"실험별 성능 비교 — {title}")
        ax.legend(fontsize=8, loc="lower right")
    plotting.save(fig, "10_main_comparison")


def plot_ablation(df: pd.DataFrame) -> None:
    factors = df["factor"].unique()
    fig, axes = plt.subplots(1, len(factors), figsize=(5 * len(factors), 4.3), sharey=True)
    for ax, factor in zip(axes, factors):
        sub = df[df["factor"] == factor]
        sns.pointplot(data=sub, x="level", y="cv_roc_auc", hue="model", ax=ax, dodge=0.3,
                      markers="o", linestyles="-")
        ax.set_title(f"[{factor}] 변경 시 CV ROC-AUC")
        ax.set_xlabel("")
        ax.legend(fontsize=7, loc="lower right")
    plotting.save(fig, "11_ablation")


def plot_fs(df: pd.DataFrame) -> None:
    rf = df[df["method"] == "rf"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
    sns.lineplot(data=rf, x="k", y="cv_roc_auc", hue="model", marker="o", ax=axes[0])
    axes[0].set_title("선택 변수 수(k)에 따른 CV ROC-AUC (RF 중요도 기반)")
    sns.lineplot(data=rf, x="k", y="overfit_gap", hue="model", marker="o", ax=axes[1])
    axes[1].set_title("선택 변수 수(k)에 따른 과적합 gap (train acc - CV acc)")
    for ax in axes:
        ax.set_xlabel("선택 변수 수 k (32 = 전체)")
    plotting.save(fig, "12_feature_selection_k")


def run() -> None:
    plotting.setup()
    out = plotting.RESULT_DIR
    data = split()
    print("train/test:", data[0].shape, data[1].shape)

    t = time.time()
    main = main_comparison(data)
    main.to_csv(out / "main_comparison.csv", index=False)
    plot_main(main)
    print(f"[main] {time.time() - t:.0f}s")
    cols = ["실험", "model", "cv_accuracy", "cv_f1", "cv_roc_auc", "overfit_gap",
            "test_accuracy", "test_precision", "test_recall", "test_f1", "test_roc_auc", "n_features"]
    print(main[cols].round(4).to_string(index=False))

    t = time.time()
    abl = ablations(data)
    abl.to_csv(out / "ablation.csv", index=False)
    plot_ablation(abl)
    print(f"[ablation] {time.time() - t:.0f}s")
    print(abl.pivot_table(index=["factor", "level"], columns="model", values="cv_roc_auc").round(4).to_string())
    print(abl.pivot_table(index=["factor", "level"], columns="model", values="cv_accuracy").round(4).to_string())

    t = time.time()
    fs = feature_selection(data)
    fs.to_csv(out / "feature_selection.csv", index=False)
    plot_fs(fs)
    print(f"[fs] {time.time() - t:.0f}s")
    print(fs.pivot_table(index=["method", "k"], columns="model", values="cv_roc_auc").round(4).to_string())
    print(fs.pivot_table(index=["method", "k"], columns="model", values="overfit_gap").round(4).to_string())


if __name__ == "__main__":
    run()
