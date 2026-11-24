"""가산점: GridSearchCV(전처리+모델 동시 탐색), 성능 곡선, SHAP, Feature Importance 비교."""
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import RobustScaler, StandardScaler

from . import plotting
from .data import RANDOM_STATE
from .experiments import CV, REFERENCE, evaluate, split
from .pipelines import build_pipeline, get_models

SCALER_CHOICES = [StandardScaler(), RobustScaler(), "passthrough"]
PREP_GRID = {"prep__num__imputer__strategy": ["mean", "median"]}

PARAM_GRIDS = {
    "LogisticRegression": {"model__C": [0.03, 0.1, 0.3, 1, 3, 10],
                           "prep__num__scaler": SCALER_CHOICES},
    "RandomForest": {"model__max_depth": [4, 6, 8, None],
                     "model__min_samples_leaf": [1, 3, 5],
                     "model__max_features": ["sqrt", 0.5]},
    "XGBoost": {"model__max_depth": [2, 3, 4, 5],
                "model__learning_rate": [0.03, 0.1],
                "model__n_estimators": [200, 400],
                "model__min_child_weight": [1, 3]},
    "LightGBM": {"model__num_leaves": [7, 15, 31],
                 "model__learning_rate": [0.03, 0.1],
                 "model__n_estimators": [200, 400],
                 "model__min_child_samples": [10, 20, 40]},
}


def _single_thread(model):
    """GridSearchCV가 병렬화를 맡으므로 모델 내부 스레드는 1로 고정(과다 구독 방지)."""
    if "n_jobs" in model.get_params():
        model.set_params(n_jobs=1)
    return model


def tune(data) -> tuple[pd.DataFrame, dict]:
    X_tr, X_te, y_tr, y_te = data
    rows, best = [], {}
    for name, model in get_models().items():
        pipe = build_pipeline(REFERENCE, _single_thread(model))
        default = evaluate(pipe, X_tr, y_tr, X_te, y_te)
        t = time.time()
        gs = GridSearchCV(pipe, {**PREP_GRID, **PARAM_GRIDS[name]}, scoring="roc_auc",
                          cv=CV, n_jobs=-1, refit=True)
        gs.fit(X_tr, y_tr)
        tuned = evaluate(gs.best_estimator_, X_tr, y_tr, X_te, y_te)
        best[name] = gs.best_estimator_
        params = {k: (type(v).__name__ if not isinstance(v, (str, int, float, type(None))) else v)
                  for k, v in gs.best_params_.items()}
        print(f"[{name}] {time.time() - t:.0f}s, {len(gs.cv_results_['params'])} 조합, best={params}")
        for tag, res in [("default", default), ("tuned", tuned)]:
            rows.append({"model": name, "setting": tag,
                         "best_params": str(params) if tag == "tuned" else "", **res})
    return pd.DataFrame(rows), best


def plot_roc_cm(best: dict, data, best_name: str) -> None:
    _, X_te, _, y_te = data
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, est in best.items():
        RocCurveDisplay.from_estimator(est, X_te, y_te, ax=ax, name=name)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_title("튜닝된 모델 ROC Curve (Test)")
    plotting.save(fig, "13_roc_curves")

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_estimator(best[best_name], X_te, y_te, ax=ax, cmap="Blues",
                                          display_labels=["사망", "생존"], colorbar=False)
    ax.set_title(f"Confusion Matrix — {best_name} (tuned)")
    plotting.save(fig, "14_confusion_matrix")


def rf_importance_selection(data, top_n: int = 10) -> pd.Series:
    """STEP 04: RF Feature Importance로 상위 N개 변수 선정 (REFERENCE 전처리 후 32개 중)."""
    X_tr, _, y_tr, _ = data
    pipe = build_pipeline(REFERENCE, RandomForestClassifier(n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1))
    pipe.fit(X_tr, y_tr)
    imp = pd.Series(pipe.named_steps["model"].feature_importances_,
                    index=pipe[:-1].get_feature_names_out()).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = ["#c44e52" if i < top_n else "#b0b0b0" for i in range(len(imp))]
    ax.barh(imp.index[::-1], imp.values[::-1], color=colors[::-1])
    ax.set_title(f"Random Forest Feature Importance — 빨간색: 상위 {top_n}개 선택")
    ax.set_xlabel("Mean decrease in impurity")
    plotting.save(fig, "15_rf_importance_topN")
    return imp


def explain(est, data, name: str) -> pd.DataFrame:
    """SHAP + 3가지 중요도(impurity / permutation / |SHAP|) 비교."""
    _, X_te, _, y_te = data
    prep, model = est[:-1], est[-1]
    Xt = prep.transform(X_te)
    feat = list(Xt.columns)

    explainer = shap.TreeExplainer(model)
    sv = explainer(Xt)
    vals = sv.values if sv.values.ndim == 2 else sv.values[:, :, 1]
    sv_pos = shap.Explanation(vals, base_values=np.ravel(sv.base_values)[: len(Xt)] if np.ndim(sv.base_values) else sv.base_values,
                              data=Xt.values, feature_names=feat)

    plt.figure()
    shap.plots.beeswarm(sv_pos, max_display=15, show=False)
    plt.title(f"SHAP Summary (beeswarm) — {name}")
    plotting.save(plt.gcf(), "16_shap_beeswarm")

    plt.figure()
    shap.plots.bar(sv_pos, max_display=15, show=False)
    plt.title(f"SHAP 평균 |값| — {name}")
    plotting.save(plt.gcf(), "17_shap_bar")

    mean_abs = pd.Series(np.abs(vals).mean(axis=0), index=feat)
    top3 = mean_abs.sort_values(ascending=False).index[:3]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, f in zip(axes, top3):
        shap.plots.scatter(sv_pos[:, f], color=sv_pos, ax=ax, show=False)
        ax.set_title(f"SHAP dependence — {f}")
    plotting.save(fig, "18_shap_dependence")

    perm = permutation_importance(model, Xt, y_te, scoring="roc_auc", n_repeats=20,
                                  random_state=RANDOM_STATE, n_jobs=1)
    imp = pd.DataFrame({
        "impurity/gain": pd.Series(model.feature_importances_, index=feat),
        "permutation (ΔAUC)": pd.Series(perm.importances_mean, index=feat),
        "mean |SHAP|": mean_abs,
    })
    norm = imp.clip(lower=0) / imp.clip(lower=0).sum()
    order = norm.mean(axis=1).sort_values(ascending=False).index[:15]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1.4, 1]})
    norm.loc[order].iloc[::-1].plot.barh(ax=axes[0], width=0.8)
    axes[0].set_title(f"3가지 Feature Importance 비교 (정규화, 상위 15) — {name}")
    ranks = imp.rank(ascending=False).loc[order]
    sns.heatmap(ranks, annot=True, fmt=".0f", cmap="YlGnBu_r", ax=axes[1], cbar=False)
    axes[1].set_title("방법별 순위 (1 = 가장 중요)")
    plotting.save(fig, "19_importance_comparison")
    return imp


def run() -> None:
    plotting.setup()
    out = plotting.RESULT_DIR
    data = split()

    imp_rf = rf_importance_selection(data)
    imp_rf.to_csv(out / "rf_importance.csv", header=["importance"])
    print("RF 중요도 상위 10:", imp_rf.head(10).round(4).to_dict())

    res, best = tune(data)
    res.to_csv(out / "tuning.csv", index=False)
    cols = ["model", "setting", "cv_roc_auc", "cv_accuracy", "overfit_gap", "test_accuracy",
            "test_precision", "test_recall", "test_f1", "test_roc_auc"]
    print(res[cols].round(4).to_string(index=False))

    tuned = res[res["setting"] == "tuned"].set_index("model")
    best_name = tuned["cv_roc_auc"].idxmax()
    print("CV ROC-AUC 기준 최종 모델:", best_name)
    plot_roc_cm(best, data, best_name)

    tree_tuned = tuned.drop(index="LogisticRegression")
    shap_name = tree_tuned["cv_roc_auc"].idxmax()
    imp = explain(best[shap_name], data, shap_name)
    imp.round(4).to_csv(out / "importance_comparison.csv")
    print(f"SHAP 대상: {shap_name}")
    print(imp.sort_values("mean |SHAP|", ascending=False).head(12).round(4).to_string())


if __name__ == "__main__":
    run()
