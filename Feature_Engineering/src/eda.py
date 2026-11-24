"""STEP 02: 탐색적 데이터 분석 (EDA) — 그림은 reports/figures, 표는 results/ 에 저장."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import plotting
from .data import LEAKAGE_COLS, TARGET, column_table, load_raw
from .features import add_derived_features

NUM_COLS = ["age", "fare", "sibsp", "parch"]


def missing_table(df: pd.DataFrame) -> pd.DataFrame:
    t = pd.DataFrame({
        "결측 수": df.isna().sum(),
        "결측 비율(%)": (df.isna().mean() * 100).round(2),
    })
    return t[t["결측 수"] > 0].sort_values("결측 비율(%)", ascending=False)


def outlier_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in NUM_COLS:
        s = df[col].dropna()
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((s < lo) | (s > hi)).sum())
        rows.append({
            "변수": col, "Q1": round(q1, 2), "Q3": round(q3, 2), "IQR": round(iqr, 2),
            "하한": round(lo, 2), "상한": round(hi, 2), "이상치 수": n_out,
            "이상치 비율(%)": round(n_out / len(s) * 100, 2),
            "최대값": round(s.max(), 2), "왜도": round(s.skew(), 2),
        })
    return pd.DataFrame(rows)


def run() -> None:
    plotting.setup()
    df = load_raw()
    out = plotting.RESULT_DIR

    # ---- STEP 01 출력물 ---------------------------------------------------
    print("데이터 shape:", df.shape)
    column_table(df).to_csv(out / "column_table.csv", index=False)

    # ---- 결측치 -----------------------------------------------------------
    mt = missing_table(df)
    mt.to_csv(out / "missing_table.csv")
    fig, ax = plt.subplots(figsize=(7, 3.6))
    sns.barplot(x=mt["결측 비율(%)"], y=mt.index, ax=ax, color="#c44e52")
    for i, v in enumerate(mt["결측 비율(%)"]):
        ax.text(v + 0.8, i, f"{v:.1f}%", va="center", fontsize=9)
    ax.set_title("컬럼별 결측치 비율 (빨간 글씨 boat/body/home.dest는 누수로 제거)")
    ax.set_xlabel("결측 비율 (%)")
    ax.set_ylabel("")
    ax.set_xlim(0, 105)
    for lbl in ax.get_yticklabels():
        if lbl.get_text() in LEAKAGE_COLS:
            lbl.set_color("#c44e52")
    plotting.save(fig, "01_missing_ratio")

    # ---- 타겟 분포 (Countplot) -----------------------------------------------
    fig, ax = plt.subplots(figsize=(4.5, 3.6))
    sns.countplot(x=TARGET, data=df, ax=ax, hue=TARGET, palette=["#4c72b0", "#dd8452"], legend=False)
    total = len(df)
    for p in ax.patches:
        h = p.get_height()
        ax.annotate(f"{int(h)} ({h / total:.1%})", (p.get_x() + p.get_width() / 2, h),
                    ha="center", va="bottom", fontsize=9)
    ax.set_xticks([0, 1], ["사망(0)", "생존(1)"])
    ax.set_title("타겟(survived) 분포")
    plotting.save(fig, "02_target_countplot")

    # ---- 수치형 분포 (Histogram) ---------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    for ax, col in zip(axes, NUM_COLS):
        sns.histplot(data=df, x=col, hue=TARGET, bins=30, kde=col in ("age", "fare"),
                     ax=ax, multiple="layer", alpha=0.55)
        ax.set_title(f"{col} 분포 (왜도={df[col].skew():.2f})")
    plotting.save(fig, "03_numeric_histograms")

    # ---- 이상치 (Boxplot) ----------------------------------------------------
    ot = outlier_table(df)
    ot.to_csv(out / "outlier_table.csv", index=False)
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    for ax, col in zip(axes, NUM_COLS):
        sns.boxplot(data=df, x=TARGET, y=col, ax=ax, hue=TARGET,
                    palette=["#4c72b0", "#dd8452"], legend=False)
        n = int(ot.loc[ot["변수"] == col, "이상치 수"].iloc[0])
        ax.set_title(f"{col} (IQR 이상치 {n}개)")
        ax.set_xticks([0, 1], ["사망", "생존"])
    plotting.save(fig, "04_boxplots")

    # ---- 범주형 vs 생존 (Countplot / Barplot) -------------------------------
    cats = ["sex", "pclass", "embarked", "sibsp", "parch"]
    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    for i, col in enumerate(cats):
        sns.countplot(data=df, x=col, hue=TARGET, ax=axes[0, i])
        axes[0, i].set_title(f"{col} 빈도 (hue=survived)")
        sns.barplot(data=df, x=col, y=TARGET, ax=axes[1, i], errorbar=("ci", 95), color="#55a868")
        axes[1, i].set_title(f"{col}별 생존율")
        axes[1, i].set_ylim(0, 1.05)
    plotting.save(fig, "05_categorical_count_bar")

    # ---- 파생 변수 미리보기 ---------------------------------------------------
    fe = add_derived_features(df)
    fig, axes = plt.subplots(1, 4, figsize=(18, 3.8))
    for ax, col in zip(axes, ["Title", "FamilySize", "Deck", "AgeGroup"]):
        order = sorted(fe[col].dropna().unique(), key=str)
        sns.barplot(data=fe, x=col, y=TARGET, ax=ax, order=order, color="#8172b3", errorbar=("ci", 95))
        ax.set_title(f"파생변수 {col}별 생존율")
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="x", rotation=30)
    plotting.save(fig, "06_derived_survival")

    # ---- 상관관계 (Heatmap) -------------------------------------------------
    corr_df = fe[["survived", "pclass", "age", "sibsp", "parch", "fare",
                  "FamilySize", "IsAlone", "FarePerPerson", "HasCabin"]].copy()
    corr_df.insert(1, "sex_male", (df["sex"] == "male").astype(int))
    corr = corr_df.corr(numeric_only=True)
    corr.round(3).to_csv(out / "correlation.csv")
    fig, ax = plt.subplots(figsize=(9, 7.5))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1,
                square=True, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("상관관계 Heatmap (Pearson, 파생변수 포함)")
    plotting.save(fig, "07_correlation_heatmap")

    # ---- 요약 출력 -----------------------------------------------------------
    print(mt.to_string())
    print(ot.to_string(index=False))
    print(df[TARGET].value_counts(normalize=True).round(4).to_string())
    print(corr["survived"].sort_values().round(3).to_string())
    print("성별 생존율:", df.groupby("sex")[TARGET].mean().round(3).to_dict())
    print("등급 생존율:", df.groupby("pclass")[TARGET].mean().round(3).to_dict())
    print("Fare=0 승객 수:", int((df["fare"] == 0).sum()))
    print("Title 분포:", fe["Title"].value_counts().to_dict())
    print("Title 생존율:", fe.groupby("Title")[TARGET].mean().round(3).to_dict())
    print("Deck 생존율:", fe.groupby("Deck")[TARGET].mean().round(3).to_dict())
    print("FamilySize 생존율:", fe.groupby("FamilySize")[TARGET].mean().round(3).to_dict())


if __name__ == "__main__":
    run()
