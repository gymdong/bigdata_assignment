"""results/*.csv → reports/figures/*.png"""
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import FIG_DIR, RESULTS_DIR
from .ground_truth import load_summary

plt.rcParams.update({
    "font.family": ["DejaVu Sans", "AppleGothic"], "axes.unicode_minus": False, "mathtext.fontset": "dejavusans",
    "figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 9.5, "legend.fontsize": 8, "legend.frameon": False,
})
C = ["#2a6fdb", "#e0663a", "#2e9e6a", "#9b59b6", "#c9a227", "#d64b7f", "#555555", "#17a2b8"]


def _r(name):
    return pd.read_csv(RESULTS_DIR / name)


def _save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, bbox_inches="tight")
    plt.close(fig)


def fig_dataset():
    mc = _r("movie_counts.csv")
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    ax[0].loglog(np.arange(1, len(mc) + 1), mc["count"], color=C[0])
    ax[0].set(xlabel="영화 순위 (빈도 내림차순)", ylabel="평점 수", title="영화별 평점 수 — 롱테일 분포")
    ax[1].hist(mc["count"], bins=np.logspace(0, math.log10(mc["count"].max()), 40), color=C[0], alpha=.8)
    ax[1].set_xscale("log")
    ax[1].set(xlabel="영화별 평점 수 (log)", ylabel="영화 수", title="빈도 히스토그램")
    _save(fig, "00_dataset_distribution.png")


# ------------------------------------------------------------------ Bloom
def fig_bloom():
    df = _r("bloom_sweep.csv")
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.8))
    for i, (bpe, g) in enumerate(df.groupby("bpe")):
        g = g.sort_values("k")
        ax[0].semilogy(g["k"], g["fpr"].clip(lower=1e-5), "o", color=C[i], ms=4, label=f"m/n={bpe}")
        ax[0].semilogy(g["k"], g["fpr_theory"], "-", color=C[i], lw=1, alpha=.7)
        best = g.loc[g["fpr"].idxmin()]
        ax[0].plot(best["k"], max(best["fpr"], 1e-5), "*", color=C[i], ms=11, mec="k", mew=.5)
    ax[0].set(xlabel="해시 함수 개수 k", ylabel="False Positive Rate (log)",
              title="FPR vs k (점=실측, 선=이론, ★=최소)")
    ax[0].legend(ncol=2)

    opt = df.loc[df.groupby("bpe")["fpr"].idxmin()]
    set_mb = load_summary()["exact_memory"]["set_pairs"]["current_bytes"] / 2**20
    ax[1].loglog(opt["bytes"] / 2**20, opt["fpr"].clip(lower=1e-5), "o-", color=C[0], label="Bloom (최적 k)")
    for _, r in opt.iterrows():
        ax[1].annotate(f"{int(r.bpe)}b/k={int(r.k)}", (r.bytes / 2**20, max(r.fpr, 1e-5)),
                       textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax[1].axvline(set_mb, color=C[1], ls="--")
    ax[1].text(set_mb * .97, 0.2, f"정확한 set\n{set_mb:.1f} MB", ha="right", color=C[1], fontsize=8)
    ax[1].set(xlabel="메모리 (MB, log)", ylabel="FPR (log)", title="메모리–정확도 trade-off", xlim=(0.15, 200))
    _save(fig, "10_bloom_fpr_vs_k.png")

    cv = _r("bloom_capacity_curve.csv")
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    for i, (lab, g) in enumerate(cv.groupby("design")):
        ax.plot(g["inserted"] / 1e3, g["fpr"] * 100, "o", ms=3, color=C[i], label=f"{lab} 실측")
        ax.plot(g["inserted"] / 1e3, g["fpr_theory"] * 100, "-", color=C[i], lw=1)
    ax.axvline(250, color="grey", ls=":")
    ax.text(255, 50, "n/4 설계 용량 초과", fontsize=8, color="grey")
    ax.set(xlabel="삽입한 원소 수 (천)", ylabel="FPR (%)", title="스트림 진행에 따른 FPR (m/n=10 설계, 점=실측 · 선=이론)")
    ax.legend()
    _save(fig, "11_bloom_capacity_curve.png")


# ------------------------------------------------------------------ CMS
def fig_cms():
    df = _r("cms_sweep.csv")
    N = load_summary()["n_records"]
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.8), sharey=True)
    for j, cons in enumerate((False, True)):
        g0 = df[df.conservative == cons]
        for i, (d, g) in enumerate(g0.groupby("d")):
            g = g.sort_values("w")
            ax[j].loglog(g["w"], g["mae"].clip(lower=0.01), "o-", ms=3, color=C[i], label=f"d={d}")
        ws = np.array(sorted(df.w.unique()))
        ax[j].loglog(ws, math.e / ws * N, "k--", lw=1, label="εN 상한 (e/w·N)")
        ax[j].set(xlabel="width w", title=("Conservative Update" if cons else "기본 CMS") + " — 평균 절대오차")
    ax[0].set_ylabel("평균 절대오차 (log)")
    ax[1].legend(ncol=2)
    _save(fig, "20_cms_error_vs_width.png")

    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.6))
    for j, cons in enumerate((False, True)):
        pv = df[df.conservative == cons].pivot(index="d", columns="w", values="mean_rel_err")
        im = ax[j].imshow(np.log10(pv.values.clip(min=1e-4)), cmap="viridis_r", aspect="auto")
        ax[j].set_xticks(range(len(pv.columns)), pv.columns, rotation=45)
        ax[j].set_yticks(range(len(pv.index)), pv.index)
        for (a, b), v in np.ndenumerate(pv.values):
            ax[j].text(b, a, f"{v:,.0f}" if v >= 10 else f"{v:.2g}", ha="center", va="center", fontsize=6.5,
                       color="white" if v > 3 else "black")
        ax[j].grid(False)
        ax[j].set(xlabel="w", ylabel="d", title=("CU" if cons else "기본") + " — 평균 상대오차 (배수, 1 = 100%)")
        fig.colorbar(im, ax=ax[j], label="log10(상대오차)")
    _save(fig, "21_cms_rel_err_heatmap.png")

    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    for i, budget in enumerate((1024, 4096)):
        for cons, ls in ((False, "-"), (True, "--")):
            g = df[(df.counters == budget) & (df.conservative == cons)].sort_values("d")
            ax.semilogy(g["d"], g["mae"].clip(lower=.01), "o" + ls, color=C[i], ms=4,
                        label=f"w·d={budget}{' CU' if cons else ''}")
    ax.set(xlabel="depth d (w = 예산/d)", ylabel="평균 절대오차 (log)", title="같은 메모리에서 width vs depth 배분")
    ax.legend()
    _save(fig, "22_cms_equal_memory.png")

    b = _r("cms_buckets.csv")
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    x = np.arange(b.bucket.nunique())
    for i, cons in enumerate((False, True)):
        g = b[b.conservative == cons]
        bars = ax.bar(x + (i - .5) * .38, g["mean_rel_err"] * 100, .38, color=C[i], label="CU" if cons else "기본")
        ax.bar_label(bars, fmt="%.1f%%", fontsize=7)
    ax.set_xticks(x, [f"{bk}\n({n}편)" for bk, n in zip(b.bucket.unique(), b[b.conservative == False].n_items)])
    ax.set_yscale("log")
    ax.set(xlabel="실제 빈도 구간", ylabel="평균 상대오차 (%, log)", title="빈도 구간별 상대오차 (w=272, d=5)")
    ax.legend()
    _save(fig, "23_cms_error_by_bucket.png")


# ------------------------------------------------------------------ HLL
def _hll_agg():
    df = _r("hll_sweep.csv")
    truth = load_summary()
    tv = {"users": truth["n_users"], "movies": truth["n_movies"], "pairs": truth["n_unique_pairs"]}
    df["true"] = df.target.map(tv)
    df["hll_rel"] = (df.hll_est - df.true) / df.true
    df["fm_rel"] = (df.fm_est - df.true) / df.true
    return df


def fig_hll():
    df = _hll_agg()
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.8))
    lab = {"users": "사용자 (6,040)", "movies": "영화 (3,706)", "pairs": "(user,movie) 쌍 (1,000,209)"}
    for i, (t, g) in enumerate(df.groupby("target", sort=False)):
        a = g.groupby("m").agg(rmse=("hll_rel", lambda s: np.sqrt((s ** 2).mean())))
        ax[0].loglog(a.index, a.rmse * 100, "o-", color=C[i], ms=4, label=lab[t])
    ms = np.array(sorted(df.m.unique()))
    ax[0].loglog(ms, 104 / np.sqrt(ms), "k--", lw=1, label="이론 1.04/√m")
    ax[0].set(xlabel="버킷 수 m = 2^p", ylabel="RMSE 상대오차 (%, 20 seeds)", title="HyperLogLog 정확도 vs 버킷 수")
    ax[0].legend()

    g = df[df.target == "pairs"].groupby("m").agg(
        hll=("hll_rel", lambda s: np.sqrt((s ** 2).mean())), fm=("fm_rel", lambda s: np.sqrt((s ** 2).mean())),
        hb=("hll_packed_bytes", "first"), fb=("fm_bytes", "first"))
    ax[1].loglog(g.hb, g.hll * 100, "o-", color=C[0], label="HLL (6bit 레지스터)")
    ax[1].loglog(g.fb, g.fm * 100, "s-", color=C[1], label="FM/PCSA (32bit 비트맵)")
    ax[1].loglog(g.hb, 104 / np.sqrt(g.hb * 8 / 6), "--", color=C[0], lw=1, label="HLL 이론 1.04/√m")
    ax[1].loglog(g.fb, 78 / np.sqrt(g.fb / 4), "--", color=C[1], lw=1, label="FM 이론 0.78/√m")
    ax[1].set(xlabel="메모리 (bytes, log)", ylabel="RMSE 상대오차 (%)", title="고유 쌍 수: HLL vs Flajolet–Martin")
    ax[1].legend()
    _save(fig, "30_hll_error_vs_m.png")

    cv = _r("hll_curve.csv")
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.4))
    for i, (p, g) in enumerate(cv.groupby("p")):
        ax[0].plot(g.processed / 1e3, g.est / 1e3, color=C[i], label=f"HLL p={p} (m={1 << p})")
        ax[1].plot(g.processed / 1e3, (g.est - g.true) / g.true * 100, color=C[i], label=f"p={p}")
        ax[1].axhspan(-104 / math.sqrt(1 << p), 104 / math.sqrt(1 << p), color=C[i], alpha=.08)
    ax[0].plot(g.processed / 1e3, g.true / 1e3, "k--", lw=1, label="실제 고유 수")
    ax[0].set(xlabel="처리 레코드 (천)", ylabel="고유 쌍 수 (천)", title="스트림 진행에 따른 추정")
    ax[1].axhline(0, color="k", lw=.8)
    ax[1].set(xlabel="처리 레코드 (천)", ylabel="상대오차 (%)", title="상대오차 (음영 = ±1σ 이론)")
    ax[0].legend(); ax[1].legend()
    _save(fig, "31_hll_stream_curve.png")


# ------------------------------------------------------------------ 메모리·시간
def fig_memory_time():
    m = _r("memory.csv")
    fig, ax = plt.subplots(figsize=(7, 3.4))
    y = np.arange(len(m))
    ax.barh(y + .2, m.struct_bytes, .38, color=C[0], label="스케치 (자료구조 크기)")
    ax.barh(y - .2, m.exact_current, .38, color=C[1], label="정확한 set/dict (tracemalloc)")
    for i, r in m.iterrows():
        ax.text(r.struct_bytes * 1.15, i + .2, _fmt_b(r.struct_bytes), va="center", fontsize=7.5)
        ax.text(r.exact_current * 1.15, i - .2, f"{_fmt_b(r.exact_current)}  (×{r.exact_current / r.struct_bytes:,.0f})",
                va="center", fontsize=7.5)
    ax.set_xscale("log")
    ax.set_yticks(y, m.algo)
    ax.set_xlim(right=m.exact_current.max() * 30)
    ax.set(xlabel="bytes (log)", title="메모리 사용량: 스케치 vs 정확한 자료구조")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=2)
    _save(fig, "40_memory.png")

    t = _r("timing_main.csv")
    fig, ax = plt.subplots(figsize=(7, 3.6))
    colors = [C[6] if "Parse" in s else (C[1] if "Exact" in s else C[0]) for s in t.method]
    bars = ax.barh(t.method, t.throughput / 1e3, color=colors)
    ax.bar_label(bars, labels=[f"{v / 1e3:,.0f}K/s ({s:.2f}s)" for v, s in zip(t.throughput, t.total_s)], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlim(right=t.throughput.max() / 1e3 * 1.35)
    ax.set(xlabel="처리량 (천 레코드/초, 파싱 포함)", title="전체 스트림(1,000,209건) 처리 시간")
    _save(fig, "41_throughput.png")

    ts = _r("timing_sweep.csv")
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
    g = ts[ts.algo == "bloom"]
    ax[0].plot(g.value, g.algo_us_per_item, "o-", color=C[0])
    ax[0].set(xlabel="k", ylabel="μs / 원소 (파싱 제외)", title="Bloom: k에 따른 삽입 비용")
    for i, a in enumerate(("cms", "cms-cu")):
        g = ts[(ts.algo == a) & (ts.param == "d")]
        ax[1].plot(g.value, g.algo_us_per_item, "o-", color=C[i], label="CU" if a == "cms-cu" else "기본")
    ax[1].set(xlabel="d", ylabel="μs / 원소", title="CMS: depth에 따른 갱신 비용")
    ax[1].legend()
    g = ts[ts.algo == "hll-estimate"]
    ax[2].loglog(2 ** g.value, g.total_s * 1e3, "o-", color=C[2])
    ax[2].set(xlabel="m = 2^p", ylabel="ms / estimate()", title="HLL: 추정 1회 비용 (O(m))")
    _save(fig, "42_time_vs_params.png")


def _fmt_b(b):
    for unit in ("B", "KB", "MB", "GB"):
        if b < 1024:
            return f"{b:,.1f} {unit}" if unit != "B" else f"{b:,.0f} B"
        b /= 1024
    return f"{b:.1f} TB"


def run():
    fig_dataset()
    fig_bloom()
    fig_cms()
    fig_hll()
    fig_memory_time()
    print("figures →", FIG_DIR)


if __name__ == "__main__":
    run()
