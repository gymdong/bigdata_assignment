"""STEP 04~07: 메인 실험(정확도·메모리·시간) + 파라미터 비교 실험.

- 메인 실험/시간 측정은 **순차 실행**(코어 경쟁 없음)으로 수행한다.
- 파라미터 스윕은 정확도가 목적이므로 ProcessPoolExecutor로 병렬 실행한다.
  (병렬 실행 중의 시간 값은 P/E 코어 혼재로 신뢰하기 어려워 별도 순차 timing 스윕을 둔다.)
- 모든 스트리밍 실험은 ratings.dat를 처음부터 한 줄씩 다시 읽는다.
"""
import math
import os
import random
import statistics as st
import sys
import time
import tracemalloc
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from .bloom import BloomFilter, optimal_k, theoretical_fpr
from .cms import CountMinSketch
from .data import RESULTS_DIR, int_key, pair_key, stream_ratings
from .fm import FlajoletMartin
from .ground_truth import NEG_PATH, POS_PATH, load_lines, load_movie_counts, load_summary
from .hll import HyperLogLog

SEEDS = [0, 1, 2, 3, 4]
N = 1_000_209  # 전체 레코드 수 = 고유 (user, movie) 쌍 수
WORKERS = os.cpu_count() or 4

BLOOM_MAIN = dict(bpe=10)  # 원소당 10비트 → k* = 7, 이론 FPR ≈ 0.82%
BLOOM_BPE = [2, 4, 6, 8, 10, 12, 16]
BLOOM_K = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12]
CMS_MAIN = dict(eps=0.01, delta=0.01)  # w=272, d=5
CMS_W = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
CMS_D = [1, 2, 3, 4, 5, 6, 8]
HLL_P = list(range(4, 17))
HLL_SEEDS = list(range(20))  # HLL은 분산 추정을 위해 seed 20개
TOPK = 50
TIMING_PREFIX = 200_000


def _log(*a):
    print(*a, flush=True)
    sys.stdout.flush()


def _save(rows, name):
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / name, index=False)
    _log(f"  → results/{name} ({len(df)} rows)")
    return df


def _pmap(fn, tasks):
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        return list(ex.map(fn, tasks, chunksize=1))


# ---------------------------------------------------------------- 시간 기준선
def _pair(r):
    return pair_key(r.user, r.movie)


def _movie(r):
    return int_key(r.movie)


def parse_only_seconds(key=_pair, limit=None) -> float:
    """스트림 파싱 + 키 생성만 하는 기준 시간 (알고리즘 순수 비용 = 전체 − 기준)."""
    t = time.perf_counter()
    for r in stream_ratings(limit=limit):
        key(r)
    return time.perf_counter() - t


# ================================================================ Bloom Filter
def bloom_sweep_task(args):
    bpe, k = args
    neg = load_lines(NEG_PATH)
    pos = load_lines(POS_PATH)
    bf = BloomFilter.for_capacity(N, bpe, k=k, seed=0)
    t = time.perf_counter()
    for r in stream_ratings():
        bf.add(pair_key(r.user, r.movie))
    t_ins = time.perf_counter() - t
    fp = sum(x in bf for x in neg)
    fn = sum(x not in bf for x in pos)
    return dict(bpe=bpe, k=k, m_bits=bf.m, bytes=bf.size_bytes(), k_opt=optimal_k(bf.m, N),
                fpr=fp / len(neg), fpr_theory=theoretical_fpr(bf.m, N, k), fn=fn,
                fill_ratio=bf.fill_ratio(), insert_s_parallel=t_ins)


def bloom_main(seed: int) -> dict:
    neg = load_lines(NEG_PATH)
    bf = BloomFilter.for_capacity(N, BLOOM_MAIN["bpe"], seed=seed)
    t = time.perf_counter()
    for r in stream_ratings():
        bf.add(pair_key(r.user, r.movie))
    t_ins = time.perf_counter() - t
    # 거짓 음성 확인: 전체 100만 개 양성을 모두 질의
    t = time.perf_counter()
    fn = 0
    for r in stream_ratings():
        if pair_key(r.user, r.movie) not in bf:
            fn += 1
    t_qpos = time.perf_counter() - t
    t = time.perf_counter()
    fp = sum(x in bf for x in neg)
    t_qneg = time.perf_counter() - t
    return dict(seed=seed, m_bits=bf.m, k=bf.k, bytes=bf.size_bytes(), fpr=fp / len(neg),
                fpr_theory=bf.theoretical_fpr(), fn=fn, fill_ratio=bf.fill_ratio(),
                insert_s=t_ins, query_pos_s=t_qpos, query_neg_s=t_qneg, n_neg=len(neg))


def bloom_capacity_curve() -> list[dict]:
    """스트림 진행에 따른 FPR: 정상 용량(n) 설계 vs 과소 설계(n/4)."""
    neg = load_lines(NEG_PATH)[:20_000]
    rows = []
    for label, cap in (("capacity=n", N), ("capacity=n/4", N // 4)):
        bf = BloomFilter.for_capacity(cap, BLOOM_MAIN["bpe"], seed=0)
        for i, r in enumerate(stream_ratings(), 1):
            bf.add(pair_key(r.user, r.movie))
            if i % 50_000 == 0 or i == N:
                rows.append(dict(design=label, inserted=i, m_bits=bf.m, k=bf.k,
                                 fpr=sum(x in bf for x in neg) / len(neg),
                                 fpr_theory=theoretical_fpr(bf.m, i, bf.k)))
    return rows


def bloom_memory() -> dict:
    tracemalloc.start()
    bf = BloomFilter.for_capacity(N, BLOOM_MAIN["bpe"], seed=0)
    for r in stream_ratings():
        bf.add(pair_key(r.user, r.movie))
    cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return dict(struct_bytes=bf.size_bytes(), traced_current=cur, traced_peak=peak)


# ================================================================ Count-Min Sketch
def _cms_eval(cms: CountMinSketch, truth: dict[bytes, int]) -> dict:
    n = sum(truth.values())
    est = {x: cms.query(x) for x in truth}
    errs = [est[x] - c for x, c in truth.items()]
    rel = [(est[x] - c) / c for x, c in truth.items()]
    bound = cms.epsilon * n
    true_top = set(sorted(truth, key=truth.get, reverse=True)[:TOPK])
    est_top = set(sorted(est, key=lambda x: (est[x], x), reverse=True)[:TOPK])
    return dict(mae=st.fmean(errs), max_abs_err=max(errs), mean_rel_err=st.fmean(rel),
                median_rel_err=st.median(rel), exact_ratio=sum(e == 0 for e in errs) / len(errs),
                within_bound=sum(e <= bound for e in errs) / len(errs), bound=bound,
                min_err=min(errs), topk_precision=len(true_top & est_top) / TOPK)


def cms_sweep_task(args):
    w, d, cons = args
    truth = load_movie_counts()
    cms = CountMinSketch(w, d, seed=0, conservative=cons)
    for r in stream_ratings():
        cms.update(int_key(r.movie))
    return dict(w=w, d=d, conservative=cons, counters=w * d, bytes=cms.size_bytes(),
                eps=cms.epsilon, delta=cms.delta, **_cms_eval(cms, truth))


def cms_main_task(args):
    seed, cons = args
    truth = load_movie_counts()
    cms = CountMinSketch.from_error(**CMS_MAIN, seed=seed, conservative=cons)
    t = time.perf_counter()
    for r in stream_ratings():
        cms.update(int_key(r.movie))
    t_upd = time.perf_counter() - t
    t = time.perf_counter()
    for x in truth:
        cms.query(x)
    t_q = time.perf_counter() - t
    res = dict(seed=seed, conservative=cons, w=cms.w, d=cms.d, bytes=cms.size_bytes(),
               update_s=t_upd, query_s=t_q, **_cms_eval(cms, truth))
    # 빈도 구간별 상대오차 (seed 0만)
    buckets = []
    if seed == 0:
        edges = [1, 10, 100, 1000, 10_000]
        for lo, hi in zip(edges[:-1], edges[1:]):
            items = [(x, c) for x, c in truth.items() if lo <= c < hi]
            rel = [(cms.query(x) - c) / c for x, c in items]
            absd = [cms.query(x) - c for x, c in items]
            buckets.append(dict(conservative=cons, bucket=f"[{lo}, {hi})", n_items=len(items),
                                mean_abs_err=st.fmean(absd), mean_rel_err=st.fmean(rel),
                                median_rel_err=st.median(rel)))
    return res, buckets


def cms_memory() -> dict:
    tracemalloc.start()
    cms = CountMinSketch.from_error(**CMS_MAIN, seed=0)
    for r in stream_ratings():
        cms.update(int_key(r.movie))
    cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return dict(struct_bytes=cms.size_bytes(), traced_current=cur, traced_peak=peak)


# ================================================================ HyperLogLog / FM
TARGETS = ("users", "movies", "pairs")


def hll_sweep_task(args):
    p, seed = args
    hll = {t: HyperLogLog(p, seed) for t in TARGETS}
    fm = {t: FlajoletMartin(1 << p, seed) for t in TARGETS}
    for r in stream_ratings():
        keys = (int_key(r.user), int_key(r.movie), pair_key(r.user, r.movie))
        for t, k in zip(TARGETS, keys):
            hll[t].add(k)
            fm[t].add(k)
    return [dict(p=p, m=1 << p, seed=seed, target=t,
                 hll_est=hll[t].estimate(), fm_est=fm[t].estimate(),
                 hll_bytes=hll[t].size_bytes(), hll_packed_bytes=hll[t].packed_size_bytes(),
                 fm_bytes=fm[t].size_bytes()) for t in TARGETS]


def hll_curve() -> list[dict]:
    """스트림 진행에 따른 고유 (user, movie) 쌍 추정 (p=10, 14) vs 실제."""
    hs = {p: HyperLogLog(p, 0) for p in (10, 14)}
    exact: set[bytes] = set()
    rows = []
    for i, r in enumerate(stream_ratings(), 1):
        k = pair_key(r.user, r.movie)
        exact.add(k)
        for h in hs.values():
            h.add(k)
        if i % 25_000 == 0 or i == N:
            for p, h in hs.items():
                rows.append(dict(processed=i, p=p, true=len(exact), est=h.estimate()))
    return rows


def hll_merge_demo() -> dict:
    """스트림을 두 노드로 나눠 처리 후 병합 → 단일 HLL과 레지스터가 완전히 동일한지 확인."""
    a, b, whole = HyperLogLog(14, 0), HyperLogLog(14, 0), HyperLogLog(14, 0)
    for i, r in enumerate(stream_ratings()):
        k = int_key(r.user)
        (a if i % 2 else b).add(k)
        whole.add(k)
    a_est, b_est = a.estimate(), b.estimate()
    a.merge(b)
    return dict(node_a_est=a_est, node_b_est=b_est, merged_est=a.estimate(),
                whole_est=whole.estimate(), identical_registers=a.reg == whole.reg)


def hll_memory() -> dict:
    tracemalloc.start()
    h = HyperLogLog(14, 0)
    for r in stream_ratings():
        h.add(pair_key(r.user, r.movie))
    cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return dict(struct_bytes=h.size_bytes(), traced_current=cur, traced_peak=peak)


# ================================================================ 순차 timing
def timing_main() -> list[dict]:
    """전체 스트림 1회 처리 시간: 각 스케치 vs 정확한 set/dict (순차, 3회 중 중앙값).

    algo_s = 전체 시간 − (같은 키를 만드는 파싱 기준 시간) → 자료구조 갱신 자체의 비용.
    """
    def clock(make, add, key, limit=None, reps=3):
        ts = []
        for _ in range(reps):
            obj = make()
            fn = add(obj)
            t = time.perf_counter()
            for r in stream_ratings(limit=limit):
                fn(key(r))
            ts.append(time.perf_counter() - t)
        return st.median(ts)

    def count_add(d):
        def f(k):
            d[k] = d.get(k, 0) + 1
        return f

    base = {k: st.median(parse_only_seconds(k) for _ in range(3)) for k in (_pair, _movie)}
    cases = [
        ("Bloom Filter (m/n=10, k=7)", lambda: BloomFilter.for_capacity(N, 10), lambda o: o.add, _pair),
        ("Exact set (pairs)", set, lambda o: o.add, _pair),
        ("HyperLogLog (p=14)", lambda: HyperLogLog(14), lambda o: o.add, _pair),
        ("Flajolet-Martin (m=16384)", lambda: FlajoletMartin(1 << 14), lambda o: o.add, _pair),
        ("CMS (w=272, d=5)", lambda: CountMinSketch.from_error(**CMS_MAIN), lambda o: o.update, _movie),
        ("CMS-CU (w=272, d=5)", lambda: CountMinSketch.from_error(**CMS_MAIN, conservative=True),
         lambda o: o.update, _movie),
        ("Exact dict (movie counts)", dict, count_add, _movie),
    ]
    rows = [dict(method=f"Parse only ({'pair' if k is _pair else 'movie'} key)", total_s=t, algo_s=0.0,
                 throughput=N / t, algo_us_per_item=0.0) for k, t in base.items()]
    for name, make, add, key in cases:
        t = clock(make, add, key)
        algo = max(t - base[key], 0.0)
        rows.append(dict(method=name, total_s=t, algo_s=algo, throughput=N / t, algo_us_per_item=algo / N * 1e6))
    return rows


def timing_sweep() -> list[dict]:
    """파라미터에 따른 처리 시간(순차, 앞 20만 건): Bloom k, CMS d·w, HLL p."""
    n = TIMING_PREFIX
    base = {k: st.median(parse_only_seconds(k, n) for _ in range(3)) for k in (_pair, _movie)}
    rows = []

    def clock(make, add, key):
        ts = []
        for _ in range(3):
            fn = add(make())
            t = time.perf_counter()
            for r in stream_ratings(limit=n):
                fn(key(r))
            ts.append(time.perf_counter() - t)
        t = st.median(ts)
        return dict(total_s=t, throughput=n / t, algo_us_per_item=max(t - base[key], 0.0) / n * 1e6)

    for k in BLOOM_K:
        rows.append(dict(algo="bloom", param="k", value=k,
                         **clock(lambda: BloomFilter(N * 10, k), lambda o: o.add, _pair)))
    for d in CMS_D:
        for cons in (False, True):
            rows.append(dict(algo="cms-cu" if cons else "cms", param="d", value=d,
                             **clock(lambda: CountMinSketch(1024, d, conservative=cons), lambda o: o.update, _movie)))
    for w in (64, 1024, 4096):
        rows.append(dict(algo="cms", param="w(d=4)", value=w,
                         **clock(lambda: CountMinSketch(w, 4), lambda o: o.update, _movie)))
    for p in (4, 8, 12, 16):
        rows.append(dict(algo="hll", param="p", value=p, **clock(lambda: HyperLogLog(p), lambda o: o.add, _pair)))
    # 추정(질의) 1회 시간: HLL은 레지스터 m개를 모두 훑으므로 O(m)
    for p in HLL_P:
        h = HyperLogLog(p)
        for i in range(10_000):
            h.add(str(i).encode())
        t = time.perf_counter()
        for _ in range(20):
            h.estimate()
        rows.append(dict(algo="hll-estimate", param="p", value=p, total_s=(time.perf_counter() - t) / 20,
                         throughput=float("nan"), algo_us_per_item=float("nan")))
    return rows


# ================================================================ 실행
def run(stage: str = "all") -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    load_summary()  # ground truth 존재 확인
    t_all = time.perf_counter()

    if stage in ("all", "bloom"):
        _log("[Bloom] 파라미터 스윕 (병렬)")
        _save(_pmap(bloom_sweep_task, [(b, k) for b in BLOOM_BPE for k in BLOOM_K]), "bloom_sweep.csv")
        _log("[Bloom] 메인 실험 5 seeds (순차)")
        _save([bloom_main(s) for s in SEEDS], "bloom_main.csv")
        _log("[Bloom] 용량 곡선")
        _save(bloom_capacity_curve(), "bloom_capacity_curve.csv")

    if stage in ("all", "cms"):
        _log("[CMS] 파라미터 스윕 (병렬)")
        _save(_pmap(cms_sweep_task, [(w, d, c) for w in CMS_W for d in CMS_D for c in (False, True)]),
              "cms_sweep.csv")
        _log("[CMS] 메인 실험 5 seeds × {기본, CU} (순차)")
        out = [cms_main_task((s, c)) for s in SEEDS for c in (False, True)]
        _save([o[0] for o in out], "cms_main.csv")
        _save([b for o in out for b in o[1]], "cms_buckets.csv")

    if stage in ("all", "hll"):
        _log("[HLL/FM] p × 20 seeds 스윕 (병렬)")
        res = _pmap(hll_sweep_task, [(p, s) for p in HLL_P for s in HLL_SEEDS])
        _save([row for rows in res for row in rows], "hll_sweep.csv")
        _log("[HLL] 스트림 곡선 / 병합")
        _save(hll_curve(), "hll_curve.csv")
        _save([hll_merge_demo()], "hll_merge.csv")

    if stage in ("all", "memory"):
        _log("[Memory] tracemalloc 측정")
        gt = load_summary()["exact_memory"]
        rows = []
        for algo, m, exact_name in (("Bloom Filter (m/n=10)", bloom_memory(), "set_pairs"),
                                    ("Count-Min Sketch (272×5)", cms_memory(), "dict_movie_counts"),
                                    ("HyperLogLog (p=14)", hll_memory(), "set_pairs")):
            rows.append(dict(algo=algo, **m, exact=exact_name, exact_current=gt[exact_name]["current_bytes"],
                             exact_peak=gt[exact_name]["peak_bytes"]))
        _save(rows, "memory.csv")

    if stage in ("all", "timing"):
        _log("[Timing] 전체 스트림 처리 시간 (순차)")
        _save(timing_main(), "timing_main.csv")
        _log("[Timing] 파라미터별 처리 시간 (순차, prefix)")
        _save(timing_sweep(), "timing_sweep.csv")

    _log(f"완료: {time.perf_counter() - t_all:.1f}s")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "all")
