"""정합성 테스트: 각 자료구조가 이론적으로 보장하는 성질을 합성 데이터로 확인."""
import math
import random
from collections import Counter

from src.bloom import BloomFilter, theoretical_fpr
from src.cms import CountMinSketch
from src.fm import FlajoletMartin
from src.hll import HyperLogLog


def keys(n, prefix="k"):
    return [f"{prefix}{i}".encode() for i in range(n)]


def test_bloom_no_false_negatives_and_fpr_near_theory():
    n = 20_000
    bf = BloomFilter.for_capacity(n, bits_per_element=8, seed=1)
    ins = keys(n)
    for x in ins:
        bf.add(x)
    assert all(x in bf for x in ins)  # FN = 0
    neg = keys(50_000, "neg")
    fpr = sum(x in bf for x in neg) / len(neg)
    theo = theoretical_fpr(bf.m, n, bf.k)
    assert abs(fpr - theo) < 0.01, (fpr, theo)


def test_cms_never_underestimates_and_within_bound():
    rng = random.Random(0)
    stream = [str(int(rng.paretovariate(1.2))).encode() for _ in range(50_000)]
    truth = Counter(stream)
    for conservative in (False, True):
        cms = CountMinSketch(w=272, d=5, seed=3, conservative=conservative)  # eps=0.01
        for x in stream:
            cms.update(x)
        errs = [cms.query(x) - c for x, c in truth.items()]
        assert min(errs) >= 0
        bound = cms.epsilon * len(stream)
        assert sum(e <= bound for e in errs) / len(errs) >= 1 - cms.delta


def test_conservative_update_not_worse():
    stream = keys(3000) * 3 + keys(200, "hot") * 50
    truth = Counter(stream)
    a, b = CountMinSketch(64, 4, seed=7), CountMinSketch(64, 4, seed=7, conservative=True)
    for x in stream:
        a.update(x)
        b.update(x)
    for x in truth:
        assert truth[x] <= b.query(x) <= a.query(x)


def test_hll_error_within_3_sigma():
    n = 100_000
    for p in (8, 12):
        h = HyperLogLog(p, seed=11)
        for x in keys(n):
            h.add(x)
            h.add(x)  # 중복은 영향 없음
        rel = abs(h.estimate() - n) / n
        assert rel < 3 * h.std_error, (p, rel)


def test_hll_small_range_and_merge():
    h = HyperLogLog(10)
    for x in keys(100):
        h.add(x)
    assert abs(h.estimate() - 100) < 5  # Linear Counting 구간
    a, b, both = HyperLogLog(10, 2), HyperLogLog(10, 2), HyperLogLog(10, 2)
    for x in keys(5000):
        a.add(x); both.add(x)
    for x in keys(5000, "z"):
        b.add(x); both.add(x)
    a.merge(b)
    assert a.reg == both.reg


def test_fm_reasonable():
    n = 200_000
    fm = FlajoletMartin(256, seed=5)
    for x in keys(n):
        fm.add(x)
    assert abs(fm.estimate() - n) / n < 3 * fm.std_error
