"""Flajolet–Martin (PCSA, Probabilistic Counting with Stochastic Averaging, 1985).

HyperLogLog의 전신으로, 비교 기준선으로 구현한다.
- m개 비트맵(각 32비트)을 두고, 해시 x의 하위 log2(m)비트로 비트맵 j를 고른 뒤
  나머지 비트의 trailing zero 개수 r에 해당하는 비트를 1로 설정.
- R_j = 비트맵 j에서 가장 낮은 0비트 위치,  E = (m/φ)·2^{mean(R_j)},  φ ≈ 0.77351
표준오차 ≈ 0.78/√m 이지만 레지스터가 32비트라 같은 메모리 기준으로는 HLL보다 불리하다.
또한 n이 m에 비해 작을 때(약 n < 10m) 편향이 크다.
"""
from .hashing import hash64

PHI = 0.77351
BITMAP_BITS = 32


class FlajoletMartin:
    __slots__ = ("m", "seed", "bitmaps", "_shift", "_mask")

    def __init__(self, m: int, seed: int = 0):
        if m & (m - 1):
            raise ValueError("m은 2의 거듭제곱")
        self.m = m
        self.seed = seed
        self.bitmaps = [0] * m
        self._shift = m.bit_length() - 1
        self._mask = m - 1

    def add(self, item: bytes) -> None:
        x = hash64(item, self.seed)
        j = x & self._mask
        rest = x >> self._shift
        r = (rest & -rest).bit_length() - 1 if rest else BITMAP_BITS - 1
        if r >= BITMAP_BITS:
            r = BITMAP_BITS - 1
        self.bitmaps[j] |= 1 << r

    def estimate(self) -> float:
        total = 0
        for b in self.bitmaps:
            total += (~b & (b + 1)).bit_length() - 1  # 가장 낮은 0비트 위치
        return self.m / PHI * 2 ** (total / self.m)

    def size_bytes(self) -> int:
        return self.m * BITMAP_BITS // 8

    @property
    def std_error(self) -> float:
        return 0.78 / self.m ** 0.5
