"""Bloom Filter — 원소 포함 여부 근사 판정 (직접 구현).

구조: m비트 배열(bytearray, 8비트/바이트로 패킹) + k개의 해시 위치.
- add(x):      g_i(x) = (h1 + i·h2) mod m, i = 0..k-1 위치의 비트를 1로 설정
- contains(x): k개 비트가 모두 1이면 "있을 수 있음", 하나라도 0이면 "확실히 없음"

성질: 거짓 음성(FN) = 0, 거짓 양성률(FPR) ≈ (1 − e^{−kn/m})^k,
최적 해시 개수 k* = (m/n)·ln 2.
"""
import math

from .hashing import hash_pair


class BloomFilter:
    __slots__ = ("m", "k", "seed", "bits", "n_added")

    def __init__(self, m: int, k: int, seed: int = 0):
        if m <= 0 or k <= 0:
            raise ValueError("m, k는 양수여야 합니다")
        self.m = m
        self.k = k
        self.seed = seed
        self.bits = bytearray((m + 7) // 8)
        self.n_added = 0

    @classmethod
    def for_capacity(cls, n: int, bits_per_element: float, k: int | None = None, seed: int = 0):
        m = max(8, int(math.ceil(n * bits_per_element)))
        if k is None:
            k = optimal_k(m, n)
        return cls(m, k, seed)

    def add(self, item: bytes) -> None:
        h1, h2 = hash_pair(item, self.seed)
        m, bits = self.m, self.bits
        for i in range(self.k):
            pos = (h1 + i * h2) % m
            bits[pos >> 3] |= 1 << (pos & 7)
        self.n_added += 1

    def __contains__(self, item: bytes) -> bool:
        h1, h2 = hash_pair(item, self.seed)
        m, bits = self.m, self.bits
        for i in range(self.k):
            pos = (h1 + i * h2) % m
            if not bits[pos >> 3] & (1 << (pos & 7)):
                return False
        return True

    contains = __contains__

    def fill_ratio(self) -> float:
        return sum(b.bit_count() for b in self.bits) / self.m

    def size_bytes(self) -> int:
        """자료구조 본체(비트 배열) 크기."""
        return len(self.bits)

    def theoretical_fpr(self, n: int | None = None) -> float:
        n = self.n_added if n is None else n
        return theoretical_fpr(self.m, n, self.k)


def optimal_k(m: int, n: int) -> int:
    return max(1, round(m / n * math.log(2)))


def theoretical_fpr(m: int, n: int, k: int) -> float:
    return (1.0 - math.exp(-k * n / m)) ** k
