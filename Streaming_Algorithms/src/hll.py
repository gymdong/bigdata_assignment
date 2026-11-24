"""HyperLogLog — 고유 원소 수(cardinality) 근사 추정 (직접 구현, Flajolet et al. 2007).

구조: m = 2^p 개 레지스터(bytearray, 레지스터당 1바이트; 이론상 6비트면 충분).
- add(x): 64비트 해시 x의 상위 p비트 → 버킷 j,
          나머지 (64−p)비트에서 첫 1비트의 위치 ρ(= 선행 0 개수 + 1) → M[j] = max(M[j], ρ)
- estimate(): E = α_m · m² / Σ 2^{−M[j]}  (조화평균)
  · 소규모 보정: E ≤ 2.5m 이고 값이 0인 레지스터 V > 0 이면 Linear Counting m·ln(m/V)
  · 64비트 해시를 쓰므로 대규모(2^32 근처) 보정은 불필요
표준오차 ≈ 1.04/√m. 두 HLL은 레지스터별 max로 손실 없이 병합(merge)할 수 있다.
"""
import math

from .hashing import hash64


def alpha(m: int) -> float:
    if m == 16:
        return 0.673
    if m == 32:
        return 0.697
    if m == 64:
        return 0.709
    return 0.7213 / (1 + 1.079 / m)


class HyperLogLog:
    __slots__ = ("p", "m", "seed", "reg", "_rest_bits", "_rest_mask")

    def __init__(self, p: int, seed: int = 0):
        if not 4 <= p <= 18:
            raise ValueError("p는 4~18 범위")
        self.p = p
        self.m = 1 << p
        self.seed = seed
        self.reg = bytearray(self.m)
        self._rest_bits = 64 - p
        self._rest_mask = (1 << (64 - p)) - 1

    def add(self, item: bytes) -> None:
        x = hash64(item, self.seed)
        j = x >> self._rest_bits
        rho = self._rest_bits - (x & self._rest_mask).bit_length() + 1
        if rho > self.reg[j]:
            self.reg[j] = rho

    def estimate(self) -> float:
        m = self.m
        z = 0.0
        zeros = 0
        for r in self.reg:
            z += 2.0 ** -r
            if r == 0:
                zeros += 1
        e = alpha(m) * m * m / z
        if e <= 2.5 * m and zeros:
            return m * math.log(m / zeros)
        return e

    def merge(self, other: "HyperLogLog") -> None:
        if (self.p, self.seed) != (other.p, other.seed):
            raise ValueError("p와 seed가 같아야 병합 가능")
        self.reg = bytearray(max(a, b) for a, b in zip(self.reg, other.reg))

    def size_bytes(self) -> int:
        return len(self.reg)

    def packed_size_bytes(self) -> int:
        """레지스터를 6비트로 패킹했을 때의 크기(실서비스 구현 기준)."""
        return math.ceil(self.m * 6 / 8)

    @property
    def std_error(self) -> float:
        return 1.04 / math.sqrt(self.m)
