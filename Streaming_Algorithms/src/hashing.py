"""공통 해시 함수.

모든 알고리즘은 원소 하나당 BLAKE2b(128비트) 해시를 **한 번만** 계산하고,
그 결과를 두 개의 64비트 정수 (h1, h2)로 나눠 사용한다.

- Bloom Filter / Count-Min Sketch: Kirsch–Mitzenmacher double hashing
  g_i(x) = h1 + i·h2 (mod m). k개의 독립 해시와 같은 점근적 FPR을 가지면서
  해시 계산 비용은 1회로 줄어든다.
- HyperLogLog / Flajolet–Martin: 64비트 h1을 균등 난수 비트열로 사용한다.

`seed`는 BLAKE2b의 key로 들어가므로 seed가 다르면 서로 독립인 해시 계열이 된다.
"""
from hashlib import blake2b

MASK64 = (1 << 64) - 1


def _key(seed: int) -> bytes:
    return seed.to_bytes(8, "little")


def hash128(data: bytes, seed: int = 0) -> int:
    return int.from_bytes(blake2b(data, digest_size=16, key=_key(seed)).digest(), "little")


def hash_pair(data: bytes, seed: int = 0) -> tuple[int, int]:
    """(h1, h2) 반환. h2는 홀수로 강제해 m이 2의 거듭제곱일 때도 모든 칸을 순회하게 한다."""
    h = hash128(data, seed)
    return h & MASK64, (h >> 64) | 1


def hash64(data: bytes, seed: int = 0) -> int:
    return int.from_bytes(blake2b(data, digest_size=8, key=_key(seed)).digest(), "little")
