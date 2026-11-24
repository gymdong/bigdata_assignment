"""Count-Min Sketch — 항목별 빈도 근사 추정 (직접 구현).

구조: d행 × w열 카운터 행렬 (array('I'), 카운터당 4바이트, 1차원으로 평탄화).
- update(x, c): 각 행 i에서 열 g_i(x) = (h1 + i·h2) mod w 의 카운터에 c를 더함
- query(x):     d개 카운터의 최솟값 반환

성질: 추정치 f̂(x) ≥ f(x) (과소추정 없음).
w = ⌈e/ε⌉, d = ⌈ln(1/δ)⌉ 이면 확률 1−δ 이상으로 f̂(x) ≤ f(x) + ε·N.

conservative=True 이면 Conservative Update(Estan & Varghese, 2002):
현재 최솟값 + c 보다 작은 카운터만 그 값으로 끌어올린다. 과소추정 없음은 유지되고
불필요한 증가가 줄어 오차가 작아진다(단, 삭제/음수 갱신은 불가).
"""
import math
from array import array

from .hashing import hash_pair


class CountMinSketch:
    __slots__ = ("w", "d", "seed", "conservative", "table", "total")

    def __init__(self, w: int, d: int, seed: int = 0, conservative: bool = False):
        if w <= 0 or d <= 0:
            raise ValueError("w, d는 양수여야 합니다")
        self.w = w
        self.d = d
        self.seed = seed
        self.conservative = conservative
        self.table = array("I", bytes(4 * w * d))
        self.total = 0

    @classmethod
    def from_error(cls, eps: float, delta: float, **kw):
        return cls(int(math.ceil(math.e / eps)), int(math.ceil(math.log(1 / delta))), **kw)

    def _cells(self, item: bytes) -> list[int]:
        h1, h2 = hash_pair(item, self.seed)
        w = self.w
        return [i * w + (h1 + i * h2) % w for i in range(self.d)]

    def update(self, item: bytes, count: int = 1) -> None:
        t = self.table
        cells = self._cells(item)
        if self.conservative:
            target = min(t[c] for c in cells) + count
            for c in cells:
                if t[c] < target:
                    t[c] = target
        else:
            for c in cells:
                t[c] += count
        self.total += count

    def query(self, item: bytes) -> int:
        t = self.table
        return min(t[c] for c in self._cells(item))

    def size_bytes(self) -> int:
        return self.table.itemsize * len(self.table)

    @property
    def epsilon(self) -> float:
        return math.e / self.w

    @property
    def delta(self) -> float:
        return math.exp(-self.d)
