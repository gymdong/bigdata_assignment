"""STEP 01: MovieLens 1M 다운로드 및 한 줄씩 읽는 스트림 제너레이터."""
import io
import os
import ssl
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterator, NamedTuple

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RATINGS_PATH = DATA_DIR / "ml-1m" / "ratings.dat"
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "reports" / "figures"
URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


class Rating(NamedTuple):
    user: int
    movie: int
    rating: int
    ts: int


def ensure_data() -> Path:
    """ml-1m.zip을 받아 data/ml-1m/에 풀어 둔다 (이미 있으면 생략)."""
    if RATINGS_PATH.exists():
        return RATINGS_PATH
    import certifi

    ctx = ssl.create_default_context(cafile=certifi.where())
    print(f"다운로드: {URL}")
    with urllib.request.urlopen(URL, context=ctx) as resp:
        payload = resp.read()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        for name in zf.namelist():
            # 경로 조작 방지: ml-1m/ 아래의 일반 파일만 추출
            if name.startswith("ml-1m/") and ".." not in name and not name.endswith("/"):
                target = DATA_DIR / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(zf.read(name))
    return RATINGS_PATH


def stream_ratings(path: Path | str = RATINGS_PATH, limit: int | None = None) -> Iterator[Rating]:
    """ratings.dat(`UserID::MovieID::Rating::Timestamp`)를 한 줄씩 파싱해 내보낸다.

    전체 파일을 메모리에 올리지 않으며, 호출할 때마다 파일을 처음부터 다시 읽는다.
    """
    with open(path, "r", encoding="latin-1") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            u, m, r, t = line.rstrip("\n").split("::")
            yield Rating(int(u), int(m), int(r), int(t))


def pair_key(user: int, movie: int) -> bytes:
    return f"{user}:{movie}".encode()


def int_key(x: int) -> bytes:
    return str(x).encode()


if __name__ == "__main__":
    p = ensure_data()
    print(p, os.path.getsize(p), "bytes")
    print(next(stream_ratings()))
