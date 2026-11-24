"""STEP 03: Ground Truth — 정확한 set/dict 기반 결과 계산.

스트림을 한 번 읽어
- 고유 (user, movie) 쌍 집합 → Bloom Filter 정답 / 고유 쌍 수
- 영화별 평점 개수 dict → Count-Min Sketch 정답
- 고유 사용자·영화 수 → HyperLogLog 정답
을 만든다. Bloom Filter 질의용 음성(존재하지 않는 쌍) 집합도 여기서 생성한다.
"""
import json
import platform
import random
import subprocess
import sys
import time
import tracemalloc
from collections import Counter

from .data import DATA_DIR, RESULTS_DIR, ensure_data, int_key, pair_key, stream_ratings

NEG_PATH = DATA_DIR / "bloom_negatives.txt"
POS_PATH = DATA_DIR / "bloom_pos_sample.txt"
N_NEG = 100_000
N_POS_SAMPLE = 100_000
SEED = 42


def exact_pass():
    pairs: set[bytes] = set()
    movies: Counter[bytes] = Counter()
    users: set[bytes] = set()
    for r in stream_ratings():
        pairs.add(pair_key(r.user, r.movie))
        movies[int_key(r.movie)] += 1
        users.add(int_key(r.user))
    return pairs, movies, users


def measure_exact_memory() -> dict:
    """각 정확 자료구조를 따로 만들면서 tracemalloc으로 실제 할당량(피크)을 측정."""
    out = {}
    builders = {
        "set_pairs": lambda: _build(set, lambda s, r: s.add(pair_key(r.user, r.movie))),
        "dict_movie_counts": lambda: _build(Counter, lambda c, r: c.__setitem__(k := int_key(r.movie), c[k] + 1)),
        "set_users": lambda: _build(set, lambda s, r: s.add(int_key(r.user))),
        "set_movies": lambda: _build(set, lambda s, r: s.add(int_key(r.movie))),
    }
    for name, fn in builders.items():
        tracemalloc.start()
        obj = fn()
        cur, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        out[name] = {"current_bytes": cur, "peak_bytes": peak, "size": len(obj)}
        del obj
    return out


def _build(factory, step):
    obj = factory()
    for r in stream_ratings():
        step(obj, r)
    return obj


def environment() -> dict:
    def sh(cmd):
        try:
            return subprocess.check_output(cmd, text=True).strip()
        except Exception:
            return "unknown"

    return {
        "os": f"{platform.system()} {platform.release()} ({platform.mac_ver()[0] or platform.version()})",
        "machine": platform.machine(),
        "cpu": sh(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "cpu_cores": sh(["sysctl", "-n", "hw.ncpu"]),
        "ram_gb": round(int(sh(["sysctl", "-n", "hw.memsize"]) or 0) / 2**30, 1),
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
    }


def run() -> None:
    ensure_data()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    pairs, movies, users = exact_pass()
    exact_time = time.perf_counter() - t0

    n_records = 0
    ratings = Counter()
    ts_min, ts_max = float("inf"), 0
    for r in stream_ratings():
        n_records += 1
        ratings[r.rating] += 1
        ts_min, ts_max = min(ts_min, r.ts), max(ts_max, r.ts)

    counts = sorted(movies.values(), reverse=True)
    summary = {
        "n_records": n_records,
        "n_unique_pairs": len(pairs),
        "n_users": len(users),
        "n_movies": len(movies),
        "rating_dist": {str(k): v for k, v in sorted(ratings.items())},
        "ts_min": ts_min,
        "ts_max": ts_max,
        "movie_count_max": counts[0],
        "movie_count_median": counts[len(counts) // 2],
        "movie_count_min": counts[-1],
        "movies_count_le_10": sum(c <= 10 for c in counts),
        "top10pct_share": sum(counts[: len(counts) // 10]) / n_records,
        "exact_pass_seconds": exact_time,
        "file_bytes": (DATA_DIR / "ml-1m" / "ratings.dat").stat().st_size,
    }

    with open(RESULTS_DIR / "movie_counts.csv", "w") as f:
        f.write("movie,count\n")
        for k, v in movies.most_common():
            f.write(f"{k.decode()},{v}\n")

    # Bloom 질의셋: 존재하지 않는 (user, movie) 쌍 + 실제 존재하는 쌍 표본
    rng = random.Random(SEED)
    user_ids = sorted(int(u) for u in users)
    movie_ids = sorted(int(m) for m in movies)
    neg: set[bytes] = set()
    while len(neg) < N_NEG:
        k = pair_key(rng.choice(user_ids), rng.choice(movie_ids))
        if k not in pairs:
            neg.add(k)
    NEG_PATH.write_bytes(b"\n".join(sorted(neg)))
    POS_PATH.write_bytes(b"\n".join(rng.sample(sorted(pairs), N_POS_SAMPLE)))
    summary["pair_density"] = len(pairs) / (len(user_ids) * len(movie_ids))
    del pairs, movies, users

    summary["exact_memory"] = measure_exact_memory()
    summary["environment"] = environment()
    (RESULTS_DIR / "ground_truth.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in summary.items() if k != "rating_dist"}, indent=2, ensure_ascii=False))


def load_lines(path) -> list[bytes]:
    return path.read_bytes().split(b"\n")


def load_movie_counts() -> dict[bytes, int]:
    out = {}
    with open(RESULTS_DIR / "movie_counts.csv") as f:
        next(f)
        for line in f:
            k, v = line.strip().split(",")
            out[k.encode()] = int(v)
    return out


def load_summary() -> dict:
    return json.loads((RESULTS_DIR / "ground_truth.json").read_text())


if __name__ == "__main__":
    run()
