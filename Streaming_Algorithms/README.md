# 스트리밍 알고리즘 구현 및 정확도·메모리 트레이드오프 분석 — MovieLens 1M

MovieLens 1M 평점 로그(1,000,209건)를 한 줄씩 들어오는 이벤트 스트림으로 보고,
**Bloom Filter / Count-Min Sketch(+Conservative Update) / HyperLogLog(+Flajolet–Martin 비교)** 를
순수 Python으로 직접 구현해 정확도·메모리·처리 시간을 정확한 set/dict 결과(Ground Truth)와 비교합니다.

최종 보고서: [`reports/report.pdf`](reports/report.pdf) (원본 [`reports/report.md`](reports/report.md))

## 실행

```bash
cd Streaming_Algorithms
uv sync                                   # Python 3.12 + 고정된 의존성(uv.lock)
uv run pytest -q                          # 자료구조 정합성 테스트 6개
uv run python -m src.main                 # 다운로드 → Ground Truth → 실험 → 그림 → PDF
uv run python -m src.main experiments     # 단계별: data | ground_truth | experiments | plotting | make_pdf
uv run python -m src.experiments bloom    # 실험 일부만: bloom | cms | hll | memory | timing
```

- 데이터는 처음 실행할 때 `https://files.grouplens.org/datasets/movielens/ml-1m.zip`에서 `data/`로 자동으로 내려받습니다.
- PDF 변환에는 로컬 Chrome/Chromium headless를 사용합니다.
- 전체 실험은 Apple M4(10코어) 기준 약 6~7분이 걸립니다. 파라미터 스윕은 병렬로, 시간 측정은 순차로 실행합니다.

## 구조

| 파일 | 내용 |
|---|---|
| `src/data.py` | 다운로드, `stream_ratings()`: 한 줄씩 파싱하는 제너레이터 |
| `src/hashing.py` | BLAKE2b-128 → (h1, h2), seed는 BLAKE2b key로 사용 |
| `src/bloom.py` | Bloom Filter (bytearray 비트 배열, double hashing) |
| `src/cms.py` | Count-Min Sketch (`array('I')` 카운터, Conservative Update 옵션) |
| `src/hll.py` | HyperLogLog (α_m 보정, Linear Counting, merge) |
| `src/fm.py` | Flajolet–Martin PCSA (비교 기준선) |
| `src/ground_truth.py` | set/dict 기반 정답, Bloom 음성 질의셋, tracemalloc 메모리 |
| `src/experiments.py` | 메인 실험(5 seeds), 파라미터 스윕, 시간 측정 → `results/*.csv` |
| `src/plotting.py` | `results/*.csv` → `reports/figures/*.png` |
| `src/make_pdf.py` | `report.md` → HTML → PDF |
| `tests/test_sketches.py` | FN=0, FPR≈이론, CMS 과소추정 없음/εN 상한, CU≤기본, HLL 3σ, merge |
