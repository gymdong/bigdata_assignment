# Feature Engineering 파이프라인 비교 실험 — Titanic

Titanic 데이터(OpenML, 1,309명)로 결측치 처리, 인코딩, 스케일링, 파생 변수, 변수 선택 전략에 따른 성능 차이를 비교하는 과제입니다.
최종 보고서: [`reports/report.pdf`](reports/report.pdf) (원본 [`reports/report.md`](reports/report.md))

## 실행

```bash
cd Feature_Engineering
uv sync                          # Python 3.12 + 고정된 의존성(uv.lock)
uv run python -m src.main        # 데이터 → EDA → 실험 → 튜닝/SHAP → PDF
uv run python -m src.main eda    # 단계별 실행: eda | experiments | tune_explain | make_pdf
```

- macOS에서 LightGBM을 쓰려면 OpenMP 런타임이 필요합니다: `brew install libomp`
- PDF 변환은 로컬 Chrome/Chromium headless를 사용합니다. 없으면 `reports/report.html`을 브라우저에서 PDF로 인쇄하면 됩니다.
- 모든 난수는 `random_state=42`로 고정했습니다. 전체 실행은 약 4분 걸립니다 (Apple Silicon 노트북 기준).

## 구조

| 파일 | 내용 |
|---|---|
| `src/data.py` | STEP 01: 다운로드/캐시(`data/titanic.csv`), 누수 컬럼 제거, 컬럼 설명 표 |
| `src/eda.py` | STEP 02: 결측·이상치·분포·상관·타겟 분석 → `reports/figures/01~07` |
| `src/features.py` | STEP 03-4: 파생 변수 7개 (`DerivedFeatures` Transformer) |
| `src/pipelines.py` | STEP 03~05: 실험 설정(`FEConfig`), 전처리·변수 선택·모델 `Pipeline` 조립 |
| `src/experiments.py` | 필수 비교표(Base, Exp-1~3), 단일 요인 ablation, 변수 선택 k 곡선 및 방법 비교 |
| `src/tune_explain.py` | GridSearchCV(전처리+모델), ROC/CM, RF top-N, SHAP, 중요도 3종 비교 |
| `src/make_pdf.py` | `report.md` → HTML → PDF |
| `results/*.csv` | 모든 실험 수치 |
