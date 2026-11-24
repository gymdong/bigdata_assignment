"""전체 파이프라인: 데이터 → Ground Truth → 실험 → 그림 → PDF.

uv run python -m src.main                 # 전체
uv run python -m src.main experiments     # 단계별: data | ground_truth | experiments | plotting | make_pdf
"""
import sys

from . import data, experiments, ground_truth, make_pdf, plotting

STEPS = {
    "data": data.ensure_data,
    "ground_truth": ground_truth.run,
    "experiments": experiments.run,
    "plotting": plotting.run,
    "make_pdf": make_pdf.run,
}

if __name__ == "__main__":
    for name in sys.argv[1:] or STEPS:
        print(f"=== {name} ===", flush=True)
        STEPS[name]()
