"""전체 재현: EDA → 비교 실험 → 튜닝/설명가능성 → PDF 보고서.

    uv run python -m src.main            # 전부 실행
    uv run python -m src.main eda        # 특정 단계만
"""
import sys
import time

STAGES = ["eda", "experiments", "tune_explain", "make_pdf"]


def main(stages: list[str]) -> None:
    import importlib

    for stage in stages:
        t = time.time()
        print(f"\n===== {stage} =====")
        importlib.import_module(f"src.{stage}").run()
        print(f"===== {stage} 완료 ({time.time() - t:.0f}s) =====")


if __name__ == "__main__":
    main(sys.argv[1:] or STAGES)
