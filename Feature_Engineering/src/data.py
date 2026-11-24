"""STEP 01: 데이터 준비 — Titanic (OpenML, 1,309 samples) 로드."""
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "titanic.csv"
TARGET = "survived"
RANDOM_STATE = 42

# 타겟 누수(leakage) 컬럼: 생존 결과 이후에 기록된 정보
LEAKAGE_COLS = ["boat", "body", "home.dest"]

COLUMN_INFO = {
    "pclass": ("범주(순서형)", "객실 등급 (1=1등석, 2=2등석, 3=3등석)"),
    "survived": ("이진(타겟)", "생존 여부 (0=사망, 1=생존)"),
    "name": ("문자열", "승객 이름 (호칭 Mr./Mrs. 등 포함)"),
    "sex": ("범주", "성별"),
    "age": ("수치", "나이"),
    "sibsp": ("수치", "동승한 형제자매/배우자 수"),
    "parch": ("수치", "동승한 부모/자녀 수"),
    "ticket": ("문자열", "티켓 번호"),
    "fare": ("수치", "운임 (파운드)"),
    "cabin": ("문자열", "객실 번호 (첫 글자=데크)"),
    "embarked": ("범주", "승선 항구 (C=Cherbourg, Q=Queenstown, S=Southampton)"),
    "boat": ("문자열", "구명보트 번호 — 누수, 제거"),
    "body": ("수치", "시신 식별 번호 — 누수, 제거"),
    "home.dest": ("문자열", "거주지/목적지 — 결측 다수·누수 가능, 제거"),
}


def _download() -> pd.DataFrame:
    import certifi
    from sklearn.datasets import fetch_openml

    # python.org 빌드의 Python은 기본 CA 번들이 없을 수 있어 certifi를 사용
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    df = fetch_openml("titanic", version=1, as_frame=True, parser="liac-arff").frame
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    return df


def load_raw() -> pd.DataFrame:
    """원본 데이터 (캐시가 없으면 OpenML에서 다운로드)."""
    if not DATA_PATH.exists():
        _download()
    df = pd.read_csv(DATA_PATH)  # dtype을 항상 CSV 기준으로 통일
    df[TARGET] = df[TARGET].astype(int)
    for col in ["pclass", "sibsp", "parch"]:
        df[col] = df[col].astype(int)
    return df


def load_xy():
    """누수 컬럼을 제거한 X, y."""
    df = load_raw()
    X = df.drop(columns=[TARGET] + LEAKAGE_COLS)
    y = df[TARGET]
    return X, y


def column_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        kind, desc = COLUMN_INFO.get(col, ("", ""))
        rows.append({
            "컬럼": col,
            "dtype": str(df[col].dtype),
            "유형": kind,
            "설명": desc,
            "결측 수": int(df[col].isna().sum()),
            "고유값 수": int(df[col].nunique()),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = load_raw()
    print("shape:", df.shape)
    print(column_table(df).to_string(index=False))
