"""STEP 03-4: 파생 변수 생성 — sklearn Transformer로 구현해 Pipeline 안에서 동작."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

RAW_TEXT_COLS = ["name", "ticket", "cabin"]  # 그대로는 모델에 넣을 수 없는 고유값 문자열

TITLE_MAP = {
    "Mr": "Mr", "Mrs": "Mrs", "Mme": "Mrs", "Miss": "Miss", "Mlle": "Miss", "Ms": "Miss",
    "Master": "Master",
}
AGE_BINS = [0, 12, 18, 35, 60, np.inf]
AGE_LABELS = ["Child", "Teen", "YoungAdult", "Adult", "Senior"]

DERIVED_NUM = ["FamilySize", "IsAlone", "FarePerPerson", "HasCabin"]
DERIVED_CAT = ["Title", "Deck", "AgeGroup"]


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """행 단위(stateless) 파생 변수 — 다른 행 정보를 쓰지 않으므로 train/test 누수가 없다."""
    out = df.copy()
    out["FamilySize"] = out["sibsp"] + out["parch"] + 1
    out["IsAlone"] = (out["FamilySize"] == 1).astype(int)
    out["FarePerPerson"] = out["fare"] / out["FamilySize"]
    title = out["name"].str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()
    out["Title"] = title.map(TITLE_MAP).fillna("Rare")
    deck = out["cabin"].str[0].fillna("U")
    out["Deck"] = deck.replace({"T": "U"})  # T 데크는 1명뿐이라 Unknown으로 병합
    out["HasCabin"] = out["cabin"].notna().astype(int)
    age_group = pd.cut(out["age"], bins=AGE_BINS, labels=AGE_LABELS, right=False)
    out["AgeGroup"] = age_group.astype(object).where(age_group.notna(), np.nan)
    return out


class DerivedFeatures(BaseEstimator, TransformerMixin):
    """enabled=True면 파생 변수를 추가하고, 항상 원본 텍스트 컬럼(name/ticket/cabin)을 제거한다."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        out = add_derived_features(X) if self.enabled else X.copy()
        return out.drop(columns=[c for c in RAW_TEXT_COLS if c in out.columns])

    def get_feature_names_out(self, input_features=None):
        return None
