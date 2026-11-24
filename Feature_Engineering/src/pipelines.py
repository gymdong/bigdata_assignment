"""STEP 03~05: 전처리 + 변수 선택 + 모델을 하나의 sklearn Pipeline으로 조립."""
from dataclasses import dataclass, replace

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectFromModel, SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, RobustScaler, StandardScaler
from xgboost import XGBClassifier

from .data import RANDOM_STATE
from .features import DERIVED_CAT, DERIVED_NUM, DerivedFeatures

BASE_NUM = ["pclass", "age", "sibsp", "parch", "fare"]
BASE_CAT = ["sex", "embarked"]

SCALERS = {
    "none": "passthrough",
    "standard": StandardScaler(),
    "minmax": MinMaxScaler(),
    "robust": RobustScaler(),
}


@dataclass(frozen=True)
class FEConfig:
    """한 실험의 특성 공학 설정."""
    impute: str = "median"        # mean | median | most_frequent | drop
    encoding: str = "onehot"      # onehot | label
    scaling: str = "standard"     # none | standard | minmax | robust
    derived: bool = True          # 파생 변수 사용 여부
    fs_k: int | None = None       # 선택할 변수 수 (None = 변수 선택 안 함)
    fs_method: str = "rf"         # rf | rfe | kbest

    def with_(self, **kw) -> "FEConfig":
        return replace(self, **kw)


# 과제 5장 필수 비교 실험 (파생 변수는 Exp-1~3 모두 포함, Base는 원본 수치형만)
EXPERIMENTS = {
    "Exp-1": FEConfig(impute="mean", encoding="onehot", scaling="standard", fs_k=None),
    "Exp-2": FEConfig(impute="median", encoding="label", scaling="minmax", fs_k=10),
    "Exp-3": FEConfig(impute="most_frequent", encoding="onehot", scaling="robust", fs_k=10),
}


def get_models() -> dict:
    return {
        "LogisticRegression": LogisticRegression(max_iter=5000),
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                                 subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
                                 random_state=RANDOM_STATE, n_jobs=4),
        "LightGBM": LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15,
                                   random_state=RANDOM_STATE, verbose=-1, n_jobs=4),
    }


def build_preprocessor(cfg: FEConfig) -> ColumnTransformer:
    num_cols = BASE_NUM + (DERIVED_NUM if cfg.derived else [])
    cat_cols = BASE_CAT + (DERIVED_CAT if cfg.derived else [])

    # drop 설정은 행 제거를 실험 코드에서 먼저 수행하므로 남은 결측(없음)에 대비해 median 사용
    num_strategy = "median" if cfg.impute == "drop" else cfg.impute
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy=num_strategy)),
        ("scaler", SCALERS[cfg.scaling]),
    ])
    if cfg.encoding == "onehot":
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    elif cfg.encoding == "label":
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    else:
        raise ValueError(cfg.encoding)
    # 범주형은 평균/중앙값이 정의되지 않으므로 항상 최빈값으로 대체
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", encoder),
    ])
    return ColumnTransformer(
        [("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)],
        verbose_feature_names_out=False,
    )


def build_selector(cfg: FEConfig):
    if cfg.fs_method == "rf":
        est = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
        return SelectFromModel(est, threshold=-np.inf, max_features=cfg.fs_k)
    if cfg.fs_method == "rfe":
        return RFE(LogisticRegression(max_iter=5000), n_features_to_select=cfg.fs_k)
    if cfg.fs_method == "kbest":
        return SelectKBest(f_classif, k=cfg.fs_k)
    raise ValueError(cfg.fs_method)


def build_pipeline(cfg: FEConfig, model) -> Pipeline:
    steps = [("derive", DerivedFeatures(enabled=cfg.derived)), ("prep", build_preprocessor(cfg))]
    if cfg.fs_k is not None:
        steps.append(("select", build_selector(cfg)))
    steps.append(("model", model))
    pipe = Pipeline(steps)
    pipe.set_output(transform="pandas")
    return pipe


def build_base_pipeline(model_name: str, model) -> Pipeline:
    """Base: 결측치 처리/인코딩/스케일링/파생변수/변수선택 모두 없음.

    원본 수치형 5개만 사용(범주형은 인코딩 없이는 모델 입력 불가라 제외).
    결측은 NaN 그대로 두고 트리 모델(RF/XGB/LGBM)의 내장 결측 처리에 맡긴다.
    LogisticRegression은 NaN을 받을 수 없어 유일하게 상수 0 채우기를 적용(최소 조건).
    """
    select = ColumnTransformer([("raw", "passthrough", BASE_NUM)], verbose_feature_names_out=False)
    steps = [("select_raw", select)]
    if model_name == "LogisticRegression":
        steps.append(("fill0", SimpleImputer(strategy="constant", fill_value=0)))
    steps.append(("model", model))
    pipe = Pipeline(steps)
    pipe.set_output(transform="pandas")
    return pipe
