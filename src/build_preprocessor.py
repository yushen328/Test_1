import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    """创建数据预处理流程，让不同类型的列用不同方式处理。"""
    # 数值列，例如面积、年份、车库大小。
    numeric_features = x.select_dtypes(include=["number"]).columns

    # 类别列，例如社区、房屋类型、是否有中央空调。
    categorical_features = x.select_dtypes(include=["object", "string"]).columns

    # 数值列处理：
    # 1. 缺失值用中位数补上
    # 2. 标准化，让不同量纲的数字更容易被线性模型学习
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # 类别列处理：
    # 1. 缺失值用出现最多的类别补上
    # 2. One-Hot 编码，把文字类别转成模型能看懂的 0/1 数字列
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # ColumnTransformer 的作用是：数值列走 numeric_pipeline，类别列走 categorical_pipeline。
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )
