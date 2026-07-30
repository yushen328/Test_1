import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """基于原始字段添加几个更容易让模型理解的新特征。"""
    df = df.copy()

    # 总面积通常比单独的地下室、一楼、二楼面积更能直接反映房子大小。
    if {"TotalBsmtSF", "1stFlrSF", "2ndFlrSF"}.issubset(df.columns):
        df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]

    # 房龄：房子卖出年份减去建造年份。
    if {"YrSold", "YearBuilt"}.issubset(df.columns):
        df["HouseAge"] = df["YrSold"] - df["YearBuilt"]

    # 翻新年龄：房子卖出年份减去最近一次装修年份。
    if {"YrSold", "YearRemodAdd"}.issubset(df.columns):
        df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]

    return df
