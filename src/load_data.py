import pandas as pd

from config import DATA_PATH


def load_data() -> pd.DataFrame:
    """读取房价数据。"""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"没有找到数据文件：{DATA_PATH}\n"
            "请先下载 Kaggle House Prices 数据集，并把 train.csv 放到 data/raw/train.csv"
        )

    return pd.read_csv(DATA_PATH)
