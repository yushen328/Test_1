from pathlib import Path


# 项目根目录：也就是 house-price-prediction-starter 这个文件夹。
ROOT = Path(__file__).resolve().parents[1]

print(ROOT)
# 训练数据的位置。程序会从这里读取 Kaggle 的 train.csv。
DATA_PATH = ROOT / "data" / "raw" / "train.csv"

# 要预测的目标列。House Prices 数据集里的房价列名就是 SalePrice。
TARGET = "SalePrice"
