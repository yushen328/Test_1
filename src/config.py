from pathlib import Path


# 项目根目录：也就是 house-price-prediction-starter 这个文件夹。
ROOT = Path(__file__).resolve().parents[1]

# 训练数据的位置。程序会从这里读取 Kaggle 的 train.csv。
DATA_PATH = ROOT / "data" / "raw" / "train.csv"
RAW_DATA_DIR = ROOT / "data" / "raw"

# 保存训练完成的模型。
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "best_model.joblib"

# 默认预测结果输出位置。
OUTPUT_DIR = ROOT / "outputs"
PREDICTION_OUTPUT_PATH = OUTPUT_DIR / "predictions.csv"

# 要预测的目标列。House Prices 数据集里的房价列名就是 SalePrice。
TARGET = "SalePrice"
