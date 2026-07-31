import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from add_features import add_features
from config import MODEL_PATH, PREDICTION_OUTPUT_PATH, RAW_DATA_DIR, TARGET


def predict(input_path: str, output_path: str = str(PREDICTION_OUTPUT_PATH)) -> pd.DataFrame:
    """加载已保存的模型，对 CSV 文件里的房屋数据进行价格预测。"""
    model_path = MODEL_PATH
    if not model_path.exists():
        raise FileNotFoundError(
            f"没有找到模型文件：{model_path}\n"
            "请先运行 python src/train.py 训练并保存模型。"
        )

    df = pd.read_csv(input_path)
    original_df = df.copy()
    df = add_features(df)

    x = df.drop(columns=[TARGET, "Id"], errors="ignore")
    model = joblib.load(model_path)

    predictions_log = model.predict(x)
    predictions = np.expm1(predictions_log)

    result = pd.DataFrame({"PredictedSalePrice": predictions})
    if "Id" in original_df.columns:
        result.insert(0, "Id", original_df["Id"])

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_file, index=False)

    return result


def resolve_input_path(input_csv: str) -> Path:
    """支持输入完整路径，也支持只输入 data/raw 里的文件名。"""
    input_path = Path(input_csv)
    if input_path.exists():
        return input_path

    raw_data_path = RAW_DATA_DIR / input_csv
    if raw_data_path.exists():
        return raw_data_path

    raise FileNotFoundError(
        f"没有找到预测 CSV：{input_csv}\n"
        f"可以输入完整路径，或把文件放到 {RAW_DATA_DIR} 后只输入文件名。"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="使用已保存的房价模型预测新数据。")
    parser.add_argument(
        "input_csv",
        nargs="?",
        help="需要预测的 CSV 文件路径；如果文件在 data/raw 里，可以只输入文件名。",
    )
    parser.add_argument(
        "--output",
        default=str(PREDICTION_OUTPUT_PATH),
        help=f"预测结果输出路径，默认：{PREDICTION_OUTPUT_PATH}",
    )
    args = parser.parse_args()

    input_csv = args.input_csv
    if not input_csv:
        input_csv = input("请输入需要预测的 CSV 文件名或路径：").strip()

    input_path = resolve_input_path(input_csv)
    result = predict(str(input_path), args.output)
    print(f"预测完成，共生成 {len(result)} 条结果。")
    print(f"预测结果已保存到：{args.output}")


if __name__ == "__main__":
    main()
