import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline


def evaluate_model(
    name: str, model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series
) -> dict:
    """在测试集上评估模型，并打印 RMSE、MAE、R2 三个指标。"""
    predictions_log = model.predict(x_test)

    # 训练时对房价取了 log1p，这里用 expm1 转回真实房价。
    predictions = np.expm1(predictions_log)
    actual = np.expm1(y_test)

    # RMSE / MAE 越小越好，R2 越接近 1 越好。
    rmse = np.sqrt(mean_squared_error(actual, predictions))
    mae = mean_absolute_error(actual, predictions)
    r2 = r2_score(actual, predictions)

    print(f"\n{name}")
    print("-" * len(name))
    print(f"RMSE: {rmse:,.2f}")
    print(f"MAE : {mae:,.2f}")
    print(f"R2  : {r2:.4f}")

    return {
        "name": name,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }
