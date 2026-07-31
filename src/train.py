import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from add_features import add_features
from build_preprocessor import build_preprocessor
from config import MODEL_PATH, TARGET
from evaluate_model import evaluate_model
from get_models import get_models
from load_data import load_data
from show_feature_importance import show_random_forest_feature_importance


def main() -> None:
    """完整训练流程：读取数据、处理特征、训练模型、评估结果。"""
    df = load_data()
    df = add_features(df)

    if TARGET not in df.columns:
        raise ValueError(f"数据中没有找到目标列：{TARGET}")

    # x 是模型输入特征；y 是我们希望模型预测的房价。
    x = df.drop(columns=[TARGET, "Id"], errors="ignore")

    # 房价差距很大，取 log 后模型通常更稳定；评估时会再转回真实房价。
    y = np.log1p(df[TARGET])

    # 把数据分成训练集和测试集：
    # 训练集用来让模型学习，测试集用来检查模型有没有真的学会。
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    models = get_models()

    trained_models = {}
    model_scores = {}

    for name, estimator in models.items():
        # Pipeline 把“预处理”和“模型训练”连在一起，避免手动重复处理数据。
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(x_train)),
                ("model", estimator),
            ]
        )

        pipeline.fit(x_train, y_train)
        scores = evaluate_model(name, pipeline, x_test, y_test)
        trained_models[name] = pipeline
        model_scores[name] = scores

    best_model_name = min(
        model_scores, key=lambda model_name: model_scores[model_name]["rmse"]
    )
    best_model = trained_models[best_model_name]
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    print(f"\n最佳模型：{best_model_name}")
    print(f"最佳 RMSE：{model_scores[best_model_name]['rmse']:,.2f}")
    print(f"模型已保存到：{MODEL_PATH}")

    print("\n下面的特征重要性来自随机森林，仅用于解释；最终保存模型由 RMSE 决定。")
    show_random_forest_feature_importance(trained_models["随机森林模型"], top_n=5)


if __name__ == "__main__":
    main()
