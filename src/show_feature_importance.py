import pandas as pd
from sklearn.pipeline import Pipeline


def show_random_forest_feature_importance(model: Pipeline) -> None:
    """输出随机森林认为最重要的 15 个特征。"""
    preprocessor = model.named_steps["preprocessor"]
    regressor = model.named_steps["model"]

    # 预处理后，类别列会被拆成很多 One-Hot 特征，所以这里取的是转换后的特征名。
    feature_names = preprocessor.get_feature_names_out()
    importances = regressor.feature_importances_

    ranking = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(15)
    )

    print("\n随机森林最重要的 15 个特征")
    print("-" * 24)
    print(ranking.to_string(index=False))
