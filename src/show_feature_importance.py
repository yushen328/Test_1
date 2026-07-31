import pandas as pd
from sklearn.pipeline import Pipeline


def show_random_forest_feature_importance(model: Pipeline, top_n: int = 5) -> None:
    """输出随机森林认为最重要的前几个特征。"""
    preprocessor = model.named_steps["preprocessor"]
    regressor = model.named_steps["model"]

    # 预处理后，类别列会被拆成很多 One-Hot 特征，所以这里取的是转换后的特征名。
    feature_names = preprocessor.get_feature_names_out()
    importances = regressor.feature_importances_

    ranking = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(top_n)
    )

    print(f"\n随机森林特征重要性分析（前 {top_n} 个，仅用于解释）")
    print("-" * 38)
    print(ranking.to_string(index=False))
