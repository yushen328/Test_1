from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression



def get_models() -> dict:
    """准备要训练的模型。"""
    return {
        "线性回归基线模型": LinearRegression(),
        "随机森林模型": RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=1,
            min_samples_leaf=2,
        ),
    }
