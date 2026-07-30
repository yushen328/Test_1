# 房价预测项目

这是一个适合机器学习入门练习的端到端回归项目。项目使用 Kaggle 的 House Prices 数据集，根据房屋面积、位置、房龄、装修、车库、地下室等特征预测最终成交价 `SalePrice`。

项目重点不是只跑出一个结果，而是完整演示一个表格类机器学习项目的常见流程：读取数据、构造特征、处理缺失值和类别变量、训练多个模型、评估模型效果，并分析哪些特征对房价影响较大。

## 项目目标

- 学习如何读取和检查表格数据。
- 学习如何区分数值特征和类别特征。
- 学习如何处理缺失值。
- 学习如何把文字类别转换成模型可理解的数值特征。
- 学习如何构造更有解释性的衍生特征。
- 训练回归模型预测房价。
- 使用 RMSE、MAE、R2 评估模型表现。
- 查看随机森林模型认为最重要的房价影响因素。

## 数据来源

推荐使用 Kaggle 的 House Prices 数据集：

https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data

下载后，把 `train.csv` 放到下面的位置：

```text
data/raw/train.csv
```

程序默认从这个路径读取训练数据。如果文件不存在，程序会抛出错误并提示需要先下载数据。

## 项目结构

```text
house-price-prediction-starter/
  README.md
  requirements.txt
  data/
    raw/
      train.csv
  src/
    config.py
    load_data.py
    add_features.py
    build_preprocessor.py
    get_models.py
    evaluate_model.py
    show_feature_importance.py
    train.py
```

各文件作用如下：

| 文件 | 作用 |
| --- | --- |
| `README.md` | 项目说明文档，介绍项目目标、流程、函数作用和运行方式。 |
| `requirements.txt` | 项目依赖列表，包括 `pandas`、`numpy`、`scikit-learn`。 |
| `data/raw/train.csv` | 原始训练数据，需要从 Kaggle 下载后放入此处。 |
| `src/config.py` | 保存项目路径、数据路径和预测目标列名等配置。 |
| `src/load_data.py` | 读取原始 CSV 数据。 |
| `src/add_features.py` | 基于原始字段新增衍生特征。 |
| `src/build_preprocessor.py` | 构建数据预处理流程。 |
| `src/get_models.py` | 定义要训练的机器学习模型。 |
| `src/evaluate_model.py` | 计算并打印模型评估指标。 |
| `src/show_feature_importance.py` | 输出随机森林模型中最重要的特征。 |
| `src/train.py` | 项目主入口，串联完整训练和评估流程。 |

## 安装依赖

建议先创建虚拟环境，再安装依赖：

```bash
pip install -r requirements.txt
```

依赖说明：

| 依赖 | 作用 |
| --- | --- |
| `pandas` | 读取 CSV、处理表格数据。 |
| `numpy` | 数学计算，例如对房价取对数、还原对数结果。 |
| `scikit-learn` | 数据预处理、模型训练、模型评估。 |

## 运行项目

在项目根目录运行：

```bash
python src/train.py
```

运行后程序会依次输出：

- 线性回归基线模型的评估结果。
- 随机森林模型的评估结果。
- 随机森林认为最重要的前 15 个特征。

## 完整流程说明

项目主流程在 `src/train.py` 的 `main()` 函数中，整体步骤如下：

1. 调用 `load_data()` 读取 `data/raw/train.csv`。
2. 调用 `add_features(df)` 增加衍生特征。
3. 检查数据中是否存在目标列 `SalePrice`。
4. 从数据中拆出输入特征 `x` 和目标值 `y`。
5. 对房价 `SalePrice` 使用 `np.log1p()` 做对数变换，让模型训练更稳定。
6. 使用 `train_test_split()` 把数据拆分成训练集和测试集。
7. 调用 `build_preprocessor(x_train)` 构建预处理器。
8. 调用 `get_models()` 获取要训练的模型。
9. 对每个模型创建 `Pipeline`，把预处理和模型训练连接在一起。
10. 调用 `pipeline.fit(x_train, y_train)` 训练模型。
11. 调用 `evaluate_model()` 在测试集上评估模型。
12. 保存训练好的模型到 `trained_models` 字典中。
13. 调用 `show_random_forest_feature_importance()` 输出随机森林的重要特征。

## 为什么要对房价取对数

房价通常跨度很大，便宜房和贵房之间可能相差数倍甚至更多。直接预测原始房价时，模型容易被高价房影响。

代码中使用：

```python
y = np.log1p(df[TARGET])
```

它的作用是对房价做 `log(1 + price)` 变换，使目标值分布更平滑。评估时再使用：

```python
predictions = np.expm1(predictions_log)
actual = np.expm1(y_test)
```

把预测值和真实值还原回真实房价后，再计算 RMSE、MAE 和 R2。

## 各模块和函数说明

### `src/config.py`

这个文件保存全局配置，避免在多个文件里重复写路径或列名。

| 名称 | 作用 |
| --- | --- |
| `ROOT` | 项目根目录，也就是 `house-price-prediction-starter` 文件夹。 |
| `DATA_PATH` | 训练数据路径，默认是 `data/raw/train.csv`。 |
| `TARGET` | 预测目标列名，House Prices 数据集中房价列为 `SalePrice`。 |

### `src/load_data.py`

#### `load_data() -> pd.DataFrame`

作用：读取训练数据，并返回一个 `pandas.DataFrame`。

执行逻辑：

1. 检查 `DATA_PATH` 是否存在。
2. 如果数据文件不存在，抛出 `FileNotFoundError`，提示用户下载 Kaggle 数据并放到指定路径。
3. 如果文件存在，使用 `pd.read_csv(DATA_PATH)` 读取数据。
4. 返回读取后的表格数据。

这个函数把“数据从哪里来”封装起来，后续训练代码只需要调用 `load_data()`，不需要关心具体文件路径。

### `src/add_features.py`

#### `add_features(df: pd.DataFrame) -> pd.DataFrame`

作用：基于原始字段添加更容易被模型利用的新特征。

执行逻辑：

1. 使用 `df.copy()` 复制一份数据，避免直接修改传入的原始数据。
2. 如果数据中存在 `TotalBsmtSF`、`1stFlrSF`、`2ndFlrSF` 三列，则新增：

```text
TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF
```

`TotalSF` 表示房屋总面积，比单独看地下室、一楼、二楼面积更直观。

3. 如果数据中存在 `YrSold` 和 `YearBuilt` 两列，则新增：

```text
HouseAge = YrSold - YearBuilt
```

`HouseAge` 表示房屋出售时的房龄。

4. 如果数据中存在 `YrSold` 和 `YearRemodAdd` 两列，则新增：

```text
RemodAge = YrSold - YearRemodAdd
```

`RemodAge` 表示距离最近一次改造或装修过去了多少年。

5. 返回添加新特征后的数据。

代码中使用 `issubset(df.columns)` 判断列是否存在，这样即使数据集缺少某些列，也不会因为直接访问不存在的列而报错。

### `src/build_preprocessor.py`

#### `build_preprocessor(x: pd.DataFrame) -> ColumnTransformer`

作用：创建数据预处理流程，让数值列和类别列使用不同的处理方式。

执行逻辑：

1. 使用 `select_dtypes(include=["number"])` 找出数值列，例如面积、年份、车库面积等。
2. 使用 `select_dtypes(include=["object", "string"])` 找出类别列，例如社区、房屋类型、装修质量等。
3. 为数值列创建 `numeric_pipeline`：

```text
SimpleImputer(strategy="median") -> StandardScaler()
```

含义：

- `SimpleImputer(strategy="median")`：用中位数填补缺失值。
- `StandardScaler()`：标准化数值，让不同量纲的特征更容易被模型学习。

4. 为类别列创建 `categorical_pipeline`：

```text
SimpleImputer(strategy="most_frequent") -> OneHotEncoder(handle_unknown="ignore")
```

含义：

- `SimpleImputer(strategy="most_frequent")`：用出现次数最多的类别填补缺失值。
- `OneHotEncoder(handle_unknown="ignore")`：把文字类别转换成 0/1 特征，并且遇到训练集中没见过的新类别时不会报错。

5. 使用 `ColumnTransformer` 把两条预处理流程合并：

- 数值列走 `numeric_pipeline`。
- 类别列走 `categorical_pipeline`。

这个函数返回的是预处理器本身，不会立刻处理数据。真正执行预处理是在 `Pipeline.fit()` 和 `Pipeline.predict()` 过程中自动完成的。

### `src/get_models.py`

#### `get_models() -> dict`

作用：返回一个字典，里面保存项目要训练和对比的模型。

当前包含两个模型：

| 模型名称 | 模型 | 作用 |
| --- | --- | --- |
| 线性回归基线模型 | `LinearRegression()` | 作为简单基线，方便判断复杂模型是否真的带来提升。 |
| 随机森林模型 | `RandomForestRegressor()` | 使用多棵决策树进行集成预测，通常能捕捉非线性关系。 |

随机森林参数说明：

| 参数 | 含义 |
| --- | --- |
| `n_estimators=300` | 使用 300 棵树。树越多通常越稳定，但训练更慢。 |
| `random_state=42` | 固定随机种子，让每次运行结果更容易复现。 |
| `n_jobs=1` | 使用 1 个 CPU 进程训练。 |
| `min_samples_leaf=2` | 每个叶子节点至少保留 2 个样本，有助于降低过拟合。 |

### `src/evaluate_model.py`

#### `evaluate_model(name: str, model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> None`

作用：在测试集上评估模型，并打印 RMSE、MAE、R2 三个指标。

执行逻辑：

1. 使用 `model.predict(x_test)` 得到预测结果。这里得到的是对数房价。
2. 使用 `np.expm1()` 把预测结果从对数房价还原成真实房价。
3. 使用 `np.expm1()` 把测试集真实值也还原成真实房价。
4. 计算评估指标：

| 指标 | 含义 | 判断方式 |
| --- | --- | --- |
| RMSE | 均方根误差，对大误差更敏感。 | 越小越好。 |
| MAE | 平均绝对误差，更容易理解为平均预测偏差。 | 越小越好。 |
| R2 | 模型解释目标变量变化的能力。 | 越接近 1 越好。 |

5. 打印模型名称和评估结果。

这个函数只负责评估和打印，不返回新数据。

### `src/show_feature_importance.py`

#### `show_random_forest_feature_importance(model: Pipeline) -> None`

作用：输出随机森林模型认为最重要的前 15 个特征。

执行逻辑：

1. 从 `Pipeline` 中取出预处理器：

```python
preprocessor = model.named_steps["preprocessor"]
```

2. 从 `Pipeline` 中取出随机森林模型：

```python
regressor = model.named_steps["model"]
```

3. 使用 `preprocessor.get_feature_names_out()` 获取预处理后的特征名。

类别变量经过 One-Hot 编码后会拆成很多新列，例如 `Neighborhood` 可能会变成多个类似 `cat__Neighborhood_NAmes` 的特征。

4. 使用 `regressor.feature_importances_` 获取随机森林计算出的特征重要性。
5. 把特征名和重要性放入 `DataFrame`。
6. 按重要性从高到低排序。
7. 取前 15 个特征并打印。

注意：这个函数适用于随机森林这类带有 `feature_importances_` 属性的模型，不适用于普通线性回归。

### `src/train.py`

#### `main() -> None`

作用：项目主入口，负责串联完整机器学习流程。

完整逻辑：

1. 读取数据：

```python
df = load_data()
```

2. 添加新特征：

```python
df = add_features(df)
```

3. 检查目标列：

```python
if TARGET not in df.columns:
    raise ValueError(...)
```

4. 构造输入特征 `x`：

```python
x = df.drop(columns=[TARGET, "Id"], errors="ignore")
```

这里会移除：

- `SalePrice`：因为它是要预测的目标，不能作为输入特征。
- `Id`：样本编号通常不包含有价值的预测信息。

5. 构造目标值 `y`：

```python
y = np.log1p(df[TARGET])
```

6. 拆分训练集和测试集：

```python
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)
```

其中：

- `test_size=0.2` 表示 20% 数据用于测试，80% 数据用于训练。
- `random_state=42` 用于保证每次拆分结果一致。

7. 构建预处理器：

```python
preprocessor = build_preprocessor(x_train)
```

8. 获取模型：

```python
models = get_models()
```

9. 遍历每个模型，创建完整训练流水线：

```python
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", estimator),
    ]
)
```

这个 `Pipeline` 的好处是，训练和预测时都会自动执行同样的预处理步骤，减少手动处理数据出错的机会。

10. 训练模型：

```python
pipeline.fit(x_train, y_train)
```

11. 评估模型：

```python
evaluate_model(name, pipeline, x_test, y_test)
```

12. 保存训练后的模型：

```python
trained_models[name] = pipeline
```

13. 输出随机森林重要特征：

```python
show_random_forest_feature_importance(trained_models["随机森林模型"])
```

#### `if __name__ == "__main__":`

这段代码表示：只有当直接运行 `python src/train.py` 时，才会执行 `main()`。如果其他文件导入 `train.py`，则不会自动开始训练。

## 模型评估指标解释

### RMSE

RMSE 是 Root Mean Squared Error，中文常叫均方根误差。它会对较大的预测错误给予更高惩罚。

例如模型对某些高价房预测偏差很大，RMSE 会明显变高。

### MAE

MAE 是 Mean Absolute Error，中文常叫平均绝对误差。它表示预测值和真实值平均相差多少。

如果 MAE 是 18000，大致可以理解为模型平均预测误差约为 18000 美元。

### R2

R2 表示模型对房价变化的解释能力。一般来说：

- 越接近 1，说明模型解释能力越强。
- 接近 0，说明模型效果接近直接预测平均值。
- 小于 0，说明模型效果很差。

## 当前模型思路

项目中同时训练两个模型：

1. 线性回归基线模型

线性回归简单、速度快、容易理解。它适合作为基线模型：如果复杂模型比线性回归好，说明复杂模型可能学到了更多非线性关系。

2. 随机森林模型

随机森林由多棵决策树组成，能处理更复杂的特征关系。对于房价这种受多个因素共同影响的问题，随机森林通常比简单线性模型更灵活。

## 可以继续改进的方向

- 增加更多特征工程，例如总浴室数、总门廊面积、是否翻新、房屋质量组合特征等。
- 尝试更多模型，例如 Ridge、Lasso、GradientBoostingRegressor、XGBoost、LightGBM。
- 使用交叉验证，让评估结果更稳定。
- 对高偏态数值特征做对数变换。
- 保存训练好的模型，方便后续直接预测新房屋价格。
- 增加预测脚本，例如输入一条房屋信息并输出预测价格。
- 添加可视化分析，例如房价分布、面积与房价关系、重要特征柱状图。

## 推荐学习路线

1. 先运行 `python src/train.py`，观察模型输出。
2. 阅读 `src/train.py`，理解整体训练流程。
3. 阅读 `src/build_preprocessor.py`，理解数值列和类别列如何分别处理。
4. 阅读 `src/add_features.py`，尝试添加自己的新特征。
5. 阅读 `src/evaluate_model.py`，理解 RMSE、MAE、R2 的含义。
6. 修改 `src/get_models.py`，尝试加入新的模型。
7. 对比不同模型的评估结果，判断哪个模型更适合当前数据。

## 简历描述示例

房价预测机器学习项目：基于 Kaggle House Prices 数据集，完成数据读取、缺失值处理、类别特征编码、特征工程、回归建模与模型评估。使用线性回归作为基线模型，并引入随机森林回归模型提升预测能力；通过 RMSE、MAE 和 R2 指标评估模型表现，同时分析随机森林特征重要性，识别影响房价的关键因素。
