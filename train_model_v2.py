# train_model_v2.py
# Improved model: uses units_sold + region + product with OneHotEncoder

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import joblib

# 1. Load dataset
df = pd.read_csv("sales.csv")

# 2. Features and target
X = df[["units_sold", "region", "product"]]
y = df["revenue"]

# 3. Preprocessing: one-hot encode categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ("onehot", OneHotEncoder(), ["region", "product"])
    ],
    remainder="passthrough"
)

# 4. Build pipeline: preprocessing + regression
model_v2 = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("regressor", LinearRegression())
])

# 5. Train improved model
model_v2.fit(X, y)

# 6. Save improved model
joblib.dump(model_v2, "revenue_model_v2.pkl")
print("Improved model saved as revenue_model_v2.pkl")
