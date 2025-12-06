# train_model_v1.py
# Baseline model: only uses units_sold as feature

import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# 1. Load dataset
df = pd.read_csv("sales.csv")

# 2. Select features and target
X = df[["units_sold"]]       # baseline: ONLY units_sold
y = df["revenue"]

# 3. Define and train model
model_v1 = LinearRegression()
model_v1.fit(X, y)

# 4. Save model
joblib.dump(model_v1, "revenue_model_v1.pkl")
print("Baseline model saved as revenue_model_v1.pkl")
