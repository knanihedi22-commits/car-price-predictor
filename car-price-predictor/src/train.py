import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

TARGET = "price"
df = pd.read_csv("data/processed/cars_clean.csv")
X, y = df.drop(columns=[TARGET]), df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = X.select_dtypes(exclude="number").columns.tolist()

preprocess = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])

def make_model(est):
    return TransformedTargetRegressor(
        regressor=Pipeline([("prep", preprocess), ("model", est)]),
        func=np.log1p, inverse_func=np.expm1,
    )

candidates = {
    "Ridge": Ridge(alpha=1.0),
    "RandomForest": RandomForestRegressor(n_estimators=60, max_depth=16, min_samples_leaf=2, n_jobs=-1, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=300, learning_rate=0.1, tree_method="hist", random_state=42, n_jobs=-1),
}

cv_results = {}
for name, est in candidates.items():
    s = cross_val_score(make_model(est), X_train, y_train, cv=3, scoring="neg_mean_absolute_error")
    cv_results[name] = round(float(-s.mean()), 1)
    print(f"{name}: CV MAE = {-s.mean():.0f}")

print("Tuning XGBoost...")
search = RandomizedSearchCV(
    make_model(XGBRegressor(tree_method="hist", random_state=42, n_jobs=-1)),
    param_distributions={
        "regressor__model__n_estimators": [300, 500, 700],
        "regressor__model__max_depth": [4, 6, 8],
        "regressor__model__learning_rate": [0.05, 0.1],
        "regressor__model__subsample": [0.8, 1.0],
    },
    n_iter=6, cv=3, scoring="neg_mean_absolute_error", random_state=42,
)
search.fit(X_train, y_train)
best = search.best_estimator_
print("Best params:", search.best_params_)

pred = best.predict(X_test)
mae = mean_absolute_error(y_test, pred)
r2 = r2_score(y_test, pred)
print(f"TEST MAE: {mae:.0f}   TEST R2: {r2:.3f}")

os.makedirs("models", exist_ok=True)
joblib.dump(best, "models/model.joblib", compress=3)

meta = {
    "num": {c: {"min": float(X[c].min()), "max": float(X[c].max()),
                "median": float(X[c].median())} for c in num_cols},
    "cat": {c: sorted(X[c].unique().tolist()) for c in cat_cols if c != "model"},
    "brand_models": {b: sorted(g["model"].unique().tolist()) for b, g in X.groupby("brand")},
    "metrics": {"cv_mae": cv_results, "test_mae": round(float(mae), 1), "test_r2": round(float(r2), 3)},
}
json.dump(meta, open("models/meta.json", "w"), indent=2)
print("Saved model and meta.")
