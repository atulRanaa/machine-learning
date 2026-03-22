"""
Lab: Gradient Boosting from Scratch + XGBoost Tuning
=====================================================
"""
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error


class GradientBoostingFromScratch:
    """Minimal gradient boosting regressor using decision tree stumps."""

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.lr = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.init_pred = None

    def fit(self, X, y):
        self.init_pred = y.mean()
        F = np.full(len(y), self.init_pred)

        for m in range(self.n_estimators):
            residuals = y - F
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            self.trees.append(tree)
            F += self.lr * tree.predict(X)

            if m % 50 == 0:
                mse = np.mean((y - F) ** 2)
                print(f"  Iteration {m:3d} | Train MSE: {mse:.6f}")

        return self

    def predict(self, X):
        F = np.full(X.shape[0], self.init_pred)
        for tree in self.trees:
            F += self.lr * tree.predict(X)
        return F


if __name__ == "__main__":
    X, y = fetch_california_housing(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("=" * 60)
    print("GRADIENT BOOSTING FROM SCRATCH")
    print("=" * 60)
    gb = GradientBoostingFromScratch(n_estimators=200, learning_rate=0.1, max_depth=4)
    gb.fit(X_train, y_train)
    y_pred = gb.predict(X_test)
    print(f"Test MSE: {mean_squared_error(y_test, y_pred):.4f}")

    # XGBoost comparison
    try:
        from xgboost import XGBRegressor

        print("\n" + "=" * 60)
        print("XGBOOST COMPARISON")
        print("=" * 60)
        xgb = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42)
        xgb.fit(X_train, y_train)
        y_pred_xgb = xgb.predict(X_test)
        print(f"XGBoost Test MSE: {mean_squared_error(y_test, y_pred_xgb):.4f}")
    except ImportError:
        print("XGBoost not installed. pip install xgboost")
