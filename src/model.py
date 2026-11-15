"""Model training, evaluation and inference helpers (ports of the original cells)."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

__all__ = ["split_and_scale", "train_linear_regression", "train_random_forest",
           "build_residual_table", "plot_actual_vs_predicted",
           "plot_residual_distribution", "plot_residuals_vs_predicted",
           "compare_models", "physical_parameters", "export_model_artifacts",
           "tlc_reference_fare", "predict_fare"]

RANDOM_STATE = 42


def split_and_scale(df2: pd.DataFrame):
    """Separate X / y, one-hot ``vendorid``, 75/25 split, standardize features.

    Returns ``X, y, X_train_scaled, X_test_scaled, y_train, y_test, scaler``.
    """
    # 1. Separate features (X) and target (y)
    X = df2.drop(columns=['fare_amount'])
    y = df2['fare_amount']

    # One-hot encode the categorical 'vendorid' column (numeric 1 or 2)
    X = pd.get_dummies(X, columns=['vendorid'], drop_first=True)

    # 2. Train-Test Split (75% train, 25% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE)

    # 3. Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"X_train shape: {X_train_scaled.shape}")
    print(f"X_test shape: {X_test_scaled.shape}")
    return X, y, X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_linear_regression(X_train_scaled, y_train, X_test_scaled, y_test):
    """Fit and evaluate the baseline Linear Regression.

    Returns ``lr_model, y_pred, rmse, r2``.
    """
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)

    y_pred = lr_model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"\n--- Baseline Linear Regression Performance ---")
    print(f"RMSE: ${rmse:.2f}")
    print(f"R^2 Score: {r2:.4f}")
    return lr_model, y_pred, rmse, r2


def train_random_forest(X_train_scaled, y_train, X_test_scaled, y_test,
                        n_estimators: int = 100, max_depth: int = 10):
    """Fit and evaluate the champion Random Forest Regressor.

    Returns ``rf_model, y_pred_rf, rmse_rf, r2_rf``.
    """
    rf_model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                     random_state=RANDOM_STATE, n_jobs=-1)

    print("Training Random Forest Regressor...")
    rf_model.fit(X_train_scaled, y_train)
    print("Model training complete!")

    y_pred_rf = rf_model.predict(X_test_scaled)
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    rmse_rf = np.sqrt(mse_rf)
    r2_rf = r2_score(y_test, y_pred_rf)

    print(f"\n--- Champion Random Forest Performance ---")
    print(f"RMSE: ${rmse_rf:.2f}")
    print(f"R^2 Score: {r2_rf:.4f}")
    return rf_model, y_pred_rf, rmse_rf, r2_rf


def build_residual_table(y_test, y_pred_rf) -> pd.DataFrame:
    """Actual / Predicted / Residual frame (prints head and residual statistics)."""
    rf_residuals_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred_rf,
        'Residual': y_test - y_pred_rf,
    })
    print("--- Model Predictions and Residuals (Random Forest) ---")
    display(rf_residuals_df.head(10))

    print("\n--- Residual Statistics ---")
    display(rf_residuals_df['Residual'].describe())
    return rf_residuals_df


def plot_actual_vs_predicted(y_test, y_pred_rf) -> None:
    """Scatterplot of actual vs. predicted fares with a 45-degree line."""
    plt.figure(figsize=(8, 6))
    sns.set_theme(style='whitegrid')
    sns.scatterplot(x=y_test, y=y_pred_rf, alpha=0.4, color='dodgerblue', edgecolor=None, s=15)

    max_val = max(y_test.max(), y_pred_rf.max())
    plt.plot([0, max_val], [0, max_val], color='red', linestyle='--', linewidth=2, label='Perfect Prediction Line')

    plt.title('Actual vs. Predicted Fare Amounts (Random Forest Model Validation)', fontsize=14, pad=15)
    plt.xlabel('Actual Fare ($)', fontsize=12)
    plt.ylabel('Predicted Fare ($)', fontsize=12)
    plt.xlim(0, max_val + 5)
    plt.ylim(0, max_val + 5)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()


def plot_residual_distribution(rf_residuals_df: pd.DataFrame) -> None:
    """Histogram of the residuals."""
    sns.histplot(rf_residuals_df['Residual'], bins=np.arange(-20, 20, 0.5))
    plt.title('Distribution of the residuals')
    plt.xlabel('residual value')
    plt.ylabel('count')
    plt.show()


def plot_residuals_vs_predicted(rf_residuals_df: pd.DataFrame) -> None:
    """Scatterplot of residuals over predicted values."""
    sns.scatterplot(x='Predicted', y='Residual', data=rf_residuals_df)
    plt.axhline(0, c='red')
    plt.title('Scatterplot of residuals over predicted values')
    plt.xlabel('predicted value')
    plt.ylabel('residual value')
    plt.show()


def compare_models(lr_model, rf_model, feature_names) -> pd.DataFrame:
    """Linear-regression coefficients next to Random-Forest importances (transposed)."""
    model_comparison_df = pd.DataFrame({
        'Linear Regression Coef': lr_model.coef_,
        'Random Forest Importance': rf_model.feature_importances_,
    }, index=feature_names)
    return model_comparison_df.T


# --------------------------------------------------------------------------- #
# Optional extras (not in the original notebook): web calculator export
# --------------------------------------------------------------------------- #
def physical_parameters(lr_model, scaler, feature_names) -> pd.DataFrame:
    """Convert standardized coefficients back to original units (per mile, per minute...).

    With z = (x - mu) / sigma:  slope_j = b_j / sigma_j,
    intercept = b0 - sum(b_j * mu_j / sigma_j).
    """
    slopes = lr_model.coef_ / scaler.scale_
    intercept = lr_model.intercept_ - np.sum(lr_model.coef_ * scaler.mean_ / scaler.scale_)
    table = pd.DataFrame({"standardized_coef": lr_model.coef_,
                          "feature_std": scaler.scale_,
                          "physical_slope": slopes}, index=list(feature_names))
    table.attrs["intercept"] = float(intercept)
    return table


def export_model_artifacts(lr_model, scaler, rf_model, feature_names,
                           root: Path | None = None) -> dict:
    """Save models to ``models/`` and calculator parameters to ``web/model_params.json``."""
    from src.api import PROJECT_ROOT
    root = Path(root or PROJECT_ROOT)
    (root / "models").mkdir(exist_ok=True)
    (root / "web").mkdir(exist_ok=True)
    joblib.dump({"model": rf_model, "scaler": scaler, "features": list(feature_names)},
                root / "models" / "random_forest.joblib")
    joblib.dump({"model": lr_model, "scaler": scaler, "features": list(feature_names)},
                root / "models" / "linear_regression.joblib")

    table = physical_parameters(lr_model, scaler, feature_names)
    params = {"intercept": table.attrs["intercept"],
              "coefficients": table["physical_slope"].to_dict()}
    (root / "web" / "model_params.json").write_text(json.dumps(params, indent=2))
    print("Saved models/ and web/model_params.json")
    display(table.round(4))
    return params


def tlc_reference_fare(distance_miles: float, duration_min: float) -> float:
    """Simplified TLC standard metered fare (before surcharges).

    $3.00 flag drop; $0.70 per 1/5 mile above 12 mph, $0.70 per minute at or
    below 12 mph, applied to the whole trip.
    """
    speed = distance_miles / (duration_min / 60) if duration_min > 0 else 0
    variable = distance_miles * 5 * 0.70 if speed > 12 else duration_min * 0.70
    return 3.00 + variable


def predict_fare(params: dict, mean_distance: float, mean_duration: float,
                 passenger_count: int = 1, rush_hour: int = 0, vendor_2: int = 1) -> float:
    """Predict a fare from exported physical-scale linear parameters."""
    c = params["coefficients"]
    return (params["intercept"] + c["mean_distance"] * mean_distance
            + c["mean_duration"] * mean_duration + c["passenger_count"] * passenger_count
            + c["rush_hour"] * rush_hour + c["vendorid_2"] * vendor_2)
