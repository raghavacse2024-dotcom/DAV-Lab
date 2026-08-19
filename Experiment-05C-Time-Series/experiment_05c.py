import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA

# Expected input file: diabetes9.csv

diabetes_data = pd.read_csv("diabetes9.csv")
print(diabetes_data.head())

plt.figure(figsize=(12, 5))
plt.plot(diabetes_data['Glucose'], label="Glucose Level")
plt.xlabel("Index")
plt.ylabel("Glucose Level")
plt.title("Time Series of Glucose Levels")
plt.legend()
plt.tight_layout()
plt.savefig("glucose_time_series.png", dpi=150)
plt.show()

# The manual uses period=30 for decomposition.
decomposition = seasonal_decompose(
    diabetes_data['Glucose'], model='additive', period=30
)
fig, axes = plt.subplots(3, 1, figsize=(12, 8))
decomposition.trend.plot(ax=axes[0], title="Trend Component")
decomposition.seasonal.plot(ax=axes[1], title="Seasonal Component")
decomposition.resid.plot(ax=axes[2], title="Residual Component")
plt.tight_layout()
plt.savefig("time_series_decomposition.png", dpi=150)
plt.show()

# Moving Average
diabetes_data['Glucose_MA'] = diabetes_data['Glucose'].rolling(window=7).mean()
plt.figure(figsize=(12, 5))
plt.plot(diabetes_data['Glucose'], label="Original", alpha=0.5)
plt.plot(diabetes_data['Glucose_MA'], label="7-day Moving Average")
plt.legend()
plt.title("Moving Average Smoothing")
plt.tight_layout()
plt.savefig("moving_average.png", dpi=150)
plt.show()

# ARIMA forecasting
train_size = int(len(diabetes_data) * 0.8)
train = diabetes_data['Glucose'][:train_size]
test = diabetes_data['Glucose'][train_size:]

model = ARIMA(train, order=(5, 1, 0))
fitted_model = model.fit()
forecast = fitted_model.forecast(steps=len(test))

plt.figure(figsize=(12, 5))
plt.plot(range(len(test)), test, label="Actual")
plt.plot(range(len(test)), forecast, label="Forecast")
plt.xlabel("Index")
plt.ylabel("Glucose Level")
plt.title("ARIMA Model Forecasting")
plt.legend()
plt.tight_layout()
plt.savefig("arima_forecast.png", dpi=150)
plt.show()
