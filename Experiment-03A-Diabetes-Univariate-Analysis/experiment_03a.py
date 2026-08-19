import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

uci_diabetes = pd.read_csv("uci_diabetes.csv")
pima_diabetes = pd.read_csv("pima_diabetes.csv")

numerical_columns = [
    "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI",
    "DiabetesPedigreeFunction", "Age"
]


def univariate_analysis(df, columns):
    stats = {}
    for col in columns:
        stats[col] = {
            "Mean": np.mean(df[col]),
            "Median": np.median(df[col]),
            "Mode": df[col].mode()[0],
            "Variance": np.var(df[col], ddof=1),
            "Standard Deviation": np.std(df[col], ddof=1),
            "Skewness": skew(df[col]),
            "Kurtosis": kurtosis(df[col]),
        }
    return pd.DataFrame(stats).T

print("UCI Diabetes Dataset Sample:")
print(uci_diabetes.head())
print("\nPima Indians Diabetes Dataset Sample:")
print(pima_diabetes.head())

print("\nUCI Diabetes Dataset Statistics:")
print(univariate_analysis(uci_diabetes, numerical_columns))

print("\nPima Indians Diabetes Dataset Statistics:")
print(univariate_analysis(pima_diabetes, numerical_columns))
