import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score

uci_diabetes = pd.read_csv("uci_diabetes.csv")
pima_diabetes = pd.read_csv("pima_diabetes.csv")

print("UCI Diabetes Dataset Sample:\n", uci_diabetes.head())
print("\nPima Indians Diabetes Dataset Sample:\n", pima_diabetes.head())


def linear_regression_analysis(df, x_column, y_column, title):
    X = df[[x_column]]
    y = df[y_column]
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)

    print(f"\nLinear Regression (Predicting {y_column} using {x_column}):")
    print(f"R² Score: {r2:.4f}")

    plt.figure(figsize=(7, 5))
    plt.scatter(X, y, label="Actual Data")
    plt.plot(X, y_pred, linewidth=2, label="Regression Line")
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return r2

linear_regression_analysis(uci_diabetes, "Glucose", "BMI", "UCI: Glucose vs BMI")
linear_regression_analysis(pima_diabetes, "Glucose", "BMI", "Pima: Glucose vs BMI")


def logistic_regression_analysis(df, features, target, dataset_name):
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nLogistic Regression ({dataset_name}):")
    print(f"Accuracy Score: {accuracy:.4f}")
    return accuracy

features = ["Glucose", "BloodPressure", "BMI", "Age"]
target = "Outcome"

logistic_regression_analysis(uci_diabetes, features, target, "UCI Diabetes")
logistic_regression_analysis(pima_diabetes, features, target, "Pima Indians Diabetes")
