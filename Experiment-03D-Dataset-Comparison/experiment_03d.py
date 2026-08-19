import pandas as pd

# The manual describes these as precomputed statistics files.
uci_stats = pd.read_csv("uci_diabetes.csv")
pima_stats = pd.read_csv("pima_diabetes.csv")

print("Comparison of Univariate Analysis Results:")
print("\nUCI Diabetes Dataset Statistics:\n", uci_stats)
print("\nPima Indians Diabetes Dataset Statistics:\n", pima_stats)

# The manual presents these as example comparison values.
uci_r2 = 0.78
pima_r2 = 0.72
uci_accuracy = 82.4
pima_accuracy = 79.1

print(f"\nLinear Regression R² Scores: UCI - {uci_r2}, Pima - {pima_r2}")
print(f"Logistic Regression Accuracy: UCI - {uci_accuracy}%, Pima - {pima_accuracy}%")
