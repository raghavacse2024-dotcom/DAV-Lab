import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

uci_diabetes = pd.read_csv("uci_diabetes.csv")
pima_diabetes = pd.read_csv("pima_diabetes.csv")

features = ["Glucose", "BloodPressure", "BMI"]
target = "Outcome"

X_uci, y_uci = uci_diabetes[features], uci_diabetes[target]
X_pima, y_pima = pima_diabetes[features], pima_diabetes[target]

X_train_uci, X_test_uci, y_train_uci, y_test_uci = train_test_split(
    X_uci, y_uci, test_size=0.2, random_state=42
)
X_train_pima, X_test_pima, y_train_pima, y_test_pima = train_test_split(
    X_pima, y_pima, test_size=0.2, random_state=42
)

model_uci = LogisticRegression(max_iter=1000)
model_uci.fit(X_train_uci, y_train_uci)
model_pima = LogisticRegression(max_iter=1000)
model_pima.fit(X_train_pima, y_train_pima)

y_pred_uci = model_uci.predict(X_test_uci)
y_pred_pima = model_pima.predict(X_test_pima)

accuracy_uci = accuracy_score(y_test_uci, y_pred_uci)
precision_uci = precision_score(y_test_uci, y_pred_uci, zero_division=0)
recall_uci = recall_score(y_test_uci, y_pred_uci, zero_division=0)
f1_uci = f1_score(y_test_uci, y_pred_uci, zero_division=0)
accuracy_pima = accuracy_score(y_test_pima, y_pred_pima)
precision_pima = precision_score(y_test_pima, y_pred_pima, zero_division=0)
recall_pima = recall_score(y_test_pima, y_pred_pima, zero_division=0)
f1_pima = f1_score(y_test_pima, y_pred_pima, zero_division=0)

print("UCI Diabetes Dataset - Logistic Regression Results:")
print(f"Accuracy: {accuracy_uci:.4f}, Precision: {precision_uci:.4f}, Recall: {recall_uci:.4f}, F1 Score: {f1_uci:.4f}")
print("\nPima Indians Diabetes Dataset - Logistic Regression Results:")
print(f"Accuracy: {accuracy_pima:.4f}, Precision: {precision_pima:.4f}, Recall: {recall_pima:.4f}, F1 Score: {f1_pima:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(confusion_matrix(y_test_uci, y_pred_uci), annot=True, fmt='d', ax=axes[0])
axes[0].set_title("UCI Diabetes - Confusion Matrix")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")
sns.heatmap(confusion_matrix(y_test_pima, y_pred_pima), annot=True, fmt='d', ax=axes[1])
axes[1].set_title("Pima Indians Diabetes - Confusion Matrix")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150)
plt.show()
