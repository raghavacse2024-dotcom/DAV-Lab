import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

uci_diabetes = pd.read_csv("uci_diabetes.csv")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.histplot(uci_diabetes["Glucose"], kde=True, stat="density", linewidth=0, ax=axes[0])
x = np.linspace(uci_diabetes["Glucose"].min(), uci_diabetes["Glucose"].max(), 100)
axes[0].plot(x, norm.pdf(x, uci_diabetes["Glucose"].mean(), uci_diabetes["Glucose"].std()), linewidth=2)
axes[0].set_title("Normal Curve - Glucose")

sns.histplot(uci_diabetes["BMI"], kde=True, stat="density", linewidth=0, ax=axes[1])
x = np.linspace(uci_diabetes["BMI"].min(), uci_diabetes["BMI"].max(), 100)
axes[1].plot(x, norm.pdf(x, uci_diabetes["BMI"].mean(), uci_diabetes["BMI"].std()), linewidth=2)
axes[1].set_title("Normal Curve - BMI")

plt.tight_layout()
plt.savefig("normal_curves.png", dpi=150)
plt.show()
