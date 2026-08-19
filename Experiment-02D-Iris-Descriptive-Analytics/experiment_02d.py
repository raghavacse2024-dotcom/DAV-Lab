import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Expected input file: iris_dataset(2d).csv

df = pd.read_csv('iris_dataset(2d).csv')

print("Basic Information:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())

print("\nSpecies Count:")
print(df['species'].value_counts())

# Feature distributions
axes = df.hist(figsize=(8, 6), edgecolor='black')
plt.suptitle('Feature Distributions')
plt.tight_layout()
plt.savefig('feature_distributions.png', dpi=150)
plt.close()

# Boxplot
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='species', y='sepal length (cm)')
plt.title('Sepal Length Comparison')
plt.tight_layout()
plt.savefig('sepal_length_comparison.png', dpi=150)
plt.close()

# Pairplot
pair = sns.pairplot(df, hue='species')
pair.savefig('iris_pairplot.png', dpi=150)
plt.close('all')

print("Plots saved: feature_distributions.png, sepal_length_comparison.png, iris_pairplot.png")
