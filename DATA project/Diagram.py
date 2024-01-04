import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Load the dataset
df = pd.read_csv("/content/wrestling_data.csv")

# Drop irrelevant columns and handle missing values
# You can use the preprocessing steps from the previous code

# Scatter plot with linear regression line
plt.figure(figsize=(12, 8))
sns.scatterplot(x='age', y='rank', data=df, alpha=0.7, label='Data Points')
sns.regplot(x='age', y='rank', data=df, scatter=False, color='red', label='Linear Regression Line')
plt.title('Scatter Plot of Age vs. Final Rank')
plt.xlabel('Age')
plt.ylabel('Final Rank')
plt.legend()
plt.show()

# Calculate correlation coefficient
correlation_coefficient = df['age'].corr(df['rank'])
print(f"Correlation Coefficient between Age and Final Rank: {correlation_coefficient}")

# Correlation matrix
correlation_matrix = df[['age', 'rank', 'height', 'weight', 'strength', 'agility', 'mental']].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# Distribution of Age and Rank
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(df['age'], bins=20, kde=True, color='skyblue')
plt.title('Distribution of Age')

plt.subplot(1, 2, 2)
sns.histplot(df['rank'], bins=20, kde=True, color='salmon')
plt.title('Distribution of Final Rank')

plt.tight_layout()
plt.show()
