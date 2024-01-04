import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the dataset
df = pd.read_csv("/content/wrestling_data.csv")

# Dropping irrelevant columns and handle missing values


# Box plot
plt.figure(figsize=(15, 6))
sns.boxplot(x='nationality', y='rank', data=df)
plt.title('Box Plot of Nationality vs. Final Rank')
plt.xlabel('Nationality')
plt.ylabel('Final Rank')
plt.xticks(rotation=45, ha='right')  # Rotating x-axis labels for better visibility
plt.show()

# Calculating correlation coefficient
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['nationality_encoded'] = le.fit_transform(df['nationality'])

correlation_coefficient = df['nationality_encoded'].corr(df['rank'])
print(f"Correlation Coefficient between Nationality and Final Rank: {correlation_coefficient}")
