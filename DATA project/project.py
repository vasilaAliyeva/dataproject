import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt

# Function to load and preprocess wrestling data
def load_wrestling_data(file_path):
    wrestling_data = pd.read_csv('/content/wrestling_data.csv')
    wrestling_data = wrestling_data.dropna()

    # Perform label encoding for categorical variables if needed
    # For simplicity, let's assume 'gender', 'nationality', and 'sports' are categorical
    wrestling_data['gender'] = pd.Categorical(wrestling_data['gender']).codes
    wrestling_data['nationality'] = pd.Categorical(wrestling_data['nationality']).codes
    wrestling_data['sports'] = pd.Categorical(wrestling_data['sports']).codes

    return wrestling_data

# Function to visualize feature importance
def plot_feature_importance(model, feature_names):
    feature_importance = model.feature_importances_
    sorted_idx = feature_importance.argsort()

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
    plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title('Random Forest Feature Importance')
    plt.show()

# Function to train a random forest regressor and make predictions
def train_random_forest(data):
    # Assuming 'rank' is the target variable
    X = data[['gender', 'age', 'height', 'weight', 'nationality', 'sports', 'hours_per_day', 'strength', 'agility', 'mental']]
    y = data['rank']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a random forest regressor model and fit it to the training data
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    # Make predictions on the test data
    predictions = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, predictions)
    print(f'Mean Squared Error: {mse}')

    # Visualize feature importance
    feature_names = X.columns
    plot_feature_importance(model, feature_names)

# Main wrestling application
def wrestling_main():
    file_path = 'your_dataset.csv'  # Replace with the actual path to your dataset
    wrestling_data = load_wrestling_data(file_path)

    # Train a random forest regressor and make predictions
    train_random_forest(wrestling_data)

if __name__== "__main__":
    wrestling_main()
