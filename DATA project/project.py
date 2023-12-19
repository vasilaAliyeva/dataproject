import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# Function to load and preprocess wrestling data
def load_wrestling_data(file_path):
    wrestling_data = pd.read_csv(file_path)
    wrestling_data = wrestling_data.dropna()
    wrestling_data['Date of Birth'] = pd.to_datetime(wrestling_data['Date of Birth'], format='%d-%m-%Y')
    
    # Perform label encoding for categorical variables if needed
    # For simplicity, let's assume 'Gender', 'Nationality', and 'Mastered Sports' are categorical
    le = LabelEncoder()
    wrestling_data['Gender'] = le.fit_transform(wrestling_data['Gender'])
    wrestling_data['Nationality'] = le.fit_transform(wrestling_data['Nationality'])
    wrestling_data['Mastered Sports'] = le.fit_transform(wrestling_data['Mastered Sports'])
    
    global gender_mapping, nationality_mapping, sports_mapping
    gender_mapping = {index: label for index, label in enumerate(le.classes_)}
    nationality_mapping = {index: label for index, label in enumerate(le.classes_)}
    sports_mapping = {index: label for index, label in enumerate(le.classes_)}
    
    return wrestling_data

# Function to show gender, nationality, and sports mappings
def show_wrestling_mappings():
    print("Gender Mapping:")
    for gender, number in gender_mapping.items():
        print(f"{gender}: {number}")
    
    print("\nNationality Mapping:")
    for nationality, number in nationality_mapping.items():
        print(f"{nationality}: {number}")
    
    print("\nMastered Sports Mapping:")
    for sport, number in sports_mapping.items():
        print(f"{sport}: {number}")

# Function to predict final rank of wrestlers
def predict_final_rank(data):
    # Assuming 'Rank' is the target variable
    X = data[['Gender', 'Age', 'Height', 'Weight', 'Nationality', 'Mastered Sports', 
              'Practice Hours per Day', 'Strength Score', 'Agility Score', 'Mental Score']]
    y = data['Rank']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create a linear regression model and fit it to the training data
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions on the test data
    predictions = model.predict(X_test)
    
    return predictions

# Main wrestling application
def wrestling_main():
    file_path = 'wrestling_tournament_data.csv'
    wrestling_data = load_wrestling_data(file_path)

    while True:
        print("\nSelect an option:")
        print("1. View gender, nationality, and sports mappings")
        print("2. Predict final rank of wrestlers")
        print("3. Exit")
        choice = input("Enter your choice (1-3): ")

        if choice == '1':
            show_wrestling_mappings()
        elif choice == '2':
            predictions = predict_final_rank(wrestling_data)
            print("Predicted final rank of wrestlers:")
            print(predictions)
        elif choice == '3':
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 3.")

if __name__ == "__main__":
    wrestling_main()
