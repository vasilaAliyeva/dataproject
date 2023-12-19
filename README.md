# dataproject
# Wrestling World Tournament
Jupyter notebook to predict the final rank of wrestlers in a wrestling tournament.

-- Project Status: [Completed]

## Methods used
-- Data Visualization 

-- Data Analysis 


## Technologies
-- Python

### Libraries
-- Numpy

-- Pandas

-- Matplotlib

-- Seaborn

-- Scikit-learn


## Motivation
Wrestling tournaments are vital in the sporting world, and predicting wrestler ranks can have a significant impact on federations' performance strategies.

## Problem Description
The dataset is diverse, including wrestler attributes such as gender, age, height, weight, nationality, mastered sports, practice hours per day, and scores in strength, agility, and mental abilities. 
Challenges include careful preprocessing for modeling, handling data quality issues, and predicting ranks due to the multifaceted nature of wrestling outcomes.

## Project Objectives
- Goal: Predict the final rank of wrestlers.
- Intended Outcomes: Build a precise predictive model, offering valuable insights for athletes and federations in future tournaments.

## Data Overview
The dataset contains information on wrestling tournament participants, including physical details, skill scores, and personal information.

## Data Preprocessing
- Drop irrelevant columns.
- Impute missing values.
- Clip outliers in 'height' and 'weight' columns.

## Model Training
- Features: Gender, age, height, weight, nationality, mastered sports, practice hours per day, strength score, agility score, mental score.
- Target Variable: Final rank.
- Model: Random Forest Regressor.

## Feature Importance
Visualize the importance of each feature in predicting wrestler ranks.

## Project Structure
- data/: Folder for storing the dataset.
- notebooks/: Jupyter notebooks for analysis and model development.
- scripts/: Python scripts for data preprocessing and model training.
- README.md: Project documentation.

## Acknowledgements
Dataset:[Wrestling World Tournament](https://www.kaggle.com/datasets/julienjta/wrestling-world-tournament)
