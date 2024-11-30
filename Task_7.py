#%% Task 7: Model Selection and Setup
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

#%% Step 1: Load the dataset
df = pd.read_csv('yield_df_pca.csv')

# Verify the dataset
print("First few rows of the dataset:")
print(df.head())

# Check for missing values
print("\nMissing values in the dataset:")
print(df.isnull().sum())

#%% Step 2: Separate features and target variable
X = df.drop('yield_t_per_ha', axis=1)
y = df['yield_t_per_ha']

# Verify the shapes
print("\nShape of Features (X):", X.shape)
print("Shape of Target (y):", y.shape)

#%% Step 3: Split into training and testing sets
# Set a random seed for reproducibility
random_seed = 42

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=random_seed
)

# Verify the shapes of the splits
print("\nTraining set shape (X_train):", X_train.shape)
print("Test set shape (X_test):", X_test.shape)
print("Training set shape (y_train):", y_train.shape)
print("Test set shape (y_test):", y_test.shape)

#%% Step 4: Initialize models
# Linear Regression model
lin_reg = LinearRegression()

# Decision Tree Regressor model
dec_tree_reg = DecisionTreeRegressor(random_state=random_seed)

#%% Models are now ready for training in Task 8
print("\nModels initialized and ready for training.")

# %%
