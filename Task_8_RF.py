# Task_8_RF.py

#%% Task 8: Model Implementation - Random Forest Regressor
import pandas as pd
import numpy as np
import tracemalloc  # For memory usage tracking
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
import joblib

#%% Start tracking memory usage
tracemalloc.start()

#%% Step 1: Load the dataset
df = pd.read_csv('yield_df_encoded.csv')  # Use the dataset before PCA

#%% Verify the dataset
print("First few rows of the dataset:")
print(df.head())

#%% Step 2: Separate features and target variable
X = df.drop('yield_t_per_ha', axis=1)
y = df['yield_t_per_ha']

#%% Step 3: Split into training and testing sets
random_seed = 42  # For reproducibility

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=random_seed
)

#%% Step 4: Define the number of PCA components
variance_threshold = 0.95  # Retain 95% of the variance

# Fit PCA on the training data to determine the number of components
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

pca = PCA()
pca.fit(X_train_scaled)
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
num_components = np.argmax(cumulative_variance >= variance_threshold) + 1
print(f"Number of PCA components to retain {variance_threshold*100}% variance: {num_components}")

#%% Step 5: Create Pipeline for Random Forest Regressor

# Random Forest Regressor pipeline
pipeline_rf = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=num_components)),
    ('regressor', RandomForestRegressor(random_state=random_seed))
])

# Define hyperparameters to tune
rf_params = {
    'regressor__n_estimators': [100, 200],
    'regressor__max_depth': [None, 10, 20],
    'regressor__min_samples_split': [2, 5]
}

# Set up GridSearchCV
grid_search_rf = GridSearchCV(
    estimator=pipeline_rf,
    param_grid=rf_params,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

# Fit the model to find the best hyperparameters
grid_search_rf.fit(X_train, y_train)

print("\nBest parameters for Random Forest Regressor:")
print(grid_search_rf.best_params_)

# Extract the best estimator
best_rf_reg = grid_search_rf.best_estimator_

# Evaluate the best estimator using cross-validation
rf_cv_scores = cross_val_score(
    best_rf_reg, X_train, y_train, cv=5, scoring='neg_mean_squared_error', n_jobs=-1
)

print("\nRandom Forest Regressor Cross-Validation MSE Scores:", -rf_cv_scores)
print("Mean MSE for Random Forest Regressor:", -rf_cv_scores.mean())

#%% Step 6: Save the best model
joblib.dump(best_rf_reg, 'trained_random_forest_regressor_model.pkl')
print("\nRandom Forest Regressor model saved as 'trained_random_forest_regressor_model.pkl'.")

#%% Stop tracking memory usage and display memory usage
current, peak = tracemalloc.get_traced_memory()
current_MB = current / 10**6
peak_MB = peak / 10**6
print(f"\nCurrent memory usage: {current_MB:.2f} MB")
print(f"Peak memory usage: {peak_MB:.2f} MB")
tracemalloc.stop()

#%% Save memory usage to a file
with open('memory_usage_Task_8_RF_updated.txt', 'w') as f:
    f.write(f"Task 8_RF - Random Forest Model Implementation Memory Usage:\n")
    f.write(f"Current memory usage: {current_MB:.2f} MB\n")
    f.write(f"Peak memory usage: {peak_MB:.2f} MB\n")

print("\nMemory usage data saved to 'memory_usage_Task_8_RF_updated.txt'.")
