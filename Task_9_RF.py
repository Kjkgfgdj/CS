# Task_9_RF.py

#%% Task 9: Model Training - Random Forest Regressor
import pandas as pd
import tracemalloc  # For memory usage tracking
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
import numpy as np

#%% Start tracking memory usage
tracemalloc.start()

#%% Step 1: Load the dataset
df = pd.read_csv('yield_df_encoded.csv')  # Use the dataset before PCA

#%% Step 2: Separate features and target variable
X = df.drop('yield_t_per_ha', axis=1)
y = df['yield_t_per_ha']

#%% Step 3: Define the number of PCA components
variance_threshold = 0.95  # Retain 95% of the variance

# Fit PCA on the entire dataset (after scaling)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA()
pca.fit(X_scaled)
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
num_components = np.argmax(cumulative_variance >= variance_threshold) + 1
print(f"Number of PCA components to retain {variance_threshold*100}% variance: {num_components}")

#%% Step 4: Load the best hyperparameters
best_rf_reg = joblib.load('trained_random_forest_regressor_model.pkl')
best_params = best_rf_reg.get_params()

#%% Step 5: Create Pipeline for Random Forest Regressor
pipeline_rf = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=num_components)),
    ('regressor', RandomForestRegressor(random_state=42))
])

# Update the pipeline with best hyperparameters
pipeline_rf.set_params(**best_params)

#%% Step 6: Train the model on the full dataset
pipeline_rf.fit(X, y)

#%% Step 7: Save the trained model
joblib.dump(pipeline_rf, 'trained_random_forest_regressor_model.pkl')

#%% Stop tracking memory usage and display memory usage
current, peak = tracemalloc.get_traced_memory()
current_MB = current / 10**6
peak_MB = peak / 10**6
print(f"\nCurrent memory usage: {current_MB:.2f} MB")
print(f"Peak memory usage: {peak_MB:.2f} MB")
tracemalloc.stop()

#%% Save memory usage to a file
with open('memory_usage_Task_9_RF_updated.txt', 'w') as f:
    f.write(f"Task 9_RF - Random Forest Model Training Memory Usage:\n")
    f.write(f"Current memory usage: {current_MB:.2f} MB\n")
    f.write(f"Peak memory usage: {peak_MB:.2f} MB\n")

print("\nMemory usage data saved to 'memory_usage_Task_9_RF_updated.txt'.")
print("\nRandom Forest Regressor model retrained on full data and saved.")
