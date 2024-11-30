# Task_12.py

#%% Task 12: Performance Evaluation
import pandas as pd
import numpy as np
import tracemalloc  # For memory usage tracking
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
import joblib

#%% Start tracking memory usage
tracemalloc.start()

#%% Step 1: Load the dataset
df = pd.read_csv('yield_df_encoded.csv')  # Use the dataset before PCA

# Verify the dataset
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

#%% Step 4: Load the trained models (pipelines)
lin_reg_pipeline = joblib.load('trained_linear_regression_model.pkl')
dec_tree_reg_pipeline = joblib.load('trained_decision_tree_regressor_model.pkl')
rf_reg_pipeline = joblib.load('trained_random_forest_regressor_model.pkl')

#%% Step 5: Predict crop yields using test data for all models

# Linear Regression predictions
y_pred_lin_reg = lin_reg_pipeline.predict(X_test)

# Decision Tree Regressor predictions
y_pred_dec_tree = dec_tree_reg_pipeline.predict(X_test)

# Random Forest Regressor predictions
y_pred_rf = rf_reg_pipeline.predict(X_test)

#%% Step 6: Calculate evaluation metrics for Linear Regression
mse_lin_reg = mean_squared_error(y_test, y_pred_lin_reg)
mae_lin_reg = mean_absolute_error(y_test, y_pred_lin_reg)
r2_lin_reg = r2_score(y_test, y_pred_lin_reg)

print("\nLinear Regression Performance on Test Data:")
print(f"MSE: {mse_lin_reg:.4f}")
print(f"MAE: {mae_lin_reg:.4f}")
print(f"R-squared: {r2_lin_reg:.4f}")

#%% Step 7: Calculate evaluation metrics for Decision Tree Regressor
mse_dec_tree = mean_squared_error(y_test, y_pred_dec_tree)
mae_dec_tree = mean_absolute_error(y_test, y_pred_dec_tree)
r2_dec_tree = r2_score(y_test, y_pred_dec_tree)

print("\nDecision Tree Regressor Performance on Test Data:")
print(f"MSE: {mse_dec_tree:.4f}")
print(f"MAE: {mae_dec_tree:.4f}")
print(f"R-squared: {r2_dec_tree:.4f}")

#%% Step 8: Calculate evaluation metrics for Random Forest Regressor
mse_rf = mean_squared_error(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print("\nRandom Forest Regressor Performance on Test Data:")
print(f"MSE: {mse_rf:.4f}")
print(f"MAE: {mae_rf:.4f}")
print(f"R-squared: {r2_rf:.4f}")

#%% Step 9: Calculate Cross-Validation Scores
cv_folds = 5

# Linear Regression cross-validation
cv_scores_lin_reg = cross_val_score(lin_reg_pipeline, X, y, cv=cv_folds, scoring='neg_mean_squared_error', n_jobs=-1)
cv_rmse_lin_reg = np.sqrt(-cv_scores_lin_reg)

# Decision Tree Regressor cross-validation
cv_scores_dec_tree = cross_val_score(dec_tree_reg_pipeline, X, y, cv=cv_folds, scoring='neg_mean_squared_error', n_jobs=-1)
cv_rmse_dec_tree = np.sqrt(-cv_scores_dec_tree)

# Random Forest Regressor cross-validation
cv_scores_rf = cross_val_score(rf_reg_pipeline, X, y, cv=cv_folds, scoring='neg_mean_squared_error', n_jobs=-1)
cv_rmse_rf = np.sqrt(-cv_scores_rf)

# Print cross-validation RMSE for each model
print("\nCross-Validation RMSE for Linear Regression:", cv_rmse_lin_reg.mean())
print("Cross-Validation RMSE for Decision Tree Regressor:", cv_rmse_dec_tree.mean())
print("Cross-Validation RMSE for Random Forest Regressor:", cv_rmse_rf.mean())

#%% Step 10: Save the evaluation metrics to a CSV file
results_df = pd.DataFrame({
    'Model': ['Linear Regression', 'Decision Tree Regressor', 'Random Forest Regressor'],
    'MSE': [mse_lin_reg, mse_dec_tree, mse_rf],
    'MAE': [mae_lin_reg, mae_dec_tree, mae_rf],
    'R-squared': [r2_lin_reg, r2_dec_tree, r2_rf],
    'Cross-Val RMSE': [cv_rmse_lin_reg.mean(), cv_rmse_dec_tree.mean(), cv_rmse_rf.mean()]
})

results_df.to_csv('model_performance_metrics_updated.csv', index=False)

print("\nEvaluation metrics have been saved to 'model_performance_metrics_updated.csv'.")

#%% Stop tracking memory usage and display memory usage
current, peak = tracemalloc.get_traced_memory()
current_MB = current / 10**6
peak_MB = peak / 10**6
print(f"\nCurrent memory usage: {current_MB:.2f} MB")
print(f"Peak memory usage: {peak_MB:.2f} MB")
tracemalloc.stop()

#%% Save memory usage to a file
with open('memory_usage_Task_12_updated.txt', 'w') as f:
    f.write(f"Task 12 - Model Evaluation Memory Usage:\n")
    f.write(f"Current memory usage: {current_MB:.2f} MB\n")
    f.write(f"Peak memory usage: {peak_MB:.2f} MB\n")

print("\nMemory usage data saved to 'memory_usage_Task_12_updated.txt'.")

#%% Step 11: Plot Actual vs Predicted
import matplotlib.pyplot as plt
import seaborn as sns

def plot_pred_vs_actual(y_true, y_pred, model_name):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('Actual Yield (t/ha)')
    plt.ylabel('Predicted Yield (t/ha)')
    plt.title(f'Actual vs Predicted Yield - {model_name}')
    plt.tight_layout()
    filename = f'actual_vs_predicted_{model_name.replace(" ", "_").lower()}.png'
    plt.savefig(filename)
    plt.show()
    print(f"Prediction vs Actual plot saved as '{filename}'.")

# Plot for Linear Regression
plot_pred_vs_actual(y_test, y_pred_lin_reg, 'Linear Regression')

# Plot for Decision Tree Regressor
plot_pred_vs_actual(y_test, y_pred_dec_tree, 'Decision Tree Regressor')

# Plot for Random Forest Regressor
plot_pred_vs_actual(y_test, y_pred_rf, 'Random Forest Regressor')
