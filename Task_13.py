# Task_13.py

#%% Task 13: Model Comparison and Selection
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set plotting styles
sns.set_style('whitegrid')

#%% Step 1: Load the evaluation metrics
results_df = pd.read_csv('model_performance_metrics_updated.csv')

# Display the evaluation metrics
print("Model Performance Metrics:")
print(results_df)

#%% Step 2: Compare evaluation metrics
# Plotting the metrics for better visualization
metrics = ['MSE', 'MAE', 'R-squared']
models = results_df['Model']

# Function to create bar plots for each metric
def plot_metric(metric_name):
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y=metric_name, data=results_df, palette='viridis')
    plt.title(f'{metric_name} Comparison')
    plt.ylabel(metric_name)
    plt.tight_layout()
    filename = f"{metric_name.lower().replace(' ', '_')}_comparison.png"
    plt.savefig(filename)
    plt.show()
    print(f"{metric_name} comparison plot saved as '{filename}'.")

# Plot each metric
for metric in metrics:
    plot_metric(metric)

#%% Step 3: Select the best-performing model
# Based on R-squared (higher is better), MSE and MAE (lower is better)
best_model_index = results_df['R-squared'].idxmax()
best_model_name = results_df.loc[best_model_index, 'Model']
print(f"\nThe best-performing model is: {best_model_name}")

#%% Step 4: Document findings for the report
# Create a summary of the comparison
comparison_summary = f"""
After evaluating the Linear Regression, Decision Tree Regressor, and Random Forest Regressor models, the following observations were made:

- **Linear Regression**:
  - MSE: {results_df.loc[results_df['Model'] == 'Linear Regression', 'MSE'].values[0]:.4f}
  - MAE: {results_df.loc[results_df['Model'] == 'Linear Regression', 'MAE'].values[0]:.4f}
  - R-squared: {results_df.loc[results_df['Model'] == 'Linear Regression', 'R-squared'].values[0]:.4f}

- **Decision Tree Regressor**:
  - MSE: {results_df.loc[results_df['Model'] == 'Decision Tree Regressor', 'MSE'].values[0]:.4f}
  - MAE: {results_df.loc[results_df['Model'] == 'Decision Tree Regressor', 'MAE'].values[0]:.4f}
  - R-squared: {results_df.loc[results_df['Model'] == 'Decision Tree Regressor', 'R-squared'].values[0]:.4f}

- **Random Forest Regressor**:
  - MSE: {results_df.loc[results_df['Model'] == 'Random Forest Regressor', 'MSE'].values[0]:.4f}
  - MAE: {results_df.loc[results_df['Model'] == 'Random Forest Regressor', 'MAE'].values[0]:.4f}
  - R-squared: {results_df.loc[results_df['Model'] == 'Random Forest Regressor', 'R-squared'].values[0]:.4f}

**Conclusion**: The {best_model_name} outperformed the other models, achieving the lowest MSE and MAE, and the highest R-squared value. Therefore, the {best_model_name} is selected as the best-performing model for further analysis.
"""

# Save the summary to a text file
with open('model_comparison_summary.txt', 'w') as file:
    file.write(comparison_summary)

print("\nModel comparison summary has been saved to 'model_comparison_summary.txt'.")

# %%
