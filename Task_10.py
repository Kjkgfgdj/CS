#%% Task 10: Feature Importance Analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Set plotting styles
sns.set_style('whitegrid')

#%% Step 1: Load the dataset and trained models
# Load the dataset before PCA
df = pd.read_csv('yield_df_encoded.csv')

# Load the trained models (pipelines)
lin_reg_pipeline = joblib.load('trained_linear_regression_model.pkl')
dec_tree_reg_pipeline = joblib.load('trained_decision_tree_regressor_model.pkl')

#%% Step 2: Extract original feature names
X = df.drop('yield_t_per_ha', axis=1)
original_feature_names = X.columns.tolist()

#%% Step 3: Access the regressor and PCA from the pipelines
lin_reg = lin_reg_pipeline.named_steps['regressor']
dec_tree_reg = dec_tree_reg_pipeline.named_steps['regressor']
pca = lin_reg_pipeline.named_steps['pca']

#%% Step 4: Transform coefficients back to original features for Linear Regression
# Get coefficients from the linear regression model
lin_reg_coefficients = lin_reg.coef_

# Transform PCA components to original feature space
pca_components = pca.components_

# Approximate original coefficients
approx_original_coefficients = np.dot(lin_reg_coefficients, pca_components)

# Create a DataFrame for the coefficients
lin_reg_coef_df = pd.DataFrame({
    'Feature': original_feature_names,
    'Coefficient': approx_original_coefficients
})

# Sort the coefficients by absolute value
lin_reg_coef_df['Absolute_Coefficient'] = lin_reg_coef_df['Coefficient'].abs()
lin_reg_coef_df_sorted = lin_reg_coef_df.sort_values(by='Absolute_Coefficient', ascending=False)

# Display the top features
print("\nTop features based on Linear Regression coefficients:")
print(lin_reg_coef_df_sorted.head(10))

#%% Step 5: Visualize the coefficients
plt.figure(figsize=(12, 6))
sns.barplot(
    x='Feature',
    y='Coefficient',
    data=lin_reg_coef_df_sorted.head(10),
    palette='viridis'
)
plt.title('Top 10 Feature Coefficients from Linear Regression')
plt.xlabel('Original Features')
plt.ylabel('Coefficient')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('lin_reg_top_features.png')
plt.show()
print("Linear Regression top features plot saved as 'lin_reg_top_features.png'.")

#%% Step 6: Transform feature importances back to original features for Decision Tree Regressor
# Get feature importances from the decision tree regressor
dec_tree_importances_pca = dec_tree_reg.feature_importances_

# Approximate original importances
approx_original_importances = np.dot(dec_tree_importances_pca, pca_components)

# Create a DataFrame for the feature importances
dec_tree_importances_df = pd.DataFrame({
    'Feature': original_feature_names,
    'Importance': approx_original_importances
})

# Take the absolute value of importances
dec_tree_importances_df['Absolute_Importance'] = dec_tree_importances_df['Importance'].abs()

# Sort the importances
dec_tree_importances_df_sorted = dec_tree_importances_df.sort_values(by='Absolute_Importance', ascending=False)

# Display the top features
print("\nTop features based on Decision Tree feature importances:")
print(dec_tree_importances_df_sorted.head(10))

#%% Step 7: Visualize the feature importances
plt.figure(figsize=(12, 6))
sns.barplot(
    x='Feature',
    y='Importance',
    data=dec_tree_importances_df_sorted.head(10),
    palette='viridis'
)
plt.title('Top 10 Feature Importances from Decision Tree Regressor')
plt.xlabel('Original Features')
plt.ylabel('Importance')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('dec_tree_top_features.png')
plt.show()
print("Decision Tree Regressor top features plot saved as 'dec_tree_top_features.png'.")

#%% Step 8: Document findings for the report
# Save the coefficients and importances to CSV files
lin_reg_coef_df_sorted.to_csv('linear_regression_coefficients.csv', index=False)
dec_tree_importances_df_sorted.to_csv('decision_tree_feature_importances.csv', index=False)

print("\nFeature importance data has been saved to 'linear_regression_coefficients.csv' and 'decision_tree_feature_importances.csv'.")
