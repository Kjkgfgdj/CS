# Task_14.py

#%% Task 14: Results Interpretation
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# For SHAP explanations
import shap

# Set plotting styles
sns.set_style('whitegrid')

#%% Step 1: Load the dataset and the best-performing model
# Load the dataset before PCA
df = pd.read_csv('yield_df_encoded.csv')

# Load the trained Random Forest Regressor pipeline
pipeline = joblib.load('trained_random_forest_regressor_model.pkl')

# Verify the model
print("Loaded the best-performing model: Random Forest Regressor within a pipeline")

#%% Step 2: Separate features and target variable
X = df.drop('yield_t_per_ha', axis=1)
y = df['yield_t_per_ha']

# Verify the shapes
print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)

#%% Step 3: Prepare for SHAP
# Since the pipeline includes PCA, SHAP values will be computed on the PCA components
# For interpretability, we need to compute SHAP values on the original features

# Create a wrapper model that can handle the pipeline
def model_predict(X_input):
    # Ensure X_input is a DataFrame with correct feature names
    if isinstance(X_input, np.ndarray):
        X_input = pd.DataFrame(X_input, columns=X.columns)
    return pipeline.predict(X_input)

# Use KernelExplainer since TreeExplainer may not work well with pipelines including PCA
X_background = shap.sample(X, 100, random_state=42)  # Background dataset for SHAP

explainer = shap.KernelExplainer(model_predict, X_background)

# Sample a subset of the data for SHAP computations
X_sample = shap.sample(X, 100, random_state=42)

# Compute SHAP values
shap_values = explainer.shap_values(X_sample)

#%% Step 4: Plot SHAP summary plot
plt.figure()
shap.summary_plot(shap_values, X_sample, show=False)
plt.title('SHAP Summary Plot')
plt.tight_layout()
plt.savefig('shap_summary_plot.png')
plt.show()
print("SHAP summary plot saved as 'shap_summary_plot.png'.")

#%% Step 5: Identify top features impacting the model
# Calculate mean absolute SHAP values for each feature
feature_importance = pd.DataFrame({
    'Feature': X_sample.columns,
    'SHAP Value': np.abs(shap_values).mean(axis=0)
})

# Sort features by importance
feature_importance.sort_values(by='SHAP Value', ascending=False, inplace=True)

# Display the top features
print("\nTop 10 features impacting the model:")
print(feature_importance.head(10))

# Save to CSV
feature_importance.to_csv('shap_feature_importance.csv', index=False)
print("\nFeature importance saved to 'shap_feature_importance.csv'.")

#%% Step 6: Visualize the dependence of yield on top features
top_features = feature_importance['Feature'].head(3).tolist()

for feature in top_features:
    plt.figure()
    shap.dependence_plot(feature, shap_values, X_sample, show=False)
    plt.title(f'SHAP Dependence Plot for {feature}')
    plt.tight_layout()
    filename = f'shap_dependence_plot_{feature}.png'
    plt.savefig(filename)
    plt.show()
    print(f"SHAP dependence plot for {feature} saved as '{filename}'.")

#%% Step 7: Interpret results in the context of agricultural practices

# Prepare interpretation text
interpretation = "### Interpretation of Top Features Impacting Crop Yield:\n"

for feature in top_features:
    mean_shap = feature_importance.loc[feature_importance['Feature'] == feature, 'SHAP Value'].values[0]
    interpretation += f"\n**Feature:** {feature}\n"
    interpretation += f"- **Mean SHAP Value:** {mean_shap:.4f}\n"
    # You can add more interpretation based on domain knowledge
    interpretation += "\n"

# Print the interpretation
print("\nInterpretation of Top Features:")
print(interpretation)

# Save the interpretation to a text file
with open('results_interpretation.txt', 'w') as f:
    f.write(interpretation)

print("Results interpretation has been saved to 'results_interpretation.txt'.")

#%% Step 8: Provide actionable insights for farmers and policymakers

# The actionable insights section remains the same as before
# You can refer to your previous insights or adjust based on the new findings

# Save the actionable insights to a text file
actionable_insights = """
### Actionable Insights for Farmers and Policymakers:

1. **Optimize Key Factors:**
   - Focus on the top features identified by SHAP analysis.
   - Implement strategies to enhance positive factors and mitigate negative ones.

2. **Data-Driven Decision Making:**
   - Utilize predictive models and feature importance analyses to make informed agricultural decisions.

3. **Policy Recommendations:**
   - Develop policies that support practices influencing the top features to improve crop yield.

"""

print("\nActionable Insights for Farmers and Policymakers:")
print(actionable_insights)

with open('actionable_insights.txt', 'w') as f:
    f.write(actionable_insights)

print("Actionable insights have been saved to 'actionable_insights.txt'.")
