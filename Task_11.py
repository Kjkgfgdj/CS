# Task_11.py

#%% Task 11: Statistical Analysis
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

#%% Step 2: Load the dataset
df = pd.read_csv('yield_df_with_new_features.csv')

# Display the first few rows
print("First few rows of the dataset:")
print(df.head())

#%% Step 3.1: Compute correlation matrix
correlation_matrix = df[['yield_t_per_ha', 'avg_temp', 'temp_deviation', 'pesticides_tonnes', 'average_rain_fall_mm_per_year']].corr()

print("\nCorrelation Matrix:")
print(correlation_matrix)

#%% Step 3.2: Visualize the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig('statistical_analysis_correlation_matrix.png')
plt.show()

#%% Step 4.1: Linear Regression Analysis
import statsmodels.api as sm

# Define the independent variables (add a constant term for the intercept)
X = df[['avg_temp', 'temp_deviation', 'pesticides_tonnes', 'average_rain_fall_mm_per_year']]
X = sm.add_constant(X)

# Define the dependent variable
y = df['yield_t_per_ha']

# Fit the linear regression model
model = sm.OLS(y, X).fit()

# Display the summary of the regression
print("\nLinear Regression Model Summary:")
print(model.summary())

#%% Step 5.1: Pearson correlation between yield and temperature deviation
corr_coef, p_value = stats.pearsonr(df['yield_t_per_ha'], df['temp_deviation'])

print(f"\nPearson Correlation between Yield and Temperature Deviation:")
print(f"Correlation Coefficient: {corr_coef:.4f}")
print(f"P-value: {p_value:.4e}")


#%% Step 5.2: Pearson correlation between yield and pesticide usage
corr_coef_pest, p_value_pest = stats.pearsonr(df['yield_t_per_ha'], df['pesticides_tonnes'])

print(f"\nPearson Correlation between Yield and Pesticide Usage:")
print(f"Correlation Coefficient: {corr_coef_pest:.4f}")
print(f"P-value: {p_value_pest:.4e}")


#%% Step 5.3.1: Scatter plot of Yield vs. Temperature Deviation
plt.figure(figsize=(10, 6))
sns.scatterplot(x='temp_deviation', y='yield_t_per_ha', data=df, alpha=0.6)
sns.regplot(x='temp_deviation', y='yield_t_per_ha', data=df, scatter=False, color='red')
plt.title('Crop Yield vs. Temperature Deviation')
plt.xlabel('Temperature Deviation (°C)')
plt.ylabel('Yield (t/ha)')
plt.tight_layout()
plt.savefig('yield_vs_temp_deviation_statistical_analysis.png')
plt.show()


#%% Step 5.3.2: Scatter plot of Yield vs. Pesticide Usage
plt.figure(figsize=(10, 6))
sns.scatterplot(x='pesticides_tonnes', y='yield_t_per_ha', data=df, alpha=0.6)
sns.regplot(x='pesticides_tonnes', y='yield_t_per_ha', data=df, scatter=False, color='red')
plt.title('Crop Yield vs. Pesticide Usage')
plt.xlabel('Pesticide Usage (tonnes)')
plt.ylabel('Yield (t/ha)')
plt.tight_layout()
plt.savefig('yield_vs_pesticides_statistical_analysis.png')
plt.show()


#%% Step 6: Save regression summary to a text file
with open('linear_regression_model_summary.txt', 'w') as f:
    f.write(model.summary().as_text())

print("\nLinear regression model summary has been saved to 'linear_regression_model_summary.txt'.")


# %%
