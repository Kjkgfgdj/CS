#%% Task 5: Dimensionality Reduction
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

#%% Step 1: Load the encoded dataset from Task 4.2
yield_df = pd.read_csv('yield_df_encoded.csv')

# Verify the DataFrame
print("Columns in yield_df:")
print(yield_df.columns)

#%% Step 2: Separate features and target variable
X = yield_df.drop(['yield_t_per_ha'], axis=1)
y = yield_df['yield_t_per_ha']

#%% Step 3: Standardize the features
# It's important to scale the data before applying PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#%% Step 4: Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

#%% Step 5: Analyze variance explained by principal components
explained_variance = pca.explained_variance_ratio_

# Create a DataFrame for visualization
pc_values = np.arange(pca.n_components_) + 1
explained_variance_df = pd.DataFrame({
    'Principal Component': pc_values,
    'Explained Variance Ratio': explained_variance
})

# Plot the explained variance
plt.figure(figsize=(12, 6))
sns.barplot(x='Principal Component', y='Explained Variance Ratio', data=explained_variance_df, palette='viridis')
plt.title('Explained Variance Ratio by Principal Components')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

#%% Step 6: Plot cumulative explained variance
cumulative_variance = np.cumsum(explained_variance)

plt.figure(figsize=(12, 6))
plt.plot(pc_values, cumulative_variance, marker='o', linestyle='--', color='b')
plt.title('Cumulative Explained Variance by Principal Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.xticks(pc_values)
plt.grid(True)
plt.tight_layout()
plt.show()

#%% Step 7: Select number of principal components to keep
variance_threshold = 0.95  # Retain 95% of the variance
num_components = np.argmax(cumulative_variance >= variance_threshold) + 1
print(f"Number of principal components to retain {variance_threshold*100}% variance: {num_components}")

#%% Step 8: Transform data using selected principal components
pca_selected = PCA(n_components=num_components)
X_pca_selected = pca_selected.fit_transform(X_scaled)

#%% Step 9: Create a DataFrame with principal components
pc_columns = [f'PC{i+1}' for i in range(num_components)]
X_pca_df = pd.DataFrame(data=X_pca_selected, columns=pc_columns)

# Add the target variable back to the DataFrame
final_df = pd.concat([X_pca_df, y.reset_index(drop=True)], axis=1)

# Display the first few rows
print("First few rows of the final DataFrame:")
print(final_df.head())

#%% Step 10: Save the final dataset with selected principal components
final_df.to_csv('yield_df_pca.csv', index=False)

#%% Step 11: Document results for your report
# Save the explained variance data for inclusion in your report
explained_variance_df.to_csv('explained_variance_ratio.csv', index=False)
cumulative_variance_df = pd.DataFrame({
    'Principal Component': pc_values,
    'Cumulative Explained Variance': cumulative_variance
})
cumulative_variance_df.to_csv('cumulative_explained_variance.csv', index=False)

#%% Step 12: Optionally, visualize the loadings (contribution of original features to PCs)
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

# Create a DataFrame for loadings
loadings_df = pd.DataFrame(loadings, index=X.columns, columns=[f'PC{i+1}' for i in range(len(X.columns))])

# Save the loadings
loadings_df.to_csv('pca_loadings.csv')

# Optionally, display the loadings for the first few principal components
print("Loadings for the first few principal components:")
print(loadings_df.iloc[:, :5])

# %%
