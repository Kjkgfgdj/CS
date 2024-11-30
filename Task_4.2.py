#%% Task 4.2: Encode Categorical Variables
import pandas as pd

#%% Step 1: Load the updated dataset from Task 4.1
yield_df = pd.read_csv('yield_df_with_new_features.csv')

#%% Step 2: Identify categorical variables to encode
categorical_vars = ['Country', 'Crop_Type']

#%% Step 3: Perform One-Hot Encoding
yield_df_encoded = pd.get_dummies(yield_df, columns=categorical_vars)

#%% Step 4: Verify the encoding
print("Encoded DataFrame Columns:")
print(yield_df_encoded.columns)

print("\nFirst few rows of the encoded DataFrame:")
print(yield_df_encoded.head())

#%% Step 5: Save the encoded dataset
yield_df_encoded.to_csv('yield_df_encoded.csv', index=False)

# %%
print("Shape of original DataFrame:", yield_df.shape)
print("Shape of encoded DataFrame:", yield_df_encoded.shape)
print("Any NaN values in the encoded DataFrame?")
print(yield_df_encoded.isnull().any().any())
