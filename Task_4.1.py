#Thursday, November 16 
# Task 4.1: 	4.1 Create new features:
#	Calculate weather indices like deviations from average temperatures.
#	Compute cumulative pesticide usage over years.


#%% **Task 4. Feature Engineering**
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

#%% Load the dataset
yield_df = pd.read_csv('yield_df_cleaned.csv')

#%% Step 1: Calculate long-term average temperature per country
country_avg_temp = yield_df.groupby('Country')['avg_temp'].mean().reset_index()
country_avg_temp.rename(columns={'avg_temp': 'long_term_avg_temp'}, inplace=True)

#%% Merge with the original dataframe
yield_df = pd.merge(yield_df, country_avg_temp, on='Country', how='left')

#%% Step 2: Calculate deviation from long-term average temperature
yield_df['temp_deviation'] = yield_df['avg_temp'] - yield_df['long_term_avg_temp']

#%% Step 3: Calculate standard deviation of temperature per country
country_temp_std = yield_df.groupby('Country')['avg_temp'].std().reset_index()
country_temp_std.rename(columns={'avg_temp': 'temp_std'}, inplace=True)

#%% Merge with the dataframe
yield_df = pd.merge(yield_df, country_temp_std, on='Country', how='left')

#%% Step 4: Calculate standardized temperature deviation (Z-score)
yield_df['temp_deviation_zscore'] = yield_df['temp_deviation'] / yield_df['temp_std']

#%% Step 5: Sort the dataframe
yield_df = yield_df.sort_values(by=['Country', 'Year'])

#%% Step 6: Compute cumulative pesticide usage over years
yield_df['cumulative_pesticides'] = yield_df.groupby('Country')['pesticides_tonnes'].cumsum()

#%% Step 7: (Optional) Normalize cumulative pesticide usage
scaler = MinMaxScaler()
yield_df[['cumulative_pesticides_normalized']] = scaler.fit_transform(yield_df[['cumulative_pesticides']])

#%% Step 8: Verify the new features
print(yield_df[['Country', 'Year', 'pesticides_tonnes', 'cumulative_pesticides']].head())

#%% Step 9: Save the updated dataframe
yield_df.to_csv('yield_df_with_new_features.csv', index=False)

#%% Final assertions
assert 'temp_deviation' in yield_df.columns, "'temp_deviation' column is missing."
assert 'temp_std' in yield_df.columns, "'temp_std' column is missing."
assert 'cumulative_pesticides' in yield_df.columns, "'cumulative_pesticides' column is missing."

# %%
