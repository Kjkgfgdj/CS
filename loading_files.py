#%% Import libraries
from IPython.display import display
import pandas as pd
from pathlib import Path
pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)
pd.set_option('expand_frame_repr', False)
#%% Load the data
urls = ['Crop Yield Prediction Dataset/archive/yield_df.csv']

data_sets = {}

for url in urls:
    try:
        filename = Path(url).stem
        data_sets[filename] = pd.read_csv(url)
        print(f"Loaded {filename} successfully.")
    except FileNotFoundError:
        print(f"Error: Could not find file {url}")
    except pd.errors.EmptyDataError:
        print(f"Error: File {url} is empty")
    except Exception as e:
        print(f"Error loading {url}: {str(e)}")

#%% Display the first few rows of each dataset
for name, df in data_sets.items():
    print(f"\nDataset: {name}")
    display(df.head())



#%% Access the yield_df dataset
yield_df = data_sets['yield_df']

#%% Remove unnecessary columns (e.g., 'Unnamed: 0')
unnecessary_columns = [col for col in yield_df.columns if 'Unnamed' in col]
yield_df = yield_df.drop(columns=unnecessary_columns)

#%% Rename columns for clarity
yield_df = yield_df.rename(columns={
    'Area': 'Country',
    'Item': 'Crop_Type',
    'Year': 'Year',
    'hg/ha_yield': 'hg_per_ha_yield'
})

#%% Convert hg/ha_yield to t/ha
yield_df['yield_t_per_ha'] = yield_df['hg_per_ha_yield'] / 10000

#%% Drop the original hg_per_ha_yield column
yield_df = yield_df.drop(columns=['hg_per_ha_yield'])



# %%  Display the first few rows to verify the conversion
display(yield_df[['Country', 'Crop_Type', 'Year', 'yield_t_per_ha']].head())

# %% Save the cleaned yield_df dataset
yield_df.to_csv('yield_df_cleaned.csv', index=False)

#%%
# Check the data types of the columns
print(yield_df.dtypes)

#%%
# Check for missing values
missing_values = yield_df.isnull().sum()
print("Missing values in yield_df:")
print(missing_values)

#%% Thursday, November 16 
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
