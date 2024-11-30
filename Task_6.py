#%% Task 6: Exploratory Data Analysis (EDA)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set plotting styles
sns.set_style('whitegrid')

#%% Step 1: Load the dataset
df = pd.read_csv('yield_df_with_new_features.csv')

# Display the first few rows
print("First few rows of the dataset:")
print(df.head())

#%% Step 2: Compute summary statistics
summary_stats = df.describe().T  # Transpose for better readability
print("\nSummary Statistics:")
print(summary_stats)

# Optionally, save the summary statistics to a CSV file
summary_stats.to_csv('summary_statistics.csv')

#%% Step 3a: Generate histogram for yield distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['yield_t_per_ha'], bins=30, kde=True, color='green')
plt.title('Distribution of Crop Yield (t/ha)')
plt.xlabel('Yield (t/ha)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('yield_distribution_histogram.png')
plt.show()

#%% Step 3b-i: Scatter plot of Yield vs. Average Temperature
plt.figure(figsize=(10, 6))
sns.scatterplot(x='avg_temp', y='yield_t_per_ha', data=df, alpha=0.6)
plt.title('Crop Yield vs. Average Temperature')
plt.xlabel('Average Temperature (°C)')
plt.ylabel('Yield (t/ha)')
plt.tight_layout()
plt.savefig('yield_vs_avg_temp.png')
plt.show()

#%% Step 3b-ii: Scatter plot of Yield vs. Pesticide Usage
plt.figure(figsize=(10, 6))
sns.scatterplot(x='pesticides_tonnes', y='yield_t_per_ha', data=df, alpha=0.6)
plt.title('Crop Yield vs. Pesticide Usage')
plt.xlabel('Pesticide Usage (tonnes)')
plt.ylabel('Yield (t/ha)')
plt.tight_layout()
plt.savefig('yield_vs_pesticides.png')
plt.show()

#%% Step 3b-iii: Scatter plot of Yield vs. Average Rainfall
plt.figure(figsize=(10, 6))
sns.scatterplot(x='average_rain_fall_mm_per_year', y='yield_t_per_ha', data=df, alpha=0.6)
plt.title('Crop Yield vs. Average Rainfall')
plt.xlabel('Average Rainfall (mm/year)')
plt.ylabel('Yield (t/ha)')
plt.tight_layout()
plt.savefig('yield_vs_rainfall.png')
plt.show()

#%% Step 3b-iv: Scatter plot of Yield vs. Temperature Deviation
plt.figure(figsize=(10, 6))
sns.scatterplot(x='temp_deviation', y='yield_t_per_ha', data=df, alpha=0.6)
plt.title('Crop Yield vs. Temperature Deviation')
plt.xlabel('Temperature Deviation (°C)')
plt.ylabel('Yield (t/ha)')
plt.tight_layout()
plt.savefig('yield_vs_temp_deviation.png')
plt.show()

#%% Step 3c-i: Line plot of Average Yield Over Time
plt.figure(figsize=(12, 6))
df_grouped_year = df.groupby('Year')['yield_t_per_ha'].mean().reset_index()
sns.lineplot(x='Year', y='yield_t_per_ha', data=df_grouped_year, marker='o')
plt.title('Average Crop Yield Over Time')
plt.xlabel('Year')
plt.ylabel('Average Yield (t/ha)')
plt.tight_layout()
plt.savefig('average_yield_over_time.png')
plt.show()

#%% Step 3c-ii: Line plot of Average Temperature Over Time
plt.figure(figsize=(12, 6))
df_grouped_year_temp = df.groupby('Year')['avg_temp'].mean().reset_index()
sns.lineplot(x='Year', y='avg_temp', data=df_grouped_year_temp, marker='o', color='red')
plt.title('Average Temperature Over Time')
plt.xlabel('Year')
plt.ylabel('Average Temperature (°C)')
plt.tight_layout()
plt.savefig('average_temperature_over_time.png')
plt.show()

#%% Step 3c-iii: Line plot of Total Pesticide Usage Over Time
plt.figure(figsize=(12, 6))
df_grouped_year_pest = df.groupby('Year')['pesticides_tonnes'].sum().reset_index()
sns.lineplot(x='Year', y='pesticides_tonnes', data=df_grouped_year_pest, marker='o', color='purple')
plt.title('Total Pesticide Usage Over Time')
plt.xlabel('Year')
plt.ylabel('Pesticide Usage (tonnes)')
plt.tight_layout()
plt.savefig('pesticide_usage_over_time.png')
plt.show()

#%% Step 3d: Correlation Heatmap
plt.figure(figsize=(12, 10))
corr_matrix = df.select_dtypes(include=[np.number]).corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.show()
# %%
