import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data from parquet file
df = pd.read_parquet("dataset.parquet")

# Preprocessing
df['created'] = pd.to_datetime(df['created'])
df['updated'] = pd.to_datetime(df['updated'])

# Extract datetime features
df['created_year'] = df['created'].dt.year
df['created_month'] = df['created'].dt.month
df['created_day'] = df['created'].dt.day
df['updated_year'] = df['updated'].dt.year
df['updated_month'] = df['updated'].dt.month
df['updated_day'] = df['updated'].dt.day
df['time_diff'] = (df['updated'] - df['created']).dt.total_seconds()

# Parse lonlat into latitude and longitude
df[['longitude', 'latitude']] = df['lonlat'].str.extract(r'([-\d.]+)[°\'"]\w\s([-\d.]+)[°\'"]\w')

# Convert longitude and latitude to numeric
df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')

# Set seaborn style
sns.set(style="whitegrid")

# Create plots for encoded features
encoded_columns = [
    "created_year", "created_month", "created_day",
    "updated_year", "updated_month", "updated_day",
    "time_diff", "longitude", "latitude"
]

for col in encoded_columns:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x="radio", y=col, data=df, palette="coolwarm")
    plt.title(f"{col} by Radio Type")
    plt.xlabel("Radio Type")
    plt.ylabel(col)
    plt.show()

print("Plots generated successfully.")