import pandas as pd

# Configure pandas to print all columns horizontally on console.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Load datasets (from excel files).
d1 = pd.read_excel("data\\dataset-a.xlsx", sheet_name="ECG1A03_new", header=1, nrows=119)
d2 = pd.read_excel("data\\dataset-b.xlsx", sheet_name="ECG1A04_new", header=1)

# Prepare d1:
# Take a look at the column names
print(d1.head(), '\n')

# Drop unneeded columns
d1_drop_cols = [
    'slweight(g)',
    'slweight / 2',
    'carcass_yield',
    'Total',
    'Unnamed: 24',
    'coldcarc/warmcarc ratio',
]
d1.drop(d1_drop_cols, axis=1, inplace=True)
# Check data after dropping the columns
print(d1.head(), '\n')

# Look for empty values
print(f"Number of empty values per column (d1):\n{d1.isna().sum()}")


# Prepare d2:
# Take a quick look
print(d2.head(), '\n')

# Specify needed columns and drop the rest
d2_keep_cols = [
    'sheepid',
    'mw%',
    'WtBefDIS',
    'LEG',
    'CHUMP',
    'LOIN',
    'BREAST',
    'BESTEND',
    'MIDNECK',
    'SHOULDER',
    'NECK'
]
d2 = d2[d2_keep_cols]
# Check dataset columns
print(d2.head(), '\n')

# Look for empty values
print(f"Number of empty values per column (d2):\n{d2.isna().sum()}")

# Merge dataframes on "sheepid"
merged_df = pd.merge(d1, d2, on='sheepid', how='left')
print(merged_df.head())
# Merged dataframe should have (d1+d2-1) columns (-1 because "sheepid" is common on both dfs).
print(f"d1 column sum: {len(d1.columns)}")
print(f"d2 column sum: {len(d2.columns)}")
print(f"merged dataframe column sum: {len(merged_df.columns)}")

# Save merged dataset to csv file
merged_df.to_csv("data\\dataset-merged.csv", index=False)

