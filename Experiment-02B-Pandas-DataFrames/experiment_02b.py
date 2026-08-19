import pandas as pd

# Load dataset into a DataFrame
# Expected input file: data.csv

df = pd.read_csv('data.csv')

print("First 5 rows:\n", df.head())
print("Last 5 rows:\n", df.tail())

df.info()
print("Summary statistics:\n", df.describe())

# Handle missing values only in numeric columns
numeric_columns = df.select_dtypes(include='number').columns
for column in numeric_columns:
    df[column] = df[column].fillna(df[column].mean())

# These names follow the structure used in the manual.
if 'existing_column' in df.columns:
    df['new_column'] = df['existing_column'] * 2
    series = df['existing_column']
    print("Series addition:", series + 10)

if {'existing_column', 'another_column'}.issubset(df.columns):
    filtered_df = df[(df['existing_column'] > 50) & (df['another_column'] < 100)]
    print("Filtered DataFrame:\n", filtered_df)

if {'category_column', 'numeric_column'}.issubset(df.columns):
    grouped = df.groupby('category_column')['numeric_column'].mean()
    print("Grouped mean:\n", grouped)
    df_sorted = df.sort_values(by='numeric_column', ascending=False)
    print("Sorted DataFrame:\n", df_sorted)
    masked_df = df[df['numeric_column'] > df['numeric_column'].median()]
    print("Masked DataFrame:\n", masked_df)

# Remove duplicates and missing rows
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# Create subset when the named columns exist
if {'column1', 'column2'}.issubset(df.columns):
    subset_df = df[['column1', 'column2']]
    subset_df.to_csv('filtered_data.csv', index=False)

if 'numeric_column' in df.columns:
    print("Total sum:", df['numeric_column'].sum())
    print("Mean:", df['numeric_column'].mean())
    print("Standard Deviation:", df['numeric_column'].std())
