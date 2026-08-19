import pandas as pd

# Expected local files: Google_data (2b.c1).csv and data (2c2).xlsx
# The web dataset is read directly from its public CSV URL.

text_df = pd.read_csv('Google_data (2b.c1).csv')
excel_df = pd.read_excel('data (2c2).xlsx', sheet_name='Sheet1')
web_df = pd.read_csv('https://raw.githubusercontent.com/cs109/2014_data/master/countries.csv')

print(text_df.head(), "\n", excel_df.head(), "\n", web_df.head())

# Current Pandas-friendly equivalents of the manual's fillna(method=...) calls.
text_df = text_df.ffill()
excel_df = excel_df.bfill()
web_df = web_df.dropna()

text_df.to_csv('processed_text.csv', index=False)
excel_df.to_excel('processed_excel.xlsx', index=False)

print("Processed text data saved to processed_text.csv")
print("Processed Excel data saved to processed_excel.xlsx")
