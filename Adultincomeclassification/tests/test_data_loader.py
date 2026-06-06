from data_loader import load_data, clean_missing_values

df = load_data()

print("Original data shape:")
print(df.shape)

print("\nMissing values before cleaning:")
print(df.isnull().sum())

df = clean_missing_values(df)

print("\nMissing values after replacing '?' with NaN:")
print(df.isnull().sum())

print("\nFirst 5 rows:")
print(df.head())