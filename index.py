import pandas as pd

# Step 1: Load the CSV file
file_path = 'sale_data.csv'
df = pd.read_csv(file_path)

# Step 2: Inspect the data (display first few rows)
print(df.head())

# Step 3: Check for missing values in the DataFrame
missing_data = df.isnull().sum()
print(missing_data)

# Step 4: Handle missing values
# Fill missing Quantity with 1
df['Quantity'].fillna(1, inplace=True)

# Fill missing Discount with 0
df['Discount'].fillna(0, inplace=True)

# Verify that there are no more missing values
print(df.isnull().sum())

# Step 5: Validate and Correct the TotalAmount Column
# Calculate the expected TotalAmount
df['CalculatedTotal'] = df['SaleAmount'] * df['Quantity'] - df['Discount']

# Identify rows where the TotalAmount does not match the calculated amount
discrepancies = df[df['TotalAmount'] != df['CalculatedTotal']]
print(discrepancies)

# Update the TotalAmount column with the correct values
df['TotalAmount'] = df['CalculatedTotal']

# Drop the CalculatedTotal column as it's no longer needed
df.drop(columns=['CalculatedTotal'], inplace=True)

# Verify the correction by checking the first few rows
print(df.head())

# Step 6: Save the cleaned DataFrame to a new CSV file
output_file_path = 'cleaned_sale_data.csv'
df.to_csv(output_file_path, index=False)

print("Cleaned data saved to:", output_file_path)
