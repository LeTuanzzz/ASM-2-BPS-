import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the CSV file
file_path = 'sale_data.csv'
data = pd.read_csv(file_path)

# Drop rows with missing target values
data_cleaned = data.dropna(subset=['TotalAmount'])

# Select features and target
features = ['SaleAmount', 'Quantity']
X = data_cleaned[features]
y = data_cleaned['TotalAmount']

# Handle missing values in features
X.fillna(X.mean(), inplace=True)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict the TotalAmount for the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Line chart: Compare actual vs predicted TotalAmount
plt.figure(figsize=(12, 6))
plt.plot(range(len(y_test)), y_test.values, marker='o', linestyle='-', color='blue', label='Actual TotalAmount')
plt.plot(range(len(y_test)), y_pred, marker='x', linestyle='--', color='red', label='Predicted TotalAmount')
plt.title('Actual vs Predicted TotalAmount')
plt.xlabel('Index')
plt.ylabel('TotalAmount')
plt.legend()
plt.grid(True)
plt.show()
