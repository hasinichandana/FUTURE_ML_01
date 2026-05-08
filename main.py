import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load dataset
df = pd.read_csv("data/SampleSuperStore.csv", encoding='latin1')

# Convert date column
df['Order Date'] = pd.to_datetime(df['Order Date'])

# Remove missing values
df = df.dropna()

# Create time features
df['Year'] = df['Order Date'].dt.year
df['Month'] = df['Order Date'].dt.month
df['Day'] = df['Order Date'].dt.day
df['Weekday'] = df['Order Date'].dt.weekday

# Aggregate sales by date
sales_data = df.groupby('Order Date')['Sales'].sum().reset_index()

# Create trend feature
sales_data['Days'] = np.arange(len(sales_data))

# Features and target
X = sales_data[['Days']]
y = sales_data['Sales']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# Train model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("MAE:", mae)
print("RMSE:", rmse)

# Plot actual vs predicted
plt.figure(figsize=(12,6))

plt.plot(y_test.values, label='Actual Sales')
plt.plot(predictions, label='Predicted Sales')

plt.legend()
plt.title("Actual vs Predicted Sales")

plt.savefig("images/forecast.png")

plt.show()

# Future forecasting
future_days = pd.DataFrame({
    'Days': np.arange(len(sales_data), len(sales_data)+30)
})

future_forecast = model.predict(future_days)

# Future graph
plt.figure(figsize=(12,6))

plt.plot(future_forecast)

plt.title("30-Day Future Sales Forecast")
plt.xlabel("Future Days")
plt.ylabel("Predicted Sales")

plt.savefig("images/future_forecast.png")

plt.show()

# Monthly sales trend
monthly_sales = df.groupby('Month')['Sales'].sum()

monthly_sales.plot(kind='bar')

plt.title("Monthly Sales Trend")

plt.savefig("images/graph_name.png")
plt.show()