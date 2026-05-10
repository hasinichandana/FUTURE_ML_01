import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Page Title
st.set_page_config(page_title="Sales Forecasting", layout="wide")

st.title("📈 Sales Forecasting Using Machine Learning")

st.write("""
This application predicts future retail sales trends using Machine Learning.
The forecasting system helps businesses improve inventory planning,
demand estimation, and business decision-making.
""")

# Load Dataset
df = pd.read_csv("data/sales.csv", encoding='latin1')

# Convert date column
df['Order Date'] = pd.to_datetime(df['Order Date'])

# Remove missing values
df = df.dropna()

# Feature Engineering
df['Year'] = df['Order Date'].dt.year
df['Month'] = df['Order Date'].dt.month
df['Day'] = df['Order Date'].dt.day
df['Weekday'] = df['Order Date'].dt.weekday

# Aggregate sales by date
sales_data = df.groupby('Order Date')['Sales'].sum().reset_index()

# Trend feature
sales_data['Days'] = np.arange(len(sales_data))

# Features and target
X = sales_data[['Days']]
y = sales_data['Sales']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# Train Model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluation Metrics
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

# Display Metrics
st.subheader("📊 Model Evaluation")

col1, col2 = st.columns(2)

with col1:
    st.metric("MAE", f"{mae:.2f}")

with col2:
    st.metric("RMSE", f"{rmse:.2f}")

# Actual vs Predicted Graph
st.subheader("📉 Actual vs Predicted Sales")

fig1, ax1 = plt.subplots(figsize=(12, 6))

ax1.plot(y_test.values, label='Actual Sales')
ax1.plot(predictions, label='Predicted Sales')

ax1.set_xlabel("Time")
ax1.set_ylabel("Sales")
ax1.legend()

st.pyplot(fig1)

# Future Forecasting
future_days = pd.DataFrame({
    'Days': np.arange(len(sales_data), len(sales_data) + 30)
})

future_forecast = model.predict(future_days)

# Future Forecast Graph
st.subheader("📈 30-Day Future Sales Forecast")

fig2, ax2 = plt.subplots(figsize=(12, 6))

ax2.plot(future_forecast)

ax2.set_xlabel("Future Days")
ax2.set_ylabel("Predicted Sales")

st.pyplot(fig2)

# Business Insights
st.subheader("💡 Business Insights")

st.write("""
- The forecasting model predicts future retail sales trends.
- Businesses can use these forecasts for inventory planning.
- Helps reduce overstocking and stock shortages.
- Supports staffing and operational planning.
- Improves business decision-making using data-driven insights.
""")

st.success("✅ Sales Forecasting Completed Successfully!")