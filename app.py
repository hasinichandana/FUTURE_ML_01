import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Page Configuration
st.set_page_config(page_title="Sales Forecasting", layout="wide")

# Title
st.title("📈 Sales Forecasting Using Machine Learning")

st.write("""
This application predicts future retail sales trends using Machine Learning.

The forecasting system helps businesses improve:
- Inventory Planning
- Demand Estimation
- Staffing Preparation
- Business Decision-Making
""")

# Load Dataset
df = pd.read_csv("data/SampleSuperStore.csv", encoding='latin1')

# Convert Date Column
df['Order Date'] = pd.to_datetime(df['Order Date'])

# Remove Missing Values
df = df.dropna()

# Feature Engineering
df['Year'] = df['Order Date'].dt.year
df['Month'] = df['Order Date'].dt.month
df['Day'] = df['Order Date'].dt.day
df['Weekday'] = df['Order Date'].dt.weekday

# Aggregate Sales by Date
sales_data = df.groupby('Order Date')['Sales'].sum().reset_index()

# Trend Feature
sales_data['Days'] = np.arange(len(sales_data))

# Features and Target
X = sales_data[['Days']]
y = sales_data['Sales']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# Train Model
model = RandomForestRegressor(random_state=42)
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

# Graph 1 - Actual vs Predicted Sales
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

# Graph 2 - Future Forecast
st.subheader("📈 30-Day Future Sales Forecast")

fig2, ax2 = plt.subplots(figsize=(12, 6))

ax2.plot(future_forecast)

ax2.set_xlabel("Future Days")
ax2.set_ylabel("Predicted Sales")

st.pyplot(fig2)

# Graph 3 - Monthly Sales Trend
st.subheader("📅 Monthly Sales Trend")

monthly_sales = df.groupby('Month')['Sales'].sum()

fig3, ax3 = plt.subplots(figsize=(10, 5))

monthly_sales.plot(kind='bar', ax=ax3)

ax3.set_xlabel("Month")
ax3.set_ylabel("Total Sales")

st.pyplot(fig3)

# Business Insights
st.subheader("💡 Business Insights")

st.write("""
- The forecasting model predicts future retail sales trends.
- Businesses can use forecasts for inventory management.
- Helps reduce overstocking and stock shortages.
- Supports staffing and operational planning.
- Improves business decision-making using data-driven insights.
""")

# Success Message
st.success("✅ Sales Forecasting Completed Successfully!")