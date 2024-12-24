import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

# Streamlit app title
st.title("Car Price Prediction App")


# Load the dataset
@st.cache
def load_data():
    return pd.read_csv("CAR DETAILS.csv")


try:
    df = load_data()
except FileNotFoundError:
    st.error(
        "Dataset not found! Please ensure 'CAR DETAILS.csv' is in the same directory as the script."
    )
    st.stop()

# Display dataset preview
if st.checkbox("Show Dataset Preview"):
    st.write(df.head())

# Data preprocessing
st.subheader("Data Preprocessing")
df["age"] = 2024 - df["year"]
df.drop(columns=["name", "year"], inplace=True)
categorical_cols = ["fuel", "seller_type", "transmission", "owner"]
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Scaling the data
X = df.drop(columns=["selling_price"])
y = df["selling_price"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train models
st.subheader("Training Models")
lr = LinearRegression()
rf = RandomForestRegressor(random_state=42)
gb = GradientBoostingRegressor(random_state=42)

lr.fit(X_train, y_train)
rf.fit(X_train, y_train)
gb.fit(X_train, y_train)


# Model evaluation
def evaluate_model(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse**0.5
    r2 = r2_score(y_test, y_pred)
    return {"Model": name, "MAE": mae, "MSE": mse, "RMSE": rmse, "R²": r2}


results = []
results.append(evaluate_model(lr, X_test, y_test, "Linear Regression"))
results.append(evaluate_model(rf, X_test, y_test, "Random Forest"))
results.append(evaluate_model(gb, X_test, y_test, "Gradient Boosting"))

st.write(pd.DataFrame(results))

# Save the best model (compressed)
best_model = rf  # Assuming Random Forest performs the best
joblib.dump(best_model, "best_model_compressed.pkl", compress=3)
st.success("Best model saved as 'best_model_compressed.pkl'.")

# Make predictions on user input
st.subheader("Make Predictions")
uploaded_file = st.file_uploader(
    "Upload a CSV file with new data (same format as the dataset):", type="csv"
)

if uploaded_file is not None:
    sample_data = pd.read_csv(uploaded_file)
    sample_data["age"] = 2024 - sample_data["year"]
    sample_data.drop(columns=["name", "year"], inplace=True)
    sample_data = pd.get_dummies(sample_data, columns=categorical_cols, drop_first=True)
    sample_data = sample_data.reindex(columns=X.columns, fill_value=0)
    sample_data_scaled = scaler.transform(sample_data)

    predictions = best_model.predict(sample_data_scaled)
    st.write("Predictions:")
    st.write(pd.DataFrame({"Predicted Selling Price": predictions}))

# Visualizations
st.subheader("Visualizations")
if st.checkbox("Show EDA Plots"):
    # Price Distribution Across Car Ages
    st.write("Price Distribution Across Different Car Ages")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x="age", y="selling_price", data=df, ax=ax)
    st.pyplot(fig)

    # Correlation Between Kilometers Driven and Selling Price
    st.write("Kilometers Driven vs. Selling Price")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x="km_driven", y="selling_price", data=df, ax=ax)
    st.pyplot(fig)
