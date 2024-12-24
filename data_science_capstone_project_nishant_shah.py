import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import KMeans
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
    st.error("Dataset not found! Please ensure 'CAR DETAILS.csv' is in the same directory as the script.")
    st.stop()

# Display dataset preview
if st.checkbox("Show Dataset Preview"):
    st.write(df.head())

# Data preprocessing
st.subheader("Data Preprocessing")
try:
    df["age"] = 2024 - df["year"]
    df.drop(columns=["name", "year"], inplace=True)
    categorical_cols = ["fuel", "seller_type", "transmission", "owner"]
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
except KeyError as e:
    st.error(f"KeyError: {e}. Ensure your dataset contains the expected columns.")
    st.stop()

# Scaling the data
X = df.drop(columns=["selling_price"])
y = df["selling_price"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

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

results = [
    evaluate_model(lr, X_test, y_test, "Linear Regression"),
    evaluate_model(rf, X_test, y_test, "Random Forest"),
    evaluate_model(gb, X_test, y_test, "Gradient Boosting"),
]

st.write(pd.DataFrame(results))

# Save the best model (compressed)
best_model = rf  # Assuming Random Forest performs the best
joblib.dump(best_model, "best_model_compressed.pkl", compress=3)
st.success("Best model saved as 'best_model_compressed.pkl'.")

# Make predictions on user input
st.subheader("Make Predictions")
uploaded_file = st.file_uploader("Upload a CSV file with new data (same format as the dataset):", type="csv")

if uploaded_file is not None:
    sample_data = pd.read_csv(uploaded_file)
    try:
        sample_data["age"] = 2024 - sample_data["year"]
        sample_data.drop(columns=["name", "year"], inplace=True)
        sample_data = pd.get_dummies(sample_data, columns=categorical_cols, drop_first=True)
        sample_data = sample_data.reindex(columns=X.columns, fill_value=0)
        sample_data_scaled = scaler.transform(sample_data)

        predictions = best_model.predict(sample_data_scaled)
        st.write("Predictions:")
        st.write(pd.DataFrame({"Predicted Selling Price": predictions}))
    except KeyError as e:
        st.error(f"KeyError: {e}. Ensure the uploaded data matches the required format.")

# Visualizations
st.subheader("Visualizations")
eda_df = df.copy()

if st.checkbox("Show Visualizations"):
    # Price Distribution Across Different Car Ages
    st.write("Price Distribution Across Different Car Ages")
    fig, ax = plt.subplots(figsize=(15, 8))
    sns.boxplot(x="age", y="selling_price", data=eda_df, ax=ax)
    ax.set_title("Price Distribution Across Different Car Ages")
    ax.set_xlabel("Age of the Car")
    ax.set_ylabel("Selling Price")
    st.pyplot(fig)

    # Boxplot for Seller Type vs Selling Price
    st.write("Seller Type vs Selling Price")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(x="seller_type", y="selling_price", data=eda_df, ax=ax)
    ax.set_title("Seller Type vs Selling Price")
    ax.set_xlabel("Seller Type")
    ax.set_ylabel("Selling Price")
    st.pyplot(fig)

    # Scatter Plot of km_driven vs selling_price, separated by Fuel Type
    st.write("Kilometers Driven vs Selling Price (By Fuel Type)")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(x="km_driven", y="selling_price", hue="fuel", data=eda_df, ax=ax)
    ax.set_title("Kilometers Driven vs Selling Price for Different Fuel Types")
    ax.set_xlabel("Kilometers Driven")
    ax.set_ylabel("Selling Price")
    st.pyplot(fig)

    # Clustering: Car Segments
    st.write("Car Segments Based on Price, Kilometers Driven, and Age")
    cluster_data = eda_df[["selling_price", "km_driven", "age"]]

    scaler = StandardScaler()
    cluster_data_scaled = scaler.fit_transform(cluster_data)

    kmeans = KMeans(n_clusters=3, random_state=100)
    eda_df["segment"] = kmeans.fit_predict(cluster_data_scaled)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        x="km_driven", y="selling_price", hue="segment", data=eda_df, palette="Set1", ax=ax
    )
    ax.set_title("Car Segments Based on Price, Kilometers Driven, and Age")
    ax.set_xlabel("Kilometers Driven")
    ax.set_ylabel("Selling Price")
    st.pyplot(fig)
