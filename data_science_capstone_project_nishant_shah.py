import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load the dataset
df = pd.read_csv("CAR DETAILS.csv")

# Check for null values
print("Null values:\n", df.isnull().sum())

# Drop duplicates
df = df.drop_duplicates()

# Add a new feature for car age and drop "year"
df["age"] = 2024 - df["year"]
df.drop(columns=["name", "year"], inplace=True)

# Handle categorical variables
categorical_cols = ["fuel", "seller_type", "transmission", "owner"]
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Verify dataset
print(df.head())

# Handle Null Values, One-Hot Encoding, Imputation, and Scaling
X = df.drop(columns=["selling_price"])
y = df["selling_price"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform EDA and Graphical Analysis
print(df.head())

# Price Distribution Across Different Car Ages
plt.figure(figsize=(15, 8))
sns.boxplot(x="age", y="selling_price", data=df)
plt.title("Price Distribution Across Different Car Ages")
plt.xlabel("Age of the Car")
plt.ylabel("Selling Price")
plt.xticks(rotation=45)
plt.show()

# Analysis of Seller Type and Its Impact on Selling Price
plt.figure(figsize=(8, 6))
sns.boxplot(x="seller_type_Individual", y="selling_price", data=df)
plt.title("Seller Type vs Selling Price")
plt.xlabel("Seller Type")
plt.ylabel("Selling Price")
plt.show()

# Correlation Between Kilometers Driven and Selling Price for Different Fuel Types
plt.figure(figsize=(12, 8))
sns.scatterplot(x="km_driven", y="selling_price", hue="fuel_Petrol", data=df)
plt.title("Kilometers Driven vs Selling Price for Different Fuel Types")
plt.xlabel("Kilometers Driven")
plt.ylabel("Selling Price")
plt.legend(title="Fuel Type")
plt.show()

# Predicting the Most Popular Car Segments Based on Features
# Select relevant features for clustering
cluster_data = df[["selling_price", "km_driven", "age"]]

# Normalize the data
scaler = StandardScaler()
cluster_data_scaled = scaler.fit_transform(cluster_data)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=100)
df["segment"] = kmeans.fit_predict(cluster_data_scaled)

# Visualize the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x="km_driven", y="selling_price", hue="segment", data=df, palette="Set1"
)
plt.title("Car Segments Based on Price, Kilometers Driven, and Age")
plt.xlabel("Kilometers Driven")
plt.ylabel("Selling Price")
plt.legend(title="Segment")
plt.show()

# Prepare Data for ML Modeling
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train Machine Learning Models
# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Random Forest
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Gradient Boosting
gb = GradientBoostingRegressor()
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)


# Evaluation Function
def evaluate_model(y_test, y_pred, model_name):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse**0.5
    r2 = r2_score(y_test, y_pred)
    print(f"{model_name} - MAE: {mae}, MSE: {mse}, RMSE: {rmse}, R²: {r2}")


evaluate_model(y_test, y_pred_lr, "Linear Regression")
evaluate_model(y_test, y_pred_rf, "Random Forest")
evaluate_model(y_test, y_pred_gb, "Gradient Boosting")

# Save and compress the best model (assuming Random Forest performs best)
joblib.dump(rf, "best_model_compressed.pkl", compress=3)

# Save the fitted scaler and encoder
joblib.dump(scaler, "scaler.pkl")
joblib.dump(encoder, "encoder.pkl")

# Load the saved model and test on a new dataset
# Load the original dataset
df = pd.read_csv("CAR DETAILS.csv")

# Randomly pick 20 data points
np.random.seed(42)
sample_data = df.sample(n=20)

# Separate features and target variable
X_sample = sample_data.drop(columns=["selling_price"])
y_sample = sample_data["selling_price"]

# Debugging: Check for null values in the sample data
print("Null values in sample data before encoding:")
print(X_sample.isnull().sum())

# Load the scaler, encoder, and model
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")
loaded_model = joblib.load("best_model_compressed.pkl")

# Encode sample data
X_sample_encoded = pd.DataFrame(
    encoder.transform(X_sample[categorical_cols]),
    columns=encoder.get_feature_names_out(categorical_cols),
)
print("X_sample_encoded shape:", X_sample_encoded.shape)

# Drop original categorical columns
X_sample = X_sample.drop(categorical_cols, axis=1)
print("X_sample shape after dropping categorical columns:", X_sample.shape)

# Ensure no duplicate rows before concatenation
X_sample.reset_index(drop=True, inplace=True)
X_sample_encoded.reset_index(drop=True, inplace=True)

# Concatenate the encoded columns
X_sample = pd.concat([X_sample, X_sample_encoded], axis=1)
print("X_sample shape after concatenation:", X_sample.shape)

# Ensure the columns match those of the training data
X_sample = X_sample.reindex(columns=X.columns, fill_value=0)
print("X_sample shape after reindexing:", X_sample.shape)

# Scale the sample data
X_sample_scaled = scaler.transform(X_sample)

# Predict using the loaded model
y_sample_pred = loaded_model.predict(X_sample_scaled)

# Verify the lengths match before creating DataFrame
if len(y_sample.values) == len(y_sample_pred):
    results = pd.DataFrame({"Actual": y_sample.values, "Predicted": y_sample_pred})
    print(results)
else:
    print(
        f"Length mismatch: Actuals={len(y_sample.values)}, Predicted={len(y_sample_pred)}"
    )

# Evaluate the model on the sampled dataset
if len(y_sample.values) == len(y_sample_pred):
    print("Sampled Data - R²:", r2_score(y_sample, y_sample_pred))
    print("Sampled Data - MAE:", mean_absolute_error(y_sample, y_sample_pred))
    print("Sampled Data - MSE:", mean_squared_error(y_sample, y_sample_pred))
else:
    print("Skipping evaluation due to length mismatch.")
