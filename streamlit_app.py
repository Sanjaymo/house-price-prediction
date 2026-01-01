import streamlit as st
import pandas as pd
import numpy as np

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="California Housing Price Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏡 California Housing Price Predictor")
st.caption(
    "This application predicts the **Median House Value in USD** using a "
    "Multiple Linear Regression model trained on the California Housing dataset."
)


# -----------------------------
# Load & Prepare Dataset
# -----------------------------
@st.cache_data
def load_data():
    housing = fetch_california_housing(as_frame=True)

    X = housing.data.copy()
    y = housing.target

    # ❌ Remove coordinates
    X = X.drop(columns=["Latitude", "Longitude"])

    # ✅ Rename columns for clarity
    X.rename(columns={
        "MedInc": "Median Income (10k USD)",
        "HouseAge": "House Age (Years)",
        "AveRooms": "Average Rooms",
        "AveBedrms": "Average Bedrooms",
        "Population": "Population in Area",
        "AveOccup": "Average Occupancy"
    }, inplace=True)

    return X, y


# -----------------------------
# Train Model
# -----------------------------
@st.cache_resource
def train_model(test_size, random_state):
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_test_pred = model.predict(X_test)

    metrics = {
        "Mean Squared Error (MSE)": mean_squared_error(y_test, y_test_pred),
        "Mean Absolute Error (MAE)": mean_absolute_error(y_test, y_test_pred),
        "R² Score": r2_score(y_test, y_test_pred)
    }

    return model, X, metrics


# -----------------------------
# Sidebar Settings
# -----------------------------
st.sidebar.header("⚙️ Model Settings")

test_size = st.sidebar.slider(
    "Test Data Size",
    min_value=0.1,
    max_value=0.4,
    value=0.2,
    step=0.05
)

random_state = st.sidebar.number_input(
    "Random State",
    min_value=0,
    max_value=9999,
    value=42,
    step=1
)

model, X, metrics = train_model(test_size, random_state)


# -----------------------------
# Prediction Section
# -----------------------------
st.subheader("🔮 Enter House Details")

inputs = {}

for col in X.columns:
    min_val = int(X[col].min())
    max_val = int(X[col].max())
    default_val = int(X[col].median())

    inputs[col] = st.slider(
        col,
        min_value=min_val,
        max_value=max_val,
        value=default_val,
        step=1
    )

input_df = pd.DataFrame([inputs]).astype(int)

st.markdown("### 🧾 Input Summary")
st.dataframe(input_df, use_container_width=True)


# -----------------------------
# Prediction Button
# -----------------------------
if st.button("🚀 Predict House Price"):
    prediction = model.predict(input_df)[0]

    # ✅ Prevent negative predictions
    prediction = max(prediction, 0)

    st.success("Prediction completed successfully ✅")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="Predicted Median House Value (USD)",
            value=f"${prediction * 100000:,.2f}"
        )

    with col2:
        st.metric(
            label="Model Output (Dataset Scale)",
            value=f"{prediction:.3f}"
        )


# -----------------------------
# Model Performance
# -----------------------------
st.markdown("---")
st.subheader("📈 Model Performance (Test Dataset)")

for metric, value in metrics.items():
    st.write(f"**{metric}:** `{value:.4f}`")

st.info(
    "📌 **Note:** The model predicts the **median house value** based on income, "
    "house age, room statistics, occupancy, and population characteristics. "
    "Coordinates were removed to improve interpretability."
)
