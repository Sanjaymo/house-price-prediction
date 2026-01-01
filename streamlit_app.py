import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# -----------------------------
# Constants
# -----------------------------
USD_TO_INR = 83  # Fixed conversion rate for stability


# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Housing Price Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏡 Housing Price Predictor")
st.caption(
    "Predicting the **Median House Value** using a "
    "Multiple Linear Regression model (California Housing Dataset)."
)


# -----------------------------
# Load Dataset (LOCAL CSV)
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("california_housing.csv")

    # Target
    y = df["MedHouseVal"]

    # ❌ Remove coordinates
    X = df.drop(columns=["MedHouseVal", "Latitude", "Longitude"])

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

    y_pred = model.predict(X_test)

    metrics = {
        "MSE": mean_squared_error(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "R² Score": r2_score(y_test, y_pred)
    }

    return model, X, metrics


# -----------------------------
# Sidebar Settings
# -----------------------------
st.sidebar.header("⚙️ Model Settings")

test_size = st.sidebar.slider(
    "Test Data Size",
    0.1, 0.4, 0.2, 0.05
)

random_state = st.sidebar.number_input(
    "Random State",
    0, 9999, 42
)

model, X, metrics = train_model(test_size, random_state)


# -----------------------------
# Prediction Section
# -----------------------------
st.subheader("🔮 Enter House Details")

inputs = {}

for col in X.columns:
    inputs[col] = st.slider(
        col,
        min_value=int(X[col].min()),
        max_value=int(X[col].max()),
        value=int(X[col].median()),
        step=1   # ✅ No decimals
    )

input_df = pd.DataFrame([inputs])

st.markdown("### 🧾 Input Summary")
st.dataframe(input_df, use_container_width=True)


# -----------------------------
# Prediction Output
# -----------------------------
if st.button("🚀 Predict House Price"):
    prediction = model.predict(input_df)[0]

    # ✅ Prevent negative values
    prediction = max(prediction, 0)

    # Convert values
    price_usd = prediction * 100000
    price_inr = price_usd * USD_TO_INR

    st.success("Prediction Successful ✅")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Predicted Median House Value (USD)",
            f"${price_usd:,.2f}"
        )

    with col2:
        st.metric(
            "Predicted Median House Value (INR)",
            f"₹{price_inr:,.2f}"
        )


# -----------------------------
# Model Performance
# -----------------------------
st.markdown("---")
st.subheader("📈 Model Performance")

for k, v in metrics.items():
    st.write(f"**{k}:** `{v:.4f}`")

st.info(
    "💡 Prices in INR are calculated using a fixed conversion rate "
    f"(1 USD = ₹{USD_TO_INR}) for stability and cloud compatibility."
)
