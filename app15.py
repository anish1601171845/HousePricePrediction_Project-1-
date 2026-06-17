import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 House Price Prediction")
st.write("Predict house prices using Machine Learning")

# ----------------------------
# Load Dataset
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Housing.csv")
    return df

df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df.head())

# ----------------------------
# Data Cleaning
# ----------------------------
numeric_cols = df.select_dtypes(include=np.number).columns
categorical_cols = df.select_dtypes(exclude=np.number).columns

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

df.drop_duplicates(inplace=True)

# ----------------------------
# Encoding
# ----------------------------
encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# ----------------------------
# Features & Target
# ----------------------------
X = df.drop("price", axis=1)
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ----------------------------
# Train Models
# ----------------------------
lr = LinearRegression()
lr.fit(X_train, y_train)

rf = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)
rf.fit(X_train, y_train)

# ----------------------------
# Evaluation
# ----------------------------
y_pred_lr = lr.predict(X_test)
y_pred_rf = rf.predict(X_test)

r2_lr = r2_score(y_test, y_pred_lr)
r2_rf = r2_score(y_test, y_pred_rf)

mae_lr = mean_absolute_error(y_test, y_pred_lr)
mae_rf = mean_absolute_error(y_test, y_pred_rf)

rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))

st.subheader("Model Performance")

results = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest"],
    "MAE": [round(mae_lr, 2), round(mae_rf, 2)],
    "RMSE": [round(rmse_lr, 2), round(rmse_rf, 2)],
    "R² Score": [round(r2_lr, 4), round(r2_rf, 4)]
})

st.dataframe(results)

best_model = lr if r2_lr > r2_rf else rf

# ----------------------------
# Prediction Section
# ----------------------------
st.header("Predict House Price")

user_input = {}

for col in X.columns:

    if col in encoders:

        options = encoders[col].classes_

        selected = st.selectbox(
            col,
            options
        )

        user_input[col] = encoders[col].transform([selected])[0]

    else:

        value = st.number_input(
            col,
            value=float(df[col].mean())
        )

        user_input[col] = value

input_df = pd.DataFrame([user_input])

if st.button("Predict Price"):

    prediction = best_model.predict(input_df)[0]

    st.success(
        f"Predicted House Price: ₹ {prediction:,.2f}"
    )

# ----------------------------
# Feature Importance
# ----------------------------
st.subheader("Feature Importance")

importance = pd.Series(
    rf.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

st.bar_chart(importance)

# ----------------------------
# Dataset Statistics
# ----------------------------
with st.expander("Dataset Statistics"):
    st.write(df.describe())