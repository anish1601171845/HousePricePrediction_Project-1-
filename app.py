import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(
    page_title="Smart House Price Predictor",
    page_icon="🏡",
    layout="wide"
)

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("Housing.csv")

df = load_data()

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("🏡 Smart Real Estate Dashboard")

page = st.sidebar.radio(
    "Navigate",
    ["Market Insights", "Price Predictor", "Investment Score"]
)

# -----------------------------
# Market Insights
# -----------------------------
if page == "Market Insights":

    st.title("📊 Housing Market Insights")

    col1, col2, col3 = st.columns(3)

    col1.metric("Average Price", f"₹ {df['price'].mean():,.0f}")
    col2.metric("Maximum Price", f"₹ {df['price'].max():,.0f}")
    col3.metric("Minimum Price", f"₹ {df['price'].min():,.0f}")

    st.markdown("---")

    c1, c2 = st.columns(2)

    with c1:
        fig = px.histogram(
            df,
            x="price",
            nbins=30,
            title="House Price Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.scatter(
            df,
            x="area",
            y="price",
            title="Area vs Price"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("🔥 Correlation Heatmap")

    numeric_df = df.select_dtypes(include=np.number)

    fig = px.imshow(
        numeric_df.corr(),
        text_auto=True,
        aspect="auto"
    )

    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Price Prediction
# -----------------------------
elif page == "Price Predictor":

    st.title("🏠 House Price Prediction")

    data = df.copy()

    binary_cols = [
        'mainroad',
        'guestroom',
        'basement',
        'hotwaterheating',
        'airconditioning',
        'prefarea'
    ]

    for col in binary_cols:
        data[col] = data[col].map({'yes': 1, 'no': 0})

    data = pd.get_dummies(
        data,
        columns=['furnishingstatus'],
        drop_first=True
    )

    X = data.drop("price", axis=1)
    y = data["price"]

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )

    model.fit(X, y)

    st.subheader("Enter Property Details")

    col1, col2 = st.columns(2)

    with col1:
        area = st.slider("Area (sq.ft)", 1000, 10000, 3000)
        bedrooms = st.slider("Bedrooms", 1, 10, 3)
        bathrooms = st.slider("Bathrooms", 1, 10, 2)
        stories = st.slider("Stories", 1, 5, 2)

    with col2:
        parking = st.slider("Parking Spaces", 0, 5, 1)
        mainroad = st.selectbox("Main Road Access", [0, 1])
        airconditioning = st.selectbox("Air Conditioning", [0, 1])
        prefarea = st.selectbox("Preferred Area", [0, 1])

    sample = X.mean().to_frame().T

    sample["area"] = area
    sample["bedrooms"] = bedrooms
    sample["bathrooms"] = bathrooms
    sample["stories"] = stories
    sample["parking"] = parking

    if "mainroad" in sample.columns:
        sample["mainroad"] = mainroad

    if "airconditioning" in sample.columns:
        sample["airconditioning"] = airconditioning

    if "prefarea" in sample.columns:
        sample["prefarea"] = prefarea

    if st.button("Predict Price"):

        prediction = model.predict(sample)[0]

        st.success(
            f"🏡 Estimated House Price: ₹ {prediction:,.0f}"
        )

        st.balloons()

        st.subheader("Feature Importance")

        importance = pd.DataFrame({
            "Feature": X.columns,
            "Importance": model.feature_importances_
        })

        importance = importance.sort_values(
            "Importance",
            ascending=False
        ).head(10)

        fig = px.bar(
            importance,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Top Factors Affecting Price"
        )

        st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Investment Score
# -----------------------------
else:

    st.title("🎯 Property Investment Score")

    area = st.slider("Area", 1000, 10000, 3000)
    parking = st.slider("Parking", 0, 5, 2)
    bathrooms = st.slider("Bathrooms", 1, 10, 2)

    score = (
        (area / 10000) * 50 +
        (parking / 5) * 25 +
        (bathrooms / 10) * 25
    )

    st.metric(
        "Investment Score",
        f"{score:.1f}/100"
    )

    if score >= 80:
        st.success("Excellent Investment Opportunity 🚀")

    elif score >= 60:
        st.info("Good Property 👍")

    else:
        st.warning("Average Investment Potential ⚠️")

    gauge = px.pie(
        values=[score, 100-score],
        names=["Score", "Remaining"],
        title="Investment Potential"
    )

    st.plotly_chart(gauge, use_container_width=True)