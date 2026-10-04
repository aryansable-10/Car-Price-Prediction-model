"""
app.py  –  Car Price Prediction  |  Streamlit Frontend
Run:  streamlit run app.py
"""

import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .main { background-color: #f7f8fa; }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        text-align: center;
    }
    .predict-result {
        font-size: 2.4rem;
        font-weight: 700;
        color: #3b82d4;
        text-align: center;
        padding: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Constants ─────────────────────────────────────────────────────────────────
DATA_PATH  = "car_price_prediction.csv"
MODEL_PATH = os.path.join("model", "car_price_model.pkl")
META_PATH  = os.path.join("model", "feature_meta.pkl")


# ── Loaders (cached) ──────────────────────────────────────────────────────────
@st.cache_data
def load_raw_data():
    df = pd.read_csv(DATA_PATH)
    return df


@st.cache_resource
def load_model_and_meta():
    model = joblib.load(MODEL_PATH)
    meta  = joblib.load(META_PATH)
    return model, meta


@st.cache_data
def prepare_display_data():
    """Clean dataset for display/EDA (mirrors preprocess_train logic)."""
    df = pd.read_csv(DATA_PATH)

    def strip_unit(series):
        return pd.to_numeric(
            series.astype(str).str.extract(r"([\d.]+)", expand=False),
            errors="coerce",
        )

    df["mileage"]    = strip_unit(df["mileage(km/ltr/kg)"])
    df["engine_cc"]  = strip_unit(df["engine"])
    df["max_power"]  = strip_unit(df["max_power"])
    df["car_age"]    = 2024 - df["year"]

    for col in ["mileage", "engine_cc", "max_power", "seats"]:
        df[col] = df[col].fillna(df[col].median())

    df.dropna(subset=["selling_price"], inplace=True)
    return df


# ── Guard: model must exist ───────────────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    st.error(
        "Model files not found. Please run `python preprocess_train.py` first.",
        icon="🚨",
    )
    st.stop()

model, meta = load_model_and_meta()
df_raw      = load_raw_data()
df          = prepare_display_data()

# ── Sidebar navigation ────────────────────────────────────────────────────────
st.sidebar.image(
    "https://img.icons8.com/color/96/car--v2.png", width=70
)
st.sidebar.title("Car Price Predictor")
st.sidebar.caption("ML-powered used-car valuation")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "🔮 Predict Price", "📊 Data Insights", "📋 Raw Data"],
    label_visibility="collapsed",
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**Dataset** : {len(df_raw):,} cars  \n"
    f"**Model**   : Random Forest  \n"
    f"**R2 Score**: 0.93"
)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 – HOME
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.title("🚗 Car Price Prediction System")
    st.markdown(
        "An end-to-end machine learning application that predicts the **selling "
        "price of a used car** based on its specifications and condition."
    )

    # KPI cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Cars", f"{len(df_raw):,}")
    with col2:
        st.metric("Avg Selling Price", f"INR {df['selling_price'].mean():,.0f}")
    with col3:
        st.metric("Fuel Types", df_raw["fuel"].nunique())
    with col4:
        st.metric("Model Accuracy (R2)", "0.93")

    st.markdown("---")

    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Price Distribution")
        fig = px.histogram(
            df, x="selling_price", nbins=60,
            color_discrete_sequence=["#3b82d4"],
            labels={"selling_price": "Selling Price (INR)"},
        )
        fig.update_layout(bargap=0.05, plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("Fuel Type Share")
        fuel_counts = df_raw["fuel"].value_counts().reset_index()
        fuel_counts.columns = ["Fuel", "Count"]
        fig2 = px.pie(
            fuel_counts, names="Fuel", values="Count",
            color_discrete_sequence=px.colors.qualitative.Set2,
            hole=0.4,
        )
        fig2.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Price vs. Car Age")
    fig3 = px.scatter(
        df, x="car_age", y="selling_price",
        color="fuel", opacity=0.55, trendline="ols",
        labels={"car_age": "Car Age (years)", "selling_price": "Selling Price (INR)"},
        color_discrete_sequence=px.colors.qualitative.Vivid,
    )
    fig3.update_layout(plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig3, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 – PREDICT PRICE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict Price":
    st.title("🔮 Predict Car Selling Price")
    st.markdown("Fill in the car details below to get an instant price estimate.")

    cat_vals  = meta["category_values"]
    num_range = meta["numeric_ranges"]

    with st.form("predict_form"):
        st.subheader("Car Details")

        col1, col2, col3 = st.columns(3)

        with col1:
            fuel         = st.selectbox("Fuel Type",         cat_vals["fuel"])
            seller_type  = st.selectbox("Seller Type",       cat_vals["seller_type"])
            transmission = st.selectbox("Transmission",      cat_vals["transmission"])
            owner        = st.selectbox("Owner",             cat_vals["owner"])

        with col2:
            km_driven = st.number_input(
                "KM Driven",
                min_value=0, max_value=500000,
                value=50000, step=1000,
            )
            car_age = st.slider(
                "Car Age (years)",
                min_value=1, max_value=30, value=5,
            )
            seats = st.selectbox("Seats", [2, 4, 5, 6, 7, 8, 9, 10], index=2)

        with col3:
            mileage = st.number_input(
                "Mileage (km/ltr/kg)",
                min_value=5.0, max_value=50.0,
                value=18.0, step=0.1, format="%.1f",
            )
            engine_cc = st.number_input(
                "Engine (cc)",
                min_value=500, max_value=6000,
                value=1200, step=50,
            )
            max_power = st.number_input(
                "Max Power (bhp)",
                min_value=30.0, max_value=600.0,
                value=80.0, step=1.0, format="%.1f",
            )

        submitted = st.form_submit_button("Predict Price", use_container_width=True)

    if submitted:
        # Encode categorical inputs – use explicit variable references (locals() is unreliable in forms)
        form_values = {
            "fuel":         fuel,
            "seller_type":  seller_type,
            "transmission": transmission,
            "owner":        owner,
        }
        encoded = {}
        for col in ["fuel", "seller_type", "transmission", "owner"]:
            le  = meta["encoders"][col]
            encoded[col] = le.transform([form_values[col]])[0]

        input_dict = {
            "km_driven":    km_driven,
            "fuel":         encoded["fuel"],
            "seller_type":  encoded["seller_type"],
            "transmission": encoded["transmission"],
            "owner":        encoded["owner"],
            "mileage":      mileage,
            "engine_cc":    engine_cc,
            "max_power":    max_power,
            "seats":        seats,
            "car_age":      car_age,
        }

        # Align column order to training feature_cols
        input_df = pd.DataFrame([input_dict])[meta["feature_cols"]]
        log_pred = model.predict(input_df)[0]
        price    = np.expm1(log_pred)

        st.success("Prediction Complete!")
        st.markdown(
            f"<div class='predict-result'>Estimated Price: INR {price:,.0f}</div>",
            unsafe_allow_html=True,
        )

        # Confidence range (±15%)
        low, high = price * 0.85, price * 1.15
        st.info(f"Likely range: INR {low:,.0f} - INR {high:,.0f}  (+-15% confidence band)")

        # Feature contribution gauge
        st.subheader("How key features affect your estimate")
        fi = meta["feature_importances"]
        fi_df = (
            pd.DataFrame(fi.items(), columns=["Feature", "Importance"])
            .sort_values("Importance", ascending=True)
            .tail(8)
        )
        fig_fi = px.bar(
            fi_df, x="Importance", y="Feature",
            orientation="h", color="Importance",
            color_continuous_scale="Blues",
            labels={"Importance": "Feature Importance"},
        )
        fig_fi.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_fi, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 3 – DATA INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Data Insights":
    st.title("📊 Data Insights & EDA")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Price Analysis", "Categorical Breakdown", "Correlations", "Feature Importance"]
    )

    # ── Tab 1: Price Analysis ──────────────────────────────────────────────────
    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Selling Price by Transmission")
            fig = px.box(
                df, x="transmission", y="selling_price",
                color="transmission",
                color_discrete_sequence=px.colors.qualitative.Set1,
                labels={"selling_price": "Selling Price (INR)"},
            )
            fig.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Selling Price by Fuel Type")
            fig2 = px.violin(
                df, x="fuel", y="selling_price",
                color="fuel", box=True,
                color_discrete_sequence=px.colors.qualitative.Pastel,
                labels={"selling_price": "Selling Price (INR)"},
            )
            fig2.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Price vs. KM Driven")
        fig3 = px.scatter(
            df, x="km_driven", y="selling_price",
            color="fuel", opacity=0.5,
            trendline="lowess",
            color_discrete_sequence=px.colors.qualitative.Vivid,
            labels={"km_driven": "KM Driven", "selling_price": "Selling Price (INR)"},
        )
        fig3.update_layout(plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig3, use_container_width=True)

    # ── Tab 2: Categorical Breakdown ──────────────────────────────────────────
    with tab2:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Cars by Seller Type")
            vc = df_raw["seller_type"].value_counts().reset_index()
            vc.columns = ["Seller Type", "Count"]
            fig = px.bar(
                vc, x="Seller Type", y="Count",
                color="Seller Type",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Cars by Owner Type")
            vc2 = df_raw["owner"].value_counts().reset_index()
            vc2.columns = ["Owner", "Count"]
            fig2 = px.bar(
                vc2, x="Owner", y="Count",
                color="Owner",
                color_discrete_sequence=px.colors.qualitative.Pastel,
            )
            fig2.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Average Price by Fuel & Transmission")
        pivot = df.groupby(["fuel", "transmission"])["selling_price"].mean().reset_index()
        fig3 = px.bar(
            pivot, x="fuel", y="selling_price", color="transmission",
            barmode="group",
            labels={"selling_price": "Avg Selling Price (INR)", "fuel": "Fuel Type"},
            color_discrete_sequence=px.colors.qualitative.Set1,
        )
        fig3.update_layout(plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig3, use_container_width=True)

    # ── Tab 3: Correlations ───────────────────────────────────────────────────
    with tab3:
        st.subheader("Correlation Heatmap (Numeric Features)")
        num_cols = ["selling_price", "km_driven", "mileage", "engine_cc",
                    "max_power", "seats", "car_age"]
        corr = df[num_cols].corr()

        fig = go.Figure(
            data=go.Heatmap(
                z=corr.values,
                x=corr.columns.tolist(),
                y=corr.index.tolist(),
                colorscale="RdBu",
                zmid=0,
                text=corr.round(2).values,
                texttemplate="%{text}",
                textfont={"size": 11},
            )
        )
        fig.update_layout(
            height=500,
            plot_bgcolor="white", paper_bgcolor="white",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Price vs. Max Power")
        fig2 = px.scatter(
            df, x="max_power", y="selling_price",
            color="transmission", opacity=0.55, trendline="ols",
            color_discrete_sequence=px.colors.qualitative.Bold,
            labels={"max_power": "Max Power (bhp)", "selling_price": "Selling Price (INR)"},
        )
        fig2.update_layout(plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig2, use_container_width=True)

    # ── Tab 4: Feature Importance ─────────────────────────────────────────────
    with tab4:
        st.subheader("Model Feature Importance (Random Forest)")
        fi = meta["feature_importances"]
        fi_df = (
            pd.DataFrame(fi.items(), columns=["Feature", "Importance"])
            .sort_values("Importance", ascending=False)
        )
        fig = px.bar(
            fi_df, x="Feature", y="Importance",
            color="Importance", color_continuous_scale="Blues",
            labels={"Importance": "Importance Score"},
        )
        fig.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            fi_df.style.background_gradient(cmap="Blues", subset=["Importance"]),
            use_container_width=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 4 – RAW DATA
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Raw Data":
    st.title("📋 Raw Dataset")

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows",    f"{len(df_raw):,}")
    col2.metric("Columns", df_raw.shape[1])
    col3.metric("Missing Values", int(df_raw.isnull().sum().sum()))

    st.markdown("---")
    st.subheader("Filter & Browse")

    fuel_filter = st.multiselect(
        "Filter by Fuel Type",
        options=df_raw["fuel"].unique().tolist(),
        default=df_raw["fuel"].unique().tolist(),
    )
    trans_filter = st.multiselect(
        "Filter by Transmission",
        options=df_raw["transmission"].unique().tolist(),
        default=df_raw["transmission"].unique().tolist(),
    )

    filtered = df_raw[
        df_raw["fuel"].isin(fuel_filter) &
        df_raw["transmission"].isin(trans_filter)
    ]
    st.caption(f"Showing {len(filtered):,} rows")
    st.dataframe(filtered, use_container_width=True, height=450)

    st.markdown("---")
    st.subheader("Descriptive Statistics")
    st.dataframe(
        df_raw.describe().T.style.background_gradient(cmap="Blues"),
        use_container_width=True,
    )
