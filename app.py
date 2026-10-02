import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

# ==============================================================================
# Page Configuration & UI Layout
# ==============================================================================
st.set_page_config(
    page_title="Rwanda House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Metric Cards & Headers
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# Data Loading and Pipeline Training Functions
# ==============================================================================
@st.cache_data
def load_and_clean_data(file_path):
    """Loads dataset and performs initial cleaning (duplicate removal)."""
    df = pd.read_csv(file_path)
    df = df.drop_duplicates()
    return df

@st.cache_resource
def train_model_pipeline(df):
    """Trains the preprocessing and Linear Regression pipeline."""
    # Define features and target
    X = df.drop(columns=["House_ID", "House_Price_Million_RWF"])
    y = df["House_Price_Million_RWF"]
    
    # Drop target nulls if present
    valid_target_idx = y.notna()
    X = X[valid_target_idx]
    y = y[valid_target_idx]

    num_features = ["Area_m2", "Bedrooms", "Bathrooms", "House_Age_Years", "Distance_to_City_km", "Parking_Spaces"]
    cat_features = ["Neighborhood"]

    # Numeric pipeline with Median Imputation
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median'))
    ])

    # Categorical pipeline with Most Frequent Imputation and One-Hot Encoding
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])

    # Combined preprocessor
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_pipeline, num_features),
        ('cat', cat_pipeline, cat_features)
    ])

    # Full Machine Learning Pipeline
    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model_pipeline.fit(X_train, y_train)

    # Predictions & Evaluation
    y_pred = model_pipeline.predict(X_test)
    metrics = {
        "R2": r2_score(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred))
    }

    return model_pipeline, metrics, X_test, y_test, y_pred

# Load data and model
DATA_PATH = "house_price_prediction_dataset.csv"
try:
    df = load_and_clean_data(DATA_PATH)
    pipeline, metrics, X_test, y_test, y_pred = train_model_pipeline(df)
except Exception as e:
    st.error(f"Error loading dataset: {e}. Please ensure 'house_price_prediction_dataset.csv' is present in the current working directory.")
    st.stop()

# ==============================================================================
# Navigation / Sidebar
# ==============================================================================
st.sidebar.image("https://img.icons8.com/color/96/000000/real-estate.png", width=80)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🏠 Price Predictor", "📊 Data Overview & EDA", "📈 Model Evaluation"])

st.sidebar.markdown("---")
st.sidebar.caption("Rwanda Housing Analytics Dashboard")

# ==============================================================================
# PAGE 1: House Price Predictor
# ==============================================================================
if page == "🏠 Price Predictor":
    st.markdown('<div class="main-header">Rwanda House Price Estimator</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Input house characteristics to generate real-time market value estimates (in Million RWF).</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📋 House Specifications")
        
        neighborhood_options = sorted(df["Neighborhood"].dropna().unique().tolist())
        neighborhood = st.selectbox("Neighborhood / Location", options=neighborhood_options)
        
        area = st.number_input(
            "Property Area (m²)", 
            min_value=20.0, 
            max_value=1500.0, 
            value=float(df["Area_m2"].median()),
            step=5.0
        )
        
        col_bed, col_bath = st.columns(2)
        with col_bed:
            bedrooms = st.slider("Bedrooms", min_value=1, max_value=8, value=int(df["Bedrooms"].median()))
        with col_bath:
            bathrooms = st.slider("Bathrooms", min_value=1, max_value=6, value=int(df["Bathrooms"].median()))

        col_age, col_dist, col_park = st.columns(3)
        with col_age:
            age = st.number_input("House Age (Years)", min_value=0.0, max_value=100.0, value=float(df["House_Age_Years"].median()))
        with col_dist:
            distance = st.number_input("Distance to City (km)", min_value=0.0, max_value=100.0, value=float(df["Distance_to_City_km"].median()))
        with col_park:
            parking = st.selectbox("Parking Spaces", options=[0, 1, 2, 3, 4], index=1)

    with col2:
        st.subheader("💡 Estimated Valuation")
        
        input_data = pd.DataFrame([{
            "Area_m2": area,
            "Bedrooms": bedrooms,
            "Bathrooms": bathrooms,
            "House_Age_Years": age,
            "Distance_to_City_km": distance,
            "Parking_Spaces": parking,
            "Neighborhood": neighborhood
        }])

        predicted_price = pipeline.predict(input_data)[0]
        predicted_price = max(0, predicted_price)  # Ensure non-negative pricing

        st.markdown("<br>", unsafe_allow_html=True)
        st.metric(
            label="Estimated Market Value", 
            value=f"{predicted_price:,.2f} Million RWF",
            delta=None
        )

        st.info("ℹ️ **Valuation Note:** Estimate calculated via a Linear Regression Machine Learning model trained on local Rwandan real estate transactions.")

        # Comparison with average neighbourhood price
        avg_price = df[df["Neighborhood"] == neighborhood]["House_Price_Million_RWF"].mean()
        if not np.isnan(avg_price):
            diff = predicted_price - avg_price
            st.write(f"Average price in **{neighborhood}**: **{avg_price:,.2f} Million RWF**")
            if diff > 0:
                st.write(f"This house is estimated to be **{abs(diff):,.2f} Million RWF higher** than the neighborhood average.")
            else:
                st.write(f"This house is estimated to be **{abs(diff):,.2f} Million RWF lower** than the neighborhood average.")

# ==============================================================================
# PAGE 2: Data Overview & EDA
# ==============================================================================
elif page == "📊 Data Overview & EDA":
    st.markdown('<div class="main-header">Dataset Overview & Exploratory Analysis</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📋 Data Summary", "📊 Feature Distributions", "🔗 Feature Correlations"])

    with tab1:
        st.subheader("Raw Dataset Preview")
        st.dataframe(df, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Rows", df.shape[0])
        col2.metric("Total Columns", df.shape[1])
        col3.metric("Unique Neighborhoods", df["Neighborhood"].nunique())

        st.subheader("Summary Statistics")
        st.dataframe(df.describe(), use_container_width=True)

    with tab2:
        st.subheader("Feature Distributions")
        selected_col = st.selectbox(
            "Select Feature to Visualize:",
            ["House_Price_Million_RWF", "Area_m2", "Distance_to_City_km", "House_Age_Years"]
        )
        
        fig = px.histogram(
            df, 
            x=selected_col, 
            color="Neighborhood",
            marginal="box",
            title=f"Distribution of {selected_col} across Neighborhoods",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Feature vs House Price")
        num_cols = ["Area_m2", "Distance_to_City_km", "House_Age_Years", "Bedrooms", "Bathrooms"]
        x_axis = st.selectbox("Select X-Axis Feature:", num_cols, index=0)

        fig_scatter = px.scatter(
            df,
            x=x_axis,
            y="House_Price_Million_RWF",
            color="Neighborhood",
            size="Area_m2",
            hover_data=["House_ID"],
            title=f"{x_axis} vs House Price (Million RWF)",
            template="plotly_white"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

# ==============================================================================
# PAGE 3: Model Evaluation
# ==============================================================================
elif page == "📈 Model Evaluation":
    st.markdown('<div class="main-header">Model Performance & Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Evaluation of the Linear Regression Pipeline on test data.</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("R² Score", f"{metrics['R2']:.4f}")
    col2.metric("Mean Absolute Error (MAE)", f"{metrics['MAE']:.2f} M RWF")
    col3.metric("Root Mean Squared Error (RMSE)", f"{metrics['RMSE']:.2f} M RWF")

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Actual vs Predicted House Prices")
        fig_eval = px.scatter(
            x=y_test, 
            y=y_pred,
            labels={'x': 'Actual Price (Million RWF)', 'y': 'Predicted Price (Million RWF)'},
            title="Actual vs Predicted Prices",
            template="plotly_white"
        )
        # 45-degree reference line for ideal predictions
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        fig_eval.add_shape(type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val,
                           line=dict(color="Red", dash="dash"))
        
        st.plotly_chart(fig_eval, use_container_width=True)

    with col_right:
        st.subheader("Residual Analysis")
        residuals = y_test - y_pred
        fig_res = px.histogram(
            residuals, 
            nbins=20,
            title="Distribution of Residuals (Actual - Predicted)",
            labels={'value': 'Residual Error'},
            template="plotly_white"
        )
        st.plotly_chart(fig_res, use_container_width=True)