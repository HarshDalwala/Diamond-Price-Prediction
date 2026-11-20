import pandas as pd
import numpy as np
import streamlit as st  # type: ignore
import plotly.express as px  # type: ignore
from sklearn.model_selection import train_test_split
import sklearn.preprocessing

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import SVR

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
    mean_absolute_error
)

import pickle
from pathlib import Path

st.set_page_config(
    page_title="Diamonds Data Analysis App",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Diamonds Data Analysis App")

# ---------------------------------------------------------
# Data Loading
# ---------------------------------------------------------
# Keep the CSV file in the same folder as this Python file.
base_folder = Path("diamonds.csv")

dataset_path = base_folder if base_folder.exists() else None

if dataset_path is None:
    st.error(
        "Dataset file not found. Please keep 'diamonds(1).csv' "
        "or 'diamonds.csv' in the same folder as this Python file."
    )
    st.stop()

df = pd.read_csv(dataset_path)

if "Unnamed: 0" in df.columns:
    df.drop(columns=["Unnamed: 0"], inplace=True)

num_col_eda = ['carat', 'depth', 'table', 'price', 'x', 'y', 'z']
cat_col_eda = ['cut', 'color', 'clarity']

# ---------------------------------------------------------
# Sidebar: Navigation ONLY
# ---------------------------------------------------------
st.sidebar.header("Navigation")

if "active_section" not in st.session_state:
    st.session_state.active_section = "Data Summary"

if st.sidebar.button("Show Data Summary", use_container_width=True):
    st.session_state.active_section = "Data Summary"

if st.sidebar.button("EDA - Exploratory Data Analysis", use_container_width=True):
    st.session_state.active_section = "EDA"

if st.sidebar.button("Show Train-Test Split Summary", use_container_width=True):
    st.session_state.active_section = "Train-Test Split"

if st.sidebar.button("Show Model Selection", use_container_width=True):
    st.session_state.active_section = "Model Selection"

if st.sidebar.button("Prediction", use_container_width=True):
    st.session_state.active_section = "Prediction"

active_section = st.session_state.active_section

# ---------------------------------------------------------
# Main Screen: Section-specific settings
# All selection controls are displayed on the main screen.
# ---------------------------------------------------------
test_size = st.session_state.get("test_size", 0.2)
random_state = st.session_state.get("random_state", 3)

model_list = [
    'Decision Tree Regressor',
    'Random Forest Regressor',
    'Gradient Boosting Regressor',
    'Linear Regression',
    'Ridge Regression',
    'Lasso Regression',
    'K-Nearest Neighbors Regressor',
    'AdaBoost Regressor',
    'Support Vector Regressor',
    'Polynomial Regression'
]

if "selected_model" not in st.session_state:
    st.session_state.selected_model = model_list[0]

model_choice = st.session_state.selected_model
degree = st.session_state.get("polynomial_degree", 2)

analysis_type = None
col_name = None
col1_name = None

if active_section == "Train-Test Split":
    st.header("Train-Test Split Settings")

    test_size = st.slider(
        "Select Test Size",
        min_value=0.1,
        max_value=0.9,
        value=float(test_size),
        step=0.1,
        key="test_size"
    )

    random_state = st.number_input(
        "Select Random State",
        min_value=0,
        max_value=100,
        value=int(random_state),
        step=1,
        key="random_state"
    )

elif active_section == "EDA":
    st.header("EDA - Exploratory Data Analysis")

    analysis_type = st.selectbox(
        "Select a feature for analysis",
        [
            'Univariate Analysis',
            'Bivariate Analysis',
            'Multivariate Analysis'
        ],
        key="eda_analysis_type"
    )

    if analysis_type == 'Univariate Analysis':
        col_name = st.selectbox(
            "Select the column for Univariate Analysis",
            df.columns,
            key="eda_univariate_column"
        )

    elif analysis_type in ['Bivariate Analysis', 'Multivariate Analysis']:
        col1_name = st.selectbox(
            "Select the first column for analysis",
            df.columns,
            key="eda_other_column"
        )

elif active_section == "Model Selection":
    st.header("Model Selection")

    model_choice = st.selectbox(
        "Select a model for training",
        model_list,
        index=model_list.index(st.session_state.selected_model),
        key="model_choice"
    )
    st.session_state.selected_model = model_choice

    if model_choice == 'Polynomial Regression':
        degree = st.slider(
            "Select the degree for Polynomial Regression",
            min_value=2,
            max_value=5,
            value=int(degree),
            step=1,
            key="polynomial_degree"
        )

# ---------------------------------------------------------
# Train-Test Split
# ---------------------------------------------------------
x = df.drop(columns=["price"])
y = df["price"]

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=test_size,
    random_state=int(random_state)
)

# ---------------------------------------------------------
# Data Preprocessing
# ---------------------------------------------------------
cat_col = ['cut', 'color', 'clarity']
num_col = ['carat', 'depth', 'table', 'x', 'y', 'z']

scaler = sklearn.preprocessing.StandardScaler()
encoder = sklearn.preprocessing.OrdinalEncoder()

# Numerical columns scaling
num_train = x_train[num_col]
num_test = x_test[num_col]

num_train_scaled = scaler.fit_transform(num_train)
num_train_scaled_df = pd.DataFrame(
    num_train_scaled,
    columns=num_col,
    index=num_train.index
)

num_test_scaled = scaler.transform(num_test)
num_test_scaled_df = pd.DataFrame(
    num_test_scaled,
    columns=num_col,
    index=num_test.index
)

# Categorical columns encoding
cat_train = x_train[cat_col]
cat_test = x_test[cat_col]

cat_train_encoded = encoder.fit_transform(cat_train)
cat_train_encoded_df = pd.DataFrame(
    cat_train_encoded,
    columns=cat_col,
    index=cat_train.index
)

cat_test_encoded = encoder.transform(cat_test)
cat_test_encoded_df = pd.DataFrame(
    cat_test_encoded,
    columns=cat_col,
    index=cat_test.index
)

# Join scaled numerical and encoded categorical columns
x_test_scaled_encoded = num_test_scaled_df.join(cat_test_encoded_df)
x_train_scaled_encoded = num_train_scaled_df.join(cat_train_encoded_df)

# ---------------------------------------------------------
# Model Training
# The selected model is remembered when navigating sections.
# ---------------------------------------------------------
model_choice = st.session_state.selected_model

if model_choice == 'Decision Tree Regressor':
    model = DecisionTreeRegressor()
    model.fit(x_train_scaled_encoded, y_train)

elif model_choice == 'Random Forest Regressor':
    model = RandomForestRegressor()
    model.fit(x_train_scaled_encoded, y_train)

elif model_choice == 'Gradient Boosting Regressor':
    model = GradientBoostingRegressor()
    model.fit(x_train_scaled_encoded, y_train)

elif model_choice == 'Linear Regression':
    model = LinearRegression()
    model.fit(x_train_scaled_encoded, y_train)

elif model_choice == 'Ridge Regression':
    model = Ridge()
    model.fit(x_train_scaled_encoded, y_train)

elif model_choice == 'Lasso Regression':
    model = Lasso()
    model.fit(x_train_scaled_encoded, y_train)

elif model_choice == 'K-Nearest Neighbors Regressor':
    model = KNeighborsRegressor()
    model.fit(x_train_scaled_encoded, y_train)

elif model_choice == 'AdaBoost Regressor':
    model = AdaBoostRegressor()
    model.fit(x_train_scaled_encoded, y_train)

elif model_choice == 'Support Vector Regressor':
    model = SVR()
    model.fit(x_train_scaled_encoded, y_train)

elif model_choice == 'Polynomial Regression':
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(x_train_scaled_encoded, y_train)

# Prepare downloadable model file
models = {
    'Pred_model': model,
    'Scaler': scaler,
    'Encoder': encoder
}
model_bytes = pickle.dumps(models)

# ---------------------------------------------------------
# Main Screen: Section Output
# ---------------------------------------------------------

# 1. Data Summary
if active_section == "Data Summary":
    st.header("Data Summary")
    st.dataframe(df.head(), use_container_width=True)

# 2. EDA
elif active_section == "EDA":
    col1, col2 = st.columns([1, 1])

    if analysis_type == 'Univariate Analysis':
        if col_name in num_col_eda:
            chart1 = px.histogram(
                df[col_name],
                title=f"Distribution of {col_name}"
            )
            col1.plotly_chart(chart1, use_container_width=True)

            chart2 = px.box(
                df[col_name],
                title=f"Box Plot of {col_name}"
            )
            col2.plotly_chart(chart2, use_container_width=True)

            chart3 = px.violin(
                df[col_name],
                title=f"Violin Plot of {col_name}"
            )
            col1.plotly_chart(chart3, use_container_width=True)

        elif col_name in cat_col_eda:
            df_cat_eda = df.groupby(col_name)['price'].count().reset_index()
            df_cat_eda.rename(columns={'price': 'count'}, inplace=True)

            chart1 = px.bar(
                df_cat_eda,
                x=col_name,
                y='count',
                title=f"Count by {col_name}"
            )
            col1.plotly_chart(chart1, use_container_width=True)

            chart2 = px.pie(
                df_cat_eda,
                names=col_name,
                values='count',
                title=f"Distribution by {col_name}"
            )
            col2.plotly_chart(chart2, use_container_width=True)

            chart3 = px.sunburst(
                df_cat_eda,
                path=[col_name],
                values='count',
                title=f"Sunburst Chart of {col_name}"
            )
            col1.plotly_chart(chart3, use_container_width=True)

    elif analysis_type == 'Bivariate Analysis':
        target_col = df['price']

        if col1_name in num_col_eda:
            chart1 = px.scatter(
                df,
                x=col1_name,
                y=target_col,
                title=f"Scatter Plot of {col1_name} vs Price"
            )
            col1.plotly_chart(chart1, use_container_width=True)

            df_num = df.groupby(col1_name)['price'].mean().reset_index()
            chart2 = px.line(
                df_num,
                x=col1_name,
                y='price',
                title=f"Line Plot of {col1_name} vs Price"
            )
            col2.plotly_chart(chart2, use_container_width=True)

            chart3 = px.box(
                df,
                x=col1_name,
                y=target_col,
                title=f"Box Plot of {col1_name} vs Price"
            )
            col1.plotly_chart(chart3, use_container_width=True)

        elif col1_name in cat_col_eda:
            df_cat_bi = df.groupby(col1_name)['price'].mean().reset_index()
            df_cat_bi.rename(columns={'price': 'Average Price'}, inplace=True)

            chart1 = px.bar(
                df_cat_bi,
                x=col1_name,
                y='Average Price',
                title=f"Average Price by {col1_name}"
            )
            col1.plotly_chart(chart1, use_container_width=True)

            chart2 = px.box(
                df,
                x=col1_name,
                y='price',
                title=f"Price Distribution by {col1_name}"
            )
            col2.plotly_chart(chart2, use_container_width=True)

            chart3 = px.sunburst(
                df_cat_bi,
                path=[col1_name],
                values='Average Price',
                title=f"Average Price Sunburst by {col1_name}"
            )
            col1.plotly_chart(chart3, use_container_width=True)

    elif analysis_type == 'Multivariate Analysis':
        chart1 = px.scatter_matrix(
            df,
            dimensions=[col1_name, 'price'],
            color='cut',
            title=f"Scatter Matrix of {col1_name} and Price by Cut"
        )
        col1.plotly_chart(chart1, use_container_width=True)

        chart2 = px.scatter_matrix(
            df,
            dimensions=[col1_name, 'price'],
            color='color',
            title=f"Scatter Matrix of {col1_name} and Price by Color"
        )
        col2.plotly_chart(chart2, use_container_width=True)

# 3. Train-Test Split Summary
elif active_section == "Train-Test Split":
    st.subheader("Train-Test Split Summary")
    st.write(f"X Train Shape: {x_train.shape}")
    st.write(f"X Test Shape: {x_test.shape}")
    st.write(f"Y Train Shape: {y_train.shape}")
    st.write(f"Y Test Shape: {y_test.shape}")

# 4. Model Selection
elif active_section == "Model Selection":
    st.write(f"Selected Model: {model_choice}")

    st.subheader("Model Performance")
    y_pred = model.predict(x_test_scaled_encoded)

    mse = mean_squared_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.write(f"Mean Squared Error (MSE) = {mse:.2f}")
    st.write(f"Root Mean Squared Error (RMSE) = {rmse:.2f}")
    st.write(f"Mean Absolute Error (MAE) = {mae:.2f}")
    st.write(f"R-squared (R²) = {r2:.2f}")

    st.download_button(
        label="Download Models",
        data=model_bytes,
        file_name="models.pkl"
    )

# 5. Prediction
elif active_section == "Prediction":
    st.header("Diamond Price Prediction")
    st.write("Enter Diamond Details to Predict the Price")

    st.download_button(
        label="Download Models",
        data=model_bytes,
        file_name="models.pkl"
    )

    with st.form("diamond_prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            carat = st.number_input(
                "Carat",
                min_value=0.1,
                value=0.5,
                step=0.01
            )
            cut = st.selectbox("Cut", df['cut'].unique())
            color = st.selectbox("Color", df['color'].unique())

        with col2:
            clarity = st.selectbox("Clarity", df['clarity'].unique())
            depth = st.number_input(
                "Depth",
                min_value=0.1,
                value=61.5,
                step=0.1
            )
            table = st.number_input(
                "Table",
                min_value=0.1,
                value=57.0,
                step=0.1
            )

        with col3:
            x_value = st.number_input(
                "X",
                min_value=0.1,
                value=5.0,
                step=0.1
            )
            y_value = st.number_input(
                "Y",
                min_value=0.1,
                value=5.0,
                step=0.1
            )
            z_value = st.number_input(
                "Z",
                min_value=0.1,
                value=3.0,
                step=0.1
            )

        predict_button = st.form_submit_button("Predict Diamond Price")

    if predict_button:
        input_data = pd.DataFrame({
            'carat': [carat],
            'depth': [depth],
            'table': [table],
            'x': [x_value],
            'y': [y_value],
            'z': [z_value],
            'cut': [cut],
            'color': [color],
            'clarity': [clarity]
        })

        input_num = scaler.transform(input_data[num_col])
        input_num_df = pd.DataFrame(input_num, columns=num_col)

        input_cat = encoder.transform(input_data[cat_col])
        input_cat_df = pd.DataFrame(input_cat, columns=cat_col)

        final_input = input_num_df.join(input_cat_df)

        prediction = model.predict(final_input)

        st.success("Prediction Completed Successfully!")
        st.metric(
            "Predicted Diamond Price",
            f"${prediction[0]:,.2f}"
        )
