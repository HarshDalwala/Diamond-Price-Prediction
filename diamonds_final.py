import pandas as pd
import numpy as np
import streamlit as st # type: ignore
import plotly.express as px # type: ignore
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

from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error, mean_absolute_error

import pickle

st.set_page_config(
    page_title="Diamonds Data Analysis App", 
    layout="wide",
    initial_sidebar_state="expanded")

st.title("Diamonds Data Analysis App")

#data load
df = pd.read_csv("C:\\Users\\admin\\Desktop\\Tops Technology\\Machine Learning\\data set\\1 Diamonds\\diamonds.csv")
df.drop(columns=["Unnamed: 0"], inplace=True)
#st.write(df.head())
#st.table(df.describe())
#st.dataframe(df)
num_col_eda = ['carat', 'depth', 'table', 'price', 'x', 'y', 'z']
cat_col_eda = ['cut', 'color', 'clarity']


#Data Summary
b1 = st.sidebar.checkbox("Show Data Summary")
if b1:
    st.write(df.head())


#EDA - Exploratory Data Analysis
col1, col2 = st.columns([1, 1])
b2 = st.sidebar.checkbox("EDA - Exploratory Data Analysis")
if b2:
        #st.write("This section provides an overview of the diamonds dataset, including visualizations and statistical summaries.")
    s1 = st.sidebar.selectbox("Select a feature for analysis", ['Univariate Analysis', 'Bivariate Analysis', 'Multivariate Analysis'])
    
    # Add EDA visualizations here
    if s1 == 'Univariate Analysis':
        col_name = st.sidebar.selectbox("Select a numerical column for univariate analysis", df.columns)
        if col_name in num_col_eda:
            chart1 = px.histogram(df[col_name], title=f"Distribution of {col_name}")
            col1.plotly_chart(chart1)

            chart2 = px.box(df[col_name], title=f"Box Plot of {col_name}")
            col2.plotly_chart(chart2)

            chart3 = px.violin(df[col_name], title=f"Violin Plot of {col_name}")
            col1.plotly_chart(chart3)
            

        elif col_name in cat_col_eda:
            df_cat_eda = df.groupby(col_name)['price'].count().reset_index()
            df_cat_eda.rename(columns={'price': 'count'}, inplace=True)
           # st.write(df_cat_eda)
            chart1 = px.bar(df_cat_eda, x=col_name, y='count', title=f"Average Price by {col_name}")
            col1.plotly_chart(chart1)

            chart2 = px.pie(df_cat_eda, names=col_name, values='count', title=f"Price Distribution by {col_name}")
            col2.plotly_chart(chart2)

            chart3 = px.sunburst(df_cat_eda, path=[col_name], values='count', title=f"Sunburst Chart of {col_name}")
            col1.plotly_chart(chart3)



    elif s1 == 'Bivariate Analysis':
        col1_name = st.sidebar.selectbox("Select the first numerical column for bivariate analysis", df.columns)
        target_col = df['price']
        if col1_name in num_col_eda:
            chart1 = px.scatter(df, x=col1_name, y=target_col, title=f"Scatter Plot of {col1_name} vs Price")
            col1.plotly_chart(chart1)

            df_num = df.groupby(col1_name)['price'].mean().reset_index()
            chart2 = px.line(df_num, x=col1_name, y='price', title=f"Line Plot of {col1_name} vs Price")
            col2.plotly_chart(chart2)

            chart3 = px.box(df, x=col1_name, y=target_col, title=f"Box Plot of {col1_name} vs Price")
            col1.plotly_chart(chart3)

        elif col1_name in cat_col_eda:
            df_cat_bi = df.groupby(col1_name)['price'].mean().reset_index()
            df_cat_bi.rename(columns={'price': 'count'}, inplace=True)
           # st.write(df_cat_bi)
            chart1 = px.bar(df_cat_bi, x=col1_name, y='count', title=f"Average Price by {col1_name}")
            col1.plotly_chart(chart1)

            chart2 = px.pie(df_cat_bi, names=col1_name, values='count', title=f"Price Distribution by {col1_name}")
            col2.plotly_chart(chart2)

            chart3 = px.sunburst(df_cat_bi, path=[col1_name], values='count', title=f"Sunburst Chart of {col1_name}")
            col1.plotly_chart(chart3)

    elif s1 == 'Multivariate Analysis':
        col1_name = st.sidebar.selectbox("Select the first numerical column for multivariate analysis", df.columns)
        pass  # Placeholder for multivariate analysis implementation

        chart1 = px.scatter_matrix(df, dimensions=[col1_name, 'price'], color='cut', title=f"Scatter Matrix of {col1_name} and Price by Cut")
        col1.plotly_chart(chart1)

        chart2 = px.scatter_matrix(df, dimensions=[col1_name, 'price'], color='color', title=f"Scatter Matrix of {col1_name} and Price by Color")
        col2.plotly_chart(chart2)


#train test split
x = df.drop(columns=["price"])
y = df["price"]

test_size = st.sidebar.slider("Test Size", min_value=0.1, max_value=0.9, value=0.2, step=0.1, key="test_size")
random_state = st.sidebar.number_input("Random State", min_value=0, max_value=100, value=3, step=1, key="random_state")
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)


b3 = st.sidebar.checkbox("Show Train-Test Split Summary")
if b3:
    st.write(f"X Train Shape: {x_train.shape}")
    st.write(f"X Test Shape: {x_test.shape}")
    st.write(f"Y Train Shape: {y_train.shape}")
    st.write(f"Y Test Shape: {y_test.shape}")


#data distribution
cat_col = ['cut', 'color', 'clarity']
num_col = ['carat', 'depth', 'table', 'x', 'y', 'z']

scaler = sklearn.preprocessing.StandardScaler()
encoder = sklearn.preprocessing.OrdinalEncoder()

#numerical columns scaling
num_train = x_train[num_col]
num_test = x_test[num_col]

num_train_scaled = scaler.fit_transform(num_train)
num_train_scaled_df = pd.DataFrame(num_train_scaled, columns=num_col, index=num_train.index)

num_test_scaled = scaler.transform(num_test)
num_test_scaled_df = pd.DataFrame(num_test_scaled, columns=num_col, index=num_test.index)


#categorical columns encoding
cat_train = x_train[cat_col]
cat_test = x_test[cat_col]

cat_train_encoded = encoder.fit_transform(cat_train)
cat_train_encoded_df = pd.DataFrame(cat_train_encoded, columns=cat_col, index=cat_train.index)

cat_test_encoded = encoder.transform(cat_test)
cat_test_encoded_df = pd.DataFrame(cat_test_encoded, columns=cat_col, index=cat_test.index)


#joining scaled numerical and encoded categorical columns
x_test_scaled_encoded = num_test_scaled_df.join(cat_test_encoded_df)
x_train_scaled_encoded = num_train_scaled_df.join(cat_train_encoded_df)



model_list = ['Decision Tree Regressor', 'Random Forest Regressor', 'Gradient Boosting Regressor', 
              'Linear Regression', 'Ridge Regression', 'Lasso Regression', 'K-Nearest Neighbors Regressor', 
              'AdaBoost Regressor', 'Support Vector Regressor','Polynomial Regression']

model_choice = st.sidebar.selectbox("Select a model for training", model_list)

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
    degree = st.sidebar.slider("Select the degree for Polynomial Regression", min_value=2, max_value=5, value=2, step=1)
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(x_train_scaled_encoded, y_train)


b4 = st.checkbox("Show Model Performance:")
if b4:
    y_pred = model.predict(x_test_scaled_encoded)
    mse = mean_squared_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.write(f"Model: {model_choice}")
    st.write(f"Mean Squared Error (MSE) = {mse:.2f}")
    st.write(f"Root Mean Squared Error (RMSE) = {rmse:.2f}")
    st.write(f"Mean Absolute Error (MAE) = {mae:.2f}")
    st.write(f"R-squared (R²) = {r2:.2f}")


models = {'Pred_model':model,"Scaler":scaler,"Encoder":encoder}
model_bytes = pickle.dumps(models)

# Download button
st.download_button(
label="Download Models",
data=model_bytes,
file_name="models.pkl")
