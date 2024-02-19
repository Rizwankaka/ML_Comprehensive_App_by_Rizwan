import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

# Function to load sample data
def load_sample_data(dataset_name):
    if dataset_name == 'tips':
        return sns.load_dataset('tips')
    elif dataset_name == 'titanic':
        return sns.load_dataset('titanic')

# Basic EDA function
def perform_eda(dataframe):
    st.write("Basic EDA Results")
    # Display summary statistics
    st.write("Summary Statistics:")
    st.write(dataframe.describe())
    # Display distribution of numerical data
    st.write("Data Distributions:")
    numerical_columns = dataframe.select_dtypes(include=['float64', 'int64']).columns
    for column in numerical_columns:
        st.write(f"Distribution for {column}:")
        fig, ax = plt.subplots()
        sns.histplot(dataframe[column], kde=True, ax=ax)
        st.pyplot(fig)
    # Display correlation matrix for numeric columns only
    st.write("Correlation Matrix:")
    numeric_df = dataframe.select_dtypes(include=['float64', 'int64'])
    if not numeric_df.empty:
        fig, ax = plt.subplots()
        sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.write("No numeric columns available for correlation matrix.")

# Function to dynamically generate plots based on user input
def generate_plots(dataframe):
    columns_to_plot = st.multiselect("Select the data columns to plot:", dataframe.columns)
    if columns_to_plot:
        data_types = dataframe[columns_to_plot].dtypes
        num_numeric = sum(pd.api.types.is_numeric_dtype(t) for t in data_types)
        num_categorical = sum(pd.api.types.is_categorical_dtype(t) or pd.api.types.is_object_dtype(t) for t in data_types)
        
        plot_options = []
        if num_numeric > 0:
            plot_options += ["Scatter Matrix", "Parallel Coordinates"]
        if len(columns_to_plot) == 1:
            if num_numeric == 1:
                plot_options += ["Histogram", "Box Plot", "Violin Plot"]
            else:
                plot_options += ["Bar Chart"]
        elif len(columns_to_plot) >= 2:
            if num_numeric == len(columns_to_plot):
                plot_options += ["Scatter Plot", "Line Chart"]
            if num_categorical > 0:
                plot_options += ["Facet Grid"]

        plot_type = st.selectbox("Select plot type:", plot_options)
        generate_plot(dataframe, columns_to_plot, plot_type, num_numeric, num_categorical)

def generate_plot(dataframe, columns_to_plot, plot_type, num_numeric, num_categorical):
    if plot_type == "Histogram":
        fig = px.histogram(dataframe, x=columns_to_plot[0])
    elif plot_type == "Box Plot":
        fig = px.box(dataframe, x=columns_to_plot[0])
    elif plot_type == "Violin Plot":
        fig = px.violin(dataframe, x=columns_to_plot[0])
    elif plot_type == "Bar Chart":
        fig = px.bar(dataframe, x=columns_to_plot[0], title=f"Count of {columns_to_plot[0]}")
    elif plot_type == "Scatter Plot":
        fig = px.scatter(dataframe, x=columns_to_plot[0], y=columns_to_plot[1])
    elif plot_type == "Line Chart":
        fig = px.line(dataframe, x=columns_to_plot[0], y=columns_to_plot[1])
    elif plot_type == "Scatter Matrix":
        fig = px.scatter_matrix(dataframe, dimensions=columns_to_plot)
    elif plot_type == "Parallel Coordinates":
        fig = px.parallel_coordinates(dataframe, color=columns_to_plot[0] if num_numeric > 0 else None)
    elif plot_type == "Facet Grid":
        # For Facet Grid, we assume the first selected column is categorical and plot against all others
        fig = px.scatter(dataframe, x=columns_to_plot[1], y=columns_to_plot[0] if num_numeric == 1 else columns_to_plot[2], facet_col=columns_to_plot[0] if len(columns_to_plot) > 2 else None, color=columns_to_plot[0])

    st.plotly_chart(fig)


# Function to encode categorical variables for feature set
def encode_features(X):
    categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
    X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)
    return X

# Handle missing values in the dataframe
def handle_missing_values(dataframe):
    st.write("Handle Missing Values")

    if dataframe.isnull().values.any():
        option = st.selectbox("Select how to handle missing values:", ["Drop rows with missing values", "Fill missing values"])
        if option == "Drop rows with missing values":
            dataframe.dropna(inplace=True)
            st.write("Rows with missing values have been dropped.")
        elif option == "Fill missing values":
            num_imputer = SimpleImputer(strategy="mean")
            cat_imputer = SimpleImputer(strategy="most_frequent")

            for col in dataframe.columns:
                if dataframe[col].dtype == np.number:
                    dataframe[col] = num_imputer.fit_transform(dataframe[[col]])
                else:
                    dataframe[col] = cat_imputer.fit_transform(dataframe[[col]])
            st.write("Missing values have been filled.")
    else:
        st.write("No missing values detected.")
    return dataframe

# Modified function to include encoding for features before training models
def select_features_target_and_train_models(dataframe):
    st.write("Machine Learning Task Setup")
    all_columns = dataframe.columns.tolist()
    X_columns = st.multiselect("Select feature columns (X):", all_columns, default=all_columns[:-1])
    y_column = st.selectbox("Select target column (y):", all_columns, index=len(all_columns)-1)

    X = dataframe[X_columns]
    y = dataframe[y_column].astype(np.float64)
    # Encode categorical features
    X_encoded = encode_features(X)

    problem_type = "classification" if y.dtype == 'object' or y.dtype.name == 'category' else "regression"
    if problem_type == "regression":
        st.markdown("<h2 style='text-align: center; color: green;'>This is a Regression Problem</h2>", unsafe_allow_html=True)
        regression_models = st.multiselect("Select regression models to train:", ["Linear Regression", "Ridge", "Lasso", "Decision Tree", "Random Forest"])
        test_ratio = st.slider("Select test split ratio:", 0.1, 0.5, 0.2, 0.05)

        if st.button("Train Models"):
            X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=test_ratio, random_state=42)
            results = train_and_evaluate_models(X_train, X_test, y_train, y_test, regression_models)
            best_model = min(results, key=lambda x: x['MSE'])
            worst_model = max(results, key=lambda x: x['MSE'])

            st.write(f"Best Model: {best_model['name']} - MSE: {best_model['MSE']}, R^2: {best_model['R2']}")
            st.write(f"Worst Model: {worst_model['name']} - MSE: {worst_model['MSE']}, R^2: {worst_model['R2']}")

# Train and evaluate selected regression models
def train_and_evaluate_models(X_train, X_test, y_train, y_test, model_names):
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest": RandomForestRegressor()
    }

    results = []

    for name in model_names:
        model = models[name]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results.append({"name": name, "MSE": mse, "R2": r2})

    return results

# Streamlit UI setup
st.title("Data Analysis Web Application")
uploaded_file = st.file_uploader("Upload your CSV data", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    sample_data = st.selectbox("Or, choose a sample dataset:", ["", "tips", "titanic"])
    if sample_data:
        df = load_sample_data(sample_data)
    else:
        df = None

if df is not None:
    df = handle_missing_values(df)  # Handle missing values right after data loading
    if st.button("Perform Basic EDA"):
        perform_eda(df)
    generate_plots(df)
    select_features_target_and_train_models(df)
