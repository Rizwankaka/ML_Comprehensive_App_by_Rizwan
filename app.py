import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

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
    # User selects columns to plot
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

# Function to encode categorical variables
def encode_categorical_variables(dataframe):
    st.write("Categorical Encoding")
    # Divide columns into categorical and numerical
    categorical_columns = dataframe.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_columns = dataframe.select_dtypes(exclude=['object', 'category']).columns.tolist()

    st.write(f"Categorical Columns: {categorical_columns}")
    st.write(f"Numerical Columns: {numerical_columns}")

    # Ask user for encoding method
    encoding_method = st.selectbox("Select the encoding method:", ["Label Encoder", "One Hot Encoding"])

    if st.button("Encode Categories"):
        if encoding_method == "Label Encoder":
            encoder = LabelEncoder()
            for col in categorical_columns:
                dataframe[col] = encoder.fit_transform(dataframe[col])
        elif encoding_method == "One Hot Encoding":
            dataframe = pd.get_dummies(dataframe, columns=categorical_columns)

        st.write("Data Preview after Encoding:")
        st.write(dataframe.head())

# Function to select features and target, and determine problem type
def select_features_target(dataframe):
    st.write("Machine Learning Task Setup")

    all_columns = dataframe.columns.tolist()
    X_columns = st.multiselect("Select feature columns (X):", all_columns, default=all_columns[:-1])
    y_column = st.selectbox("Select target column (y):", all_columns, index=len(all_columns)-1)

    # Determine if the task is classification or regression
    if dataframe[y_column].dtype == 'object' or dataframe[y_column].dtype.name == 'category':
        st.markdown("<h2 style='text-align: center; color: yellow;'>This is a Classification Problem</h2>", unsafe_allow_html=True)
    else:
        st.markdown("<h2 style='text-align: center; color: green;'>This is a Regression Problem</h2>", unsafe_allow_html=True)
    
    # Ask user to select the ratio for train-test split
    test_ratio = st.slider("Select test split ratio:", 0.1, 0.5, 0.2, 0.05)

    if st.button("Split Data"):
        X = dataframe[X_columns]
        y = dataframe[y_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=42)
        st.write(f"Training set size: {X_train.shape[0]} rows")
        st.write(f"Test set size: {X_test.shape[0]} rows")

# Streamlit UI setup
st.title("Data Analysis Web Application")

# Data upload
uploaded_file = st.file_uploader("Upload your CSV data", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    # Sample data selection
    sample_data = st.selectbox("Or, choose a sample dataset:", ["", "tips", "titanic"])
    if sample_data:
        df = load_sample_data(sample_data)
    else:
        df = None

if df is not None:
    if st.button("Perform Basic EDA"):
        perform_eda(df)
    generate_plots(df)
    encode_categorical_variables(df)
    select_features_target(df)
