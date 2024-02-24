import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from pydantic_settings import BaseSettings
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, roc_auc_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVR, SVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor, AdaBoostClassifier
from sklearn.impute import SimpleImputer
import math

# Function to load sample data
def load_sample_data(dataset_name):
    if dataset_name == 'tips':
        return sns.load_dataset('tips')
    elif dataset_name == 'titanic':
        return sns.load_dataset('titanic')
    elif dataset_name == 'iris':
        # This will load the Iris dataset
        return sns.load_dataset('iris')

# Basic EDA function
# def perform_eda(dataframe):
    # st.write("Basic EDA Results")
    # # Display summary statistics
    # st.write("Summary Statistics:")
    # st.write(dataframe.describe())
    # # Display distribution of numerical data
    # st.write("Data Distributions:")
    # numerical_columns = dataframe.select_dtypes(include=['float64', 'int64']).columns
    # for column in numerical_columns:
    #     st.write(f"Distribution for {column}:")
    #     fig, ax = plt.subplots()
    #     sns.histplot(dataframe[column], kde=True, ax=ax)
    #     st.pyplot(fig)
    # # Display correlation matrix for numeric columns only
    # st.write("Correlation Matrix:")
    # numeric_df = dataframe.select_dtypes(include=['float64', 'int64'])
    # if not numeric_df.empty:
    #     fig, ax = plt.subplots()
    #     sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    #     st.pyplot(fig)
    # else:
    #     st.write("No numeric columns available for correlation matrix.")
def perform_eda(dataframe):
    st.write("Basic EDA Results with Pandas Profiling")
    
    # Generate the Pandas Profiling report
    profile = ProfileReport(dataframe, explorative=True, minimal=True)
    
    # Display the report in the Streamlit app
    st_profile_report(profile)
# Function to dynamically generate plots based on user input
def generate_plots(dataframe, columns_to_plot):
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
        generate_individual_plot(dataframe, columns_to_plot, plot_type, num_numeric, num_categorical)

def generate_individual_plot(dataframe, columns_to_plot, plot_type, num_numeric, num_categorical):
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
        fig = px.scatter(dataframe, x=columns_to_plot[1], y=columns_to_plot[0] if num_numeric == 1 else columns_to_plot[2], facet_col=columns_to_plot[0] if len(columns_to_plot) > 2 else None, color=columns_to_plot[0])

    st.plotly_chart(fig)

# Function to encode categorical variables for feature set
def encode_features(X):
    categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
    X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)
    return X

# Function to visualize encoding of categorical variables
def visualize_encoding(dataframe):
    st.sidebar.write("## Encoding Visualization")

    # Select categorical columns to encode
    categorical_columns = dataframe.select_dtypes(include=['object', 'category']).columns.tolist()
    if not categorical_columns:
        st.sidebar.write("No categorical columns to encode.")
        return

    columns_to_encode = st.sidebar.multiselect("Select categorical columns to encode:", categorical_columns, key='encode_cols')
    encoding_method = st.sidebar.selectbox("Select encoding method:", ["get_dummies", "Label Encoding", "One-Hot Encoding"], key='encode_method')

    if columns_to_encode:
        if encoding_method == "get_dummies":
            # Perform one-hot encoding using get_dummies
            encoded_df = pd.get_dummies(dataframe, columns=columns_to_encode)
        elif encoding_method == "Label Encoding":
            # Perform label encoding
            le = LabelEncoder()
            encoded_df = dataframe.copy()
            for col in columns_to_encode:
                encoded_df[col] = le.fit_transform(encoded_df[col])
        elif encoding_method == "One-Hot Encoding":
            # Perform one-hot encoding using OneHotEncoder from sklearn with updated parameter
            ohe = OneHotEncoder(sparse_output=False)
            encoded_values = ohe.fit_transform(dataframe[columns_to_encode])
            # Create a DataFrame with the encoded variables
            encoded_vars = pd.DataFrame(encoded_values, columns=np.concatenate(ohe.categories_))
            encoded_df = dataframe.join(encoded_vars).drop(columns=columns_to_encode)
            # Reset index to align rows
            encoded_df.reset_index(drop=True, inplace=True)

        # Display original and encoded data side by side for comparison
        col1, col2 = st.columns(2)
        with col1:
            st.write("Original Data")
            st.dataframe(dataframe[columns_to_encode].head())
        with col2:
            st.write("Encoded Data")
            if encoding_method == "get_dummies":
                # Display only the newly created dummy columns
                dummy_columns = [col for col in encoded_df if col.startswith(tuple(columns_to_encode))]
                st.dataframe(encoded_df[dummy_columns].head())
            else:
                # For Label Encoding and One-Hot Encoding, display the modified dataframe
                if encoding_method == "Label Encoding":
                    st.dataframe(encoded_df[columns_to_encode].head())
                elif encoding_method == "One-Hot Encoding":
                    st.dataframe(encoded_df[list(encoded_vars.columns)].head())


# Handle missing values in the dataframe
def handle_missing_values(dataframe, option):
    st.write("Handle Missing Values")

    if dataframe.isnull().values.any():
        if option == "Drop rows with missing values":
            dataframe.dropna(inplace=True)
            st.sidebar.write("Rows with missing values have been dropped.")
        elif option == "Fill missing values":
            # Create imputers
            num_imputer = SimpleImputer(strategy="mean")
            cat_imputer = SimpleImputer(strategy="most_frequent")

            for col in dataframe.columns:
                if dataframe[col].dtype == "number":  # Numeric columns
                    dataframe[col] = num_imputer.fit_transform(dataframe[[col]]).ravel()
                elif dataframe[col].dtype == "bool":  # Boolean columns, treated as categorical
                    # Temporarily convert boolean to object for imputation
                    dataframe[col] = dataframe[col].astype(object)
                    dataframe[col] = cat_imputer.fit_transform(dataframe[[col]]).ravel()
                    # Optionally convert back to boolean
                    dataframe[col] = dataframe[col].astype(bool)
                else:  # Categorical/object columns
                    dataframe[col] = cat_imputer.fit_transform(dataframe[[col]]).ravel()
            st.sidebar.write("Missing values have been filled.")
    else:
        st.sidebar.write("No missing values detected.")
    return dataframe

# Modified function to include encoding for features before training models
def select_features_target_and_train_models(dataframe):
    st.write("Machine Learning Task Setup")
    all_columns = dataframe.columns.tolist()
    X_columns = st.multiselect("Select feature columns (X):", all_columns, default=all_columns[:-1])
    y_column = st.selectbox("Select target column (y):", all_columns, index=len(all_columns)-1)

    X = dataframe[X_columns]
    y = dataframe[y_column]
    # Encode categorical features
    X_encoded = encode_features(X)

    problem_type = "classification" if y.dtype == 'object' or y.dtype.name == 'category' else "regression"
    if problem_type == "regression":
        st.markdown("This is a Regression Problem", unsafe_allow_html=True)
        regression_models = st.multiselect("Select regression models to train:", ["Linear Regression", "Ridge", "Lasso", "Decision Tree", "Random Forest", "Support Vector Machines", "K-Nearest Neighbors","Gradient Boosting", "AdaBoost"])
        test_ratio = st.slider("Select test split ratio:", 0.1, 0.5, 0.2, 0.05)

        if st.button("Train Regression Models"):
            X_train, X_test, y_train, y_test = train_test_split(X_encoded, y.astype(float), test_size=test_ratio, random_state=42)
            results_df = train_and_evaluate_regression_models(X_train, X_test, y_train, y_test, regression_models)
        
            # Display the regression model evaluation results in tabular form
            st.write("## Regression Model Evaluation Results")
            st.dataframe(results_df, width=700, height=400)

            # Optionally, highlight the best model based on a specific metric, such as MSE
            # This part is optional and can be adjusted based on the metric of interest
            # Find the best model based on MSE
            best_model_idx = results_df['MSE'].idxmin()
            st.write(f"**Best Model Based on MSE:** {results_df.iloc[best_model_idx]['Model']} - MSE: {results_df.iloc[best_model_idx]['MSE']:.4f}, RMSE: {results_df.iloc[best_model_idx]['RMSE']:.4f}, MAE: {results_df.iloc[best_model_idx]['MAE']:.4f}, R2: {results_df.iloc[best_model_idx]['R2']:.4f}, MAPE: {results_df.iloc[best_model_idx]['MAPE']:.4f}")

            # Find the worst model based on MSE
            worst_model_idx = results_df['MSE'].idxmax()
            st.write(f"**Worst Model Based on MSE:** {results_df.iloc[worst_model_idx]['Model']} - MSE: {results_df.iloc[worst_model_idx]['MSE']:.4f}, RMSE: {results_df.iloc[worst_model_idx]['RMSE']:.4f}, MAE: {results_df.iloc[worst_model_idx]['MAE']:.4f}, R2: {results_df.iloc[worst_model_idx]['R2']:.4f}, MAPE: {results_df.iloc[worst_model_idx]['MAPE']:.4f}")

    elif problem_type == "classification":
        st.markdown("This is a Classification Problem", unsafe_allow_html=True)
        classification_models = st.multiselect("Select classification models to train:", ["Logistic Regression", "Decision Tree Classifier", "Random Forest Classifier", "Support Vector Machines", "K-Nearest Neighbors", "Gradient Boosting", "AdaBoost", "Gaussian Naive Bayes", "Multinomial Naive Bayes", "Bernoulli Naive Bayes", "Stochastic Gradient Descent" ])
        test_ratio = st.slider("Select test split ratio for classification:", 0.1, 0.5, 0.2, 0.05)

        if st.button("Train Classification Models"):
            X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=test_ratio, random_state=42)
            classification_results_df = train_and_evaluate_classification_models(X_train, X_test, y_train, y_test, classification_models)
            # Display the classification model evaluation results
            st.write("## Classification Model Evaluation Results")
            st.dataframe(classification_results_df, width=700, height=400)
            # Find the best and worst classifiers based on Accuracy
            best_classifier = classification_results_df.loc[classification_results_df['Accuracy'].idxmax()]
            worst_classifier = classification_results_df.loc[classification_results_df['Accuracy'].idxmin()]

            st.write(f"**Best Classifier:** {best_classifier['Model']} - Accuracy: {best_classifier['Accuracy']:.4f}, Precision: {best_classifier['Precision']:.4f}, Recall: {best_classifier['Recall']:.4f}, F1 Score: {best_classifier['F1 Score']:.4f}, AUC: {best_classifier['AUC'] if 'AUC' in best_classifier else 'N/A'}")
            st.write(f"**Worst Classifier:** {worst_classifier['Model']} - Accuracy: {worst_classifier['Accuracy']:.4f}, Precision: {worst_classifier['Precision']:.4f}, Recall: {worst_classifier['Recall']:.4f}, F1 Score: {worst_classifier['F1 Score']:.4f}, AUC: {worst_classifier['AUC'] if 'AUC' in worst_classifier else 'N/A'}")

# Train and evaluate selected regression models
def train_and_evaluate_regression_models(X_train, X_test, y_train, y_test, model_names):
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest": RandomForestRegressor(),
        "Support Vector Machines": SVR(),
        "K-Nearest Neighbors": KNeighborsRegressor(),
        "Gradient Boosting": GradientBoostingRegressor(),
        "AdaBoost": AdaBoostRegressor(),
    }
    
    results = []

    for name in model_names:
        model = models[name]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = math.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        
        results.append({
            "Model": name,
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
            "MAPE": mape
        })

    return pd.DataFrame(results)

# Train and evaluate selected classification models
def train_and_evaluate_classification_models(X_train, X_test, y_train, y_test, model_names):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree Classifier": DecisionTreeClassifier(),
        "Random Forest Classifier": RandomForestClassifier(),
        "Support Vector Machines": SVC(probability=True),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "AdaBoost": AdaBoostClassifier(),
        "Gaussian Naive Bayes": GaussianNB(),
        "Multinomial Naive Bayes": MultinomialNB(),
        "Bernoulli Naive Bayes": BernoulliNB(),
        "Stochastic Gradient Descent": SGDClassifier(),
    }
    
    results = []

    for name in model_names:
        model = models[name]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None  # For multi-class AUC

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted')
        auc = "N/A"  # Default to N/A if AUC can't be computed
        
        if y_prob is not None:
            # Handle multi-class scenario for AUC
            if len(np.unique(y_test)) == 2:  # Binary classification
                auc = roc_auc_score(y_test, y_prob[:, 1])
            else:  # Multi-class classification
                auc = roc_auc_score(y_test, y_prob, multi_class="ovo", average="weighted")

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "AUC": auc if auc != "N/A" else "N/A"  # Keep as N/A if AUC couldn't be computed
        })

    return pd.DataFrame(results)


# Streamlit UI setup
st.title('''📊Data Analysis Web Application🚀
**Credit:** App built in `Python` + `Streamlit` by [Rizwan](https://www.linkedin.com/in/rizwan-rizwan-1351a650)''')

uploaded_file = st.sidebar.file_uploader("Upload your CSV data", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    sample_data = st.sidebar.selectbox("Choose a sample dataset:", ["", "tips", "titanic", "iris"])
    if sample_data:
        df = load_sample_data(sample_data)
    else:
        df = None

if df is not None:
    # Handling missing values option in sidebar
    missing_value_handling_option = st.sidebar.selectbox(
        "Handle Missing Values",
        ["", "Drop rows with missing values", "Fill missing values"]
    )
    if missing_value_handling_option != "":
        df = handle_missing_values(df, missing_value_handling_option)
    
    # Perform EDA button in sidebar
    if st.sidebar.button("Perform Basic EDA"):
        perform_eda(df)
    
    # Optional Encoding Visualization in sidebar
    if st.sidebar.checkbox("Show Encoding Visualization"):
        visualize_encoding(df)  # Adjusted to be called when checkbox is checked

    # Selecting data columns to plot in sidebar
    columns_to_plot = st.sidebar.multiselect("Select columns to plot:", df.columns, key='plot')
    if columns_to_plot:
        generate_plots(df, columns_to_plot)
    
    # Machine Learning Task Setup in sidebar
    if st.sidebar.checkbox("Setup Machine Learning Task"):
        select_features_target_and_train_models(df)
else:
    st.write("Please upload a dataset or select a sample dataset to begin.")
# Add your contact details at the bottom of the sidebar
contact_info = """
#### Contact Information

**Author:** Rizwan

**GitHub:** [![GitHub](https://img.shields.io/badge/GitHub-Profile-blue?style=for-the-badge&logo=github)](https://github.com/Rizwankaka)

**LinkedIn:** [![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/rizwan-rizwan-1351a650/)

**Twitter:** [![Twitter](https://img.shields.io/badge/Twitter-Profile-blue?style=for-the-badge&logo=twitter)](https://twitter.com/RizwanRizwan_)

**Facebook:** [![Facebook](https://img.shields.io/badge/Facebook-Profile-blue?style=for-the-badge&logo=facebook)](https://www.facebook.com/RIZWANNAZEEER)

**Gmail:** [![Gmail](https://img.shields.io/badge/Gmail-Contact%20Me-red?style=for-the-badge&logo=gmail)](mailto:riwan.rewala@gmail.com)
"""

st.sidebar.markdown(contact_info, unsafe_allow_html=True)

st.markdown("""
    <style>
        .stApp {
        background: url("https://images.unsplash.com/photo-1472289065668-ce650ac443d2?q=80&w=2069&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
        background-size: cover;
        }
    </style>""", unsafe_allow_html=True)