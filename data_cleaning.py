import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.preprocessing import OrdinalEncoder
from fancyimpute import IterativeImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import random
import pandas as pd

class DatasetCleaning:
    def __init__(self, data):
        self.data = data
    def remove_duplicates(self, data):
        if data.duplicated().sum() > 0:
            # Show amount of duplicate values
            print('Number of duplicate rows:', data.duplicated().sum(), "removed.")
            data = data.drop_duplicates()
        return data  # return the original data if the user's input is not 'yes'

    def remove_missing(self, data):
        # Show amount of missing values
        missing_values = data.isnull().sum()
        print('Number of total missing values:', missing_values.sum())
        print('Missing values in each column:')
        for col in missing_values[missing_values>0].index:
            print(col, ':', missing_values[col])
        data = self.impute_data(data) 
        return data
    def impute_data(self, data):
        # Check for missing values
        if not data.isnull().values.any():
            print("No missing values found. Imputation not required.")
            return data.copy()

        # Analyze missingness pattern
        missing_info = data.isnull().sum()
        missing_ratio = missing_info / len(data)

        # Handle missing values based on scenario
        for col in data.columns:
            # Scenario 1: Missing Completely at Random (MCAR) - Low percentage
            if (missing_ratio[col] <= 0.1) & (data[col].dtype != object):
                # Use mean/median for numerical data, mode for categorical
                if pd.api.types.is_categorical_dtype(data[col]):
                    imp = SimpleImputer(strategy='most_frequent')
                    data[col] = imp.fit_transform(data[[col]])
                    print(f"Imputed column '{col}' using mode (MCAR, low percentage).")
                elif np.issubdtype(data[col].dtype, np.datetime64):
                    # Fill datetime column with the mean
                    data[col].fillna(data[col].mean(), inplace=True)
                    print(f"Imputed column '{col}' using mean value (MCAR, high percentage).")
                elif pd.api.types.is_numeric_dtype(data[col]):
                    imp = SimpleImputer(strategy='mean')
                    data[col] = imp.fit_transform(data[[col]])
                    print(f"Imputed column '{col}' using mean/median/mode (MCAR, low percentage).")


            # Scenario 2: MCAR - High percentage (consider dropping column if not critical)
            elif (missing_ratio[col] > 0.1) & (data[col].dtype != object):
                # Arbitrary value imputation for numerical columns
                if pd.api.types.is_numeric_dtype(data[col]):
                    imp = SimpleImputer(strategy='constant', fill_value=0)
                    data[col] = imp.fit_transform(data[[col]])
                    print(f"Imputed column '{col}' using arbitrary value (MCAR, high percentage).")
                # Constant imputation for categorical columns
                else:
                    imp = SimpleImputer(strategy='constant', fill_value='Missing')
                    data[col] = imp.fit_transform(data[[col]])
                    print(f"Imputed column '{col}' using constant value (MCAR, high percentage).")

            # Scenario 3: Missing Not at Random (MNAR) - Categorical, explore domain knowledge
            elif data[col].dtype == object:
                # Multiple imputation by chained equations (MICE)
                imp = IterativeImputer(max_iter=10, random_state=0)
                enc = OrdinalEncoder()
                data_encoded = enc.fit_transform(data[[col]])
                data_encoded_imputed = imp.fit_transform(data_encoded)
                data.loc[:, col] = enc.inverse_transform(data_encoded_imputed)
                print(f"Imputed column '{col}' using MICE (MNAR, categorical).")

            # Scenario 4: Missing At Random (MAR) - Numerical, consider KNN imputation
            elif (data[col].dtype != object) & (missing_ratio[col] <= 0.1):
                imp = KNNImputer(n_neighbors=5)  # Adjust parameters as needed
                data[col] = imp.fit_transform(data[[col]])
                print(f"Imputed column '{col}' using KNN (MAR, low percentage).")

        return data
 
    def remove_outliers(self, data):
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        outliers_count = {}
        for col in numeric_cols:
            initial_count = data.shape[0]
            z_scores = (data[col] - data[col].mean()) / data[col].std()
            data = data[(np.abs(z_scores) < 3)]
            final_count = data.shape[0]
            outliers_count[col] = initial_count - final_count
            if (outliers_count[col] > 0):
                print(f"Removed {outliers_count[col]} outliers from column '{col}'.")
        return data, outliers_count
    def onehot_encode(self,df_imputed, threshold=5):
        categorical_cols = df_imputed.select_dtypes(include=['object', 'category'])
        cols_to_encode = categorical_cols.columns[categorical_cols.nunique() < threshold]
        if not cols_to_encode.empty:
            # Perform one-hot encoding
            encoder = OneHotEncoder(drop='first', sparse_output=False)
            encoded_data = encoder.fit_transform(df_imputed[cols_to_encode])
            # Get feature names after one-hot encoding
            encoded_columns = encoder.get_feature_names_out(cols_to_encode)
            # Create DataFrame with encoded columns
            encoded_df = pd.DataFrame(encoded_data, columns=encoded_columns, index=df_imputed.index)
            # Drop original columns and concatenate with the encoded DataFrame
            df_imputed = df_imputed.drop(cols_to_encode, axis=1)
            df_fixed = pd.concat([df_imputed, encoded_df], axis=1)
            return df_fixed
        else:
            return df_imputed

    def object_columns(self, df, max_unique_values=15):
        object_columns = {}  # Dictionary to store object columns and their unique values

        # Iterate over DataFrame columns
        for col in df.columns:
            if df[col].dtype == 'object':  # Check if dtype is object
                unique_values = df[col].unique().tolist()
                if len(unique_values) > max_unique_values:
                    # If there are more than max_unique_values, select max_unique_values random values
                    unique_values = random.sample(unique_values, max_unique_values)
                # Store column name and its unique values in the dictionary
                object_columns[col] = unique_values

        return object_columns

    def convert_dtypes(self, df, obj_cols):
        for col, dtype in obj_cols.items():
            try:
                if dtype == 'datetime':
                    df[col] = pd.to_datetime(df[col])
                else:
                    df[col] = df[col].astype(dtype)
            except Exception:
                df[col] = df[col].astype('object')
        return df
    def plot_feature_importance(self, df, target_variable):
        # Exclude non-numeric columns
        numeric_columns = df.select_dtypes(include=['number']).columns

        if target_variable in numeric_columns:  # Check if target_variable is numeric
            X = df[numeric_columns].drop(target_variable, axis=1)
            y = df[target_variable]

            model = RandomForestRegressor()
            model.fit(X, y)

            importance = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
            importance.sort_values(by='Importance', ascending=False, inplace=True)

            plt.figure(figsize=(10, 8))
            sns.barplot(x='Importance', y='Feature', data=importance)
            plt.title(f'Feature Importance for Target Variable: {target_variable}')
            plt.show()
        else:
            print(f"Column '{target_variable}' not found or not numeric in DataFrame")
