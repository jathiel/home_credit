import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.pipeline import make_pipeline

import dask.dataframe as dd
from dask_ml.preprocessing import StandardScaler as daskss
from dask_ml.preprocessing import OneHotEncoder as daskohe
from sklearn.compose import ColumnTransformer


# Function to replace NaNs with unique values
def replace_nans_with_unique(df: pd.DataFrame, column: str):
    """
    Parameters:
    - df: pandas.DataFrame
    - column: column name

    Returns:
    - Dataframe with a new value replacing NaNs
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    # Only look for a value above max. This choice is arbitrary.
    max_value = df[column].max()
    if max_value is np.nan:
        max_value = 0  # Assuming positive values, adjust as necessary
    new_value = max_value + 1
    # Replace NaNs with the unique value
    df[column] = df[column].fillna(new_value)
    return df


def vectorize_dataframe(df_main):
    # Dictionary to store the mappings for each column
    mappings = {}

    # Function to handle numerical columns
    def handle_numerical(df, column):
        df = replace_nans_with_unique(df, column)
        _, bins = pd.qcut(df[column], 10, retbins=True, duplicates="drop", labels=False)
        df[column] = pd.cut(df[column], bins=bins, labels=False, include_lowest=True)
        return df, bins

    # Function to handle categorical columns
    def handle_categorical(df, column):
        df = replace_nans_with_unique(df, column)
        mapping = {cat: i for i, cat in enumerate(df[column].unique())}
        df[column] = df[column].map(mapping)
        return df, mapping

    # Function to handle datetime columns
    def handle_datetime(df, column):
        # df = replace_nans_with_unique(df, column)
        df[column] = pd.to_datetime(df[column])
        # Convert datetime to a numerical feature, e.g., UNIX timestamp
        df[column] = (df[column] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
        return df, "timestamp"

    # Iterate over each column in the DataFrame
    for column in df_main.columns:
        if pd.api.types.is_datetime64_any_dtype(df_main[column]):
            df_main, mapping = handle_datetime(df_main, column)
            mappings[column] = mapping
        elif pd.api.types.is_string_dtype(df_main[column]):
            df_main, mapping = handle_categorical(df_main, column)
            mappings[column] = mapping
        elif pd.api.types.is_numeric_dtype(df_main[column]):
            df_main, bins = handle_numerical(df_main, column)
            mappings[column] = bins
        else:
            mappings[column] = None  # For unsupported column types

    return df_main, mappings


def vectorize_dataframe_for_nn(df, chunk_size=10000):
    # Define transformers for different column types
    scaler = StandardScaler()
    onehot = OneHotEncoder(sparse_output=False)

    # Identify numeric and categorical columns
    numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns

    # Create a column transformer to apply the appropriate transformations
    transformer = ColumnTransformer(
        [('num', scaler, numeric_columns),
         ('cat', onehot, categorical_columns)],
        remainder='passthrough'  # Include other types without transformation
    )
    print(transformer)
    # Fit and transform the data
    # transformed_data = transformer.fit_transform(df)
    # feature_names = transformer.get_feature_names_out()  # Get new column names after one-hot encoding

    # for start in range(0, df.shape[0], chunk_size):
    #     end = start + chunk_size
    #     chunk = df.iloc[start:end]
    #     transformed_data = transformer.fit_transform(chunk)
    #     yield transformed_data, transformer
    # Convert the output to a dense DataFrame if necessary (useful if sparse_output=True in OneHotEncoder)
    # if isinstance(transformed_data, np.ndarray):
    #     df_transformed = pd.DataFrame(transformed_data, columns=feature_names)
    # else:
    #     df_transformed = pd.DataFrame(transformed_data.toarray(), columns=feature_names)
    # Initialize an empty DataFrame to collect transformed batches
    df_transformed = pd.DataFrame()

    # Process data in batches
    for start in range(0, len(df), chunk_size):
        end = min(start + chunk_size, len(df))
        batch = df.iloc[start:end]
        transformed_batch = None
        if start == 0:
            # Fit and transform the first batch
            transformed_batch = transformer.fit_transform(batch)
        else:
            # Only transform subsequent batches
            transformed_batch = transformer.transform(batch)

        # Convert transformed data to DataFrame and append to the result
        batch_df = pd.DataFrame(transformed_batch, columns=transformer.get_feature_names_out())
        df_transformed = pd.concat([df_transformed, batch_df], ignore_index=True)

    # Store the preprocessing model for inverse_transform or transforming new data
    return df_transformed, transformer


if __name__ == "__main__":
    # Example usage`
    data = {
        "A": [1, 2, 3, 4, np.nan],
        "B": ["apple", "banana", "apple", "orange", None],
        "C": ["2021-01-01", "2021-02-01", np.nan, "2021-04-01", "2021-05-01"],
        "D": [np.nan, "a115b35", "a115b36", "a115b37", "a115b38"],
    }

    # df = pd.DataFrame(data)
    # vectorized_df, col_mappings = vectorize_dataframe(df)
    # print(f"{vectorized_df}\n\n")
    # print(col_mappings)
    df = pd.DataFrame(data)
    print(df.dtypes.unique())

    df["C"] = pd.to_datetime(
        df["C"]
    )  # Convert dates to datetime format (optional handling)
    vectorized_df, transformer = vectorize_dataframe_for_nn(df)
    # vectorized_df, transformer = vectorize_dataframe(df)
    print(df)
    print(vectorized_df)
    print(df == vectorized_df)
    print(transformer)

# def process_numerical_column(column):
#     """
#     Process a numerical column by filling missing values with the median
#     and then standardizing the column.

#     - A processed pd.Series object.
#     """
#     # Replace missing values with nan
#     imputer = SimpleImputer()
#     column_filled = imputer.fit_transform(column.values.reshape(-1, 1)).flatten()

#     # Standardize the column
#     scaler = StandardScaler()
#     column_standardized = scaler.fit_transform(column_filled.reshape(-1, 1)).flatten()


#     return pd.Series(column_standardized)


# def process_categorical_column(column, placeholder='Missing'):
#     """
#     Process a categorical column by filling missing values with a placeholder,
#     then applying one-hot encoding.
#     """
#     # Fill missing values with a placeholder
#     column_filled = column.fillna(placeholder)

#     # One-hot encode the column
#     encoder = OneHotEncoder(sparse=False)
#     encoded = encoder.fit_transform(column_filled.values.reshape(-1, 1))

#     return pd.DataFrame(encoded, columns=encoder.get_feature_names_out())


# def process_datetime_column(column):
#     """
#     Process a datetime column stored as strings, splitting it into separate
#     columns for year, month, and day.

#     Parameters:
#     - column: pd.Series containing datetime information as strings.

#     Returns:
#     - A pd.DataFrame with separate columns for year, month, and day.
#     """
#     # Convert the column to datetime
#     datetime_col = pd.to_datetime(column, errors='coerce') # 'coerce' turns invalid parsing into NaT (Not a Time)

#     # Extract year, month, and day as separate columns
#     year = datetime_col.dt.year
#     month = datetime_col.dt.month
#     day = datetime_col.dt.day

#     # Assuming 'df' is your DataFrame and 'Date' is the datetime column in string format
#     datetime_processed_df = process_datetime_column(df['Date'])
#     # Combine the extracted columns into a DataFrame

#     # Drop the original 'Date' column from 'df'
#     df = df.drop('Date', axis=1)

#     # Concatenate the processed datetime DataFrame with the original 'df'
#     df = pd.concat([df, datetime_processed_df], axis=1)


#     return pd.DataFrame({'Year': year, 'Month': month, 'Day': day})


# def process_string_column(column):
#     """
#     Process a string column by filling missing values with a placeholder
#     and then encoding the strings to numerical labels.

#     Parameters:
#     - column: pd.Series

#     Returns:
#     - A processed pd.Series object with numerical labels.
#     """
#     # Fill missing values with a placeholder
#     column_filled = column.fillna('Missing')

#     # Encode the column with labels
#     encoder = LabelEncoder()
#     column_encoded = encoder.fit_transform(column_filled)

#     return pd.Series(column_encoded)


# def vectorize_dataframe(df):
#     processed_columns = []
#     for column_name, column_data in df.items():
#         if column_data.dtype == 'float64' or column_data.dtype == 'int64':
#             processed_columns.append(process_numerical_column(column_data))
#         elif column_data.dtype == 'object': #?
#             # Assuming object dtype implies categorical or string data
#             processed_columns.append(process_categorical_column(column_data))
#     return pd.concat(processed_columns, axis=1)
