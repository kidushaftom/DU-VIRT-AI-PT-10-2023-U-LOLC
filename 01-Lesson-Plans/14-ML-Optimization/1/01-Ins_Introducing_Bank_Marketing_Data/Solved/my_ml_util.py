# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Function to perform iterative imputation on a DataFrame
def Iterative_Imputer(df, target_col):
    """
    Perform iterative imputation on a DataFrame.

    Parameters:
        df (DataFrame): Input DataFrame containing both features and target column.
        target_col (str): Name of the target column to be predicted.

    Returns:
        Tuple: Four DataFrames representing the imputed training and testing data, and the corresponding target variables.
    """

    # Split the DataFrame into features (x) and target variable (y)
    x = df.drop(columns=target_col, axis=1)
    y = df[target_col]

    # Identify numerical and categorical columns
    numerical_cols = x.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = x.select_dtypes(include=['object']).columns

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)

    # Display information about missing values and shape before imputing
    print(f'x_train missing value before imputing {x_train.isna().sum()}')
    print(f'x_train shape before imputing {x_train.shape}')
    print('-' * 100)

    # Impute missing values in numerical columns using iterative imputer
    iterative_imputer = IterativeImputer()
    iterative_imputer.fit(x_train[numerical_cols])
    x_train[numerical_cols] = iterative_imputer.transform(x_train[numerical_cols])
    x_test[numerical_cols] = iterative_imputer.transform(x_test[numerical_cols])

    # For categorical columns, fill missing values with the most frequent value (mode)
    x_train[categorical_cols] = x_train[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])
    x_test[categorical_cols] = x_test[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])

    # Display information about missing values and shape after imputing
    print(f'x_train missing value after imputing {x_train.isna().sum()}')
    print(f'x_train shape after imputing {x_train.shape}')
    print('-' * 100)

    # Return the imputed training and testing data along with target variables
    return x_train, x_test, y_train, y_test
