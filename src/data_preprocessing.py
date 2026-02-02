import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(filepath: str) -> pd.DataFrame:
    """Load the student dataset from the specified filepath."""
    return pd.read_csv(filepath)

def split_features_target(df: pd.DataFrame):
    """
    Separate feature variables (X) and target variable (y).
    Drops G3 to prevent data leakage.
    """
    x = df.drop(columns=["pass", "G3"])
    y = df["pass"]
    return x, y

def identify_feature_types(x: pd.DataFrame):
    """
    Identify categorical and numerical feature columns.
    """
    categorical_features = x.select_dtypes(include="object").columns
    numerical_features = x.select_dtypes(include=["int64", "float64"]).columns
    return categorical_features, numerical_features

def encode_categorical_features(x: pd.DataFrame, categorical_features):
    """
    Apply one-hot encoding to categorical features.
    """
    x_encoded = pd.get_dummies(
        x,
        columns = categorical_features,
        drop_first = True
    )
    return x_encoded

def scale_numerical_features(x: pd.DataFrame, numerical_features):
    """
    Scale numerical features using StandardScaler.
    """
    scalar = StandardScaler()
    x[numerical_features] = scalar.fit_transform(x[numerical_features])
    return x, scalar

def train_test_split_data(x, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets using stratified sampling.
    """
    return train_test_split(
        x,
        y,
        test_size = test_size,
        random_state = random_state,
        stratify = y
    )

def preprocess_data(filepath: str):
    """
    Complete preprocessing pipeline.

    Returns:
    X_train, X_test, y_train, y_test, scaler
    """

    # Load data
    df = load_data(filepath)

    # Split features and target
    X, y = split_features_target(df)

    # Identify feature types
    categorical_features, numerical_features = identify_feature_types(X)

    # Encode categorical variables
    X_encoded = encode_categorical_features(X, categorical_features)

    # Scale numerical features
    X_scaled, scaler = scale_numerical_features(
        X_encoded, numerical_features
    )

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split_data(
        X_scaled, y
    )

    return X_train, X_test, y_train, y_test, scaler