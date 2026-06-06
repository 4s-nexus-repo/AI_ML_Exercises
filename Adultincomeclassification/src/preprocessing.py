from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def split_features_and_target(df):
    """
    Convert the income target column into binary values
    and split the dataset into features X and target y.
    """

    df = df.copy()

    df["income"] = df["income"].map({
        "<=50K": 0,
        ">50K": 1
    })

    X = df.drop("income", axis=1)
    y = df["income"]

    return X, y


def identify_column_types(X):
    """
    Identify numerical and categorical columns.
    """

    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    return numerical_cols, categorical_cols


def split_train_test(X, y):
    """
    Split data into training and test sets.
    Stratify is used to keep the same target class balance.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def create_preprocessor(numerical_cols, categorical_cols):
    """
    Create preprocessing pipeline for numerical and categorical columns.
    """

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    try:
        one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", one_hot_encoder)
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, numerical_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ])

    return preprocessor


def preprocess_data(X_train, X_test, numerical_cols, categorical_cols):
    """
    Fit the preprocessor on training data and transform both train and test data.
    """

    preprocessor = create_preprocessor(numerical_cols, categorical_cols)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    return X_train_processed, X_test_processed, preprocessor