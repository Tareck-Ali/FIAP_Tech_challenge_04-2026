import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def df_mini_preproc(df: pd.DataFrame) -> pd.DataFrame:
    if "Churn" not in df.columns:
        raise ValueError("Coluna alvo 'Churn' não existe")

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Convert numeric safely
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Manual binary mapping BEFORE encoding
    binary_map = {
        "Yes": 1, "No": 0,
        "No phone service": 0,
        "No internet service": 0,
        "Female": 1, "Male": 0
    }

    cat_cols = ["InternetService"]

    for col in df.select_dtypes(include="object").columns:
        if col not in cat_cols:
            df[col] = df[col].str.strip().replace(binary_map)
    return df


def data_pipeline(X, model) -> Pipeline:

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X.select_dtypes(include="object").columns

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])