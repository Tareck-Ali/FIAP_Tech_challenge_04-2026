from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib

from data_preproc import data_pipeline, df_mini_preproc
import settings as sett

df = pd.read_csv(sett.RAW_DATA / "WA_Fn-UseC_-Telco-Customer-Churn.csv")
target = "Churn"


df = df_mini_preproc(df)

df.to_csv(sett.PROCESSED_DATA / "processed.csv")

X = df.drop(columns = [target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=sett.rng
)

logreg_pipeline = data_pipeline(X,LogisticRegression(max_iter=10000))
dummy_pipeline = data_pipeline(X,DummyClassifier())

logreg_pipeline.fit(X_train, y_train)
dummy_pipeline.fit(X_train, y_train)

joblib.dump(logreg_pipeline, sett.MODELS / "logreg_baseline.pkl")
joblib.dump(logreg_pipeline, sett.MODELS / "dummy_baseline.pkl")

mlflow.set_tracking_uri(f"file:{sett.PROCESSED_DATA / 'mlruns'}")
mlflow.set_experiment("telco-churn-classification")

with mlflow.start_run(run_name="logreg_vs_dummy"):

    # ---- Dataset version ----
    dataset_path = sett.PROCESSED_DATA / "processed.csv"
    dataset_version = pd.util.hash_pandas_object(df).sum()

    mlflow.log_param("dataset_path", str(dataset_path))
    mlflow.log_param("dataset_version", dataset_version)

    # ---- Model params ----
    logreg_pipeline.fit(X_train, y_train)
    dummy_pipeline.fit(X_train, y_train)

    # Log Logistic Regression hyperparams
    logreg_model = logreg_pipeline.named_steps["model"]
    mlflow.log_param("logreg_max_iter", logreg_model.max_iter)
    mlflow.log_param("logreg_model", "LogisticRegression")

    # Dummy params
    mlflow.log_param("dummy_model", "DummyClassifier")

    # ---- Metrics ----
    dummy_acc = accuracy_score(y_test, dummy_pipeline.predict(X_test))
    logreg_acc = accuracy_score(y_test, logreg_pipeline.predict(X_test))

    mlflow.log_metric("dummy_accuracy", dummy_acc)
    mlflow.log_metric("logreg_accuracy", logreg_acc)

    # ---- Save models ----
    mlflow.sklearn.log_model(logreg_pipeline, "logreg_model")
    mlflow.sklearn.log_model(dummy_pipeline, "dummy_model")