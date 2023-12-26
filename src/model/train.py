# Import libraries

import argparse
import glob
import os
import mlflow
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
# from mlflow.models import infer_signature
# from mlflow.utils.environment import _mlflow_conda_env


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--training_data", dest='training_data',
                        type=str)
    parser.add_argument("--reg_rate", dest='reg_rate',
                        type=float, default=0.01)

    # parse args
    args = parser.parse_args()

    # return args
    return args


def get_csvs_df(path):
    if not os.path.exists(path):
        raise RuntimeError(f"Cannot use non-existent path provided: {path}")
    csv_files = glob.glob(f"{path}/*.csv")
    if not csv_files:
        raise RuntimeError(f"No CSV files found in provided data path: {path}")
    return pd.concat((pd.read_csv(f) for f in csv_files), sort=False)


def split_data(df):
    X, y = df[['Pregnancies', 'PlasmaGlucose', 'DiastolicBloodPressure',
               'TricepsThickness', 'SerumInsulin', 'BMI', 'DiabetesPedigree',
               'Age']].values, df['Diabetic'].values
    return train_test_split(X, y, test_size=0.30, random_state=0)


def train_model(reg_rate, X_train, y_train):
    # train model
    mlflow.autolog()
    lg = LogisticRegression(C=1/reg_rate, solver="liblinear")
    lg.fit(X_train, y_train)
#    print("Registering the model via MLFlow")
#    mlflow.sklearn.log_model(
#        sk_model=lg,
#        registered_model_name='log_regression',
#        artifact_path='log_regression',
#    )
    return lg


def get_model_metrics(class_model, X_test, y_test):
    preds = class_model.predict(X_test)
    accuracy = accuracy_score(preds, y_test)
    f1 = f1_score(preds, y_test)
    metrics = {"accuracy": accuracy, "f1": f1}
    return metrics


def main():
    # enable autologging
    # mlflow.sklearn.autolog(log_models=False)
    mlflow.sklearn.autolog()
    args = parse_args()

    # read data
    df = get_csvs_df(args.training_data)

    # split data
    X_train, X_test, y_train, y_test = split_data(df)

    # train model
    lg = train_model(args.reg_rate, X_train, y_train)
    model_metrics = get_model_metrics(lg, X_test, y_test)
    print(model_metrics["accuracy"])
    print(model_metrics["f1"])
    # Signature
#    signature = infer_signature(X_test, y_test)
#
#    # Conda environment
#    custom_env =_mlflow_conda_env(
#        additional_conda_deps=None,
#        #additional_pip_deps=["xgboost==1.5.2"],
#        additional_conda_channels=None,
#    )
#
#    # Sample
#    #input_example = X_train.sample(n=1)
#    input_example = X_train[random.randint(0,len(X_train)-1)]
#
#    # Log the model manually
#    mlflow.sklearn.log_model(lg,
#                             artifact_path="classifier",
#                             conda_env=custom_env,
#                             signature=signature,
#                             input_example=input_example)

    model_name = "logistic_regression_diabetes.pkl"
    joblib.dump(value=lg, filename=model_name)


# run script
if __name__ == "__main__":
    # add space in logs
    print("\n\n")
    print("*" * 60)

    # run main function
    main()

    # add space in logs
    print("*" * 60)
    print("\n\n")
