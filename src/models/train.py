# src/models/train.py
import json
import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.models.evaluate import calculate_metrics


def load_yaml(file_path: str) -> dict:
  with open(file_path, "r") as f:
    return yaml.safe_load(f)


def run_training_pipeline():
  config = load_yaml("config/config.yaml")
  params = load_yaml("config/params.yaml")

  mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
  mlflow.set_experiment(config["mlflow"]["experiment_name"])

  # Load preprocessed datasets
  X_train = pd.read_csv(config["data"]["train_features"])
  y_train = pd.read_csv(config["data"]["train_target"]).values.ravel()
  X_test = pd.read_csv(config["data"]["test_features"])
  y_test = pd.read_csv(config["data"]["test_target"]).values.ravel()
  preprocessor = joblib.load(config["data"]["preprocessor_path"])

  # Model catalog using params.yaml
  rf_cfg = params["train"]["models"]["RandomForest"]
  lr_cfg = params["train"]["models"]["LogisticRegression"]
  dt_cfg = params["train"]["models"]["DecisionTree"]

  candidates = {
      "RandomForest": RandomForestClassifier(
          n_estimators=rf_cfg["n_estimators"],
          max_depth=rf_cfg["max_depth"],
          min_samples_split=rf_cfg["min_samples_split"],
          random_state=params["base"]["random_state"],
      ),
      "LogisticRegression": LogisticRegression(
          C=lr_cfg["C"],
          max_iter=lr_cfg["max_iter"],
          random_state=params["base"]["random_state"],
      ),
      "DecisionTree": DecisionTreeClassifier(
          max_depth=dt_cfg["max_depth"],
          random_state=params["base"]["random_state"],
      ),
  }

  best_metric_score = 0.0
  best_model_name = ""
  best_pipeline = None
  best_metrics = {}

  for name, model in candidates.items():
    with mlflow.start_run(run_name=f"DVC_Stage_{name}"):
      # Log hyperparameters
      mlflow.log_params(model.get_params())
      mlflow.log_param("model_type", name)

      # Train & evaluate
      model.fit(X_train, y_train)
      y_pred = model.predict(X_test)
      y_proba = model.predict_proba(X_test)[:, 1]

      metrics = calculate_metrics(y_test, y_pred, y_proba)
      mlflow.log_metrics(metrics)

      # Complete end-to-end pipeline
      full_pipeline = Pipeline(
          steps=[("preprocessor", preprocessor), ("classifier", model)]
      )

      # Log pipeline model to MLflow artifact store
      mlflow.sklearn.log_model(
          sk_model=full_pipeline,
          artifact_path="churn_model",
          name="model_pipeline",
          skops_trusted_types=["numpy.dtype", "sklearn.tree._tree.Tree"],
          registered_model_name=f"churn_{name.lower()}",
      )

      selection_metric = params["evaluation"]["metric_for_selection"]
      if metrics[selection_metric] > best_metric_score:
        best_metric_score = metrics[selection_metric]
        best_model_name = name
        best_pipeline = full_pipeline
        best_metrics = metrics

  # Save winning pipeline artifact for DVC tracking & deployment
  os.makedirs(config["artifacts"]["model_dir"], exist_ok=True)
  joblib.dump(best_pipeline, config["artifacts"]["model_path"])

  # Export metrics JSON for DVC metrics tracking
  os.makedirs(os.path.dirname(config["artifacts"]["metrics_file"]), exist_ok=True)
  with open(config["artifacts"]["metrics_file"], "w") as f:
    json.dump(best_metrics, f, indent=4)

  print(
      f"Finished Training. Winning model: {best_model_name} "
      f"({selection_metric}: {best_metric_score:.4f})"
  )


if __name__ == "__main__":
  run_training_pipeline()