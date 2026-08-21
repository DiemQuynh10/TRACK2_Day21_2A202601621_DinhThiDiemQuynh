import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

EVAL_THRESHOLD = 0.70

DRIFT_WARNING_THRESHOLD = 0.10


def _build_model(params: dict):
    """Khoi tao model theo model_type trong params (mac dinh: random_forest)."""
    model_type = params.get("model_type", "random_forest")

    if model_type == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", 3),
            random_state=42,
        )
    if model_type == "logistic_regression":
        return LogisticRegression(
            max_iter=1000,
            random_state=42,
        )
    return RandomForestClassifier(
        n_estimators=params.get("n_estimators", 100),
        max_depth=params.get("max_depth", None),
        min_samples_split=params.get("min_samples_split", 2),
        random_state=42,
    )


def _check_label_distribution(y_train) -> dict:
    """Tinh ti le tung lop trong tap huan luyen, canh bao neu lop nao qua it (< 10%)."""
    counts = y_train.value_counts(normalize=True).sort_index()
    distribution = {str(label): float(ratio) for label, ratio in counts.items()}

    for label, ratio in distribution.items():
        if ratio < DRIFT_WARNING_THRESHOLD:
            print(
                f"CANH BAO DATA DRIFT: lop '{label}' chi chiem {ratio:.2%} "
                f"tap huan luyen (< {DRIFT_WARNING_THRESHOLD:.0%})."
            )

    return distribution


def train(
    params: dict,
    data_path: str = "data/train_phase1.csv",
    eval_path: str = "data/eval.csv",
) -> float:
    """
    Huan luyen mo hinh va ghi nhan ket qua vao MLflow.

    Tham so:
        params     : dict chua cac sieu tham so cho model.
        data_path  : duong dan den file du lieu huan luyen.
        eval_path  : duong dan den file du lieu danh gia.

    Tra ve:
        accuracy (float): do chinh xac tren tap danh gia.
    """

    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    label_distribution = _check_label_distribution(y_train)

    with mlflow.start_run():
        mlflow.log_params(params)

        model = _build_model(params)
        model.fit(X_train, y_train)

        preds = model.predict(X_eval)
        acc = accuracy_score(y_eval, preds)
        f1 = f1_score(y_eval, preds, average="weighted")

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, "model")

        print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")

        os.makedirs("outputs", exist_ok=True)
        with open("outputs/metrics.json", "w") as f:
            json.dump(
                {
                    "accuracy": acc,
                    "f1_score": f1,
                    "label_distribution": label_distribution,
                },
                f,
            )

        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")

    return acc


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
