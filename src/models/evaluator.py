
import pandas as pd
import numpy as np
import logging
import mlflow
import mlflow.sklearn
from pathlib import Path
from sklearn.metrics import (
    f1_score, roc_auc_score,
    classification_report,
    confusion_matrix,
    RocCurveDisplay
)
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT_DIR   = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = ROOT_DIR / "models"
REPORT_DIR = ROOT_DIR / "reports"

LABEL_NAMES = {
    0: "Normal",
    1: "Desgaste (Valve Plate)",
    2: "Falha Forçada 1",
    3: "Falha Forçada 2",
}



def load_best_model(model_name: str = "PumpFailurePredictor_XGBoost_Tuned"):

    db_path = (ROOT_DIR / "mlflow.db").as_posix()
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")

    # Carrega a versão mais recente do modelo registrado
    model_uri = f"models:/{model_name}/latest"
    model = mlflow.sklearn.load_model(model_uri)

    logger.info(f" Model loaded: {model_uri}")

    return model


def evaluate_on_test(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "XGBoost_Tuned"
) -> dict:

    y_pred       = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    metrics = {
        "test_f1_macro":    f1_score(y_test, y_pred, average="macro"),
        "test_f1_weighted": f1_score(y_test, y_pred, average="weighted"),
        "test_roc_auc":     roc_auc_score(
                                y_test, y_pred_proba,
                                multi_class="ovr"
                            ),
        "test_accuracy":    (y_test == y_pred).mean(),

        # F1 por classe — o mais importante para o negócio
        "test_f1_normal":   f1_score(y_test, y_pred, average=None)[0],
        "test_f1_desgaste": f1_score(y_test, y_pred, average=None)[1],
        "test_f1_falha1":   f1_score(y_test, y_pred, average=None)[2],
        "test_f1_falha2":   f1_score(y_test, y_pred, average=None)[3],
    }

    logger.info(f"\n Test Metrics — {model_name}:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    # Classification report completo
    report = classification_report(
        y_test, y_pred,
        target_names=list(LABEL_NAMES.values())
    )
    logger.info(f"\n Classification Report (TEST):\n{report}")

    return metrics, y_pred, y_pred_proba



def plot_roc_curves(
    y_test: pd.Series,
    y_pred_proba: np.ndarray,
    model_name: str
) -> str:

    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_curve, auc

    # Binariza as classes para calcular ROC por classe
    y_bin = label_binarize(y_test, classes=[0, 1, 2, 3])

    fig, ax = plt.subplots(figsize=(10, 7))

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]

    for i, (label, color) in enumerate(zip(LABEL_NAMES.values(), colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_pred_proba[:, i])
        auc_score   = auc(fpr, tpr)

        ax.plot(
            fpr, tpr,
            color=color,
            linewidth=2,
            label=f"{label} (AUC = {auc_score:.4f})"
        )

    # Linha de referência — AUC=0.5 = chute aleatório
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC = 0.50)")

    ax.set_title(f"ROC Curve per Class — {model_name}")
    ax.set_xlabel("False Positive Rate (Alarmes Falsos)")
    ax.set_ylabel("True Positive Rate (Falhas Detectadas)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = str(REPORT_DIR / f"roc_curves_{model_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()

    logger.info(f" ROC curves recorded at: {path}")

    return path


def plot_test_confusion_matrix(
    y_test: pd.Series,
    y_pred: np.ndarray,
    model_name: str
) -> str:

    cm = confusion_matrix(y_test, y_pred, normalize="true")
    labels = list(LABEL_NAMES.values())

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2%",        # formato percentual
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax
    )
    ax.set_title(f"Normalized Confusion Matrix — {model_name}\n(test set)")
    ax.set_ylabel("Real")
    ax.set_xlabel("Predito")
    plt.tight_layout()

    path = str(REPORT_DIR / f"confusion_matrix_test_{model_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()

    return path


def run_evaluation(
    X_test:     pd.DataFrame,
    y_test:     pd.Series,
    model_name: str = "XGBoost_Tuned"
) -> None:
    """
    Run the full evaluation and log it into MLflow.
    """
    logger.info(" Starting final evaluationg at test set...\n")

    # Carrega modelo do Registry
    model = load_best_model(f"PumpFailurePredictor_{model_name}")

    # Avaliação
    metrics, y_pred, y_pred_proba = evaluate_on_test(
        model, X_test, y_test, model_name
    )

    # Gráficos
    roc_path = plot_roc_curves(y_test, y_pred_proba, model_name)
    cm_path  = plot_test_confusion_matrix(y_test, y_pred, model_name)

    # Loga métricas e artefatos no run existente do MLflow
    db_path = (ROOT_DIR / "mlflow.db").as_posix()
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    mlflow.set_experiment("pump-failure-prediction")

    with mlflow.start_run(run_name=f"{model_name}_Test_Evaluation"):
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(roc_path)
        mlflow.log_artifact(cm_path)

    logger.info("\n Avaliation finished and recorded into MLflow!")
    logger.info(" Access http://127.0.0.1:5000 to visulize")


    if __name__ == "__main__":
        run_evaluation(X_test, y_test)
