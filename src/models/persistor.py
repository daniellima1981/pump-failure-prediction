import mlflow
import mlflow.sklearn
import joblib
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT_DIR   = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

LABEL_NAMES = {
    0: "Normal",
    1: "Desgaste (Valve Plate)",
    2: "Falha Forçada 1",
    3: "Falha Forçada 2",
}


def export_model_to_joblib(
    model_name: str = "PumpFailurePredictor_XGBoost_Tuned",
    output_filename: str = "xgboost_model.joblib"
) -> Path:

    db_path = (ROOT_DIR / "mlflow.db").as_posix()
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")

    model_uri = f"models:/{model_name}/latest"
    model     = mlflow.sklearn.load_model(model_uri)

    logger.info(f"Model loaded from Registry: {model_uri}")

    output_path = MODELS_DIR / output_filename
    joblib.dump(model, output_path)

    logger.info(f"Model exported to: {output_path}")

    return output_path


def validate_artifacts() -> bool:

    artifacts = {
        "scaler": MODELS_DIR / "scaler.joblib",
        "model":  MODELS_DIR / "xgboost_model.joblib",
    }

    all_ok = True
    logger.info("\nValidating artifacts...")

    for name, path in artifacts.items():

        if not path.exists():
            logger.error(f"  {name}: file not found → {path}")
            all_ok = False
            continue

        try:
            obj = joblib.load(path)
            logger.info(f"  {name}: successfully loaded → {type(obj).__name__}")
        except Exception as e:
            logger.error(f"  {name}: failed to load → {e}")
            all_ok = False
            continue

        if name == "model":
            for method in ["predict", "predict_proba"]:
                if not hasattr(obj, method):
                    logger.error(f"  {name}: method '{method}' not found")
                    all_ok = False
                else:
                    logger.info(f"     → method '{method}' ")

        if name == "scaler":
            if not hasattr(obj, "mean_") or not hasattr(obj, "scale_"):
                logger.error(f"  {name}: scaler not fitted correctly")
                all_ok = False
            else:
                logger.info(f"     → {len(obj.mean_)} features scaled ")

    if all_ok:
        logger.info("\n All artifacts validated — dashboard ready to run!")
    else:
        logger.error("\n Some artifacts failed — fix before running dashboard")

    return all_ok


def save_model_metadata(feature_names: list = None) -> None:
    
    if feature_names is None:
        model = joblib.load(MODELS_DIR / "xgboost_model.joblib")
        if hasattr(model, "feature_names_in_"):
            feature_names = model.feature_names_in_.tolist()
        else:
            raise ValueError(
                "feature_names not found in model — pass explicitly"
            )

    metadata = {
        "model_name":   "XGBoost_Tuned",
        "n_features":   len(feature_names),
        "feature_names": feature_names,        # 37 features do modelo
        "base_feature_names": [                # 7 features brutas do CSV
            "Pressure - leak line",
            "Temperature - leak line",
            "Pressure - output",
            "Temperature - output",
            "Flow - leak line",
            "Flow - output",
            "Temp. diff",
        ],
        "label_names":  LABEL_NAMES,
        "scaler_path":  str(MODELS_DIR / "scaler.joblib"),
        "model_path":   str(MODELS_DIR / "xgboost_model.joblib"),
    }

    metadata_path = MODELS_DIR / "model_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    logger.info(f"\n Metadata saved: {metadata_path}")
    logger.info(f"   Model features ({len(feature_names)}):")
    for feat in feature_names:
        logger.info(f"   → {feat}")


def run_persistence_pipeline(
    model_name: str = "PumpFailurePredictor_XGBoost_Tuned"
) -> None:

    logger.info(" Starting persistence pipeline...\n")

    # 1. Exporta modelo
    export_model_to_joblib(
        model_name=model_name,
        output_filename="xgboost_model.joblib"
    )

    # 2. Valida artefatos
    all_ok = validate_artifacts()

    if not all_ok:
        raise RuntimeError(" Artifacts validation failed — pipeline stopped")

    logger.info("\n Persistence pipeline concluded!")
    logger.info("\n Artifacts ready for dashboard:")
    logger.info(f"  {MODELS_DIR / 'scaler.joblib'}")
    logger.info(f"  {MODELS_DIR / 'xgboost_model.joblib'}")
    logger.info(f"  {MODELS_DIR / 'model_metadata.json'}")


if __name__ == "__main__":
    run_persistence_pipeline()