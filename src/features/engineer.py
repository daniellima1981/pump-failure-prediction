import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sklearn.feature_selection import VarianceThreshold

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


ROOT_DIR     = Path(__file__).resolve().parent.parent.parent
FEATURES_DIR = ROOT_DIR / "data" / "features"

#FEATURES_DIR = Path("data/features")

# AVAILABLE FEATURES AFTER PREPROCESSOR

BASE_FEATURES = [
    "Pressure - leak line",
    "Temperature - leak line",
    "Pressure - output",
    "Temperature - output",
    "Flow - leak line",
    "Flow - output",
    "Temp. diff",
]

def create_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    
    # Ratio 1 - Flow efficience
    # How many from the total flow is leaking
    # Increase progressively with valve plate wear
    df["ratio_flow_leak_output"] = (
        df["Flow - leak line"] / (df["Flow - output"] + 1e-8)
        ) #avoid division by zero

    # Ratio 2 - Normalized pressure gradient
    # Pressure drop relative to outlet pressure
    # Instability here indicates problems with the valve.
    df["ratio_pressure_leak_output"] = (
        df["Pressure - leak line"] / (df["Pressure - output"] + 1e-8)
    )

    # Ratio 3 - Normalized temperature delta
    # Heating relative to the outlet temperature
    # Excessive friction generates heat → indicates mechanical wear
    df["ratio_temp_diff_output"] = (
        df["Temp. diff"] /
        (df["Temperature - output"] + 1e-8)
    )

    logger.info(f"Created ratio features: 3")

    return df

def create_rolling_features(
        df: pd.DataFrame,
        windows: list[int] = [5, 10, 30]
) -> pd.DataFrame:
    """
    5s → fast variations (abrupt failure)
    10s → medium variations (state transition)
    30s → tendency towards degradation (progressive wear)

    Args:
    windows: window sizes in number of samples
    (dataset has a resolution of 1 sample/second)
    """

    
    ROLLING_FEATURES = [
        "Pressure - leak line",
        "Temperature - leak line",
        "Flow - leak line",
        "Flow - output",
    ]

    n_created = 0

    for feature in ROLLING_FEATURES:
        for window in windows:
            prefix = feature.lower().replace(" ", "_").replace("-", "")

            #Mean to capture the central tendance
            col_mean = f"{prefix}_mean_{window}s"
            df[col_mean] = (
                df[feature]
                .rolling(window=window, min_periods=1)
                .mean()
            )

            #Std - capture instability/variance
            #min_periods=1 to avoid NaN
            col_std = f"{prefix}_std_{window}s"
            df[col_std] = (
                df[feature]
                .rolling(window=window, min_periods=1)
                .std()
                .fillna(0) # std 1 element = NaN -> replace by 0 
            )

            n_created += 2


    logger.info(f" Rolling features created: {n_created} "
                f"({len(ROLLING_FEATURES)} sensors × "
                f"{len(windows)} window × 2 statistics)")

    return df

def create_delta_features(df: pd.DataFrame) -> pd.DataFrame:

    DELTA_FEATURES = [
        "Pressure - output",
        "Pressure - leak line",
        "Flow - output",
    ]

    for feature in DELTA_FEATURES:
        col_name = feature.lower().replace(" ", "_").replace("-", "")
        df[f"{col_name}_delta"] = df[feature].diff().fillna(0)


    logger.info(f"Delta features created: {len(DELTA_FEATURES)}")

    return df


def remove_low_variance_features(
    df: pd.DataFrame,
    threshold: float = 0.01
) -> pd.DataFrame:
    """
    Remove low variance features.

    threshold=0.01 → remove features with variance < 1%
    """
    feature_cols = [
        col for col in df.select_dtypes(include="number").columns
        if col != "label"
    ]

    selector = VarianceThreshold(threshold=threshold)
    selector.fit(df[feature_cols])

    removed = [
        col for col, support in
        zip(feature_cols, selector.get_support())
        if not support
    ]

    if removed:
        df = df.drop(columns=removed)
        logger.warning(f" Removed Features due to low variance: {removed}")
    else:
        logger.info(" No feature removed due to low variance")

    return df

def run_feature_engineering(
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    RUN THE FEATURE ENGINEERING PIPELINE
    """

    logger.info(" Starting Feature Engineering...\n")

    #Apply transformation over the 3 sets
    for name, dataset in [("train", X_train), ("val", X_val), ("test", X_test)]:
        logger.info(f"--- Processing {name} ---")
        dataset = create_ratio_features(dataset)
        dataset = create_rolling_features(dataset)
        dataset = create_delta_features(dataset)

        if name == "train":
            X_train = dataset
        elif name == "val":
            X_val = dataset
        else:
            X_test = dataset
    
    # Variance Threshold
    feature_cols = [col for col in X_train.columns if col != "label"]
    selector = VarianceThreshold(threshold=0.01)
    selector.fit(X_train[feature_cols])

    
    cols_to_keep = [
        col for col, support in
        zip(feature_cols, selector.get_support())
        if support
    ]

    X_train = X_train[cols_to_keep]
    X_val   = X_val[cols_to_keep]
    X_test  = X_test[cols_to_keep]

    logger.info(f"\n Feature Engineering Finished !")
    logger.info(f"  Original Features: {len(BASE_FEATURES)}")
    logger.info(f"     Final Features:    {len(cols_to_keep)}")
    logger.info(f"   Created Features:   {len(cols_to_keep) - len(BASE_FEATURES)}")
    logger.info(f"\n  Full List:\n  {cols_to_keep}")

    return X_train, X_val, X_test  
