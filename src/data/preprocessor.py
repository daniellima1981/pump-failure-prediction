import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import BorderlineSMOTE
import joblib


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#MODELS_DIR = Path("models")
#PROCESSED_DIR = Path("data/processed")

ROOT_DIR      = Path(__file__).resolve().parent.parent.parent
MODELS_DIR    = ROOT_DIR / "models"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

SCALER_PATH = MODELS_DIR / "scaler.joblib"
INPUT_PATH  = PROCESSED_DIR / "dataset_merged.csv"



#NO FEATURE COLUMNS LIST

COLS_TO_DROP = [
    "Czas",         # timestamp, no feature
    "Czas2",         # timestamp, no feature
    "label_name",   # label
    "stan",         # incomplet subset of label
]

def drop_irrelevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    REMOVE NO FEATURE COLUMNS.
    REMOVE ONLY EXISTING COLUMNS, NO ERROS IF IT DOESN'T EXIST
    """
    cols_present = [c for c in COLS_TO_DROP if c in df.columns]
    df = df.drop(columns=cols_present)

    logger.info(f"Removed Columns: {cols_present}")
    logger.info(f"Shape after removal: {df.shape}")

    return df

#Removing variables with more then 0.992 correlation

REDUNDAT_COLS = [
    "Temperature - suction line"
]

def drop_redundant_columns(df: pd.DataFrame)-> pd.DataFrame:
    """
    REMOVING FEATURES WITH HIGH CORRELATION.
    KEEPS ONLY ONE OF THE REDUNDANT PAIR.
    """
    cols_present = [c for c in REDUNDAT_COLS if c in df.columns]
    df = df.drop(columns=cols_present)

    logger.info(f"Redundat columns removed: {cols_present}")
    logger.info(f"Shape after removal: {df.shape}")

    return df

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle NaN values for numeric features.

    Applying the Median
    """
    features_col = [
        col for col in df.select_dtypes(include="number").columns
        if col != "label"
    ]

    nan_counts = df[features_col].isnull().sum()
    nan_found = nan_counts[nan_counts > 0]

    if nan_counts.empty:
        logger.info("NaN not found.")
        return df
    
    logger.info("\n Nan found:")
    logger.info(nan_found.to_string())

    for col in nan_found.index:
        median = df[col].median()
        df[col] = df[col].fillna(median)
        logger.info(f"  '{col}': {nan_found[col]} NaN → filled with median ({median:.4f})")
    
    logger.info(f"\n NaN treated. Total filled: {nan_found.sum()}")

    return df


def clip_outliers(df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
    """
    TO REDUCE EXTREME OUTLIERS USING Z-SCORE BY NUMERIC COLUMN.
    VALUES BEYOND THE THRESHOLD STANDARD DEVIATION ARE CLIPPED TO THE LIMIT
    """

    feature_cols = [
        col for col in df.select_dtypes(include="number").columns
        if col != "label"
    ]

    clipped_report = {}

    for col in feature_cols:
        mean = df[col].mean()
        std = df[col].std()

        lower = mean - threshold * std
        upper = mean + threshold * std

        n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()

        if n_outliers > 0:
            df[col] = df[col].clip(lower=lower, upper=upper)
            clipped_report[col] = n_outliers

        if clipped_report:
            logger.info(f"\nOutliers clipped (threshold={threshold} std):")
            for col, n in clipped_report.items():
                logger.info(f"  {col}: {n:,} adjusted values")
        else:
            logger.info("No extreme outliers found.")
        
        return df
    
def split_dataset(
        df: pd.DataFrame,
        val_size: float = 0.15,
        test_size: float = 0.15,
        random_state: int = 42
    ) -> tuple:
        
    """
    Stractified split train / validation / test.

        Returns: 
        X_train, X_val, X_test, y_train, y_val, y_test
    """

    X = df.drop(columns=["label"])
    y = df["label"]

    # 1º split - reserve test from the rest
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    #2º - reserve test from the rest
    val_size_adjusted = val_size / (1 - test_size)

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size_adjusted,
        stratify=y_temp,
        random_state=random_state
    )

   
    logger.info(f"\nDataset Split:")
    logger.info(f"  Train:      {len(X_train):,} samples ({len(X_train)/len(df)*100:.1f}%)")
    logger.info(f"  Validation:  {len(X_val):,}  samples ({len(X_val)/len(df)*100:.1f}%)")
    logger.info(f"  Test:        {len(X_test):,} samples ({len(X_test)/len(df)*100:.1f}%)")

    return X_train, X_val, X_test, y_train, y_val, y_test



#NORMALIZATION

def fit_and_apply_scaler(
        df: pd.DataFrame,
        fit: bool = True,
        scaler_path: str = SCALER_PATH
) -> tuple[pd.DataFrame, StandardScaler]:
    """
    APPLIES STANDARDSCALER TO NUMERICAL FEATURES.

    ARGS:
    DF: DATAFRAME WITH FEATURES
    FIT: TRUE = FITS AND TRANSFORMS (TRAIN SET ONLY)
    FALSE = TRANSFORMS ONLY (VALIDATION AND TEST)
    SCALER_PATH: PATH TO SAVE/LOAD THE SCALER

    WHY STANDARDSCALER INSTEAD OF MINMAXSCALER?

    MINMAXSCALER IS SENSITIVE TO OUTLIERS
    STANDARDSCALER (MEAN=0, STD=1) IS MORE ROBUST
    FOR INDUSTRIAL SENSOR DATA

    WHY SAVE THE SCALER?

    IT MUST BE THE SAME FOR TRAINING AND INFERENCE
    RE-FITTING IN PRODUCTION WOULD CAUSE INCONSISTENT VALUES
    """

    feature_cols = [
        col for col in df.select_dtypes(include="number").columns
        if col != "label"
    ]

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if fit:
        scaler = StandardScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
        joblib.dump(scaler, scaler_path)
        logger.info(f"Trained Scaler and save in: {scaler_path}")
    else:
        scaler = joblib.load(scaler_path)
        df[feature_cols] = scaler.transform(df[feature_cols])
        logger.info(f" Scaler loaded and applied from: {scaler_path}")
    
    return df, scaler


def apply_smote(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        random_state: int = 42
) -> tuple[pd.DataFrame, pd.Series]:
    """
    APPLY BORDERLINE-SMOTE ONLY IN TRAINING SET.

    """

    logger.info(f"\nDistribution before SMOTE: ")
    logger.info(y_train.value_counts().sort_index().to_string())

    smote = BorderlineSMOTE(
        sampling_strategy="not majority",
        random_state=random_state,
        k_neighbors = 5
    )

    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    logger.info(f"Distribution after SMOTE:")

    logger.info(pd.Series(y_resampled).value_counts().sort_index().to_string())
    logger.info(f"Amostras adicionadas: {len(X_resampled) - len(X_train):,}")

    return(
        pd.DataFrame(X_resampled, columns=X_train.columns),
        pd.Series(y_resampled)
    )

def run_preprocessing_pipeline(
        input_path: str = INPUT_PATH,
) -> tuple:
    """
    Execute full pre-processing pipeline.
    Return the se for Feature Engineering.

    Order:
    1.Columns Cleaning
    2.Split
    3.Normalization
    4.SMOTE
    """
    logger.info(" Starting pre-processing pipeline...\n")

    df = pd.read_csv(input_path)

    #Cleaning
    df = drop_irrelevant_columns(df)
    df = drop_redundant_columns(df)
    df = handle_missing_values(df)
    df = clip_outliers(df)

    #Split
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df)

    #Normalization
    X_train, scaler = fit_and_apply_scaler(X_train, fit=True)
    X_val,   _      = fit_and_apply_scaler(X_val,   fit=False)
    X_test,  _      = fit_and_apply_scaler(X_test,  fit=False)

    #SMOTE - only for training
    X_train, y_train = apply_smote(X_train, y_train)


    logger.info("\nPipeline Finished!")
    logger.info(f"  X_train: {X_train.shape}")
    logger.info(f"  X_val:   {X_val.shape}")
    logger.info(f"  X_test:  {X_test.shape}")

    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__=="__main__":
    run_preprocessing_pipeline()
