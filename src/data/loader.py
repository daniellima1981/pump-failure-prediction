import pandas as pd
from pathlib import Path
import logging


# LOG BASIC CONFIGURATION USEFUL TO CONTROLE THE LEVEL (DEBUG, INFO, WARNING)
# AND REDIRECT TO A FILE WHEN REQUIRED"


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#PATHLIB-BETTER THAN STRINGS PATH THE CODE CAN BE USED IN WINDOWS, LINUX OR MAC

BASE_DIR = Path(__file__).resolve().parents[2]          ############################################################
                                                        ## WHEN EXECUTING AS A MODELO NEED TO CHANGE THIS EXCERPT ##
RAW_DATA_DIR = BASE_DIR / "data" / "raw"                ## TO RIGHT RESOLV THE  PATH                              ##
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"    ############################################################


# FILE DICTIONARY

# FILE -> LABEL

FILE_LABEL_MAP = {
    "dane_OT.csv":  0, # NORMAL OPERATING CONDITIONS
    "dane_UT1.csv": 1, # VALVE PLATE WEAR
    "dane_UT2.csv": 2, # SIMULATED FAILURE 1
    "dane_UT3.csv": 3, # SIMULATED FAILURE 2
}

#REVERSE NUMBER -> DESCRIPTION

LABEL_NAME_MAP = {
    0: "NORMAL",
    1: "VALVE PLATE WEAR",
    2: "SIMULATED FAILURE 1",
    3: "SIMULATED FAILURE 2",
}

# SCHEMA ENFORCEMENT

COLUMN_SCHEMA = {
    "Czas": "str",
    "Czas2": "str",
    "Pressure - leak line":     "float64",
    "Temperature - leak line":  "float64",
    "Pressure - output":        "float64",
    "Temperature - suction line": "float64",
    "Temperature - output":     "float64",
    "Flow - leak line":         "float64",
    "Flow - output":            "float64",
    "Temp. diff":               "float64",
    "stan":                     "int64",
}

def enforce_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    ENFORCE THE DATA TYPE AS DEFINED IN SCHEMA.
    RECORD WARNINGS FOR UNEXPECTED COLUMNS OF CONVERSION FAILURE.
    """
    for col, dtype in COLUMN_SCHEMA.items():

        #TREAT MISSED COLUMN
        if col not in df.columns:
            logger.warning(f"Expected column not found: '{col}'")

        #SKIP STRING COLUMNS
        if dtype == 'str':
            continue

        try:
            df[col] = df[col].astype(dtype)
        except (ValueError, TypeError) as e:
            logger.warning(f"Failure while converting '{col}' to {dtype}: {e}")

    unexpected = set(df.columns) - set(COLUMN_SCHEMA.keys()) - {"label", "label_name"}
    if unexpected:
        logger.warning(f"Unexpected columns detected: {unexpected}")
    
    return df


#LOADING FILES - FUNCTION TO LOAD SINGLE FILE

def load_sigle_file(filename: str, label: int) -> pd.DataFrame:
    """
    READ A CSV FILE FROM RAW DIRECTORY AND ADD THE LABEL COLUMN.

    ARGS:
        FILENAME: NAME OF THE FILE (EX: "DANE_OT.CSV")
        LABEL: INTEGER REPRESENTING A CLASS
    
    RETURNS:
        A DATAFRAME WITH DATA + COLUMN 'LABEL'
    """
    filepath =  RAW_DATA_DIR / filename

    # CHECK TO AVOID PANDAR ERROR WHEN THERE ARE NO FILE
    if not filepath.exists():
        raise FileNotFoundError(
            f"File not found: {filepath}\n"
            f"Verify whether the file exist in the directory: {RAW_DATA_DIR}"
        )

    logger.info(f"Reading {filename} ...")
    df = pd.read_csv(filepath)

    #ADDING THE LABEL BEFORE ANY TRANSFORMATION
    df["label"] = label

    #ADDING THE CLASS NAME, TO BE REMOVED AFTER THE TRAINING(METADATA)
    df["label_name"] = LABEL_NAME_MAP[label]

    logger.info(f"  → {len(df):,} amostras carregadas | label={label} \
                ({LABEL_NAME_MAP[label]})")
    
    return df


def load_and_merge_all() -> pd.DataFrame:
    """
    READ ALL CSV's FROM THE DATASET, ADD THE LABELs 
    AND RETURN A CONSOLIDATED FILE.

    RETURNS:

        CONSOLIDATED DATAFRAME WITH ALL CLASSES
    """

    dataframes = []

    for filename, label in FILE_LABEL_MAP.items():
        df = load_sigle_file(filename, label)
        dataframes.append(df)
    
    #USING PD.CONCAT BECAUSE IS MORE EFFICIENT THAN DF1 + DF2 + DF3...
    #IGNORE_INDEX=TRUE RESET THE SEQ INDEX

    merged = pd.concat(dataframes, ignore_index=True)

    logger.info(f"\nDataset consolidado: {len(merged):,} amostras | \
                {merged.shape[1]} colunas")
    
    return merged


def validate_dataset(df: pd.DataFrame) -> None:
    """
    TO MAKE THE FIRST DATA VALIDATION BEFORE LOAD THE DATA
    """

    print("\n" + "="*50)
    print("DATASET CHECK")
    print("="*50)

    #SHAPE
    print(f"\n Shape: {df.shape,} linhas × {df.shape[1]} colunas")

    #CLASS DISTRIBUTION

    print("\n CLASS DISTRIBUTION:")
    dist = df.groupby(["label", "label_name"]).size().reset_index(name="count")
    dist["percentual"] = (dist["count"] / len(df) * 100).round(2)
    print(dist.to_string(index=False))

    #NULL VALUES
    
    print("\n NULL VALUES BY COLUMN:")
    nulls = df.isnull().sum()
    nulls_found = nulls[nulls > 0]
    if len(nulls_found) == 0:
        print(" NO NULL VALUES FOUND")
    else:
        print(nulls_found)
    
    #DATA TYPE
    print("\n DATA TYPE:")
    print(df.dtypes.value_counts().to_string())


    # VERY BASIC STATISTICS
    print("\n DESCRIPTIVE ESTATISTICS:")
    print(df.describe().round(3).to_string())

    print("\n" + "="*50 + "\n")

# SAVING

def save_processed(df: pd.DataFrame, filename: str = "dataset_merged.csv") -> None:
    """
    SAVE THE CONSOLIDATED DATAFRAME IN DATA/PROCESSED/
    
    """
    #CREATE THE DIRECTORY IF IT DOESN'T EXISTS
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    output_path =  PROCESSED_DATA_DIR / filename
    df.to_csv(output_path, index=False)

    logger.info(f" DATASET SAVED TO: {output_path}")

#ENTRYPOINT

def main():
    df = load_and_merge_all()
    validate_dataset(df)
    save_processed(df)
    return df

if __name__ == "__main__":
    main()