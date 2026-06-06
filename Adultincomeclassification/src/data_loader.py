# load data and clean hidden files ,?, values
from pathlib import Path
import numpy as np
import pandas as pd


def load_data():
    """
    Load the Adult Income dataset from the data folder.
    """

    base_dir = Path(__file__).resolve().parents[1]
    data_path = base_dir / "data" / "adult.csv"

    df = pd.read_csv(data_path)

    return df


def clean_missing_values(df):
    """
    Replace hidden missing values stored as '?' with NaN.
    """

    df = df.copy()
    df = df.replace("?", np.nan)

    return df
    