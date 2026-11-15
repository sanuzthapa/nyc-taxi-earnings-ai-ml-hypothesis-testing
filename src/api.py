"""Data access: NYC OpenData SoQL client and local raw / processed file I/O."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

__all__ = ["PROJECT_ROOT", "RAW_DIR", "PROCESSED_DIR", "RAW_FILENAME",
           "PROCESSED_FILENAME", "API_URL", "SELECTED_COLUMNS", "NUMERIC_COLS",
           "DATETIME_COLS", "fetch_taxi_data", "save_dataframe",
           "load_raw_data", "save_processed_data", "load_processed_data"]

# <repo>/src/api.py -> <repo>
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_FILENAME = "nyc_yellow_taxi_data_2023.csv"
PROCESSED_FILENAME = "nyc_yellow_taxi_2023_clean.csv"

# 1. API endpoint for NYC 2023 Yellow Taxi Data
API_URL = "https://data.cityofnewyork.us/resource/4b4i-vvec.json"

# 2. The exact 17 requested columns
SELECTED_COLUMNS = [
    "vendorid", "tpep_pickup_datetime", "tpep_dropoff_datetime",
    "passenger_count", "trip_distance", "ratecodeid", "store_and_fwd_flag",
    "pulocationid", "dolocationid", "payment_type", "fare_amount", "extra",
    "mta_tax", "tip_amount", "tolls_amount", "improvement_surcharge",
    "total_amount",
]
NUMERIC_COLS = [
    "passenger_count", "trip_distance", "ratecodeid", "payment_type",
    "fare_amount", "extra", "mta_tax", "tip_amount", "tolls_amount",
    "improvement_surcharge", "total_amount", "pulocationid", "dolocationid",
]
DATETIME_COLS = ["tpep_pickup_datetime", "tpep_dropoff_datetime"]


def fetch_taxi_data(limit: int = 50000, app_token: str | None = None) -> pd.DataFrame:
    """Fetch trips with server-side SoQL parameters (``$select/$limit/$order``)."""
    # 3. Server-side SoQL parameters
    params = {
        "$select": ", ".join(SELECTED_COLUMNS),
        "$limit": limit,            # Fetching 50,000 rows for our analysis
        "$order": ":id",
    }
    headers = {"User-Agent": "Google-Colab-Data-Fetcher/1.0"}
    if app_token:
        headers["X-App-Token"] = app_token

    print("Fetching records from NYC OpenData API...")
    response = requests.get(API_URL, params=params, headers=headers)
    response.raise_for_status()

    # 4. Load directly into DataFrame
    df = pd.DataFrame(response.json())

    # 5. Type casting
    df[NUMERIC_COLS] = df[NUMERIC_COLS].apply(pd.to_numeric, errors="coerce")
    for col in DATETIME_COLS:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # 6. Verify columns and structure
    print(f"Loaded {len(df):,} rows with {df.shape[1]} columns.\n")
    print(df.info())
    return df


def save_dataframe(df: pd.DataFrame, directory=RAW_DIR,
                   filename: str = RAW_FILENAME) -> Path:
    """Save the dataframe to CSV, creating the target directory if needed."""
    Path(directory).mkdir(parents=True, exist_ok=True)
    output_file_path = Path(directory) / filename
    df.to_csv(output_file_path, index=False)
    print(f"DataFrame successfully saved to: {output_file_path}")
    return output_file_path


def load_raw_data(directory=RAW_DIR, filename: str = RAW_FILENAME) -> pd.DataFrame:
    """Read the raw snapshot saved by notebook 01 and show its structure."""
    df = pd.read_csv(Path(directory) / filename)
    df.info()
    return df


def save_processed_data(df: pd.DataFrame, directory=PROCESSED_DIR,
                        filename: str = PROCESSED_FILENAME) -> Path:
    """Save the cleaned dataset (notebook 02 -> notebooks 03 and 04)."""
    Path(directory).mkdir(parents=True, exist_ok=True)
    out = Path(directory) / filename
    df.to_csv(out, index=False)
    print(f"Saved {len(df):,} cleaned rows -> {out}")
    return out


def load_processed_data(directory=PROCESSED_DIR,
                        filename: str = PROCESSED_FILENAME) -> pd.DataFrame:
    """Load the cleaned dataset with datetime columns parsed."""
    df = pd.read_csv(Path(directory) / filename, parse_dates=DATETIME_COLS)
    print(f"Loaded {len(df):,} cleaned rows")
    return df
