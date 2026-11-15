"""Cleaning and feature engineering (faithful ports of the original notebook cells)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from IPython.display import display

__all__ = [
    "convert_datetime_columns", "add_duration", "clean_trip_data",
    "clean_fare_amount", "add_pickup_dropoff", "route_group_mean",
    "grouped_to_dict", "map_mean_distance", "route_example",
    "add_mean_duration", "add_day_month", "add_rush_hour", "rush_hour_counts",
    "filter_rush_hour", "select_model_features", "RoutePairAggregator",
]


# --------------------------------------------------------------------------- #
# Datetime conversion and duration
# --------------------------------------------------------------------------- #
def convert_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert pickup / dropoff columns to datetime (prints dtypes before/after)."""
    df = df.copy()
    print('Data type of tpep_pickup_datetime before:', df['tpep_pickup_datetime'].dtype)
    print('Data type of tpep_dropoff_datetime before:', df['tpep_dropoff_datetime'].dtype)

    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'], format='%Y-%m-%d %H:%M:%S')
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'], format='%Y-%m-%d %H:%M:%S')

    print('Data type of tpep_pickup_datetime after:', df['tpep_pickup_datetime'].dtype)
    print('Data type of tpep_dropoff_datetime after:', df['tpep_dropoff_datetime'].dtype)
    return df


def add_duration(df: pd.DataFrame) -> pd.DataFrame:
    """Create the ``duration`` column (minutes)."""
    df = df.copy()
    df['duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']) / np.timedelta64(1, 'm')
    return df


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
def clean_trip_data(df: pd.DataFrame, quantile: float = 0.99) -> pd.DataFrame:
    """Remove passenger-count, distance and duration anomalies / outliers."""
    # Define cleaning thresholds dynamically
    distance_cap = df['trip_distance'].quantile(quantile)
    duration_cap = df['duration'].quantile(quantile)

    # Apply sequential filters
    df_cleaned = df[
        # 1. Filter passenger counts: only realistic values (1 to 4 passengers)
        (df['passenger_count'] >= 1) &
        (df['passenger_count'] <= 4) &

        # 2. Filter trip distance: greater than 0 and cap extreme 1% outliers
        (df['trip_distance'] > 0) &
        (df['trip_distance'] <= distance_cap) &

        # 3. Filter trip duration: greater than 0 and cap extreme 1% outliers
        (df['duration'] > 0) &
        (df['duration'] <= duration_cap)
    ]

    # Summary of the cleaning process
    rows_removed = len(df) - len(df_cleaned)
    percent_removed = (rows_removed / len(df)) * 100

    print(f"--- Data Cleaning Summary ---")
    print(f"Original rows: {len(df):,}")
    print(f"Cleaned rows:  {len(df_cleaned):,}")
    print(f"Rows removed:  {rows_removed:,} ({percent_removed:.2f}% of total data)")
    print(f"\n--- Cleaned Data Distributions ---")
    print(f"Passenger limits: {df_cleaned['passenger_count'].min()} to {df_cleaned['passenger_count'].max()} passengers")
    print(f"Distance limits:  {df_cleaned['trip_distance'].min():.2f} to {df_cleaned['trip_distance'].max():.2f} miles")
    print(f"Duration limits:  {df_cleaned['duration'].min():.2f} to {df_cleaned['duration'].max():.2f} minutes")
    return df_cleaned


def clean_fare_amount(df_cleaned: pd.DataFrame, upper_quantile: float = 0.995) -> pd.DataFrame:
    """Investigate fare anomalies, keep positive fares up to the 99.5th percentile."""
    # 1. Identify and describe fare_amount anomalies
    negative_fares = df_cleaned[df_cleaned['fare_amount'] < 0]
    zero_fares = df_cleaned[df_cleaned['fare_amount'] == 0]

    # Let's check a high-end percentile to handle extreme values safely
    fare_99th = df_cleaned['fare_amount'].quantile(0.99)

    print("--- Fare Amount Anomaly Investigation ---")
    print(f"Trips with negative fares: {len(negative_fares)} records (Min fare: ${df_cleaned['fare_amount'].min()})")
    print(f"Trips with zero fares:     {len(zero_fares)} records")
    print(f"99th percentile of fare:   ${fare_99th:.2f}")
    print(f"Max fare in dataset:       ${df_cleaned['fare_amount'].max():.2f}\n")

    # 2. Handle anomalies: Keep only positive fares and cap at the 99.5th percentile
    #    to keep premium airport runs but drop extreme errors
    fare_upper_limit = df_cleaned['fare_amount'].quantile(upper_quantile)
    df_cleaned = df_cleaned[(df_cleaned['fare_amount'] > 0) & (df_cleaned['fare_amount'] <= fare_upper_limit)]

    print("--- Post-Fare Cleaning Summary ---")
    print(f"New final dataset size:    {len(df_cleaned):,} rows")
    print(f"Fare range:                ${df_cleaned['fare_amount'].min():.2f} to ${df_cleaned['fare_amount'].max():.2f}")
    return df_cleaned


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
def add_pickup_dropoff(df: pd.DataFrame) -> pd.DataFrame:
    """Create the ``pickup_dropoff`` route key."""
    df = df.copy()
    df['pickup_dropoff'] = df['pulocationid'].astype(str) + ' ' + df['dolocationid'].astype(str)
    return df


def route_group_mean(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Mean of ``column`` for each pickup/dropoff group."""
    return df.groupby('pickup_dropoff').mean(numeric_only=True)[[column]]


def grouped_to_dict(grouped: pd.DataFrame, column: str) -> dict:
    """Convert a one-column group table to ``{route: value}``."""
    # 1. Convert `grouped` to a dictionary
    grouped_dict = grouped.to_dict()
    # 2. Reassign to only contain the inner dictionary
    return grouped_dict[column]


def map_mean_distance(df: pd.DataFrame, grouped_dict: dict) -> pd.DataFrame:
    """Add ``mean_distance`` by mapping each trip's route to its mean distance."""
    df = df.copy()
    # 1. Create a mean_distance column that is a copy of the pickup_dropoff helper column
    df['mean_distance'] = df['pickup_dropoff']
    # 2. Map `grouped_dict` to the `mean_distance` column
    df['mean_distance'] = df['mean_distance'].map(grouped_dict)
    return df


def route_example(df: pd.DataFrame, column: str, pu: int = 100, do: int = 231) -> pd.DataFrame:
    """Confirm a route feature using one example route (default 100 -> 231)."""
    return df[(df['pulocationid'] == pu) & (df['dolocationid'] == do)][[column]]


def add_mean_duration(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``mean_duration`` (mean trip duration per pickup/dropoff pair)."""
    df = df.copy()
    # 1. Group by the 'pickup_dropoff' feature and calculate mean duration
    grouped_duration = df.groupby('pickup_dropoff').mean(numeric_only=True)[['duration']]
    # 2. Convert to a dictionary mapping pickup_dropoff to mean duration
    grouped_duration_dict = grouped_duration.to_dict()['duration']
    # 3. Create and map the 'mean_duration' column
    df['mean_duration'] = df['pickup_dropoff'].map(grouped_duration_dict)
    # 4. Confirm that it worked
    display(route_example(df, 'mean_duration').head())
    return df


def add_day_month(df: pd.DataFrame) -> pd.DataFrame:
    """Add lowercase ``day`` name and short ``month`` name columns."""
    df = df.copy()
    # 1. Create 'day' column using lowercase day names
    df['day'] = df['tpep_pickup_datetime'].dt.day_name().str.lower()
    # 2. Create 'month' column using lowercase short-form month names
    df['month'] = df['tpep_pickup_datetime'].dt.strftime('%b').str.lower()
    # 3. Display the newly added columns along with the pickup datetime
    display(df[['tpep_pickup_datetime', 'day', 'month']].head())
    return df


def add_rush_hour(df: pd.DataFrame) -> pd.DataFrame:
    """Binary ``rush_hour``: weekday AND pickup 06:00-10:00 or 16:00-20:00."""
    df = df.copy()
    # 1. Extract hour and weekday flag
    hours = df['tpep_pickup_datetime'].dt.hour
    is_weekday = ~df['day'].isin(['saturday', 'sunday'])

    # 2. Define rush hour intervals
    morning_rush = (hours >= 6) & (hours < 10)
    evening_rush = (hours >= 16) & (hours < 20)

    # 3. Combine conditions to assign a binary flag (1 for rush hour, 0 otherwise)
    df['rush_hour'] = (is_weekday & (morning_rush | evening_rush)).astype(int)

    # 4. Confirm the feature creation by displaying its distribution
    print("Distribution of rush_hour trips:")
    display(df['rush_hour'].value_counts())

    # 5. Display sample records to visually inspect the results
    display(df[['tpep_pickup_datetime', 'day', 'rush_hour']].head(10))
    return df


def rush_hour_counts(df: pd.DataFrame) -> None:
    """Count rush-hour trips two ways."""
    # Option A: Count the number of rows matching the condition directly
    print("Number of rush hour trips (using shape):", df[df['rush_hour'] == 1].shape[0])
    # Option B: Sum only the binary 'rush_hour' column (avoiding datetime columns)
    print("Number of rush hour trips (using column sum):", df['rush_hour'].sum())


def filter_rush_hour(df: pd.DataFrame) -> pd.DataFrame:
    """Return only the rush-hour trips."""
    return df[df['rush_hour'] == 1]


def select_model_features(df_cleaned: pd.DataFrame) -> pd.DataFrame:
    """Drop redundant, irrelevant or not-available-at-deployment columns."""
    df2 = df_cleaned.copy()
    df2 = df2.drop(['tpep_dropoff_datetime', 'tpep_pickup_datetime',
                    'trip_distance', 'ratecodeid', 'store_and_fwd_flag',
                    'pulocationid', 'dolocationid', 'payment_type', 'extra',
                    'mta_tax', 'tip_amount', 'tolls_amount',
                    'improvement_surcharge', 'total_amount', 'duration',
                    'pickup_dropoff', 'day', 'month'], axis=1)
    df2.info()
    return df2


# --------------------------------------------------------------------------- #
# OPTIONAL: leakage-safe alternative (not used by the notebooks by default)
# --------------------------------------------------------------------------- #
class RoutePairAggregator:
    """Learn mean distance / duration per route on the TRAINING split only.

    The notebooks (like the original analysis) compute route means over all rows
    before splitting, which lets a test trip contribute to its own feature.
    Use this class instead -- ``fit`` on the train split, ``transform`` both
    splits -- for a leakage-free evaluation. Unseen routes fall back to the
    global training mean.
    """

    def __init__(self, key: str = "pickup_dropoff"):
        self.key = key
        self.sources = {"trip_distance": "mean_distance", "duration": "mean_duration"}

    def fit(self, df: pd.DataFrame) -> "RoutePairAggregator":
        self.route_means_ = df.groupby(self.key)[list(self.sources)].mean().rename(columns=self.sources)
        self.global_means_ = {v: float(df[k].mean()) for k, v in self.sources.items()}
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for new_col, fallback in self.global_means_.items():
            out[new_col] = out[self.key].map(self.route_means_[new_col]).fillna(fallback)
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)
