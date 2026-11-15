"""Exploratory data analysis helpers (faithful ports of the original notebook cells)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display

__all__ = ["data_quality_report", "plot_boxplots", "verify_distance_outliers",
           "plot_duration_vs_fare", "top_high_fares", "show_fare_rows",
           "plot_pairplot", "plot_correlation_heatmap"]


def data_quality_report(df: pd.DataFrame) -> pd.Series:
    """Check for duplicates and missing values; returns missing values per column."""
    # Check for duplicates
    print('Shape of dataframe:', df.shape)
    print('Shape of dataframe with duplicates dropped:', df.drop_duplicates().shape)

    # Check for missing values in dataframe
    print('Total count of missing values:', df.isna().sum().sum())

    # Display missing values per column in dataframe
    print('Missing values per column:')
    return df.isna().sum()


def plot_boxplots(df: pd.DataFrame, title: str = 'Boxplots for outlier detection') -> None:
    """Boxplots of trip_distance, fare_amount and duration."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 2))
    fig.suptitle(title)
    sns.boxplot(ax=axes[0], x=df['trip_distance'])
    sns.boxplot(ax=axes[1], x=df['fare_amount'])
    sns.boxplot(ax=axes[2], x=df['duration'])
    plt.show()


def verify_distance_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """IQR check of trip_distance, then a 0-distance / >P99 filtered frame."""
    # 1. Verify Outliers in trip_distance using IQR method
    Q1 = df['trip_distance'].quantile(0.25)
    Q3 = df['trip_distance'].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = max(0, Q1 - 1.5 * IQR)  # trip distance cannot be negative
    upper_bound = Q3 + 1.5 * IQR

    outliers_upper = df[df['trip_distance'] > upper_bound]
    outliers_zero = df[df['trip_distance'] == 0]

    print(f"--- Outlier Verification ---")
    print(f"IQR: {IQR:.2f} miles (Q1: {Q1:.2f}, Q3: {Q3:.2f})")
    print(f"Theoretical limits for normal distribution: {lower_bound:.2f} to {upper_bound:.2f} miles")
    print(f"Trips with 0.0 distance: {len(outliers_zero)} ({len(outliers_zero)/len(df)*100:.2f}% of data)")
    print(f"Trips exceeding upper bound ({upper_bound:.2f} miles): {len(outliers_upper)} ({len(outliers_upper)/len(df)*100:.2f}% of data)")
    print(f"99th percentile of trip distance: {df['trip_distance'].quantile(0.99):.2f} miles\n")

    # 2. Deal with Outliers: filter out 0-mile trips and extreme outliers (above P99)
    df_clean = df[(df['trip_distance'] > 0) & (df['trip_distance'] <= df['trip_distance'].quantile(0.99))]
    print(f"--- Post-cleaning Analysis ---")
    print(f"Original dataset size: {len(df)} rows")
    print(f"Cleaned dataset size: {len(df_clean)} rows")
    print(f"Cleaned dataset max distance: {df_clean['trip_distance'].max():.2f} miles")
    return df_clean


def plot_duration_vs_fare(df: pd.DataFrame) -> None:
    """Scatter of mean_duration vs fare_amount with a line of best fit."""
    sns.set(style='whitegrid')
    f = plt.figure()
    f.set_figwidth(5)
    f.set_figheight(5)
    sns.regplot(x=df['mean_duration'], y=df['fare_amount'],
                scatter_kws={'alpha': 0.5, 's': 5},
                line_kws={'color': 'red'})
    plt.ylim(0, 70)
    plt.xlim(0, 70)
    plt.title('Mean duration x fare amount')
    plt.show()


def top_high_fares(df: pd.DataFrame, threshold: float = 50) -> pd.Series:
    """Most frequent fare amounts above ``threshold``."""
    return df[df['fare_amount'] > threshold]['fare_amount'].value_counts().head()


def show_fare_rows(df: pd.DataFrame, fare: float = 70, n: int = 30) -> pd.DataFrame:
    """All columns for the first ``n`` trips with an exact fare."""
    pd.set_option('display.max_columns', None)   # Set pandas to display all columns
    return df[df['fare_amount'] == fare].head(n)


def plot_pairplot(df: pd.DataFrame) -> None:
    """Pairplot of fare_amount, mean_duration and mean_distance."""
    sns.pairplot(df[['fare_amount', 'mean_duration', 'mean_distance']],
                 plot_kws={'alpha': 0.4, 'size': 5})
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Pearson correlation heatmap."""
    plt.figure(figsize=(6, 4))
    sns.heatmap(df.corr(method='pearson'), annot=True, cmap='Reds')
    plt.title('Correlation heatmap', fontsize=18)
    plt.show()
