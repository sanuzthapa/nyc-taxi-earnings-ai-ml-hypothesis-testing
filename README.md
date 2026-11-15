# NYC Taxi Fares: Payment-Method A/B Test & Fare Prediction

Does the payment method (credit card vs. cash) relate to the metered base fare of NYC yellow taxi trips, and can that fare be predicted from pre-trip information? An end-to-end project: SoQL ingestion → cleaning → hypothesis test → ML models.

The analysis lives in four notebooks (markdown, findings and outputs from the original single notebook), and every piece of code they run lives in `src/`.

## Repository layout

```
├── data/
│   ├── raw/                 # Raw Socrata snapshot (created by notebook 01)
│   └── processed/           # Cleaned dataset (created by notebook 02)
├── 01_soql_ingestion.ipynb
├── 02_eda_cleaning.ipynb
├── 03_hypothesis_testing.ipynb
├── 04_model_training.ipynb
├── src/
│   ├── colab_setup.py       # Google Colab: mount Drive, cd into project folder
│   ├── api.py               # SoQL client wrapper + raw/processed file I/O
│   ├── features.py          # Cleaning, feature engineering, route-pair aggregation
│   ├── eda.py               # EDA plots and diagnostics
│   ├── stats.py             # Welch's t-test (card vs. cash)
│   └── model.py             # Estimators, evaluation, diagnostics, inference helpers
├── web/
│   └── nyc_taxi_ml_blog.html  # Standalone interactive technical blog & calculator
├── requirements.txt
└── README.md
```

## Notebook → module map

| Notebook | Calls into |
|---|---|
| `01_soql_ingestion` | `api.fetch_taxi_data`, `api.save_dataframe` |
| `02_eda_cleaning` | `api.load_raw_data`, `features.convert_datetime_columns / add_duration / clean_trip_data / clean_fare_amount`, `eda.data_quality_report / plot_boxplots / verify_distance_outliers`, `api.save_processed_data` |
| `03_hypothesis_testing` | `api.load_processed_data`, `stats.welch_ttest` |
| `04_model_training` | `features.add_pickup_dropoff / add_mean_duration / add_day_month / add_rush_hour / select_model_features`, `eda.plot_*`, `model.split_and_scale / train_linear_regression / train_random_forest / build_residual_table / plot_* / compare_models / export_model_artifacts` |

Simple one-line pandas displays from the original notebook (`df.head()`, `df.describe()`, ...) stay inline.

## Running in Google Colab

Put the repository in Google Drive at `MyDrive/Coursera/projects_github/nyc_taxi_2023_v2` (change `PROJECT_PATHS_TO_TRY` in `src/colab_setup.py` if yours differs). The first two code cells of every notebook are:

```python
from google.colab import drive
drive.mount('/content/drive')
```
```python
import sys
from google.colab import drive

drive.mount('/content/drive')
sys.path.append('/content/drive/MyDrive/Coursera/projects_github/nyc_taxi_2023_v2')

from src.colab_setup import setup_environment
setup_environment()

from src.api import *
from src.features import *
```

The `sys.path.append` line is what lets Python find `src/`; `setup_environment()` then moves into the project folder, applies the plotting defaults and creates `data/raw` and `data/processed`. Run the notebooks in order; data is read and written inside the Drive folder.

Locally: `pip install -r requirements.txt`, open the notebooks from the repository root and skip the Colab cells.

In notebook 01 the API fetch and save calls are commented out, as in the original. Uncomment them to download the data (`fetch_taxi_data(limit=50000, app_token=...)`; the token is optional).

## Results (from the original run)

| Item | Value |
|---|---|
| Rows after cleaning | 45,192 (from 50,000) |
| Mean fare, card (n = 33,948) | $19.76 |
| Mean fare, cash (n = 10,728) | $19.24 |
| Welch t-test | t = 2.92, p = 0.0035 |
| Linear Regression | RMSE $5.27, R² 0.8895 |
| Random Forest | RMSE $5.16, R² 0.8941 |

Saved cell outputs in the notebooks are from the original run; re-run them to refresh.

## Limitations

- **One-day sample.** The query takes the first 50,000 rows by `:id`, almost all from Sunday 1 January 2023, so rush hour has a single positive row and nothing here speaks to weekdays or seasonality.
- **Possible data leakage.** `mean_distance` and `mean_duration` are computed over all rows before the train/test split (as in the original analysis), so the model scores are probably optimistic. `features.RoutePairAggregator` learns route means on the training split only if you want a leakage-free version.
- **Association, not causation.** Payment type is chosen by passengers, and `fare_amount` excludes tips.

## Data source
[NYC TLC Yellow Taxi Trip Data 2023](https://data.cityofnewyork.us/resource/4b4i-vvec) via NYC OpenData (Socrata).
