
"""

Interface to run the Historical Processing System

Adjust Parameters Below

"""

import logging
from pathlib import Path

import numpy as np
import polars as pl
# from decimal import Decimal
from datetime import datetime, timezone

from data_types import DataSourceType, AvailableDatasets
from historical_processor import HistoricalDataProcessor
from buffer import (
    HistoricalConfig, 
    HelperFunctions,
    DeltasDistribution,
    NormalCDF,
    StudentTCDF,
    SkewedTCDF,
    EmpiricalCDF,
)




# ============================================================================
# [START] - Logging Setup
# ============================================================================


# Configure basic logging to the console with INFO level
# Verbosity Levels: DEBUG/INFO/WARNING/ERROR/CRITICAL
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

logger = logging.getLogger(__name__)


# ============================================================================
# [END] - Logging Setup
# ============================================================================


# ============================================================================
# [START] - File Paths for Import & Export
# ============================================================================

# WORKING DIRECTORY
MAIN_DIR = Path("/Projects/")

# HISTORICAL DATA
HISTORICAL_DATA = MAIN_DIR / "0_data" / "historical"
# SPECIFIC REQUEST


# class AvailableDatasets:
# """Available Datasets, returns request ID"""
# FULL_YEAR_2024__CME_JPY_V: str = "GLBX-20251028-HAE9P7SP3U"
# FULL_YEAR_2023__CME_JPY_V: str = "GLBX-20251116-MJ5JKQA9A4"
# FULL_YEAR_2022__CME_JPY_V: str = "GLBX-20251115-DJHUGKSLWB"

request_id = AvailableDatasets.FULL_YEAR_2023__CME_JPY_V


# IMPORT PATH
PARQUET_DIR = HISTORICAL_DATA / "parquet"

# EXPORT PATH
EXPORT_DIR = MAIN_DIR / "99_exports" / "pool_5"
TTEMPORARY_DIR = ""
LOGS_DIR = ""

# ============================================================================
# [END] - File Paths for Import & Export
# ============================================================================




# ============================================================================
# [START] - Polars Display Configurations
# ============================================================================

pl.Config.set_tbl_rows(1000)      # show up to 100 rows
pl.Config.set_tbl_cols(50)       # show up to 50 columns
# pl.Config.set_tbl_width_chars(0) # 0 = no width limit
pl.Config.set_tbl_hide_dataframe_shape(False)  # keep shape info

# [ DEFAULT DECIMAL DISPLAY ]
# # Option 1: Set display precision globally
# pl.Config.set_fmt_float("full")

# Option 2: Set specific decimal places
pl.Config.set_float_precision(7)

# # Option 3: Check individual values
# print(result["price_decimal"][7])  # Will show: 0.0070995

# ============================================================================
# [END] - Polars Display Configurations
# ============================================================================




# ============================================================================
# [START] - PARAMETERS
# ============================================================================


# Contracts per VolumeBar
VOLUME_BAR_SIZE = 100

# Number of rows to maintain in memory, before archiving to disk
DATA_ARCHIVE_SIZE = 20_000


CONTRACT_SIZE__JPY_USD = 12_500_000
# 0.0000005 per JPY increment = $6.25
# Spreads: 0.0000002 per JPY increment = $2.50
# Quarterly contracts (Mar, Jun, Sep, Dec) listed for 20 consecutive quarters and serial contracts listed for 3 months


# Number of Trading Days per Year
RECORDED_TRADING_DAYS = 313 # Dates w/ transactions taking place. Timezone is UTC, but CME is CT (-5/-6) and Japan is JST (+9)
STANDARD_TRADING_DAYS = 245 # Somewhere between 240-255

# Exact Calculations
CONTRACT_TRADING_VOLUME__2024 = 37_581_060
SPECIFIC_ADV_CALCULATION = np.round(CONTRACT_TRADING_VOLUME__2024 / STANDARD_TRADING_DAYS).astype(int) # or use `RECORDED_TRADING_DAYS`


# 120-155K per day depending on number of trading days
ADV_CONTRACTS = 150_000_000

# Typical number of buckets per day (~1 every 5 minutes)
# (Lookback window required to generate metric)
V__BUCKETS_PER_METRIC = 250

# Contracts per Bucket
BUCKET_SIZE___CONTRACTS = np.round(ADV_CONTRACTS / V__BUCKETS_PER_METRIC).astype(int)
BUCKET_SIZE___VOLUME_BARS = np.round(BUCKET_SIZE___CONTRACTS / VOLUME_BAR_SIZE).astype(int)


# ============================================================================
# [END] - PARAMETERS
# ============================================================================




# ============================================================================
# [ START ] - MAIN
# ============================================================================



def main(vb_size: int, export: str):


    # ============================================================================
    # [START] - Import Filters
    # ============================================================================


    # Convert START_DATE and END_DATE to nanosecond epochs (integers)
    START_DATE_NS = int(datetime(2022, 1, 1, tzinfo=timezone.utc).timestamp() * 1e9)
    END_DATE_NS   = int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp() * 1e9)


    # # Data files to glob for processing
    # DATA_GLOB = pl.scan_parquet(str(PARQUET_DIR / request_id / "*.parquet"))

    # Method 2: List of glob patterns for MULTIPLE DIRECTORIES
    DATA_GLOB = pl.scan_parquet([
        str(PARQUET_DIR / AvailableDatasets.FULL_YEAR_2022__CME_JPY_V / "*.parquet"),
        str(PARQUET_DIR / AvailableDatasets.FULL_YEAR_2023__CME_JPY_V / "*.parquet"),
        str(PARQUET_DIR / AvailableDatasets.FULL_YEAR_2024__CME_JPY_V / "*.parquet"),
    ])


    # [ ACTUAL FILTER REQUEST FOR PROCESSING ]
    lf = (

        # 0) Glob across all files in data directory
        DATA_GLOB
        
        # 1) Select only needed columns FIRST (reduces data volume)
        .select([
            "ts_recv",
            "price",
            "size",
            "side",
            "sequence",
            "instrument_id",
        ])

        # 2.) Convert to physical int representation (just stripping data, saved as int64 in Parquet file)
        .with_columns(
            pl.col("ts_recv").dt.epoch(time_unit="ns").cast(pl.UInt64).alias("ts_recv")
        )
        
        # 3) Filter on raw uint64 epoch values (fastest)
        .filter(
            (pl.col("ts_recv") >= START_DATE_NS) & 
            (pl.col("ts_recv") < END_DATE_NS)
        )
        .filter(
            pl.col("size") > VOLUME_BAR_SIZE
        )
        # .filter(pl.col("side").is_in(["A", "B"])) # Remove "N" side orders
        # .limit(1_000)

    )



    
    # ============================================================================
    # [END] - Import Filters
    # ============================================================================


    # [ Define & Validate Configuration Settings ]
    # (Verifies parameter values are correct via assertions in post_init)
    
    config = HistoricalConfig(
    
        # Volume Bar
        volumebar_contracts_per_bar=vb_size,
        volumebar_lookback_buffer_size=50,
        volumebar_archive_size=DATA_ARCHIVE_SIZE,

        # Bucket (Fixed)
        enable_fixed_buckets=False,
        fixed_bucket_volumebars_per_bucket=40, #V__BUCKETS_PER_METRIC
        fixed_bucket_archive_size=DATA_ARCHIVE_SIZE,

        # Bucket (Interval)
        enable_interval_buckets=False, #V__BUCKETS_PER_METRIC
        interval_bucket_volumebars_per_bucket=30,
        interval_bucket_archive_size=DATA_ARCHIVE_SIZE,

        # Metrics (On/Off)
        training=True, # If Training: No metrics are generated
        # Metrics (Fixed)
        fixed_metrics_buckets_per_metric=50,
        fixed_metrics_archive_size=500,
        # Metrics (Interval)
        interval_metrics_buckets_per_metric=50,
        interval_metrics_archive_size=500,

        # Dataset Info
        query_start_date=START_DATE_NS,
        query_end_date=END_DATE_NS,
        query_dataset_directory=str(PARQUET_DIR / "*.parquet"),

        # Override Defaults
        output_dir=EXPORT_DIR / export,
        max_workers=3,

        include_debug_timestamps=True,

        test_schema=False,
        test_schema_show_valid_message=False,

    )


    # [ Visually Review the Request ]
    initial_total_rows, initial_total_size, initial_date_min, initial_date_max = HelperFunctions.get_dataset_summary(config=config, lf=lf, start_date_ns=START_DATE_NS, end_date_ns=END_DATE_NS)


    """

    GET `price_deltas` FROM DATA!
    Filter NaN values in query(?)

    """
    price_deltas = pl.scan_parquet(str(PARQUET_DIR / "price_deltas/filtered_price_deltas.parquet")).collect()['price_deltas'].to_numpy()


    # [ Initialize Returns Distributions for Use During <BVC> Calculations ]
    # ( Normalizing Price_Delta based on Standard_Deviation() of ALL Price Deltas)
    returns_distribution = DeltasDistribution(deltas=price_deltas)


    # [ Initialize Classifier Distribution for Use During <BVC> Calculations ]
    """
    # Choose your CDF model
    normal = NormalCDF()
    student = StudentTCDF(df=5)
    skewed = SkewedTCDF(df=5, lam=-0.2)  # Negative skew typical for equities
    empirical = EmpiricalCDF(z)
    """
    classifier_distribution = NormalCDF()
    

    # [ Initialize Processor ]
    main_processor = HistoricalDataProcessor(
        data_source_type=DataSourceType.MDP3_HISTORICAL,
        config=config,
        historical_data_lazyframe=lf,
        returns_distribution=returns_distribution,
        classifier_distribution=classifier_distribution,
        total_rows=initial_total_rows,
        start_ts_ns=START_DATE_NS,
    )
    

    # [ Run ]
    result = main_processor.process_parquet_data()


    # [ Evaluate if all initally requested orders / volumes were processed ]
    # Pretty print results
    # HelperFunctions.evaluate_processing_results() is called from pretty_print_processing_verification_results()
    HelperFunctions.pretty_print_processing_verification_results(result, initial_total_rows, initial_total_size, config)


    # [ End ]
    HelperFunctions.completion_sound()
    print("\n\n[ Historical calculations complete! ] \n\n\n\n")




if __name__ == '__main__':


    main(vb_size=VOLUME_BAR_SIZE, export=(f"vb_{VOLUME_BAR_SIZE}"))
    # sizes = [
    #     25,
    #     50,
    #     75,
    #     100,
    #     125,
    #     150,
    #     175,
    #     200,
    #     250,
    #     300,
    #     500,
    #     500,
    #     750,
    #     1000,
    # ]

    # for size in sizes:


    #     # [ Begin ]
    #     print("\n\nStarting historical calculations...\n\n")
    #     HelperFunctions.completion_sound()


    #     main(vb_size=size, export=(f"vb_{size}"))



