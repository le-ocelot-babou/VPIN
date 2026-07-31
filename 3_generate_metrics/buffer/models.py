"""
Data structures and models for volumebar calculations system.

Includes:
- VOLUMEBAR_DTYPE: 54-field structured array dtype
- BUCKET_DTYPE: Same as VOLUMEBAR_DTYPE (aggregated volumebars)
- StructuredCircularBuffer: Circular buffer for structured arrays
- ArchivalTask: Data package for archival operations
"""

import os
import sys

import ast
import inspect
import textwrap

import numpy as np
import polars as pl
from scipy import stats
from scipy.special import erf
from scipy.interpolate import interp1d

import logging

from datetime import datetime, timezone

from enum import Enum
from dataclasses import dataclass
from numpy.typing import NDArray
from typing import Any, Callable, Dict, Optional, Union, List

from .config import HistoricalConfig




logger = logging.getLogger(__name__)


# ============================================================================
# STRUCTURED ARRAY DTYPES
# ============================================================================

# Small epsilon for numerical stability
EPSILON = np.float64(1e-6)




VOLUMEBAR_DTYPE = np.dtype([
    # ========================================================================
    # Configuration & Identifiers
    # ========================================================================
    ('id', np.uint32),
    ('bar_volume_size', np.uint32),
    
    # ========================================================================
    # Metadata Fields
    # ========================================================================
    ('gap_return', np.bool_),
    ('max_time_gap_ns', np.uint64),
    ('contains_oversized_order', np.bool_),
    ('has_resized', np.bool_),
    
    # ========================================================================
    # Order & Volume Statistics (Ambiguous/Total)
    # ========================================================================
    ('order_count', np.uint32),
    ('order_splits', np.uint32),
    ('volume_total', np.uint32),
    ('bar_complete', np.bool_),
    
    # ========================================================================
    # Active Order Counts
    # ========================================================================
    ('active_order_count_buy', np.uint32),
    ('active_order_count_sell', np.uint32),
    ('active_order_count_none', np.uint32),
    
    # ========================================================================
    # Active Volumes
    # ========================================================================
    ('active_volume_buy', np.uint32),
    ('active_volume_sell', np.uint32),
    ('active_volume_none', np.uint32),
    
    # ========================================================================
    # Active Imbalance Metrics
    # ========================================================================
    ('active_imbalance_signed', np.int64),
    ('active_imbalance_abs', np.uint64),
    ('active_imbalance_signed_ratio', np.float64),
    ('active_imbalance_abs_ratio', np.float64),
    ('active_imbalance_buy_ratio', np.float64),
    
    # ========================================================================
    # Active Link Function Transforms
    # ========================================================================
    ('active_imbalance_signed_ratio_atanh', np.float64),
    ('active_imbalance_buy_ratio_logit', np.float64),
    ('active_imbalance_abs_ratio_logit', np.float64),
    
    # ========================================================================
    # Active Price Metrics (VWAP)
    # Stored in 1e-9 scaled units (e.g., 1085505555.742)
    # Preserves sub-tick precision from VWAP calculations
    # ========================================================================
    ('active_buy_vwap', np.float64),
    ('active_sell_vwap', np.float64),
    ('active_none_vwap', np.float64),
    ('active_spread_vwap', np.float64),
    ('active_midpoint_vwap', np.float64),
    
    # ========================================================================
    # Active Weighted Midpoints
    # ========================================================================
    ('active_mid_imbalance_weighted', np.float64),
    ('active_mid_flow_weighted', np.float64),
    ('active_mid_aggressor_weighted', np.float64),
    
    # ========================================================================
    # Active Price Range Metrics
    # ========================================================================
    ('active_buy_price_min', np.float64),
    ('active_buy_price_max', np.float64),
    ('active_buy_price_range', np.float64),
    ('active_sell_price_min', np.float64),
    ('active_sell_price_max', np.float64),
    ('active_sell_price_range', np.float64),
    ('active_none_price_min', np.float64),
    ('active_none_price_max', np.float64),
    ('active_none_price_range', np.float64),
    
    # ========================================================================
    # Active Pace Metrics
    # ========================================================================
    ('active_buy_pace', np.float64),
    ('active_sell_pace', np.float64),
    ('active_buy_pace_log', np.float64),
    ('active_sell_pace_log', np.float64),
    
    # ========================================================================
    # Active N-Side Inferred Classification
    # ========================================================================
    ('active_none_inferred_buy_volume', np.float64),
    ('active_none_inferred_sell_volume', np.float64),
    ('active_none_inferred_buy_vwap', np.float64),
    ('active_none_inferred_sell_vwap', np.float64),
    
    # ========================================================================
    # Adjusted Volumes (Active + N-Side Inferred)
    # ========================================================================
    ('adjusted_volume_buy', np.float64),
    ('adjusted_volume_sell', np.float64),
    
    # ========================================================================
    # Adjusted Imbalance Metrics
    # ========================================================================
    ('adjusted_imbalance_signed', np.float64),
    ('adjusted_imbalance_abs', np.float64),
    ('adjusted_imbalance_signed_ratio', np.float64),
    ('adjusted_imbalance_abs_ratio', np.float64),
    ('adjusted_imbalance_buy_ratio', np.float64),
    
    # ========================================================================
    # Adjusted Link Function Transforms
    # ========================================================================
    ('adjusted_imbalance_signed_ratio_atanh', np.float64),
    ('adjusted_imbalance_buy_ratio_logit', np.float64),
    ('adjusted_imbalance_abs_ratio_logit', np.float64),
    
    # ========================================================================
    # Adjusted Price Metrics (VWAP)
    # ========================================================================
    ('adjusted_buy_vwap', np.float64),
    ('adjusted_sell_vwap', np.float64),
    ('adjusted_spread_vwap', np.float64),
    ('adjusted_midpoint_vwap', np.float64),
    
    # ========================================================================
    # Adjusted Weighted Midpoints
    # ========================================================================
    ('adjusted_mid_imbalance_weighted', np.float64),
    ('adjusted_mid_flow_weighted', np.float64),
    ('adjusted_mid_aggressor_weighted', np.float64),
    
    # ========================================================================
    # Passive Midprice Metrics
    # ========================================================================
    ('passive_midprice', np.float64),
    ('passive_midprice_delta_price', np.float64),
    ('passive_midprice_delta_percent', np.float64),
    ('passive_midprice_delta_log', np.float64),
    
    # ========================================================================
    # Passive Normalized and CDF Metrics
    # ========================================================================
    ('passive_midprice_delta_normalized', np.float64),
    ('passive_midprice_delta_cdf', np.float64),
    
    # ========================================================================
    # Passive Volume Classification
    # ========================================================================
    ('passive_buy_volume', np.float64),
    ('passive_sell_volume', np.float64),
    
    # ========================================================================
    # Passive Imbalance Metrics
    # ========================================================================
    ('passive_imbalance_signed', np.float64),
    ('passive_imbalance_abs', np.float64),
    ('passive_imbalance_signed_ratio', np.float64),
    ('passive_imbalance_abs_ratio', np.float64),
    ('passive_imbalance_buy_ratio', np.float64),
    
    # ========================================================================
    # Passive Link Function Transforms
    # ========================================================================
    ('passive_imbalance_signed_ratio_atanh', np.float64),
    ('passive_imbalance_buy_ratio_logit', np.float64),
    ('passive_imbalance_abs_ratio_logit', np.float64),
    
    # ========================================================================
    # Temporal Metrics
    # ========================================================================
    ('start_ts_ns', np.uint64),
    ('end_ts_ns', np.uint64),
    ('time_elapsed_ns', np.uint64),
    ('pace_of_contracts_traded', np.float64),
    ('time_elapsed_ns_log', np.float64),
    ('pace_of_contracts_traded_log', np.float64),
    
    # ========================================================================
    # Instrument Tracking
    # ========================================================================
    ('contract_roll', np.bool_),
    ('latest_instrument_id', np.uint32),
    
    # ========================================================================
    # Divergence Metrics (Active vs Passive)
    # ========================================================================
    ('divergence_buy_volume', np.float64),
    ('divergence_sell_volume', np.float64),
    ('divergence_imbalance_signed', np.float64),
    ('divergence_imbalance_signed_ratio', np.float64),
    ('divergence_buy_ratio', np.float64),
    
    # ========================================================================
    # Derived Indicator
    # ========================================================================
    ('derived_price_direction_positive', np.bool_),
    
    # ========================================================================
    # Delta Calculations - Ambiguous/Total Deltas
    # ========================================================================
    ('delta_order_count', np.int32),
    ('delta_order_count_pct', np.float64),
    ('delta_order_count_log', np.float64),
    ('delta_order_splits', np.int32),
    ('delta_volume_total', np.int32),
    ('delta_volume_total_pct', np.float64),
    ('delta_volume_total_log', np.float64),
    
    # ========================================================================
    # Delta Calculations - Active Order Count Deltas
    # ========================================================================
    ('delta_active_order_count_buy', np.int32),
    ('delta_active_order_count_buy_pct', np.float64),
    ('delta_active_order_count_buy_log', np.float64),
    ('delta_active_order_count_sell', np.int32),
    ('delta_active_order_count_sell_pct', np.float64),
    ('delta_active_order_count_sell_log', np.float64),
    ('delta_active_order_count_none', np.int32),
    ('delta_active_order_count_none_pct', np.float64),
    ('delta_active_order_count_none_log', np.float64),
    
    # ========================================================================
    # Delta Calculations - Active Volume Deltas
    # ========================================================================
    ('delta_active_volume_buy', np.int32),
    ('delta_active_volume_buy_pct', np.float64),
    ('delta_active_volume_buy_log', np.float64),
    ('delta_active_volume_sell', np.int32),
    ('delta_active_volume_sell_pct', np.float64),
    ('delta_active_volume_sell_log', np.float64),
    ('delta_active_volume_none', np.int32),
    ('delta_active_volume_none_pct', np.float64),
    ('delta_active_volume_none_log', np.float64),
    
    # ========================================================================
    # Delta Calculations - Active Imbalance Deltas
    # ========================================================================
    ('delta_active_imbalance_signed', np.int64),
    ('delta_active_imbalance_abs', np.int64),
    ('delta_active_imbalance_signed_ratio', np.float64),
    ('delta_active_imbalance_abs_ratio', np.float64),
    ('delta_active_imbalance_buy_ratio', np.float64),
    
    # ========================================================================
    # Delta Calculations - Active Link Function Transform Deltas
    # ========================================================================
    ('delta_active_imbalance_signed_ratio_atanh', np.float64),
    ('delta_active_imbalance_buy_ratio_logit', np.float64),
    ('delta_active_imbalance_abs_ratio_logit', np.float64),
    
    # ========================================================================
    # Delta Calculations - Active Price Deltas (VWAP)
    # ========================================================================
    ('delta_active_buy_vwap', np.float64),
    ('delta_active_buy_vwap_pct', np.float64),
    ('delta_active_buy_vwap_log', np.float64),
    ('delta_active_sell_vwap', np.float64),
    ('delta_active_sell_vwap_pct', np.float64),
    ('delta_active_sell_vwap_log', np.float64),
    ('delta_active_none_vwap', np.float64),
    ('delta_active_none_vwap_pct', np.float64),
    ('delta_active_none_vwap_log', np.float64),
    ('delta_active_spread_vwap', np.float64),
    ('delta_active_spread_vwap_pct', np.float64),
    ('delta_active_spread_vwap_log', np.float64),
    ('delta_active_midpoint_vwap', np.float64),
    ('delta_active_midpoint_vwap_pct', np.float64),
    ('delta_active_midpoint_vwap_log', np.float64),
    
    # ========================================================================
    # Delta Calculations - Active Weighted Midpoint Deltas
    # ========================================================================
    ('delta_active_mid_imbalance_weighted', np.float64),
    ('delta_active_mid_imbalance_weighted_pct', np.float64),
    ('delta_active_mid_imbalance_weighted_log', np.float64),
    ('delta_active_mid_flow_weighted', np.float64),
    ('delta_active_mid_flow_weighted_pct', np.float64),
    ('delta_active_mid_flow_weighted_log', np.float64),
    ('delta_active_mid_aggressor_weighted', np.float64),
    ('delta_active_mid_aggressor_weighted_pct', np.float64),
    ('delta_active_mid_aggressor_weighted_log', np.float64),
    
    # ========================================================================
    # Delta Calculations - Active Price Range Deltas
    # ========================================================================
    ('delta_active_buy_price_min', np.float64),
    ('delta_active_buy_price_min_pct', np.float64),
    ('delta_active_buy_price_min_log', np.float64),
    ('delta_active_buy_price_max', np.float64),
    ('delta_active_buy_price_max_pct', np.float64),
    ('delta_active_buy_price_max_log', np.float64),
    ('delta_active_buy_price_range', np.float64),
    ('delta_active_buy_price_range_pct', np.float64),
    ('delta_active_buy_price_range_log', np.float64),
    ('delta_active_sell_price_min', np.float64),
    ('delta_active_sell_price_min_pct', np.float64),
    ('delta_active_sell_price_min_log', np.float64),
    ('delta_active_sell_price_max', np.float64),
    ('delta_active_sell_price_max_pct', np.float64),
    ('delta_active_sell_price_max_log', np.float64),
    ('delta_active_sell_price_range', np.float64),
    ('delta_active_sell_price_range_pct', np.float64),
    ('delta_active_sell_price_range_log', np.float64),
    ('delta_active_none_price_min', np.float64),
    ('delta_active_none_price_min_pct', np.float64),
    ('delta_active_none_price_min_log', np.float64),
    ('delta_active_none_price_max', np.float64),
    ('delta_active_none_price_max_pct', np.float64),
    ('delta_active_none_price_max_log', np.float64),
    ('delta_active_none_price_range', np.float64),
    ('delta_active_none_price_range_pct', np.float64),
    ('delta_active_none_price_range_log', np.float64),
    
    # ========================================================================
    # Delta Calculations - Active Pace Deltas
    # ========================================================================
    ('delta_active_buy_pace', np.float64),
    ('delta_active_buy_pace_pct', np.float64),
    ('delta_active_buy_pace_log', np.float64),
    ('delta_active_sell_pace', np.float64),
    ('delta_active_sell_pace_pct', np.float64),
    ('delta_active_sell_pace_log', np.float64),
    
    # ========================================================================
    # Delta Calculations - Active Pace Log Transform Deltas (delta of base log field)
    # ========================================================================
    ('delta_active_buy_pace_log_base', np.float64),
    ('delta_active_sell_pace_log_base', np.float64),
    
    # ========================================================================
    # Delta Calculations - Active N-Side Inferred Volume Deltas
    # ========================================================================
    ('delta_active_none_inferred_buy_volume', np.float64),
    ('delta_active_none_inferred_buy_volume_pct', np.float64),
    ('delta_active_none_inferred_buy_volume_log', np.float64),
    ('delta_active_none_inferred_sell_volume', np.float64),
    ('delta_active_none_inferred_sell_volume_pct', np.float64),
    ('delta_active_none_inferred_sell_volume_log', np.float64),
    
    # ========================================================================
    # Delta Calculations - Active N-Side Inferred VWAP Deltas
    # ========================================================================
    ('delta_active_none_inferred_buy_vwap', np.float64),
    ('delta_active_none_inferred_buy_vwap_pct', np.float64),
    ('delta_active_none_inferred_buy_vwap_log', np.float64),
    ('delta_active_none_inferred_sell_vwap', np.float64),
    ('delta_active_none_inferred_sell_vwap_pct', np.float64),
    ('delta_active_none_inferred_sell_vwap_log', np.float64),
    
    # ========================================================================
    # Delta Calculations - Adjusted Volume Deltas
    # ========================================================================
    ('delta_adjusted_volume_buy', np.float64),
    ('delta_adjusted_volume_buy_pct', np.float64),
    ('delta_adjusted_volume_buy_log', np.float64),
    ('delta_adjusted_volume_sell', np.float64),
    ('delta_adjusted_volume_sell_pct', np.float64),
    ('delta_adjusted_volume_sell_log', np.float64),
    
    # ========================================================================
    # Delta Calculations - Adjusted Imbalance Deltas
    # ========================================================================
    ('delta_adjusted_imbalance_signed', np.float64),
    ('delta_adjusted_imbalance_abs', np.float64),
    ('delta_adjusted_imbalance_signed_ratio', np.float64),
    ('delta_adjusted_imbalance_abs_ratio', np.float64),
    ('delta_adjusted_imbalance_buy_ratio', np.float64),
    
    # ========================================================================
    # Delta Calculations - Adjusted Link Function Transform Deltas
    # ========================================================================
    ('delta_adjusted_imbalance_signed_ratio_atanh', np.float64),
    ('delta_adjusted_imbalance_buy_ratio_logit', np.float64),
    ('delta_adjusted_imbalance_abs_ratio_logit', np.float64),
    
    # ========================================================================
    # Delta Calculations - Adjusted Price Deltas (VWAP)
    # ========================================================================
    ('delta_adjusted_buy_vwap', np.float64),
    ('delta_adjusted_buy_vwap_pct', np.float64),
    ('delta_adjusted_buy_vwap_log', np.float64),
    ('delta_adjusted_sell_vwap', np.float64),
    ('delta_adjusted_sell_vwap_pct', np.float64),
    ('delta_adjusted_sell_vwap_log', np.float64),
    ('delta_adjusted_spread_vwap', np.float64),
    ('delta_adjusted_spread_vwap_pct', np.float64),
    ('delta_adjusted_spread_vwap_log', np.float64),
    ('delta_adjusted_midpoint_vwap', np.float64),
    ('delta_adjusted_midpoint_vwap_pct', np.float64),
    ('delta_adjusted_midpoint_vwap_log', np.float64),
    
    # ========================================================================
    # Delta Calculations - Adjusted Weighted Midpoint Deltas
    # ========================================================================
    ('delta_adjusted_mid_imbalance_weighted', np.float64),
    ('delta_adjusted_mid_imbalance_weighted_pct', np.float64),
    ('delta_adjusted_mid_imbalance_weighted_log', np.float64),
    ('delta_adjusted_mid_flow_weighted', np.float64),
    ('delta_adjusted_mid_flow_weighted_pct', np.float64),
    ('delta_adjusted_mid_flow_weighted_log', np.float64),
    ('delta_adjusted_mid_aggressor_weighted', np.float64),
    ('delta_adjusted_mid_aggressor_weighted_pct', np.float64),
    ('delta_adjusted_mid_aggressor_weighted_log', np.float64),
    
    # ========================================================================
    # Delta Calculations - Passive Midprice First-Order Deltas
    # ========================================================================
    ('delta_passive_midprice', np.float64),
    ('delta_passive_midprice_pct', np.float64),
    ('delta_passive_midprice_log', np.float64),
    
    # ========================================================================
    # Delta Calculations - Passive Midprice Second-Order Deltas (Acceleration)
    # ========================================================================
    ('delta_passive_midprice_delta_price', np.float64),
    ('delta_passive_midprice_delta_price_pct', np.float64),
    ('delta_passive_midprice_delta_price_log', np.float64),
    ('delta_passive_midprice_delta_percent', np.float64),
    ('delta_passive_midprice_delta_log', np.float64),
    
    # ========================================================================
    # Delta Calculations - Passive Normalized/CDF Deltas
    # ========================================================================
    ('delta_passive_midprice_delta_normalized', np.float64),
    ('delta_passive_midprice_delta_cdf', np.float64),
    
    # ========================================================================
    # Delta Calculations - Passive Volume Deltas
    # ========================================================================
    ('delta_passive_buy_volume', np.float64),
    ('delta_passive_buy_volume_pct', np.float64),
    ('delta_passive_buy_volume_log', np.float64),
    ('delta_passive_sell_volume', np.float64),
    ('delta_passive_sell_volume_pct', np.float64),
    ('delta_passive_sell_volume_log', np.float64),
    
    # ========================================================================
    # Delta Calculations - Passive Imbalance Deltas
    # ========================================================================
    ('delta_passive_imbalance_signed', np.float64),
    ('delta_passive_imbalance_abs', np.float64),
    ('delta_passive_imbalance_signed_ratio', np.float64),
    ('delta_passive_imbalance_abs_ratio', np.float64),
    ('delta_passive_imbalance_buy_ratio', np.float64),
    
    # ========================================================================
    # Delta Calculations - Passive Link Function Transform Deltas
    # ========================================================================
    ('delta_passive_imbalance_signed_ratio_atanh', np.float64),
    ('delta_passive_imbalance_buy_ratio_logit', np.float64),
    ('delta_passive_imbalance_abs_ratio_logit', np.float64),
    
    # ========================================================================
    # Delta Calculations - Temporal Deltas
    # ========================================================================
    ('delta_time_elapsed_ns', np.int64),
    ('delta_time_elapsed_ns_pct', np.float64),
    ('delta_time_elapsed_ns_log', np.float64),
    ('delta_pace_of_contracts_traded', np.float64),
    ('delta_pace_of_contracts_traded_pct', np.float64),
    ('delta_pace_of_contracts_traded_log', np.float64),
    
    # ========================================================================
    # Delta Calculations - Temporal Log Base Deltas (delta of base log field)
    # ========================================================================
    ('delta_time_elapsed_ns_log_base', np.float64),
    ('delta_pace_of_contracts_traded_log_base', np.float64),
    
    # ========================================================================
    # Delta Calculations - Divergence Deltas
    # ========================================================================
    ('delta_divergence_buy_volume', np.float64),
    ('delta_divergence_sell_volume', np.float64),
    ('delta_divergence_imbalance_signed', np.float64),
    ('delta_divergence_imbalance_signed_ratio', np.float64),
    ('delta_divergence_buy_ratio', np.float64),
])




BUCKET_DTYPE = np.dtype([

    # ========================================================================
    # Section 2: Meta / Structural
    # ========================================================================
    ('id', np.uint32),
    ('bucket_type', 'U16'),
    ('num_bars', np.uint32),
    ('all_bars_complete', np.bool_),
    ('bar_volume_size', np.uint32),
    ('contract_roll_any', np.bool_),
    ('contract_roll_count', np.uint32),
    ('latest_instrument_id', np.uint32),
    ('start_ts_ns', np.uint64),
    ('end_ts_ns', np.uint64),
    ('time_elapsed_ns_total', np.uint64),
    ('gap_return_any', np.bool_),
    ('gap_return_count', np.uint32),
    ('contains_oversized_order_any', np.bool_),
    ('contains_oversized_order_count', np.uint32),
    
    # ========================================================================
    # Section 3: Order & Volume Statistics (Total/Ambiguous)
    # ========================================================================
    
    # Volume metrics
    ('volume_total_sum', np.uint64),
    ('volume_total_mean', np.float64),
    ('volume_total_std', np.float64),
    
    # Order count metrics
    ('order_count_sum', np.uint64),
    ('order_count_mean', np.float64),
    ('order_count_std', np.float64),
    
    # Order splits
    ('order_splits_sum', np.uint64),
    ('order_splits_mean', np.float64),
    
    # Volume log deltas
    ('delta_volume_total_log_mean', np.float64),
    ('delta_volume_total_log_std', np.float64),
    ('delta_volume_total_log_max', np.float64),
    
    # Order count log deltas
    ('delta_order_count_log_mean', np.float64),
    ('delta_order_count_log_std', np.float64),
    ('delta_order_count_log_max', np.float64),
    
    # ========================================================================
    # Section 4: Active Metrics
    # ========================================================================
    
    # ------------------------------------------------------------------------
    # 4.1 Active Order Counts
    # ------------------------------------------------------------------------
    ('active_order_count_buy_sum', np.uint64),
    ('active_order_count_buy_mean', np.float64),
    ('active_order_count_buy_std', np.float64),
    ('active_order_count_sell_sum', np.uint64),
    ('active_order_count_sell_mean', np.float64),
    ('active_order_count_sell_std', np.float64),
    ('active_order_count_none_sum', np.uint64),
    ('active_order_count_none_mean', np.float64),
    ('active_order_count_none_std', np.float64),
    
    # ------------------------------------------------------------------------
    # 4.2 Active Volumes
    # ------------------------------------------------------------------------
    ('active_volume_buy_sum', np.uint64),
    ('active_volume_buy_mean', np.float64),
    ('active_volume_buy_std', np.float64),
    ('active_volume_sell_sum', np.uint64),
    ('active_volume_sell_mean', np.float64),
    ('active_volume_sell_std', np.float64),
    ('active_volume_none_sum', np.uint64),
    ('active_volume_none_mean', np.float64),
    ('active_volume_none_std', np.float64),
    
    # ------------------------------------------------------------------------
    # 4.3 Active Volume Log Deltas
    # ------------------------------------------------------------------------
    ('delta_active_volume_buy_log_mean', np.float64),
    ('delta_active_volume_buy_log_std', np.float64),
    ('delta_active_volume_buy_log_max', np.float64),
    ('delta_active_volume_sell_log_mean', np.float64),
    ('delta_active_volume_sell_log_std', np.float64),
    ('delta_active_volume_sell_log_max', np.float64),
    ('delta_active_volume_none_log_mean', np.float64),
    ('delta_active_volume_none_log_std', np.float64),
    ('delta_active_volume_none_log_max', np.float64),
    
    # ------------------------------------------------------------------------
    # 4.4 Active Order Count Log Deltas
    # ------------------------------------------------------------------------
    ('delta_active_order_count_buy_log_mean', np.float64),
    ('delta_active_order_count_buy_log_std', np.float64),
    ('delta_active_order_count_buy_log_max', np.float64),
    ('delta_active_order_count_sell_log_mean', np.float64),
    ('delta_active_order_count_sell_log_std', np.float64),
    ('delta_active_order_count_sell_log_max', np.float64),
    ('delta_active_order_count_none_log_mean', np.float64),
    ('delta_active_order_count_none_log_std', np.float64),
    ('delta_active_order_count_none_log_max', np.float64),
    
    # ------------------------------------------------------------------------
    # 4.5 Active Imbalance Metrics
    # ------------------------------------------------------------------------
    ('active_imbalance_signed', np.int64),
    ('active_imbalance_abs', np.uint64),
    ('active_imbalance_signed_ratio', np.float64),
    ('active_imbalance_abs_ratio', np.float64),
    ('active_imbalance_buy_ratio', np.float64),
    ('active_imbalance_signed_ratio_std', np.float64),
    ('active_imbalance_abs_ratio_std', np.float64),
    ('active_imbalance_buy_ratio_std', np.float64),
    
    # ------------------------------------------------------------------------
    # 4.6 Active Link Function Transforms
    # ------------------------------------------------------------------------
    ('active_imbalance_signed_ratio_atanh', np.float64),
    ('active_imbalance_signed_ratio_atanh_std', np.float64),
    ('active_imbalance_buy_ratio_logit', np.float64),
    ('active_imbalance_buy_ratio_logit_std', np.float64),
    ('active_imbalance_abs_ratio_logit', np.float64),
    ('active_imbalance_abs_ratio_logit_std', np.float64),
    
    # ------------------------------------------------------------------------
    # 4.7 Active Imbalance Deltas
    # ------------------------------------------------------------------------
    ('delta_active_imbalance_signed_ratio_mean', np.float64),
    ('delta_active_imbalance_signed_ratio_std', np.float64),
    ('delta_active_imbalance_buy_ratio_mean', np.float64),
    ('delta_active_imbalance_buy_ratio_std', np.float64),
    
    # ------------------------------------------------------------------------
    # 4.8 Active Derived Imbalance Metrics
    # ------------------------------------------------------------------------
    ('active_cumulative_signed_imbalance', np.int64),
    ('active_imbalance_persistence', np.float64),
    ('active_imbalance_volatility', np.float64),
    
    # ------------------------------------------------------------------------
    # 4.9 Active Price Metrics (VWAP)
    # ------------------------------------------------------------------------
    ('active_buy_vwap', np.float64),
    ('active_sell_vwap', np.float64),
    ('active_none_vwap', np.float64),
    ('active_spread_vwap', np.float64),
    ('active_midpoint_vwap', np.float64),
    ('active_midpoint_vwap_std', np.float64),
    ('active_midpoint_vwap_range', np.float64),
    ('active_spread_vwap_std', np.float64),
    ('active_spread_vwap_range', np.float64),
    
    # ------------------------------------------------------------------------
    # 4.10 Active Weighted Midpoints
    # ------------------------------------------------------------------------
    ('active_mid_imbalance_weighted', np.float64),
    ('active_mid_flow_weighted', np.float64),
    ('active_mid_aggressor_weighted', np.float64),
    ('active_mid_imbalance_weighted_std', np.float64),
    ('active_mid_flow_weighted_std', np.float64),
    ('active_mid_aggressor_weighted_std', np.float64),
    
    # ------------------------------------------------------------------------
    # 4.11 Active Price Log Deltas
    # ------------------------------------------------------------------------
    ('delta_active_midpoint_vwap_log_sum', np.float64),
    ('delta_active_midpoint_vwap_log_mean', np.float64),
    ('delta_active_midpoint_vwap_log_std', np.float64),
    ('delta_active_midpoint_vwap_log_skew', np.float64),
    ('delta_active_spread_vwap_log_mean', np.float64),
    ('delta_active_spread_vwap_log_std', np.float64),
    ('delta_active_buy_vwap_log_mean', np.float64),
    ('delta_active_buy_vwap_log_std', np.float64),
    ('delta_active_sell_vwap_log_mean', np.float64),
    ('delta_active_sell_vwap_log_std', np.float64),
    
    # ------------------------------------------------------------------------
    # 4.12 Active Weighted Midpoint Log Deltas
    # ------------------------------------------------------------------------
    ('delta_active_mid_imbalance_weighted_log_sum', np.float64),
    ('delta_active_mid_imbalance_weighted_log_mean', np.float64),
    ('delta_active_mid_imbalance_weighted_log_std', np.float64),
    ('delta_active_mid_imbalance_weighted_log_skew', np.float64),
    ('delta_active_mid_flow_weighted_log_sum', np.float64),
    ('delta_active_mid_flow_weighted_log_mean', np.float64),
    ('delta_active_mid_flow_weighted_log_std', np.float64),
    ('delta_active_mid_flow_weighted_log_skew', np.float64),
    ('delta_active_mid_aggressor_weighted_log_sum', np.float64),
    ('delta_active_mid_aggressor_weighted_log_mean', np.float64),
    ('delta_active_mid_aggressor_weighted_log_std', np.float64),
    ('delta_active_mid_aggressor_weighted_log_skew', np.float64),
    
    # ------------------------------------------------------------------------
    # 4.13 Active Price Range Metrics
    # ------------------------------------------------------------------------
    # Bucket-level extremes
    ('active_buy_price_min', np.float64),
    ('active_buy_price_max', np.float64),
    ('active_buy_price_range', np.float64),
    ('active_sell_price_min', np.float64),
    ('active_sell_price_max', np.float64),
    ('active_sell_price_range', np.float64),
    ('active_none_price_min', np.float64),
    ('active_none_price_max', np.float64),
    ('active_none_price_range', np.float64),
    # Bar-level range statistics
    ('active_buy_price_range_mean', np.float64),
    ('active_buy_price_range_std', np.float64),
    ('active_sell_price_range_mean', np.float64),
    ('active_sell_price_range_std', np.float64),
    ('active_none_price_range_mean', np.float64),
    ('active_none_price_range_std', np.float64),
    
    # ------------------------------------------------------------------------
    # 4.14 Active Pace Metrics
    # ------------------------------------------------------------------------
    # Bucket-level pace
    ('active_buy_pace', np.float64),
    ('active_sell_pace', np.float64),
    # Bar-level statistics
    ('active_buy_pace_mean', np.float64),
    ('active_buy_pace_std', np.float64),
    ('active_sell_pace_mean', np.float64),
    ('active_sell_pace_std', np.float64),
    # Log-transformed pace deltas
    ('delta_active_buy_pace_log_mean', np.float64),
    ('delta_active_buy_pace_log_std', np.float64),
    ('delta_active_sell_pace_log_mean', np.float64),
    ('delta_active_sell_pace_log_std', np.float64),
    
    # ------------------------------------------------------------------------
    # 4.15 Active N-Side Inferred Aggregations
    # ------------------------------------------------------------------------
    ('active_none_inferred_buy_volume_sum', np.float64),
    ('active_none_inferred_sell_volume_sum', np.float64),
    ('active_none_inferred_buy_vwap', np.float64),
    ('active_none_inferred_sell_vwap', np.float64),
    
    # ========================================================================
    # Section 5: Adjusted Metrics
    # ========================================================================
    
    # ------------------------------------------------------------------------
    # 5.1 Adjusted Volumes
    # ------------------------------------------------------------------------
    ('adjusted_volume_buy_sum', np.float64),
    ('adjusted_volume_sell_sum', np.float64),
    ('adjusted_volume_buy_mean', np.float64),
    ('adjusted_volume_buy_std', np.float64),
    ('adjusted_volume_sell_mean', np.float64),
    ('adjusted_volume_sell_std', np.float64),
    
    # ------------------------------------------------------------------------
    # 5.2 Adjusted Imbalance Metrics
    # ------------------------------------------------------------------------
    ('adjusted_imbalance_signed', np.float64),
    ('adjusted_imbalance_abs', np.float64),
    ('adjusted_imbalance_signed_ratio', np.float64),
    ('adjusted_imbalance_abs_ratio', np.float64),
    ('adjusted_imbalance_buy_ratio', np.float64),
    ('adjusted_imbalance_signed_ratio_std', np.float64),
    ('adjusted_imbalance_abs_ratio_std', np.float64),
    ('adjusted_imbalance_buy_ratio_std', np.float64),
    
    # ------------------------------------------------------------------------
    # 5.3 Adjusted Link Function Transforms
    # ------------------------------------------------------------------------
    ('adjusted_imbalance_signed_ratio_atanh', np.float64),
    ('adjusted_imbalance_signed_ratio_atanh_std', np.float64),
    ('adjusted_imbalance_buy_ratio_logit', np.float64),
    ('adjusted_imbalance_buy_ratio_logit_std', np.float64),
    ('adjusted_imbalance_abs_ratio_logit', np.float64),
    ('adjusted_imbalance_abs_ratio_logit_std', np.float64),
    
    # ------------------------------------------------------------------------
    # 5.4 Adjusted Imbalance Deltas
    # ------------------------------------------------------------------------
    ('delta_adjusted_imbalance_signed_ratio_mean', np.float64),
    ('delta_adjusted_imbalance_signed_ratio_std', np.float64),
    ('delta_adjusted_imbalance_buy_ratio_mean', np.float64),
    ('delta_adjusted_imbalance_buy_ratio_std', np.float64),
    ('delta_adjusted_imbalance_signed_ratio_skew', np.float64),
    
    # ------------------------------------------------------------------------
    # 5.5 Adjusted Derived Imbalance Metrics
    # ------------------------------------------------------------------------
    ('adjusted_cumulative_signed_imbalance', np.float64),
    ('adjusted_imbalance_persistence', np.float64),
    ('adjusted_imbalance_volatility', np.float64),
    
    # ------------------------------------------------------------------------
    # 5.6 Adjusted Price Metrics (VWAP)
    # ------------------------------------------------------------------------
    ('adjusted_buy_vwap', np.float64),
    ('adjusted_sell_vwap', np.float64),
    ('adjusted_spread_vwap', np.float64),
    ('adjusted_midpoint_vwap', np.float64),
    ('adjusted_midpoint_vwap_std', np.float64),
    ('adjusted_midpoint_vwap_range', np.float64),
    ('adjusted_spread_vwap_std', np.float64),
    ('adjusted_spread_vwap_range', np.float64),
    
    # ------------------------------------------------------------------------
    # 5.7 Adjusted Weighted Midpoints
    # ------------------------------------------------------------------------
    ('adjusted_mid_imbalance_weighted', np.float64),
    ('adjusted_mid_flow_weighted', np.float64),
    ('adjusted_mid_aggressor_weighted', np.float64),
    ('adjusted_mid_imbalance_weighted_std', np.float64),
    ('adjusted_mid_flow_weighted_std', np.float64),
    ('adjusted_mid_aggressor_weighted_std', np.float64),
    
    # ------------------------------------------------------------------------
    # 5.8 Adjusted Price Log Deltas
    # ------------------------------------------------------------------------
    ('delta_adjusted_midpoint_vwap_log_sum', np.float64),
    ('delta_adjusted_midpoint_vwap_log_mean', np.float64),
    ('delta_adjusted_midpoint_vwap_log_std', np.float64),
    ('delta_adjusted_midpoint_vwap_log_skew', np.float64),
    ('delta_adjusted_spread_vwap_log_mean', np.float64),
    ('delta_adjusted_spread_vwap_log_std', np.float64),
    ('delta_adjusted_buy_vwap_log_mean', np.float64),
    ('delta_adjusted_buy_vwap_log_std', np.float64),
    ('delta_adjusted_sell_vwap_log_mean', np.float64),
    ('delta_adjusted_sell_vwap_log_std', np.float64),
    
    # ------------------------------------------------------------------------
    # 5.9 Adjusted Weighted Midpoint Log Deltas
    # ------------------------------------------------------------------------
    ('delta_adjusted_mid_imbalance_weighted_log_sum', np.float64),
    ('delta_adjusted_mid_imbalance_weighted_log_mean', np.float64),
    ('delta_adjusted_mid_imbalance_weighted_log_std', np.float64),
    ('delta_adjusted_mid_imbalance_weighted_log_skew', np.float64),
    ('delta_adjusted_mid_flow_weighted_log_sum', np.float64),
    ('delta_adjusted_mid_flow_weighted_log_mean', np.float64),
    ('delta_adjusted_mid_flow_weighted_log_std', np.float64),
    ('delta_adjusted_mid_flow_weighted_log_skew', np.float64),
    ('delta_adjusted_mid_aggressor_weighted_log_sum', np.float64),
    ('delta_adjusted_mid_aggressor_weighted_log_mean', np.float64),
    ('delta_adjusted_mid_aggressor_weighted_log_std', np.float64),
    ('delta_adjusted_mid_aggressor_weighted_log_skew', np.float64),
    
    # ========================================================================
    # Section 6: Passive Metrics
    # ========================================================================
    
    # ------------------------------------------------------------------------
    # 6.1 Passive Midprice
    # ------------------------------------------------------------------------
    ('passive_midprice', np.float64),
    ('passive_midprice_std', np.float64),
    ('passive_midprice_range', np.float64),
    
    # ------------------------------------------------------------------------
    # 6.2 Passive Midprice Deltas
    # ------------------------------------------------------------------------
    ('delta_passive_midprice_log_sum', np.float64),
    ('delta_passive_midprice_log_mean', np.float64),
    ('delta_passive_midprice_log_std', np.float64),
    ('delta_passive_midprice_log_skew', np.float64),
    
    # ------------------------------------------------------------------------
    # 6.3 Passive CDF Statistics
    # ------------------------------------------------------------------------
    ('passive_midprice_delta_cdf_mean', np.float64),
    ('passive_midprice_delta_cdf_std', np.float64),
    
    # ------------------------------------------------------------------------
    # 6.4 Passive Volume Classifications
    # ------------------------------------------------------------------------
    ('passive_buy_volume_sum', np.float64),
    ('passive_sell_volume_sum', np.float64),
    ('passive_buy_volume_mean', np.float64),
    ('passive_buy_volume_std', np.float64),
    ('passive_sell_volume_mean', np.float64),
    ('passive_sell_volume_std', np.float64),
    
    # ------------------------------------------------------------------------
    # 6.5 Passive Imbalance Metrics
    # ------------------------------------------------------------------------
    ('passive_imbalance_signed', np.float64),
    ('passive_imbalance_abs', np.float64),
    ('passive_imbalance_signed_ratio', np.float64),
    ('passive_imbalance_abs_ratio', np.float64),
    ('passive_imbalance_buy_ratio', np.float64),
    ('passive_imbalance_signed_ratio_std', np.float64),
    ('passive_imbalance_abs_ratio_std', np.float64),
    ('passive_imbalance_buy_ratio_std', np.float64),
    
    # ------------------------------------------------------------------------
    # 6.6 Passive Link Function Transforms
    # ------------------------------------------------------------------------
    ('passive_imbalance_signed_ratio_atanh', np.float64),
    ('passive_imbalance_signed_ratio_atanh_std', np.float64),
    ('passive_imbalance_buy_ratio_logit', np.float64),
    ('passive_imbalance_buy_ratio_logit_std', np.float64),
    ('passive_imbalance_abs_ratio_logit', np.float64),
    ('passive_imbalance_abs_ratio_logit_std', np.float64),
    
    # ------------------------------------------------------------------------
    # 6.7 Passive Imbalance Deltas
    # ------------------------------------------------------------------------
    ('delta_passive_imbalance_signed_ratio_mean', np.float64),
    ('delta_passive_imbalance_signed_ratio_std', np.float64),
    ('delta_passive_imbalance_signed_ratio_skew', np.float64),
    ('delta_passive_imbalance_buy_ratio_mean', np.float64),
    ('delta_passive_imbalance_buy_ratio_std', np.float64),
    
    # ------------------------------------------------------------------------
    # 6.8 Passive Derived Imbalance Metrics
    # ------------------------------------------------------------------------
    ('passive_cumulative_signed_imbalance', np.float64),
    ('passive_imbalance_persistence', np.float64),
    ('passive_imbalance_volatility', np.float64),
    
    # ========================================================================
    # Section 7: Divergence Metrics
    # ========================================================================
    
    # ------------------------------------------------------------------------
    # 7.1 Aggregated Bar-Level Divergences
    # ------------------------------------------------------------------------
    ('divergence_buy_volume_mean', np.float64),
    ('divergence_buy_volume_std', np.float64),
    ('divergence_sell_volume_mean', np.float64),
    ('divergence_sell_volume_std', np.float64),
    ('divergence_imbalance_signed_mean', np.float64),
    ('divergence_imbalance_signed_std', np.float64),
    ('divergence_imbalance_signed_ratio_mean', np.float64),
    ('divergence_imbalance_signed_ratio_std', np.float64),
    ('divergence_buy_ratio_mean', np.float64),
    ('divergence_buy_ratio_std', np.float64),
    
    # ------------------------------------------------------------------------
    # 7.2 Bucket-Level Divergences
    # ------------------------------------------------------------------------
    ('divergence_buy_volume_bucket', np.float64),
    ('divergence_sell_volume_bucket', np.float64),
    ('divergence_imbalance_signed_bucket', np.float64),
    ('divergence_imbalance_signed_ratio_bucket', np.float64),
    ('divergence_buy_ratio_bucket', np.float64),
    
    # ------------------------------------------------------------------------
    # 7.3 Divergence Deltas
    # ------------------------------------------------------------------------
    ('delta_divergence_buy_volume_mean', np.float64),
    ('delta_divergence_sell_volume_mean', np.float64),
    ('delta_divergence_imbalance_signed_ratio_mean', np.float64),
    ('delta_divergence_buy_ratio_mean', np.float64),
    
    # ========================================================================
    # Section 8: Temporal / Tempo Structure
    # ========================================================================
    
    # ------------------------------------------------------------------------
    # 8.1 Time Elapsed Metrics
    # ------------------------------------------------------------------------
    ('time_elapsed_ns_mean', np.float64),
    ('time_elapsed_ns_std', np.float64),
    ('time_elapsed_ns_min', np.uint64),
    ('time_elapsed_ns_max', np.uint64),
    
    # ------------------------------------------------------------------------
    # 8.2 Pace of Contracts Traded
    # ------------------------------------------------------------------------
    ('pace_of_contracts_traded', np.float64),
    ('pace_of_contracts_traded_mean', np.float64),
    ('pace_of_contracts_traded_std', np.float64),
    
    # ------------------------------------------------------------------------
    # 8.3 Temporal Log Deltas
    # ------------------------------------------------------------------------
    ('delta_time_elapsed_ns_log_mean', np.float64),
    ('delta_time_elapsed_ns_log_std', np.float64),
    ('delta_pace_of_contracts_traded_log_mean', np.float64),
    ('delta_pace_of_contracts_traded_log_std', np.float64),
    
    # ------------------------------------------------------------------------
    # 8.4 Derived Tempo Metrics
    # ------------------------------------------------------------------------
    ('tempo_stability', np.float64),
    ('tempo_acceleration', np.float64),
    
    # ========================================================================
    # Section 9: Directional Signals
    # ========================================================================
    
    # ------------------------------------------------------------------------
    # 9.1 Active Directional Signals
    # ------------------------------------------------------------------------
    ('active_pct_bars_positive_direction', np.float64),
    ('active_directional_streak_max', np.uint32),
    ('active_directional_reversals_count', np.uint32),
    ('active_net_direction', np.float64),
    
    # ------------------------------------------------------------------------
    # 9.2 Adjusted Directional Signals
    # ------------------------------------------------------------------------
    ('adjusted_pct_bars_positive_direction', np.float64),
    ('adjusted_directional_streak_max', np.uint32),
    ('adjusted_directional_reversals_count', np.uint32),
    ('adjusted_net_direction', np.float64),
    
    # ------------------------------------------------------------------------
    # 9.3 Passive Directional Signals
    # ------------------------------------------------------------------------
    ('passive_pct_bars_positive_direction', np.float64),
    ('passive_directional_streak_max', np.uint32),
    ('passive_directional_reversals_count', np.uint32),
    ('passive_net_direction', np.float64),
    
    # ========================================================================
    # Section 10: Volatility and Stability Overlays
    # ========================================================================
    ('active_price_volatility', np.float64),
    ('active_spread_volatility', np.float64),
    ('active_return_volatility', np.float64),
    ('active_intermediate_volatility', np.float64),
    ('active_spread_stability', np.float64),
    ('active_flow_volatility', np.float64),
    ('adjusted_price_volatility', np.float64),
    ('adjusted_spread_volatility', np.float64),
    ('passive_price_volatility', np.float64),
    
    # ========================================================================
    # Section 11: Inter-Bucket Deltas (Regime Transitions)
    # ========================================================================
    
    # ------------------------------------------------------------------------
    # 11.1 Active Inter-Bucket Deltas
    # ------------------------------------------------------------------------
    ('bucket_delta_active_spread_vwap_log', np.float64),
    ('bucket_delta_active_spread_vwap_std_log', np.float64),
    ('bucket_delta_active_spread_volatility', np.float64),
    ('bucket_delta_active_return_volatility_log', np.float64),
    ('bucket_delta_active_intermediate_volatility_log', np.float64),
    ('bucket_delta_active_flow_volatility_log', np.float64),
    ('bucket_delta_active_pace_of_contracts_traded_log', np.float64),
    ('bucket_delta_active_tempo_stability', np.float64),
    ('bucket_delta_active_time_elapsed_ns_total_log', np.float64),
    ('bucket_delta_active_imbalance_signed_ratio_atanh', np.float64),
    ('bucket_delta_active_imbalance_abs_ratio_logit', np.float64),
    ('bucket_delta_active_imbalance_persistence', np.float64),
    ('bucket_delta_active_cumulative_signed_imbalance_norm', np.float64),
    ('bucket_delta_active_imbalance_volatility', np.float64),
    ('bucket_delta_active_pct_bars_positive_direction', np.float64),
    ('bucket_delta_active_directional_streak_max', np.int32),
    ('bucket_delta_active_directional_reversals_count', np.int32),
    
    # ------------------------------------------------------------------------
    # 11.2 Adjusted Inter-Bucket Deltas
    # ------------------------------------------------------------------------
    ('bucket_delta_adjusted_spread_vwap_log', np.float64),
    ('bucket_delta_adjusted_spread_volatility', np.float64),
    ('bucket_delta_adjusted_return_volatility_log', np.float64),
    ('bucket_delta_adjusted_imbalance_signed_ratio_atanh', np.float64),
    ('bucket_delta_adjusted_imbalance_abs_ratio_logit', np.float64),
    ('bucket_delta_adjusted_imbalance_persistence', np.float64),
    ('bucket_delta_adjusted_imbalance_volatility', np.float64),
    
    # ------------------------------------------------------------------------
    # 11.3 Second-Order Deltas (Acceleration)
    # ------------------------------------------------------------------------
    ('bucket_delta2_active_return_volatility_log', np.float64),
    ('bucket_delta2_active_imbalance_signed_ratio_atanh', np.float64),
    ('bucket_delta2_active_pace_of_contracts_traded_log', np.float64),
    ('bucket_delta2_adjusted_return_volatility_log', np.float64),
    ('bucket_delta2_adjusted_imbalance_signed_ratio_atanh', np.float64),
    
    # ========================================================================
    # Section 12: Debug / Diagnostic
    # ========================================================================
    ('bar_ids_checksum', np.uint64),
    ('missing_bars_flag', np.bool_),
    ('processing_time_ns', np.uint64),
    ('nan_count_active_midpoint', np.uint32),
    ('nan_count_adjusted_midpoint', np.uint32),
    ('nan_count_passive_midprice', np.uint32),

])




# Placeholder for metrics dtype
# TODO: Define METRICS_DTYPE based on requirements
METRICS_DTYPE = np.dtype([
    ('id', np.uint32),
    ('bucket_type', 'U16'),
    # TODO: Add metric fields
])




# Datatypes
class BufferName:
    VOLUMEBAR = 'volumebar'
    BUCKET = 'bucket'
    METRIC = 'metric'
    ALL = [VOLUMEBAR, BUCKET, METRIC]

class ChunkType:
    RAW = 'raw'
    FIXED = 'fixed'
    INTERVAL = 'interval'
    ALL = [RAW, FIXED, INTERVAL]




# ============================================================================
# CIRCULAR BUFFER FOR STRUCTURED ARRAYS
# ============================================================================

class StructuredCircularBuffer:
    """
    Circular buffer for structured numpy arrays.
    Preserves all fields with correct dtypes.
    
    Storage is a 1D structured array where each element is a complete record
    with named fields (not a 2D array).
    """
    
    def __init__(self, lookback_window: int, dtype: np.dtype):
        """
        Initialize circular buffer.
        
        Args:
            lookback_window: Number of records to maintain (e.g., 50)
            dtype: Structured array dtype (e.g., VOLUMEBAR_DTYPE)
        """
        self.lookback = lookback_window
        self.dtype = dtype
        self.data = np.empty(lookback_window, dtype=dtype)
        self.count = 0
    



    def add_row(self, data: Union[Dict[str, Any], np.ndarray]) -> None:
        """
        Add data to buffer (accepts dict or structured numpy array).
        
        Args:
            data: Either a dictionary with keys matching dtype fields,
                or a structured numpy array (single row) with matching dtype.
                Values should already be in correct numpy dtypes.
        """
        idx = self.count % self.lookback
        
        # Handle numpy structured array
        if isinstance(data, np.ndarray):
            if data.dtype != self.dtype:
                raise ValueError(
                    f"Array dtype mismatch: expected {self.dtype}, got {data.dtype}"
                )
            # Direct assignment of structured array row
            self.data[idx] = data
        
        # Handle dictionary
        elif isinstance(data, dict):
            # Field-by-field assignment from dict
            if self.dtype.names is not None:
                for field_name in self.dtype.names:
                    if field_name in data:
                        self.data[idx][field_name] = data[field_name]
        
        else:
            raise TypeError(
                f"data must be Dict or np.ndarray, got {type(data).__name__}"
            )
        
        self.count += 1
    



    def get_previous_row_view(self) -> Optional[np.ndarray]:
        """
        Return view of previous bar's statistics.
        
        Returns:
            Structured numpy record (single row with all fields)
            or None if buffer empty.
        """
        if self.count == 0:
            return None
        
        prev_idx = (self.count - 1) % self.lookback
        return self.data[prev_idx]
    


    def get_last_n_rows(self, n: int) -> Optional[np.ndarray]: # (!!!) Not tested
        """
        Get the last N rows from the buffer.
        
        Args:
            n: Number of rows to retrieve
        
        Returns:
            Structured numpy array with last N rows, or None if insufficient data.
            Returns None if n > count (not enough data available).
            Rows are returned in chronological order (oldest to newest).
        """

        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")
        
        if n > self.count:
            # Not enough data available
            return None
        
        if n > self.lookback:
            raise ValueError(
                f"Cannot retrieve {n} rows: exceeds lookback window ({self.lookback})"
            )
        
        # If buffer not yet full (count < lookback)
        if self.count < self.lookback:
            # Data is stored sequentially from index 0
            start_idx = self.count - n
            end_idx = self.count
            return self.data[start_idx:end_idx].copy()
        
        # Else
        # Buffer is full (circular wrapping)
        current_idx = self.count % self.lookback
        
        # Calculate starting position for last N rows
        start_idx = (current_idx - n) % self.lookback
        
        # Two cases: data wraps around or does not wrap
        if start_idx < current_idx:
            # No wrap: continuous slice
            return self.data[start_idx:current_idx].copy()
        else:
            # Wrap: concatenate two slices
            # From start_idx to end, then from 0 to current_idx
            return np.concatenate([
                self.data[start_idx:],
                self.data[:current_idx]
            ])
    



    def is_warm(self) -> bool:
        """
        Check if buffer has enough data for calculations.
        
        Returns:
            True if buffer has at least lookback_window records.
        """
        return self.count >= self.lookback
    



    
    def get_field(self, field_name: str) -> np.ndarray:
        """
        Get all values for a specific field.
        
        Args:
            field_name: Name of field (e.g., 'volume_total')
        
        Returns:
            1D array of values for that field across all rows.
        """
        return self.data[field_name]




# ============================================================================
# ARCHIVAL TASK
# ============================================================================

@dataclass
class ArchivalTask:
    """
    Data package for archival operations.
    
    Contains all information needed to write chunk to Parquet.
    """
    buffer_name: str              # 'volumebar', 'buckets', 'metrics'
    chunk_type: str               # 'raw', 'fixed', 'interval'
    chunk_id: int                 # Unique chunk identifier
    data: np.ndarray              # Structured array to archive
    metadata: Dict[str, Any]      # Start/end rows, counts, etc.
    processed_timestamp_ns: Optional[int]   # Optional debug timestamp (milliseconds)
    
    def __post_init__(self):
        """Validate task data."""
        if not isinstance(self.data, np.ndarray):
            raise TypeError(f"data must be numpy array, got {type(self.data)}")
        # Check if structured array by checking for named fields
        if not hasattr(self.data.dtype, 'names') or self.data.dtype.names is None:
            raise TypeError("data must be structured array with named fields")




# ============================================================================
# Distribution Containers
# ============================================================================


class DeltasDistribution:
    """
    Empirical distribution summary for price changes (ΔP).
    
    Computes a single global standard deviation from the sample data.
    This σ is:
        - Empirical: computed directly from data, no distributional assumptions
        - Unconditional: single value across all observations
        - Global: represents the overall scale/spread of price changes
    
    Primary use: normalizing raw price changes into z-scores.
        z_i = ΔP_i / σ
    
    These z-scores can then be passed to ClassifierDistribution.cdf()
    to obtain probability-like values in [0, 1].
    """
    
    __slots__ = ('_std',)  # Prevent __dict__ allocation for memory efficiency
    
    def __init__(self, deltas: np.ndarray):

        """
        Parameters
        ----------
        deltas : np.ndarray
            1D array of price changes ΔP_i.
        
        Process
        -------
        1. Cast input to float64 for numerical precision
        2. Remove NaN values if present, otherwise use original array
        3. Compute population standard deviation:
           
           σ = sqrt( (1/n) * Σ(x_i - x̄)² )
           
           Using ddof=0 (population std) rather than ddof=1 (sample std)
           because we treat the observed data as the full population
           for scaling purposes, not as a sample estimating a larger population.
        """

        values = np.asarray(deltas, dtype=np.float64)
        mask = np.isnan(values)
        
        # Only filter if NaNs are present; avoids unnecessary copy
        if mask.any():
            values = values[~mask]
        
        if values.size == 0:
            raise ValueError("Requires at least one value.")
        
        self._std = float(values.std(ddof=0))
    



    @property
    def std(self) -> float:
        """
        Global empirical standard deviation σ.
        
        Returns
        -------
        float
            Population standard deviation of the input price changes.
        """
        return self._std




    def std(self) -> float:
        """
        Global empirical standard deviation σ_ΔP.

        Intended usage:
            z_i = ΔP_i / returns_distribution.std()
        """
        return self._std




class NormalCDF:
    """
    Maps normalized returns (z-scores) to probabilities via the standard normal CDF.
    
    This is a nonlinear S-curve transformation, NOT an assumption that returns
    are normally distributed. The normal CDF is used purely as a sigmoid-like
    function that:
        - Maps (-∞, +∞) → (0, 1)
        - Is symmetric around z=0 → 0.5
        - Has smooth, monotonic behavior
        - Compresses extreme values toward 0 and 1
    
    Interpretation of output:
        - Φ(z) ≈ 0.0: extreme negative move (far left tail)
        - Φ(z) = 0.5: exactly at the mean (z = 0)
        - Φ(z) ≈ 1.0: extreme positive move (far right tail)
    
    Reference values:
        z = -3  →  Φ(z) = 0.0013  (bottom 0.13%)
        z = -2  →  Φ(z) = 0.0228  (bottom 2.3%)
        z = -1  →  Φ(z) = 0.1587  (bottom 15.9%)
        z =  0  →  Φ(z) = 0.5000  (median)
        z = +1  →  Φ(z) = 0.8413  (top 15.9%)
        z = +2  →  Φ(z) = 0.9772  (top 2.3%)
        z = +3  →  Φ(z) = 0.9987  (top 0.13%)
    
    Latency: ~0.1-0.5µs per query (direct formula, no lookup)
    Memory: Zero (stateless, no fitted parameters)
    """
    
    __slots__ = ()
    
    # Precomputed constant: sqrt(2) ≈ 1.4142135623730951
    # Computed once at class definition, avoids repeated calculation.
    _SQRT2 = np.sqrt(2.0)
    
    def cdf(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Standard normal CDF: Φ(z) = P(Z ≤ z) for Z ~ N(0,1).
        
        Parameters
        ----------
        z : float or np.ndarray
            Normalized values (z-scores). Typically z_i = ΔP_i / σ.
        
        Returns
        -------
        float or np.ndarray
            Probability values in [0, 1].
        
        Mathematical Definition
        -----------------------
        The standard normal CDF is defined as:
        
            Φ(z) = ∫_{-∞}^{z} (1/√(2π)) * exp(-t²/2) dt
        
        This integral has no closed-form solution, but relates to the
        error function (erf) via:
        
            Φ(z) = 0.5 * [1 + erf(z / √2)]
        
        where:
            erf(x) = (2/√π) * ∫_{0}^{x} exp(-t²) dt
        
        erf -> from SciPy
        """
        return 0.5 * (1.0 + erf(np.asarray(z, dtype=np.float64) / self._SQRT2))




class StudentTCDF:
    """
    Maps normalized returns to probabilities via Student's t-distribution CDF.
    
    The t-distribution has heavier tails than the normal, controlled by
    degrees of freedom (df). This makes it more appropriate for financial
    returns which exhibit excess kurtosis (fat tails).
    
    Tail behavior comparison (probability of |z| > 3):
        Normal (df=∞):  0.27%
        t (df=30):      0.54%
        t (df=10):      1.0%
        t (df=5):       2.0%
        t (df=3):       4.3%
    
    As df → ∞, the t-distribution converges to the normal distribution.
    
    Properties:
        - Maps (-∞, +∞) → (0, 1)
        - Symmetric around z=0 → 0.5
        - Heavier tails than normal (more probability mass in extremes)
        - Single parameter: degrees of freedom (df)
    
    Latency: ~1-2µs per query (scipy function call)
    Memory: 8 bytes (one float for df)
    """
    
    __slots__ = ('_df', '_dist')
    
    def __init__(self, df: float = 5.0):
        """
        Parameters
        ----------
        df : float, default=5.0
            Degrees of freedom. Lower values = heavier tails.
            
            Guidance for financial data:
                df=3-5:   Very heavy tails (high-frequency, crisis periods)
                df=5-10:  Moderate tails (typical equity returns)
                df=10-30: Light tails (approaching normal)
                df>30:    Approximately normal
        
        Process
        -------
        1. Store degrees of freedom parameter
        2. Create frozen scipy distribution object for efficient repeated queries
           (avoids parameter validation overhead on each cdf() call)
        """
        if df <= 0:
            raise ValueError("Degrees of freedom must be positive.")
        self._df = float(df)
        self._dist = stats.t(df=self._df)
    
    @property
    def df(self) -> float:
        """Degrees of freedom parameter."""
        return self._df
    
    def cdf(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Student's t CDF: F(z; df) = P(T ≤ z) for T ~ t(df).
        
        Parameters
        ----------
        z : float or np.ndarray
            Normalized values (z-scores). Typically z_i = ΔP_i / σ.
        
        Returns
        -------
        float or np.ndarray
            Probability values in [0, 1].
        
        Mathematical Definition
        -----------------------
        The Student's t CDF is defined as:
        
            F(z; df) = ∫_{-∞}^{z} Γ((df+1)/2) / (√(df·π) · Γ(df/2)) · (1 + t²/df)^(-(df+1)/2) dt
        
        where Γ is the gamma function.
        
        Key differences from normal:
            - PDF decays as |z|^(-(df+1)) vs exp(-z²/2) for normal
            - This polynomial decay produces heavier tails
            - Variance = df/(df-2) for df>2 (undefined for df≤2)
        """
        return self._dist.cdf(np.asarray(z, dtype=np.float64))




class SkewedTCDF:
    """
    Maps normalized returns to probabilities via Hansen's skewed t-distribution CDF.
    
    Extends Student's t with an asymmetry parameter, capturing both:
        - Fat tails (via degrees of freedom)
        - Asymmetry (via skewness parameter)
    
    This is particularly relevant for financial returns which often exhibit:
        - Negative skewness (larger negative returns than positive)
        - Excess kurtosis (fat tails on both sides)
    
    Properties:
        - Maps (-∞, +∞) → (0, 1)
        - Asymmetric when lambda ≠ 0
        - Heavier tails than normal
        - Two parameters: df (tail weight) and lambda (skewness)
    
    Skewness parameter interpretation:
        lambda < 0:  Left-skewed (negative tail heavier, common in equities)
        lambda = 0:  Symmetric (reduces to Student's t)
        lambda > 0:  Right-skewed (positive tail heavier)
    
    Latency: ~2-5µs per query (scipy function call)
    Memory: 16 bytes (two floats)
    """
    
    __slots__ = ('_df', '_lambda', '_dist')
    
    def __init__(self, df: float = 5.0, lam: float = 0.0):
        """
        Parameters
        ----------
        df : float, default=5.0
            Degrees of freedom. Lower values = heavier tails.
            Must be > 0.
            
        lam : float, default=0.0
            Skewness parameter in range (-1, 1).
            Negative = left-skewed, positive = right-skewed.
        
        Process
        -------
        1. Validate parameters (df > 0, -1 < lambda < 1)
        2. Create frozen scipy nct (noncentral t) distribution
           
           Note: scipy.stats.nct uses a different parameterization.
           We use the location-scale transformation to achieve
           Hansen's skewed t behavior.
        """
        if df <= 0:
            raise ValueError("Degrees of freedom must be positive.")
        if not -1 < lam < 1:
            raise ValueError("Skewness parameter lambda must be in (-1, 1).")
        
        self._df = float(df)
        self._lambda = float(lam)
        
        # scipy.stats.skewnorm for skewed normal, but for skewed t
        # we use nct (noncentral t) with nc parameter controlling skewness
        # The relationship: nc ≈ lambda * sqrt(df) for small skewness
        nc = lam * np.sqrt(df) * 0.5  # Approximate mapping to nct
        self._dist = stats.nct(df=self._df, nc=nc)
    
    @property
    def df(self) -> float:
        """Degrees of freedom parameter."""
        return self._df
    
    @property
    def skewness(self) -> float:
        """Skewness parameter lambda."""
        return self._lambda
    
    def cdf(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Skewed t CDF: F(z; df, λ) = P(T ≤ z) for T ~ skewed-t(df, λ).
        
        Parameters
        ----------
        z : float or np.ndarray
            Normalized values (z-scores). Typically z_i = ΔP_i / σ.
        
        Returns
        -------
        float or np.ndarray
            Probability values in [0, 1].
        
        Mathematical Definition
        -----------------------
        Hansen's skewed t PDF is defined piecewise:
        
            f(z; df, λ) = 
                bc · (1 + (bz+a)²/(df(1-λ)²))^(-(df+1)/2)  if z < -a/b
                bc · (1 + (bz+a)²/(df(1+λ)²))^(-(df+1)/2)  if z ≥ -a/b
        
        where:
            a = 4λc · (df-2)/(df-1)
            b² = 1 + 3λ² - a²
            c = Γ((df+1)/2) / (√(π(df-2)) · Γ(df/2))
        
        The CDF is the integral of this PDF. No closed form exists;
        scipy computes it numerically.
        
        Implementation Note
        -------------------
        We approximate Hansen's skewed t using scipy's noncentral t (nct).
        For precise Hansen's skewed t, consider the arch package:
            from arch.univariate import SkewStudent
        """
        return self._dist.cdf(np.asarray(z, dtype=np.float64))




class EmpiricalCDF:
    """
    Maps values to probabilities via the empirical cumulative distribution function.
    
    The empirical CDF makes no distributional assumptions. It directly uses
    the observed data to compute:
    
        F̂(z) = (number of observations ≤ z) / (total observations)
    
    This is the most accurate representation of your actual data distribution,
    but requires storing and sorting the full dataset.
    
    Properties:
        - Maps [min(data), max(data)] → (0, 1]
        - Values below min → 0, values above max → 1
        - Step function (discrete jumps at each data point)
        - Interpolated for smooth queries between observed values
    
    Tradeoffs vs parametric distributions:
        + No distributional assumptions (perfect accuracy for observed data)
        + Captures any shape: multimodal, skewed, heavy-tailed
        - Requires O(n) memory to store sorted data
        - O(n log n) initialization cost for sorting
        - Cannot extrapolate beyond observed range
    
    Latency: ~2-5µs per query (binary search interpolation)
    Memory: O(n) - stores full sorted dataset
    """
    
    __slots__ = ('_sorted_values', '_cdf_values', '_interp_func', '_n')
    
    def __init__(self, values: np.ndarray):
        """
        Parameters
        ----------
        values : np.ndarray
            1D array of observed values to build the empirical distribution.
        
        Process
        -------
        1. Cast to float64 and remove NaNs if present
        2. Sort values in ascending order - O(n log n)
        3. Compute CDF values: F̂(x_i) = i/n for sorted x_i
        4. Build interpolation function for O(log n) queries
           
           Interpolation uses linear interpolation between observed points,
           with bounds_error=False to handle queries outside observed range:
               - Below min(data) → 0
               - Above max(data) → 1
        """
        arr = np.asarray(values, dtype=np.float64)
        mask = np.isnan(arr)
        
        if mask.any():
            arr = arr[~mask]
        
        if arr.size == 0:
            raise ValueError("Requires at least one value.")
        
        self._n = arr.size
        self._sorted_values = np.sort(arr)
        
        # CDF values: P(X ≤ x_i) = i/n for the i-th sorted value
        # Using 1-indexed: first value maps to 1/n, last to n/n = 1
        self._cdf_values = np.arange(1, self._n + 1) / self._n
        
        # Interpolation function for fast queries
        # fill_value=(0, 1): below min → 0, above max → 1
        self._interp_func = interp1d(
            self._sorted_values,
            self._cdf_values,
            kind='linear',
            bounds_error=False,
            fill_value=(0.0, 1.0)
        )
    
    @property
    def n(self) -> int:
        """Number of observations in the empirical distribution."""
        return self._n
    
    def cdf(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Empirical CDF: F̂(z) = (# observations ≤ z) / n.
        
        Parameters
        ----------
        z : float or np.ndarray
            Query values.
        
        Returns
        -------
        float or np.ndarray
            Probability values in [0, 1].
        
        Mathematical Definition
        -----------------------
        The empirical CDF is defined as:
        
            F̂(z) = (1/n) · Σᵢ 𝟙(Xᵢ ≤ z)
        
        where 𝟙 is the indicator function.
        
        This is a step function with jumps of 1/n at each observed value.
        We use linear interpolation between observed points for smooth output.
        
        Implementation
        --------------
        Uses scipy.interpolate.interp1d which performs binary search
        on the sorted values array - O(log n) per scalar query,
        O(k log n) for k query values.
        """
        result = self._interp_func(np.asarray(z, dtype=np.float64))
        
        # interp1d returns array even for scalar input; preserve scalar type
        if np.ndim(z) == 0:
            return float(result)
        return result
    
    def percentile(self, q: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Inverse CDF (quantile function): returns value at given percentile.
        
        Parameters
        ----------
        q : float or np.ndarray
            Percentile(s) in range [0, 100].
        
        Returns
        -------
        float or np.ndarray
            Value(s) at the requested percentile(s).
        """
        return np.percentile(self._sorted_values, q)




# ============================================================================
# HELPER FUNCTIONS
# ============================================================================




class HelperFunctions:


    @staticmethod
    def dict_to_structured_array(data: Dict[str, Any], dtype: np.dtype) -> np.ndarray:
        """
        Convert dictionary to single structured array row.
        
        Args:
            data: Dictionary with field names as keys
            dtype: Target structured array dtype
        
        Returns:
            Single-element structured array
        """
        row = np.empty(1, dtype=dtype)
        if dtype.names is not None:
            for field_name in dtype.names:
                if field_name in data:
                    row[field_name] = data[field_name]
        return row[0]




    @staticmethod
    def get_field_names_by_type(dtype: np.dtype, target_type: type) -> list:
        """
        Get list of field names matching a specific numpy dtype.
        
        Args:
            dtype: Structured array dtype
            target_type: Target numpy type (e.g., np.float64)
        
        Returns:
            List of field names with matching type
        
        Example:
            >>> float_fields = get_field_names_by_type(VOLUMEBAR_DTYPE, np.float64)
            >>> # Returns: ['imbalance_size_abs_ratio', 'price_bid_vwap', ...]
        """
        matching_fields = []
        if dtype.names is not None:
            for name in dtype.names:
                field_dtype = dtype.fields[name][0] # type: ignore
                if field_dtype == target_type:
                    matching_fields.append(name)
        return matching_fields
    



    @staticmethod
    def validate_full_schema(stats: Dict[str, Any], schema: np.dtype, show_valid_messages: bool = False) -> bool:
        """
        Validate that a dictionary matches the complete schema.
        Checks field names, count, and data types.
        
        Args:
            stats: Dictionary to validate
            schema: NumPy dtype schema to validate against
            show_valid_messages: If False, suppress output when validation passes.
                                 If True, print regardless of result.
            
        Returns:
            True if valid, False otherwise
        """
        expected_fields = {name: dtype for name, dtype in schema.descr}
        is_valid = True
        
        # Collect validation results
        field_count_ok = len(stats) == len(expected_fields)
        if not field_count_ok:
            is_valid = False
        
        missing_fields = set(expected_fields.keys()) - set(stats.keys())
        if missing_fields:
            is_valid = False
        
        extra_fields = set(stats.keys()) - set(expected_fields.keys())
        if extra_fields:
            is_valid = False
        
        type_mismatches = []
        for field_name, expected_dtype in expected_fields.items():
            if field_name not in stats:
                continue
            value = stats[field_name]
            expected_type = np.dtype(expected_dtype).type
            if not isinstance(value, expected_type):
                type_mismatches.append((field_name, type(value).__name__, expected_type.__name__))
        
        if type_mismatches:
            is_valid = False
        
        # Only print if validation failed or show_valid_messages is True
        if not is_valid or show_valid_messages:
            print("\n" + "="*80)
            print("FULL SCHEMA VALIDATION")
            print("="*80)
            
            print(f"\nField Count Check:")
            print(f"  Provided: {len(stats)} fields")
            print(f"  Expected: {len(expected_fields)} fields")
            if not field_count_ok:
                print(f"  X MISMATCH: Difference of {abs(len(stats) - len(expected_fields))} fields")
            else:
                print(f"  ✓ PASS")
            
            print(f"\nMissing Fields Check:")
            if missing_fields:
                print(f"  X FAILED: {len(missing_fields)} missing field(s)")
                for field in sorted(missing_fields):
                    print(f"     - '{field}' (expected type: {expected_fields[field]})")
            else:
                print(f"  ✓ PASS: All expected fields present")
            
            print(f"\nUnexpected Fields Check:")
            if extra_fields:
                print(f"  X FAILED: {len(extra_fields)} unexpected field(s)")
                for field in sorted(extra_fields):
                    print(f"     - '{field}' (not in schema)")
            else:
                print(f"  ✓ PASS: No unexpected fields")
            
            print(f"\nData Type Validation:")
            if type_mismatches:
                print(f"  X FAILED: {len(type_mismatches)} type mismatch(es)")
                for field_name, actual_type, expected_type in type_mismatches:
                    print(f"     - '{field_name}': got {actual_type}, expected {expected_type}")
            else:
                print(f"  ✓ PASS: All field types correct")
            
            print("\n" + "="*80)
            if is_valid:
                print("RESULT: ✓ VALIDATION PASSED")
            else:
                print("RESULT: X VALIDATION FAILED")
            print("="*80 + "\n")
        
        return is_valid




    @staticmethod
    def validate_statistics_schema(stats: Dict[str, Any], schema: np.dtype, delta_prefixes: Optional[List[str]] = None, show_valid_messages: bool = False) -> bool:
        """
        Validate that a dictionary contains only non-delta (statistics) fields from schema.
        Checks field names, count, and data types.
        
        Args:
            stats: Dictionary to validate
            schema: NumPy dtype schema to validate against
            delta_prefixes: List of prefixes to identify delta fields to exclude. 
                        Defaults to ['delta_', 'derived_'] if None.
            show_valid_messages: If False, suppress output when validation passes.
                                 If True, print regardless of result.
            
        Returns:
            True if valid, False otherwise
        """
        if delta_prefixes is None:
            delta_prefixes = ['delta_', 'derived_']
        
        expected_fields = {name: dtype for name, dtype in schema.descr 
                        if not any(name.startswith(prefix) for prefix in delta_prefixes)}
        is_valid = True
        
        # Collect validation results
        field_count_ok = len(stats) == len(expected_fields)
        if not field_count_ok:
            is_valid = False
        
        missing_fields = set(expected_fields.keys()) - set(stats.keys())
        if missing_fields:
            is_valid = False
        
        extra_fields = set(stats.keys()) - set(expected_fields.keys())
        if extra_fields:
            is_valid = False
        
        type_mismatches = []
        for field_name, expected_dtype in expected_fields.items():
            if field_name not in stats:
                continue
            value = stats[field_name]
            expected_type = np.dtype(expected_dtype).type
            if not isinstance(value, expected_type):
                type_mismatches.append((field_name, type(value).__name__, expected_type.__name__))
        
        if type_mismatches:
            is_valid = False
        
        # Only print if validation failed or show_valid_messages is True
        if not is_valid or show_valid_messages:
            print("\n" + "="*80)
            print("STATISTICS SCHEMA VALIDATION")
            print(f"Excluding prefixes: {delta_prefixes}")
            print("="*80)
            
            print(f"\nField Count Check:")
            print(f"  Provided: {len(stats)} fields")
            print(f"  Expected: {len(expected_fields)} statistics fields")
            if not field_count_ok:
                print(f"  X MISMATCH: Difference of {abs(len(stats) - len(expected_fields))} fields")
            else:
                print(f"  ✓ PASS")
            
            print(f"\nMissing Statistics Fields Check:")
            if missing_fields:
                print(f"  X FAILED: {len(missing_fields)} missing statistics field(s)")
                for field in sorted(missing_fields):
                    print(f"     - '{field}' (expected type: {expected_fields[field]})")
            else:
                print(f"  ✓ PASS: All expected statistics fields present")
            
            print(f"\nUnexpected Fields Check:")
            if extra_fields:
                print(f"  X FAILED: {len(extra_fields)} unexpected field(s)")
                
                delta_fields_present = [f for f in extra_fields 
                                        if any(f.startswith(prefix) for prefix in delta_prefixes)]
                if delta_fields_present:
                    print(f"     Delta fields (should not be present in statistics):")
                    for field in sorted(delta_fields_present):
                        print(f"       - '{field}'")
                
                non_schema_fields = [f for f in extra_fields 
                                    if not any(f.startswith(prefix) for prefix in delta_prefixes)]
                if non_schema_fields:
                    print(f"     Fields not in schema:")
                    for field in sorted(non_schema_fields):
                        print(f"       - '{field}'")
            else:
                print(f"  ✓ PASS: No unexpected fields")
            
            print(f"\nData Type Validation:")
            if type_mismatches:
                print(f"  X FAILED: {len(type_mismatches)} type mismatch(es)")
                for field_name, actual_type, expected_type in type_mismatches:
                    print(f"     - '{field_name}': got {actual_type}, expected {expected_type}")
            else:
                print(f"  ✓ PASS: All field types correct")
            
            print("\n" + "="*80)
            if is_valid:
                print("RESULT: ✓ VALIDATION PASSED")
            else:
                print("RESULT: X VALIDATION FAILED")
            print("="*80 + "\n")
        
        return is_valid




    @staticmethod
    def validate_delta_schema(stats: Dict[str, Any], schema: np.dtype, delta_prefixes: Optional[List[str]] = None, show_valid_messages: bool = False) -> bool:
        """
        Validate that a dictionary contains only delta fields from schema.
        Checks field names, count, and data types.
        
        Args:
            stats: Dictionary to validate
            schema: NumPy dtype schema to validate against
            delta_prefixes: List of prefixes to identify delta fields. 
                        Defaults to ['delta_', 'derived_'] if None.
            show_valid_messages: If False, suppress output when validation passes.
                                 If True, print regardless of result.
            
        Returns:
            True if valid, False otherwise
        """
        if delta_prefixes is None:
            delta_prefixes = ['delta_', 'derived_']
        
        expected_fields = {name: dtype for name, dtype in schema.descr 
                        if any(name.startswith(prefix) for prefix in delta_prefixes)}
        is_valid = True
        
        # Collect validation results
        field_count_ok = len(stats) == len(expected_fields)
        if not field_count_ok:
            is_valid = False
        
        missing_fields = set(expected_fields.keys()) - set(stats.keys())
        if missing_fields:
            is_valid = False
        
        extra_fields = set(stats.keys()) - set(expected_fields.keys())
        if extra_fields:
            is_valid = False
        
        type_mismatches = []
        for field_name, expected_dtype in expected_fields.items():
            if field_name not in stats:
                continue
            value = stats[field_name]
            expected_type = np.dtype(expected_dtype).type
            if not isinstance(value, expected_type):
                type_mismatches.append((field_name, type(value).__name__, expected_type.__name__))
        
        if type_mismatches:
            is_valid = False
        
        # Only print if validation failed or show_valid_messages is True
        if not is_valid or show_valid_messages:
            print("\n" + "="*80)
            print("DELTA SCHEMA VALIDATION")
            print(f"Delta prefixes: {delta_prefixes}")
            print("="*80)
            
            print(f"\nField Count Check:")
            print(f"  Provided: {len(stats)} fields")
            print(f"  Expected: {len(expected_fields)} delta fields")
            if not field_count_ok:
                print(f"  X MISMATCH: Difference of {abs(len(stats) - len(expected_fields))} fields")
            else:
                print(f"  ✓ PASS")
            
            print(f"\nMissing Delta Fields Check:")
            if missing_fields:
                print(f"  X FAILED: {len(missing_fields)} missing delta field(s)")
                for field in sorted(missing_fields):
                    print(f"     - '{field}' (expected type: {expected_fields[field]})")
            else:
                print(f"  ✓ PASS: All expected delta fields present")
            
            print(f"\nUnexpected Fields Check:")
            if extra_fields:
                print(f"  X FAILED: {len(extra_fields)} unexpected field(s)")
                
                for prefix in delta_prefixes:
                    prefix_fields = [f for f in extra_fields if f.startswith(prefix)]
                    if prefix_fields:
                        print(f"     Fields with '{prefix}' prefix not in schema:")
                        for field in sorted(prefix_fields):
                            print(f"       - '{field}'")
                
                non_delta_fields = [f for f in extra_fields 
                                if not any(f.startswith(prefix) for prefix in delta_prefixes)]
                if non_delta_fields:
                    print(f"     Non-delta fields (should not be present):")
                    for field in sorted(non_delta_fields):
                        print(f"       - '{field}'")
            else:
                print(f"  ✓ PASS: No unexpected fields")
            
            print(f"\nData Type Validation:")
            if type_mismatches:
                print(f"  X FAILED: {len(type_mismatches)} type mismatch(es)")
                for field_name, actual_type, expected_type in type_mismatches:
                    print(f"     - '{field_name}': got {actual_type}, expected {expected_type}")
            else:
                print(f"  ✓ PASS: All field types correct")
            
            print("\n" + "="*80)
            if is_valid:
                print("RESULT: ✓ VALIDATION PASSED")
            else:
                print("RESULT: X VALIDATION FAILED")
            print("="*80 + "\n")
        
        return is_valid




    @staticmethod
    def validate_function_column_access(func: Callable, schema: np.dtype, show_valid_messages: bool = False) -> bool:
        """
        Dynamically validate that all column accesses in a function exist in schema.
        Parses the function's source code to extract column names being accessed.
        
        Args:
            func: Function to validate (must access array columns via subscript notation)
            schema: NumPy dtype schema to validate against
            show_valid_messages: If False, suppress output when validation passes.
                                 If True, print regardless of result.
            
        Returns:
            True if all accessed columns exist in schema, False otherwise
        """
        is_valid = True
        
        # Get function source code
        try:
            source = inspect.getsource(func)
            source = textwrap.dedent(source)
        except OSError:
            print("\n" + "="*80)
            print(f"DYNAMIC COLUMN ACCESS VALIDATION: {func.__name__}")
            print("="*80)
            print("X Cannot retrieve source code for function")
            print("="*80 + "\n")
            return False
        
        # Parse source code into AST
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            print("\n" + "="*80)
            print(f"DYNAMIC COLUMN ACCESS VALIDATION: {func.__name__}")
            print("="*80)
            print(f"X Cannot parse function source code: {e}")
            print("="*80 + "\n")
            return False
        
        # Extract all subscript accesses (array['column_name'])
        accessed_columns = set()
        
        class ColumnAccessVisitor(ast.NodeVisitor):
            def visit_Subscript(self, node):
                if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
                    accessed_columns.add(node.slice.value)
                elif isinstance(node.slice, ast.Index) and isinstance(node.slice.value, ast.Constant):
                    if isinstance(node.slice.value.value, str):
                        accessed_columns.add(node.slice.value.value)
                self.generic_visit(node)
        
        visitor = ColumnAccessVisitor()
        visitor.visit(tree)
        
        # Get schema field names
        schema_fields = set(schema.names)
        
        # Check for invalid column access
        invalid_columns = accessed_columns - schema_fields
        if invalid_columns:
            is_valid = False
        
        # Only print if validation failed or show_valid_messages is True
        if not is_valid or show_valid_messages:
            print("\n" + "="*80)
            print(f"DYNAMIC COLUMN ACCESS VALIDATION: {func.__name__}")
            print("="*80)
            
            print(f"\nFunction: {func.__name__}")
            print(f"Columns accessed in code: {len(accessed_columns)}")
            print(f"Columns in schema: {len(schema_fields)}")
            
            if accessed_columns:
                print(f"\nAccessed columns:")
                for col in sorted(accessed_columns):
                    print(f"  - '{col}'")
            
            print(f"\nInvalid Column Access Check:")
            if invalid_columns:
                print(f"  X FAILED: {len(invalid_columns)} column(s) do not exist in schema")
                for column in sorted(invalid_columns):
                    print(f"     - '{column}' (will cause KeyError at runtime)")
            else:
                print(f"  ✓ PASS: All accessed columns exist in schema")
            
            if invalid_columns:
                print(f"\nPossible Typos:")
                for invalid_col in sorted(invalid_columns):
                    similar = []
                    for field in schema_fields:
                        if field.lower().replace('_', '') == invalid_col.lower().replace('_', ''):
                            similar.append(field)
                        elif len(field) == len(invalid_col):
                            diff_count = sum(1 for a, b in zip(field, invalid_col) if a != b)
                            if diff_count == 1:
                                similar.append(field)
                    
                    if similar:
                        print(f"     - '{invalid_col}' might be: {', '.join(similar)}")
            
            unused_columns = schema_fields - accessed_columns
            if unused_columns and len(unused_columns) < 20:
                print(f"\nInfo: {len(unused_columns)} schema column(s) not accessed:")
                for column in sorted(unused_columns):
                    print(f"     - '{column}'")
            elif unused_columns:
                print(f"\nInfo: {len(unused_columns)} schema column(s) not accessed (list truncated)")
            
            print("\n" + "="*80)
            if is_valid:
                print("RESULT: ✓ VALIDATION PASSED - All column accesses are safe")
            else:
                print("RESULT: X VALIDATION FAILED - Code will fail with KeyError")
            print("="*80 + "\n")
        
        return is_valid




    @staticmethod
    def round_to_half_pip_float64(
        values: NDArray[np.float64] | float,
        pip_size: float = 0.0001
    ) -> NDArray[np.float64] | float:
        """
        Round float prices to nearest 0.5 pip.
        
        Parameters:
        - values: array or scalar of prices in float format
        - pip_size: base pip (0.0001 for most pairs, 0.01 for JPY pairs)
        
        Returns: values rounded to nearest 0.5 pip
        """
        half_pip = pip_size / 2
        return np.round(values / half_pip) * half_pip




    @staticmethod
    def round_to_pipette_float64(
        values: NDArray[np.float64] | float,
        pip_size: float = 0.0001
    ) -> NDArray[np.float64] | float:
        """
        Round float prices to nearest pipette/fractional pip.
        
        Parameters:
        - values: array or scalar of prices in float format
        - pip_size: base pip (0.0001 for most pairs, 0.01 for JPY pairs)
        
        Returns: values rounded to nearest 0.1 pip (pipette)
        """
        pipette = pip_size / 10
        return np.round(values / pipette) * pipette




    @staticmethod
    def round_to_half_pip_int64(
        prices_int64: NDArray[np.int64] | np.int64,
        pip_size: float = 0.0001
    ) -> NDArray[np.int64] | np.int64:
        """
        Round int64 prices (1 unit = 1e-9) to nearest 0.5 pip.
        
        Parameters:
        - prices_int64: array or scalar of prices in int64 format (1 unit = 1e-9)
        - pip_size: base pip in float terms (0.0001 for most pairs, 0.01 for JPY pairs)
        
        Returns: prices rounded to nearest 0.5 pip in int64 format
        
        Example:
        - pip_size=0.0001 -> half_pip=0.00005 -> 50000 units at 1e-9 scale
        - pip_size=0.01 (JPY) -> half_pip=0.005 -> 5000000 units at 1e-9 scale
        """
        half_pip_units = int((pip_size / 2) / 1e-9)
        rounded = np.round(prices_int64 / half_pip_units) * half_pip_units
        return rounded.astype(np.int64) if isinstance(prices_int64, np.ndarray) else np.int64(rounded)




    @staticmethod
    def round_to_pipette_int64(
        prices_int64: NDArray[np.int64] | np.int64,
        pip_size: float = 0.0001
    ) -> NDArray[np.int64] | np.int64:
        """
        Round int64 prices (1 unit = 1e-9) to nearest pipette.
        
        Parameters:
        - prices_int64: array or scalar of prices in int64 format (1 unit = 1e-9)
        - pip_size: base pip in float terms (0.0001 for most pairs, 0.01 for JPY pairs)
        
        Returns: prices rounded to nearest pipette (0.1 pip) in int64 format
        
        Example:
        - pip_size=0.0001 -> pipette=0.00001 -> 10000 units at 1e-9 scale
        - pip_size=0.01 (JPY) -> pipette=0.001 -> 1000000 units at 1e-9 scale
        """
        pipette_units = int((pip_size / 10) / 1e-9)
        rounded = np.round(prices_int64 / pipette_units) * pipette_units
        return rounded.astype(np.int64) if isinstance(prices_int64, np.ndarray) else np.int64(rounded)
    
    
    
    
    @staticmethod
    def format_currency_numpy(
        values: Union[pl.Series, np.ndarray],
        currency_symbol: str = "",
        scale_factor: float = 1e-9,
        min_decimals: int = 2,
        max_decimals: int = 15,
        truncate: bool = True,
        use_commas: bool = True
    ) -> list[str]:
        """
        Vectorized currency formatting with scaling support.
        
        Args:
            values: Polars Series or numpy array of numeric values
            currency_symbol: Currency symbol to use (e.g., "$", "¥", "€"). If empty, no symbol shown.
            scale_factor: Multiplier to apply (default 1e-9 for converting int64 nano-units)
            min_decimals: Minimum decimal places to show, 0 or higher (0 = no decimal point if not needed)
            max_decimals: Maximum decimal places to show
            truncate: If True, truncate; if False, round
            use_commas: If True, add thousand separators
        
        Returns:
            List of formatted strings
        """
        
        # Validate min_decimals
        if min_decimals < 0:
            raise ValueError("min_decimals must be 0 or greater")
        
        # Convert to numpy array if it's a Polars Series
        arr = np.array(values, dtype=np.float64)
        
        # Apply scaling
        arr = arr * scale_factor
        
        # Apply truncation or rounding (vectorized)
        if truncate:
            multiplier = 10 ** max_decimals
            processed = np.floor(arr * multiplier) / multiplier
        else:
            processed = np.round(arr, max_decimals)
        
        # Clamp to max_decimals to override original precision
        processed = np.round(processed, max_decimals)
        
        # Determine if symbol should be shown
        show_symbol = len(currency_symbol) > 0
        symbol_prefix = currency_symbol if show_symbol else ""
        
        # Format each value
        results = []
        
        for val in processed:
            if np.isnan(val):
                results.append("N/A")
                continue
            
            # Format with max precision
            formatted = f"{val:.{max_decimals}f}"
            
            # Split into integer and decimal parts
            if '.' in formatted:
                integer_part, decimal_part = formatted.split('.')
                
                # Strip trailing zeros but keep at least min_decimals
                decimal_stripped = decimal_part.rstrip('0')
                
                # Format integer part with optional commas
                int_val = int(float(integer_part))
                if use_commas:
                    integer_formatted = f"{int_val:,}"
                else:
                    integer_formatted = str(int_val)
                
                # Handle decimal display based on min_decimals
                if min_decimals == 0 and len(decimal_stripped) == 0:
                    # No decimal point if min_decimals is 0 and no significant decimals
                    results.append(f"{symbol_prefix}{integer_formatted}")
                else:
                    decimal_padded = decimal_stripped.ljust(min_decimals, '0')
                    results.append(f"{symbol_prefix}{integer_formatted}.{decimal_padded}")
            else:
                # No decimal part
                int_val = int(float(formatted))
                if use_commas:
                    integer_formatted = f"{int_val:,}"
                else:
                    integer_formatted = str(int_val)
                results.append(f"{symbol_prefix}{integer_formatted}")
        
        return results




    # Helper function for Polars DataFrame integration
    @staticmethod
    def format_currency_columns(
        df: pl.DataFrame,
        columns: Union[str, list[str]],
        currency_symbol: str = "",
        scale_factor: float = 1e-9,
        min_decimals: int = 2,
        max_decimals: int = 2,
        truncate: bool = False,
        use_commas: bool = True,
        suffix: str = "_formatted"
    ) -> pl.DataFrame:
        """
        Apply currency formatting to specified columns in a Polars DataFrame.
        
        Args:
            df: Input DataFrame
            columns: Single column name or list of column names to format
            currency_symbol: Currency symbol to use. If empty, no symbol shown.
            scale_factor: Multiplier (default 1e-9 for int64 nano-units)
            min_decimals: Minimum decimal places (0 = no decimal point if not needed)
            max_decimals: Maximum decimal places (for rounding override)
            truncate: If True, truncate; if False, round
            use_commas: Add thousand separators
            suffix: Suffix to add to new column names
        
        Returns:
            DataFrame with new formatted columns
        """
        
        # Ensure columns is a list
        if isinstance(columns, str):
            columns = [columns]
        
        # Create new columns with formatted values
        result = df.clone()
        
        for col in columns:
            formatted_values = HelperFunctions.format_currency_numpy(
                values=df[col],
                currency_symbol=currency_symbol,
                scale_factor=scale_factor,
                min_decimals=min_decimals,
                max_decimals=max_decimals,
                truncate=truncate,
                use_commas=use_commas
            )
            result = result.with_columns(
                pl.Series(name=f"{col}{suffix}", values=formatted_values)
            )
        
        return result




    @staticmethod
    def get_dataset_summary(config, lf, start_date_ns, end_date_ns):
        """
        Analyze the dataset and return summary statistics.
        
        Parameters
        ----------
        lf : pl.LazyFrame
            LazyFrame containing the data to analyze
        start_date_ns : int
            Requested start timestamp in nanoseconds
        end_date_ns : int
            Requested end timestamp in nanoseconds
            
        Returns
        -------
        tuple
            (total_rows, total_size, date_min, date_max)
        """
        request_data = (
            lf.select([
                pl.len().alias("total_rows"),
                pl.col("size").sum().alias("total_size"),
                pl.col("ts_recv").min().alias("ts_min"),
                pl.col("ts_recv").max().alias("ts_max"),
            ])
            .collect()
            .with_columns([
                pl.from_epoch("ts_min", time_unit="ns").dt.strftime("%A, %B %d, %Y at %H:%M UTC (%Y-%m-%d)").alias("date_min"),
                pl.from_epoch("ts_max", time_unit="ns").dt.strftime("%A, %B %d, %Y at %H:%M UTC (%Y-%m-%d)").alias("date_max"),
            ])
        )

        initial_total_rows = request_data["total_rows"][0]
        initial_total_size = request_data["total_size"][0]
        initial_date_min = request_data["date_min"][0]
        initial_date_max = request_data["date_max"][0]
        ts_min = request_data["ts_min"][0]
        ts_max = request_data["ts_max"][0]

        start_date_str = datetime.fromtimestamp(start_date_ns / 1e9, tz=timezone.utc).strftime("%A, %B %d, %Y at %H:%M UTC (%Y-%m-%d)")
        end_date_str = datetime.fromtimestamp(end_date_ns / 1e9, tz=timezone.utc).strftime("%A, %B %d, %Y at %H:%M UTC (%Y-%m-%d)")

        # Calculate time deltas
        start_delta_ns = ts_min - start_date_ns  # Positive if data starts after requested
        end_delta_ns = end_date_ns - ts_max      # Positive if data ends before requested


        # [ Provide synopsis of REQUEST ]
        print(f"""
        \n\n
┌─ DATA PROCESSING SUMMARY {'─' * 39}
│
│  Preparing to process {initial_total_rows:,} orders
│
│  {initial_total_size:,} total contracts traded
│
│
│
│  (Requested Start):   {start_date_str}
│  Data begins on:      {initial_date_min}""")
        

        # Display start gap warning if exists
        if start_delta_ns != 0:
            delta_str = HelperFunctions.format_time_delta(abs(start_delta_ns))
            if start_delta_ns > 0:
                print(f"│")
                print(f"│  ⚠ Time Gap at Start:")
                print(f"│    Missing first {delta_str}")
                print(f"│")
            else:
                print(f"│")
                print(f"│  ⚠ Time Gap at Start:")
                print(f"│    Data includes extra {delta_str} before requested start")
                print(f"│")

        # End
        print(f"""│
│
│
│  (Requested End):     {end_date_str}
│  Data ends on:        {initial_date_max}
│""")
        
        # Display end gap warning if exists
        if end_delta_ns != 0:
            delta_str = HelperFunctions.format_time_delta(abs(end_delta_ns))
            if end_delta_ns > 0:
                print(f"│")
                print(f"│  ⚠ Time Gap at End:")
                print(f"│    Missing last {delta_str}")
                print(f"│")
            else:
                print(f"│")
                print(f"│  ⚠ Time Gap at End:")
                print(f"│    Data includes extra {delta_str} after requested end")
                print(f"│")

        print(f"│")
        print(f"└─{'─' * 60}")
        print(f"\n\n")
    

        # [ Provide synopsis of CONFIGURATION ]
        print(f"""
\n\n
┌─ PROCESSING CONFIGURATION {'─' * 38}
│
│  Volume Bars:
│    Contracts per bar:        {config.volumebar_contracts_per_bar:>8,}
│    Lookback Buffer size:               {config.volumebar_lookback_buffer_size:>8,}
│
│  Buckets:
│    Fixed Buckets:             {"ENABLED" if config.enable_fixed_buckets else "DISABLED"}""")

        if config.enable_fixed_buckets:
            print(f"│      VolumeBars per Fixed_Bucket:  {config.fixed_bucket_volumebars_per_bucket:>8,}")

        print(f"│")
        print(f"│    Interval Buckets:         {'ENABLED' if config.enable_interval_buckets else 'DISABLED'}")

        if config.enable_interval_buckets:
            print(f"│      VolumeBars per Interval_Bucket:  {config.interval_bucket_volumebars_per_bucket:>8,}")

        print(f"│")
        print(f"│  Metrics:")

        if config.training:
            print(f"│    Mode:                      TRAINING (no metrics generated)")
        else:
            print(f"│    Mode:                      PRODUCTION")
            print(f"│")
            print(f"│    Fixed Metrics:")
            print(f"│      Buckets per metric:     {config.fixed_metrics_buckets_per_metric:>8,}")
            print(f"│")
            print(f"│    Interval Metrics:")
            print(f"│      Buckets per metric:     {config.interval_metrics_buckets_per_metric:>8,}")

        print(f"│")
        print(f"└─{'─' * 60}")
        print(f"\n\n")
        

        # Return data for later use in validation
        return (
            initial_total_rows, initial_total_size, initial_date_min, initial_date_max
        )




    @staticmethod
    def _generate_sync_report(
        strict: bool,
        completed: bool,
        processor_bars_finalized: int,
        vb_id: int,
        contract_volume_processed: int,
        vb_contract_volume_processed: int,
        rows_processed: int,
        vb_orders_processed: int,
        vb_start_ts_ns: int,
        vb_end_ts_ns: int,
    ) -> tuple[bool, str]:
        """
        Validate synchronization between HistoricalDataProcessor and VolumeBar.
        
        Args:
            completed: True for "Completed Processing", False for "In-Processing"
            processor_bars_finalized: Number of bars finalized by processor
            vb_id: VolumeBar ID
            contract_volume_processed: Volume processed by processor
            vb_contract_volume_processed: Volume processed by VolumeBar
            rows_processed: Rows processed by processor
            vb_orders_processed: Orders processed by VolumeBar
            vb_start_ts_ns: VolumeBar start timestamp in nanoseconds
            vb_end_ts_ns: VolumeBar end timestamp in nanoseconds
            strict: If True, include volume match in sync status evaluation
        
        Returns:
            tuple[bool, str]: (sync_status, formatted_message)
                - sync_status: True if synchronized, False if discrepancy detected
                - formatted_message: Formatted log message with sync details
        """


        # {VolumeBars} start at (row/index 0), thus the need to deincrement for direct comparison
        processor_bars_finalized -= 1
        
        source = "<<< *** Completed Processing *** >>>" if completed else "In-Processing"
        bars_match = processor_bars_finalized == vb_id
        rows_match = rows_processed == vb_orders_processed
        volume_match = contract_volume_processed == vb_contract_volume_processed
        
        if strict:
            sync_status = bars_match and rows_match and volume_match
        else:
            sync_status = bars_match and rows_match
        
        if sync_status:
            # All Clear: Synchronized
            message = (
                "\n\n"
                "\n================================================================================\n"
                f"FROM: < {source} >\n"
                "Processor-VolumeBar sync verified\n\n"
                
                "[BAR COUNTS]\n"
                f"  processor_bars_finalized = {processor_bars_finalized:,}\n"
                f"  vb_id                    = {vb_id:,}\n\n"
                
                "[VOLUME PROCESSED]\n"
                f"  contract_volume_processed    = {contract_volume_processed:,}\n"
                f"  vb_contract_volume_processed = {vb_contract_volume_processed:,}\n\n"
                
                "[ROW/ORDER COUNTS]\n"
                f"  rows_processed      = {rows_processed:,}\n"
                f"  vb_orders_processed = {vb_orders_processed:,}\n\n"
                
                "[TIMESTAMPS]\n"
                f"  vb_start_ts_ns = {vb_start_ts_ns}\n"
                f"  vb_end_ts_ns   = {vb_end_ts_ns}\n"
                "\n================================================================================\n"
                "\n\n"
            )
        else:
            # Critical: Discrepancy detected
            message = (
                "\n\n"
                "\n"
                "################################################################################\n"
                "###                     SYNC DISCREPANCY DETECTED                           ###\n"
                "################################################################################\n"
                f"FROM: < {source} >\n\n"
                
                "[BAR COUNTS]\n"
                f"  processor_bars_finalized = {processor_bars_finalized}\n"
                f"  vb_id                    = {vb_id}\n"
                f"  bars_match               = <{bars_match}>\n\n"
                
                "[VOLUME PROCESSED]\n"
                f"  contract_volume_processed    = {contract_volume_processed}\n"
                f"  vb_contract_volume_processed = {vb_contract_volume_processed}\n"
                f"  volume_match                 = <{volume_match}>\n\n"
                
                "[ROW/ORDER COUNTS]\n"
                f"  rows_processed      = {rows_processed}\n"
                f"  vb_orders_processed = {vb_orders_processed}\n"
                f"  rows_match          = <{rows_match}>\n\n"
                
                "[TIMESTAMPS]\n"
                f"  vb_start_ts_ns = {vb_start_ts_ns}\n"
                f"  vb_end_ts_ns   = {vb_end_ts_ns}\n"
                "\n################################################################################\n"
                "\n\n"
            )
        
        return sync_status, message




    # processing_results["incomplete_number_of_volumebar_raw"]
    @staticmethod
    def evaluate_processing_results(
        processing_results: dict,
        initial_total_rows: int,
        initial_total_size: int,
        config: HistoricalConfig,
    ) -> bool:
        """
        Evaluate if processing completed successfully based on buffer/chunk types present.
        
        Validation checks:
        - VolumeBar: Verify all orders/volume processed
        - Bucket: Verify sufficient buckets generated based on volumebar count
        - Metric: Verify sufficient metrics generated based on bucket count
        
        Args:
            processing_results: Dictionary containing processing statistics
            initial_total_rows: Expected number of orders to process
            initial_total_size: Expected total volume to process
            config: HistoricalConfig with lookback window settings
            
        Returns:
            True if all validations pass, False otherwise
        """
        
        success = processing_results.get("success", False)
        
        if not success:
            logger.error("Processing marked as failed in results")
            return False
        
        all_checks_passed = True
        
        # [ VolumeBar Validation ]
        # Must always exist
        try:
            final_total_rows = processing_results["total_rows"]
            final_total_size = processing_results["total_size"]
            
            rows_match = final_total_rows == initial_total_rows
            size_match = final_total_size == initial_total_size
            
            if not rows_match:
                logger.error(f"VolumeBar row count mismatch: expected {initial_total_rows}, got {final_total_rows}")
                all_checks_passed = False
                
            if not size_match:
                logger.error(f"VolumeBar volume mismatch: expected {initial_total_size}, got {final_total_size}")
                all_checks_passed = False
                
        except KeyError as e:
            logger.error(f"Missing required volumebar field: {e}")
            return False
        
        
        # [ Fixed Bucket Validation ]
        if "total_number_of_bucket_fixed_generated" in processing_results:
            try:
                actual_fixed_buckets = processing_results["total_number_of_bucket_fixed_generated"]
                total_volumebar_count = processing_results["total_number_of_volumebar_raw_generated"]
                incomplete_volumebar_count = processing_results.get("incomplete_number_of_volumebar_raw", 0)
                completed_volumebar_count = total_volumebar_count - incomplete_volumebar_count
                
                # For fixed buckets, first bucket is generated at first multiple of bucket_size
                first_bucket_at = config.fixed_bucket_volumebars_per_bucket
                
                # Calculate expected buckets from first bucket to last completed volumebar
                # (Fencepost problem, count all fenceposts-not just those inbetween)
                if completed_volumebar_count >= first_bucket_at:
                    expected_fixed_buckets = ((completed_volumebar_count - first_bucket_at) // config.fixed_bucket_volumebars_per_bucket) + 1
                else:
                    expected_fixed_buckets = 0
                
                if actual_fixed_buckets != expected_fixed_buckets:
                    logger.error(
                        f"Fixed bucket count mismatch: expected {expected_fixed_buckets}, got {actual_fixed_buckets} "
                        f"(total_volumebars={total_volumebar_count}, incomplete={incomplete_volumebar_count}, "
                        f"completed={completed_volumebar_count}, first_bucket_at={first_bucket_at}, "
                        f"volumebars_per_bucket={config.fixed_bucket_volumebars_per_bucket})"
                    )
                    all_checks_passed = False
                    
            except KeyError as e:
                logger.error(f"Missing required fixed bucket field: {e}")
                all_checks_passed = False
        
        
        # [ Interval Bucket Validation ]
        if "total_number_of_bucket_interval_generated" in processing_results:
            try:
                actual_interval_buckets = processing_results["total_number_of_bucket_interval_generated"]
                total_volumebar_count = processing_results["total_number_of_volumebar_raw_generated"]
                incomplete_volumebar_count = processing_results.get("incomplete_number_of_volumebar_raw", 0)
                completed_volumebar_count = total_volumebar_count - incomplete_volumebar_count
                
                # For interval buckets, first bucket is generated when interval_lookback volumebars are available
                warmup_loss = config.interval_bucket_volumebars_per_bucket - 1
                
                # After warmup, a bucket is generated on EVERY volumebar
                expected_interval_buckets = max(0, completed_volumebar_count - warmup_loss)
                
                if actual_interval_buckets != expected_interval_buckets:
                    logger.error(
                        f"Interval bucket count mismatch: expected {expected_interval_buckets}, got {actual_interval_buckets} "
                        f"(total_volumebars={total_volumebar_count}, incomplete={incomplete_volumebar_count}, "
                        f"completed={completed_volumebar_count}, warmup_loss={warmup_loss}, "
                        f"interval_lookback={config.interval_bucket_volumebars_per_bucket})"
                    )
                    all_checks_passed = False
                    
            except KeyError as e:
                logger.error(f"Missing required interval bucket field: {e}")
                all_checks_passed = False
        
        
        # [ Fixed Metric Validation ]
        if "total_number_of_metric_fixed_generated" in processing_results:
            try:
                actual_fixed_metrics = processing_results["total_number_of_metric_fixed_generated"]
                
                # Metrics based on bucket count
                if "total_number_of_bucket_fixed_generated" in processing_results:
                    bucket_count = processing_results["total_number_of_bucket_fixed_generated"]
                else:
                    logger.error("Fixed metrics exist but no fixed buckets found")
                    all_checks_passed = False
                    bucket_count = 0
                
                # Expected: floor(bucket_count / fixed_metrics_buckets_per_metric)
                expected_fixed_metrics = bucket_count // config.fixed_metrics_buckets_per_metric
                
                if actual_fixed_metrics != expected_fixed_metrics:
                    logger.error(
                        f"Fixed metric count mismatch: expected {expected_fixed_metrics}, got {actual_fixed_metrics} "
                        f"(buckets={bucket_count}, lookback={config.fixed_metrics_buckets_per_metric})"
                    )
                    all_checks_passed = False
                    
            except KeyError as e:
                logger.error(f"Missing required fixed metric field: {e}")
                all_checks_passed = False
        
        
        # [ Interval Metric Validation ]
        if "total_number_of_metric_interval_generated" in processing_results:
            try:
                actual_interval_metrics = processing_results["total_number_of_metric_interval_generated"]
                
                # Metrics based on bucket count
                if "total_number_of_bucket_interval_generated" in processing_results:
                    bucket_count = processing_results["total_number_of_bucket_interval_generated"]
                else:
                    logger.error("Interval metrics exist but no interval buckets found")
                    all_checks_passed = False
                    bucket_count = 0
                
                # Expected: max(0, bucket_count - interval_metrics_buckets_per_metric)
                expected_interval_metrics = max(0, bucket_count - config.interval_metrics_buckets_per_metric)
                
                if actual_interval_metrics != expected_interval_metrics:
                    logger.error(
                        f"Interval metric count mismatch: expected {expected_interval_metrics}, got {actual_interval_metrics} "
                        f"(buckets={bucket_count}, lookback={config.interval_metrics_buckets_per_metric})"
                    )
                    all_checks_passed = False
                    
            except KeyError as e:
                logger.error(f"Missing required interval metric field: {e}")
                all_checks_passed = False
        
        return all_checks_passed



    @staticmethod
    def format_time_delta(delta_ns: int) -> str:
        """
        Convert nanosecond delta to human-readable format.
        
        Args:
            delta_ns: Time difference in nanoseconds
            
        Returns:
            Human-readable string like "5 minutes 32 seconds" or "2 days 3 hours 15 minutes"
        """
        if delta_ns < 0:
            return "NEGATIVE DELTA (ERROR)"
        
        # Convert to seconds
        total_seconds = delta_ns / 1_000_000_000
        
        days = int(total_seconds // 86400)
        remaining = total_seconds % 86400
        hours = int(remaining // 3600)
        remaining = remaining % 3600
        minutes = int(remaining // 60)
        seconds = int(remaining % 60)
        
        parts = []
        if days > 0:
            parts.append(f"{days} day{'s' if days != 1 else ''}")
        if hours > 0:
            parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
        if minutes > 0:
            parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
        if seconds > 0 or len(parts) == 0:
            parts.append(f"{seconds} second{'s' if seconds != 1 else ''}")
        
        return " ".join(parts)



    @staticmethod
    def pretty_print_processing_verification_results(
        processing_results: dict,
        initial_total_rows: int,
        initial_total_size: int,
        config: HistoricalConfig,
    ) -> None:
        """
        Display formatted processing verification results.
        
        Shows validation for volumebars, buckets, and metrics with checkmarks,
        timestamp ranges, and time deltas between buffer levels.
        
        Args:
            processing_results: Dictionary containing processing statistics
            initial_total_rows: Expected number of orders to process
            initial_total_size: Expected total volume to process
            config: HistoricalConfig with lookback window settings
        """
        
        print("\n" + "═" * 80)
        print("PROCESSING VERIFICATION REPORT".center(80))
        print("═" * 80)
        
        # [ VolumeBar Section ]
        print("\n┌─ VOLUMEBARS (Raw) " + "─" * 58)
        
        final_total_rows = processing_results.get("total_rows", 0)
        final_total_size = processing_results.get("total_size", 0)
        
        rows_match = final_total_rows == initial_total_rows
        size_match = final_total_size == initial_total_size
        
        print(f"│")
        print(f"│  Orders Processed:")
        print(f"│    Expected:  {initial_total_rows:>15,}")
        print(f"│    Actual:    {final_total_rows:>15,}  {'✓' if rows_match else 'X MISMATCH'}")
        print(f"│")
        print(f"│  Volume Processed:")
        print(f"│    Expected:  {initial_total_size:>15,}")
        print(f"│    Actual:    {final_total_size:>15,}  {'✓' if size_match else 'X MISMATCH'}")
        print(f"│")
        
        if "date_min" in processing_results and "date_max" in processing_results:
            print(f"│  Time Range:")
            print(f"│    Start:  {processing_results['date_min']}")
            print(f"│    End:    {processing_results['date_max']}")
        
        volumebar_ts_max = processing_results.get("volumebar_raw_ts_max")
        
        
        # [ Fixed Bucket Section ]
        if "total_number_of_bucket_fixed_generated" in processing_results:
            print("\n┌─ BUCKETS (Fixed) " + "─" * 61)
            
            actual_buckets = processing_results["total_number_of_bucket_fixed_generated"]
            
            # Get actual volumebar counts from processing results
            total_volumebar_count = processing_results["total_number_of_volumebar_raw_generated"]
            incomplete_volumebar_count = processing_results.get("incomplete_number_of_volumebar_raw", 0)
            completed_volumebar_count = total_volumebar_count - incomplete_volumebar_count
            
            # For fixed buckets, first bucket is generated at first multiple of bucket_size
            first_bucket_at = config.fixed_bucket_volumebars_per_bucket
            warmup_loss = first_bucket_at - 1
            
            # Calculate expected buckets
            if completed_volumebar_count >= first_bucket_at:
                expected_buckets = ((completed_volumebar_count - first_bucket_at) // config.fixed_bucket_volumebars_per_bucket) + 1
            else:
                expected_buckets = 0
            
            expected_buckets_raw = float(expected_buckets)
            buckets_match = actual_buckets == expected_buckets
            
            bars_used = processing_results.get("number_of_bars_used", 0)
            
            print(f"│")
            print(f"│")
            print(f"│  Calculation Context:")
            print(f"│    Total Contracts:           {final_total_size:>15,}")
            print(f"│    Total VolumeBars:          {total_volumebar_count:>15,}  (from processing)")
            print(f"│      - Incomplete:            {incomplete_volumebar_count:>15,}")
            print(f"│      - Completed:             {completed_volumebar_count:>15,}")
            print(f"│")
            print(f"│    First Bucket At:           {first_bucket_at:>15,}  (first multiple of bucket_size)")
            print(f"│    - Warmup Loss:             {warmup_loss:>15,}  (volumebars 1-{warmup_loss})")
            print(f"│")
            print(f"│")
            print(f"│    fixed_bucket_volumebars_per_bucket:         / {config.fixed_bucket_volumebars_per_bucket:>8,}")
            print(f"│    → Buckets:                 {expected_buckets:>15,}  (from VB {first_bucket_at})")
            print(f"│")
            print(f"│")
            print(f"│  Buckets Generated:")
            print(f"│    Expected:  {expected_buckets:>15,}  ((completed - first_bucket) / size + 1) aka Fencepost Problem")
            print(f"│    Actual:    {actual_buckets:>15,}  {'✓' if buckets_match else 'X MISMATCH'}")
            print(f"│    Difference:    {(expected_buckets - actual_buckets):>15,}")
            print(f"│")
            print(f"│")
            print(f"│  VolumeBars Used:  {bars_used:>15,}")
            print(f"│  Lookback Window:  {config.fixed_bucket_volumebars_per_bucket:>15,}")
            print(f"│")
            print(f"│")
            
            bucket_fixed_ts_min = processing_results.get("bucket_fixed_ts_min")
            bucket_fixed_ts_max = processing_results.get("bucket_fixed_ts_max")
            
            if bucket_fixed_ts_min and bucket_fixed_ts_max:
                # Convert timestamps to readable format
                date_min = pl.from_epoch([bucket_fixed_ts_min], time_unit="ns").dt.strftime("%A, %B %d, %Y at %H:%M UTC")[0]
                date_max = pl.from_epoch([bucket_fixed_ts_max], time_unit="ns").dt.strftime("%A, %B %d, %Y at %H:%M UTC")[0]
                
                print(f"│")
                print(f"│  Time Range:")
                print(f"│    Start:  {date_min}")
                print(f"│    End:    {date_max}")
                print(f"│")
            
            # Calculate delta from volumebar
            if volumebar_ts_max and bucket_fixed_ts_max:
                delta_ns = volumebar_ts_max - bucket_fixed_ts_max
                if delta_ns > 0:
                    delta_str = HelperFunctions.format_time_delta(delta_ns)
                    print(f"│")
                    print(f"│")
                    print(f"│")
                    print(f"│  ⚠ Time Gap from VolumeBars:")
                    print(f"│    Missing last {delta_str}")
                    print(f"│")
                    print(f"│")
        
        
        # [ Interval Bucket Section ]
        if "total_number_of_bucket_interval_generated" in processing_results:
            print("\n┌─ BUCKETS (Interval) " + "─" * 58)
            
            actual_buckets = processing_results["total_number_of_bucket_interval_generated"]
            
            # Get actual volumebar counts from processing results
            total_volumebar_count = processing_results["total_number_of_volumebar_raw_generated"]
            incomplete_volumebar_count = processing_results.get("incomplete_number_of_volumebar_raw", 0)
            completed_volumebar_count = total_volumebar_count - incomplete_volumebar_count
            
            # For interval buckets, first bucket is generated when interval_lookback volumebars are available
            warmup_loss = config.interval_bucket_volumebars_per_bucket - 1
            processable_volumebars = max(0, completed_volumebar_count - warmup_loss)
            
            expected_buckets = processable_volumebars  # One bucket per volumebar after warmup
            expected_buckets_raw = float(processable_volumebars)
            
            buckets_match = actual_buckets == expected_buckets
            
            bars_used = processing_results.get("number_of_bars_used", 0)
            
            print(f"│")
            print(f"│")
            print(f"│  Calculation Context:")
            print(f"│    Total Contracts:           {final_total_size:>15,}")
            print(f"│    Total VolumeBars:          {total_volumebar_count:>15,}  (from processing)")
            print(f"│      - Incomplete:            {incomplete_volumebar_count:>15,}")
            print(f"│      - Completed:             {completed_volumebar_count:>15,}")
            print(f"│")
            print(f"│    First Bucket At:           {warmup_loss + 1:>15,}  (after interval lookback)")
            print(f"│    - Warmup Loss:             {warmup_loss:>15,}  (volumebars 1-{warmup_loss})")
            print(f"│")
            print(f"│")
            print(f"│    interval_bucket_volumebars_per_bucket: {config.interval_bucket_volumebars_per_bucket:>6,} (lookback)")
            print(f"│    → Buckets (one per VB):    {expected_buckets:>15,}  (from VB {warmup_loss + 1})")
            print(f"│")
            print(f"│")
            print(f"│  Buckets Generated:")
            print(f"│    Expected:  {expected_buckets:>15,}  (processable_vbs - 0 for interval)")
            print(f"│    Actual:    {actual_buckets:>15,}  {'✓' if buckets_match else 'X MISMATCH'}")
            print(f"│    Difference:    {(expected_buckets - actual_buckets):>15,}")
            print(f"│")
            print(f"│")
            print(f"│  VolumeBars Used:  {bars_used:>15,}")
            print(f"│  Lookback Window:  {config.interval_bucket_volumebars_per_bucket:>15,}")
            print(f"│")
            
            bucket_interval_ts_min = processing_results.get("bucket_interval_ts_min")
            bucket_interval_ts_max = processing_results.get("bucket_interval_ts_max")
            
            if bucket_interval_ts_min and bucket_interval_ts_max:
                date_min = pl.from_epoch([bucket_interval_ts_min], time_unit="ns").dt.strftime("%A, %B %d, %Y at %H:%M UTC")[0]
                date_max = pl.from_epoch([bucket_interval_ts_max], time_unit="ns").dt.strftime("%A, %B %d, %Y at %H:%M UTC")[0]
                
                print(f"│")
                print(f"│  Time Range:")
                print(f"│    Start:  {date_min}")
                print(f"│    End:    {date_max}")
                print(f"│")
            
            # Calculate delta from volumebar
            if volumebar_ts_max and bucket_interval_ts_max:
                delta_ns = volumebar_ts_max - bucket_interval_ts_max
                if delta_ns > 0:
                    delta_str = HelperFunctions.format_time_delta(delta_ns)
                    print(f"│")
                    print(f"│")
                    print(f"│  ⚠ Time Gap from VolumeBars:")
                    print(f"│    Missing last {delta_str}")
                    print(f"│")
        
        
        # [ Fixed Metric Section ]
        if "total_number_of_metric_fixed_generated" in processing_results:
            print("\n┌─ METRICS (Fixed) " + "─" * 61)
            
            actual_metrics = processing_results["total_number_of_metric_fixed_generated"]
            
            bucket_count = processing_results.get("total_number_of_bucket_fixed_generated", 0)
            expected_metrics = bucket_count // config.fixed_metrics_buckets_per_metric
            expected_metrics_raw = bucket_count / config.fixed_metrics_buckets_per_metric
            
            metrics_match = actual_metrics == expected_metrics
            
            buckets_used = processing_results.get("num_buckets", 0)
            
            print(f"│")
            print(f"│  Calculation Context:")
            print(f"│    Buckets:                   {bucket_count:>15,}")
            print(f"│    fixed_metrics_buckets_per_metric:    {config.fixed_metrics_buckets_per_metric:>8,}")
            print(f"│    → Metrics (floor):         {expected_metrics:>15,}  ({expected_metrics_raw:.2f})")
            print(f"│")
            print(f"│  Metrics Generated:")
            print(f"│    Expected:  {expected_metrics:>15,}  (buckets / lookback_window)")
            print(f"│    Actual:    {actual_metrics:>15,}  {'✓' if metrics_match else 'X MISMATCH'}")
            print(f"│")
            print(f"│  Buckets Used:     {buckets_used:>15,}")
            print(f"│  Lookback Window:  {config.fixed_metrics_buckets_per_metric:>15,}")
            print(f"│")
            
            metric_fixed_ts_min = processing_results.get("metric_fixed_ts_min")
            metric_fixed_ts_max = processing_results.get("metric_fixed_ts_max")
            
            if metric_fixed_ts_min and metric_fixed_ts_max:
                date_min = pl.from_epoch([metric_fixed_ts_min], time_unit="ns").dt.strftime("%A, %B %d, %Y at %H:%M UTC")[0]
                date_max = pl.from_epoch([metric_fixed_ts_max], time_unit="ns").dt.strftime("%A, %B %d, %Y at %H:%M UTC")[0]
                
                print(f"│  Time Range:")
                print(f"│    Start:  {date_min}")
                print(f"│    End:    {date_max}")
                print(f"│")
            
            # Calculate deltas
            bucket_fixed_ts_max = processing_results.get("bucket_fixed_ts_max")
            
            if volumebar_ts_max and metric_fixed_ts_max:
                delta_ns = volumebar_ts_max - metric_fixed_ts_max
                if delta_ns > 0:
                    delta_str = HelperFunctions.format_time_delta(delta_ns)
                    print(f"│")
                    print(f"│  ⚠ Time Gap from VolumeBars:")
                    print(f"│    Missing last {delta_str}")
                    print(f"│")
            
            if bucket_fixed_ts_max and metric_fixed_ts_max:
                delta_ns = bucket_fixed_ts_max - metric_fixed_ts_max
                if delta_ns > 0:
                    delta_str = HelperFunctions.format_time_delta(delta_ns)
                    print(f"│")
                    print(f"│  ⚠ Time Gap from Fixed Buckets:")
                    print(f"│    Missing last {delta_str}")
                    print(f"│")
        
        
        # [ Interval Metric Section ]
        if "total_number_of_metric_interval_generated" in processing_results:
            print("\n┌─ METRICS (Interval) " + "─" * 58)
            
            actual_metrics = processing_results["total_number_of_metric_interval_generated"]
            
            bucket_count = processing_results.get("total_number_of_bucket_interval_generated", 0)
            expected_metrics = max(0, bucket_count - config.interval_metrics_buckets_per_metric)
            expected_metrics_raw = bucket_count - config.interval_metrics_buckets_per_metric
            
            metrics_match = actual_metrics == expected_metrics
            
            buckets_used = processing_results.get("num_buckets", 0)
            
            print(f"│")
            print(f"│  Calculation Context:")
            print(f"│    Buckets:                   {bucket_count:>15,}")
            print(f"│    interval_metrics_buckets_per_metric: {config.interval_metrics_buckets_per_metric:>8,}")
            print(f"│    → Metrics (raw):           {expected_metrics:>15,}  ({expected_metrics_raw:.2f})")
            print(f"│")
            print(f"│  Metrics Generated:")
            print(f"│    Expected:  {expected_metrics:>15,}  (buckets - lookback_window)")
            print(f"│    Actual:    {actual_metrics:>15,}  {'✓' if metrics_match else 'X MISMATCH'}")
            print(f"│")
            print(f"│  Buckets Used:     {buckets_used:>15,}")
            print(f"│  Lookback Window:  {config.interval_metrics_buckets_per_metric:>15,}")
            print(f"│")
            
            metric_interval_ts_min = processing_results.get("metric_interval_ts_min")
            metric_interval_ts_max = processing_results.get("metric_interval_ts_max")
            
            if metric_interval_ts_min and metric_interval_ts_max:
                date_min = pl.from_epoch([metric_interval_ts_min], time_unit="ns").dt.strftime("%A, %B %d, %Y at %H:%M UTC")[0]
                date_max = pl.from_epoch([metric_interval_ts_max], time_unit="ns").dt.strftime("%A, %B %d, %Y at %H:%M UTC")[0]
                
                print(f"│  Time Range:")
                print(f"│    Start:  {date_min}")
                print(f"│    End:    {date_max}")
                print(f"│")
            
            # Calculate deltas
            bucket_interval_ts_max = processing_results.get("bucket_interval_ts_max")
            
            if volumebar_ts_max and metric_interval_ts_max:
                delta_ns = volumebar_ts_max - metric_interval_ts_max
                if delta_ns > 0:
                    delta_str = HelperFunctions.format_time_delta(delta_ns)
                    print(f"│")
                    print(f"│  ⚠ Time Gap from VolumeBars:")
                    print(f"│    Missing last {delta_str}")
                    print(f"│")
            
            if bucket_interval_ts_max and metric_interval_ts_max:
                delta_ns = bucket_interval_ts_max - metric_interval_ts_max
                if delta_ns > 0:
                    delta_str = HelperFunctions.format_time_delta(delta_ns)
                    print(f"│")
                    print(f"│  ⚠ Time Gap from Interval Buckets:")
                    print(f"│    Missing last {delta_str}")
                    print(f"│")
        
        
        # [ Final Status ]
        print("\n" + "═" * 80)
        
        overall_success = HelperFunctions.evaluate_processing_results(
            processing_results, 
            initial_total_rows, 
            initial_total_size, 
            config
        )
        
        if overall_success:
            print("STATUS: ✓ ALL CHECKS PASSED".center(80))
        else:
            print("STATUS: X VALIDATION FAILED - See errors above".center(80))
        
        print("═" * 80 + "\n")




    @staticmethod
    def completion_sound():
        if sys.platform == 'darwin':
            os.system('afplay /System/Library/Sounds/Glass.aiff &')
        elif sys.platform == 'win32':
            import winsound
            winsound.PlaySound('SystemExclamation', winsound.SND_ALIAS)



