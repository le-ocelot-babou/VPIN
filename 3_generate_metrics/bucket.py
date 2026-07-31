
import numpy as np
from scipy import stats as scipy_stats

from typing import Optional, Dict, Any

from buffer import HelperFunctions, BUCKET_DTYPE, EPSILON


from enum import Enum
from dataclasses import dataclass
from numpy.typing import NDArray
from typing import Any, Callable, Dict, Optional, Union, List




# ============================================================================
# CALCULATION SPECIFICATION
# ============================================================================

class CalculationType(Enum):
    """Type of calculation timing."""
    FIXED = "fixed"       # Every Nth result
    INTERVAL = "interval" # Every result after warmup

    



class Bucket:


    # ============================================================================
    # Calculation Functions
    # ============================================================================


    @staticmethod
    def calculate_bucket_statistics(
        bars: np.ndarray, 
        bucket_id: np.uint32, 
        bucket_type: str
    ) -> Dict[str, Any]:
        """
        Calculate all summary statistics for a Volume Bucket from an array of VolumeBars.
        Uses vectorized NumPy operations for optimal performance.
        
        Args:
            bars: NumPy structured array of VolumeBars (VOLUMEBAR_DTYPE)
            bucket_id: Unique identifier for this bucket
            bucket_type: Type of bucket ('fixed', 'interval', 'adaptive')
            
        Returns:
            Dictionary with all bucket statistics
            
        <><><> MUST MATCH "BUCKET_DTYPE" SCHEMA <><><>
        
        Key Design Principles:
        - Bucket-level metrics: Calculate directly from aggregated raw data
        - Distribution metrics (std, range, skew): Use bar-level values
        - Delta metrics: Average bar-to-bar changes
        - Link transforms: Apply to bucket-level ratios; track std of bar-level transforms
        """
        
        current: Dict[str, Any] = {}
        num_bars = len(bars)
        
        if num_bars == 0:
            raise ValueError("Cannot calculate statistics for empty bucket")
        
        # ========================================================================
        # Section 2: Meta / Structural
        # ========================================================================
        
        current['id'] = bucket_id
        current['bucket_type'] = np.str_(bucket_type)
        current['num_bars'] = np.uint32(num_bars)
        current['all_bars_complete'] = np.bool_(np.all(bars['bar_complete']))
        current['bar_volume_size'] = bars['bar_volume_size'][0]
        current['contract_roll_any'] = np.bool_(np.any(bars['contract_roll']))
        current['contract_roll_count'] = np.uint32(np.sum(bars['contract_roll']))
        current['latest_instrument_id'] = bars['latest_instrument_id'][-1]
        current['start_ts_ns'] = np.min(bars['start_ts_ns'])
        current['end_ts_ns'] = np.max(bars['end_ts_ns'])
        current['time_elapsed_ns_total'] = current['end_ts_ns'] - current['start_ts_ns']
        current['gap_return_any'] = np.bool_(np.any(bars['gap_return']))
        current['gap_return_count'] = np.uint32(np.sum(bars['gap_return']))
        current['contains_oversized_order_any'] = np.bool_(np.any(bars['contains_oversized_order']))
        current['contains_oversized_order_count'] = np.uint32(np.sum(bars['contains_oversized_order']))
        
        # ========================================================================
        # Section 3: Order & Volume Statistics (Total/Ambiguous)
        # ========================================================================
        
        # Volume metrics
        current['volume_total_sum'] = np.sum(bars['volume_total'], dtype=np.uint64)
        current['volume_total_mean'] = np.float64(np.mean(bars['volume_total'].astype(np.float64)))
        current['volume_total_std'] = Bucket._safe_nanstd(bars['volume_total'].astype(np.float64), num_bars)
        
        # Order count metrics
        current['order_count_sum'] = np.sum(bars['order_count'], dtype=np.uint64)
        current['order_count_mean'] = np.float64(np.mean(bars['order_count'].astype(np.float64)))
        current['order_count_std'] = Bucket._safe_nanstd(bars['order_count'].astype(np.float64), num_bars)
        
        # Order splits
        current['order_splits_sum'] = np.sum(bars['order_splits'], dtype=np.uint64)
        current['order_splits_mean'] = np.float64(np.mean(bars['order_splits'].astype(np.float64)))
        
        # Volume log deltas
        current['delta_volume_total_log_mean'] = Bucket._safe_nanmean(bars['delta_volume_total_log'])
        current['delta_volume_total_log_std'] = Bucket._safe_nanstd(bars['delta_volume_total_log'], num_bars)
        current['delta_volume_total_log_max'] = Bucket._safe_nanmax(np.abs(bars['delta_volume_total_log'])) if num_bars > 0 else np.float64(np.nan)
        
        # Order count log deltas
        current['delta_order_count_log_mean'] = Bucket._safe_nanmean(bars['delta_order_count_log'])
        current['delta_order_count_log_std'] = Bucket._safe_nanstd(bars['delta_order_count_log'], num_bars)
        current['delta_order_count_log_max'] = Bucket._safe_nanmax(np.abs(bars['delta_order_count_log'])) if num_bars > 0 else np.float64(np.nan)
        
        # ========================================================================
        # Section 4: Active Metrics
        # ========================================================================
        
        # ========================================================================
        # 4.1 Active Order Counts
        # ========================================================================
        current['active_order_count_buy_sum'] = np.sum(bars['active_order_count_buy'], dtype=np.uint64)
        current['active_order_count_buy_mean'] = np.float64(np.mean(bars['active_order_count_buy'].astype(np.float64)))
        current['active_order_count_buy_std'] = Bucket._safe_nanstd(bars['active_order_count_buy'].astype(np.float64), num_bars)
        
        current['active_order_count_sell_sum'] = np.sum(bars['active_order_count_sell'], dtype=np.uint64)
        current['active_order_count_sell_mean'] = np.float64(np.mean(bars['active_order_count_sell'].astype(np.float64)))
        current['active_order_count_sell_std'] = Bucket._safe_nanstd(bars['active_order_count_sell'].astype(np.float64), num_bars)
        
        current['active_order_count_none_sum'] = np.sum(bars['active_order_count_none'], dtype=np.uint64)
        current['active_order_count_none_mean'] = np.float64(np.mean(bars['active_order_count_none'].astype(np.float64)))
        current['active_order_count_none_std'] = Bucket._safe_nanstd(bars['active_order_count_none'].astype(np.float64), num_bars)
        
        # ========================================================================
        # 4.2 Active Volumes
        # ========================================================================
        current['active_volume_buy_sum'] = np.sum(bars['active_volume_buy'], dtype=np.uint64)
        current['active_volume_buy_mean'] = np.float64(np.mean(bars['active_volume_buy'].astype(np.float64)))
        current['active_volume_buy_std'] = Bucket._safe_nanstd(bars['active_volume_buy'].astype(np.float64), num_bars)
        
        current['active_volume_sell_sum'] = np.sum(bars['active_volume_sell'], dtype=np.uint64)
        current['active_volume_sell_mean'] = np.float64(np.mean(bars['active_volume_sell'].astype(np.float64)))
        current['active_volume_sell_std'] = Bucket._safe_nanstd(bars['active_volume_sell'].astype(np.float64), num_bars)
        
        current['active_volume_none_sum'] = np.sum(bars['active_volume_none'], dtype=np.uint64)
        current['active_volume_none_mean'] = np.float64(np.mean(bars['active_volume_none'].astype(np.float64)))
        current['active_volume_none_std'] = Bucket._safe_nanstd(bars['active_volume_none'].astype(np.float64), num_bars)
        
        # ========================================================================
        # 4.3 Active Volume Log Deltas
        # ========================================================================
        current['delta_active_volume_buy_log_mean'] = Bucket._safe_nanmean(bars['delta_active_volume_buy_log'])
        current['delta_active_volume_buy_log_std'] = Bucket._safe_nanstd(bars['delta_active_volume_buy_log'], num_bars)
        current['delta_active_volume_buy_log_max'] = Bucket._safe_nanmax(np.abs(bars['delta_active_volume_buy_log'])) if num_bars > 0 else np.float64(np.nan)
        
        current['delta_active_volume_sell_log_mean'] = Bucket._safe_nanmean(bars['delta_active_volume_sell_log'])
        current['delta_active_volume_sell_log_std'] = Bucket._safe_nanstd(bars['delta_active_volume_sell_log'], num_bars)
        current['delta_active_volume_sell_log_max'] = Bucket._safe_nanmax(np.abs(bars['delta_active_volume_sell_log'])) if num_bars > 0 else np.float64(np.nan)
        
        current['delta_active_volume_none_log_mean'] = Bucket._safe_nanmean(bars['delta_active_volume_none_log'])
        current['delta_active_volume_none_log_std'] = Bucket._safe_nanstd(bars['delta_active_volume_none_log'], num_bars)
        current['delta_active_volume_none_log_max'] = Bucket._safe_nanmax(np.abs(bars['delta_active_volume_none_log'])) if num_bars > 0 else np.float64(np.nan)
        
        # ========================================================================
        # 4.4 Active Order Count Log Deltas
        # ========================================================================
        current['delta_active_order_count_buy_log_mean'] = Bucket._safe_nanmean(bars['delta_active_order_count_buy_log'])
        current['delta_active_order_count_buy_log_std'] = Bucket._safe_nanstd(bars['delta_active_order_count_buy_log'], num_bars)
        current['delta_active_order_count_buy_log_max'] = Bucket._safe_nanmax(np.abs(bars['delta_active_order_count_buy_log'])) if num_bars > 0 else np.float64(np.nan)
        
        current['delta_active_order_count_sell_log_mean'] = Bucket._safe_nanmean(bars['delta_active_order_count_sell_log'])
        current['delta_active_order_count_sell_log_std'] = Bucket._safe_nanstd(bars['delta_active_order_count_sell_log'], num_bars)
        current['delta_active_order_count_sell_log_max'] = Bucket._safe_nanmax(np.abs(bars['delta_active_order_count_sell_log'])) if num_bars > 0 else np.float64(np.nan)
        
        current['delta_active_order_count_none_log_mean'] = Bucket._safe_nanmean(bars['delta_active_order_count_none_log'])
        current['delta_active_order_count_none_log_std'] = Bucket._safe_nanstd(bars['delta_active_order_count_none_log'], num_bars)
        current['delta_active_order_count_none_log_max'] = Bucket._safe_nanmax(np.abs(bars['delta_active_order_count_none_log'])) if num_bars > 0 else np.float64(np.nan)
        
        # ========================================================================
        # 4.5 Active Imbalance Metrics
        # ========================================================================
        
        # Bucket-level imbalances from aggregated volumes
        active_buy_sum = np.float64(current['active_volume_buy_sum'])
        active_sell_sum = np.float64(current['active_volume_sell_sum'])
        active_total = active_buy_sum + active_sell_sum
        
        current['active_imbalance_signed'] = np.int64(current['active_volume_buy_sum']) - np.int64(current['active_volume_sell_sum'])
        current['active_imbalance_abs'] = np.uint64(np.abs(current['active_imbalance_signed']))
        
        if active_total > 0:
            current['active_imbalance_signed_ratio'] = np.float64(current['active_imbalance_signed']) / active_total
            current['active_imbalance_abs_ratio'] = np.float64(current['active_imbalance_abs']) / active_total
            current['active_imbalance_buy_ratio'] = active_buy_sum / active_total
        else:
            current['active_imbalance_signed_ratio'] = np.float64(np.nan)
            current['active_imbalance_abs_ratio'] = np.float64(np.nan)
            current['active_imbalance_buy_ratio'] = np.float64(np.nan)
        
        # Bar-level distribution statistics
        current['active_imbalance_signed_ratio_std'] = Bucket._safe_nanstd(bars['active_imbalance_signed_ratio'], num_bars)
        current['active_imbalance_abs_ratio_std'] = Bucket._safe_nanstd(bars['active_imbalance_abs_ratio'], num_bars)
        current['active_imbalance_buy_ratio_std'] = Bucket._safe_nanstd(bars['active_imbalance_buy_ratio'], num_bars)
        
        # ========================================================================
        # 4.6 Active Link Function Transforms
        # ========================================================================
        
        # Bucket-level transforms
        if not np.isnan(current['active_imbalance_signed_ratio']):
            signed_ratio_clipped = np.clip(current['active_imbalance_signed_ratio'], -0.9999, 0.9999)
            current['active_imbalance_signed_ratio_atanh'] = np.float64(np.arctanh(signed_ratio_clipped))
        else:
            current['active_imbalance_signed_ratio_atanh'] = np.float64(np.nan)
        
        if not np.isnan(current['active_imbalance_buy_ratio']):
            buy_ratio_clipped = np.clip(current['active_imbalance_buy_ratio'], 0.0001, 0.9999)
            current['active_imbalance_buy_ratio_logit'] = np.float64(np.log(buy_ratio_clipped / (1.0 - buy_ratio_clipped)))
        else:
            current['active_imbalance_buy_ratio_logit'] = np.float64(np.nan)
        
        if not np.isnan(current['active_imbalance_abs_ratio']):
            abs_ratio_clipped = np.clip(current['active_imbalance_abs_ratio'], 0.0001, 0.9999)
            current['active_imbalance_abs_ratio_logit'] = np.float64(np.log(abs_ratio_clipped / (1.0 - abs_ratio_clipped)))
        else:
            current['active_imbalance_abs_ratio_logit'] = np.float64(np.nan)
        
        # Bar-level transform std
        current['active_imbalance_signed_ratio_atanh_std'] = Bucket._safe_nanstd(bars['active_imbalance_signed_ratio_atanh'], num_bars)
        current['active_imbalance_buy_ratio_logit_std'] = Bucket._safe_nanstd(bars['active_imbalance_buy_ratio_logit'], num_bars)
        current['active_imbalance_abs_ratio_logit_std'] = Bucket._safe_nanstd(bars['active_imbalance_abs_ratio_logit'], num_bars)
        
        # ========================================================================
        # 4.7 Active Imbalance Deltas
        # ========================================================================
        current['delta_active_imbalance_signed_ratio_mean'] = Bucket._safe_nanmean(bars['delta_active_imbalance_signed_ratio'])
        current['delta_active_imbalance_signed_ratio_std'] = Bucket._safe_nanstd(bars['delta_active_imbalance_signed_ratio'], num_bars)
        current['delta_active_imbalance_buy_ratio_mean'] = Bucket._safe_nanmean(bars['delta_active_imbalance_buy_ratio'])
        current['delta_active_imbalance_buy_ratio_std'] = Bucket._safe_nanstd(bars['delta_active_imbalance_buy_ratio'], num_bars)
        
        # ========================================================================
        # 4.8 Active Derived Imbalance Metrics
        # ========================================================================
        current['active_cumulative_signed_imbalance'] = current['active_imbalance_signed']
        current['active_imbalance_persistence'] = Bucket._calculate_imbalance_persistence(
            bars['active_imbalance_signed'],
            current['active_imbalance_signed'],
            num_bars
        )
        current['active_imbalance_volatility'] = current['active_imbalance_signed_ratio_std']
        
        # ========================================================================
        # 4.9 Active Price Metrics (VWAP)
        # ========================================================================
        
        # Bucket-level VWAPs calculated from aggregated volumes
        if current['active_volume_buy_sum'] > 0:
            current['active_buy_vwap'] = np.float64(
                np.nansum(bars['active_buy_vwap'] * bars['active_volume_buy'].astype(np.float64)) / active_buy_sum
            )
        else:
            current['active_buy_vwap'] = np.float64(np.nan)
        
        if current['active_volume_sell_sum'] > 0:
            current['active_sell_vwap'] = np.float64(
                np.nansum(bars['active_sell_vwap'] * bars['active_volume_sell'].astype(np.float64)) / active_sell_sum
            )
        else:
            current['active_sell_vwap'] = np.float64(np.nan)
        
        active_none_sum = np.float64(current['active_volume_none_sum'])
        if current['active_volume_none_sum'] > 0:
            current['active_none_vwap'] = np.float64(
                np.nansum(bars['active_none_vwap'] * bars['active_volume_none'].astype(np.float64)) / active_none_sum
            )
        else:
            current['active_none_vwap'] = np.float64(np.nan)
        
        # Spread and midpoint
        if not np.isnan(current['active_buy_vwap']) and not np.isnan(current['active_sell_vwap']):
            current['active_spread_vwap'] = np.float64(current['active_buy_vwap'] - current['active_sell_vwap'])
            current['active_midpoint_vwap'] = np.float64((current['active_buy_vwap'] + current['active_sell_vwap']) * 0.5)
        else:
            current['active_spread_vwap'] = np.float64(np.nan)
            current['active_midpoint_vwap'] = np.float64(np.nan)
        
        # Bar-level statistics
        current['active_midpoint_vwap_std'] = Bucket._safe_nanstd(bars['active_midpoint_vwap'], num_bars)
        current['active_midpoint_vwap_range'] = Bucket._safe_nanmax(bars['active_midpoint_vwap']) - Bucket._safe_nanmin(bars['active_midpoint_vwap']) if num_bars > 0 else np.float64(0.0)
        current['active_spread_vwap_std'] = Bucket._safe_nanstd(bars['active_spread_vwap'], num_bars)
        current['active_spread_vwap_range'] = Bucket._safe_nanmax(bars['active_spread_vwap']) - Bucket._safe_nanmin(bars['active_spread_vwap']) if num_bars > 0 else np.float64(0.0)
        
        # ========================================================================
        # 4.10 Active Weighted Midpoints
        # ========================================================================
        
        # Bucket-level weighted midpoints
        if not np.isnan(current['active_midpoint_vwap']) and not np.isnan(current['active_spread_vwap']) and not np.isnan(current['active_imbalance_signed_ratio']):
            current['active_mid_imbalance_weighted'] = np.float64(
                current['active_midpoint_vwap'] + (current['active_spread_vwap'] * 0.5 * np.clip(current['active_imbalance_signed_ratio'], -1, 1))
            )
        else:
            current['active_mid_imbalance_weighted'] = np.float64(np.nan)
        
        if active_total > 0 and not np.isnan(current['active_buy_vwap']) and not np.isnan(current['active_sell_vwap']):
            # Flow-weighted: same-side weighting
            current['active_mid_flow_weighted'] = np.float64(
                (current['active_sell_vwap'] * active_sell_sum + current['active_buy_vwap'] * active_buy_sum) / active_total
            )
            # Aggressor-weighted: opposite-side weighting
            current['active_mid_aggressor_weighted'] = np.float64(
                (current['active_sell_vwap'] * active_buy_sum + current['active_buy_vwap'] * active_sell_sum) / active_total
            )
        else:
            current['active_mid_flow_weighted'] = np.float64(np.nan)
            current['active_mid_aggressor_weighted'] = np.float64(np.nan)
        
        # Bar-level statistics
        current['active_mid_imbalance_weighted_std'] = Bucket._safe_nanstd(bars['active_mid_imbalance_weighted'], num_bars)
        current['active_mid_flow_weighted_std'] = Bucket._safe_nanstd(bars['active_mid_flow_weighted'], num_bars)
        current['active_mid_aggressor_weighted_std'] = Bucket._safe_nanstd(bars['active_mid_aggressor_weighted'], num_bars)
        
        # ========================================================================
        # 4.11 Active Price Log Deltas
        # ========================================================================
        current['delta_active_midpoint_vwap_log_sum'] = np.float64(np.nansum(bars['delta_active_midpoint_vwap_log']))
        current['delta_active_midpoint_vwap_log_mean'] = Bucket._safe_nanmean(bars['delta_active_midpoint_vwap_log'])
        current['delta_active_midpoint_vwap_log_std'] = Bucket._safe_nanstd(bars['delta_active_midpoint_vwap_log'], num_bars)
        current['delta_active_midpoint_vwap_log_skew'] = Bucket._safe_skew(bars['delta_active_midpoint_vwap_log'], num_bars)
        
        current['delta_active_spread_vwap_log_mean'] = Bucket._safe_nanmean(bars['delta_active_spread_vwap_log'])
        current['delta_active_spread_vwap_log_std'] = Bucket._safe_nanstd(bars['delta_active_spread_vwap_log'], num_bars)
        
        current['delta_active_buy_vwap_log_mean'] = Bucket._safe_nanmean(bars['delta_active_buy_vwap_log'])
        current['delta_active_buy_vwap_log_std'] = Bucket._safe_nanstd(bars['delta_active_buy_vwap_log'], num_bars)
        
        current['delta_active_sell_vwap_log_mean'] = Bucket._safe_nanmean(bars['delta_active_sell_vwap_log'])
        current['delta_active_sell_vwap_log_std'] = Bucket._safe_nanstd(bars['delta_active_sell_vwap_log'], num_bars)
        
        # ========================================================================
        # 4.12 Active Weighted Midpoint Log Deltas
        # ========================================================================
        current['delta_active_mid_imbalance_weighted_log_sum'] = np.float64(np.nansum(bars['delta_active_mid_imbalance_weighted_log']))
        current['delta_active_mid_imbalance_weighted_log_mean'] = Bucket._safe_nanmean(bars['delta_active_mid_imbalance_weighted_log'])
        current['delta_active_mid_imbalance_weighted_log_std'] = Bucket._safe_nanstd(bars['delta_active_mid_imbalance_weighted_log'], num_bars)
        current['delta_active_mid_imbalance_weighted_log_skew'] = Bucket._safe_skew(bars['delta_active_mid_imbalance_weighted_log'], num_bars)
        
        current['delta_active_mid_flow_weighted_log_sum'] = np.float64(np.nansum(bars['delta_active_mid_flow_weighted_log']))
        current['delta_active_mid_flow_weighted_log_mean'] = Bucket._safe_nanmean(bars['delta_active_mid_flow_weighted_log'])
        current['delta_active_mid_flow_weighted_log_std'] = Bucket._safe_nanstd(bars['delta_active_mid_flow_weighted_log'], num_bars)
        current['delta_active_mid_flow_weighted_log_skew'] = Bucket._safe_skew(bars['delta_active_mid_flow_weighted_log'], num_bars)
        
        current['delta_active_mid_aggressor_weighted_log_sum'] = np.float64(np.nansum(bars['delta_active_mid_aggressor_weighted_log']))
        current['delta_active_mid_aggressor_weighted_log_mean'] = Bucket._safe_nanmean(bars['delta_active_mid_aggressor_weighted_log'])
        current['delta_active_mid_aggressor_weighted_log_std'] = Bucket._safe_nanstd(bars['delta_active_mid_aggressor_weighted_log'], num_bars)
        current['delta_active_mid_aggressor_weighted_log_skew'] = Bucket._safe_skew(bars['delta_active_mid_aggressor_weighted_log'], num_bars)
        
        # ========================================================================
        # 4.13 Active Price Range Metrics
        # ========================================================================
        
        # Bucket-level extremes
        current['active_buy_price_min'] = Bucket._safe_nanmin(bars['active_buy_price_min'])
        current['active_buy_price_max'] = Bucket._safe_nanmax(bars['active_buy_price_max'])
        current['active_buy_price_range'] = np.float64(current['active_buy_price_max'] - current['active_buy_price_min'])
        
        current['active_sell_price_min'] = Bucket._safe_nanmin(bars['active_sell_price_min'])
        current['active_sell_price_max'] = Bucket._safe_nanmax(bars['active_sell_price_max'])
        current['active_sell_price_range'] = np.float64(current['active_sell_price_max'] - current['active_sell_price_min'])
        
        current['active_none_price_min'] = Bucket._safe_nanmin(bars['active_none_price_min'])
        current['active_none_price_max'] = Bucket._safe_nanmax(bars['active_none_price_max'])
        current['active_none_price_range'] = np.float64(current['active_none_price_max'] - current['active_none_price_min'])
        
        # Bar-level range statistics
        current['active_buy_price_range_mean'] = Bucket._safe_nanmean(bars['active_buy_price_range'])
        current['active_buy_price_range_std'] = Bucket._safe_nanstd(bars['active_buy_price_range'], num_bars)
        current['active_sell_price_range_mean'] = Bucket._safe_nanmean(bars['active_sell_price_range'])
        current['active_sell_price_range_std'] = Bucket._safe_nanstd(bars['active_sell_price_range'], num_bars)
        current['active_none_price_range_mean'] = Bucket._safe_nanmean(bars['active_none_price_range'])
        current['active_none_price_range_std'] = Bucket._safe_nanstd(bars['active_none_price_range'], num_bars)
        
        # ========================================================================
        # 4.14 Active Pace Metrics
        # ========================================================================
        
        time_elapsed_total = np.float64(current['time_elapsed_ns_total'])
        
        # Bucket-level pace
        if time_elapsed_total > 0:
            current['active_buy_pace'] = active_buy_sum / time_elapsed_total
            current['active_sell_pace'] = active_sell_sum / time_elapsed_total
        else:
            current['active_buy_pace'] = np.float64(np.nan)
            current['active_sell_pace'] = np.float64(np.nan)
        
        # Bar-level statistics
        current['active_buy_pace_mean'] = Bucket._safe_nanmean(bars['active_buy_pace'])
        current['active_buy_pace_std'] = Bucket._safe_nanstd(bars['active_buy_pace'], num_bars)
        current['active_sell_pace_mean'] = Bucket._safe_nanmean(bars['active_sell_pace'])
        current['active_sell_pace_std'] = Bucket._safe_nanstd(bars['active_sell_pace'], num_bars)
        
        # Log-transformed pace deltas
        current['delta_active_buy_pace_log_mean'] = Bucket._safe_nanmean(bars['delta_active_buy_pace_log'])
        current['delta_active_buy_pace_log_std'] = Bucket._safe_nanstd(bars['delta_active_buy_pace_log'], num_bars)
        current['delta_active_sell_pace_log_mean'] = Bucket._safe_nanmean(bars['delta_active_sell_pace_log'])
        current['delta_active_sell_pace_log_std'] = Bucket._safe_nanstd(bars['delta_active_sell_pace_log'], num_bars)
        
        # ========================================================================
        # 4.15 Active N-Side Inferred Aggregations
        # ========================================================================
        current['active_none_inferred_buy_volume_sum'] = np.float64(np.sum(bars['active_none_inferred_buy_volume']))
        current['active_none_inferred_sell_volume_sum'] = np.float64(np.sum(bars['active_none_inferred_sell_volume']))
        
        inferred_buy_sum = current['active_none_inferred_buy_volume_sum']
        inferred_sell_sum = current['active_none_inferred_sell_volume_sum']
        
        if inferred_buy_sum > 0:
            current['active_none_inferred_buy_vwap'] = np.float64(
                np.nansum(bars['active_none_inferred_buy_vwap'] * bars['active_none_inferred_buy_volume']) / inferred_buy_sum
            )
        else:
            current['active_none_inferred_buy_vwap'] = np.float64(np.nan)
        
        if inferred_sell_sum > 0:
            current['active_none_inferred_sell_vwap'] = np.float64(
                np.nansum(bars['active_none_inferred_sell_vwap'] * bars['active_none_inferred_sell_volume']) / inferred_sell_sum
            )
        else:
            current['active_none_inferred_sell_vwap'] = np.float64(np.nan)
        
        # ========================================================================
        # Section 5: Adjusted Metrics
        # ========================================================================
        
        # 5.1 Adjusted Volumes
        current['adjusted_volume_buy_sum'] = np.float64(current['active_volume_buy_sum']) + current['active_none_inferred_buy_volume_sum']
        current['adjusted_volume_sell_sum'] = np.float64(current['active_volume_sell_sum']) + current['active_none_inferred_sell_volume_sum']
        
        current['adjusted_volume_buy_mean'] = Bucket._safe_nanmean(bars['adjusted_volume_buy'])
        current['adjusted_volume_buy_std'] = Bucket._safe_nanstd(bars['adjusted_volume_buy'], num_bars)
        current['adjusted_volume_sell_mean'] = Bucket._safe_nanmean(bars['adjusted_volume_sell'])
        current['adjusted_volume_sell_std'] = Bucket._safe_nanstd(bars['adjusted_volume_sell'], num_bars)
        
        # 5.2 Adjusted Imbalance Metrics
        adjusted_buy_sum = current['adjusted_volume_buy_sum']
        adjusted_sell_sum = current['adjusted_volume_sell_sum']
        volume_total_sum = np.float64(current['volume_total_sum'])
        
        current['adjusted_imbalance_signed'] = np.float64(adjusted_buy_sum - adjusted_sell_sum)
        current['adjusted_imbalance_abs'] = np.float64(np.abs(current['adjusted_imbalance_signed']))
        
        if volume_total_sum > 0:
            current['adjusted_imbalance_signed_ratio'] = current['adjusted_imbalance_signed'] / volume_total_sum
            current['adjusted_imbalance_abs_ratio'] = current['adjusted_imbalance_abs'] / volume_total_sum
            current['adjusted_imbalance_buy_ratio'] = adjusted_buy_sum / volume_total_sum
        else:
            current['adjusted_imbalance_signed_ratio'] = np.float64(np.nan)
            current['adjusted_imbalance_abs_ratio'] = np.float64(np.nan)
            current['adjusted_imbalance_buy_ratio'] = np.float64(np.nan)
        
        current['adjusted_imbalance_signed_ratio_std'] = Bucket._safe_nanstd(bars['adjusted_imbalance_signed_ratio'], num_bars)
        current['adjusted_imbalance_abs_ratio_std'] = Bucket._safe_nanstd(bars['adjusted_imbalance_abs_ratio'], num_bars)
        current['adjusted_imbalance_buy_ratio_std'] = Bucket._safe_nanstd(bars['adjusted_imbalance_buy_ratio'], num_bars)
        
        # 5.3 Adjusted Link Function Transforms
        if not np.isnan(current['adjusted_imbalance_signed_ratio']):
            signed_ratio_clipped = np.clip(current['adjusted_imbalance_signed_ratio'], -0.9999, 0.9999)
            current['adjusted_imbalance_signed_ratio_atanh'] = np.float64(np.arctanh(signed_ratio_clipped))
        else:
            current['adjusted_imbalance_signed_ratio_atanh'] = np.float64(np.nan)
        
        if not np.isnan(current['adjusted_imbalance_buy_ratio']):
            buy_ratio_clipped = np.clip(current['adjusted_imbalance_buy_ratio'], 0.0001, 0.9999)
            current['adjusted_imbalance_buy_ratio_logit'] = np.float64(np.log(buy_ratio_clipped / (1.0 - buy_ratio_clipped)))
        else:
            current['adjusted_imbalance_buy_ratio_logit'] = np.float64(np.nan)
        
        if not np.isnan(current['adjusted_imbalance_abs_ratio']):
            abs_ratio_clipped = np.clip(current['adjusted_imbalance_abs_ratio'], 0.0001, 0.9999)
            current['adjusted_imbalance_abs_ratio_logit'] = np.float64(np.log(abs_ratio_clipped / (1.0 - abs_ratio_clipped)))
        else:
            current['adjusted_imbalance_abs_ratio_logit'] = np.float64(np.nan)
        
        current['adjusted_imbalance_signed_ratio_atanh_std'] = Bucket._safe_nanstd(bars['adjusted_imbalance_signed_ratio_atanh'], num_bars)
        current['adjusted_imbalance_buy_ratio_logit_std'] = Bucket._safe_nanstd(bars['adjusted_imbalance_buy_ratio_logit'], num_bars)
        current['adjusted_imbalance_abs_ratio_logit_std'] = Bucket._safe_nanstd(bars['adjusted_imbalance_abs_ratio_logit'], num_bars)
        
        # 5.4 Adjusted Imbalance Deltas
        current['delta_adjusted_imbalance_signed_ratio_mean'] = Bucket._safe_nanmean(bars['delta_adjusted_imbalance_signed_ratio'])
        current['delta_adjusted_imbalance_signed_ratio_std'] = Bucket._safe_nanstd(bars['delta_adjusted_imbalance_signed_ratio'], num_bars)
        current['delta_adjusted_imbalance_buy_ratio_mean'] = Bucket._safe_nanmean(bars['delta_adjusted_imbalance_buy_ratio'])
        current['delta_adjusted_imbalance_buy_ratio_std'] = Bucket._safe_nanstd(bars['delta_adjusted_imbalance_buy_ratio'], num_bars)
        current['delta_adjusted_imbalance_signed_ratio_skew'] = Bucket._safe_skew(bars['delta_adjusted_imbalance_signed_ratio'], num_bars)
        
        # 5.5 Adjusted Derived Imbalance Metrics
        current['adjusted_cumulative_signed_imbalance'] = current['adjusted_imbalance_signed']
        current['adjusted_imbalance_persistence'] = Bucket._calculate_imbalance_persistence(
            bars['adjusted_imbalance_signed'],
            current['adjusted_imbalance_signed'],
            num_bars
        )
        current['adjusted_imbalance_volatility'] = current['adjusted_imbalance_signed_ratio_std']
        
        # 5.6 Adjusted Price Metrics (VWAP)
        if adjusted_buy_sum > 0:
            active_buy_contrib = current['active_buy_vwap'] * active_buy_sum if not np.isnan(current['active_buy_vwap']) else 0.0
            inferred_buy_contrib = current['active_none_inferred_buy_vwap'] * inferred_buy_sum if not np.isnan(current['active_none_inferred_buy_vwap']) else 0.0
            current['adjusted_buy_vwap'] = np.float64((active_buy_contrib + inferred_buy_contrib) / adjusted_buy_sum)
        else:
            current['adjusted_buy_vwap'] = np.float64(np.nan)
        
        if adjusted_sell_sum > 0:
            active_sell_contrib = current['active_sell_vwap'] * active_sell_sum if not np.isnan(current['active_sell_vwap']) else 0.0
            inferred_sell_contrib = current['active_none_inferred_sell_vwap'] * inferred_sell_sum if not np.isnan(current['active_none_inferred_sell_vwap']) else 0.0
            current['adjusted_sell_vwap'] = np.float64((active_sell_contrib + inferred_sell_contrib) / adjusted_sell_sum)
        else:
            current['adjusted_sell_vwap'] = np.float64(np.nan)
        
        if not np.isnan(current['adjusted_buy_vwap']) and not np.isnan(current['adjusted_sell_vwap']):
            current['adjusted_spread_vwap'] = np.float64(current['adjusted_buy_vwap'] - current['adjusted_sell_vwap'])
            current['adjusted_midpoint_vwap'] = np.float64((current['adjusted_buy_vwap'] + current['adjusted_sell_vwap']) * 0.5)
        else:
            current['adjusted_spread_vwap'] = np.float64(np.nan)
            current['adjusted_midpoint_vwap'] = np.float64(np.nan)
        
        current['adjusted_midpoint_vwap_std'] = Bucket._safe_nanstd(bars['adjusted_midpoint_vwap'], num_bars)
        current['adjusted_midpoint_vwap_range'] = Bucket._safe_nanmax(bars['adjusted_midpoint_vwap']) - Bucket._safe_nanmin(bars['adjusted_midpoint_vwap']) if num_bars > 0 else np.float64(0.0)
        current['adjusted_spread_vwap_std'] = Bucket._safe_nanstd(bars['adjusted_spread_vwap'], num_bars)
        current['adjusted_spread_vwap_range'] = Bucket._safe_nanmax(bars['adjusted_spread_vwap']) - Bucket._safe_nanmin(bars['adjusted_spread_vwap']) if num_bars > 0 else np.float64(0.0)
        
        # 5.7 Adjusted Weighted Midpoints
        adjusted_total = adjusted_buy_sum + adjusted_sell_sum
        
        if not np.isnan(current['adjusted_midpoint_vwap']) and not np.isnan(current['adjusted_spread_vwap']) and not np.isnan(current['adjusted_imbalance_signed_ratio']):
            current['adjusted_mid_imbalance_weighted'] = np.float64(
                current['adjusted_midpoint_vwap'] + (current['adjusted_spread_vwap'] * 0.5 * np.clip(current['adjusted_imbalance_signed_ratio'], -1, 1))
            )
        else:
            current['adjusted_mid_imbalance_weighted'] = np.float64(np.nan)
        
        if adjusted_total > 0 and not np.isnan(current['adjusted_buy_vwap']) and not np.isnan(current['adjusted_sell_vwap']):
            current['adjusted_mid_flow_weighted'] = np.float64(
                (current['adjusted_sell_vwap'] * adjusted_sell_sum + current['adjusted_buy_vwap'] * adjusted_buy_sum) / adjusted_total
            )
            current['adjusted_mid_aggressor_weighted'] = np.float64(
                (current['adjusted_sell_vwap'] * adjusted_buy_sum + current['adjusted_buy_vwap'] * adjusted_sell_sum) / adjusted_total
            )
        else:
            current['adjusted_mid_flow_weighted'] = np.float64(np.nan)
            current['adjusted_mid_aggressor_weighted'] = np.float64(np.nan)
        
        current['adjusted_mid_imbalance_weighted_std'] = Bucket._safe_nanstd(bars['adjusted_mid_imbalance_weighted'], num_bars)
        current['adjusted_mid_flow_weighted_std'] = Bucket._safe_nanstd(bars['adjusted_mid_flow_weighted'], num_bars)
        current['adjusted_mid_aggressor_weighted_std'] = Bucket._safe_nanstd(bars['adjusted_mid_aggressor_weighted'], num_bars)
        
        # 5.8-5.9 Adjusted Price Log Deltas (continuing pattern)
        current['delta_adjusted_midpoint_vwap_log_sum'] = np.float64(np.nansum(bars['delta_adjusted_midpoint_vwap_log']))
        current['delta_adjusted_midpoint_vwap_log_mean'] = Bucket._safe_nanmean(bars['delta_adjusted_midpoint_vwap_log'])
        current['delta_adjusted_midpoint_vwap_log_std'] = Bucket._safe_nanstd(bars['delta_adjusted_midpoint_vwap_log'], num_bars)
        current['delta_adjusted_midpoint_vwap_log_skew'] = Bucket._safe_skew(bars['delta_adjusted_midpoint_vwap_log'], num_bars)
        current['delta_adjusted_spread_vwap_log_mean'] = Bucket._safe_nanmean(bars['delta_adjusted_spread_vwap_log'])
        current['delta_adjusted_spread_vwap_log_std'] = Bucket._safe_nanstd(bars['delta_adjusted_spread_vwap_log'], num_bars)
        current['delta_adjusted_buy_vwap_log_mean'] = Bucket._safe_nanmean(bars['delta_adjusted_buy_vwap_log'])
        current['delta_adjusted_buy_vwap_log_std'] = Bucket._safe_nanstd(bars['delta_adjusted_buy_vwap_log'], num_bars)
        current['delta_adjusted_sell_vwap_log_mean'] = Bucket._safe_nanmean(bars['delta_adjusted_sell_vwap_log'])
        current['delta_adjusted_sell_vwap_log_std'] = Bucket._safe_nanstd(bars['delta_adjusted_sell_vwap_log'], num_bars)
        
        current['delta_adjusted_mid_imbalance_weighted_log_sum'] = np.float64(np.nansum(bars['delta_adjusted_mid_imbalance_weighted_log']))
        current['delta_adjusted_mid_imbalance_weighted_log_mean'] = Bucket._safe_nanmean(bars['delta_adjusted_mid_imbalance_weighted_log'])
        current['delta_adjusted_mid_imbalance_weighted_log_std'] = Bucket._safe_nanstd(bars['delta_adjusted_mid_imbalance_weighted_log'], num_bars)
        current['delta_adjusted_mid_imbalance_weighted_log_skew'] = Bucket._safe_skew(bars['delta_adjusted_mid_imbalance_weighted_log'], num_bars)
        current['delta_adjusted_mid_flow_weighted_log_sum'] = np.float64(np.nansum(bars['delta_adjusted_mid_flow_weighted_log']))
        current['delta_adjusted_mid_flow_weighted_log_mean'] = Bucket._safe_nanmean(bars['delta_adjusted_mid_flow_weighted_log'])
        current['delta_adjusted_mid_flow_weighted_log_std'] = Bucket._safe_nanstd(bars['delta_adjusted_mid_flow_weighted_log'], num_bars)
        current['delta_adjusted_mid_flow_weighted_log_skew'] = Bucket._safe_skew(bars['delta_adjusted_mid_flow_weighted_log'], num_bars)
        current['delta_adjusted_mid_aggressor_weighted_log_sum'] = np.float64(np.nansum(bars['delta_adjusted_mid_aggressor_weighted_log']))
        current['delta_adjusted_mid_aggressor_weighted_log_mean'] = Bucket._safe_nanmean(bars['delta_adjusted_mid_aggressor_weighted_log'])
        current['delta_adjusted_mid_aggressor_weighted_log_std'] = Bucket._safe_nanstd(bars['delta_adjusted_mid_aggressor_weighted_log'], num_bars)
        current['delta_adjusted_mid_aggressor_weighted_log_skew'] = Bucket._safe_skew(bars['delta_adjusted_mid_aggressor_weighted_log'], num_bars)
        
        # ========================================================================
        # Section 6: Passive Metrics
        # ========================================================================
        
        # 6.1 Passive Midprice
        if volume_total_sum > 0:
            current['passive_midprice'] = np.float64(
                np.nansum(bars['passive_midprice'] * bars['volume_total'].astype(np.float64)) / volume_total_sum
            )
        else:
            current['passive_midprice'] = np.float64(np.nan)
        
        current['passive_midprice_std'] = Bucket._safe_nanstd(bars['passive_midprice'], num_bars)
        current['passive_midprice_range'] = Bucket._safe_nanmax(bars['passive_midprice']) - np.nanmin(bars['passive_midprice']) if num_bars > 0 else np.float64(0.0)
        
        # 6.2 Passive Midprice Deltas
        current['delta_passive_midprice_log_sum'] = np.float64(np.nansum(bars['passive_midprice_delta_log']))
        current['delta_passive_midprice_log_mean'] = Bucket._safe_nanmean(bars['passive_midprice_delta_log'])
        current['delta_passive_midprice_log_std'] = Bucket._safe_nanstd(bars['passive_midprice_delta_log'], num_bars)
        current['delta_passive_midprice_log_skew'] = Bucket._safe_skew(bars['passive_midprice_delta_log'], num_bars)
        
        # 6.3 Passive CDF Statistics
        current['passive_midprice_delta_cdf_mean'] = Bucket._safe_nanmean(bars['passive_midprice_delta_cdf'])
        current['passive_midprice_delta_cdf_std'] = Bucket._safe_nanstd(bars['passive_midprice_delta_cdf'], num_bars)
        
        # 6.4 Passive Volume Classifications
        current['passive_buy_volume_sum'] = np.float64(np.sum(bars['passive_buy_volume']))
        current['passive_sell_volume_sum'] = np.float64(np.sum(bars['passive_sell_volume']))
        current['passive_buy_volume_mean'] = Bucket._safe_nanmean(bars['passive_buy_volume'])
        current['passive_buy_volume_std'] = Bucket._safe_nanstd(bars['passive_buy_volume'], num_bars)
        current['passive_sell_volume_mean'] = Bucket._safe_nanmean(bars['passive_sell_volume'])
        current['passive_sell_volume_std'] = Bucket._safe_nanstd(bars['passive_sell_volume'], num_bars)
        
        # 6.5 Passive Imbalance Metrics
        passive_buy_sum = current['passive_buy_volume_sum']
        passive_sell_sum = current['passive_sell_volume_sum']
        
        current['passive_imbalance_signed'] = np.float64(passive_buy_sum - passive_sell_sum)
        current['passive_imbalance_abs'] = np.float64(np.abs(current['passive_imbalance_signed']))
        
        if volume_total_sum > 0:
            current['passive_imbalance_signed_ratio'] = current['passive_imbalance_signed'] / volume_total_sum
            current['passive_imbalance_abs_ratio'] = current['passive_imbalance_abs'] / volume_total_sum
            current['passive_imbalance_buy_ratio'] = passive_buy_sum / volume_total_sum
        else:
            current['passive_imbalance_signed_ratio'] = np.float64(np.nan)
            current['passive_imbalance_abs_ratio'] = np.float64(np.nan)
            current['passive_imbalance_buy_ratio'] = np.float64(np.nan)
        
        current['passive_imbalance_signed_ratio_std'] = Bucket._safe_nanstd(bars['passive_imbalance_signed_ratio'], num_bars)
        current['passive_imbalance_abs_ratio_std'] = Bucket._safe_nanstd(bars['passive_imbalance_abs_ratio'], num_bars)
        current['passive_imbalance_buy_ratio_std'] = Bucket._safe_nanstd(bars['passive_imbalance_buy_ratio'], num_bars)
        
        # 6.6 Passive Link Function Transforms
        if not np.isnan(current['passive_imbalance_signed_ratio']):
            signed_ratio_clipped = np.clip(current['passive_imbalance_signed_ratio'], -0.9999, 0.9999)
            current['passive_imbalance_signed_ratio_atanh'] = np.float64(np.arctanh(signed_ratio_clipped))
        else:
            current['passive_imbalance_signed_ratio_atanh'] = np.float64(np.nan)
        
        if not np.isnan(current['passive_imbalance_buy_ratio']):
            buy_ratio_clipped = np.clip(current['passive_imbalance_buy_ratio'], 0.0001, 0.9999)
            current['passive_imbalance_buy_ratio_logit'] = np.float64(np.log(buy_ratio_clipped / (1.0 - buy_ratio_clipped)))
        else:
            current['passive_imbalance_buy_ratio_logit'] = np.float64(np.nan)
        
        if not np.isnan(current['passive_imbalance_abs_ratio']):
            abs_ratio_clipped = np.clip(current['passive_imbalance_abs_ratio'], 0.0001, 0.9999)
            current['passive_imbalance_abs_ratio_logit'] = np.float64(np.log(abs_ratio_clipped / (1.0 - abs_ratio_clipped)))
        else:
            current['passive_imbalance_abs_ratio_logit'] = np.float64(np.nan)
        
        current['passive_imbalance_signed_ratio_atanh_std'] = Bucket._safe_nanstd(bars['passive_imbalance_signed_ratio_atanh'], num_bars)
        current['passive_imbalance_buy_ratio_logit_std'] = Bucket._safe_nanstd(bars['passive_imbalance_buy_ratio_logit'], num_bars)
        current['passive_imbalance_abs_ratio_logit_std'] = Bucket._safe_nanstd(bars['passive_imbalance_abs_ratio_logit'], num_bars)
        
        # 6.7 Passive Imbalance Deltas
        current['delta_passive_imbalance_signed_ratio_mean'] = Bucket._safe_nanmean(bars['delta_passive_imbalance_signed_ratio'])
        current['delta_passive_imbalance_signed_ratio_std'] = Bucket._safe_nanstd(bars['delta_passive_imbalance_signed_ratio'], num_bars)
        current['delta_passive_imbalance_signed_ratio_skew'] = Bucket._safe_skew(bars['delta_passive_imbalance_signed_ratio'], num_bars)
        current['delta_passive_imbalance_buy_ratio_mean'] = Bucket._safe_nanmean(bars['delta_passive_imbalance_buy_ratio'])
        current['delta_passive_imbalance_buy_ratio_std'] = Bucket._safe_nanstd(bars['delta_passive_imbalance_buy_ratio'], num_bars)
        
        # 6.8 Passive Derived Imbalance Metrics
        current['passive_cumulative_signed_imbalance'] = current['passive_imbalance_signed']
        current['passive_imbalance_persistence'] = Bucket._calculate_imbalance_persistence(
            bars['passive_imbalance_signed'],
            current['passive_imbalance_signed'],
            num_bars
        )
        current['passive_imbalance_volatility'] = current['passive_imbalance_signed_ratio_std']
        
        # ========================================================================
        # Section 7: Divergence Metrics
        # ========================================================================
        
        # 7.1 Aggregated Bar-Level Divergences
        current['divergence_buy_volume_mean'] = Bucket._safe_nanmean(bars['divergence_buy_volume'])
        current['divergence_buy_volume_std'] = Bucket._safe_nanstd(bars['divergence_buy_volume'], num_bars)
        current['divergence_sell_volume_mean'] = Bucket._safe_nanmean(bars['divergence_sell_volume'])
        current['divergence_sell_volume_std'] = Bucket._safe_nanstd(bars['divergence_sell_volume'], num_bars)
        current['divergence_imbalance_signed_mean'] = Bucket._safe_nanmean(bars['divergence_imbalance_signed'])
        current['divergence_imbalance_signed_std'] = Bucket._safe_nanstd(bars['divergence_imbalance_signed'], num_bars)
        current['divergence_imbalance_signed_ratio_mean'] = Bucket._safe_nanmean(bars['divergence_imbalance_signed_ratio'])
        current['divergence_imbalance_signed_ratio_std'] = Bucket._safe_nanstd(bars['divergence_imbalance_signed_ratio'], num_bars)
        current['divergence_buy_ratio_mean'] = Bucket._safe_nanmean(bars['divergence_buy_ratio'])
        current['divergence_buy_ratio_std'] = Bucket._safe_nanstd(bars['divergence_buy_ratio'], num_bars)
        
        # 7.2 Bucket-Level Divergences
        current['divergence_buy_volume_bucket'] = np.float64(current['active_volume_buy_sum']) - current['passive_buy_volume_sum']
        current['divergence_sell_volume_bucket'] = np.float64(current['active_volume_sell_sum']) - current['passive_sell_volume_sum']
        current['divergence_imbalance_signed_bucket'] = np.float64(current['active_imbalance_signed']) - current['passive_imbalance_signed']
        
        if not np.isnan(current['active_imbalance_signed_ratio']) and not np.isnan(current['passive_imbalance_signed_ratio']):
            current['divergence_imbalance_signed_ratio_bucket'] = current['active_imbalance_signed_ratio'] - current['passive_imbalance_signed_ratio']
        else:
            current['divergence_imbalance_signed_ratio_bucket'] = np.float64(np.nan)
        
        if not np.isnan(current['active_imbalance_buy_ratio']) and not np.isnan(current['passive_imbalance_buy_ratio']):
            current['divergence_buy_ratio_bucket'] = current['active_imbalance_buy_ratio'] - current['passive_imbalance_buy_ratio']
        else:
            current['divergence_buy_ratio_bucket'] = np.float64(np.nan)
        
        # 7.3 Divergence Deltas
        current['delta_divergence_buy_volume_mean'] = Bucket._safe_nanmean(bars['delta_divergence_buy_volume'])
        current['delta_divergence_sell_volume_mean'] = Bucket._safe_nanmean(bars['delta_divergence_sell_volume'])
        current['delta_divergence_imbalance_signed_ratio_mean'] = Bucket._safe_nanmean(bars['delta_divergence_imbalance_signed_ratio'])
        current['delta_divergence_buy_ratio_mean'] = Bucket._safe_nanmean(bars['delta_divergence_buy_ratio'])
        
        # ========================================================================
        # Section 8: Temporal / Tempo Structure
        # ========================================================================
        
        # 8.1 Time Elapsed Metrics
        current['time_elapsed_ns_mean'] = np.float64(np.mean(bars['time_elapsed_ns'].astype(np.float64)))
        current['time_elapsed_ns_std'] = Bucket._safe_nanstd(bars['time_elapsed_ns'].astype(np.float64), num_bars)
        current['time_elapsed_ns_min'] = np.min(bars['time_elapsed_ns'])
        current['time_elapsed_ns_max'] = np.max(bars['time_elapsed_ns'])
        
        # 8.2 Pace of Contracts Traded
        if time_elapsed_total > 0:
            current['pace_of_contracts_traded'] = volume_total_sum / time_elapsed_total
        else:
            current['pace_of_contracts_traded'] = np.float64(np.nan)
        
        current['pace_of_contracts_traded_mean'] = Bucket._safe_nanmean(bars['pace_of_contracts_traded'])
        current['pace_of_contracts_traded_std'] = Bucket._safe_nanstd(bars['pace_of_contracts_traded'], num_bars)
        
        # 8.3 Temporal Log Deltas
        current['delta_time_elapsed_ns_log_mean'] = Bucket._safe_nanmean(bars['delta_time_elapsed_ns_log'])
        current['delta_time_elapsed_ns_log_std'] = Bucket._safe_nanstd(bars['delta_time_elapsed_ns_log'], num_bars)
        current['delta_pace_of_contracts_traded_log_mean'] = Bucket._safe_nanmean(bars['delta_pace_of_contracts_traded_log'])
        current['delta_pace_of_contracts_traded_log_std'] = Bucket._safe_nanstd(bars['delta_pace_of_contracts_traded_log'], num_bars)
        
        # 8.4 Derived Tempo Metrics
        current['tempo_stability'] = current['delta_time_elapsed_ns_log_std']
        
        if num_bars > 1:
            pace_log_deltas = bars['delta_pace_of_contracts_traded_log']
            valid_mask = ~np.isnan(pace_log_deltas)
            if np.sum(valid_mask) > 1:
                x = np.arange(num_bars, dtype=np.float64)[valid_mask]
                y = pace_log_deltas[valid_mask]
                x_mean = np.mean(x)
                y_mean = np.mean(y)
                numerator = np.sum((x - x_mean) * (y - y_mean))
                denominator = np.sum((x - x_mean) ** 2)
                if denominator > EPSILON:
                    current['tempo_acceleration'] = np.float64(numerator / denominator)
                else:
                    current['tempo_acceleration'] = np.float64(0.0)
            else:
                current['tempo_acceleration'] = np.float64(0.0)
        else:
            current['tempo_acceleration'] = np.float64(0.0)
        
        # ========================================================================
        # Section 9: Directional Signals
        # ========================================================================
        
        # 9.1 Active Directional Signals
        active_midpoints = bars['active_midpoint_vwap']
        if num_bars > 1:
            active_directions = np.zeros(num_bars, dtype=np.bool_)
            active_directions[1:] = active_midpoints[1:] > active_midpoints[:-1]
        else:
            active_directions = np.zeros(num_bars, dtype=np.bool_)
        
        active_dir_signals = Bucket._calculate_directional_signals(active_directions, num_bars)
        current['active_pct_bars_positive_direction'] = active_dir_signals['pct_positive']
        current['active_directional_streak_max'] = active_dir_signals['streak_max']
        current['active_directional_reversals_count'] = active_dir_signals['reversals_count']
        current['active_net_direction'] = active_dir_signals['net_direction']
        
        # 9.2 Adjusted Directional Signals
        adjusted_midpoints = bars['adjusted_midpoint_vwap']
        if num_bars > 1:
            adjusted_directions = np.zeros(num_bars, dtype=np.bool_)
            adjusted_directions[1:] = adjusted_midpoints[1:] > adjusted_midpoints[:-1]
        else:
            adjusted_directions = np.zeros(num_bars, dtype=np.bool_)
        
        adjusted_dir_signals = Bucket._calculate_directional_signals(adjusted_directions, num_bars)
        current['adjusted_pct_bars_positive_direction'] = adjusted_dir_signals['pct_positive']
        current['adjusted_directional_streak_max'] = adjusted_dir_signals['streak_max']
        current['adjusted_directional_reversals_count'] = adjusted_dir_signals['reversals_count']
        current['adjusted_net_direction'] = adjusted_dir_signals['net_direction']
        
        # 9.3 Passive Directional Signals
        passive_midprices = bars['passive_midprice']
        if num_bars > 1:
            passive_directions = np.zeros(num_bars, dtype=np.bool_)
            passive_directions[1:] = passive_midprices[1:] > passive_midprices[:-1]
        else:
            passive_directions = np.zeros(num_bars, dtype=np.bool_)
        
        passive_dir_signals = Bucket._calculate_directional_signals(passive_directions, num_bars)
        current['passive_pct_bars_positive_direction'] = passive_dir_signals['pct_positive']
        current['passive_directional_streak_max'] = passive_dir_signals['streak_max']
        current['passive_directional_reversals_count'] = passive_dir_signals['reversals_count']
        current['passive_net_direction'] = passive_dir_signals['net_direction']
        
        # ========================================================================
        # Section 10: Volatility and Stability Overlays
        # ========================================================================
        current['active_price_volatility'] = current['delta_active_midpoint_vwap_log_std']
        current['active_spread_volatility'] = current['delta_active_spread_vwap_log_std']
        current['active_return_volatility'] = current['active_price_volatility']
        current['active_intermediate_volatility'] = current['active_price_volatility']
        current['active_spread_stability'] = current['active_spread_volatility']
        current['active_flow_volatility'] = current['delta_volume_total_log_std']
        current['adjusted_price_volatility'] = current['delta_adjusted_midpoint_vwap_log_std']
        current['adjusted_spread_volatility'] = current['delta_adjusted_spread_vwap_log_std']
        current['passive_price_volatility'] = current['delta_passive_midprice_log_std']
        
        # ========================================================================
        # Section 12: Debug / Diagnostic
        # ========================================================================
        current['bar_ids_checksum'] = np.sum(bars['id'], dtype=np.uint64)
        current['missing_bars_flag'] = np.bool_(False)
        current['processing_time_ns'] = np.uint64(0)
        current['nan_count_active_midpoint'] = np.uint32(np.sum(np.isnan(bars['active_midpoint_vwap'])))
        current['nan_count_adjusted_midpoint'] = np.uint32(np.sum(np.isnan(bars['adjusted_midpoint_vwap'])))
        current['nan_count_passive_midprice'] = np.uint32(np.sum(np.isnan(bars['passive_midprice'])))

        
        return current




    @staticmethod
    def calculate_bucket_deltas(
        current: Dict[str, Any], 
        previous: Dict[str, Any],
        previous_deltas: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Calculate inter-bucket delta statistics by comparing current bucket to previous bucket.
        
        Args:
            current: Dictionary of current bucket statistics
            previous: Dictionary of previous bucket statistics
            previous_deltas: Optional dictionary of previous bucket's deltas (for second-order)
            
        Returns:
            Dictionary with all inter-bucket delta statistics
        """
        
        deltas: Dict[str, Any] = {}
        
        # Active spread changes
        deltas['bucket_delta_active_spread_vwap_log'] = Bucket._safe_log_ratio(
            current['active_spread_vwap'], previous['active_spread_vwap']
        )
        deltas['bucket_delta_active_spread_vwap_std_log'] = Bucket._safe_log_ratio(
            current['active_spread_vwap_std'], previous['active_spread_vwap_std']
        )
        
        if not np.isnan(current['active_spread_volatility']) and not np.isnan(previous['active_spread_volatility']):
            deltas['bucket_delta_active_spread_volatility'] = np.float64(
                current['active_spread_volatility'] - previous['active_spread_volatility']
            )
        else:
            deltas['bucket_delta_active_spread_volatility'] = np.float64(np.nan)
        
        # Volatility changes
        deltas['bucket_delta_active_return_volatility_log'] = Bucket._safe_log_ratio(
            current['active_return_volatility'], previous['active_return_volatility']
        )
        deltas['bucket_delta_active_intermediate_volatility_log'] = Bucket._safe_log_ratio(
            current['active_intermediate_volatility'], previous['active_intermediate_volatility']
        )
        deltas['bucket_delta_active_flow_volatility_log'] = Bucket._safe_log_ratio(
            current['active_flow_volatility'], previous['active_flow_volatility']
        )
        
        # Tempo changes
        deltas['bucket_delta_active_pace_of_contracts_traded_log'] = Bucket._safe_log_ratio(
            current['pace_of_contracts_traded'], previous['pace_of_contracts_traded']
        )
        
        if not np.isnan(current['tempo_stability']) and not np.isnan(previous['tempo_stability']):
            deltas['bucket_delta_active_tempo_stability'] = np.float64(
                current['tempo_stability'] - previous['tempo_stability']
            )
        else:
            deltas['bucket_delta_active_tempo_stability'] = np.float64(np.nan)
        
        current_time = np.float64(current['time_elapsed_ns_total'])
        previous_time = np.float64(previous['time_elapsed_ns_total'])
        if current_time > 0 and previous_time > 0:
            deltas['bucket_delta_active_time_elapsed_ns_total_log'] = np.float64(
                np.log((current_time + EPSILON) / (previous_time + EPSILON))
            )
        else:
            deltas['bucket_delta_active_time_elapsed_ns_total_log'] = np.float64(np.nan)
        
        # Imbalance changes
        deltas['bucket_delta_active_imbalance_signed_ratio_atanh'] = Bucket._safe_atanh_diff(
            current['active_imbalance_signed_ratio'], previous['active_imbalance_signed_ratio']
        )
        deltas['bucket_delta_active_imbalance_abs_ratio_logit'] = Bucket._safe_logit_diff(
            current['active_imbalance_abs_ratio'], previous['active_imbalance_abs_ratio']
        )
        
        if not np.isnan(current['active_imbalance_persistence']) and not np.isnan(previous['active_imbalance_persistence']):
            deltas['bucket_delta_active_imbalance_persistence'] = np.float64(
                current['active_imbalance_persistence'] - previous['active_imbalance_persistence']
            )
        else:
            deltas['bucket_delta_active_imbalance_persistence'] = np.float64(np.nan)
        
        current_volume_total = np.float64(current['volume_total_sum'])
        previous_volume_total = np.float64(previous['volume_total_sum'])
        if current_volume_total > 0 and previous_volume_total > 0:
            current_cumul_norm = np.float64(current['active_cumulative_signed_imbalance']) / (current_volume_total + EPSILON)
            previous_cumul_norm = np.float64(previous['active_cumulative_signed_imbalance']) / (previous_volume_total + EPSILON)
            deltas['bucket_delta_active_cumulative_signed_imbalance_norm'] = np.float64(current_cumul_norm - previous_cumul_norm)
        else:
            deltas['bucket_delta_active_cumulative_signed_imbalance_norm'] = np.float64(np.nan)
        
        if not np.isnan(current['active_imbalance_volatility']) and not np.isnan(previous['active_imbalance_volatility']):
            deltas['bucket_delta_active_imbalance_volatility'] = np.float64(
                current['active_imbalance_volatility'] - previous['active_imbalance_volatility']
            )
        else:
            deltas['bucket_delta_active_imbalance_volatility'] = np.float64(np.nan)
        
        # Directional changes
        if not np.isnan(current['active_pct_bars_positive_direction']) and not np.isnan(previous['active_pct_bars_positive_direction']):
            deltas['bucket_delta_active_pct_bars_positive_direction'] = np.float64(
                current['active_pct_bars_positive_direction'] - previous['active_pct_bars_positive_direction']
            )
        else:
            deltas['bucket_delta_active_pct_bars_positive_direction'] = np.float64(np.nan)
        
        deltas['bucket_delta_active_directional_streak_max'] = np.int32(
            np.int64(current['active_directional_streak_max']) - np.int64(previous['active_directional_streak_max'])
        )
        deltas['bucket_delta_active_directional_reversals_count'] = np.int32(
            np.int64(current['active_directional_reversals_count']) - np.int64(previous['active_directional_reversals_count'])
        )
        
        # Adjusted inter-bucket deltas
        deltas['bucket_delta_adjusted_spread_vwap_log'] = Bucket._safe_log_ratio(
            current['adjusted_spread_vwap'], previous['adjusted_spread_vwap']
        )
        
        if not np.isnan(current['adjusted_spread_volatility']) and not np.isnan(previous['adjusted_spread_volatility']):
            deltas['bucket_delta_adjusted_spread_volatility'] = np.float64(
                current['adjusted_spread_volatility'] - previous['adjusted_spread_volatility']
            )
        else:
            deltas['bucket_delta_adjusted_spread_volatility'] = np.float64(np.nan)
        
        deltas['bucket_delta_adjusted_return_volatility_log'] = Bucket._safe_log_ratio(
            current['adjusted_price_volatility'], previous['adjusted_price_volatility']
        )
        
        deltas['bucket_delta_adjusted_imbalance_signed_ratio_atanh'] = Bucket._safe_atanh_diff(
            current['adjusted_imbalance_signed_ratio'], previous['adjusted_imbalance_signed_ratio']
        )
        deltas['bucket_delta_adjusted_imbalance_abs_ratio_logit'] = Bucket._safe_logit_diff(
            current['adjusted_imbalance_abs_ratio'], previous['adjusted_imbalance_abs_ratio']
        )
        
        if not np.isnan(current['adjusted_imbalance_persistence']) and not np.isnan(previous['adjusted_imbalance_persistence']):
            deltas['bucket_delta_adjusted_imbalance_persistence'] = np.float64(
                current['adjusted_imbalance_persistence'] - previous['adjusted_imbalance_persistence']
            )
        else:
            deltas['bucket_delta_adjusted_imbalance_persistence'] = np.float64(np.nan)
        
        if not np.isnan(current['adjusted_imbalance_volatility']) and not np.isnan(previous['adjusted_imbalance_volatility']):
            deltas['bucket_delta_adjusted_imbalance_volatility'] = np.float64(
                current['adjusted_imbalance_volatility'] - previous['adjusted_imbalance_volatility']
            )
        else:
            deltas['bucket_delta_adjusted_imbalance_volatility'] = np.float64(np.nan)
        
        # Second-order deltas
        if previous_deltas is not None:
            current_delta = deltas['bucket_delta_active_return_volatility_log']
            previous_delta = previous_deltas.get('bucket_delta_active_return_volatility_log', np.nan)
            if not np.isnan(current_delta) and not np.isnan(previous_delta):
                deltas['bucket_delta2_active_return_volatility_log'] = np.float64(current_delta - previous_delta)
            else:
                deltas['bucket_delta2_active_return_volatility_log'] = np.float64(np.nan)
            
            current_delta = deltas['bucket_delta_active_imbalance_signed_ratio_atanh']
            previous_delta = previous_deltas.get('bucket_delta_active_imbalance_signed_ratio_atanh', np.nan)
            if not np.isnan(current_delta) and not np.isnan(previous_delta):
                deltas['bucket_delta2_active_imbalance_signed_ratio_atanh'] = np.float64(current_delta - previous_delta)
            else:
                deltas['bucket_delta2_active_imbalance_signed_ratio_atanh'] = np.float64(np.nan)
            
            current_delta = deltas['bucket_delta_active_pace_of_contracts_traded_log']
            previous_delta = previous_deltas.get('bucket_delta_active_pace_of_contracts_traded_log', np.nan)
            if not np.isnan(current_delta) and not np.isnan(previous_delta):
                deltas['bucket_delta2_active_pace_of_contracts_traded_log'] = np.float64(current_delta - previous_delta)
            else:
                deltas['bucket_delta2_active_pace_of_contracts_traded_log'] = np.float64(np.nan)
            
            current_delta = deltas['bucket_delta_adjusted_return_volatility_log']
            previous_delta = previous_deltas.get('bucket_delta_adjusted_return_volatility_log', np.nan)
            if not np.isnan(current_delta) and not np.isnan(previous_delta):
                deltas['bucket_delta2_adjusted_return_volatility_log'] = np.float64(current_delta - previous_delta)
            else:
                deltas['bucket_delta2_adjusted_return_volatility_log'] = np.float64(np.nan)
            
            current_delta = deltas['bucket_delta_adjusted_imbalance_signed_ratio_atanh']
            previous_delta = previous_deltas.get('bucket_delta_adjusted_imbalance_signed_ratio_atanh', np.nan)
            if not np.isnan(current_delta) and not np.isnan(previous_delta):
                deltas['bucket_delta2_adjusted_imbalance_signed_ratio_atanh'] = np.float64(current_delta - previous_delta)
            else:
                deltas['bucket_delta2_adjusted_imbalance_signed_ratio_atanh'] = np.float64(np.nan)
        else:
            deltas['bucket_delta2_active_return_volatility_log'] = np.float64(np.nan)
            deltas['bucket_delta2_active_imbalance_signed_ratio_atanh'] = np.float64(np.nan)
            deltas['bucket_delta2_active_pace_of_contracts_traded_log'] = np.float64(np.nan)
            deltas['bucket_delta2_adjusted_return_volatility_log'] = np.float64(np.nan)
            deltas['bucket_delta2_adjusted_imbalance_signed_ratio_atanh'] = np.float64(np.nan)
        
        return deltas




    # ============================================================================
    # Empty Value Functions
    # ============================================================================


    # @staticmethod
    # def _get_empty_bucket_statistics() -> Dict[str, Any]:
    #     """
    #     Generate initial values for all non-delta bucket statistics.
    #     Used for the first bucket or when initializing bucket data structures.
        
    #     Returns:
    #         Dictionary with all non-delta bucket fields set to appropriate initial values
            
    #     <><><> MUST MATCH "BUCKET_DTYPE" SCHEMA <><><>
        
    #     Reasoning for initial values:
    #     - NaN for continuous metrics (unknown/unmeasured)
    #     - 0 for counts and sums (nothing accumulated yet)
    #     - False for boolean flags (no events occurred)
    #     - Empty strings for identifiers (not yet set)
    #     """
        
    #     current: Dict[str, Any] = {}
        
    #     # ========================================================================
    #     # I. Meta / Structural
    #     # ========================================================================
    #     current['bucket_id'] = np.uint32(0)
    #     current['bucket_type'] = np.str_('')
    #     current['num_bars'] = np.uint32(0)
    #     current['all_bars_complete'] = np.bool_(False)
    #     current['bar_volume_size'] = np.uint32(0)
    #     current['contract_roll_any'] = np.bool_(False)
    #     current['latest_instrument_id'] = np.uint32(0)
    #     current['start_ts_ns'] = np.uint64(0)
    #     current['end_ts_ns'] = np.uint64(0)
    #     current['time_elapsed_ns_total'] = np.uint64(0)
        
    #     # ========================================================================
    #     # II. Flow & Order Activity - Aggregations
    #     # ========================================================================
    #     current['volume_total_sum'] = np.uint64(0)
    #     current['volume_total_mean'] = np.float64(np.nan)
    #     current['volume_total_std'] = np.float64(np.nan)
    #     current['volume_buy_sum'] = np.uint64(0)
    #     current['volume_buy_mean'] = np.float64(np.nan)
    #     current['volume_buy_std'] = np.float64(np.nan)
    #     current['volume_sell_sum'] = np.uint64(0)
    #     current['volume_sell_mean'] = np.float64(np.nan)
    #     current['volume_sell_std'] = np.float64(np.nan)
        
    #     current['order_count_sum'] = np.uint64(0)
    #     current['order_count_mean'] = np.float64(np.nan)
    #     current['order_count_std'] = np.float64(np.nan)
    #     current['order_count_buy_sum'] = np.uint64(0)
    #     current['order_count_buy_mean'] = np.float64(np.nan)
    #     current['order_count_buy_std'] = np.float64(np.nan)
    #     current['order_count_sell_sum'] = np.uint64(0)
    #     current['order_count_sell_mean'] = np.float64(np.nan)
    #     current['order_count_sell_std'] = np.float64(np.nan)
        
    #     current['order_splits_sum'] = np.uint64(0)
    #     current['order_splits_mean'] = np.float64(np.nan)
        
    #     current['delta_volume_total_log_mean'] = np.float64(np.nan)
    #     current['delta_volume_total_log_std'] = np.float64(np.nan)
    #     current['delta_volume_total_log_max'] = np.float64(np.nan)
    #     current['delta_volume_buy_log_mean'] = np.float64(np.nan)
    #     current['delta_volume_buy_log_std'] = np.float64(np.nan)
    #     current['delta_volume_buy_log_max'] = np.float64(np.nan)
    #     current['delta_volume_sell_log_mean'] = np.float64(np.nan)
    #     current['delta_volume_sell_log_std'] = np.float64(np.nan)
    #     current['delta_volume_sell_log_max'] = np.float64(np.nan)
        
    #     current['delta_order_count_log_mean'] = np.float64(np.nan)
    #     current['delta_order_count_log_std'] = np.float64(np.nan)
    #     current['delta_order_count_log_max'] = np.float64(np.nan)
    #     current['delta_order_count_buy_log_mean'] = np.float64(np.nan)
    #     current['delta_order_count_buy_log_std'] = np.float64(np.nan)
    #     current['delta_order_count_buy_log_max'] = np.float64(np.nan)
    #     current['delta_order_count_sell_log_mean'] = np.float64(np.nan)
    #     current['delta_order_count_sell_log_std'] = np.float64(np.nan)
    #     current['delta_order_count_sell_log_max'] = np.float64(np.nan)
        
    #     # ========================================================================
    #     # III. Imbalance and Directional Pressure
    #     # ========================================================================
    #     current['imbalance_size_signed_ratio_mean'] = np.float64(np.nan)
    #     current['imbalance_size_signed_ratio_std'] = np.float64(np.nan)
    #     current['imbalance_size_abs_ratio_mean'] = np.float64(np.nan)
    #     current['imbalance_size_abs_ratio_std'] = np.float64(np.nan)
    #     current['imbalance_buy_ratio_mean'] = np.float64(np.nan)
    #     current['imbalance_buy_ratio_std'] = np.float64(np.nan)
        
    #     current['imbalance_size_signed_ratio_atanh_mean'] = np.float64(np.nan)
    #     current['imbalance_size_signed_ratio_atanh_std'] = np.float64(np.nan)
    #     current['imbalance_buy_ratio_logit_mean'] = np.float64(np.nan)
    #     current['imbalance_buy_ratio_logit_std'] = np.float64(np.nan)
    #     current['imbalance_size_abs_ratio_logit_mean'] = np.float64(np.nan)
    #     current['imbalance_size_abs_ratio_logit_std'] = np.float64(np.nan)
        
    #     current['delta_imbalance_size_signed_ratio_mean'] = np.float64(np.nan)
    #     current['delta_imbalance_size_signed_ratio_std'] = np.float64(np.nan)
    #     current['delta_imbalance_buy_ratio_mean'] = np.float64(np.nan)
    #     current['delta_imbalance_buy_ratio_std'] = np.float64(np.nan)
        
    #     current['cumulative_signed_imbalance'] = np.int64(0)
    #     current['imbalance_persistence'] = np.float64(np.nan)
    #     current['imbalance_volatility'] = np.float64(np.nan)
        
    #     # ========================================================================
    #     # IV. Price / VWAP Metrics
    #     # ========================================================================
    #     current['midpoint_vwap_mean'] = np.float64(np.nan)
    #     current['midpoint_vwap_std'] = np.float64(np.nan)
    #     current['midpoint_vwap_range'] = np.float64(np.nan)
    #     current['spread_vwap_mean'] = np.float64(np.nan)
    #     current['spread_vwap_std'] = np.float64(np.nan)
    #     current['spread_vwap_range'] = np.float64(np.nan)
        
    #     current['delta_midpoint_vwap_log_sum'] = np.float64(np.nan)
    #     current['delta_midpoint_vwap_log_mean'] = np.float64(np.nan)
    #     current['delta_midpoint_vwap_log_std'] = np.float64(np.nan)
    #     current['delta_midpoint_vwap_log_skew'] = np.float64(np.nan)
        
    #     current['delta_mid_imbalance_weighted_log_sum'] = np.float64(np.nan)
    #     current['delta_mid_imbalance_weighted_log_mean'] = np.float64(np.nan)
    #     current['delta_mid_imbalance_weighted_log_std'] = np.float64(np.nan)
    #     current['delta_mid_imbalance_weighted_log_skew'] = np.float64(np.nan)
        
    #     current['delta_mid_flow_weighted_log_sum'] = np.float64(np.nan)
    #     current['delta_mid_flow_weighted_log_mean'] = np.float64(np.nan)
    #     current['delta_mid_flow_weighted_log_std'] = np.float64(np.nan)
    #     current['delta_mid_flow_weighted_log_skew'] = np.float64(np.nan)
        
    #     current['delta_mid_aggressor_weighted_log_sum'] = np.float64(np.nan)
    #     current['delta_mid_aggressor_weighted_log_mean'] = np.float64(np.nan)
    #     current['delta_mid_aggressor_weighted_log_std'] = np.float64(np.nan)
    #     current['delta_mid_aggressor_weighted_log_skew'] = np.float64(np.nan)
        
    #     current['delta_spread_vwap_log_mean'] = np.float64(np.nan)
    #     current['delta_spread_vwap_log_std'] = np.float64(np.nan)
        
    #     current['delta_price_buy_vwap_log_mean'] = np.float64(np.nan)
    #     current['delta_price_buy_vwap_log_std'] = np.float64(np.nan)
    #     current['delta_price_sell_vwap_log_mean'] = np.float64(np.nan)
    #     current['delta_price_sell_vwap_log_std'] = np.float64(np.nan)
        
    #     current['price_volatility'] = np.float64(np.nan)
    #     current['spread_volatility'] = np.float64(np.nan)
    #     current['bucket_return_log_sum'] = np.float64(np.nan)
    #     current['intermediate_volatility'] = np.float64(np.nan)
    #     current['return_volatility'] = np.float64(np.nan)
    #     current['spread_stability'] = np.float64(np.nan)
        
    #     # ========================================================================
    #     # V. Temporal / Tempo Structure
    #     # ========================================================================
    #     current['time_elapsed_ns_mean'] = np.float64(np.nan)
    #     current['time_elapsed_ns_std'] = np.float64(np.nan)
    #     current['time_elapsed_ns_min'] = np.uint64(0)
    #     current['time_elapsed_ns_max'] = np.uint64(0)
        
    #     current['pace_of_orders_mean'] = np.float64(np.nan)
    #     current['pace_of_orders_std'] = np.float64(np.nan)
        
    #     current['delta_time_elapsed_ns_log_mean'] = np.float64(np.nan)
    #     current['delta_time_elapsed_ns_log_std'] = np.float64(np.nan)
    #     current['delta_pace_of_orders_log_mean'] = np.float64(np.nan)
    #     current['delta_pace_of_orders_log_std'] = np.float64(np.nan)
        
    #     current['tempo_stability'] = np.float64(np.nan)
    #     current['tempo_acceleration'] = np.float64(np.nan)
        
    #     # ========================================================================
    #     # VI. Directional Signals
    #     # ========================================================================
    #     current['pct_bars_positive_direction'] = np.float64(np.nan)
    #     current['directional_streak_max'] = np.uint32(0)
    #     current['directional_reversals_count'] = np.uint32(0)
    #     current['net_direction'] = np.float64(np.nan)
    #     current['direction_streak_len'] = np.uint32(0)
        
    #     # ========================================================================
    #     # VII. Volatility and Stability Overlays
    #     # ========================================================================
    #     current['flow_volatility'] = np.float64(np.nan)
        
    #     # ========================================================================
    #     # IX. Debug / Diagnostic
    #     # ========================================================================
    #     current['bar_ids_checksum'] = np.uint64(0)
    #     current['missing_bars_flag'] = np.bool_(False)
    #     current['processing_time_ns'] = np.uint64(0)
        
    #     return current




    # @staticmethod
    # def _get_empty_bucket_deltas() -> Dict[str, Any]:
    #     """
    #     Generate NaN values for all inter-bucket delta features.
    #     Used for the first bucket where there is no previous bucket to compare against.
        
    #     Returns:
    #         Dictionary with all inter-bucket delta fields set to NaN or zero
            
    #     <><><> MUST MATCH "BUCKET_DTYPE" SCHEMA <><><>
        
    #     Reasoning for using NaN:
    #     1. Semantically correct (unknown ≠ zero change)
    #     2. Prevents model confusion
    #     3. Enables proper missing value handling
    #     4. Maintains statistical integrity
    #     5. Gives flexibility in downstream processing
    #     """
        
    #     deltas: Dict[str, Any] = {}
        
    #     # All float64 deltas get NaN (no previous bucket to compare)
    #     deltas['bucket_delta_spread_vwap_mean_log'] = np.float64(np.nan)
    #     deltas['bucket_delta_spread_vwap_std_log'] = np.float64(np.nan)
    #     deltas['bucket_delta_spread_volatility'] = np.float64(np.nan)
    #     deltas['bucket_delta_return_volatility_log'] = np.float64(np.nan)
    #     deltas['bucket_delta_intermediate_volatility_log'] = np.float64(np.nan)
    #     deltas['bucket_delta_flow_volatility_log'] = np.float64(np.nan)
    #     deltas['bucket_delta_pace_of_orders_mean_log'] = np.float64(np.nan)
    #     deltas['bucket_delta_tempo_stability'] = np.float64(np.nan)
    #     deltas['bucket_delta_time_elapsed_ns_total_log'] = np.float64(np.nan)
    #     deltas['bucket_delta_imbalance_size_signed_ratio_mean_atanh'] = np.float64(np.nan)
    #     deltas['bucket_delta_imbalance_size_abs_ratio_mean_logit'] = np.float64(np.nan)
    #     deltas['bucket_delta_imbalance_persistence'] = np.float64(np.nan)
    #     deltas['bucket_delta_cumulative_signed_imbalance_norm'] = np.float64(np.nan)
    #     deltas['bucket_delta_imbalance_volatility'] = np.float64(np.nan)
    #     deltas['bucket_delta_pct_bars_positive_direction'] = np.float64(np.nan)
        
    #     # Integer deltas get 0 (no previous value to compare)
    #     deltas['bucket_delta_directional_streak_max'] = np.int32(0)
    #     deltas['bucket_delta_directional_reversals_count'] = np.int32(0)
        
    #     # Lag features get NaN (no previous bucket exists)
    #     deltas['lag1_bucket_return_log_sum'] = np.float64(np.nan)
    #     deltas['lag1_abs_bucket_return_log_sum'] = np.float64(np.nan)
    #     deltas['lag1_sign_bucket_return'] = np.int8(0)
        
    #     return deltas




    @staticmethod
    def _get_empty_bucket_statistics() -> Dict[str, Any]:
        """
        Generate initial values for all non-delta bucket statistics.
        Used for the first bucket or when initializing bucket data structures.
        Returns:
            Dictionary with all non-delta bucket fields set to appropriate initial values
        <><><> MUST MATCH "BUCKET_DTYPE" SCHEMA <><><>
        Reasoning for initial values:
        - NaN for continuous metrics (unknown/unmeasured)
        - 0 for counts and sums (nothing accumulated yet)
        - False for boolean flags (no events occurred)
        - Empty strings for identifiers (not yet set)
        Edge case handling:
        - uint/int types: 0
        - float types: NaN (insufficient data or undefined)
        - bool types: False
        """

        current: Dict[str, Any] = {}

        # ========================================================================
        # 2. Meta / Structural
        # ========================================================================
        current['id'] = np.uint32(0)
        current['bucket_type'] = np.str_('')  # 'fixed', 'interval', 'adaptive'
        current['num_bars'] = np.uint32(0)
        current['all_bars_complete'] = np.bool_(False)
        current['bar_volume_size'] = np.uint32(0)
        current['contract_roll_any'] = np.bool_(False)
        current['contract_roll_count'] = np.uint32(0)
        current['latest_instrument_id'] = np.uint32(0)
        current['start_ts_ns'] = np.uint64(0)
        current['end_ts_ns'] = np.uint64(0)
        current['time_elapsed_ns_total'] = np.uint64(0)
        current['gap_return_any'] = np.bool_(False)
        current['gap_return_count'] = np.uint32(0)
        current['contains_oversized_order_any'] = np.bool_(False)
        current['contains_oversized_order_count'] = np.uint32(0)

        # ========================================================================
        # 3. Order & Volume Statistics (Total/Ambiguous)
        # ========================================================================
        # Volume Metrics
        current['volume_total_sum'] = np.uint64(0)
        current['volume_total_mean'] = np.float64(np.nan)
        current['volume_total_std'] = np.float64(np.nan)

        # Order Count Metrics
        current['order_count_sum'] = np.uint64(0)
        current['order_count_mean'] = np.float64(np.nan)
        current['order_count_std'] = np.float64(np.nan)

        # Order Splits
        current['order_splits_sum'] = np.uint64(0)
        current['order_splits_mean'] = np.float64(np.nan)

        # Volume Log Deltas
        current['delta_volume_total_log_mean'] = np.float64(np.nan)
        current['delta_volume_total_log_std'] = np.float64(np.nan)
        current['delta_volume_total_log_max'] = np.float64(np.nan)

        # Order Count Log Deltas
        current['delta_order_count_log_mean'] = np.float64(np.nan)
        current['delta_order_count_log_std'] = np.float64(np.nan)
        current['delta_order_count_log_max'] = np.float64(np.nan)

        # ========================================================================
        # 4. Active Metrics
        # ========================================================================
        # 4.1 Active Order Counts
        current['active_order_count_buy_sum'] = np.uint64(0)
        current['active_order_count_buy_mean'] = np.float64(np.nan)
        current['active_order_count_buy_std'] = np.float64(np.nan)
        current['active_order_count_sell_sum'] = np.uint64(0)
        current['active_order_count_sell_mean'] = np.float64(np.nan)
        current['active_order_count_sell_std'] = np.float64(np.nan)
        current['active_order_count_none_sum'] = np.uint64(0)
        current['active_order_count_none_mean'] = np.float64(np.nan)
        current['active_order_count_none_std'] = np.float64(np.nan)

        # 4.2 Active Volumes
        current['active_volume_buy_sum'] = np.uint64(0)
        current['active_volume_buy_mean'] = np.float64(np.nan)
        current['active_volume_buy_std'] = np.float64(np.nan)
        current['active_volume_sell_sum'] = np.uint64(0)
        current['active_volume_sell_mean'] = np.float64(np.nan)
        current['active_volume_sell_std'] = np.float64(np.nan)
        current['active_volume_none_sum'] = np.uint64(0)
        current['active_volume_none_mean'] = np.float64(np.nan)
        current['active_volume_none_std'] = np.float64(np.nan)

        # 4.3 Active Volume Log Deltas
        current['delta_active_volume_buy_log_mean'] = np.float64(np.nan)
        current['delta_active_volume_buy_log_std'] = np.float64(np.nan)
        current['delta_active_volume_buy_log_max'] = np.float64(np.nan)
        current['delta_active_volume_sell_log_mean'] = np.float64(np.nan)
        current['delta_active_volume_sell_log_std'] = np.float64(np.nan)
        current['delta_active_volume_sell_log_max'] = np.float64(np.nan)
        current['delta_active_volume_none_log_mean'] = np.float64(np.nan)
        current['delta_active_volume_none_log_std'] = np.float64(np.nan)
        current['delta_active_volume_none_log_max'] = np.float64(np.nan)

        # 4.4 Active Order Count Log Deltas
        current['delta_active_order_count_buy_log_mean'] = np.float64(np.nan)
        current['delta_active_order_count_buy_log_std'] = np.float64(np.nan)
        current['delta_active_order_count_buy_log_max'] = np.float64(np.nan)
        current['delta_active_order_count_sell_log_mean'] = np.float64(np.nan)
        current['delta_active_order_count_sell_log_std'] = np.float64(np.nan)
        current['delta_active_order_count_sell_log_max'] = np.float64(np.nan)
        current['delta_active_order_count_none_log_mean'] = np.float64(np.nan)
        current['delta_active_order_count_none_log_std'] = np.float64(np.nan)
        current['delta_active_order_count_none_log_max'] = np.float64(np.nan)

        # 4.5 Active Imbalance Metrics
        current['active_imbalance_signed'] = np.int64(0)
        current['active_imbalance_abs'] = np.uint64(0)
        current['active_imbalance_signed_ratio'] = np.float64(np.nan)
        current['active_imbalance_abs_ratio'] = np.float64(np.nan)
        current['active_imbalance_buy_ratio'] = np.float64(np.nan)
        current['active_imbalance_signed_ratio_std'] = np.float64(np.nan)
        current['active_imbalance_abs_ratio_std'] = np.float64(np.nan)
        current['active_imbalance_buy_ratio_std'] = np.float64(np.nan)

        # 4.6 Active Link Function Transforms
        current['active_imbalance_signed_ratio_atanh'] = np.float64(np.nan)
        current['active_imbalance_signed_ratio_atanh_std'] = np.float64(np.nan)
        current['active_imbalance_buy_ratio_logit'] = np.float64(np.nan)
        current['active_imbalance_buy_ratio_logit_std'] = np.float64(np.nan)
        current['active_imbalance_abs_ratio_logit'] = np.float64(np.nan)
        current['active_imbalance_abs_ratio_logit_std'] = np.float64(np.nan)

        # 4.7 Active Imbalance Deltas
        current['delta_active_imbalance_signed_ratio_mean'] = np.float64(np.nan)
        current['delta_active_imbalance_signed_ratio_std'] = np.float64(np.nan)
        current['delta_active_imbalance_buy_ratio_mean'] = np.float64(np.nan)
        current['delta_active_imbalance_buy_ratio_std'] = np.float64(np.nan)

        # 4.8 Active Derived Imbalance Metrics
        current['active_cumulative_signed_imbalance'] = np.int64(0)
        current['active_imbalance_persistence'] = np.float64(np.nan)
        current['active_imbalance_volatility'] = np.float64(np.nan)

        # 4.9 Active Price Metrics (VWAP)
        current['active_buy_vwap'] = np.float64(np.nan)
        current['active_sell_vwap'] = np.float64(np.nan)
        current['active_none_vwap'] = np.float64(np.nan)
        current['active_spread_vwap'] = np.float64(np.nan)
        current['active_midpoint_vwap'] = np.float64(np.nan)
        current['active_midpoint_vwap_std'] = np.float64(np.nan)
        current['active_midpoint_vwap_range'] = np.float64(np.nan)
        current['active_spread_vwap_std'] = np.float64(np.nan)
        current['active_spread_vwap_range'] = np.float64(np.nan)

        # 4.10 Active Weighted Midpoints
        current['active_mid_imbalance_weighted'] = np.float64(np.nan)
        current['active_mid_flow_weighted'] = np.float64(np.nan)
        current['active_mid_aggressor_weighted'] = np.float64(np.nan)
        current['active_mid_imbalance_weighted_std'] = np.float64(np.nan)
        current['active_mid_flow_weighted_std'] = np.float64(np.nan)
        current['active_mid_aggressor_weighted_std'] = np.float64(np.nan)

        # 4.11 Active Price Log Deltas
        current['delta_active_midpoint_vwap_log_sum'] = np.float64(np.nan)
        current['delta_active_midpoint_vwap_log_mean'] = np.float64(np.nan)
        current['delta_active_midpoint_vwap_log_std'] = np.float64(np.nan)
        current['delta_active_midpoint_vwap_log_skew'] = np.float64(np.nan)
        current['delta_active_spread_vwap_log_mean'] = np.float64(np.nan)
        current['delta_active_spread_vwap_log_std'] = np.float64(np.nan)
        current['delta_active_buy_vwap_log_mean'] = np.float64(np.nan)
        current['delta_active_buy_vwap_log_std'] = np.float64(np.nan)
        current['delta_active_sell_vwap_log_mean'] = np.float64(np.nan)
        current['delta_active_sell_vwap_log_std'] = np.float64(np.nan)

        # 4.12 Active Weighted Midpoint Log Deltas
        current['delta_active_mid_imbalance_weighted_log_sum'] = np.float64(np.nan)
        current['delta_active_mid_imbalance_weighted_log_mean'] = np.float64(np.nan)
        current['delta_active_mid_imbalance_weighted_log_std'] = np.float64(np.nan)
        current['delta_active_mid_imbalance_weighted_log_skew'] = np.float64(np.nan)
        current['delta_active_mid_flow_weighted_log_sum'] = np.float64(np.nan)
        current['delta_active_mid_flow_weighted_log_mean'] = np.float64(np.nan)
        current['delta_active_mid_flow_weighted_log_std'] = np.float64(np.nan)
        current['delta_active_mid_flow_weighted_log_skew'] = np.float64(np.nan)
        current['delta_active_mid_aggressor_weighted_log_sum'] = np.float64(np.nan)
        current['delta_active_mid_aggressor_weighted_log_mean'] = np.float64(np.nan)
        current['delta_active_mid_aggressor_weighted_log_std'] = np.float64(np.nan)
        current['delta_active_mid_aggressor_weighted_log_skew'] = np.float64(np.nan)

        # 4.13 Active Price Range Metrics - Bucket-Level Extremes
        current['active_buy_price_min'] = np.float64(np.nan)
        current['active_buy_price_max'] = np.float64(np.nan)
        current['active_buy_price_range'] = np.float64(np.nan)
        current['active_sell_price_min'] = np.float64(np.nan)
        current['active_sell_price_max'] = np.float64(np.nan)
        current['active_sell_price_range'] = np.float64(np.nan)
        current['active_none_price_min'] = np.float64(np.nan)
        current['active_none_price_max'] = np.float64(np.nan)
        current['active_none_price_range'] = np.float64(np.nan)

        # 4.13 Active Price Range Metrics - Bar-Level Statistics
        current['active_buy_price_range_mean'] = np.float64(np.nan)
        current['active_buy_price_range_std'] = np.float64(np.nan)
        current['active_sell_price_range_mean'] = np.float64(np.nan)
        current['active_sell_price_range_std'] = np.float64(np.nan)
        current['active_none_price_range_mean'] = np.float64(np.nan)
        current['active_none_price_range_std'] = np.float64(np.nan)

        # 4.14 Active Pace Metrics - Bucket-Level
        current['active_buy_pace'] = np.float64(np.nan)
        current['active_sell_pace'] = np.float64(np.nan)

        # 4.14 Active Pace Metrics - Bar-Level Statistics
        current['active_buy_pace_mean'] = np.float64(np.nan)
        current['active_buy_pace_std'] = np.float64(np.nan)
        current['active_sell_pace_mean'] = np.float64(np.nan)
        current['active_sell_pace_std'] = np.float64(np.nan)

        # 4.14 Active Pace Metrics - Log-Transformed Deltas
        current['delta_active_buy_pace_log_mean'] = np.float64(np.nan)
        current['delta_active_buy_pace_log_std'] = np.float64(np.nan)
        current['delta_active_sell_pace_log_mean'] = np.float64(np.nan)
        current['delta_active_sell_pace_log_std'] = np.float64(np.nan)

        # 4.15 Active N-Side Inferred Aggregations
        current['active_none_inferred_buy_volume_sum'] = np.float64(0.0)
        current['active_none_inferred_sell_volume_sum'] = np.float64(0.0)
        current['active_none_inferred_buy_vwap'] = np.float64(np.nan)
        current['active_none_inferred_sell_vwap'] = np.float64(np.nan)

        # ========================================================================
        # 5. Adjusted Metrics
        # ========================================================================
        # 5.1 Adjusted Volumes
        current['adjusted_volume_buy_sum'] = np.float64(0.0)
        current['adjusted_volume_sell_sum'] = np.float64(0.0)
        current['adjusted_volume_buy_mean'] = np.float64(np.nan)
        current['adjusted_volume_buy_std'] = np.float64(np.nan)
        current['adjusted_volume_sell_mean'] = np.float64(np.nan)
        current['adjusted_volume_sell_std'] = np.float64(np.nan)

        # 5.2 Adjusted Imbalance Metrics
        current['adjusted_imbalance_signed'] = np.float64(0.0)
        current['adjusted_imbalance_abs'] = np.float64(0.0)
        current['adjusted_imbalance_signed_ratio'] = np.float64(np.nan)
        current['adjusted_imbalance_abs_ratio'] = np.float64(np.nan)
        current['adjusted_imbalance_buy_ratio'] = np.float64(np.nan)
        current['adjusted_imbalance_signed_ratio_std'] = np.float64(np.nan)
        current['adjusted_imbalance_abs_ratio_std'] = np.float64(np.nan)
        current['adjusted_imbalance_buy_ratio_std'] = np.float64(np.nan)

        # 5.3 Adjusted Link Function Transforms
        current['adjusted_imbalance_signed_ratio_atanh'] = np.float64(np.nan)
        current['adjusted_imbalance_signed_ratio_atanh_std'] = np.float64(np.nan)
        current['adjusted_imbalance_buy_ratio_logit'] = np.float64(np.nan)
        current['adjusted_imbalance_buy_ratio_logit_std'] = np.float64(np.nan)
        current['adjusted_imbalance_abs_ratio_logit'] = np.float64(np.nan)
        current['adjusted_imbalance_abs_ratio_logit_std'] = np.float64(np.nan)

        # 5.4 Adjusted Imbalance Deltas
        current['delta_adjusted_imbalance_signed_ratio_mean'] = np.float64(np.nan)
        current['delta_adjusted_imbalance_signed_ratio_std'] = np.float64(np.nan)
        current['delta_adjusted_imbalance_buy_ratio_mean'] = np.float64(np.nan)
        current['delta_adjusted_imbalance_buy_ratio_std'] = np.float64(np.nan)
        current['delta_adjusted_imbalance_signed_ratio_skew'] = np.float64(np.nan)

        # 5.5 Adjusted Derived Imbalance Metrics
        current['adjusted_cumulative_signed_imbalance'] = np.float64(0.0)
        current['adjusted_imbalance_persistence'] = np.float64(np.nan)
        current['adjusted_imbalance_volatility'] = np.float64(np.nan)

        # 5.6 Adjusted Price Metrics (VWAP)
        current['adjusted_buy_vwap'] = np.float64(np.nan)
        current['adjusted_sell_vwap'] = np.float64(np.nan)
        current['adjusted_spread_vwap'] = np.float64(np.nan)
        current['adjusted_midpoint_vwap'] = np.float64(np.nan)
        current['adjusted_midpoint_vwap_std'] = np.float64(np.nan)
        current['adjusted_midpoint_vwap_range'] = np.float64(np.nan)
        current['adjusted_spread_vwap_std'] = np.float64(np.nan)
        current['adjusted_spread_vwap_range'] = np.float64(np.nan)

        # 5.7 Adjusted Weighted Midpoints
        current['adjusted_mid_imbalance_weighted'] = np.float64(np.nan)
        current['adjusted_mid_flow_weighted'] = np.float64(np.nan)
        current['adjusted_mid_aggressor_weighted'] = np.float64(np.nan)
        current['adjusted_mid_imbalance_weighted_std'] = np.float64(np.nan)
        current['adjusted_mid_flow_weighted_std'] = np.float64(np.nan)
        current['adjusted_mid_aggressor_weighted_std'] = np.float64(np.nan)

        # 5.8 Adjusted Price Log Deltas
        current['delta_adjusted_midpoint_vwap_log_sum'] = np.float64(np.nan)
        current['delta_adjusted_midpoint_vwap_log_mean'] = np.float64(np.nan)
        current['delta_adjusted_midpoint_vwap_log_std'] = np.float64(np.nan)
        current['delta_adjusted_midpoint_vwap_log_skew'] = np.float64(np.nan)
        current['delta_adjusted_spread_vwap_log_mean'] = np.float64(np.nan)
        current['delta_adjusted_spread_vwap_log_std'] = np.float64(np.nan)
        current['delta_adjusted_buy_vwap_log_mean'] = np.float64(np.nan)
        current['delta_adjusted_buy_vwap_log_std'] = np.float64(np.nan)
        current['delta_adjusted_sell_vwap_log_mean'] = np.float64(np.nan)
        current['delta_adjusted_sell_vwap_log_std'] = np.float64(np.nan)

        # 5.9 Adjusted Weighted Midpoint Log Deltas
        current['delta_adjusted_mid_imbalance_weighted_log_sum'] = np.float64(np.nan)
        current['delta_adjusted_mid_imbalance_weighted_log_mean'] = np.float64(np.nan)
        current['delta_adjusted_mid_imbalance_weighted_log_std'] = np.float64(np.nan)
        current['delta_adjusted_mid_imbalance_weighted_log_skew'] = np.float64(np.nan)
        current['delta_adjusted_mid_flow_weighted_log_sum'] = np.float64(np.nan)
        current['delta_adjusted_mid_flow_weighted_log_mean'] = np.float64(np.nan)
        current['delta_adjusted_mid_flow_weighted_log_std'] = np.float64(np.nan)
        current['delta_adjusted_mid_flow_weighted_log_skew'] = np.float64(np.nan)
        current['delta_adjusted_mid_aggressor_weighted_log_sum'] = np.float64(np.nan)
        current['delta_adjusted_mid_aggressor_weighted_log_mean'] = np.float64(np.nan)
        current['delta_adjusted_mid_aggressor_weighted_log_std'] = np.float64(np.nan)
        current['delta_adjusted_mid_aggressor_weighted_log_skew'] = np.float64(np.nan)

        # ========================================================================
        # 6. Passive Metrics
        # ========================================================================
        # 6.1 Passive Midprice
        current['passive_midprice'] = np.float64(np.nan)
        current['passive_midprice_std'] = np.float64(np.nan)
        current['passive_midprice_range'] = np.float64(np.nan)

        # 6.2 Passive Midprice Deltas
        current['delta_passive_midprice_log_sum'] = np.float64(np.nan)
        current['delta_passive_midprice_log_mean'] = np.float64(np.nan)
        current['delta_passive_midprice_log_std'] = np.float64(np.nan)
        current['delta_passive_midprice_log_skew'] = np.float64(np.nan)

        # 6.3 Passive CDF Statistics
        current['passive_midprice_delta_cdf_mean'] = np.float64(np.nan)
        current['passive_midprice_delta_cdf_std'] = np.float64(np.nan)

        # 6.4 Passive Volume Classifications
        current['passive_buy_volume_sum'] = np.float64(0.0)
        current['passive_sell_volume_sum'] = np.float64(0.0)
        current['passive_buy_volume_mean'] = np.float64(np.nan)
        current['passive_buy_volume_std'] = np.float64(np.nan)
        current['passive_sell_volume_mean'] = np.float64(np.nan)
        current['passive_sell_volume_std'] = np.float64(np.nan)

        # 6.5 Passive Imbalance Metrics
        current['passive_imbalance_signed'] = np.float64(0.0)
        current['passive_imbalance_abs'] = np.float64(0.0)
        current['passive_imbalance_signed_ratio'] = np.float64(np.nan)
        current['passive_imbalance_abs_ratio'] = np.float64(np.nan)
        current['passive_imbalance_buy_ratio'] = np.float64(np.nan)
        current['passive_imbalance_signed_ratio_std'] = np.float64(np.nan)
        current['passive_imbalance_abs_ratio_std'] = np.float64(np.nan)
        current['passive_imbalance_buy_ratio_std'] = np.float64(np.nan)

        # 6.6 Passive Link Function Transforms
        current['passive_imbalance_signed_ratio_atanh'] = np.float64(np.nan)
        current['passive_imbalance_signed_ratio_atanh_std'] = np.float64(np.nan)
        current['passive_imbalance_buy_ratio_logit'] = np.float64(np.nan)
        current['passive_imbalance_buy_ratio_logit_std'] = np.float64(np.nan)
        current['passive_imbalance_abs_ratio_logit'] = np.float64(np.nan)
        current['passive_imbalance_abs_ratio_logit_std'] = np.float64(np.nan)

        # 6.7 Passive Imbalance Deltas
        current['delta_passive_imbalance_signed_ratio_mean'] = np.float64(np.nan)
        current['delta_passive_imbalance_signed_ratio_std'] = np.float64(np.nan)
        current['delta_passive_imbalance_signed_ratio_skew'] = np.float64(np.nan)
        current['delta_passive_imbalance_buy_ratio_mean'] = np.float64(np.nan)
        current['delta_passive_imbalance_buy_ratio_std'] = np.float64(np.nan)

        # 6.8 Passive Derived Imbalance Metrics
        current['passive_cumulative_signed_imbalance'] = np.float64(0.0)
        current['passive_imbalance_persistence'] = np.float64(np.nan)
        current['passive_imbalance_volatility'] = np.float64(np.nan)

        # ========================================================================
        # 7. Divergence Metrics
        # ========================================================================
        # 7.1 Aggregated Bar-Level Divergences
        current['divergence_buy_volume_mean'] = np.float64(np.nan)
        current['divergence_buy_volume_std'] = np.float64(np.nan)
        current['divergence_sell_volume_mean'] = np.float64(np.nan)
        current['divergence_sell_volume_std'] = np.float64(np.nan)
        current['divergence_imbalance_signed_mean'] = np.float64(np.nan)
        current['divergence_imbalance_signed_std'] = np.float64(np.nan)
        current['divergence_imbalance_signed_ratio_mean'] = np.float64(np.nan)
        current['divergence_imbalance_signed_ratio_std'] = np.float64(np.nan)
        current['divergence_buy_ratio_mean'] = np.float64(np.nan)
        current['divergence_buy_ratio_std'] = np.float64(np.nan)

        # 7.2 Bucket-Level Divergences
        current['divergence_buy_volume_bucket'] = np.float64(np.nan)
        current['divergence_sell_volume_bucket'] = np.float64(np.nan)
        current['divergence_imbalance_signed_bucket'] = np.float64(np.nan)
        current['divergence_imbalance_signed_ratio_bucket'] = np.float64(np.nan)
        current['divergence_buy_ratio_bucket'] = np.float64(np.nan)

        # 7.3 Divergence Deltas
        current['delta_divergence_buy_volume_mean'] = np.float64(np.nan)
        current['delta_divergence_sell_volume_mean'] = np.float64(np.nan)
        current['delta_divergence_imbalance_signed_ratio_mean'] = np.float64(np.nan)
        current['delta_divergence_buy_ratio_mean'] = np.float64(np.nan)

        # ========================================================================
        # 8. Temporal / Tempo Structure
        # ========================================================================
        # 8.1 Time Elapsed Metrics
        current['time_elapsed_ns_mean'] = np.float64(np.nan)
        current['time_elapsed_ns_std'] = np.float64(np.nan)
        current['time_elapsed_ns_min'] = np.uint64(0)
        current['time_elapsed_ns_max'] = np.uint64(0)

        # 8.2 Pace of Contracts Traded
        current['pace_of_contracts_traded'] = np.float64(np.nan)
        current['pace_of_contracts_traded_mean'] = np.float64(np.nan)
        current['pace_of_contracts_traded_std'] = np.float64(np.nan)

        # 8.3 Temporal Log Deltas
        current['delta_time_elapsed_ns_log_mean'] = np.float64(np.nan)
        current['delta_time_elapsed_ns_log_std'] = np.float64(np.nan)
        current['delta_pace_of_contracts_traded_log_mean'] = np.float64(np.nan)
        current['delta_pace_of_contracts_traded_log_std'] = np.float64(np.nan)

        # 8.4 Derived Tempo Metrics
        current['tempo_stability'] = np.float64(np.nan)
        current['tempo_acceleration'] = np.float64(np.nan)

        # ========================================================================
        # 9. Directional Signals
        # ========================================================================
        # 9.1 Active Directional Signals
        current['active_pct_bars_positive_direction'] = np.float64(np.nan)
        current['active_directional_streak_max'] = np.uint32(0)
        current['active_directional_reversals_count'] = np.uint32(0)
        current['active_net_direction'] = np.float64(np.nan)

        # 9.2 Adjusted Directional Signals
        current['adjusted_pct_bars_positive_direction'] = np.float64(np.nan)
        current['adjusted_directional_streak_max'] = np.uint32(0)
        current['adjusted_directional_reversals_count'] = np.uint32(0)
        current['adjusted_net_direction'] = np.float64(np.nan)

        # 9.3 Passive Directional Signals
        current['passive_pct_bars_positive_direction'] = np.float64(np.nan)
        current['passive_directional_streak_max'] = np.uint32(0)
        current['passive_directional_reversals_count'] = np.uint32(0)
        current['passive_net_direction'] = np.float64(np.nan)

        # ========================================================================
        # 10. Volatility and Stability Overlays
        # ========================================================================
        current['active_price_volatility'] = np.float64(np.nan)
        current['active_spread_volatility'] = np.float64(np.nan)
        current['active_return_volatility'] = np.float64(np.nan)
        current['active_intermediate_volatility'] = np.float64(np.nan)
        current['active_spread_stability'] = np.float64(np.nan)
        current['active_flow_volatility'] = np.float64(np.nan)
        current['adjusted_price_volatility'] = np.float64(np.nan)
        current['adjusted_spread_volatility'] = np.float64(np.nan)
        current['passive_price_volatility'] = np.float64(np.nan)

        # ========================================================================
        # 12. Debug / Diagnostic
        # ========================================================================
        current['bar_ids_checksum'] = np.uint64(0)
        current['missing_bars_flag'] = np.bool_(False)
        current['processing_time_ns'] = np.uint64(0)
        current['nan_count_active_midpoint'] = np.uint32(0)
        current['nan_count_adjusted_midpoint'] = np.uint32(0)
        current['nan_count_passive_midprice'] = np.uint32(0)

        return current




    @staticmethod
    def _get_empty_bucket_deltas() -> Dict[str, Any]:
        """
        Generate NaN values for all inter-bucket delta features.
        Used for the first bucket where there is no previous bucket to compare against.
        Returns:
            Dictionary with all inter-bucket delta fields set to NaN or zero
        <><><> MUST MATCH "BUCKET_DTYPE" SCHEMA <><><>
        Reasoning for using NaN:
        1. Semantically correct (unknown ≠ zero change)
        2. Prevents model confusion
        3. Enables proper missing value handling
        4. Maintains statistical integrity
        5. Gives flexibility in downstream processing
        """

        deltas: Dict[str, Any] = {}

        # ========================================================================
        # 11.1 Active Inter-Bucket Deltas
        # ========================================================================
        deltas['bucket_delta_active_spread_vwap_log'] = np.float64(np.nan)
        deltas['bucket_delta_active_spread_vwap_std_log'] = np.float64(np.nan)
        deltas['bucket_delta_active_spread_volatility'] = np.float64(np.nan)
        deltas['bucket_delta_active_return_volatility_log'] = np.float64(np.nan)
        deltas['bucket_delta_active_intermediate_volatility_log'] = np.float64(np.nan)
        deltas['bucket_delta_active_flow_volatility_log'] = np.float64(np.nan)
        deltas['bucket_delta_active_pace_of_contracts_traded_log'] = np.float64(np.nan)
        deltas['bucket_delta_active_tempo_stability'] = np.float64(np.nan)
        deltas['bucket_delta_active_time_elapsed_ns_total_log'] = np.float64(np.nan)
        deltas['bucket_delta_active_imbalance_signed_ratio_atanh'] = np.float64(np.nan)
        deltas['bucket_delta_active_imbalance_abs_ratio_logit'] = np.float64(np.nan)
        deltas['bucket_delta_active_imbalance_persistence'] = np.float64(np.nan)
        deltas['bucket_delta_active_cumulative_signed_imbalance_norm'] = np.float64(np.nan)
        deltas['bucket_delta_active_imbalance_volatility'] = np.float64(np.nan)
        deltas['bucket_delta_active_pct_bars_positive_direction'] = np.float64(np.nan)
        deltas['bucket_delta_active_directional_streak_max'] = np.int32(0)
        deltas['bucket_delta_active_directional_reversals_count'] = np.int32(0)

        # ========================================================================
        # 11.2 Adjusted Inter-Bucket Deltas
        # ========================================================================
        deltas['bucket_delta_adjusted_spread_vwap_log'] = np.float64(np.nan)
        deltas['bucket_delta_adjusted_spread_volatility'] = np.float64(np.nan)
        deltas['bucket_delta_adjusted_return_volatility_log'] = np.float64(np.nan)
        deltas['bucket_delta_adjusted_imbalance_signed_ratio_atanh'] = np.float64(np.nan)
        deltas['bucket_delta_adjusted_imbalance_abs_ratio_logit'] = np.float64(np.nan)
        deltas['bucket_delta_adjusted_imbalance_persistence'] = np.float64(np.nan)
        deltas['bucket_delta_adjusted_imbalance_volatility'] = np.float64(np.nan)

        # ========================================================================
        # 11.3 Second-Order Deltas (Acceleration)
        # ========================================================================
        deltas['bucket_delta2_active_return_volatility_log'] = np.float64(np.nan)
        deltas['bucket_delta2_active_imbalance_signed_ratio_atanh'] = np.float64(np.nan)
        deltas['bucket_delta2_active_pace_of_contracts_traded_log'] = np.float64(np.nan)
        deltas['bucket_delta2_adjusted_return_volatility_log'] = np.float64(np.nan)
        deltas['bucket_delta2_adjusted_imbalance_signed_ratio_atanh'] = np.float64(np.nan)

        return deltas




    # ============================================================================
    # Test Functions
    # ============================================================================



    @staticmethod
    def test__bucket_full_schema(full: Dict[str, Any], empty: bool, show_valid_messages: bool = False) -> None:
        source = "EMPTY DATA" if empty else "ACTUAL DATA"
        error_message = f"Full schema validation failed for << {source} >>"
        assert HelperFunctions.validate_full_schema(
            stats=full,
            schema=BUCKET_DTYPE,
            show_valid_messages=show_valid_messages,
        ), error_message




    @staticmethod
    def test__bucket_statistics_schema(stats: Dict[str, Any], empty: bool, show_valid_messages: bool = False) -> None:
        source = "EMPTY DATA" if empty else "ACTUAL DATA"
        error_message = f"Calculations (non-delta) schema validation failed for << {source} >>"
        assert HelperFunctions.validate_statistics_schema(
            stats=stats, 
            schema=BUCKET_DTYPE, 
            delta_prefixes=["bucket_delta"],
            show_valid_messages=show_valid_messages,
        ), error_message




    @staticmethod
    def test__bucket_deltas_schema(deltas: Dict[str, Any], empty: bool, show_valid_messages: bool = False) -> None:
        source = "EMPTY DATA" if empty else "ACTUAL DATA"
        error_message = f"Delta schema validation failed for << {source} >>"
        assert HelperFunctions.validate_delta_schema(
            stats=deltas, 
            schema=BUCKET_DTYPE,
            delta_prefixes=['bucket_delta_', 'bucket_delta2_'],
            show_valid_messages=show_valid_messages,
        ), error_message




    # ============================================================================
    # Helper Functions
    # ============================================================================


    @staticmethod
    def _safe_nanstd(arr: np.ndarray, num_bars: int) -> np.float64:
        """Calculate std with ddof=1, returning 0.0 if insufficient data."""
        if num_bars <= 1:
            return np.float64(0.0)
        valid_count = np.count_nonzero(~np.isnan(arr))
        if valid_count <= 1:
            return np.float64(0.0)
        return np.float64(np.nanstd(arr, ddof=1))
    



    @staticmethod 
    def _safe_nanmean(arr: np.ndarray) -> np.float64:
        """Calculate mean, returning NaN if array is empty or all NaN."""
        if np.count_nonzero(~np.isnan(arr)) == 0:
            return np.float64(np.nan)
        return np.float64(np.nanmean(arr))




    @staticmethod
    def _safe_nanmin(arr: np.ndarray) -> np.float64:
        valid = arr[~np.isnan(arr)]
        return np.float64(np.nanmin(valid)) if len(valid) > 0 else np.float64(np.nan)




    @staticmethod
    def _safe_nanmax(arr: np.ndarray) -> np.float64:
        valid = arr[~np.isnan(arr)]
        return np.float64(np.nanmax(valid)) if len(valid) > 0 else np.float64(np.nan)




    @staticmethod
    def _safe_skew(arr: np.ndarray, num_bars: int) -> np.float64:
        """Calculate skewness, returning 0.0 if insufficient data."""
        if num_bars <= 2:
            return np.float64(0.0)
        valid = arr[~np.isnan(arr)]
        if len(valid) <= 2:
            return np.float64(0.0)
        result = scipy_stats.skew(valid)
        return np.float64(result) if not np.isnan(result) else np.float64(0.0)




    @staticmethod
    def _safe_log_ratio(current: float, previous: float) -> np.float64:
        """Calculate log ratio with epsilon for stability."""
        if np.isnan(current) or np.isnan(previous):
            return np.float64(np.nan)
        if current <= 0 or previous <= 0:
            return np.float64(np.nan)
        return np.float64(np.log((current + EPSILON) / (previous + EPSILON)))




    @staticmethod
    def _safe_atanh_diff(current_ratio: float, previous_ratio: float) -> np.float64:
        """Calculate difference in atanh-transformed ratios."""
        if np.isnan(current_ratio) or np.isnan(previous_ratio):
            return np.float64(np.nan)
        current_clipped = np.clip(current_ratio, -0.9999, 0.9999)
        previous_clipped = np.clip(previous_ratio, -0.9999, 0.9999)
        return np.float64(np.arctanh(current_clipped) - np.arctanh(previous_clipped))




    @staticmethod
    def _safe_logit_diff(current_ratio: float, previous_ratio: float) -> np.float64:
        """Calculate difference in logit-transformed ratios."""
        if np.isnan(current_ratio) or np.isnan(previous_ratio):
            return np.float64(np.nan)
        current_clipped = np.clip(current_ratio, 0.0001, 0.9999)
        previous_clipped = np.clip(previous_ratio, 0.0001, 0.9999)
        current_logit = np.log(current_clipped / (1.0 - current_clipped))
        previous_logit = np.log(previous_clipped / (1.0 - previous_clipped))
        return np.float64(current_logit - previous_logit)




    @staticmethod
    def _calculate_directional_signals(
        directions: np.ndarray, 
        num_bars: int
    ) -> Dict[str, Any]:
        """
        Calculate directional signal metrics from boolean direction array.
        
        Args:
            directions: Boolean array where True = positive direction
            num_bars: Number of bars
            
        Returns:
            Dictionary with pct_positive, streak_max, reversals_count, net_direction
        """
        if num_bars == 0:
            return {
                'pct_positive': np.float64(np.nan),
                'streak_max': np.uint32(0),
                'reversals_count': np.uint32(0),
                'net_direction': np.float64(np.nan),
            }
        
        positive_count = np.sum(directions)
        pct_positive = np.float64(positive_count) / np.float64(num_bars)
        
        # Directional streak calculation
        direction_int = directions.astype(np.int8)
        direction_changes = np.diff(direction_int) != 0
        streak_lengths = np.diff(np.concatenate(([0], np.where(direction_changes)[0] + 1, [num_bars])))
        streak_max = np.uint32(np.max(streak_lengths)) if len(streak_lengths) > 0 else np.uint32(0)
        
        # Reversals count
        reversals_count = np.uint32(np.sum(direction_changes))
        
        # Net direction
        up_count = positive_count
        down_count = num_bars - positive_count
        net_direction = np.float64(up_count - down_count) / np.float64(num_bars)
        
        return {
            'pct_positive': pct_positive,
            'streak_max': streak_max,
            'reversals_count': reversals_count,
            'net_direction': net_direction,
        }




    @staticmethod
    def _calculate_imbalance_persistence(
        bar_signed_values: np.ndarray,
        bucket_signed_value: float,
        num_bars: int
    ) -> np.float64:
        """
        Calculate fraction of bars where sign matches bucket-level sign.
        
        Args:
            bar_signed_values: Array of bar-level signed imbalance values
            bucket_signed_value: Bucket-level signed imbalance
            num_bars: Number of bars
            
        Returns:
            Persistence ratio [0, 1] or 0.0 if bucket sign is 0
        """
        bucket_sign = np.sign(bucket_signed_value)
        if bucket_sign == 0 or num_bars == 0:
            return np.float64(0.0)
        
        matching_signs = np.sum(np.sign(bar_signed_values) == bucket_sign)
        return np.float64(matching_signs) / np.float64(num_bars)



