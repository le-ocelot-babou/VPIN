"""
Historical calculations system.

Synchronous processing with threading ONLY for disk I/O.

Data flow:
1. User provides pre-calculated volumebar stats via save_latest_volume_bar()
2. Store in volumebar circular buffer
3. Derive buckets from last N volumebars (fixed and/or interval)
4. Calculate metrics on buckets (if training disabled)
5. Archive to Parquet when chunks full
6. Consolidate all chunks at end

Usage:
    config = HistoricalConfig(
        # Volume Bar
        volumebar_contracts_per_bar=VOLUME_BAR_SIZE,
        volumebar_lookback_buffer_size=50,
        volumebar_archive_size=DATA_ARCHIVE_SIZE,

        # Bucket (Fixed)
        enable_fixed_buckets=True,
        fixed_bucket_volumebars_per_bucket=50, #V__BUCKETS_PER_METRIC
        fixed_bucket_archive_size=DATA_ARCHIVE_SIZE,

        # Bucket (Interval)
        enable_interval_buckets=True, #V__BUCKETS_PER_METRIC
        interval_bucket_volumebars_per_bucket=50,
        interval_bucket_archive_size=DATA_ARCHIVE_SIZE,

        # Metrics (On/Off)
        training=True, # If Training: No metrics are generated
        # Metrics (Fixed)
        fixed_metrics_buckets_per_metric=50,
        fixed_metrics_archive_size=500,
        # Metrics (Interval)
        interval_metrics_buckets_per_metric=50,
        interval_metric_archive_size=500,

        # Override Defaults
        output_dir=EXPORT_DIR,
        max_workers=3,
        include_debug_timestamps=True,
    )

    calculator = HistoricalCalculations(config)
    
    for volumebar_stats in your_data:
        calculator.save_latest_volume_bar(volumebar_stats)
    
    calculator.processing_complete()
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Dict, Any, List, Optional, Union
import numpy as np

from .config import HistoricalConfig
from .models import (
    StructuredCircularBuffer,
    VOLUMEBAR_DTYPE,
    BUCKET_DTYPE,
    METRICS_DTYPE,
    BufferName,
    ChunkType,
    ArchivalTask
)
from .archival_manager import HistoricalArchivalManager


logger = logging.getLogger(__name__)




class HistoricalCalculations:
    """
    Historical volumebar calculations with structured array storage.
    
    Processes volumebars and derives buckets (fixed/interval) and metrics.
    Uses synchronous processing with threading only for disk I/O archival.
    """
    
    def __init__(self, config: HistoricalConfig, bucket_methods):
        """
        Initialize historical calculator.
        
        Args:
            config: Historical configuration with bucket and metrics settings
        """

        self.config = config
        self.bucket_methods = bucket_methods
        

        # VolumeBar: [ circular buffer ]
        self.volumebar_buffer = StructuredCircularBuffer(
            lookback_window=config.volumebar_lookback_buffer_size,
            dtype=VOLUMEBAR_DTYPE
        )
        
        # VolumeBar: [ archival chunk ]
        self.volumebar_chunk = {
            'data': np.empty(config.volumebar_archive_size, dtype=VOLUMEBAR_DTYPE),
            'count': 0,
            'start_id': 0
        }
        

        # Fixed Bucket: [ components ] (if enabled)
        self.fixed_buckets_buffer: Optional[StructuredCircularBuffer] = None
        self.fixed_bucket_chunk: Optional[Dict[str, Any]] = None

        if config.enable_fixed_buckets:

            self.fixed_buckets_buffer = StructuredCircularBuffer(
                lookback_window=config.fixed_bucket_volumebars_per_bucket,
                dtype=BUCKET_DTYPE
            )
            self.fixed_bucket_chunk = {
                'data': np.empty(config.fixed_bucket_archive_size, dtype=BUCKET_DTYPE),
                'count': 0,
                'start_id': 0
            }
        

        # Interval Bucket: [ components ] (if enabled)
        self.interval_buckets_buffer: Optional[StructuredCircularBuffer] = None
        self.interval_bucket_chunk: Optional[Dict[str, Any]] = None

        if config.enable_interval_buckets:
            self.interval_buckets_buffer = StructuredCircularBuffer(
                lookback_window=config.interval_bucket_volumebars_per_bucket,
                dtype=BUCKET_DTYPE
            )
            self.interval_bucket_chunk = {
                'data': np.empty(config.interval_bucket_archive_size, dtype=BUCKET_DTYPE),
                'count': 0,
                'start_id': 0
            }
        

        # Fixed Metrics: [ components ] (if training disabled)
        self.fixed_metrics_buffer: Optional[StructuredCircularBuffer] = None
        self.fixed_metrics_chunk: Optional[Dict[str, Any]] = None

        if not config.training and config.enable_fixed_buckets:
            self.fixed_metrics_buffer = StructuredCircularBuffer(
                lookback_window=config.fixed_bucket_volumebars_per_bucket, #fixed_metrics_buckets_per_metric
                dtype=METRICS_DTYPE
            )
            self.fixed_metrics_chunk = {
                'data': np.empty(config.fixed_metrics_archive_size, dtype=METRICS_DTYPE),
                'count': 0,
                'start_id': 0
            }
        

        # Interval Metrics: [ components ] (if training disabled)
        self.interval_metrics_buffer: Optional[StructuredCircularBuffer] = None
        self.interval_metrics_chunk: Optional[Dict[str, Any]] = None

        if not config.training and config.enable_interval_buckets:
            self.interval_metrics_buffer = StructuredCircularBuffer(
                lookback_window=config.interval_bucket_volumebars_per_bucket, #interval_metrics_buckets_per_metric
                dtype=METRICS_DTYPE
            )
            self.interval_metrics_chunk = {
                'data': np.empty(config.interval_metric_archive_size, dtype=METRICS_DTYPE),
                'count': 0,
                'start_id': 0
            }
        

        # Thread pool for archival
        self.thread_pool = ThreadPoolExecutor(max_workers=config.max_workers)
        self.pending_futures: List[Future] = []

        # Test Bucket Schema: Enable/Disable
        self.test_schema: bool = config.test_schema
        self.test_schema_show_valid_message: bool = config.test_schema_show_valid_message
        
        # Archival manager
        self.archival_manager = HistoricalArchivalManager(config)
        
        # Counters
        self.volumebar_count = 0
        self.volumebar_incomplete_count = 0
        self.fixed_bucket_count = 0
        self.interval_bucket_count = 0
        self.chunks_archived = {
            'volumebar': 0,
            'fixed_buckets': 0,
            'interval_buckets': 0,
            'fixed_metrics': 0,
            'interval_metrics': 0
        }
    



    def save_latest_volume_bar(self, stats: Dict[str, Any]) -> None:
        """
        Add pre-calculated volumebar statistics.
        
        This triggers bucket and metrics calculations based on configuration.
        
        Args:
            stats: Dictionary with all VOLUMEBAR_DTYPE fields.
                   Values already in correct numpy dtypes.
                   Must include 'bar_complete' field.
        """
        # Add to volumebar buffer
        self.volumebar_buffer.add_row(stats)
        self.volumebar_count += 1
        
        # Store raw volumebar for archival
        self._store_volumebar_raw()

        if stats is not None and not stats['bar_complete']:

            # If incomplete, update respective counter
            self.volumebar_incomplete_count += 1
        
        else:

            # Process [ Buckets ] if COMPLETE VolumeBar(s) are available
            #
            # Logic inside of `_process_buckets()`:
            #  - Only runs if the latest VolumeBar was COMPLETE
            #  - Considers if there is sufficient data in the lookback buffer
            self._process_buckets()

        
        # Archive if any chunks are full
        if self._should_archive():
            self._archive_all()
    



    def get_previous_volumebar_row_view(self):
        """
        Get view of previous VolumeBar row.
        
        Used in VolumeBar Delta Calculations
        """
        return self.volumebar_buffer.get_previous_row_view()
    



    def _store_volumebar_raw(self) -> None:
        """Store complete volumebar row for archival."""
        prev_row = self.volumebar_buffer.get_previous_row_view()
        if prev_row is None:
            return
        
        # Iterator already updated for latest addition
        idx = self.volumebar_chunk['count']
        self.volumebar_chunk['data'][idx] = prev_row
        
        if self.volumebar_chunk['count'] == 0:
            self.volumebar_chunk['start_id'] = int(prev_row['id'])
        
        # Update iterator
        self.volumebar_chunk['count'] += 1
    



    def _process_buckets(self) -> None:
        """
        Called by VolumeBar.save_latest_volume_bar()

        Process bucket calculations based on configuration.
        
        Checks if fixed and/or interval buckets should be calculated,
        then triggers metrics calculations if training is disabled.
        Only processes if the most recent volumebar is complete.
        """
        # # Get the most recent VolumeBar
        # last_bar = self.volumebar_buffer.get_previous_row_view()
    
        # # Only process buckets if the last volumebar is complete
        # if last_bar is None or not last_bar['bar_complete']:
        #     return
        

        # [ CHECK IF ]
        # - There configuration is enabled AND
        # - There is enough lookback data in the buffer to perform the calculation

        # Process [ Fixed Bucket ]
        if self.config.enable_fixed_buckets and self._should_calculate_fixed_bucket():
            self._calculate_fixed_bucket()
            
            # Calculate [ Fixed Metric ] if training disabled
            if not self.config.training and self.fixed_metrics_buffer is not None:
                self._calculate_fixed_metrics()
        

        # Process [ Interval Buckets ]
        if self.config.enable_interval_buckets and self._should_calculate_interval_bucket():
            self._calculate_interval_bucket()
            
            # Calculate [ Interval Metrics ] if training disabled
            if not self.config.training and self.interval_metrics_buffer is not None:
                self._calculate_interval_metrics()
        



    def _should_calculate_fixed_bucket(self) -> bool:
        """
        Determine if fixed bucket should be calculated.
        
        Fixed buckets calculated every N volumebars (where N = fixed_bucket_volumebars_per_bucket).
        
        Returns:
            True if fixed bucket calculation should run
        """

        lookback = self.config.fixed_bucket_volumebars_per_bucket


        # # #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
        # # #### [ TEST ] - If Bucket Generation Interval Matches Configulartion
        # if self.volumebar_count < (lookback * 3):
        #     if (self.volumebar_count >= lookback and 
        #             self.volumebar_count % lookback == 0):
        #         print(f"DEBUG [FIXED_BUCKET]: Bucket generated at volumebar_count={self.volumebar_count}")
        # # #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
            

        # Only processes buckets IF the last VolumeBar is complete! See _process_buckets() logic
        return (self.volumebar_count >= lookback and 
                self.volumebar_count % lookback == 0)
    




    def _should_calculate_interval_bucket(self) -> bool:
        """
        Determine if interval bucket should be calculated.
        
        Interval buckets calculated on every volumebar after warmup.
        
        Returns:
            True if interval bucket calculation should run
        """

        # # #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
        # # #### [ TEST ] - If Bucket Generation Interval Matches Configulartion
        # if self.volumebar_count < (self.config.fixed_bucket_volumebars_per_bucket * 3):
        #     if (self.volumebar_count >= self.config.interval_bucket_volumebars_per_bucket):
        #         print(f"DEBUG [INTERVAL_BUCKET]: Bucket generated at volumebar_count={self.volumebar_count}")
        # # #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####


        # Only processes buckets IF the last VolumeBar is complete! See _process_buckets() logic
        return self.volumebar_count >= self.config.interval_bucket_volumebars_per_bucket
    



    def _calculate_fixed_bucket(self) -> None:
        """

        Calculate fixed bucket from volumebar buffer.
        
        Steps:
        1. Get last N volumebars (where N = fixed_bucket_volumebars_per_bucket)
        2. Calculate bucket statistics
        3. Calculate bucket deltas (using previous fixed bucket or empty deltas)
        4. Store in fixed bucket buffer
        5. Store for archival

        """

        if self.fixed_buckets_buffer is None:
            return
        
        # Get volumebar data for calculation
        lookback = self.config.fixed_bucket_volumebars_per_bucket
        volumebar_data = self.volumebar_buffer.get_last_n_rows(lookback)
        
        if volumebar_data is None or len(volumebar_data) < lookback:
            logger.warning(f"Insufficient volumebar data for fixed bucket calculation")
            return
        
        try:

            # Calculate bucket statistics
            bucket_id = np.uint32(self.fixed_bucket_count)
            bucket_stats = self.bucket_methods.calculate_bucket_statistics(
                bars=volumebar_data,
                bucket_id=bucket_id,
                bucket_type="fixed"
            )


            # #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
            # #### [ TEST ] - Test Bucket Schema (ONLY statistics)
            if self.test_schema:
                self.bucket_methods.test__bucket_statistics_schema(stats=bucket_stats, empty=False, show_valid_messages=self.test_schema_show_valid_message)
            # #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
            

            # Calculate bucket deltas
            previous_bucket = self.fixed_buckets_buffer.get_previous_row_view()

            # Not the first bucket - use actual deltas
            if previous_bucket is not None:

                bucket_deltas = self.bucket_methods.calculate_bucket_deltas(
                    current=bucket_stats,
                    previous=previous_bucket
                )

                # #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
                # #### [ TEST ] - Test Bucket Schema (ONLY deltas)
                if self.test_schema:
                    self.bucket_methods.test__bucket_deltas_schema(deltas=bucket_deltas, empty=False, show_valid_messages=self.test_schema_show_valid_message)
                # #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####


            # First bucket - use empty deltas
            else:
                bucket_deltas = self.bucket_methods._get_empty_bucket_deltas()

                # #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
                # #### [ TEST ] - Test EMPTY Bucket Schema (Empty x3: Deltas, Statistics, and Full)
                if self.test_schema:
                    self.bucket_methods.test__bucket_deltas_schema(deltas=bucket_deltas, empty=True, show_valid_messages=self.test_schema_show_valid_message)

                    empty_stats = self.bucket_methods._get_empty_bucket_statistics()
                    self.bucket_methods.test__bucket_statistics_schema(empty_stats, empty=True, show_valid_messages=self.test_schema_show_valid_message)
                    empty_stats.update(bucket_deltas) # Does not alter `bucket_deltas`, solely a dictionary read
                    self.bucket_methods.test__bucket_full_schema(full=empty_stats, empty=True, show_valid_messages=self.test_schema_show_valid_message)
                # #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
            

            
            # Merge bucket stats with deltas (dict)
            bucket_stats.update(bucket_deltas)


            # #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
            # #### [ TEST ] - Test Bucket Schema (ONLY full)
            if self.test_schema:
                self.bucket_methods.test__bucket_full_schema(full=bucket_stats, empty=False, show_valid_messages=self.test_schema_show_valid_message)
            # #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####


            # Create structured NumPy array from dict
            bucket_ndarray = np.array(
                [tuple(bucket_stats[name] for name in BUCKET_DTYPE.names)], 
                dtype=BUCKET_DTYPE
            )

            # Add to buffer (using add_row for numpy array)
            self.fixed_buckets_buffer.add_row(bucket_ndarray)
            
            # Store for archival
            self._store_bucket_result(ChunkType.FIXED, bucket_ndarray)
            
            # Update iterator for next addition
            self.fixed_bucket_count += 1
            
        except Exception as e:
            logger.error(f"Interval bucket calculation failed: {e}", exc_info=True)
    



    def _calculate_interval_bucket(self) -> None:
        """

        Calculate interval bucket from volumebar buffer.
        
        Steps:
        1. Get last N volumebars (where N = interval_bucket_volumebars_per_bucket)
        2. Calculate bucket statistics
        3. Calculate bucket deltas (using previous interval bucket or empty deltas)
        4. Store in interval bucket buffer
        5. Store for archival

        """

        if self.interval_buckets_buffer is None:
            return
        
        # Get volumebar data for calculation
        lookback = self.config.interval_bucket_volumebars_per_bucket
        volumebar_data = self.volumebar_buffer.get_last_n_rows(lookback)
        
        if volumebar_data is None or len(volumebar_data) < lookback:
            logger.warning(f"Insufficient volumebar data for interval bucket calculation")
            return
        
        try:

            # Calculate bucket statistics
            bucket_id = np.uint32(self.interval_bucket_count)
            bucket_stats = self.bucket_methods.calculate_bucket_statistics(
                bars=volumebar_data,
                bucket_id=bucket_id,
                bucket_type="interval"
            )


            # #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
            # #### [ TEST ] - Test Bucket Schema (ONLY statistics)
            if self.test_schema:
                self.bucket_methods.test__bucket_statistics_schema(stats=bucket_stats, empty=False, show_valid_messages=self.test_schema_show_valid_message)
            # #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
            
            
            # Calculate bucket deltas
            previous_bucket = self.interval_buckets_buffer.get_previous_row_view()

            # Not the first bucket - use actual deltas
            if previous_bucket is not None:

                bucket_deltas = self.bucket_methods.calculate_bucket_deltas(
                    current=bucket_stats,
                    previous=previous_bucket
                )

                # #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
                # #### [ TEST ] - Test Bucket Schema (ONLY deltas)
                if self.test_schema:
                    self.bucket_methods.test__bucket_deltas_schema(deltas=bucket_deltas, empty=False, show_valid_messages=self.test_schema_show_valid_message)
                # #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####


            # First bucket - use empty deltas
            else:
                bucket_deltas = self.bucket_methods._get_empty_bucket_deltas()

                # #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
                # #### [ TEST ] - Test EMPTY Bucket Schema (Empty x3: Deltas, Statistics, and Full)
                if self.test_schema:
                    self.bucket_methods.test__bucket_deltas_schema(deltas=bucket_deltas, empty=True, show_valid_messages=self.test_schema_show_valid_message)

                    empty_stats = self.bucket_methods._get_empty_bucket_statistics()
                    self.bucket_methods.test__bucket_statistics_schema(empty_stats, empty=True, show_valid_messages=self.test_schema_show_valid_message)
                    empty_stats.update(bucket_deltas) # Does not alter `bucket_deltas`, solely a dictionary read
                    self.bucket_methods.test__bucket_full_schema(full=empty_stats, empty=True, show_valid_messages=self.test_schema_show_valid_message)
                # #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
            


            # Merge bucket stats with deltas (dict)
            bucket_stats.update(bucket_deltas)


            # #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
            # #### [ TEST ] - Test Bucket Schema (ONLY full)
            if self.test_schema:
                self.bucket_methods.test__bucket_full_schema(full=bucket_stats, empty=False, show_valid_messages=self.test_schema_show_valid_message)
            # #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####


            # Create structured NumPy array from dict
            bucket_ndarray = np.array(
                [tuple(bucket_stats[name] for name in BUCKET_DTYPE.names)], 
                dtype=BUCKET_DTYPE
            )

            # Add to buffer (using add_row for numpy array)
            self.interval_buckets_buffer.add_row(bucket_ndarray)
            
            # Store for archival
            self._store_bucket_result(ChunkType.INTERVAL, bucket_ndarray)
            
            # Update iterator for next addition
            self.interval_bucket_count += 1
            
        except Exception as e:
            logger.error(f"Interval bucket calculation failed: {e}", exc_info=True)
    



    def _store_bucket_result(self, bucket_type: str, bucket_stats: np.ndarray) -> None:
        """
        Store bucket result for archival.
        
        Args:
            bucket_type: ChunkType.FIXED or ChunkType.INTERVAL
            bucket_stats: Structured array with BUCKET_DTYPE
        """

        if bucket_type == ChunkType.FIXED and self.fixed_bucket_chunk is not None:
            idx = self.fixed_bucket_chunk['count']
            self.fixed_bucket_chunk['data'][idx] = bucket_stats
            
            if self.fixed_bucket_chunk['count'] == 0:
                self.fixed_bucket_chunk['start_id'] = int(bucket_stats['id'])
            
            self.fixed_bucket_chunk['count'] += 1
            

        elif bucket_type == ChunkType.INTERVAL and self.interval_bucket_chunk is not None:
            idx = self.interval_bucket_chunk['count']
            self.interval_bucket_chunk['data'][idx] = bucket_stats
            
            if self.interval_bucket_chunk['count'] == 0:
                self.interval_bucket_chunk['start_id'] = int(bucket_stats['id'])
            
            self.interval_bucket_chunk['count'] += 1
    



    def _calculate_fixed_metrics(self) -> None:
        """
        Calculate metrics on fixed bucket buffer.
        
        TODO: Implement metrics calculation logic.
        - Input: fixed_buckets_buffer.data (last N buckets)
        - Input: previous_metric from fixed_metrics_buffer (index=-1)
        - Output: metric_stats dictionary with METRICS_DTYPE fields
        
        Placeholder implementation.
        """
        if self.fixed_metrics_buffer is None:
            return
        
        try:
            # Get bucket buffer data
            bucket_data = self.fixed_buckets_buffer.data if self.fixed_buckets_buffer else None
            previous_metric = self.fixed_metrics_buffer.get_previous_row_view()
            
            # TODO: Calculate metrics from bucket data
            # metric_stats = your_metrics_calculation_function(bucket_data, previous_metric)
            
            # Placeholder: create empty metric
            metric_stats = {
                'id': self.fixed_bucket_count - 1,
                'bucket_type': ChunkType.FIXED
                # TODO: Add calculated metric fields
            }
            
            # Add to buffer (add_row accepts either ndarray or dict)
            self.fixed_metrics_buffer.add_row(metric_stats)
            
            # Store for archival
            self._store_metrics_result(ChunkType.FIXED, metric_stats)


            # (!!!) (!?) <<< PERFORM SCHEMA TEST HERE >>>
            if self.test_schema:
                # self.bucket_methods.
                #     test__bucket_full_schema(full: Dict[str, Any], empty: bool)
                #     test__bucket_statistics_schema(stats: Dict[str, Any], empty: bool)
                #     test__bucket_deltas_schema(deltas: Dict[str, Any], empty: bool)
                pass

            
        except Exception as e:
            logger.error(f"Fixed metrics calculation failed: {e}")
    



    def _calculate_interval_metrics(self) -> None:
        """
        Calculate metrics on interval bucket buffer.
        
        TODO: Implement metrics calculation logic.
        - Input: interval_buckets_buffer.data (last N buckets)
        - Input: previous_metric from interval_metrics_buffer (index=-1)
        - Output: metric_stats dictionary with METRICS_DTYPE fields
        
        Placeholder implementation.
        """
        if self.interval_metrics_buffer is None:
            return
        
        try:
            # Get bucket buffer data
            bucket_data = self.interval_buckets_buffer.data if self.interval_buckets_buffer else None
            previous_metric = self.interval_metrics_buffer.get_previous_row_view()
            
            # TODO: Calculate metrics from bucket data
            # metric_stats = your_metrics_calculation_function(bucket_data, previous_metric)
            
            # Placeholder: create empty metric
            metric_stats = {
                'id': self.interval_bucket_count - 1,
                'bucket_type': ChunkType.INTERVAL
                # TODO: Add calculated metric fields
            }
            
            # Add to buffer
            self.interval_metrics_buffer.add_row(metric_stats)
            
            # Store for archival
            self._store_metrics_result(ChunkType.INTERVAL, metric_stats)


            # (!!!) (!?) <<< PERFORM SCHEMA TEST HERE >>>
            if self.test_schema:
                # self.bucket_methods.
                #     test__bucket_full_schema(full: Dict[str, Any], empty: bool)
                #     test__bucket_statistics_schema(stats: Dict[str, Any], empty: bool)
                #     test__bucket_deltas_schema(deltas: Dict[str, Any], empty: bool)
                pass
            
            
        except Exception as e:
            logger.error(f"Interval metrics calculation failed: {e}")
    



    def _store_metrics_result(self, metrics_type: str, metric_stats: Dict[str, Any]) -> None:
        """
        Store metrics result for archival.
        
        Args:
            metrics_type: ChunkType.FIXED or ChunkType.INTERVAL
            metric_stats: Dictionary with METRICS_DTYPE fields
        """

        # Convert dict to structured array row (Metric: !!!)
        metric_row = np.array(
            [(metric_stats.get(name, 0) for name in METRICS_DTYPE.names)],
            dtype=METRICS_DTYPE
        )[0]
        
        if metrics_type == ChunkType.FIXED and self.fixed_metrics_chunk is not None:
            idx = self.fixed_metrics_chunk['count']
            self.fixed_metrics_chunk['data'][idx] = metric_row
            
            if self.fixed_metrics_chunk['count'] == 0:
                self.fixed_metrics_chunk['start_id'] = int(metric_stats['id'])
            
            self.fixed_metrics_chunk['count'] += 1
            
        elif metrics_type == ChunkType.INTERVAL and self.interval_metrics_chunk is not None:
            idx = self.interval_metrics_chunk['count']
            self.interval_metrics_chunk['data'][idx] = metric_row
            
            if self.interval_metrics_chunk['count'] == 0:
                self.interval_metrics_chunk['start_id'] = int(metric_stats['id'])
            
            self.interval_metrics_chunk['count'] += 1
    



    def _should_archive(self) -> bool:
        """
        Check if any chunk is full and needs archival.
        
        Returns:
            True if archival needed
        """
        # Check volumebar chunk
        if self.volumebar_chunk['count'] >= self.config.volumebar_archive_size:
            return True
        
        # Check fixed bucket chunk
        if (self.fixed_bucket_chunk is not None and 
            self.fixed_bucket_chunk['count'] >= self.config.fixed_bucket_archive_size):
            return True
        
        # Check interval bucket chunk
        if (self.interval_bucket_chunk is not None and 
            self.interval_bucket_chunk['count'] >= self.config.interval_bucket_archive_size):
            return True
        
        # Check fixed metrics chunk
        if (self.fixed_metrics_chunk is not None and 
            self.fixed_metrics_chunk['count'] >= self.config.fixed_metrics_archive_size):
            return True
        
        # Check interval metrics chunk
        if (self.interval_metrics_chunk is not None and 
            self.interval_metrics_chunk['count'] >= self.config.interval_metric_archive_size):
            return True
        
        return False
    



    def _archive_all(self) -> None:
        """
        Archive all full chunks using thread pool.
        
        Main thread continues processing while archival happens in background.
        """
        # Archive volumebar
        if self.volumebar_chunk['count'] > 0:
            count = self.volumebar_chunk['count']
            data_copy = np.copy(self.volumebar_chunk['data'][:count])
            start_id = int(self.volumebar_chunk['start_id'])
            chunk_id = self.chunks_archived['volumebar']
            
            future = self.thread_pool.submit(
                self._archive_volumebar_sync,
                data_copy,
                start_id,
                chunk_id
            )
            self.pending_futures.append(future)
            
            self._reset_volumebar_chunk()
            self.chunks_archived['volumebar'] += 1
        
        # Archive fixed buckets
        if self.fixed_bucket_chunk is not None and self.fixed_bucket_chunk['count'] > 0:
            count = self.fixed_bucket_chunk['count']
            data_copy = np.copy(self.fixed_bucket_chunk['data'][:count])
            start_id = int(self.fixed_bucket_chunk['start_id'])
            chunk_id = self.chunks_archived['fixed_buckets']
            
            future = self.thread_pool.submit(
                self._archive_bucket_sync,
                ChunkType.FIXED,
                data_copy,
                start_id,
                chunk_id
            )
            self.pending_futures.append(future)
            
            self._reset_bucket_chunk(ChunkType.FIXED)
            self.chunks_archived['fixed_buckets'] += 1
        
        # Archive interval buckets
        if self.interval_bucket_chunk is not None and self.interval_bucket_chunk['count'] > 0:
            count = self.interval_bucket_chunk['count']
            data_copy = np.copy(self.interval_bucket_chunk['data'][:count])
            start_id = int(self.interval_bucket_chunk['start_id'])
            chunk_id = self.chunks_archived['interval_buckets']
            
            future = self.thread_pool.submit(
                self._archive_bucket_sync,
                ChunkType.INTERVAL,
                data_copy,
                start_id,
                chunk_id
            )
            self.pending_futures.append(future)
            
            self._reset_bucket_chunk(ChunkType.INTERVAL)
            self.chunks_archived['interval_buckets'] += 1
        
        # Archive fixed metrics
        if self.fixed_metrics_chunk is not None and self.fixed_metrics_chunk['count'] > 0:
            count = self.fixed_metrics_chunk['count']
            data_copy = np.copy(self.fixed_metrics_chunk['data'][:count])
            start_id = int(self.fixed_metrics_chunk['start_id'])
            chunk_id = self.chunks_archived['fixed_metrics']
            
            future = self.thread_pool.submit(
                self._archive_metrics_sync,
                ChunkType.FIXED,
                data_copy,
                start_id,
                chunk_id
            )
            self.pending_futures.append(future)
            
            self._reset_metrics_chunk(ChunkType.FIXED)
            self.chunks_archived['fixed_metrics'] += 1
        
        # Archive interval metrics
        if self.interval_metrics_chunk is not None and self.interval_metrics_chunk['count'] > 0:
            count = self.interval_metrics_chunk['count']
            data_copy = np.copy(self.interval_metrics_chunk['data'][:count])
            start_id = int(self.interval_metrics_chunk['start_id'])
            chunk_id = self.chunks_archived['interval_metrics']
            
            future = self.thread_pool.submit(
                self._archive_metrics_sync,
                ChunkType.INTERVAL,
                data_copy,
                start_id,
                chunk_id
            )
            self.pending_futures.append(future)
            
            self._reset_metrics_chunk(ChunkType.INTERVAL)
            self.chunks_archived['interval_metrics'] += 1
    



    def _archive_volumebar_sync(
        self,
        data: np.ndarray,
        start_id: int,
        chunk_id: int
    ) -> None:
        """
        Archive volumebar chunk (blocking, runs in thread pool).
        
        Args:
            data: Copy of volumebar data to archive
            start_id: Starting ID for this chunk
            chunk_id: Chunk identifier
        """
        end_id = int(data['id'][-1])
        count = len(data)
        
        task = ArchivalTask(
            buffer_name=BufferName.VOLUMEBAR,
            chunk_type=ChunkType.RAW,
            chunk_id=chunk_id,
            data=data,
            metadata={
                'start_row': start_id,
                'end_row': end_id,
                'count': count
            },
            processed_timestamp_ns=int(time.time_ns()) if self.config.include_debug_timestamps else None
        )
        
        self.archival_manager.write_to_parquet(task)
        logger.info(f"Volumebar chunk {chunk_id} archived")
    



    def _archive_bucket_sync(
        self,
        bucket_type: str,
        data: np.ndarray,
        start_id: int,
        chunk_id: int
    ) -> None:
        """
        Archive bucket chunk (blocking, runs in thread pool).
        
        Args:
            bucket_type: ChunkType.FIXED or ChunkType.INTERVAL
            data: Copy of bucket data to archive
            start_id: Starting ID for this chunk
            chunk_id: Chunk identifier
        """
        end_id = int(data['id'][-1])
        count = len(data)
        
        task = ArchivalTask(
            buffer_name=BufferName.BUCKET,
            chunk_type=f'{bucket_type}',
            chunk_id=chunk_id,
            data=data,
            metadata={
                'start_row': start_id,
                'end_row': end_id,
                'count': count,
                'bucket_type': bucket_type
            },
            processed_timestamp_ns=int(time.time_ns()) if self.config.include_debug_timestamps else None
        )
        
        self.archival_manager.write_to_parquet(task)
        logger.info(f"{bucket_type.capitalize()} bucket chunk {chunk_id} archived")
    



    def _archive_metrics_sync(
        self,
        metrics_type: str,
        data: np.ndarray,
        start_id: int,
        chunk_id: int
    ) -> None:
        """
        Archive metrics chunk (blocking, runs in thread pool).
        
        Args:
            metrics_type: ChunkType.FIXED or ChunkType.INTERVAL
            data: Copy of metrics data to archive
            start_id: Starting ID for this chunk
            chunk_id: Chunk identifier
        """
        end_id = int(data['id'][-1])
        count = len(data)
        
        task = ArchivalTask(
            buffer_name=BufferName.METRIC,
            chunk_type=f'{metrics_type}',
            chunk_id=chunk_id,
            data=data,
            metadata={
                'start_row': start_id,
                'end_row': end_id,
                'count': count,
                'metrics_type': metrics_type
            },
            processed_timestamp_ns=int(time.time_ns()) if self.config.include_debug_timestamps else None
        )
        
        self.archival_manager.write_to_parquet(task)
        logger.info(f"{metrics_type.capitalize()} metrics chunk {chunk_id} archived")
    



    def _reset_volumebar_chunk(self) -> None:
        """Reset volumebar chunk for new data."""
        self.volumebar_chunk['count'] = 0
        self.volumebar_chunk['start_id'] = 0
    



    def _reset_bucket_chunk(self, bucket_type: str) -> None:
        """
        Reset bucket chunk for new data.
        
        Args:
            bucket_type: ChunkType.FIXED or ChunkType.INTERVAL
        """
        if bucket_type == ChunkType.FIXED and self.fixed_bucket_chunk is not None:
            self.fixed_bucket_chunk['count'] = 0
            self.fixed_bucket_chunk['start_id'] = 0
        elif bucket_type == ChunkType.INTERVAL and self.interval_bucket_chunk is not None:
            self.interval_bucket_chunk['count'] = 0
            self.interval_bucket_chunk['start_id'] = 0
    



    def _reset_metrics_chunk(self, metrics_type: str) -> None:
        """
        Reset metrics chunk for new data.
        
        Args:
            metrics_type: ChunkType.FIXED or ChunkType.INTERVAL
        """
        if metrics_type == ChunkType.FIXED and self.fixed_metrics_chunk is not None:
            self.fixed_metrics_chunk['count'] = 0
            self.fixed_metrics_chunk['start_id'] = 0
        elif metrics_type == ChunkType.INTERVAL and self.interval_metrics_chunk is not None:
            self.interval_metrics_chunk['count'] = 0
            self.interval_metrics_chunk['start_id'] = 0
    



    def processing_complete(self) -> Dict[Any, Any]:
        """
        Finalize historical processing.
        
        Steps:
        1. Archive any remaining data
        2. Wait for all pending archives to complete
        3. Shutdown thread pool
        4. Consolidate all chunks into large files
        
        Returns:
            Evaluation data from consolidation
        """
        logger.info("Completing historical processing...")
        
        # Archive remaining data
        # If we used _should_archive() at finalization, the final volumebars would be lost because the chunk never reached 10,000 rows.
        if self.volumebar_chunk['count'] > 0:
            self._archive_all()
        
        # Wait for all pending archives
        logger.info(f"Waiting for {len(self.pending_futures)} pending archives...")
        for future in self.pending_futures:
            try:
                future.result()
            except Exception as e:
                logger.error(f"Archive future failed: {e}")
        
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)
        logger.info("All archives complete")
        
        # Consolidate all chunks
        logger.info("Consolidating chunks...")
        result = self.archival_manager.consolidate_all()
        
        logger.info("Historical processing complete")
        return result
    



    def get_stats(self) -> Dict[str, Any]:
        """
        Get current processing statistics.
        
        Returns:
            Dictionary of statistics including volumebars, buckets, and metrics
        """
        stats = {
            'volumebars_processed': self.volumebar_count,
            'volumebars_incomplete': self.volumebar_incomplete_count,
            'volumebars_in_buffer': min(self.volumebar_count, self.config.volumebar_lookback_buffer_size),
            'volumebar_chunk_count': self.volumebar_chunk['count'],
            'chunks_archived': self.chunks_archived.copy(),
            'pending_archives': len(self.pending_futures)
        }
        
        # Add fixed bucket stats
        if self.config.enable_fixed_buckets:
            stats['fixed_buckets_processed'] = self.fixed_bucket_count
            stats['fixed_buckets_in_buffer'] = (
                min(self.fixed_bucket_count, self.config.fixed_bucket_volumebars_per_bucket)
                if self.fixed_buckets_buffer else 0
            )
            stats['fixed_bucket_chunk_count'] = (
                self.fixed_bucket_chunk['count'] if self.fixed_bucket_chunk else 0
            )
        
        # Add interval bucket stats
        if self.config.enable_interval_buckets:
            stats['interval_buckets_processed'] = self.interval_bucket_count
            stats['interval_buckets_in_buffer'] = (
                min(self.interval_bucket_count, self.config.interval_bucket_volumebars_per_bucket)
                if self.interval_buckets_buffer else 0
            )
            stats['interval_bucket_chunk_count'] = (
                self.interval_bucket_chunk['count'] if self.interval_bucket_chunk else 0
            )
        
        # Add fixed metrics stats
        if self.config.enable_fixed_buckets and not self.config.training:
            stats['fixed_metrics_enabled'] = True
            stats['fixed_metrics_processed'] = self.fixed_bucket_count  # Metrics match bucket count
            stats['fixed_metrics_in_buffer'] = (
                min(self.fixed_bucket_count, self.config.fixed_bucket_volumebars_per_bucket)
                if self.fixed_metrics_buffer else 0
            )
            stats['fixed_metrics_chunk_count'] = (
                self.fixed_metrics_chunk['count'] if self.fixed_metrics_chunk else 0
            )
        else:
            stats['fixed_metrics_enabled'] = False
        
        # Add interval metrics stats
        if self.config.enable_interval_buckets and not self.config.training:
            stats['interval_metrics_enabled'] = True
            stats['interval_metrics_processed'] = self.interval_bucket_count  # Metrics match bucket count
            stats['interval_metrics_in_buffer'] = (
                min(self.interval_bucket_count, self.config.interval_bucket_volumebars_per_bucket)
                if self.interval_metrics_buffer else 0
            )
            stats['interval_metrics_chunk_count'] = (
                self.interval_metrics_chunk['count'] if self.interval_metrics_chunk else 0
            )
        else:
            stats['interval_metrics_enabled'] = False
        
        return stats
    


