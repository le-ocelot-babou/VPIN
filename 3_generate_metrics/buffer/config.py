"""
Configuration classes for volumebar calculations system.

Includes:
- BaseConfig: Shared configuration parameters
- HistoricalConfig: Configuration for historical processing
- LiveConfig: Configuration for live processing
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class BaseConfig:
    """
    Shared configuration for both historical and live processing.
    """
    
    # Parquet compression settings
    compression: str = 'zstd'
    compression_level: int = 3
    row_group_size: int = 1000
    
    # Directory paths
    output_dir: Path = Path('data')
    failed_dir: Path = Path('data/failed')
    
    # Optional debug timestamps
    # TODO: For live mode, implement efficient millisecond timestamp generation
    include_debug_timestamps: bool = False
    
    # Thread pool size for disk I/O
    max_workers: int = 3

    # # (for LINTER, Live Only) Retry logic for failed archival
    # max_retries: int = 3
    # retry_backoff_base: float = 2.0
    
    def __post_init__(self):
        """Ensure directories are Path objects."""
        if not isinstance(self.output_dir, Path):
            self.output_dir = Path(self.output_dir)
        if not isinstance(self.failed_dir, Path):
            self.failed_dir = Path(self.failed_dir)




@dataclass
class HistoricalConfig(BaseConfig):
    """
    Configuration for historical processing.
    
    Historical mode uses synchronous processing with threading
    only for disk I/O operations.
    
    Supports volumebar processing with optional bucket derivation
    (fixed and/or interval) and metrics calculation.
    """
    mode: str = 'historical'
    

    # Volume Bar
    volumebar_contracts_per_bar: int = 100  # Number of contracts to accumulate per volumebar
    volumebar_lookback_buffer_size: int = 50  # Number of volumebars to keep in buffer (volumebars_per_bucket + 1 for delta calculations)
    volumebar_archive_size: int = 10_000

    # Bucket (Fixed)
    enable_fixed_buckets: bool = True
    fixed_bucket_volumebars_per_bucket: int = 1  # How many volumebars per fixed bucket
    fixed_bucket_archive_size: int = 10_000

    # Bucket (Interval)
    enable_interval_buckets: bool = False
    interval_bucket_volumebars_per_bucket: int = 1  # How many volumebars per interval bucket (rolling window)
    interval_bucket_archive_size: int = 10_000

    # Metrics (On/Off)
    training: bool = True  # If Training: No metrics are generated
    
    # Metrics (Fixed)
    enable_fixed_metrics: bool = True  # (controlled by training flag)
    fixed_metrics_buckets_per_metric: int = 50  # How many buckets per fixed metric
    fixed_metrics_archive_size: int = 500  # Number of metrics to keep in buffer (buckets_per_metric + 1 for delta calculations)

    # Metrics (Interval)
    enable_interval_metrics: bool = True  # (controlled by training flag)
    interval_metrics_buckets_per_metric: int = 50  # How many buckets per interval metric (rolling window)
    interval_metrics_archive_size: int = 500  # Number of metrics to keep in buffer (buckets_per_metric + 1 for delta calculations)
    
    # Dataset Info
    query_start_date: int = 0
    query_end_date: int = 1
    query_dataset_directory: str = ""

    # Threading settings
    max_workers: int = 2
    include_debug_timestamps: bool = False

    # Test Schema
    test_schema: bool = False
    test_schema_show_valid_message: bool = False
    

    def __post_init__(self):
        """Validate configuration."""
        super().__post_init__()

        # Validate Query Start/End Date
        assert isinstance(self.query_start_date, int) and isinstance(self.query_end_date, int) and self.query_end_date > self.query_start_date, \
            "query_start_date must be greater than query_end_date"
        
        # Volume Bar Assertions
        assert isinstance(self.volumebar_contracts_per_bar, int) and self.volumebar_contracts_per_bar > 0, \
            "volumebar_contracts_per_bar must be a positive integer"
        assert isinstance(self.volumebar_lookback_buffer_size, int) and self.volumebar_lookback_buffer_size >= 2, \
            "volumebar_lookback_buffer_size must be >= 2 (minimum for delta calculations)"
        assert isinstance(self.volumebar_archive_size, int) and self.volumebar_archive_size > 0, \
            "volumebar_archive_size must be a positive integer"
        
        # Volume Bar Archive Size Warning
        if self.volumebar_archive_size < self.volumebar_lookback_buffer_size * 10:
            import warnings
            warnings.warn(
                f"volumebar_archive_size ({self.volumebar_archive_size}) should be at least "
                f"10x volumebar_lookback_buffer_size ({self.volumebar_lookback_buffer_size}). "
                f"Current ratio: {self.volumebar_archive_size/self.volumebar_lookback_buffer_size:.1f}x. "
                "Frequent archival may impact performance."
            )

        # Bucket (Fixed) Assertions
        assert isinstance(self.enable_fixed_buckets, bool), \
            "enable_fixed_buckets must be a boolean"
        if self.enable_fixed_buckets:
            assert isinstance(self.fixed_bucket_volumebars_per_bucket, int) and self.fixed_bucket_volumebars_per_bucket > 0, \
                "fixed_bucket_volumebars_per_bucket must be a positive integer"
            assert self.fixed_bucket_volumebars_per_bucket <= self.volumebar_lookback_buffer_size, \
                f"fixed_bucket_volumebars_per_bucket ({self.fixed_bucket_volumebars_per_bucket}) cannot exceed volumebar_lookback_buffer_size ({self.volumebar_lookback_buffer_size})"
            assert isinstance(self.fixed_bucket_archive_size, int) and self.fixed_bucket_archive_size > 0, \
                "fixed_bucket_archive_size must be a positive integer"
            
            # Fixed Bucket Archive Size Warning
            if self.fixed_bucket_archive_size < self.fixed_bucket_volumebars_per_bucket * 5:
                import warnings
                warnings.warn(
                    f"fixed_bucket_archive_size ({self.fixed_bucket_archive_size}) should be at least "
                    f"5x fixed_bucket_volumebars_per_bucket ({self.fixed_bucket_volumebars_per_bucket}). "
                    f"Current ratio: {self.fixed_bucket_archive_size/self.fixed_bucket_volumebars_per_bucket:.1f}x. "
                    "Frequent archival may impact performance."
                )

        # Bucket (Interval) Assertions
        assert isinstance(self.enable_interval_buckets, bool), \
            "enable_interval_buckets must be a boolean"
        if self.enable_interval_buckets:
            assert isinstance(self.interval_bucket_volumebars_per_bucket, int) and self.interval_bucket_volumebars_per_bucket > 0, \
                "interval_bucket_volumebars_per_bucket must be a positive integer"
            assert self.interval_bucket_volumebars_per_bucket <= self.volumebar_lookback_buffer_size, \
                f"interval_bucket_volumebars_per_bucket ({self.interval_bucket_volumebars_per_bucket}) cannot exceed volumebar_lookback_buffer_size ({self.volumebar_lookback_buffer_size})"
            assert isinstance(self.interval_bucket_archive_size, int) and self.interval_bucket_archive_size > 0, \
                "interval_bucket_archive_size must be a positive integer"
            
            # Interval Bucket Archive Size Warning
            if self.interval_bucket_archive_size < self.interval_bucket_volumebars_per_bucket * 5:
                import warnings
                warnings.warn(
                    f"interval_bucket_archive_size ({self.interval_bucket_archive_size}) should be at least "
                    f"5x interval_bucket_volumebars_per_bucket ({self.interval_bucket_volumebars_per_bucket}). "
                    f"Current ratio: {self.interval_bucket_archive_size/self.interval_bucket_volumebars_per_bucket:.1f}x. "
                    "Frequent archival may impact performance."
                )

        # Both bucket types disabled warning
        if not self.enable_fixed_buckets and not self.enable_interval_buckets:
            import warnings
            warnings.warn(
                "Both fixed and interval buckets are disabled. "
                "Only volumebar processing will occur."
            )

        # Both bucket types cannot be disabled simultaneously if metrics are enabled
        if not self.training:
            assert self.enable_fixed_buckets or self.enable_interval_buckets, \
                "At least one bucket type (fixed or interval) must be enabled when training=False"

        # Metrics (Fixed) Assertions
        assert isinstance(self.enable_fixed_metrics, bool), \
            "enable_fixed_metrics must be a boolean"
        if self.enable_fixed_metrics and not self.training:
            assert self.enable_fixed_buckets, \
                "enable_fixed_buckets must be True when enable_fixed_metrics is True and training is False"
            assert isinstance(self.fixed_metrics_buckets_per_metric, int) and self.fixed_metrics_buckets_per_metric > 0, \
                "fixed_metrics_buckets_per_metric must be a positive integer"
            assert self.fixed_bucket_archive_size >= self.fixed_metrics_buckets_per_metric + 1, \
                f"fixed_bucket_archive_size ({self.fixed_bucket_archive_size}) must be >= fixed_metrics_buckets_per_metric + 1 ({self.fixed_metrics_buckets_per_metric + 1})"
            assert isinstance(self.fixed_metrics_archive_size, int) and self.fixed_metrics_archive_size > 0, \
                "fixed_metrics_archive_size must be a positive integer"

        # Metrics (Interval) Assertions
        assert isinstance(self.enable_interval_metrics, bool), \
            "enable_interval_metrics must be a boolean"
        if self.enable_interval_metrics and not self.training:
            assert self.enable_interval_buckets, \
                "enable_interval_buckets must be True when enable_interval_metrics is True and training is False"
            assert isinstance(self.interval_metrics_buckets_per_metric, int) and self.interval_metrics_buckets_per_metric > 0, \
                "interval_metrics_buckets_per_metric must be a positive integer"
            assert self.interval_bucket_archive_size >= self.interval_metrics_buckets_per_metric + 1, \
                f"interval_bucket_archive_size ({self.interval_bucket_archive_size}) must be >= interval_metrics_buckets_per_metric + 1 ({self.interval_metrics_buckets_per_metric + 1})"
            assert isinstance(self.interval_metric_archive_size, int) and self.interval_metric_archive_size > 0, \
                "interval_metric_archive_size must be a positive integer"

        # Training Mode Assertions
        assert isinstance(self.training, bool), \
            "training must be a boolean"
        
        # Validate metrics archive sizes when not in training mode
        if not self.training:
            if self.enable_fixed_buckets:
                assert self.fixed_metrics_archive_size > 0, \
                    "fixed_metrics_archive_size must be positive when training=False"
            
            if self.enable_interval_buckets:
                assert self.interval_metrics_archive_size > 0, \
                    "interval_metrics_archive_size must be positive when training=False"

        # Threading Settings Assertions
        assert isinstance(self.max_workers, int) and self.max_workers > 0, \
            "max_workers must be a positive integer"




@dataclass
class LiveConfig(BaseConfig):
    """
    Configuration for live processing.
    
    Live mode uses asyncio event loop with threading for disk I/O.
    """
    mode: str = 'live'
    
    # Backpressure settings
    queue_maxsize: int = 10
    
    # Retry logic for failed archival
    max_retries: int = 3
    retry_backoff_base: float = 2.0
    
    def __post_init__(self):
        """Validate configuration."""
        super().__post_init__()
        
        if self.archive_size < self.lookback_window * 10:
            import warnings
            warnings.warn(
                f"archive_size ({self.archive_size}) should be at least "
                f"10× lookback_window ({self.lookback_window}). "
                f"Current ratio: {self.archive_size/self.lookback_window:.1f}×"
            )
        
        if self.queue_maxsize < 3:
            import warnings
            warnings.warn(
                f"queue_maxsize ({self.queue_maxsize}) is very small. "
                "Consider at least 5-10 for smooth operation."
            )
        
        # TODO: For live mode, consider SQLite migration for >10K chunks
        # See archival_manager.py for migration strategy
        