"""
Archival managers for writing structured arrays to Parquet files.

Includes:
- BaseArchivalManager: Shared archival logic with dtype preservation
- HistoricalArchivalManager: Historical-specific archival with consolidation
- LiveArchivalManager: Live-specific archival with retry logic
"""


import yaml
import json
import logging

import time
import fcntl
import threading

from pathlib import Path
from datetime import datetime, timezone

from typing import Dict, Any
import numpy as np
import pandas as pd

from .config import BaseConfig, HistoricalConfig, LiveConfig
from .models import (
    ArchivalTask,
    HelperFunctions,
    BufferName,
    ChunkType
)

logger = logging.getLogger(__name__)


class BaseArchivalManager:
    """
    Base archival manager with dtype-preserving Parquet writing.
    
    Handles:
    - Writing structured arrays to Parquet
    - Preserving exact dtypes (uint32, int64, float64, bool, etc.)
    - Registry management with file locking
    - Tiered failure handling (Parquet → JSON+NPY)
    """
    
    def __init__(self, config: BaseConfig):
        """
        Initialize archival manager.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.registry_lock = threading.Lock()
        self.registry: dict = {'volumebar': {}, 'buckets': {}, 'metrics': {}}
        self._load_registry()
    
    def _load_registry(self) -> None:
        """Load existing registry from disk or create new one."""
        registry_file = self.config.output_dir / 'chunk_registry.json'
        
        if registry_file.exists():
            try:
                with open(registry_file, 'r') as f:
                    self.registry = json.load(f)
                logger.info(f"Loaded registry from {registry_file}")
            except Exception as e:
                logger.warning(f"Could not load registry: {e}")
                self.registry = {'volumebar': {}, 'buckets': {}, 'metrics': {}}
        else:
            # Create directory and empty registry
            self.config.output_dir.mkdir(parents=True, exist_ok=True)
            with open(registry_file, 'w') as f:
                json.dump(self.registry, f, indent=2)
            logger.info(f"Created new registry at {registry_file}")
    
    def write_to_parquet(self, task: ArchivalTask) -> None:
        """
        Write archival task to Parquet file.
        
        This is a BLOCKING operation intended to run in thread pool.
        
        Args:
            task: ArchivalTask containing structured array data
        """
        try:
            df = self._structured_array_to_dataframe(task)
            output_path = self._get_output_path(task)
            
            # Ensure directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write to Parquet with PyArrow (preserves dtypes)
            df.to_parquet(
                path=str(output_path),
                engine='pyarrow',
                compression=self.config.compression,
                compression_level=self.config.compression_level,
                row_group_size=self.config.row_group_size,
                index=False
            ) # type: ignore
            
            self._update_registry(task, output_path)
            logger.info(f"Archived {task.buffer_name}/{task.chunk_type} chunk {task.chunk_id}")
            
        except Exception as e:
            logger.error(f"Archive failed: {e}")
            self._handle_failure_tiered(task, e)
    
    def _structured_array_to_dataframe(self, task: ArchivalTask) -> pd.DataFrame:
        """
        Convert structured numpy array to DataFrame preserving dtypes.
        
        Args:
            task: ArchivalTask with structured array data
        
        Returns:
            DataFrame with all fields as columns, dtypes preserved
        """
        # Pandas automatically preserves dtypes from structured arrays
        df = pd.DataFrame(task.data)
        
        # Add optional debug timestamp (nanoseconds)
        if self.config.include_debug_timestamps and task.processed_timestamp_ns is not None:
            df['processed_timestamp_ns'] = task.processed_timestamp_ns
        
        return df
    
    def _get_output_path(self, task: ArchivalTask) -> Path:
        """
        Generate output path for chunk file.
        
        Args:
            task: ArchivalTask
        
        Returns:
            Path to output Parquet file in `tmp/`
        """
        return (
            self.config.output_dir /
            "tmp" /
            task.buffer_name /
            task.chunk_type /
            f"chunk_{task.chunk_id:05d}.parquet"
        )
    
    def _update_registry(self, task: ArchivalTask, file_path: Path) -> None:
        """
        Update JSON registry with thread-safe file locking.
        
        NOTE FOR LIVE MODE:
        For >10K chunks, consider migrating to SQLite:
        
        Benefits of SQLite:
        - Concurrent writes with WAL mode
        - Efficient queries with indexes
        - Better scalability (millions of chunks)
        - ACID transactions
        
        Migration strategy:
        1. Keep JSON as primary format initially
        2. Add SQLite as cache/index layer
        3. Rebuild SQLite from JSON on startup
        4. Query SQLite, persist to JSON
        5. Eventually deprecate JSON for live mode
        
        SQLite schema example:
        CREATE TABLE chunks (
            buffer_name TEXT,
            chunk_type TEXT,
            chunk_id INTEGER,
            start_row INTEGER,
            end_row INTEGER,
            count INTEGER,
            file_path TEXT,
            processed_timestamp_ns INTEGER,
            file_size_bytes INTEGER,
            PRIMARY KEY (buffer_name, chunk_type, chunk_id)
        );
        CREATE INDEX idx_row_range
        ON chunks(buffer_name, chunk_type, start_row, end_row);
        
        For now, using JSON with file locking.
        
        Args:
            task: ArchivalTask
            file_path: Path to archived file
        """
        with self.registry_lock:
            registry_file = self.config.output_dir / 'chunk_registry.json'
            
            # File-level lock for multi-process safety
            with open(registry_file, 'r+') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                
                try:
                    registry = json.load(f)
                    
                    # Ensure structure exists
                    if task.buffer_name not in registry:
                        registry[task.buffer_name] = {}
                    if task.chunk_type not in registry[task.buffer_name]:
                        registry[task.buffer_name][task.chunk_type] = []
                    
                    # Add entry
                    entry = {
                        'chunk_id': task.chunk_id,
                        'start_row': task.metadata.get('start_row', 0),
                        'end_row': task.metadata.get('end_row', 0),
                        'count': task.metadata.get('count', len(task.data)),
                        'file': str(file_path.relative_to(self.config.output_dir)),
                        'file_size_bytes': file_path.stat().st_size
                    }
                    
                    if task.processed_timestamp_ns is not None:
                        entry['processed_timestamp_ns'] = task.processed_timestamp_ns
                    
                    registry[task.buffer_name][task.chunk_type].append(entry)
                    
                    # Write back
                    f.seek(0)
                    json.dump(registry, f, indent=2)
                    f.truncate()
                    
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    
    def _handle_failure_tiered(self, task: ArchivalTask, error: Exception) -> None:
        """
        Tiered failure handling.
        
        Tier 1: Try Parquet (most useful format)
        Tier 2: Fallback to JSON metadata + NPY data
        Tier 3: Critical error if both fail
        
        Args:
            task: Failed ArchivalTask
            error: Exception that caused failure
        """
        failed_dir = self.config.failed_dir
        failed_dir.mkdir(parents=True, exist_ok=True)
        
        base_name = (
            f"{task.buffer_name}_{task.chunk_type}_"
            f"chunk_{task.chunk_id:05d}_failed"
        )
        
        # Tier 1: Try Parquet
        try:
            parquet_path = failed_dir / f"{base_name}.parquet"
            df = self._structured_array_to_dataframe(task)
            df.to_parquet(parquet_path, engine='pyarrow')
            logger.info(f"Failed chunk saved as Parquet: {parquet_path}")
            return
        except Exception as parquet_error:
            logger.warning(f"Cannot save as Parquet: {parquet_error}")
        
        # Tier 2: JSON + NPY
        try:
            json_path = failed_dir / f"{base_name}.json"
            with open(json_path, 'w') as f:
                json.dump({
                    'metadata': task.metadata,
                    'buffer_name': task.buffer_name,
                    'chunk_type': task.chunk_type,
                    'chunk_id': task.chunk_id,
                    'processed_timestamp_ns': task.processed_timestamp_ns,
                    'original_error': str(error),
                    'dtype_descr': str(task.data.dtype.descr)
                }, f, indent=2)
            
            # Save data as NPY
            npy_path = failed_dir / f"{base_name}.npy"
            np.save(npy_path, task.data)
            
            logger.info(f"Failed chunk saved as JSON+NPY: {json_path}")
        except Exception as e:
            logger.critical(f"CRITICAL: Cannot save failed chunk: {e}")


class HistoricalArchivalManager(BaseArchivalManager):
    """
    Historical-specific archival manager.
    
    Adds consolidation functionality for merging chunks at end of processing.
    """
    
    def __init__(self, config: HistoricalConfig):
        """Initialize historical archival manager."""
        super().__init__(config)
        self.config = config  # Override with HistoricalConfig type
    
    def consolidate_all(self) -> Dict[Any, Any]:
        """

        Consolidate all chunks into large files.
        
        This is called at the end of historical processing to merge
        many small Parquet files into fewer large ones for efficient storage.

        Perform queries to test if program generated optimal output.

        """


        try:
            import polars as pl
        except ImportError:
            logger.error("Polars required for consolidation. Install: pip install polars")
            return
        

        # [ START ] - Create Export Summary
        timestamp = datetime.now()
        timestamp_folder_name = timestamp.strftime("%H%M_%b_%d_%y")
        timestamp_dir = self.config.output_dir / timestamp_folder_name
        # Create timestamp-based directory
        timestamp_dir.mkdir(parents=True, exist_ok=True)

        # Export configuration to YAML (most appropriate for config readability)
        config_dict = {
            'volumebar_contracts_per_bar': self.config.volumebar_contracts_per_bar,
            'volumebar_lookback_buffer_size': self.config.volumebar_lookback_buffer_size,
            'volumebar_archive_size': self.config.volumebar_archive_size,
            'enable_fixed_buckets': self.config.enable_fixed_buckets,
            'fixed_bucket_volumebars_per_bucket': self.config.fixed_bucket_volumebars_per_bucket,
            'fixed_bucket_archive_size': self.config.fixed_bucket_archive_size,
            'enable_interval_buckets': self.config.enable_interval_buckets,
            'interval_bucket_volumebars_per_bucket': self.config.interval_bucket_volumebars_per_bucket,
            'interval_bucket_archive_size': self.config.interval_bucket_archive_size,
            'training': self.config.training,
            'fixed_metrics_buckets_per_metric': self.config.fixed_metrics_buckets_per_metric,
            'fixed_metrics_archive_size': self.config.fixed_metrics_archive_size,
            'interval_metrics_buckets_per_metric': self.config.interval_metrics_buckets_per_metric,
            'interval_metrics_archive_size': self.config.interval_metrics_archive_size,
            'max_workers': self.config.max_workers,
            'include_debug_timestamps': self.config.include_debug_timestamps,
            'export_timestamp': timestamp.isoformat(),
            'query_start_date': {
                'epoch_ns': self.config.query_start_date,
                'human_readable': datetime.fromtimestamp(self.config.query_start_date / 1e9, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
            },
            'query_end_date': {
                'epoch_ns': self.config.query_end_date,
                'human_readable': datetime.fromtimestamp(self.config.query_end_date / 1e9, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
            },
            'query_dataset_directory': self.config.query_dataset_directory,
        }


        print("\n\n┌─ GENERATED CONSOLIDATED FILES " + "─" * 58)


        # Attempt to Export Config File
        config_file = timestamp_dir / 'config.yaml'
        try:
            with open(config_file, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

            print("│")
            print("│")
            print("| FOR: [ config_file ] ")
            print(f"| {config_file}")
            print("│")
            print("│")

        except Exception as e:
            print(f"Warning: Failed to export configuration YAML: {e}")
            # Optionally, try JSON fallback
            try:
                json_file = timestamp_dir / 'config.json'
                with open(json_file, 'w') as f:
                    json.dump(config_dict, f, indent=2)
                print(f"Configuration exported to JSON instead: {json_file}")
            except Exception as json_error:
                print(f"Error: Failed to export configuration in any format: {json_error}")
        
        # [ END ] - Create Export Summary


        # [ Define variables to save query results for subsequent tests ]
        processing_results: dict = {}

        # Define tmp directory path
        tmp_dir = self.config.output_dir / "tmp"


        # [ START ] - CONSOLIDATION OF ARCHIVES INTO A SINGULAR FILE (as well as queries for subsequent tests)
        for buffer_name in BufferName.ALL:
            for chunk_type in ChunkType.ALL:


                # Read from original location
                pattern = (
                    tmp_dir /
                    buffer_name /
                    chunk_type /
                    'chunk_*.parquet'
                )
                
                if not pattern.parent.exists():
                    continue
                
                files = list(pattern.parent.glob('chunk_*.parquet'))
                if len(files) == 0:
                    continue
                else:
                    # IF DATA EXISTS FOR THE BUFFER/CHUNK TYPE...
                    # Create buffer/chunk subdirectories within timestamp folder
                    buffer_chunk_dir = timestamp_dir / buffer_name / chunk_type
                    buffer_chunk_dir.mkdir(parents=True, exist_ok=True)
                
                logger.info(
                    f"Consolidating {len(files)} "
                    f"{buffer_name}/{chunk_type} files..."
                )
                

                try:

                    # [ START ] - Queries on dataset for subsequent Quality Assurance Tests

                    # Read all chunks for a given BUFFER/CHUNK type
                    lf = pl.scan_parquet(str(pattern))




                    # [ Different Queries Based on Buffer: VolumeBar, Bucket, or Metric ]

                    # [ VolumeBar ]
                    if buffer_name == "volumebar" and chunk_type == "raw":

                        # [ Verify All Orders Were Processed in VOLUMEBARS (Separate Query) ]
                        check_results = (
                            lf.select([
                                pl.len().alias(f"total_number_of_{buffer_name}_{chunk_type}_generated"),
                                (pl.col("bar_complete") == False).sum().alias(f"incomplete_number_of_{buffer_name}_{chunk_type}"),
                                (pl.col("order_count").sum() - pl.col("order_splits").sum()).alias("total_rows"),
                                pl.col("volume_total").sum().alias("total_size"),
                                pl.col("start_ts_ns").min().alias(f"{buffer_name}_{chunk_type}_ts_min"),
                                pl.col("end_ts_ns").max().alias(f"{buffer_name}_{chunk_type}_ts_max"),
                            ])
                            .collect()
                            .with_columns([
                                pl.from_epoch(f"{buffer_name}_{chunk_type}_ts_min", time_unit="ns").dt.strftime("%A, %B %d, %Y at %H:%M UTC (%Y-%m-%d)").alias("date_min"),
                                pl.from_epoch(f"{buffer_name}_{chunk_type}_ts_max", time_unit="ns").dt.strftime("%A, %B %d, %Y at %H:%M UTC (%Y-%m-%d)").alias("date_max"),
                            ])
                        )

                        result_dict = check_results.to_dicts()[0]
                        processing_results.update(result_dict)
                    

                    elif buffer_name == BufferName.BUCKET:

                        # if chunk_type == ChunkType.FIXED:
                        # elif chunk_type == ChunkType.INTERVAL:

                        # [ Verify Quantity of BUCKETS Generated (Separate Query) ]
                        check_results = (
                            lf.select([
                                pl.len().alias(f"total_number_of_{buffer_name}_{chunk_type}_generated"),
                                pl.col("num_bars").sum().alias("number_of_bars_used"),
                                pl.col("start_ts_ns").min().alias(f"{buffer_name}_{chunk_type}_ts_min"),
                                pl.col("end_ts_ns").max().alias(f"{buffer_name}_{chunk_type}_ts_max"),
                            ])
                            .collect()
                        )

                        result_dict = check_results.to_dicts()[0]
                        processing_results.update(result_dict)


                    elif buffer_name == BufferName.METRIC:

                        # if chunk_type == ChunkType.FIXED:
                        # elif chunk_type == ChunkType.INTERVAL:

                        # [ Verify Quantity of METRICS Generated (Separate Query) ]
                        check_results = (
                            lf.select([
                                pl.len().alias(f"total_number_of_{buffer_name}_{chunk_type}_generated"),
                                # pl.col("num_bars").sum().alias("number_of_buckets_used"),
                                # pl.col("start_ts_ns").min().alias("{buffer_name}_{chunk_type}_ts_min"),
                                # pl.col("end_ts_ns").max().alias("{buffer_name}_{chunk_type}_ts_max"),
                                # !!!
                            ])
                            .collect()
                        )

                        result_dict = check_results.to_dicts()[0]
                        processing_results.update(result_dict)


                    else:
                        raise ValueError(
                            f"Unknown buffer/chunk type, got: BUFFER={buffer_name}, CHUNK={chunk_type}"
                        )


                    # [ END ] - Queries on dataset for subsequent Quality Assurance Tests




                    # [ START ] - Save to singular file ==============================

                    # [ The Globbed DataFrame of All Recently Processed Data ]
                    df = lf.collect()

                    # Write consolidated file to new timestamp-based location
                    output_file = (
                        buffer_chunk_dir /
                        f"{buffer_name}_{chunk_type}_consolidated.parquet"
                    )

                    df.write_parquet(
                        output_file,
                        compression=self.config.compression, # type: ignore
                        row_group_size=100_000  # Larger row groups for consolidated
                    )
                    
                    # Delete original chunks
                    for f in files:
                        f.unlink()
                    

                    # Print Newly Generated File(s) to Log:
                    print("│")
                    print("│")
                    print(f"| FOR:  [ {buffer_name}_{chunk_type} ]")
                    print("│")
                    print(f"| {output_file}")
                    print("│")
                    print("│")

                    # [ END ] - Save to singular file ==============================




                    # [ START ] - Generate formatted columns for DISPLAY
                    # !!!
                    # [ END ] - Generate formatted columns for DISPLAY



                    
                except Exception as e:
                    logger.error(f"Consolidation failed for {buffer_name}/{chunk_type}: {e}", exc_info=True)
                    return {"success": False}
        

        # [ END ] - CONSOLIDATION OF ARCHIVES INTO A SINGULAR FILE (as well as queries for subsequent tests)


        # [START] - Clean up tmp directory and its subdirectories
        if tmp_dir.exists():
            try:
                import shutil
                shutil.rmtree(tmp_dir)
                logger.info(f"Successfully removed tmp directory: {tmp_dir}")
            except Exception as e:
                logger.warning(f"Failed to remove tmp directory: {e}")
        # [END] - Clean up tmp directory
        

        # Padding for the printed results
        print("\n\n\n")
        
        # Save evaluation data ("Were all orders/volumes processed based on the initial task?")
        return {"success": True, **processing_results}




class LiveArchivalManager(BaseArchivalManager):
    """
    Live-specific archival manager.
    
    Adds retry logic for handling transient failures in live mode.
    """
    
    def __init__(self, config: LiveConfig):
        """Initialize live archival manager."""
        super().__init__(config)
        self.config = config  # Override with LiveConfig type
    
    def write_to_parquet_with_retry(self, task: ArchivalTask) -> None:
        """
        Write to Parquet with retry logic for live mode.
        
        Args:
            task: ArchivalTask to write
        """
        for attempt in range(self.config.max_retries):
            try:
                self.write_to_parquet(task)
                if attempt > 0:
                    logger.info(f"Retry {attempt} succeeded")
                return
            except Exception as e:
                if attempt < self.config.max_retries - 1:
                    sleep_time = self.config.retry_backoff_base ** attempt # type: ignore
                    logger.warning(f"Retry {attempt + 1} failed: {e}. Sleeping {sleep_time}s")
                    time.sleep(sleep_time)
                else:
                    logger.error(f"All {self.config.max_retries} retries exhausted")
                    self._handle_failure_tiered(task, e)
