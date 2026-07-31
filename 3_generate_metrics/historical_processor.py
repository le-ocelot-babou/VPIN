"""

Historical data processor for volume bar generation from parquet files.

"""


import time
import logging

import tracemalloc
import psutil
import os

import numpy as np
import polars as pl
from scipy import stats as scipy_stats

from pathlib import Path
from typing import Optional, Dict, Any

from bucket import Bucket
from volume_bar import VolumeBar
from buffer import (
    HistoricalCalculations, 
    HistoricalConfig, 
    HelperFunctions,
    DeltasDistribution,
    NormalCDF,
    StudentTCDF,
    SkewedTCDF,
    EmpiricalCDF,
    #, VOLUMEBAR_DTYPE, BUCKET_DTYPE, EPSILON
)




logger = logging.getLogger(__name__)




class HistoricalDataProcessor:
    """

    Processes historical data from parquet files and generates volume bars.
    Supports MDP2.0 and MDP3.0 historical data formats.

    """
    
    def __init__(
        self,
        data_source_type: str,
        config: HistoricalConfig,
        historical_data_lazyframe: pl.LazyFrame,
        returns_distribution: DeltasDistribution,
        classifier_distribution: NormalCDF,
        total_rows: int,
        start_ts_ns: int,
    ):
        """

        Initialize historical data processor.
        
        Args:
            data_source_type: Type of data source (from DataSourceType)
            bar_volume_size: Volume capacity for each bar
            parquet_directory: Directory containing parquet files

        """

        
        # Buffer Initialization
        self.buffer = HistoricalCalculations(
            config=config,
            bucket_methods=Bucket,
        )


        self.data_source_type = data_source_type
        self.bar_volume_size = int(config.volumebar_contracts_per_bar)
        self.historical_data_lazyframe = historical_data_lazyframe

        self.returns_distribution = returns_distribution
        self.classifier_distribution = classifier_distribution

        self.start_ts_ns = start_ts_ns
        
        self.preprocessor = None
        self.current_bar = None

        self.total_rows: int = total_rows

        _one_percent = total_rows * 0.01
        self.progress_increment = max(5000, round(_one_percent / 5000) * 5000)
        
        self.rows_processed: int = 0
        self.bars_finalized: int = 0
        self.contract_volume_processed: int = 0

        self.vb_orders_processed: int = 0
        self.vb_contract_volume_processed: int = 0

        self.perf_start_time = None
        self.test_schema: bool = config.test_schema
        self.test_schema_show_valid_message: bool = config.test_schema_show_valid_message


        # # #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
        # # #### [ TEST ] - Test Memory Usage
        # self.memory_profiler = MemoryProfiler()
        # self.memory_profiler.start()
        # # #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####


        self._initialize_data_source(data_source_type)
    



    def _initialize_data_source(self, data_type: str) -> None:
        """
        Initialize data source and preprocessor if needed.
        
        Args:
            data_type: Type of data source to initialize
        """

        # [ NOT SUPPORTED YET ] 
        # # if DataSourceType.requires_preprocessing(data_type):
            # # self.preprocessor = TimeBarPreProcessor()
        
        self.current_bar = VolumeBar(
            bar_volume_size=self.bar_volume_size, 
            previous_volume_bar_end_ts=self.start_ts_ns,
            expected_avg_order_size=3,
            time_gap_size_hours=8,
            test_schema=self.test_schema,
            test_schema_show_valid_message=self.test_schema_show_valid_message,
        )




    def process_parquet_data(self) -> Dict[Any, Any]:

        """

        Process the data in the LazyFrame
        (Optimize for performance i.e. streaming or batches)

        """


        self.perf_start_time = time.perf_counter()

        
        try:

            # [ START ] - DATA PROCESSING VIA ITER_ROWS
            print("[ PROGRESS ]\n\n")
            
            # [ MOST EFFICIENT - STREAMING MODE ]
            # (if query supports it)
            #
            df = self.historical_data_lazyframe.collect(streaming=True)
            for row in df.iter_rows(named=True):
                
                # Attempt to process row first, so that counts are accurate during Exception handling
                self._ingest_order(row)

                self.rows_processed += 1
                self.contract_volume_processed += row["size"]

                # (Optional) Status Updates
                if(self.rows_processed % self.progress_increment == 0):
                    _progress = int((self.rows_processed / self.total_rows) * 100) if self.total_rows else 0
                    print(f"<<< {_progress}% >>>    ( {self.rows_processed:,} / {self.total_rows:,} )")


            # # [ STRUCTURED EFFICIENCY - BATCH MODE (30-50% slower) ]
            # #
            # BATCH_SIZE = 100_000  # Adjust based on row size
            # total_rows = self.historical_data_lazyframe.select(pl.count()).collect().item()

            # for offset in range(0, total_rows, BATCH_SIZE):
            #     batch = self.historical_data_lazyframe.slice(offset, BATCH_SIZE).collect()
            #     for row in batch.iter_rows(named=True):
            #         < INSERT LATEST "row processing" LOGIC HERE >
                

            # [ INEFFICIENT - LOADS ALL ROWS INTO MEMORY ]
            #
            # # Stream processing using iter_rows
            # for row in df.collect().iter_rows(named=True):
            #     < INSERT LATEST "row processing" LOGIC HERE >

            # [ END ] - DATA PROCESSING VIA ITER_ROWS
            



            # [START] - Finalize any remaining partial bar
            if self.current_bar.current_volume > 0:

                # Finalize the last remaining partial {VolumeBar}
                vb_results: dict = self.current_bar.finalize(
                    buffer=self.buffer,
                    returns_distribution=self.returns_distribution,
                    classifier_distribution=self.classifier_distribution
                )

                # [ Update Internal Counters ]
                # Count of bars actually finished, i.e. if 1 bar is complete, index must be 1
                self.bars_finalized += 1
                
                # Update Internal Counters BASED ON {VolumeBar} Data
                # (i.e. exact orders and contract volume processed)
                self.vb_orders_processed += vb_results["orders_processed"]
                self.vb_contract_volume_processed += vb_results["contract_volume_processed"]
            
            # [END] - Finalize any remaining partial bar


            # [ VALIDATION CHECK for {HistoricalDataProcessor} vs {VolumeBar} Synchronization ]
            _discrepancy, _sync_report = HelperFunctions._generate_sync_report(
                strict=True,
                completed=True,
                processor_bars_finalized=self.bars_finalized, # Deincremented inside sync report for direct comparison
                vb_id=vb_results['id'],
                contract_volume_processed=self.contract_volume_processed,
                vb_contract_volume_processed=self.vb_contract_volume_processed,
                rows_processed=self.rows_processed,
                vb_orders_processed=self.vb_orders_processed,
                vb_start_ts_ns=vb_results['start_ts_ns'],
                vb_end_ts_ns=vb_results['end_ts_ns'],
            )

            if _discrepancy:
                print(_sync_report)

            else:
                # Critical: Discrepancy detected between {HistoricalDataProcessor} and {VolumeBar}
                logger.critical(_sync_report)
            



            # Save evaluation data from buffer/archive
            logger.info("Finalizing buffer...")
            result = self.buffer.processing_complete()

            self._export_results()
            self._generate_performance_report()

            logger.info("process_parquet_data() in HistoricalDataProcessor")

            # Evaluation Data
            return result
        

        except Exception as e:

            logger.error(f"Failed processing at row {self.rows_processed + 1}: {e}")
            raise
    



    # def _process_row(self, row: Dict[str, Any]) -> None:
    #     """
    #     Process a single row from the data source.
        
    #     Args:
    #         row: Dictionary containing order data
    #     """

    #     # Route through preprocessor if needed
    #     if self.preprocessor:
    #         order = self._route_to_preprocessor(row)
    #         logger.critical("Feature Does Not Exist Yet: Routed to Pre-Processor(?)")

    #     else:
    #         order = row
        
    #     self._ingest_order(order)
    



    def _route_to_preprocessor(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send order through time-bar preprocessor.
        
        Args:
            order: Raw order dictionary
            
        Returns:
            Preprocessed order with aggressor side
        """

        # For single order processing, wrap in iterator
        processed = next(self.preprocessor.generate_time_bars(iter([order])))
        return processed
    



    def _ingest_order(self, order: Dict[str, Any]) -> None:
        """

        Ingest order into current volume bar.
        Handles splits recursively without affecting official counts.
        
        Args:
            order: Order dictionary with required fields

        """


        # # #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
        # # #### [ TEST ] - Test Memory Usage of Bar Generation
        # if self.bars_finalized < 20001:
        #    self.memory_profiler.checkpoint(f"Start - Bar #{self.bars_finalized}")
        # # #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####


        # Safely extract continuation flag
        # (Defaults to "False" for original dataset rows)
        is_continuation: bool = order.get("is_continuation", False)
        
        is_full, remaining_order = self.current_bar.add_order(
            buffer=self.buffer,
            ts=order["ts_recv"],
            price=order["price"],
            size=order["size"],
            side=order["side"],
            sequence=order["sequence"],
            instrument_id=order["instrument_id"],
            is_continuation=is_continuation,
        )


        # if is_oversized_order:
        # "pop out" of this function for recursive _ingest_order (?)
        # or just handle here? Best practice?

        
        if is_full:

            # Finalize the current Volume Bar
            # (Runs the calculations and saves to buffer)
            vb_results: dict = self.current_bar.finalize(
                buffer=self.buffer,
                returns_distribution=self.returns_distribution,
                classifier_distribution=self.classifier_distribution
            )




            # [ Update Internal Counters ]
            # Count of bars actually finished, i.e. if 1 bar is complete, index must be 1
            self.bars_finalized += 1
            
            # Update Internal Counters BASED ON {VolumeBar} Data
            # (i.e. exact orders and contract volume processed)
            self.vb_orders_processed += vb_results["orders_processed"]
            self.vb_contract_volume_processed += vb_results["contract_volume_processed"]

            # For Synchronization Check - Accounting for pending increment
            # {HistoricalDataProcessor} updates its count AFTER `_ingest_order` completes (for exception handling purposes)
            # Thus, an increment of 1 needs to be added to the current row (if it's not a continuation)
            pending_row_count = 0 if is_continuation else 1
            pending_volume = 0 if is_continuation else order["size"]


            # # [ VALIDATION CHECK for {HistoricalDataProcessor} vs {VolumeBar} Synchronization ]
            # # Information will not align until processing has completed...
            # # This is because the counts for row and order ingestion / processing are all different
            
            # expected_rows = self.rows_processed + pending_row_count
            # expected_volume = self.contract_volume_processed + pending_volume
            
            # _discrepancy, _sync_report = HelperFunctions._generate_sync_report(
            #     strict=False,
            #     completed=False,
            #     processor_bars_finalized=self.bars_finalized,  # Deincremented inside sync report for direct comparison
            #     vb_id=vb_results['id'],
            #     contract_volume_processed=expected_volume,
            #     vb_contract_volume_processed=self.vb_contract_volume_processed,
            #     rows_processed=expected_rows,
            #     vb_orders_processed=self.vb_orders_processed,
            #     vb_start_ts_ns=vb_results['start_ts_ns'],
            #     vb_end_ts_ns=vb_results['end_ts_ns'],
            # )

            # if _discrepancy:
            #     logger.info(_sync_report)

            # else:
            #     # Critical: Discrepancy detected between {HistoricalDataProcessor} and {VolumeBar}
            #     logger.critical(_sync_report)
            



            # Initizalize new Volume Bar
            # (Dereferences/garbage collects the old Volume Bar)
            self.current_bar = VolumeBar(
                bar_volume_size=self.bar_volume_size, 
                previous_volume_bar_end_ts=vb_results["end_ts_ns"],
                expected_avg_order_size=3,
                time_gap_size_hours=8,
                test_schema=self.test_schema,
                test_schema_show_valid_message=self.test_schema_show_valid_message,
            )
            
            # If order was split, submit the remainder to the new Volume Bar
            if remaining_order:

                # Recursively return to ingestion function
                # (Handles edge case of multiple-continuation orders)
                self._ingest_order(remaining_order)


        # # #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
        # # #### [ TEST ] - Test Memory Usage of Bar Generation
        # if self.bars_finalized < 20001:
        #    self.memory_profiler.checkpoint(f"End - Bar #{self.bars_finalized}")
        # # #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
        
    


    def _export_results(self) -> None:
        """Placeholder for exporting finalized bars."""
        # TODO: Implement export logic
        # Finalize processing

        logger.info("Historical processing complete!")
        pass
    



    def _generate_performance_report(self) -> Dict[str, Any]:
        """
        Generate performance report for processing run.
        
        Returns:
            Dictionary containing performance metrics
        """

        # Final buffer stats
        final_stats = self.buffer.get_stats()
        logger.info(f"Final buffer stats: {final_stats}")


        # Time-Based Performance Analysis
        elapsed_time = time.perf_counter() - self.perf_start_time

        
        # # #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
        # # #### [ TEST ] - Test Memory Usage of Bar Generation
        # # Memory-Based Performance Analysis
        # self.memory_profiler.stop()
        # # #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
        
        
        report = {
            "data_source_type": self.data_source_type,
            "rows_processed": self.rows_processed,
            "bars_finalized": self.bars_finalized,
            "elapsed_time_seconds": elapsed_time,
            "rows_per_second": self.rows_processed / elapsed_time if elapsed_time > 0 else 0,
            "bars_per_second": self.bars_finalized / elapsed_time if elapsed_time > 0 else 0
        }
        
        # Build dynamic sections based on what is enabled
        volumebar_section = f"""
        ║  Volumebars:
        ║    Processed:           {final_stats['volumebars_processed']:>28,}
        ║    Incomplete:          {final_stats['volumebars_incomplete']:>28,}
        ║    In Buffer:           {final_stats['volumebars_in_buffer']:>28,}
        ║    Chunk Count:         {final_stats['volumebar_chunk_count']:>28,}"""
        
        fixed_bucket_section = ""
        if final_stats.get('fixed_buckets_processed') is not None:
            fixed_bucket_section = f"""
        ║  Fixed Buckets:
        ║    Processed:           {final_stats['fixed_buckets_processed']:>28,}
        ║    In Buffer:           {final_stats['fixed_buckets_in_buffer']:>28,}
        ║    Chunk Count:         {final_stats['fixed_bucket_chunk_count']:>28,}"""
        
        interval_bucket_section = ""
        if final_stats.get('interval_buckets_processed') is not None:
            interval_bucket_section = f"""
        ║  Interval Buckets:
        ║    Processed:           {final_stats['interval_buckets_processed']:>28,}
        ║    In Buffer:           {final_stats['interval_buckets_in_buffer']:>28,}
        ║    Chunk Count:         {final_stats['interval_bucket_chunk_count']:>28,}"""
        
        fixed_metrics_section = ""
        if final_stats.get('fixed_metrics_enabled'):
            fixed_metrics_section = f"""
        ║  Fixed Metrics:
        ║    Processed:           {final_stats['fixed_metrics_processed']:>28,}
        ║    In Buffer:           {final_stats['fixed_metrics_in_buffer']:>28,}
        ║    Chunk Count:         {final_stats['fixed_metrics_chunk_count']:>28,}"""
        
        interval_metrics_section = ""
        if final_stats.get('interval_metrics_enabled'):
            interval_metrics_section = f"""
        ║  Interval Metrics:
        ║    Processed:           {final_stats['interval_metrics_processed']:>28,}
        ║    In Buffer:           {final_stats['interval_metrics_in_buffer']:>28,}
        ║    Chunk Count:         {final_stats['interval_metrics_chunk_count']:>28,}"""
        
        archive_section = f"""
        ║  Archives:
        ║    Chunks Archived:     {sum(final_stats['chunks_archived'].values()):>28,}
        ║    Pending Futures:     {final_stats['pending_archives']:>28,}"""
        
        print(f"""
        \n
        \n
        ╔═════════════════════════════════════════════════════════════════
        ║              PROCESSING PERFORMANCE REPORT                 
        ╠═════════════════════════════════════════════════════════════════
        ║                                                       
        ║  Data Source:           {report['data_source_type']:<28}
        ║  Rows Processed:        {report['rows_processed']:>28,}
        ║  Bars Finalized:        {report['bars_finalized']:>28,}
        ║                                                       
        ║  Elapsed Time:          {report['elapsed_time_seconds']:>23.2f} sec
        ║  Processing Rate:       {report['rows_per_second']:>21,.0f} rows/sec
        ║  Bar Generation Rate:   {report['bars_per_second']:>22,.0f} bars/sec
        ║                                                       
        ╠═════════════════════════════════════════════════════════════════
        ║              BUFFER STATISTICS                 
        ╠════════════════════════════════════════════════════════════════={volumebar_section}{fixed_bucket_section}{interval_bucket_section}{fixed_metrics_section}{interval_metrics_section}{archive_section}
        ║                                                       
        ╚═════════════════════════════════════════════════════════════════
        \n
        \n
        """)


        return report




    def _generate_sync_report(
        self,
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




# ============================================================================
# TEST PERFORMANCE OF MEMORY
# ============================================================================




class MemoryProfiler:

    def __init__(self):
        self.snapshots = []
        self.process = psutil.Process(os.getpid())
        

    def start(self):
        """Start memory tracking"""
        tracemalloc.start()
        self.baseline = self.process.memory_info().rss / 1024 / 1024  # MB
        

    def checkpoint(self, label=""):
        """Take a memory snapshot"""
        current, peak = tracemalloc.get_traced_memory()
        rss = self.process.memory_info().rss / 1024 / 1024
        self.snapshots.append({
            'label': label,
            'current_mb': current / 1024 / 1024,
            'peak_mb': peak / 1024 / 1024,
            'rss_mb': rss,
            'rss_delta_mb': rss - self.baseline
        })
        

    def stop(self):
        """Stop tracking and print summary"""
        tracemalloc.stop()
        print("\nMemory Profile:")
        print("-" * 70)
        for snap in self.snapshots:
            print(f"{snap['label']:20s} | "
                f"Current: {snap['current_mb']:8.2f} MB | "
                f"Peak: {snap['peak_mb']:8.2f} MB | "
                f"RSS Δ: {snap['rss_delta_mb']:+8.2f} MB")
