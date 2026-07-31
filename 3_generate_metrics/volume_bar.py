
"""

VolumeBar Implementation with Optimized 2D NumPy Arrays

Volume-based bar that accumulates orders until reaching a volume threshold.
Uses optimized 2D NumPy arrays for high-performance processing.

"""


import logging
import numpy as np
import pandas as pd

from buffer import (
    HistoricalCalculations, 
    HelperFunctions, 
    VOLUMEBAR_DTYPE, 
    EPSILON,
    DeltasDistribution,
    NormalCDF,
    StudentTCDF,
    SkewedTCDF,
    EmpiricalCDF,
)

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, TYPE_CHECKING




# Configure logging
logger = logging.getLogger(__name__)








class VolumeBar:
    """
    
    Volume-based bar that accumulates orders until reaching a volume threshold.
    Uses optimized 2D NumPy arrays for high-performance processing.

    ---

    """
    

    def __init__(
        self,
        bar_volume_size: int,
        previous_volume_bar_end_ts: int,
        expected_avg_order_size: int = 3,
        time_gap_size_hours: int = 8,
        test_schema: bool = False,
        test_schema_show_valid_message: bool = False,
    ) -> None:
        
        """

        Initialize VolumeBar with capacity management.
        
        Args:
            bar_volume_size: Volume threshold to trigger finalization (e.g., 100 contracts)
            expected_avg_order_size: Expected average order size for initial capacity estimation

        """

        # Enable / Disable Schema Tests (For Each VolumeBar Finalization)
        self.test_schema: bool = test_schema
        self.test_schema_show_valid_message: bool = test_schema_show_valid_message


        # Target volume that triggers bar completion
        self.bar_volume_size: int = bar_volume_size
        
        # Initial Capacity: attempt to reduce memory usage, based on average order size with 1.5x buffer
        self._initial_capacity: int = int(bar_volume_size / (expected_avg_order_size * 1.5))
        self._capacity: int = self._initial_capacity

        # Maximum Capacity: allows for worst case: all 1-contract orders plus buffer
        self._max_capacity: int = bar_volume_size + 10
        self._has_resized: bool = False

        # Oversized Order: Was an order much larger than a VolumeBar encountered?
        # (negatively affects accuracy of PASSIVE metrics)
        self._oversized_order: bool = False
        # Defined Size of Oversized Order
        self._oversized_order_size: int = int(bar_volume_size * 2)


        ## [ CENTRAL_TIMESTAMP ]
        # Central timestamps in nanoseconds for timing reference of ALL order types
        self._central_timestamps: np.ndarray = np.zeros(self._max_capacity, dtype=np.uint64)

        # Current number of central timestamps stored
        self._central_timestamp_idx: int = 0

        # Gap Return: Was there a large time gap while processing orders that should be reviewed?
        self._gap_return: bool = False

        # Evaluation Criteria for Gap Return: Hours to nanoseconds
        # Typically 48 hours ( a weekend i.e. 24/5) but sometimes up to 72 hours (holidays, etc.)
        # Convern hours to nanoseconds for use in `add_order`
        self._time_gap_size_ns: int = time_gap_size_hours * 3_600_000_000_000

        # Save the Timestamp of the Previous Volume Bar's `end_ts_ns` for Gap Return Evaluation
        self._last_timestamp = previous_volume_bar_end_ts
        
        
        

        # #### [ START ] - Pre-allocated Arrays
        # Initialize aggressor side arrays - all pre-allocated for performance

        ## [ A_SELL_AGGRESSOR ]
        # Timestamps in nanoseconds for high-precision timing
        self._a_sell_aggressor_timestamps: np.ndarray = np.zeros(self._capacity, dtype=np.uint64)

        # Prices stored as int64 in 1e-9 units (e.g., 150.50 → 150_500_000_000)
        self._a_sell_aggressor_prices: np.ndarray = np.zeros(self._capacity, dtype=np.int64)

        # Actual volume contributed to this bar (may be less than size if split)
        self._a_sell_aggressor_size_contributed: np.ndarray = np.zeros(self._capacity, dtype=np.uint32)

        # Track which orders were split across bars
        self._ask_is_continuation: np.ndarray = np.zeros(self._capacity, dtype=np.bool_)

        # Message sequence numbers for order tracking
        self._a_sell_aggressor_sequences: np.ndarray = np.zeros(self._capacity, dtype=np.uint32)

        # Contract identifiers for roll detection
        self._a_sell_aggressor_instrument_ids: np.ndarray = np.zeros(self._capacity, dtype=np.uint32)

        # Current number of sell aggressor orders stored
        self._a_sell_aggressor_idx: int = 0
        

        ## [ B_BUY_AGGRESSOR ] - same structure as sell aggressor side
        self._b_buy_aggressor_timestamps: np.ndarray = np.zeros(self._capacity, dtype=np.uint64)
        self._b_buy_aggressor_prices: np.ndarray = np.zeros(self._capacity, dtype=np.int64)
        self._b_buy_aggressor_size_contributed: np.ndarray = np.zeros(self._capacity, dtype=np.uint32)
        self._b_buy_aggressor_is_continuation: np.ndarray = np.zeros(self._capacity, dtype=np.bool_)
        self._b_buy_aggressor_sequences: np.ndarray = np.zeros(self._capacity, dtype=np.uint32)
        self._b_buy_aggressor_instrument_ids: np.ndarray = np.zeros(self._capacity, dtype=np.uint32)

        # Current number of buy aggressor orders stored
        self._b_buy_aggressor_idx: int = 0


        ## [ N_NONE_AGGRESSOR ] - same structure as other aggressor sides
        self._n_none_aggressor_timestamps: np.ndarray = np.zeros(self._capacity, dtype=np.uint64)
        self._n_none_aggressor_prices: np.ndarray = np.zeros(self._capacity, dtype=np.int64)
        self._n_none_aggressor_size_contributed: np.ndarray = np.zeros(self._capacity, dtype=np.uint32)
        self._n_none_aggressor_is_continuation: np.ndarray = np.zeros(self._capacity, dtype=np.bool_)
        self._n_none_aggressor_sequences: np.ndarray = np.zeros(self._capacity, dtype=np.uint32)
        self._n_none_aggressor_instrument_ids: np.ndarray = np.zeros(self._capacity, dtype=np.uint32)

        # Current number of none aggressor orders stored
        self._n_none_aggressor_idx: int = 0


        # #### [ END ] - Pre-allocated Arrays

    


    @property
    def remaining_capacity(self) -> int:
        """Provide remaining capacity of VolumeBar."""

        # Calculate current accumulated volume from all sides
        # By using size_contributed arrays to get actual volume in this bar
        volume_accumulated: int = int(
            np.sum(self._a_sell_aggressor_size_contributed[:self._a_sell_aggressor_idx]) +
            np.sum(self._b_buy_aggressor_size_contributed[:self._b_buy_aggressor_idx]) +
            np.sum(self._n_none_aggressor_size_contributed[:self._n_none_aggressor_idx])
        )
        
        return self.bar_volume_size - volume_accumulated

    


    @property
    def current_volume(self) -> int:
        """Provide current volume of VolumeBar."""

        # Calculate current accumulated volume from all sides
        # By using size_contributed arrays to get actual volume in this bar
        volume_accumulated: int = int(
            np.sum(self._a_sell_aggressor_size_contributed[:self._a_sell_aggressor_idx]) +
            np.sum(self._b_buy_aggressor_size_contributed[:self._b_buy_aggressor_idx]) +
            np.sum(self._n_none_aggressor_size_contributed[:self._n_none_aggressor_idx])
        )
        
        return volume_accumulated
    
    


    @property
    def is_full(self) -> bool:
        """Check if VolumeBar has reached capacity."""

        # Calculate current accumulated volume from all sides
        # By using size_contributed arrays to get actual volume in this bar
        volume_accumulated: int = int(
            np.sum(self._a_sell_aggressor_size_contributed[:self._a_sell_aggressor_idx]) +
            np.sum(self._b_buy_aggressor_size_contributed[:self._b_buy_aggressor_idx]) +
            np.sum(self._n_none_aggressor_size_contributed[:self._n_none_aggressor_idx])
        )
        
        if volume_accumulated < self.bar_volume_size:
            # Bar not full yet, continue accumulating
            return False
        else:
            return True


    
    def add_order(
        self,
        buffer: HistoricalCalculations,
        ts: int,
        price: int,
        size: int,
        side: str,
        sequence: int,
        instrument_id: int,
        is_continuation: bool = False
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """

        Add new order to the VolumeBar
        Automatically finalizes if capacity reached.
        Handles capacity checking, splitting, array storage.
        
        Args:
            ts: Timestamp in nanoseconds (uint64)
            price: Price in 1e-9 units (int64)
            size: Order size in contracts (uint32)
            side: "B" for bid, "A" for ask
            sequence: Message sequence number (uint32)
            instrument_id: Contract identifier (uint32)
            is_continuation: True if this is a continuation of a previously split order
            
        Returns:
            Tuple of (finalized, remaining_order):
             - finalized: True if bar reached capacity and was finalized
             - remaining_order: Dict with unsplit portion if order was split, else None
        
        """


        # [ START ] - VALIDATE PARAMETER DATA BEFORE PROCEEDING

        # Validate order size - must be positive
        assert size > 0, f"Invalid order size: {size}"
        assert side in ("B", "A", "N"), f"Invalid side: {side}"
        # IF: *LIVE VERSION* (THIS SHOULD BE HANDLED AT THE DATA FEED LEVEL)

        # [ END ] - VALIDATE PARAMETER DATA BEFORE PROCEEDING




        # ============================================================
        # [ START ] - EVALUATE UNIQUE ATTRIBUTES THAT AFFECT DATA PROCESSING

        
        ## Evaluate if the current order being processed is much larger than the size of a typical VolmueBar
        # (negatively affects accuracy of PASSIVE metrics)
        if size >= self._oversized_order_size:

            
            if self.test_schema_show_valid_message:
                logger.warning(f"\n(!) Current order size ({size}) is much larger than a typical VolumeBar size ({self.bar_volume_size})\n(negatively affects accuracy of PASSIVE metrics)\n")


            self._oversized_order = True


            new_orders__partially_weighted_deltas = self._handle_oversized_order(
                buffer,
                ts,
                price,
                size,
                side,
                sequence,
                instrument_id,
                is_continuation,
            )


            ## !!!
            # place this as the FIRST function
            # reject / return here?

            # new return flag for special handling, use dict instead of tuple? no...

            # return (True, remaining_order)


            # partially_weighted_deltas = [] self.()

            # return partially_weighted_deltas
            # ------------------------------------------------------------
            #
            # [ Potential Solutions ]
            #
            # (see notes on "Handling Block Trades - Granularity Versus Atomicity" for more detail)
            #
            # Potential Flags to Use: is_multi_bar_order, no_independent_delta
            # 
            # Current Implementation: "Once and Done," delta applies only to the first VolumeBar, subsequent bars have 0 delta
            #
            # Most Promising Solution: 
            # Spread the jump across bins with weights proportional to each bin’s fraction of the chunk’s volume
            # A bin holding 40% of the oversized chunk would show 40% of the delta
            #
            # <<Reject from VolumeBar>>
            # - Identify Oversized Order
            # - Prepare proportional statistics
            # - Return to HistoricalProcessor with special REJECT flag
            # - Processor sends prepared "proportional_statistics" through loop
            # - Multiple VolumeBars are prepared, then continue as normal
            #
            # ------------------------------------------------------------




        # Evaluate if there is a Gap Return
        last_timestamp: int = 0

        # If this VolumeBar has a previous stamp, access it
        if self._central_timestamp_idx > 0:
            # Get previous central timestamp (returns 0 if no previous entry exists)
            last_timestamp = self._central_timestamps[self._central_timestamp_idx - 1]

        # Else, fallback to previous VolumeBar's `end_ts_ns`
        else:
            last_timestamp = self._last_timestamp

        ## Check if a Gap Return has elapsed since the previous order
        if (ts - last_timestamp) > self._time_gap_size_ns:
            self._gap_return = True
        



        # [ END ] - EVALUATE UNIQUE ATTRIBUTES THAT AFFECT DATA PROCESSING
        # ============================================================

        
        

        # ============================================================
        # [ START ] - CALCULATE REMAINING CAPACITY / IF ORDER NEEDS TO BE SPLIT
        _remaining_capacity: int = self.remaining_capacity

        # Determine if order needs to be split across bars
        size_to_add: int
        remaining_size: int
        mark_as_continuation: bool

        # Order fits entirely in current bar
        if size <= _remaining_capacity:
            size_to_add = size
            remaining_size = 0

            # If this was already a continuation, it stays a continuation
            # If this was an original (un-split) order and fits entirely, it stays original
            mark_as_continuation = is_continuation

        # Order exceeds remaining capacity - must split
        else:

            # !!!

            size_to_add = _remaining_capacity
            remaining_size = size - _remaining_capacity

            # Any remainder created by splitting an order is automatically a continuation
            # BUT original/parent one IS NOT a continuation
            mark_as_continuation = is_continuation
        

        # [ END ] - CALCULATE REMAINING CAPACITY
        # ============================================================




        # ============================================================
        # [ START ] - STORE ORDER DATA IN RESPECTIVE ARRAY (BID/ASK/NONE)


        # [ BID ]
        if side == "B":

            # (For memory optimization)
            # Check if we need to resize arrays (one-time operation)
            if self._b_buy_aggressor_idx >= self._capacity:
                self._resize_to_max()
            

            # Store all bid order attributes at current index
            self._b_buy_aggressor_timestamps[self._b_buy_aggressor_idx] = ts
            self._b_buy_aggressor_prices[self._b_buy_aggressor_idx] = price
            self._b_buy_aggressor_size_contributed[self._b_buy_aggressor_idx] = size_to_add  # Actual volume added/contributed to this VolumeBar
            self._b_buy_aggressor_is_continuation[self._b_buy_aggressor_idx] = mark_as_continuation
            self._b_buy_aggressor_sequences[self._b_buy_aggressor_idx] = sequence
            self._b_buy_aggressor_instrument_ids[self._b_buy_aggressor_idx] = instrument_id
            self._b_buy_aggressor_idx += 1
        

        # [ ASK ]
        elif side == "A":

            # (For memory optimization)
            # Check if we need to resize arrays (one-time operation)
            if self._a_sell_aggressor_idx >= self._capacity:
                self._resize_to_max()
            

            # Store all ask order attributes at current index
            self._a_sell_aggressor_timestamps[self._a_sell_aggressor_idx] = ts
            self._a_sell_aggressor_prices[self._a_sell_aggressor_idx] = price
            self._a_sell_aggressor_size_contributed[self._a_sell_aggressor_idx] = size_to_add  # Actual volume added/contributed to this VolumeBar
            self._ask_is_continuation[self._a_sell_aggressor_idx] = mark_as_continuation
            self._a_sell_aggressor_sequences[self._a_sell_aggressor_idx] = sequence
            self._a_sell_aggressor_instrument_ids[self._a_sell_aggressor_idx] = instrument_id
            self._a_sell_aggressor_idx += 1
        

        # [ N_NONE_AGGRESSOR ]
        elif side == "N":

            # (For memory optimization)
            # Check if we need to resize arrays (one-time operation)
            if self._n_none_aggressor_idx >= self._capacity:
                self._resize_to_max()
            

            # Store all none aggressor order attributes at current index
            self._n_none_aggressor_timestamps[self._n_none_aggressor_idx] = ts
            self._n_none_aggressor_prices[self._n_none_aggressor_idx] = price
            self._n_none_aggressor_size_contributed[self._n_none_aggressor_idx] = size_to_add  # Actual volume added/contributed to this VolumeBar
            self._n_none_aggressor_is_continuation[self._n_none_aggressor_idx] = mark_as_continuation
            self._n_none_aggressor_sequences[self._n_none_aggressor_idx] = sequence
            self._n_none_aggressor_instrument_ids[self._n_none_aggressor_idx] = instrument_id
            self._n_none_aggressor_idx += 1
        

        else:
            logger.critical(f"Unknown Order Side: '{side}' - (should have been caught by Polars query and `assert` statements!)")
            assert False, f"Invalid side (2): {side}"




        # Store current order's timestamp in CENTRAL_TIMESTAMP at current index (aggressor agnostic)
        self._central_timestamps[self._central_timestamp_idx] = ts
        self._central_timestamp_idx += 1
        

        # [ END ] - STORE ORDER DATA IN RESPECTIVE ARRAY (BID/ASK)
        # ============================================================



        

        # ============================================================
        # [ START ] - ORDER SPLITTING LOGIC & RETURN

        # Bar is full, prepare remaining order dictionary if split occurred
        if self.is_full:

            # !!!
            
            remaining_order: Optional[Dict[str, Any]] = None

            # If part of the order remains, prepare the information for the next {VolumeBar}
            if remaining_size > 0:
                # Prepare data in a format that can be accepted by {HistoricalDataProcessor}
                remaining_order = {
                    "ts_recv": ts,
                    "price": price,
                    "size": remaining_size,
                    "side": side,
                    "sequence": sequence,
                    "instrument_id": instrument_id,
                    "is_continuation": True  # Remainder of an order is always a continuation
                }
            

            # Signal that bar is ready for finalization
            # Main processor must then call finalize() and create new bar
            # New bar will receive the remaining portion of the order
            return (True, remaining_order)
        

        # Bar not full yet, continue accumulating
        else:
            return (False, None)
        

        # [ END ] - ORDER SPLITTING LOGIC & RETURN
        # ============================================================
    



    def finalize(self, buffer: HistoricalCalculations, returns_distribution: DeltasDistribution, classifier_distribution: NormalCDF) -> Dict[str, int]:
        """
        Calculate all statistics and send to buffer.
        Called by main processor when add_order() signals bar completion.
        
        Args:
            buffer: HistoricalCalculations instance with required interface methods:
                - get_previous_volumebar_row_view() -> Optional[np.ndarray]
                - save_latest_volume_bar(stats: Dict[str, Any]) -> None
                
        Process:
            1. Calculate  statistics from arrays (no cached values)
            2. Retrieve previous bar statistics from buffer
            3. Calculate delta statistics if previous bar exists
            4. Send complete statistics dictionary to buffer
        """



        # Step 1: Obtain previous VolumeBar's statistics from buffer
        # This returns a NumPy structured array view for fast field access
        previous_row: Optional[np.ndarray] = buffer.get_previous_volumebar_row_view()


        # Step 2: Calculate all statistics from current VolumeBar"s arrays
        # Does not use cached values to ensure data integrity
        current_row: Dict[str, Any] = self._calculate_statistics(
            previous=previous_row,
            returns_distribution=returns_distribution,
            classifier_distribution=classifier_distribution,
        )




        # #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
        # #### [ TEST ] - Test Statistics Schema (current_row, ONLY statistics)
        if self.test_schema:
            self.test__row_statistics_schema(current_row, empty=False, show_valid_messages=self.test_schema_show_valid_message)
        # #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####

        # #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
        # #### [ TEST ] - Test Empty Statistics Schema (current_row, ONLY empty statistics i.e. ?)
        if self.test_schema:
            self.test__row_statistics_schema(self._get_empty_statistics(), empty=True, show_valid_messages=self.test_schema_show_valid_message)
        # #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####




        # Step 3: Calculate deltas only if we have a previous bar
        # (First bar will not have any deltas, nothing to compare against)

        # This is the first VolumeBar / Row in the Buffer
        if previous_row is None:

            # Obtain a zero-filled (blank) dict with required column names
            deltas: Dict[str, Any] = self._get_empty_deltas()

            # Merge deltas into the statistics dictionary
            current_row.update(deltas)


            # #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
            # #### [ TEST ] - Test Empty Deltas Schema (current_row, ONLY EMPTY deltas i.e. row 0)
            if self.test_schema:
                self.test__row_deltas_schema(self._get_empty_deltas(), empty=True, show_valid_messages=self.test_schema_show_valid_message)
            # #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####

            
        # Otherwise, we have data to compare against
        else:

            # Compare current to previous and calculate all deltas
            deltas: Dict[str, Any] = self._calculate_deltas(current_row, previous_row)

            # Merge deltas into the statistics dictionary
            current_row.update(deltas)


            # #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
            # #### [ TEST ] - Test Deltas Schema (current_row, ONLY deltas)
            if self.test_schema:
                self.test__row_deltas_schema(deltas, empty=False, show_valid_messages=self.test_schema_show_valid_message)
            # #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####




        # Step 4: Send complete statistics to buffer for storage
        # Buffer handles appending to its internal structured array
        buffer.save_latest_volume_bar(current_row)


        ## For Debugging Purposes:
        # print(current_row)

        # #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
        # #### [ TEST ] - Test Full Schema (current_row, statistics AND deltas)
        if self.test_schema:
            self.test__row_full_schema(current_row, empty=False, show_valid_messages=self.test_schema_show_valid_message)
        # #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####




        # [ Return (Order/Volume Processed Counts) to the {HistoricalDataProcessor} to monitor for potential data leaks ]
        #
        # i.e. Ensure the {VolumeBar} is not losing datapoints along the way
        # For simplicity and lower latency, (Order/Volume Processed Counts) data is provided upon every {VolumeBar} finalization

        _orders_processed: int = current_row["order_count"] - current_row["order_splits"]
        _contract_volume_processed: int = current_row["volume_total"]

        return {

            "id": current_row["id"],

            "orders_processed": _orders_processed,
            "contract_volume_processed": _contract_volume_processed,

            "start_ts_ns": current_row["start_ts_ns"],
            "end_ts_ns": current_row["end_ts_ns"],

        }


        # # [ Assist GarbageCollector by De-Referencing Class Attributes? ]
        # ## [ B-Buy ]
        # self._b_buy_aggressor_timestamps: np.ndarray = np.zeros(self._capacity, dtype=np.uint64)
        # self._b_buy_aggressor_prices: np.ndarray = np.zeros(self._capacity, dtype=np.int64)
        # self._b_buy_aggressor_size_contributed: np.ndarray = np.zeros(self._capacity, dtype=np.uint32)
        # self._b_buy_aggressor_is_continuation: np.ndarray = np.zeros(self._capacity, dtype=np.bool_)
        # self._b_buy_aggressor_sequences: np.ndarray = np.zeros(self._capacity, dtype=np.uint32)
        # self._b_buy_aggressor_instrument_ids: np.ndarray = np.zeros(self._capacity, dtype=np.uint32)
        
        # ## [ A-Sell ] - same structure as B side
        # ## [ N-NONE ] - same structure as B side




    def _handle_oversized_order(
        self,
        buffer: HistoricalCalculations,
        ts: int,
        price: int,
        size: int,
        side: str,
        sequence: int,
        instrument_id: int,
        is_continuation: bool = False
    ) -> Tuple: #Tuple[bool, Optional[Dict[str, Any]]]:

        # !!!


        # ============================================================
        # [ START ] - GENERATE ORDER SPLITS FROM OVERSIZED

        # Define Iterative Variables
        order_split_sizes: list = []
        order_adjusted_prices: list = []

        price_of_oversized_order: int = price
        remaining_size_of_oversized_order: int = size
        

        # Calculate (REMAINING CAPACITY / SIZE OF VOLUME BAR) for splits
        remaining_volume_bar_capacity: int = self.remaining_capacity
        current_volume_bar_capacity: int = self.current_volume

        initial_order_size: int = remaining_volume_bar_capacity
        capacity_of_volume_bar: int = self.bar_volume_size


        # Append (initial order value) -> to fill the remaining capacity of the current VolumeBar
        order_split_sizes.append(initial_order_size)
        remaining_size_of_oversized_order -= initial_order_size

        # Calculate the number of FULL_BARS AND the REMAINDER value
        number_of_full_bars = remaining_size_of_oversized_order // capacity_of_volume_bar
        remainder = remaining_size_of_oversized_order % capacity_of_volume_bar

        # Append values of FULL_BARS to the array
        order_split_sizes.extend([capacity_of_volume_bar] * number_of_full_bars)

        # Add REMAINDER if it exists
        if remainder > 0:
            order_split_sizes.append(remainder)
        

        # [ END ] - GENERATE ORDER SPLITS FROM OVERSIZED
        # ============================================================




        # (TEMPORARY / TESTING)
        print("\n\n[ ORDER SPLIT SIZES ]\n\n", order_split_sizes)




        # ============================================================
        # [ START ] - MEASURE PRICE DELTA


        # Step 1: Obtain previous VolumeBar's statistics from buffer
        # This returns a NumPy structured array view for fast field access
        previous_row: Optional[np.ndarray] = buffer.get_previous_volumebar_row_view()


        # Reference Price to be used in future calculations
        reference_price_from_last_volume_bar: int = 0
        # Chosen Price Type
        reference_price_field: str = "passive_midprice"


        # [ GET PREVIOUS VOLUME BAR DATA]

        # ( This is the first VolumeBar / Row in the Buffer)
        if previous_row is None:

            # (Use "first row" treatment here -> `np.nan`)
            # No prior values to compare against
            #
            # Unable to compute a price delta, thus "proportionally weighted price deltas" are not possible
            # Do not set last_price to 0, that will display a sudden drop(spike) that DID NOT occur

            reference_price_from_last_volume_bar = np.float64(np.nan)

        # (NOT the first VolumeBar / Row, we have data to compare against)
        else:

            # Reference Price from Previous VolumeBar
            reference_price_from_last_volume_bar = np.float64(previous_row[reference_price_field])

        
        # [ END ] - GET LAST PRICE DELTA
        # ============================================================




        # ============================================================
        # [ START ] - CALCULATE PRICE DELTAS - 3 VERSIONS


        # [ Is there a reference price available from the previous VolumeBar? ]
        # (If so, "proportionally weighted deltas" are possible)


        # 3x (PRICES) and (PRICE_DELTAS) to measure
        # - Last
        # - Current
        # - Interpolated

        # Prices
        price_last_volume_bar = np.float64(np.nan)
        price_current_volume_bar = np.float64(np.nan)
        price_interpolated_volume_bar = np.float64(np.nan)
        # Deltas
        delta_price_last_volume_bar = np.float64(np.nan)
        delta_price_current_volume_bar = np.float64(np.nan)
        delta_price_interpolated_volume_bar = np.float64(np.nan)


        # [ GENERATE "proportionally weighted prices" ]
        # ( Reference price available! )
        if not np.isnan(reference_price_from_last_volume_bar):


            # --------------------------------------------------
            # [ START ] - (LAST VOLUME_BAR PRICE)

            price_last_volume_bar = np.float64(reference_price_from_last_volume_bar)

            # [ END ] - (LAST VOLUME_BAR PRICE)
            # --------------------------------------------------


            # [ Are there sufficient contracts in the current VolumeBar to generate a price? ]
            # (i.e. is it over the Minimum Capacity Threshold? Else, just noise. )

            # Define VolumeBar's (Minimum Capacity Threshold) to generate a STABLE current/interpolated price
            minimum_capacity_threshold: float = 0.50
            current_capacity_ratio: float = current_volume_bar_capacity / capacity_of_volume_bar

            # Count the number of orders in current VolumeBar (used for `_calculate_passive_midprice` method call)
            total_orders = (
                self._a_sell_aggressor_idx + 
                self._b_buy_aggressor_idx + 
                self._n_none_aggressor_idx
            )

            if (total_orders > 0) and (current_capacity_ratio > minimum_capacity_threshold):

                # --------------------------------------------------
                # [ START ] - (CURRENT VOLUME_BAR PRICE)

                # ========================================================================
                # Passive Midprice (VWAP of ALL trades: A + B + N)
                # ========================================================================

                # Innerworkings of `_calculate_passive_midprice`:
                # From each "Trade Side" NumPy Array
                # (i.e. "_b_buy_aggressor...", "_a_sell_aggressor...", "_n_none_aggressor...")
                #
                # Aggregates into 1 list:
                # - Prices Contributed (all_prices)
                # - Size Contributed (all_volumes)
                #
                # Performs element-wise multiplication for exact VWAP calculation
                # Returns `passive_midprice` if (total_orders > 0 and total_volume > 0)
                # Else returns `np.nan`

                price_current_volume_bar = np.float64(self._calculate_passive_midprice(total_orders=total_orders))

                # [ END ] - (CURRENT VOLUME_BAR PRICE)
                # --------------------------------------------------


                # --------------------------------------------------
                # [ START ] - (INTERPOLATED VOLUME_BAR PRICE)

                # Weighting: (Current price by the CURRENT capacity ratio) AND (the last price by the REMAINDER)
                price_interpolated_volume_bar = np.float64(np.average(

                    [price_current_volume_bar, price_last_volume_bar],
                    weights=[current_capacity_ratio, 1.0]

                ))

                # [ END ] - (INTERPOLATED VOLUME_BAR PRICE)
                # --------------------------------------------------
            
            # [ INSUFFICIENT data in current VolumeBar to generate "proportionally weighted prices" ]
            else:

                # Simply reference the last VolumeBar's if there is insufficient data
                price_current_volume_bar = np.float64(price_last_volume_bar)
                price_interpolated_volume_bar = np.float64(price_last_volume_bar)




            # --------------------------------------------------
            # [ START ] - (CALCULATE PRICE DELTAS)

            # Last
            delta_price_last_volume_bar = np.float64(price_of_oversized_order - price_last_volume_bar)

            # Current
            delta_price_current_volume_bar = np.float64(price_of_oversized_order - price_current_volume_bar)

            # Interpolated
            delta_price_interpolated_volume_bar = np.float64(price_of_oversized_order - price_interpolated_volume_bar)

            # [ END ] - (CALCULATE PRICE DELTAS)
            # --------------------------------------------------


        # [ END ] - CALCULATE PRICE DELTAS - 3 VERSIONS
        # ============================================================




        # ============================================================
        # [ START ] - ASSIGN (PROPORTIONALLY WEIGHTED PRICE DELTAS)


        # order_split_sizes: list = []
        # order_adjusted_prices: list = [] -> should these be numpy arrays?

        if not np.isnan(delta_price_last_volume_bar):
            pass
        else:
            # nothing to reference, first row
            # Handle NaN


        # # Last
        # delta_price_last_volume_bar = np.float64
        # # Current
        # delta_price_current_volume_bar
        # # Interpolated
        # delta_price_interpolated_volume_bar

        # For NumPy Array
        value: float = 1.0 -> SHOULD WE USE NumPy arrays INSTEAD of python lists? fix git. store exports in 0_data?
        sizes = np.array([1, 2, 2, 1])
        https://claude.ai/chat/708d4ee4-d6cb-4f00-8d91-b144a6532bae

        # Calculate proportional weights
        total_size = np.sum(sizes)
        proportions = sizes / total_size

        # Assign value proportionally
        allocated_values = value * proportions

        # Result: [0.16666667, 0.33333333, 0.33333333, 0.16666667]

        # For Python Lists
        value: float = 1.0
        sizes = [1, 2, 2, 1]

        total_size = sum(sizes)
        allocated_values = [value * (size / total_size) for size in sizes]


        # [ END ] - ASSIGN (PROPORTIONALLY WEIGHTED PRICE DELTAS)
        # ============================================================




        # ============================================================
        # [ START ] - PACKAGE UPDATED ORDERS

        # order_split_sizes: list = []
        # order_adjusted_prices: list = []
        
        # create CUSTOM flag/CUSTOM price metric
        # add to TRADITIONAL order values

        # remaining_order = {
        #     "ts_recv": ts,
        #     "price": price,
        #     "size": remaining_size,
        #     "side": side,
        #     "sequence": sequence,
        #     "instrument_id": instrument_id,
        #     "is_continuation": True  # Remainder of an order is always a continuation
        # }
        

        # [ END ] - PACKAGE UPDATED ORDERS
        # ============================================================




        # [ RETURN array (or tuple?) of orders here ]
        return ()




    def _resize_to_max(self) -> None:
        """

        One-time resize of all arrays to maximum capacity.
        Called when initial optimistic capacity is exceeded.

        """

        if self._has_resized:
            logger.error("Attempted second resize - bar overfilled beyond max capacity")
            return
        
        logger.info(f"Resizing arrays from {self._capacity} to {self._max_capacity}")
        self._has_resized = True
        
        new_capacity = self._max_capacity
        
        # Resize A-Sell aggressor arrays
        self._a_sell_aggressor_timestamps = np.resize(self._a_sell_aggressor_timestamps, new_capacity)
        self._a_sell_aggressor_prices = np.resize(self._a_sell_aggressor_prices, new_capacity)
        self._a_sell_aggressor_size_contributed = np.resize(self._a_sell_aggressor_size_contributed, new_capacity)
        self._ask_is_continuation = np.resize(self._ask_is_continuation, new_capacity)
        self._a_sell_aggressor_sequences = np.resize(self._a_sell_aggressor_sequences, new_capacity)
        self._a_sell_aggressor_instrument_ids = np.resize(self._a_sell_aggressor_instrument_ids, new_capacity)
        
        # Resize B-Buy aggressor arrays
        self._b_buy_aggressor_timestamps = np.resize(self._b_buy_aggressor_timestamps, new_capacity)
        self._b_buy_aggressor_prices = np.resize(self._b_buy_aggressor_prices, new_capacity)
        self._b_buy_aggressor_size_contributed = np.resize(self._b_buy_aggressor_size_contributed, new_capacity)
        self._b_buy_aggressor_is_continuation = np.resize(self._b_buy_aggressor_is_continuation, new_capacity)
        self._b_buy_aggressor_sequences = np.resize(self._b_buy_aggressor_sequences, new_capacity)
        self._b_buy_aggressor_instrument_ids = np.resize(self._b_buy_aggressor_instrument_ids, new_capacity)
        
        # Resize N-None aggressor arrays
        self._n_none_aggressor_timestamps = np.resize(self._n_none_aggressor_timestamps, new_capacity)
        self._n_none_aggressor_prices = np.resize(self._n_none_aggressor_prices, new_capacity)
        self._n_none_aggressor_size_contributed = np.resize(self._n_none_aggressor_size_contributed, new_capacity)
        self._n_none_aggressor_is_continuation = np.resize(self._n_none_aggressor_is_continuation, new_capacity)
        self._n_none_aggressor_sequences = np.resize(self._n_none_aggressor_sequences, new_capacity)
        self._n_none_aggressor_instrument_ids = np.resize(self._n_none_aggressor_instrument_ids, new_capacity)
        
        self._capacity = new_capacity
    
    


    def _calculate_passive_midprice(
        self,
        total_orders: int,
    ) -> Dict[str, Any]:
    

        # Used in the following sections:
        # ========================================================================
        # Passive Midprice (VWAP of ALL trades: A + B + N)
        # ========================================================================

        # Innerworkings of `_calculate_passive_midprice`:
        # From each "Trade Side" NumPy Array
        # (i.e. "_b_buy_aggressor...", "_a_sell_aggressor...", "_n_none_aggressor...")
        #
        # Aggregates into 1 list:
        # - Prices Contributed (all_prices)
        # - Size Contributed (all_volumes)
        #
        # Performs element-wise multiplication for exact VWAP calculation
        # Returns `passive_midprice` if (total_orders > 0 and total_volume > 0)
        # Else returns `np.nan`
        
        if total_orders == 0:
            passive_midprice = np.float64(np.nan)
        else:
            # Collect all prices and volumes from all three sides
            all_prices = []
            all_volumes = []
            
            if self._b_buy_aggressor_idx > 0:
                all_prices.append(self._b_buy_aggressor_prices[:self._b_buy_aggressor_idx].astype(np.float64))
                all_volumes.append(self._b_buy_aggressor_size_contributed[:self._b_buy_aggressor_idx].astype(np.float64))
            
            if self._a_sell_aggressor_idx > 0:
                all_prices.append(self._a_sell_aggressor_prices[:self._a_sell_aggressor_idx].astype(np.float64))
                all_volumes.append(self._a_sell_aggressor_size_contributed[:self._a_sell_aggressor_idx].astype(np.float64))
            
            if self._n_none_aggressor_idx > 0:
                all_prices.append(self._n_none_aggressor_prices[:self._n_none_aggressor_idx].astype(np.float64))
                all_volumes.append(self._n_none_aggressor_size_contributed[:self._n_none_aggressor_idx].astype(np.float64))
            
            # Concatenate all prices and volumes
            combined_prices = np.concatenate(all_prices)
            combined_volumes = np.concatenate(all_volumes)
            
            # Element-wise multiplication for exact VWAP calculation
            total_weighted_value = np.sum(combined_prices * combined_volumes, dtype=np.float64)
            total_volume = np.sum(combined_volumes, dtype=np.float64)
            
            if total_volume > 0:
                passive_midprice = np.float64(total_weighted_value / total_volume)
            else:
                passive_midprice = np.float64(np.nan)


        return passive_midprice




    def _calculate_statistics(
        self,
        previous: Optional[np.ndarray],
        returns_distribution: DeltasDistribution,
        classifier_distribution: NormalCDF,
    ) -> Dict[str, Any]:
        """
        Calculate all summary statistics from arrays.
        Always recalculates from arrays, never uses cached values.
        Fully optimized for NumPy performance - all operations stay in NumPy domain.
        
        Args:
            previous: Previous bar's statistics as structured NumPy array (None for first bar)
            returns_distribution: Distribution lookup for std() used in normalization
            classifier_distribution: Distribution lookup for cdf() used in passive classification
        
        Returns:
            Dictionary with all raw statistics, properly typed with NumPy dtypes
        
        <><><> MUST MATCH "models.VOLUMEBAR_DTYPE" SCHEMA <><><>
        
        Key Optimizations:
        - Price fields stored as float64 to preserve sub-tick precision
        - All calculations use pure NumPy operations (no Python float() conversions)
        - Type conversions minimized and done efficiently
        - Division operations use NumPy scalars, not Python floats
        """

        """
        =============================================================================
        [ Trade Side - Explanation ]
        =============================================================================

        | Trade Side ["side"] =     |
        | "A" (Aggressor = Seller)  | 
        | "B" (Aggressor = Buyer)   | 
        | "N" (No side / undefined) |

        -----------------------------------------------------------------------------

        [ "A" = ASKING/SELLING ] - Aggressor Hit the Bid

        Aggressor = Seller → Hit the Bid → Selling (Short) → Negative Imbalance
        The seller crossed the spread to execute immediately at the bid.

        BID = SELLING = HITTING THE BID → Crossing the Spread Downward

        -----------------------------------------------------------------------------

        [ "B" = BIDDING/BUYING ] - Aggressor Lifted the Offer

        Aggressor = Buyer → Lift the Ask → Buying (Long) → Positive Imbalance
        The buyer crossed the spread to execute immediately at the ask.

        ASK = BUYING = PAYING THE ASK → Crossing the Spread Upward

        -----------------------------------------------------------------------------

        [ "N" = None ]

        No side specified (e.g., spread trades, auctions, crosses, or unclassified events).

        =============================================================================
        
        Aggressor Flag Logic (Aggressor Side FIX tag 5796)
        Taker: Who removed the liquidity? Crossed the spread?
        Prefixes/descriptors:
            active_buy_*  → aggressor lifted the offer (buyer crossed spread)
            active_sell_* → aggressor hit the bid (seller crossed spread)
            active_none_* → aggressor side unknown (N-side)
        
        =============================================================================
        """

        current: Dict[str, Any] = {}
        
        # ========================================================================
        # Configuration of the bar
        # ========================================================================


        # [ (!) CURRENT ROW "id" CALCULATION ]
        if previous is None:
            # This is the first row
            current["id"] = np.uint32(0)
        else:
            current["id"] = np.uint32(previous["id"] + 1)


        current["bar_volume_size"] = np.uint32(self.bar_volume_size)
        
        # ========================================================================
        # Metadata Fields
        # ========================================================================

        current["gap_return"] = np.bool_(self._gap_return)
        
        # Calculate max time gap only if gap_return is True
        if self._gap_return and self._central_timestamp_idx > 1:
            timestamp_diffs = np.diff(self._central_timestamps[:self._central_timestamp_idx])
            current["max_time_gap_ns"] = np.uint64(np.max(timestamp_diffs))
        else:
            current["max_time_gap_ns"] = np.uint64(0)
        
        current["contains_oversized_order"] = np.bool_(self._oversized_order)
        current["has_resized"] = np.bool_(self._has_resized)

        # ========================================================================
        # Order Counts (Ambiguous/Total)
        # ========================================================================

        current["order_count"] = np.uint32(
            self._a_sell_aggressor_idx + 
            self._b_buy_aggressor_idx + 
            self._n_none_aggressor_idx
        )
        
        # Count split orders (orders that spanned multiple bars)
        buy_splits = np.sum(
            self._b_buy_aggressor_is_continuation[:self._b_buy_aggressor_idx], 
            dtype=np.uint32
        ) if self._b_buy_aggressor_idx > 0 else np.uint32(0)
        
        sell_splits = np.sum(
            self._ask_is_continuation[:self._a_sell_aggressor_idx], 
            dtype=np.uint32
        ) if self._a_sell_aggressor_idx > 0 else np.uint32(0)
        
        none_splits = np.sum(
            self._n_none_aggressor_is_continuation[:self._n_none_aggressor_idx], 
            dtype=np.uint32
        ) if self._n_none_aggressor_idx > 0 else np.uint32(0)
        
        current["order_splits"] = np.uint32(buy_splits + sell_splits + none_splits)

        # ========================================================================
        # Active Order Counts
        # ========================================================================

        current["active_order_count_buy"] = np.uint32(self._b_buy_aggressor_idx)
        current["active_order_count_sell"] = np.uint32(self._a_sell_aggressor_idx)
        current["active_order_count_none"] = np.uint32(self._n_none_aggressor_idx)
        
        # ========================================================================
        # Active Volume Calculations
        # ========================================================================

        # Calculate actual volume contributed by each side
        active_volume_buy = np.sum(
            self._b_buy_aggressor_size_contributed[:self._b_buy_aggressor_idx], 
            dtype=np.uint32
        ) if self._b_buy_aggressor_idx > 0 else np.uint32(0)
        
        active_volume_sell = np.sum(
            self._a_sell_aggressor_size_contributed[:self._a_sell_aggressor_idx], 
            dtype=np.uint32
        ) if self._a_sell_aggressor_idx > 0 else np.uint32(0)
        
        active_volume_none = np.sum(
            self._n_none_aggressor_size_contributed[:self._n_none_aggressor_idx], 
            dtype=np.uint32
        ) if self._n_none_aggressor_idx > 0 else np.uint32(0)
        
        volume_total = np.uint32(active_volume_buy + active_volume_sell + active_volume_none)

        current["active_volume_buy"] = active_volume_buy
        current["active_volume_sell"] = active_volume_sell
        current["active_volume_none"] = active_volume_none
        current["volume_total"] = volume_total
        
        # Defensive check
        if volume_total > self.bar_volume_size:
            logger.warning(f"Bar overfilled: {volume_total} > {self.bar_volume_size}")

        # Evaluate if VolumeBar is "properly filled" or incomplete
        current["bar_complete"] = np.bool_(volume_total == self.bar_volume_size)
        
        # ========================================================================
        # Active Imbalance Metrics
        # ========================================================================

        # Convert to signed integers for imbalance calculations
        active_volume_buy_signed = np.int64(active_volume_buy)
        active_volume_sell_signed = np.int64(active_volume_sell)

        # Calculate buy/sell imbalance in various forms
        # If there is more buying volume ("BUYING/BIDDING", side: "B"), the Imbalance will be a positive number
        # Vice versa for more selling volume ("ASKING/SELLING", side: "A"), the Imbalance will be a negative number
        active_imbalance_signed = active_volume_buy_signed - active_volume_sell_signed
        active_imbalance_abs = np.uint64(np.abs(active_imbalance_signed))
        
        current["active_imbalance_signed"] = active_imbalance_signed
        current["active_imbalance_abs"] = active_imbalance_abs
        
        # Imbalance ratios - pure NumPy operations
        if volume_total == 0:
            current["active_imbalance_abs_ratio"] = np.float64(np.nan)
            current["active_imbalance_signed_ratio"] = np.float64(np.nan)
            current["active_imbalance_buy_ratio"] = np.float64(np.nan)
        else:
            # Convert to float64 once for all ratio calculations
            volume_total_float = np.float64(volume_total)
            
            # Pure NumPy division - no Python float() conversions
            current["active_imbalance_abs_ratio"] = np.float64(active_imbalance_abs) / volume_total_float
            current["active_imbalance_signed_ratio"] = np.float64(active_imbalance_signed) / volume_total_float
            current["active_imbalance_buy_ratio"] = np.float64(active_volume_buy) / volume_total_float
        
        # ========================================================================
        # Active Link Function Transforms for Ratios
        # ========================================================================

        # atanh transform for signed ratio in (-1, 1)
        # Clip to avoid numerical issues at boundaries
        active_signed_ratio_clipped = np.clip(
            current["active_imbalance_signed_ratio"], 
            np.float64(-0.9999), 
            np.float64(0.9999)
        )
        current["active_imbalance_signed_ratio_atanh"] = np.arctanh(active_signed_ratio_clipped)
        
        # logit transform for buy ratio in [0, 1]
        # logit(x) = log(x / (1 - x))
        active_buy_ratio_clipped = np.clip(
            current["active_imbalance_buy_ratio"], 
            np.float64(0.0001), 
            np.float64(0.9999)
        )
        current["active_imbalance_buy_ratio_logit"] = np.log(
            active_buy_ratio_clipped / (np.float64(1.0) - active_buy_ratio_clipped)
        )
        
        # logit transform for absolute imbalance ratio in [0, 1]
        active_abs_ratio_clipped = np.clip(
            current["active_imbalance_abs_ratio"], 
            np.float64(0.0001), 
            np.float64(0.9999)
        )
        current["active_imbalance_abs_ratio_logit"] = np.log(
            active_abs_ratio_clipped / (np.float64(1.0) - active_abs_ratio_clipped)
        )

        # ========================================================================
        # Active VWAP Calculations - OPTIMIZED FOR NUMPY
        # ========================================================================

        # [ BUY AGGRESSOR ] - VWAP (Lifted the Offer)
        if self._b_buy_aggressor_idx == 0:
            active_buy_vwap = np.float64(np.nan)
        else:
            # Get array slices (view, not copy - efficient)
            buy_prices = self._b_buy_aggressor_prices[:self._b_buy_aggressor_idx].astype(np.float64)
            buy_volumes = self._b_buy_aggressor_size_contributed[:self._b_buy_aggressor_idx]
            
            # Element-wise multiplication: each price weighted by its volume
            # Values are then summed for exact VWAP calculation
            # (Vectorized operations - fast C-level computation)
            total_weighted_buy_value = np.sum(buy_prices * buy_volumes, dtype=np.float64)
            total_buy_volume = np.sum(buy_volumes, dtype=np.uint64)

            # Pure NumPy division - stays in NumPy domain
            if total_buy_volume > 0:
                active_buy_vwap = total_weighted_buy_value / np.float64(total_buy_volume)
            else:
                active_buy_vwap = np.float64(np.nan)
        
        # [ SELL AGGRESSOR ] - VWAP (Hit the Bid)
        if self._a_sell_aggressor_idx == 0:
            active_sell_vwap = np.float64(np.nan)
        else:
            # Get array slices
            sell_prices = self._a_sell_aggressor_prices[:self._a_sell_aggressor_idx].astype(np.float64)
            sell_volumes = self._a_sell_aggressor_size_contributed[:self._a_sell_aggressor_idx]
            
            # Element-wise multiplication: each price weighted by its volume
            # Values are then summed for exact VWAP calculation
            # (Vectorized operations - fast C-level computation)
            total_weighted_sell_value = np.sum(sell_prices * sell_volumes, dtype=np.float64)
            total_sell_volume = np.sum(sell_volumes, dtype=np.uint64)

            # Pure NumPy division
            if total_sell_volume > 0:
                active_sell_vwap = total_weighted_sell_value / np.float64(total_sell_volume)
            else:
                active_sell_vwap = np.float64(np.nan)
        
        # [ NONE AGGRESSOR ] - VWAP (Unknown Side)
        if self._n_none_aggressor_idx == 0:
            active_none_vwap = np.float64(np.nan)
        else:
            # Get array slices
            none_prices = self._n_none_aggressor_prices[:self._n_none_aggressor_idx].astype(np.float64)
            none_volumes = self._n_none_aggressor_size_contributed[:self._n_none_aggressor_idx]
            
            # Element-wise multiplication: each price weighted by its volume
            # Values are then summed for exact VWAP calculation
            # (Vectorized operations - fast C-level computation)
            total_weighted_none_value = np.sum(none_prices * none_volumes, dtype=np.float64)
            total_none_volume = np.sum(none_volumes, dtype=np.uint64)

            # Pure NumPy division
            if total_none_volume > 0:
                active_none_vwap = total_weighted_none_value / np.float64(total_none_volume)
            else:
                active_none_vwap = np.float64(np.nan)

        # Store directly - already np.float64, no conversion needed
        current["active_buy_vwap"] = np.float64(active_buy_vwap)
        current["active_sell_vwap"] = np.float64(active_sell_vwap)
        current["active_none_vwap"] = np.float64(active_none_vwap)

        # ========================================================================
        # Active Spread and Midpoint VWAP Calculations
        # ========================================================================
        
        # [ REGULAR SPREAD & MIDPOINT ] - VWAP
        # All operations in NumPy domain, preserving float64 precision
        
        buy_vwap_valid = not np.isnan(active_buy_vwap)
        sell_vwap_valid = not np.isnan(active_sell_vwap)
        
        if not buy_vwap_valid and sell_vwap_valid:
            # Only SELL aggressor orders
            current["active_spread_vwap"] = np.float64(np.nan)
            current["active_midpoint_vwap"] = np.float64(active_sell_vwap)
        
        elif not sell_vwap_valid and buy_vwap_valid:
            # Only BUY aggressor orders
            current["active_spread_vwap"] = np.float64(np.nan)
            current["active_midpoint_vwap"] = np.float64(active_buy_vwap)
        
        elif buy_vwap_valid and sell_vwap_valid:
            # Both sides available - pure NumPy arithmetic
            current["active_spread_vwap"] = np.float64(active_buy_vwap - active_sell_vwap)
            # Use true division (/) not floor division (//)
            current["active_midpoint_vwap"] = np.float64((active_buy_vwap + active_sell_vwap) * np.float64(0.5))
        
        else:
            # Neither side available
            current["active_spread_vwap"] = np.float64(np.nan)
            current["active_midpoint_vwap"] = np.float64(np.nan)

        # ========================================================================
        # Active Weighted Midpoints
        # ========================================================================

        """
        [ IMBALANCE-WEIGHTED MIDPOINT PRICE ]

        > IWM = MIDPRICE + ((SPREAD * 0.5) * IMBALANCE)
        
        The gravity of transacted prices pulls towards most "accurate" price (linear tilt)
        
        Use of ["active_signed_ratio_clipped"] ratio, value between (-1, 1)
        Clip to avoid numerical issues at boundaries
        (i.e. 1 = 100% buy imbalance)
        """

        if np.isnan(current["active_midpoint_vwap"]) or np.isnan(current["active_spread_vwap"]):
            current["active_mid_imbalance_weighted"] = np.float64(np.nan)
        else:
            current["active_mid_imbalance_weighted"] = current["active_midpoint_vwap"] + (
                (np.float64(0.5) * current["active_spread_vwap"]) * active_signed_ratio_clipped
            )

        """
        [ FLOW-WEIGHTED MIDPOINT PRICE ]

        > FWM = (Price_Sell * Volume_Sell) + (Price_Buy * Volume_Buy) / total_ab_volume
        
        Uses SAME-SIDE weighting!

        "Where we traded." pulls toward side with more recent trades.
        Tells you "Where price actually traded."
        "Center of mass" of trading actually happened.

        NOTE: Uses only A/B volume (excludes N-side) for weighting
        """
        
        # Calculate total A/B volume (excluding N-side)
        total_ab_volume = np.uint64(active_volume_buy) + np.uint64(active_volume_sell)
        
        if total_ab_volume == 0 or not (buy_vwap_valid and sell_vwap_valid):
            current["active_mid_flow_weighted"] = np.float64(np.nan)
        else:
            # Calculate flow-weighted midpoint using same-side weighting
            flow_weighted_numerator = (
                (active_sell_vwap * np.float64(active_volume_sell)) + 
                (active_buy_vwap * np.float64(active_volume_buy))
            )
            current["active_mid_flow_weighted"] = flow_weighted_numerator / np.float64(total_ab_volume)

        """
        [ AGGRESSOR-WEIGHTED MIDPOINT PRICE ]

        > AWM = (Price_Sell * Volume_Buy) + (Price_Buy * Volume_Sell) / total_ab_volume

        Use OPPOSITE-SIDE weighting! (microprice analogue, predictive tilt)
        
        A predictive metric, works as a pressure indicator similar to microprice (for BBO)
        "Where we may trade next."

        i.e. "Given that buy flow dominated, the next move is likely upward, so the pressure-weighted mid starts pulling toward the sell side"
        
        NOTE: Uses only A/B volume (excludes N-side) for weighting
        """
        
        if total_ab_volume == 0 or not (buy_vwap_valid and sell_vwap_valid):
            current["active_mid_aggressor_weighted"] = np.float64(np.nan)
        else:
            # Calculate aggressor-weighted midpoint using opposite-side weighting
            aggressor_weighted_numerator = (
                (active_sell_vwap * np.float64(active_volume_buy)) + 
                (active_buy_vwap * np.float64(active_volume_sell))
            )
            current["active_mid_aggressor_weighted"] = aggressor_weighted_numerator / np.float64(total_ab_volume)
        
        # ========================================================================
        # Active Price Range Metrics
        # ========================================================================

        # [ BUY AGGRESSOR ] - Price Range
        if self._b_buy_aggressor_idx == 0:
            current["active_buy_price_min"] = np.float64(np.nan)
            current["active_buy_price_max"] = np.float64(np.nan)
            current["active_buy_price_range"] = np.float64(np.nan)
        else:
            buy_prices_float = self._b_buy_aggressor_prices[:self._b_buy_aggressor_idx].astype(np.float64)
            current["active_buy_price_min"] = np.float64(np.min(buy_prices_float))
            current["active_buy_price_max"] = np.float64(np.max(buy_prices_float))
            current["active_buy_price_range"] = current["active_buy_price_max"] - current["active_buy_price_min"]

        # [ SELL AGGRESSOR ] - Price Range
        if self._a_sell_aggressor_idx == 0:
            current["active_sell_price_min"] = np.float64(np.nan)
            current["active_sell_price_max"] = np.float64(np.nan)
            current["active_sell_price_range"] = np.float64(np.nan)
        else:
            sell_prices_float = self._a_sell_aggressor_prices[:self._a_sell_aggressor_idx].astype(np.float64)
            current["active_sell_price_min"] = np.float64(np.min(sell_prices_float))
            current["active_sell_price_max"] = np.float64(np.max(sell_prices_float))
            current["active_sell_price_range"] = current["active_sell_price_max"] - current["active_sell_price_min"]

        # [ NONE AGGRESSOR ] - Price Range
        if self._n_none_aggressor_idx == 0:
            current["active_none_price_min"] = np.float64(np.nan)
            current["active_none_price_max"] = np.float64(np.nan)
            current["active_none_price_range"] = np.float64(np.nan)
        else:
            none_prices_float = self._n_none_aggressor_prices[:self._n_none_aggressor_idx].astype(np.float64)
            current["active_none_price_min"] = np.float64(np.min(none_prices_float))
            current["active_none_price_max"] = np.float64(np.max(none_prices_float))
            current["active_none_price_range"] = current["active_none_price_max"] - current["active_none_price_min"]

        # ========================================================================
        # Timestamp Calculations
        # ========================================================================

        if self._central_timestamp_idx == 0:
            current["start_ts_ns"] = np.uint64(0)
            current["end_ts_ns"] = np.uint64(0)
            current["time_elapsed_ns"] = np.uint64(0)
        else:
            # Use central timestamps array
            timestamps = self._central_timestamps[:self._central_timestamp_idx]

            # NumPy min/max operations
            current["start_ts_ns"] = np.uint64(np.min(timestamps))
            current["end_ts_ns"] = np.uint64(np.max(timestamps))
            current["time_elapsed_ns"] = current["end_ts_ns"] - current["start_ts_ns"]
        
        # ========================================================================
        # Pace Metrics (Total)
        # ========================================================================

        # Pace of Order Flow (contracts per nanosecond)
        if current["time_elapsed_ns"] > 0:
            current["pace_of_contracts_traded"] = np.float64(
                np.float64(volume_total) / np.float64(current["time_elapsed_ns"])
            )
        else:
            current["pace_of_contracts_traded"] = np.float64(np.nan)
        
        # ========================================================================
        # Active Pace Metrics (Buy/Sell)
        # ========================================================================

        if current["time_elapsed_ns"] > 0:
            time_elapsed_float = np.float64(current["time_elapsed_ns"])
            
            # Buy pace
            current["active_buy_pace"] = np.float64(active_volume_buy) / time_elapsed_float
            
            # Sell pace
            current["active_sell_pace"] = np.float64(active_volume_sell) / time_elapsed_float
        else:
            current["active_buy_pace"] = np.float64(np.nan)
            current["active_sell_pace"] = np.float64(np.nan)

        # Log transforms for pace
        if current["active_buy_pace"] > 0:
            current["active_buy_pace_log"] = np.float64(np.log(current["active_buy_pace"]))
        else:
            current["active_buy_pace_log"] = np.float64(np.nan)
        
        if current["active_sell_pace"] > 0:
            current["active_sell_pace_log"] = np.float64(np.log(current["active_sell_pace"]))
        else:
            current["active_sell_pace_log"] = np.float64(np.nan)

        # ========================================================================
        # Log Transforms for Time/Pace (Level)
        # ========================================================================

        # Log of time elapsed
        if current["time_elapsed_ns"] > 0:
            current["time_elapsed_ns_log"] = np.float64(np.log(np.float64(current["time_elapsed_ns"])))
        else:
            current["time_elapsed_ns_log"] = np.float64(np.nan)
        
        # Log of pace of orders
        if current["pace_of_contracts_traded"] > 0:
            current["pace_of_contracts_traded_log"] = np.float64(np.log(current["pace_of_contracts_traded"]))
        else:
            current["pace_of_contracts_traded_log"] = np.float64(np.nan)

        # ========================================================================
        # Active N-Side Inferred Classification
        # ========================================================================

        """
        N-side trades are classified as "likely buy" or "likely sell" by comparing
        trade price to a reference price derived from active VWAPs.

        Reference Price Logic:
        - If both buy and sell VWAPs available: use midpoint
        - If only buy VWAP available: use buy VWAP
        - If only sell VWAP available: use sell VWAP
        - If neither available: fall back to previous bar's midpoint
        - If no previous bar: cannot classify (return 0.0)

        Classification Logic (per N-side trade):
        - If trade_price > reference: inferred as buy
        - If trade_price <= reference: inferred as sell

        SPECIAL CASE - PREVIOUS BAR FALLBACK:
        When the entire bar consists of N-side orders only (no A-side or B-side),
        we cannot derive a reference price from current bar data. This typically
        occurs when an oversized N-side order (e.g., auction, cross, or spread trade)
        fills the entire volume bar. In this case, we fall back to the previous
        bar's midpoint to enable classification. This is logged as CRITICAL because
        it indicates unusual market conditions that may affect metric reliability.
        """

        if self._n_none_aggressor_idx == 0:
            # No N-side trades to classify
            current["active_none_inferred_buy_volume"] = np.float64(0.0)
            current["active_none_inferred_sell_volume"] = np.float64(0.0)
            current["active_none_inferred_buy_vwap"] = np.float64(np.nan)
            current["active_none_inferred_sell_vwap"] = np.float64(np.nan)
        else:
            # Determine reference price
            used_previous_bar_fallback = False
            
            if buy_vwap_valid and sell_vwap_valid:
                reference_price = (active_buy_vwap + active_sell_vwap) * np.float64(0.5)
            elif buy_vwap_valid:
                reference_price = active_buy_vwap
            elif sell_vwap_valid:
                reference_price = active_sell_vwap
            elif previous is not None and not np.isnan(previous["active_midpoint_vwap"]):
                # FALLBACK: No A/B trades in current bar, use previous bar's active midpoint
                reference_price = np.float64(previous["active_midpoint_vwap"])
                used_previous_bar_fallback = True
            elif previous is not None and not np.isnan(previous["passive_midprice"]):
                # FALLBACK: No active midpoint available, use previous bar's passive midprice
                reference_price = np.float64(previous["passive_midprice"])
                used_previous_bar_fallback = True
            else:
                # Cannot classify without reference price
                reference_price = np.float64(np.nan)
            
            # Log critical warning if fallback was used
            if used_previous_bar_fallback:
                logger.critical(
                    f"N-SIDE ONLY BAR: Entire bar filled by N-side orders. "
                    f"Using previous bar midpoint as reference price for (ACTIVE) classification. "
                    f"N-side volume: {active_volume_none}, reference_price: {reference_price}"
                )
            
            if np.isnan(reference_price):
                current["active_none_inferred_buy_volume"] = np.float64(0.0)
                current["active_none_inferred_sell_volume"] = np.float64(0.0)
                current["active_none_inferred_buy_vwap"] = np.float64(np.nan)
                current["active_none_inferred_sell_vwap"] = np.float64(np.nan)
            else:
                # Get N-side prices and volumes
                none_prices = self._n_none_aggressor_prices[:self._n_none_aggressor_idx].astype(np.float64)
                none_volumes = self._n_none_aggressor_size_contributed[:self._n_none_aggressor_idx].astype(np.float64)
                
                # Vectorized classification: price > reference → buy, else → sell
                buy_mask = none_prices > reference_price
                
                # Inferred volumes
                current["active_none_inferred_buy_volume"] = np.float64(np.sum(none_volumes[buy_mask]))
                current["active_none_inferred_sell_volume"] = np.float64(np.sum(none_volumes[~buy_mask]))
                
                # Inferred VWAPs (for precise adjusted calculations)
                inferred_buy_prices = none_prices[buy_mask]
                inferred_buy_volumes = none_volumes[buy_mask]
                inferred_sell_prices = none_prices[~buy_mask] # The tilde (~) in NumPy is the bitwise NOT operator, which inverts boolean arrays.
                inferred_sell_volumes = none_volumes[~buy_mask] # Applying inverting boolean "buy" mask
                
                # Inferred Buy VWAP
                if np.sum(inferred_buy_volumes) > 0:
                    current["active_none_inferred_buy_vwap"] = np.float64(
                        np.sum(inferred_buy_prices * inferred_buy_volumes) / np.sum(inferred_buy_volumes)
                    )
                else:
                    current["active_none_inferred_buy_vwap"] = np.float64(np.nan)
                
                # Inferred Sell VWAP
                if np.sum(inferred_sell_volumes) > 0:
                    current["active_none_inferred_sell_vwap"] = np.float64(
                        np.sum(inferred_sell_prices * inferred_sell_volumes) / np.sum(inferred_sell_volumes)
                    )
                else:
                    current["active_none_inferred_sell_vwap"] = np.float64(np.nan)
        
        # ========================================================================
        # Adjusted Metrics (Active + N-Side Inferred)
        # ========================================================================

        """
        Adjusted metrics incorporate N-side inferred classifications into the
        active imbalance and VWAP calculations. This provides a more complete
        picture of order flow by including trades where the aggressor side
        was unknown but could be inferred from price relative to a reference.
        
        Calculation approach:
        - adjusted_volume_* = active_volume_* + active_none_inferred_*_volume
        - adjusted_*_vwap = weighted average of active VWAP and inferred N-side VWAP
        """

        # ========================================================================
        # Adjusted Volumes
        # ========================================================================

        adjusted_volume_buy = np.float64(active_volume_buy) + current["active_none_inferred_buy_volume"]
        adjusted_volume_sell = np.float64(active_volume_sell) + current["active_none_inferred_sell_volume"]

        current["adjusted_volume_buy"] = np.float64(adjusted_volume_buy)
        current["adjusted_volume_sell"] = np.float64(adjusted_volume_sell)

        # ========================================================================
        # Adjusted Imbalance Metrics
        # ========================================================================

        adjusted_imbalance_signed = adjusted_volume_buy - adjusted_volume_sell
        adjusted_imbalance_abs = np.abs(adjusted_imbalance_signed)

        current["adjusted_imbalance_signed"] = np.float64(adjusted_imbalance_signed)
        current["adjusted_imbalance_abs"] = np.float64(adjusted_imbalance_abs)

        # Adjusted imbalance ratios
        if volume_total == 0:
            current["adjusted_imbalance_signed_ratio"] = np.float64(np.nan)
            current["adjusted_imbalance_abs_ratio"] = np.float64(np.nan)
            current["adjusted_imbalance_buy_ratio"] = np.float64(np.nan)
        else:
            volume_total_float = np.float64(volume_total)
            current["adjusted_imbalance_signed_ratio"] = adjusted_imbalance_signed / volume_total_float
            current["adjusted_imbalance_abs_ratio"] = adjusted_imbalance_abs / volume_total_float
            current["adjusted_imbalance_buy_ratio"] = adjusted_volume_buy / volume_total_float

        # ========================================================================
        # Adjusted Link Function Transforms
        # ========================================================================

        # atanh transform for signed ratio in (-1, 1)
        adjusted_signed_ratio_clipped = np.clip(
            current["adjusted_imbalance_signed_ratio"],
            np.float64(-0.9999),
            np.float64(0.9999)
        )
        current["adjusted_imbalance_signed_ratio_atanh"] = np.arctanh(adjusted_signed_ratio_clipped)

        # logit transform for buy ratio in [0, 1]
        adjusted_buy_ratio_clipped = np.clip(
            current["adjusted_imbalance_buy_ratio"],
            np.float64(0.0001),
            np.float64(0.9999)
        )
        current["adjusted_imbalance_buy_ratio_logit"] = np.log(
            adjusted_buy_ratio_clipped / (np.float64(1.0) - adjusted_buy_ratio_clipped)
        )

        # logit transform for absolute imbalance ratio in [0, 1]
        adjusted_abs_ratio_clipped = np.clip(
            current["adjusted_imbalance_abs_ratio"],
            np.float64(0.0001),
            np.float64(0.9999)
        )
        current["adjusted_imbalance_abs_ratio_logit"] = np.log(
            adjusted_abs_ratio_clipped / (np.float64(1.0) - adjusted_abs_ratio_clipped)
        )

        # ========================================================================
        # Adjusted VWAP Calculations
        # ========================================================================

        """
        Adjusted VWAPs combine active VWAPs with inferred N-side VWAPs,
        weighted by their respective volumes. This provides a more accurate
        price representation when N-side trades are significant.
        
        Formula:
        adjusted_buy_vwap = (active_buy_vwap * active_volume_buy + 
                            active_none_inferred_buy_vwap * active_none_inferred_buy_volume) /
                           adjusted_volume_buy
        """

        # [ ADJUSTED BUY VWAP ]
        # No adjusted_volume, thus no new calculations need to be made
        if adjusted_volume_buy == 0:
            adjusted_buy_vwap = np.float64(np.nan)

        # Both active and inferred N-side buy volume exist
        elif active_volume_buy > 0 and current["active_none_inferred_buy_volume"] > 0:
            if not np.isnan(current["active_none_inferred_buy_vwap"]):
                adjusted_buy_vwap_numerator = (
                    (active_buy_vwap * np.float64(active_volume_buy)) +
                    (current["active_none_inferred_buy_vwap"] * current["active_none_inferred_buy_volume"])
                )
                adjusted_buy_vwap = adjusted_buy_vwap_numerator / adjusted_volume_buy

            # Inferred VWAP is NaN, use active only
            else:
                adjusted_buy_vwap = active_buy_vwap

        # Only active buy volume exists
        elif active_volume_buy > 0:
            adjusted_buy_vwap = active_buy_vwap

        # Only inferred N-side buy volume exists
        elif current["active_none_inferred_buy_volume"] > 0 and not np.isnan(current["active_none_inferred_buy_vwap"]):
            adjusted_buy_vwap = current["active_none_inferred_buy_vwap"]
        else:
            adjusted_buy_vwap = np.float64(np.nan)

        current["adjusted_buy_vwap"] = np.float64(adjusted_buy_vwap)


        # [ ADJUSTED SELL VWAP ]
        # No adjusted_volume, thus no new calculations need to be made
        if adjusted_volume_sell == 0:
            adjusted_sell_vwap = np.float64(np.nan)

        # Both active and inferred N-side sell volume exist
        elif active_volume_sell > 0 and current["active_none_inferred_sell_volume"] > 0:
            if not np.isnan(current["active_none_inferred_sell_vwap"]):
                adjusted_sell_vwap_numerator = (
                    (active_sell_vwap * np.float64(active_volume_sell)) +
                    (current["active_none_inferred_sell_vwap"] * current["active_none_inferred_sell_volume"])
                )
                adjusted_sell_vwap = adjusted_sell_vwap_numerator / adjusted_volume_sell

            # Inferred VWAP is NaN, use active only
            else:
                adjusted_sell_vwap = active_sell_vwap

        # Only active sell volume exists
        elif active_volume_sell > 0:
            adjusted_sell_vwap = active_sell_vwap
        
        # Only inferred N-side sell volume exists
        elif current["active_none_inferred_sell_volume"] > 0 and not np.isnan(current["active_none_inferred_sell_vwap"]):
            adjusted_sell_vwap = current["active_none_inferred_sell_vwap"]
        else:
            adjusted_sell_vwap = np.float64(np.nan)

        current["adjusted_sell_vwap"] = np.float64(adjusted_sell_vwap)

        # ========================================================================
        # Adjusted Spread and Midpoint VWAP
        # ========================================================================

        adjusted_buy_vwap_valid = not np.isnan(adjusted_buy_vwap)
        adjusted_sell_vwap_valid = not np.isnan(adjusted_sell_vwap)

        if adjusted_buy_vwap_valid and adjusted_sell_vwap_valid:
            current["adjusted_spread_vwap"] = np.float64(adjusted_buy_vwap - adjusted_sell_vwap)
            current["adjusted_midpoint_vwap"] = np.float64((adjusted_buy_vwap + adjusted_sell_vwap) * np.float64(0.5))
        elif adjusted_buy_vwap_valid:
            current["adjusted_spread_vwap"] = np.float64(np.nan)
            current["adjusted_midpoint_vwap"] = np.float64(adjusted_buy_vwap)
        elif adjusted_sell_vwap_valid:
            current["adjusted_spread_vwap"] = np.float64(np.nan)
            current["adjusted_midpoint_vwap"] = np.float64(adjusted_sell_vwap)
        else:
            current["adjusted_spread_vwap"] = np.float64(np.nan)
            current["adjusted_midpoint_vwap"] = np.float64(np.nan)

        # ========================================================================
        # Adjusted Weighted Midpoints
        # ========================================================================

        """
        [ ADJUSTED IMBALANCE-WEIGHTED MIDPOINT PRICE ]

        > IWM = MIDPRICE + ((SPREAD * 0.5) * IMBALANCE)
        
        Uses adjusted midpoint, spread, and imbalance ratio.
        """

        if np.isnan(current["adjusted_midpoint_vwap"]) or np.isnan(current["adjusted_spread_vwap"]):
            current["adjusted_mid_imbalance_weighted"] = np.float64(np.nan)
        else:
            current["adjusted_mid_imbalance_weighted"] = current["adjusted_midpoint_vwap"] + (
                (np.float64(0.5) * current["adjusted_spread_vwap"]) * adjusted_signed_ratio_clipped
            )

        """
        [ ADJUSTED FLOW-WEIGHTED MIDPOINT PRICE ]

        > FWM = (Price_Sell * Volume_Sell) + (Price_Buy * Volume_Buy) / total_adjusted_volume
        
        Uses SAME-SIDE weighting with adjusted volumes and VWAPs.
        "Where we traded" including inferred N-side classification.
        """

        total_adjusted_volume = adjusted_volume_buy + adjusted_volume_sell

        if total_adjusted_volume == 0 or not (adjusted_buy_vwap_valid and adjusted_sell_vwap_valid):
            current["adjusted_mid_flow_weighted"] = np.float64(np.nan)
        else:
            adjusted_flow_weighted_numerator = (
                (adjusted_sell_vwap * adjusted_volume_sell) +
                (adjusted_buy_vwap * adjusted_volume_buy)
            )
            current["adjusted_mid_flow_weighted"] = adjusted_flow_weighted_numerator / total_adjusted_volume

        """
        [ ADJUSTED AGGRESSOR-WEIGHTED MIDPOINT PRICE ]

        > AWM = (Price_Sell * Volume_Buy) + (Price_Buy * Volume_Sell) / total_adjusted_volume

        Uses OPPOSITE-SIDE weighting with adjusted volumes and VWAPs.
        Predictive metric: "Where we may trade next" including inferred N-side.
        """

        if total_adjusted_volume == 0 or not (adjusted_buy_vwap_valid and adjusted_sell_vwap_valid):
            current["adjusted_mid_aggressor_weighted"] = np.float64(np.nan)
        else:
            adjusted_aggressor_weighted_numerator = (
                (adjusted_sell_vwap * adjusted_volume_buy) +
                (adjusted_buy_vwap * adjusted_volume_sell)
            )
            current["adjusted_mid_aggressor_weighted"] = adjusted_aggressor_weighted_numerator / total_adjusted_volume
        
        # ========================================================================
        # Instrument ID Tracking
        # ========================================================================

        total_orders = (
            self._a_sell_aggressor_idx + 
            self._b_buy_aggressor_idx + 
            self._n_none_aggressor_idx
        )

        if total_orders == 0:
            current["contract_roll"] = np.bool_(False)
            current["latest_instrument_id"] = np.uint32(0)
        else:
            # Collect instrument IDs from all sides
            all_instrument_ids = []
            all_timestamps = []
            
            if self._b_buy_aggressor_idx > 0:
                all_instrument_ids.append(self._b_buy_aggressor_instrument_ids[:self._b_buy_aggressor_idx])
                all_timestamps.append(self._b_buy_aggressor_timestamps[:self._b_buy_aggressor_idx])
            
            if self._a_sell_aggressor_idx > 0:
                all_instrument_ids.append(self._a_sell_aggressor_instrument_ids[:self._a_sell_aggressor_idx])
                all_timestamps.append(self._a_sell_aggressor_timestamps[:self._a_sell_aggressor_idx])
            
            if self._n_none_aggressor_idx > 0:
                all_instrument_ids.append(self._n_none_aggressor_instrument_ids[:self._n_none_aggressor_idx])
                all_timestamps.append(self._n_none_aggressor_timestamps[:self._n_none_aggressor_idx])
            
            combined_ids = np.concatenate(all_instrument_ids)
            combined_timestamps = np.concatenate(all_timestamps)
            
            unique_ids = np.unique(combined_ids)

            # Detect contract roll if more than one unique instrument_id
            current["contract_roll"] = np.bool_(len(unique_ids) > 1)
            
            # Find latest instrument ID by timestamp
            latest_idx = np.argmax(combined_timestamps)
            current["latest_instrument_id"] = np.uint32(combined_ids[latest_idx])

        # ========================================================================
        # Passive Metrics
        # ========================================================================

        """
        Passive variables treat all trades equally, ignoring the aggressor flag.
        Buy/sell classification is inferred from price movement using a CDF-based
        probabilistic model.
        
        Calculation Dependencies:
        - returns_distribution.std() → Global σ for normalizing price changes
        - classifier_distribution.cdf() → Standard normal CDF for probability conversion
        """

        # ========================================================================
        # Passive Midprice (VWAP of ALL trades: A + B + N)
        # ========================================================================

        # Innerworkings of `_calculate_passive_midprice`:
        # From each "Trade Side" NumPy Array
        # (i.e. "_b_buy_aggressor...", "_a_sell_aggressor...", "_n_none_aggressor...")
        #
        # Aggregates into 1 list:
        # - Prices Contributed (all_prices)
        # - Size Contributed (all_volumes)
        #
        # Performs element-wise multiplication for exact VWAP calculation
        # Returns `passive_midprice` if (total_orders > 0 and total_volume > 0)
        # Else returns `np.nan`

        current["passive_midprice"] = np.float64(self._calculate_passive_midprice(total_orders=total_orders))

        # ========================================================================
        # Passive Midprice Deltas (vs Previous Bar)
        # ========================================================================

        if previous is None or np.isnan(passive_midprice):
            current["passive_midprice_delta_price"] = np.float64(np.nan)
            current["passive_midprice_delta_percent"] = np.float64(np.nan)
            current["passive_midprice_delta_log"] = np.float64(np.nan)
        else:
            prev_passive_midprice = np.float64(previous["passive_midprice"])
            
            if np.isnan(prev_passive_midprice):
                current["passive_midprice_delta_price"] = np.float64(np.nan)
                current["passive_midprice_delta_percent"] = np.float64(np.nan)
                current["passive_midprice_delta_log"] = np.float64(np.nan)
            else:
                # Raw delta
                current["passive_midprice_delta_price"] = passive_midprice - prev_passive_midprice
                
                # Percent delta (decimal form)
                if prev_passive_midprice != 0:
                    current["passive_midprice_delta_percent"] = (
                        (passive_midprice - prev_passive_midprice) / prev_passive_midprice
                    )
                else:
                    current["passive_midprice_delta_percent"] = np.float64(np.nan)
                
                # Log delta
                if prev_passive_midprice > 0 and passive_midprice > 0:
                    current["passive_midprice_delta_log"] = np.log(passive_midprice / prev_passive_midprice)
                else:
                    current["passive_midprice_delta_log"] = np.float64(np.nan)

        # ========================================================================
        # Passive Normalized and CDF Metrics
        # ========================================================================

        # Get standard deviation from distribution lookup
        returns_std = np.float64(returns_distribution.std())
        
        # Normalize price delta by global σ (z-score)
        if np.isnan(current["passive_midprice_delta_price"]) or returns_std == 0:
            current["passive_midprice_delta_normalized"] = np.float64(np.nan)
        else:
            current["passive_midprice_delta_normalized"] = (
                
                current["passive_midprice_delta_price"] / returns_std
                
            )


        # CDF probability score
        if np.isnan(current["passive_midprice_delta_normalized"]):
            current["passive_midprice_delta_cdf"] = np.float64(np.nan)
        elif current["passive_midprice_delta_normalized"] == 0:
            # Neutral probability (no directional signal)
            current["passive_midprice_delta_cdf"] = np.float64(0.5)
        else:
            current["passive_midprice_delta_cdf"] = np.float64(

                classifier_distribution.cdf(current["passive_midprice_delta_normalized"])

            )


        # ========================================================================
        # Passive Volume Classification
        # ========================================================================

        """
        The CDF value represents the probability that the price movement was driven
        by buying pressure. Values close to 1 indicate strong buying; values close
        to 0 indicate strong selling; 0.5 indicates neutral.
        
        Note: Fractional volumes are intentional. Errors average out over time,
        and fractional representation preserves probabilistic information.
        """

        if np.isnan(current["passive_midprice_delta_cdf"]):
            current["passive_buy_volume"] = np.float64(0.0)
            current["passive_sell_volume"] = np.float64(0.0)
        else:
            current["passive_buy_volume"] = np.float64(

                current["passive_midprice_delta_cdf"] * np.float64(volume_total)

            )
            current["passive_sell_volume"] = np.float64(

                np.float64(volume_total) - current["passive_buy_volume"]

            )

        # ========================================================================
        # Passive Imbalance Metrics
        # ========================================================================

        passive_imbalance_signed = current["passive_buy_volume"] - current["passive_sell_volume"]
        passive_imbalance_abs = np.abs(passive_imbalance_signed)

        current["passive_imbalance_signed"] = np.float64(passive_imbalance_signed)
        current["passive_imbalance_abs"] = np.float64(passive_imbalance_abs)

        if volume_total == 0:
            current["passive_imbalance_signed_ratio"] = np.float64(np.nan)
            current["passive_imbalance_abs_ratio"] = np.float64(np.nan)
            current["passive_imbalance_buy_ratio"] = np.float64(np.nan)
        else:
            volume_total_float = np.float64(volume_total)
            current["passive_imbalance_signed_ratio"] = passive_imbalance_signed / volume_total_float
            current["passive_imbalance_abs_ratio"] = passive_imbalance_abs / volume_total_float
            current["passive_imbalance_buy_ratio"] = current["passive_buy_volume"] / volume_total_float

        # ========================================================================
        # Passive Link Function Transforms
        # ========================================================================

        # atanh transform for signed ratio in (-1, 1)
        passive_signed_ratio_clipped = np.clip(
            current["passive_imbalance_signed_ratio"], 
            np.float64(-0.9999), 
            np.float64(0.9999)
        )
        current["passive_imbalance_signed_ratio_atanh"] = np.arctanh(passive_signed_ratio_clipped)
        
        # logit transform for buy ratio in [0, 1]
        passive_buy_ratio_clipped = np.clip(
            current["passive_imbalance_buy_ratio"], 
            np.float64(0.0001), 
            np.float64(0.9999)
        )
        current["passive_imbalance_buy_ratio_logit"] = np.log(
            passive_buy_ratio_clipped / (np.float64(1.0) - passive_buy_ratio_clipped)
        )
        
        # logit transform for absolute imbalance ratio in [0, 1]
        passive_abs_ratio_clipped = np.clip(
            current["passive_imbalance_abs_ratio"], 
            np.float64(0.0001), 
            np.float64(0.9999)
        )
        current["passive_imbalance_abs_ratio_logit"] = np.log(
            passive_abs_ratio_clipped / (np.float64(1.0) - passive_abs_ratio_clipped)
        )

        # ========================================================================
        # Divergence Metrics (Active vs Passive)
        # ========================================================================

        """
        Divergence metrics compare active (aggressor-known) classification to
        passive (price-inferred) classification.
        
        Interpretation:
        - Positive divergence_buy_volume: Active detected more buying than passive inferred.
          Could indicate large buy orders absorbed without price impact, or spread dynamics
          not captured by midprice.
        - Negative divergence_buy_volume: Active detected less buying than passive inferred.
          Could indicate price moved up despite balanced or sell-heavy flow.
        - Large absolute divergence: Signals potential predictive opportunity or liquidity
          dynamics worth investigating.
        """

        ## ?? use inferred here?

        # Volume divergences
        if np.isnan(current["passive_buy_volume"]):
            current["divergence_buy_volume"] = np.float64(np.nan)
            current["divergence_sell_volume"] = np.float64(np.nan)
        else:
            current["divergence_buy_volume"] = np.float64(active_volume_buy) - current["passive_buy_volume"]
            current["divergence_sell_volume"] = np.float64(active_volume_sell) - current["passive_sell_volume"]

        # Imbalance divergences
        if np.isnan(current["passive_imbalance_signed"]):
            current["divergence_imbalance_signed"] = np.float64(np.nan)
        else:
            current["divergence_imbalance_signed"] = (
                np.float64(active_imbalance_signed) - current["passive_imbalance_signed"]
            )

        # Ratio divergences
        if np.isnan(current["passive_imbalance_signed_ratio"]) or np.isnan(current["active_imbalance_signed_ratio"]):
            current["divergence_imbalance_signed_ratio"] = np.float64(np.nan)
        else:
            current["divergence_imbalance_signed_ratio"] = (
                current["active_imbalance_signed_ratio"] - current["passive_imbalance_signed_ratio"]
            )

        if np.isnan(current["passive_imbalance_buy_ratio"]) or np.isnan(current["active_imbalance_buy_ratio"]):
            current["divergence_buy_ratio"] = np.float64(np.nan)
        else:
            current["divergence_buy_ratio"] = (
                current["active_imbalance_buy_ratio"] - current["passive_imbalance_buy_ratio"]
            )

        
        return current




    def _calculate_deltas(
        self,
        current: Dict[str, Any],
        previous: np.ndarray
    ) -> Dict[str, Any]:
        """
        Calculate delta statistics by comparing current VolumeBar (i) to previous VolumeBar (i - 1).
        Fully optimized for NumPy performance with float64 price precision.
        
        Args:
            current: Raw statistics dictionary for current bar
            previous: NumPy structured array view from buffer.get_previous_view()
                    Access fields via bracket notation: previous["field_name"]
                    
        Returns:
            Dictionary with all delta statistics including:
            - Active: order counts, volumes, imbalances, ratios, link transforms,
                     VWAPs, weighted midpoints, price ranges, pace, N-side inferred
            - Adjusted: volumes, imbalances, ratios, link transforms, VWAPs, weighted midpoints
            - Passive: midprice deltas (second-order), normalized/CDF, volumes, 
                      imbalances, ratios, link transforms
            - Temporal: time elapsed, pace of contracts traded
            - Divergence: all divergence metrics
            - Derived: price direction indicator (boolean)
        
        Delta Types:
            - Raw delta: value[i] - value[i-1]
            - Percent delta (decimal): (value[i] - value[i-1]) / value[i-1]
            - Log delta: log((value[i] + ε) / (value[i-1] + ε))
        
        Edge Case Handling:
            - Percent delta: NaN if previous value is 0
            - Log delta: Uses EPSILON for numerical stability
            - NaN propagation: If either value is NaN, result is NaN
        
        <><><> MUST MATCH "models.VOLUMEBAR_DTYPE" SCHEMA <><><>
        """

        deltas: Dict[str, Any] = {}
        
        # ========================================================================
        # HELPER FUNCTIONS
        # ========================================================================
        
        def _delta_amount_int32(curr_val: Any, prev_val: Any) -> np.int32:
            """Calculate integer amount delta, casting unsigned to signed."""
            return np.int32(np.int64(curr_val) - np.int64(prev_val))
        
        def _delta_amount_int64(curr_val: Any, prev_val: Any) -> np.int64:
            """Calculate int64 amount delta."""
            return np.int64(curr_val) - np.int64(prev_val)
        
        def _delta_amount_float64(curr_val: Any, prev_val: Any) -> np.float64:
            """Calculate float64 amount delta."""
            return np.float64(curr_val) - np.float64(prev_val)
        
        def _delta_pct_float64(curr_val: Any, prev_val: Any) -> np.float64:
            """Calculate percentage delta. Returns NaN if prev is 0."""
            prev_f = np.float64(prev_val)
            if np.abs(prev_f) > EPSILON:
                return np.float64((np.float64(curr_val) - prev_f) / prev_f)
            else:
                return np.float64(np.nan)
        
        def _delta_log_float64(curr_val: Any, prev_val: Any) -> np.float64:
            """Calculate log delta with epsilon for stability."""
            curr_adj = np.float64(curr_val) + EPSILON
            prev_adj = np.float64(prev_val) + EPSILON
            if curr_adj > 0 and prev_adj > 0:
                return np.float64(np.log(curr_adj / prev_adj))
            else:
                return np.float64(np.nan)
        
        # ========================================================================
        # 2. AMBIGUOUS/TOTAL DELTAS
        # ========================================================================
        
        # order_count: amount, pct, log
        deltas["delta_order_count"] = _delta_amount_int32(
            current["order_count"], previous["order_count"]
        )
        deltas["delta_order_count_pct"] = _delta_pct_float64(
            current["order_count"], previous["order_count"]
        )
        deltas["delta_order_count_log"] = _delta_log_float64(
            current["order_count"], previous["order_count"]
        )
        
        # order_splits: amount only
        deltas["delta_order_splits"] = _delta_amount_int32(
            current["order_splits"], previous["order_splits"]
        )
        
        # volume_total: amount, pct, log
        deltas["delta_volume_total"] = _delta_amount_int32(
            current["volume_total"], previous["volume_total"]
        )
        deltas["delta_volume_total_pct"] = _delta_pct_float64(
            current["volume_total"], previous["volume_total"]
        )
        deltas["delta_volume_total_log"] = _delta_log_float64(
            current["volume_total"], previous["volume_total"]
        )
        
        # ========================================================================
        # 3. ACTIVE DELTAS
        # ========================================================================
        
        # ----------------------------------------------------------------------
        # 3.1 Active Order Count Deltas (buy, sell, none) - amount, pct, log
        # ----------------------------------------------------------------------
        
        for side in ["buy", "sell", "none"]:
            field = f"active_order_count_{side}"
            deltas[f"delta_{field}"] = _delta_amount_int32(
                current[field], previous[field]
            )
            deltas[f"delta_{field}_pct"] = _delta_pct_float64(
                current[field], previous[field]
            )
            deltas[f"delta_{field}_log"] = _delta_log_float64(
                current[field], previous[field]
            )
        
        # ----------------------------------------------------------------------
        # 3.2 Active Volume Deltas (buy, sell, none) - amount, pct, log
        # ----------------------------------------------------------------------
        
        for side in ["buy", "sell", "none"]:
            field = f"active_volume_{side}"
            deltas[f"delta_{field}"] = _delta_amount_int32(
                current[field], previous[field]
            )
            deltas[f"delta_{field}_pct"] = _delta_pct_float64(
                current[field], previous[field]
            )
            deltas[f"delta_{field}_log"] = _delta_log_float64(
                current[field], previous[field]
            )
        
        # ----------------------------------------------------------------------
        # 3.3 Active Imbalance Deltas (signed, abs) - amount only
        # ----------------------------------------------------------------------
        
        deltas["delta_active_imbalance_signed"] = _delta_amount_int64(
            current["active_imbalance_signed"], previous["active_imbalance_signed"]
        )
        deltas["delta_active_imbalance_abs"] = _delta_amount_int64(
            current["active_imbalance_abs"], previous["active_imbalance_abs"]
        )
        
        # ----------------------------------------------------------------------
        # 3.4 Active Ratio Deltas (signed_ratio, abs_ratio, buy_ratio) - amount only
        # ----------------------------------------------------------------------
        
        deltas["delta_active_imbalance_signed_ratio"] = _delta_amount_float64(
            current["active_imbalance_signed_ratio"], 
            previous["active_imbalance_signed_ratio"]
        )
        deltas["delta_active_imbalance_abs_ratio"] = _delta_amount_float64(
            current["active_imbalance_abs_ratio"], 
            previous["active_imbalance_abs_ratio"]
        )
        deltas["delta_active_imbalance_buy_ratio"] = _delta_amount_float64(
            current["active_imbalance_buy_ratio"], 
            previous["active_imbalance_buy_ratio"]
        )
        
        # ----------------------------------------------------------------------
        # 3.5 Active Link Function Transform Deltas - amount only
        # ----------------------------------------------------------------------
        
        deltas["delta_active_imbalance_signed_ratio_atanh"] = _delta_amount_float64(
            current["active_imbalance_signed_ratio_atanh"],
            previous["active_imbalance_signed_ratio_atanh"]
        )
        deltas["delta_active_imbalance_buy_ratio_logit"] = _delta_amount_float64(
            current["active_imbalance_buy_ratio_logit"],
            previous["active_imbalance_buy_ratio_logit"]
        )
        deltas["delta_active_imbalance_abs_ratio_logit"] = _delta_amount_float64(
            current["active_imbalance_abs_ratio_logit"],
            previous["active_imbalance_abs_ratio_logit"]
        )
        
        # ----------------------------------------------------------------------
        # 3.6 Active VWAP Deltas (buy, sell, none, spread, midpoint) - amount, pct, log
        # ----------------------------------------------------------------------
        
        active_vwap_fields = [
            "active_buy_vwap",
            "active_sell_vwap",
            "active_none_vwap",
            "active_spread_vwap",
            "active_midpoint_vwap"
        ]
        
        for field in active_vwap_fields:
            deltas[f"delta_{field}"] = _delta_amount_float64(
                current[field], previous[field]
            )
            deltas[f"delta_{field}_pct"] = _delta_pct_float64(
                current[field], previous[field]
            )
            deltas[f"delta_{field}_log"] = _delta_log_float64(
                current[field], previous[field]
            )
        
        # ----------------------------------------------------------------------
        # 3.7 Active Weighted Midpoint Deltas - amount, pct, log
        # ----------------------------------------------------------------------
        
        active_weighted_fields = [
            "active_mid_imbalance_weighted",
            "active_mid_flow_weighted",
            "active_mid_aggressor_weighted"
        ]
        
        for field in active_weighted_fields:
            deltas[f"delta_{field}"] = _delta_amount_float64(
                current[field], previous[field]
            )
            deltas[f"delta_{field}_pct"] = _delta_pct_float64(
                current[field], previous[field]
            )
            deltas[f"delta_{field}_log"] = _delta_log_float64(
                current[field], previous[field]
            )
        
        # ----------------------------------------------------------------------
        # 3.8 Active Price Range Deltas (min, max, range for buy/sell/none) - amount, pct, log
        # ----------------------------------------------------------------------
        
        for side in ["buy", "sell", "none"]:
            for metric in ["min", "max", "range"]:
                field = f"active_{side}_price_{metric}"
                deltas[f"delta_{field}"] = _delta_amount_float64(
                    current[field], previous[field]
                )
                deltas[f"delta_{field}_pct"] = _delta_pct_float64(
                    current[field], previous[field]
                )
                deltas[f"delta_{field}_log"] = _delta_log_float64(
                    current[field], previous[field]
                )
        
        # ----------------------------------------------------------------------
        # 3.9 Active Pace Deltas (buy, sell) - amount, pct, log
        # ----------------------------------------------------------------------
        
        for side in ["buy", "sell"]:
            field = f"active_{side}_pace"
            deltas[f"delta_{field}"] = _delta_amount_float64(
                current[field], previous[field]
            )
            deltas[f"delta_{field}_pct"] = _delta_pct_float64(
                current[field], previous[field]
            )
            deltas[f"delta_{field}_log"] = _delta_log_float64(
                current[field], previous[field]
            )
        
        # ----------------------------------------------------------------------
        # 3.10 Active Pace Log Transform Deltas (delta of base log field)
        # ----------------------------------------------------------------------
        
        for side in ["buy", "sell"]:
            field = f"active_{side}_pace_log"
            deltas[f"delta_{field}_base"] = _delta_amount_float64(
                current[field], previous[field]
            )
        
        # ----------------------------------------------------------------------
        # 3.11 Active N-Side Inferred Volume Deltas - amount, pct, log
        # ----------------------------------------------------------------------
        
        for side in ["buy", "sell"]:
            field = f"active_none_inferred_{side}_volume"
            deltas[f"delta_{field}"] = _delta_amount_float64(
                current[field], previous[field]
            )
            deltas[f"delta_{field}_pct"] = _delta_pct_float64(
                current[field], previous[field]
            )
            deltas[f"delta_{field}_log"] = _delta_log_float64(
                current[field], previous[field]
            )
        
        # ----------------------------------------------------------------------
        # 3.12 Active N-Side Inferred VWAP Deltas - amount, pct, log
        # ----------------------------------------------------------------------
        
        for side in ["buy", "sell"]:
            field = f"active_none_inferred_{side}_vwap"
            deltas[f"delta_{field}"] = _delta_amount_float64(
                current[field], previous[field]
            )
            deltas[f"delta_{field}_pct"] = _delta_pct_float64(
                current[field], previous[field]
            )
            deltas[f"delta_{field}_log"] = _delta_log_float64(
                current[field], previous[field]
            )
        
        # ========================================================================
        # 4. ADJUSTED DELTAS
        # ========================================================================
        
        # ----------------------------------------------------------------------
        # 4.1 Adjusted Volume Deltas (buy, sell) - amount, pct, log
        # ----------------------------------------------------------------------
        
        for side in ["buy", "sell"]:
            field = f"adjusted_volume_{side}"
            deltas[f"delta_{field}"] = _delta_amount_float64(
                current[field], previous[field]
            )
            deltas[f"delta_{field}_pct"] = _delta_pct_float64(
                current[field], previous[field]
            )
            deltas[f"delta_{field}_log"] = _delta_log_float64(
                current[field], previous[field]
            )
        
        # ----------------------------------------------------------------------
        # 4.2 Adjusted Imbalance Deltas (signed, abs) - amount only
        # ----------------------------------------------------------------------
        
        deltas["delta_adjusted_imbalance_signed"] = _delta_amount_float64(
            current["adjusted_imbalance_signed"], previous["adjusted_imbalance_signed"]
        )
        deltas["delta_adjusted_imbalance_abs"] = _delta_amount_float64(
            current["adjusted_imbalance_abs"], previous["adjusted_imbalance_abs"]
        )
        
        # ----------------------------------------------------------------------
        # 4.3 Adjusted Ratio Deltas - amount only
        # ----------------------------------------------------------------------
        
        deltas["delta_adjusted_imbalance_signed_ratio"] = _delta_amount_float64(
            current["adjusted_imbalance_signed_ratio"],
            previous["adjusted_imbalance_signed_ratio"]
        )
        deltas["delta_adjusted_imbalance_abs_ratio"] = _delta_amount_float64(
            current["adjusted_imbalance_abs_ratio"],
            previous["adjusted_imbalance_abs_ratio"]
        )
        deltas["delta_adjusted_imbalance_buy_ratio"] = _delta_amount_float64(
            current["adjusted_imbalance_buy_ratio"],
            previous["adjusted_imbalance_buy_ratio"]
        )
        
        # ----------------------------------------------------------------------
        # 4.4 Adjusted Link Function Transform Deltas - amount only
        # ----------------------------------------------------------------------
        
        deltas["delta_adjusted_imbalance_signed_ratio_atanh"] = _delta_amount_float64(
            current["adjusted_imbalance_signed_ratio_atanh"],
            previous["adjusted_imbalance_signed_ratio_atanh"]
        )
        deltas["delta_adjusted_imbalance_buy_ratio_logit"] = _delta_amount_float64(
            current["adjusted_imbalance_buy_ratio_logit"],
            previous["adjusted_imbalance_buy_ratio_logit"]
        )
        deltas["delta_adjusted_imbalance_abs_ratio_logit"] = _delta_amount_float64(
            current["adjusted_imbalance_abs_ratio_logit"],
            previous["adjusted_imbalance_abs_ratio_logit"]
        )
        
        # ----------------------------------------------------------------------
        # 4.5 Adjusted VWAP Deltas - amount, pct, log
        # ----------------------------------------------------------------------
        
        adjusted_vwap_fields = [
            "adjusted_buy_vwap",
            "adjusted_sell_vwap",
            "adjusted_spread_vwap",
            "adjusted_midpoint_vwap"
        ]
        
        for field in adjusted_vwap_fields:
            deltas[f"delta_{field}"] = _delta_amount_float64(
                current[field], previous[field]
            )
            deltas[f"delta_{field}_pct"] = _delta_pct_float64(
                current[field], previous[field]
            )
            deltas[f"delta_{field}_log"] = _delta_log_float64(
                current[field], previous[field]
            )
        
        # ----------------------------------------------------------------------
        # 4.6 Adjusted Weighted Midpoint Deltas - amount, pct, log
        # ----------------------------------------------------------------------
        
        adjusted_weighted_fields = [
            "adjusted_mid_imbalance_weighted",
            "adjusted_mid_flow_weighted",
            "adjusted_mid_aggressor_weighted"
        ]
        
        for field in adjusted_weighted_fields:
            deltas[f"delta_{field}"] = _delta_amount_float64(
                current[field], previous[field]
            )
            deltas[f"delta_{field}_pct"] = _delta_pct_float64(
                current[field], previous[field]
            )
            deltas[f"delta_{field}_log"] = _delta_log_float64(
                current[field], previous[field]
            )
        
        # ========================================================================
        # 5. PASSIVE DELTAS
        # ========================================================================
        
        # ----------------------------------------------------------------------
        # 5.1 Passive Midprice Deltas (second-order: delta of delta)
        # ----------------------------------------------------------------------
        
        # passive_midprice: amount, pct, log (first-order, comparing midprices)
        deltas["delta_passive_midprice"] = _delta_amount_float64(
            current["passive_midprice"], previous["passive_midprice"]
        )
        deltas["delta_passive_midprice_pct"] = _delta_pct_float64(
            current["passive_midprice"], previous["passive_midprice"]
        )
        deltas["delta_passive_midprice_log"] = _delta_log_float64(
            current["passive_midprice"], previous["passive_midprice"]
        )
        
        # Second-order deltas (acceleration): delta of the delta_price field
        deltas["delta_passive_midprice_delta_price"] = _delta_amount_float64(
            current["passive_midprice_delta_price"],
            previous["passive_midprice_delta_price"]
        )
        deltas["delta_passive_midprice_delta_price_pct"] = _delta_pct_float64(
            current["passive_midprice_delta_price"],
            previous["passive_midprice_delta_price"]
        )
        deltas["delta_passive_midprice_delta_price_log"] = _delta_log_float64(
            current["passive_midprice_delta_price"],
            previous["passive_midprice_delta_price"]
        )
        
        # Deltas for percent and log return fields
        deltas["delta_passive_midprice_delta_percent"] = _delta_amount_float64(
            current["passive_midprice_delta_percent"],
            previous["passive_midprice_delta_percent"]
        )
        deltas["delta_passive_midprice_delta_log"] = _delta_amount_float64(
            current["passive_midprice_delta_log"],
            previous["passive_midprice_delta_log"]
        )
        
        # ----------------------------------------------------------------------
        # 5.2 Passive Normalized/CDF Deltas - amount only
        # ----------------------------------------------------------------------
        
        deltas["delta_passive_midprice_delta_normalized"] = _delta_amount_float64(
            current["passive_midprice_delta_normalized"],
            previous["passive_midprice_delta_normalized"]
        )
        deltas["delta_passive_midprice_delta_cdf"] = _delta_amount_float64(
            current["passive_midprice_delta_cdf"],
            previous["passive_midprice_delta_cdf"]
        )
        
        # ----------------------------------------------------------------------
        # 5.3 Passive Volume Deltas (buy, sell) - amount, pct, log
        # ----------------------------------------------------------------------
        
        for side in ["buy", "sell"]:
            field = f"passive_{side}_volume"
            deltas[f"delta_{field}"] = _delta_amount_float64(
                current[field], previous[field]
            )
            deltas[f"delta_{field}_pct"] = _delta_pct_float64(
                current[field], previous[field]
            )
            deltas[f"delta_{field}_log"] = _delta_log_float64(
                current[field], previous[field]
            )
        
        # ----------------------------------------------------------------------
        # 5.4 Passive Imbalance Deltas (signed, abs) - amount only
        # ----------------------------------------------------------------------
        
        deltas["delta_passive_imbalance_signed"] = _delta_amount_float64(
            current["passive_imbalance_signed"], previous["passive_imbalance_signed"]
        )
        deltas["delta_passive_imbalance_abs"] = _delta_amount_float64(
            current["passive_imbalance_abs"], previous["passive_imbalance_abs"]
        )
        
        # ----------------------------------------------------------------------
        # 5.5 Passive Ratio Deltas - amount only
        # ----------------------------------------------------------------------
        
        deltas["delta_passive_imbalance_signed_ratio"] = _delta_amount_float64(
            current["passive_imbalance_signed_ratio"],
            previous["passive_imbalance_signed_ratio"]
        )
        deltas["delta_passive_imbalance_abs_ratio"] = _delta_amount_float64(
            current["passive_imbalance_abs_ratio"],
            previous["passive_imbalance_abs_ratio"]
        )
        deltas["delta_passive_imbalance_buy_ratio"] = _delta_amount_float64(
            current["passive_imbalance_buy_ratio"],
            previous["passive_imbalance_buy_ratio"]
        )
        
        # ----------------------------------------------------------------------
        # 5.6 Passive Link Function Transform Deltas - amount only
        # ----------------------------------------------------------------------
        
        deltas["delta_passive_imbalance_signed_ratio_atanh"] = _delta_amount_float64(
            current["passive_imbalance_signed_ratio_atanh"],
            previous["passive_imbalance_signed_ratio_atanh"]
        )
        deltas["delta_passive_imbalance_buy_ratio_logit"] = _delta_amount_float64(
            current["passive_imbalance_buy_ratio_logit"],
            previous["passive_imbalance_buy_ratio_logit"]
        )
        deltas["delta_passive_imbalance_abs_ratio_logit"] = _delta_amount_float64(
            current["passive_imbalance_abs_ratio_logit"],
            previous["passive_imbalance_abs_ratio_logit"]
        )
        
        # ========================================================================
        # 6. TEMPORAL DELTAS
        # ========================================================================
        
        # time_elapsed_ns: amount, pct, log
        deltas["delta_time_elapsed_ns"] = _delta_amount_int64(
            current["time_elapsed_ns"], previous["time_elapsed_ns"]
        )
        deltas["delta_time_elapsed_ns_pct"] = _delta_pct_float64(
            current["time_elapsed_ns"], previous["time_elapsed_ns"]
        )
        deltas["delta_time_elapsed_ns_log"] = _delta_log_float64(
            current["time_elapsed_ns"], previous["time_elapsed_ns"]
        )
        
        # pace_of_contracts_traded: amount, pct, log
        deltas["delta_pace_of_contracts_traded"] = _delta_amount_float64(
            current["pace_of_contracts_traded"], previous["pace_of_contracts_traded"]
        )
        deltas["delta_pace_of_contracts_traded_pct"] = _delta_pct_float64(
            current["pace_of_contracts_traded"], previous["pace_of_contracts_traded"]
        )
        deltas["delta_pace_of_contracts_traded_log"] = _delta_log_float64(
            current["pace_of_contracts_traded"], previous["pace_of_contracts_traded"]
        )
        
        # Deltas for the log-transformed base fields (delta of base log field)
        deltas["delta_time_elapsed_ns_log_base"] = _delta_amount_float64(
            current["time_elapsed_ns_log"], previous["time_elapsed_ns_log"]
        )
        deltas["delta_pace_of_contracts_traded_log_base"] = _delta_amount_float64(
            current["pace_of_contracts_traded_log"], previous["pace_of_contracts_traded_log"]
        )
        
        # ========================================================================
        # 7. DIVERGENCE DELTAS
        # ========================================================================
        
        divergence_fields = [
            "divergence_buy_volume",
            "divergence_sell_volume",
            "divergence_imbalance_signed",
            "divergence_imbalance_signed_ratio",
            "divergence_buy_ratio"
        ]
        
        for field in divergence_fields:
            deltas[f"delta_{field}"] = _delta_amount_float64(
                current[field], previous[field]
            )
        
        # ========================================================================
        # 8. DERIVED INDICATOR
        # ========================================================================
        
        # Boolean flag: True if active midpoint moved up, False otherwise
        # Handles NaN values explicitly
        prev_midpoint = np.float64(previous["active_midpoint_vwap"])
        curr_midpoint = np.float64(current["active_midpoint_vwap"])
        
        if np.isnan(prev_midpoint) or np.isnan(curr_midpoint):
            deltas["derived_price_direction_positive"] = np.bool_(False)
        else:
            deltas["derived_price_direction_positive"] = np.bool_(curr_midpoint > prev_midpoint)
        
        return deltas




    def _get_empty_statistics(self) -> Dict[str, Any]:
        """
        Return statistics dictionary with all *non-delta* values set to appropriate
        empty values. Must match models.VOLUMEBAR_DTYPE for non-delta fields only.

        Initialization Rules (per reference guide Edge Case Fill column):
        - Counts/volumes (uint32, int64): 0
        - Boolean flags: False
        - Timestamps (uint64): 0
        - Ratios requiring division: NaN (undefined when denominator = 0)
        - VWAPs/prices requiring trades: NaN (undefined when no trades)
        - Link function transforms: NaN (undefined when input ratio is NaN)
        - Pace metrics: NaN (undefined when time_elapsed_ns = 0)
        - Divergence metrics: NaN (undefined when component is NaN)

        NOTE: Per spec, any keys prefixed with 'derived_' or 'delta_'
        belong in _get_empty_deltas(), not here.
        """
        _NAN = np.float64(np.nan)
        
        return {
            # ================================================================
            # Configuration & Identifiers
            # ================================================================
            "id": np.uint32(0),
            "bar_volume_size": np.uint32(0),

            # ================================================================
            # Metadata Fields
            # ================================================================
            "gap_return": np.bool_(False),
            "max_time_gap_ns": np.uint64(0),
            "contains_oversized_order": np.bool_(False),
            "has_resized": np.bool_(False),

            # ================================================================
            # Order & Volume Statistics (Ambiguous/Total)
            # ================================================================
            "order_count": np.uint32(0),
            "order_splits": np.uint32(0),
            "volume_total": np.uint32(0),
            "bar_complete": np.bool_(False),

            # ================================================================
            # Active Order Counts
            # ================================================================
            "active_order_count_buy": np.uint32(0),
            "active_order_count_sell": np.uint32(0),
            "active_order_count_none": np.uint32(0),

            # ================================================================
            # Active Volumes
            # ================================================================
            "active_volume_buy": np.uint32(0),
            "active_volume_sell": np.uint32(0),
            "active_volume_none": np.uint32(0),

            # ================================================================
            # Active Imbalance Metrics
            # ================================================================
            "active_imbalance_signed": np.int64(0),
            "active_imbalance_abs": np.uint64(0),
            "active_imbalance_signed_ratio": _NAN,        # NaN if volume_total = 0
            "active_imbalance_abs_ratio": _NAN,           # NaN if volume_total = 0
            "active_imbalance_buy_ratio": _NAN,           # NaN if volume_total = 0

            # ================================================================
            # Active Link Function Transforms
            # ================================================================
            "active_imbalance_signed_ratio_atanh": _NAN,  # NaN if ratio is NaN
            "active_imbalance_buy_ratio_logit": _NAN,     # NaN if ratio is NaN
            "active_imbalance_abs_ratio_logit": _NAN,     # NaN if ratio is NaN

            # ================================================================
            # Active Price Metrics (VWAP)
            # ================================================================
            "active_buy_vwap": _NAN,                      # NaN if no buy trades
            "active_sell_vwap": _NAN,                     # NaN if no sell trades
            "active_none_vwap": _NAN,                     # NaN if no N-side trades
            "active_spread_vwap": _NAN,                   # NaN if either VWAP is NaN
            "active_midpoint_vwap": _NAN,                 # NaN if either VWAP is NaN

            # ================================================================
            # Active Weighted Midpoints
            # ================================================================
            "active_mid_imbalance_weighted": _NAN,        # NaN if midpoint/spread is NaN
            "active_mid_flow_weighted": _NAN,             # NaN if no A/B trades
            "active_mid_aggressor_weighted": _NAN,        # NaN if no A/B trades

            # ================================================================
            # Active Price Range Metrics
            # ================================================================
            "active_buy_price_min": _NAN,                 # NaN if no buy trades
            "active_buy_price_max": _NAN,                 # NaN if no buy trades
            "active_buy_price_range": _NAN,               # NaN if no buy trades
            "active_sell_price_min": _NAN,                # NaN if no sell trades
            "active_sell_price_max": _NAN,                # NaN if no sell trades
            "active_sell_price_range": _NAN,              # NaN if no sell trades
            "active_none_price_min": _NAN,                # NaN if no N-side trades
            "active_none_price_max": _NAN,                # NaN if no N-side trades
            "active_none_price_range": _NAN,              # NaN if no N-side trades

            # ================================================================
            # Active Pace Metrics
            # ================================================================
            "active_buy_pace": _NAN,                      # NaN if time_elapsed_ns = 0
            "active_sell_pace": _NAN,                     # NaN if time_elapsed_ns = 0
            "active_buy_pace_log": _NAN,                  # NaN if pace <= 0 or NaN
            "active_sell_pace_log": _NAN,                 # NaN if pace <= 0 or NaN

            # ================================================================
            # Active N-Side Inferred Classification
            # ================================================================
            "active_none_inferred_buy_volume": np.float64(0.0),   # 0.0 if no N-side trades
            "active_none_inferred_sell_volume": np.float64(0.0),  # 0.0 if no N-side trades
            "active_none_inferred_buy_vwap": _NAN,        # NaN if no inferred buy volume
            "active_none_inferred_sell_vwap": _NAN,       # NaN if no inferred sell volume

            # ================================================================
            # Adjusted Volumes (Active + N-Side Inferred)
            # ================================================================
            "adjusted_volume_buy": np.float64(0.0),       # 0.0 if both components are 0
            "adjusted_volume_sell": np.float64(0.0),      # 0.0 if both components are 0

            # ================================================================
            # Adjusted Imbalance Metrics
            # ================================================================
            "adjusted_imbalance_signed": np.float64(0.0), # 0.0
            "adjusted_imbalance_abs": np.float64(0.0),    # 0.0
            "adjusted_imbalance_signed_ratio": _NAN,      # NaN if volume_total = 0
            "adjusted_imbalance_abs_ratio": _NAN,         # NaN if volume_total = 0
            "adjusted_imbalance_buy_ratio": _NAN,         # NaN if volume_total = 0

            # ================================================================
            # Adjusted Link Function Transforms
            # ================================================================
            "adjusted_imbalance_signed_ratio_atanh": _NAN,  # NaN if ratio is NaN
            "adjusted_imbalance_buy_ratio_logit": _NAN,     # NaN if ratio is NaN
            "adjusted_imbalance_abs_ratio_logit": _NAN,     # NaN if ratio is NaN

            # ================================================================
            # Adjusted Price Metrics (VWAP)
            # ================================================================
            "adjusted_buy_vwap": _NAN,                    # NaN if adjusted_volume_buy = 0
            "adjusted_sell_vwap": _NAN,                   # NaN if adjusted_volume_sell = 0
            "adjusted_spread_vwap": _NAN,                 # NaN if either VWAP is NaN
            "adjusted_midpoint_vwap": _NAN,               # NaN if either VWAP is NaN

            # ================================================================
            # Adjusted Weighted Midpoints
            # ================================================================
            "adjusted_mid_imbalance_weighted": _NAN,      # NaN if midpoint/spread is NaN
            "adjusted_mid_flow_weighted": _NAN,           # NaN if total adjusted volume = 0
            "adjusted_mid_aggressor_weighted": _NAN,      # NaN if total adjusted volume = 0

            # ================================================================
            # Passive Midprice Metrics
            # ================================================================
            "passive_midprice": _NAN,                     # NaN if no trades
            "passive_midprice_delta_price": _NAN,         # NaN if first bar
            "passive_midprice_delta_percent": _NAN,       # NaN if first bar or prev = 0
            "passive_midprice_delta_log": _NAN,           # NaN if first bar or prev <= 0

            # ================================================================
            # Passive Normalized and CDF Metrics
            # ================================================================
            "passive_midprice_delta_normalized": _NAN,    # NaN if delta is NaN or sigma = 0
            "passive_midprice_delta_cdf": _NAN,           # NaN if normalized is NaN

            # ================================================================
            # Passive Volume Classification
            # ================================================================
            "passive_buy_volume": np.float64(0.0),        # 0.0 if CDF is NaN
            "passive_sell_volume": np.float64(0.0),       # 0.0 if CDF is NaN

            # ================================================================
            # Passive Imbalance Metrics
            # ================================================================
            "passive_imbalance_signed": np.float64(0.0),  # 0.0 if volumes are 0
            "passive_imbalance_abs": np.float64(0.0),     # 0.0
            "passive_imbalance_signed_ratio": _NAN,       # NaN if volume_total = 0
            "passive_imbalance_abs_ratio": _NAN,          # NaN if volume_total = 0
            "passive_imbalance_buy_ratio": _NAN,          # NaN if volume_total = 0

            # ================================================================
            # Passive Link Function Transforms
            # ================================================================
            "passive_imbalance_signed_ratio_atanh": _NAN, # NaN if ratio is NaN
            "passive_imbalance_buy_ratio_logit": _NAN,    # NaN if ratio is NaN
            "passive_imbalance_abs_ratio_logit": _NAN,    # NaN if ratio is NaN

            # ================================================================
            # Temporal Metrics
            # ================================================================
            "start_ts_ns": np.uint64(0),                  # 0 if no timestamps
            "end_ts_ns": np.uint64(0),                    # 0 if no timestamps
            "time_elapsed_ns": np.uint64(0),              # 0
            "pace_of_contracts_traded": _NAN,             # NaN if time_elapsed_ns = 0
            "time_elapsed_ns_log": _NAN,                  # NaN if time_elapsed_ns = 0
            "pace_of_contracts_traded_log": _NAN,         # NaN if pace <= 0 or NaN

            # ================================================================
            # Instrument Tracking
            # ================================================================
            "contract_roll": np.bool_(False),
            "latest_instrument_id": np.uint32(0),         # 0 if no orders

            # ================================================================
            # Divergence Metrics (Active vs Passive)
            # ================================================================
            "divergence_buy_volume": _NAN,                # NaN if either is NaN
            "divergence_sell_volume": _NAN,               # NaN if either is NaN
            "divergence_imbalance_signed": _NAN,          # NaN if either is NaN
            "divergence_imbalance_signed_ratio": _NAN,    # NaN if either is NaN
            "divergence_buy_ratio": _NAN,                 # NaN if either is NaN
        }




    def _get_empty_deltas(self) -> Dict[str, Any]:
        """
        Return dictionary with ONLY 'derived_' and 'delta_' keys initialized.
        Must match models.VOLUMEBAR_DTYPE for all delta/derived fields.

        ALL delta fields are NaN for empty/first bar (no previous bar to compare).
        Boolean derived fields are False.
        """
        _NAN = np.float64(np.nan)
        # For int32/int64 deltas, we need a sentinel. Using 0 cast pattern
        # but note: in practice these should be NaN-like. Since numpy int
        # types cannot hold NaN, we use 0 as the sentinel value.
        # The calling code should check bar index before using these values.
        
        return {
            # ================================================================
            # Derived Indicator
            # ================================================================
            "derived_price_direction_positive": np.bool_(False),

            # ================================================================
            # Delta Calculations - Ambiguous/Total Deltas
            # ================================================================
            "delta_order_count": np.int32(0),             # NaN semantic (int cannot hold NaN)
            "delta_order_count_pct": _NAN,
            "delta_order_count_log": _NAN,
            "delta_order_splits": np.int32(0),            # NaN semantic (int cannot hold NaN)
            "delta_volume_total": np.int32(0),            # NaN semantic (int cannot hold NaN)
            "delta_volume_total_pct": _NAN,
            "delta_volume_total_log": _NAN,

            # ================================================================
            # Delta Calculations - Active Order Count Deltas
            # ================================================================
            "delta_active_order_count_buy": np.int32(0),  # NaN semantic
            "delta_active_order_count_buy_pct": _NAN,
            "delta_active_order_count_buy_log": _NAN,
            "delta_active_order_count_sell": np.int32(0), # NaN semantic
            "delta_active_order_count_sell_pct": _NAN,
            "delta_active_order_count_sell_log": _NAN,
            "delta_active_order_count_none": np.int32(0), # NaN semantic
            "delta_active_order_count_none_pct": _NAN,
            "delta_active_order_count_none_log": _NAN,

            # ================================================================
            # Delta Calculations - Active Volume Deltas
            # ================================================================
            "delta_active_volume_buy": np.int32(0),       # NaN semantic
            "delta_active_volume_buy_pct": _NAN,
            "delta_active_volume_buy_log": _NAN,
            "delta_active_volume_sell": np.int32(0),      # NaN semantic
            "delta_active_volume_sell_pct": _NAN,
            "delta_active_volume_sell_log": _NAN,
            "delta_active_volume_none": np.int32(0),      # NaN semantic
            "delta_active_volume_none_pct": _NAN,
            "delta_active_volume_none_log": _NAN,

            # ================================================================
            # Delta Calculations - Active Imbalance Deltas
            # ================================================================
            "delta_active_imbalance_signed": np.int64(0), # NaN semantic
            "delta_active_imbalance_abs": np.int64(0),    # NaN semantic
            "delta_active_imbalance_signed_ratio": _NAN,
            "delta_active_imbalance_abs_ratio": _NAN,
            "delta_active_imbalance_buy_ratio": _NAN,

            # ================================================================
            # Delta Calculations - Active Link Function Transform Deltas
            # ================================================================
            "delta_active_imbalance_signed_ratio_atanh": _NAN,
            "delta_active_imbalance_buy_ratio_logit": _NAN,
            "delta_active_imbalance_abs_ratio_logit": _NAN,

            # ================================================================
            # Delta Calculations - Active Price Deltas (VWAP)
            # ================================================================
            "delta_active_buy_vwap": _NAN,
            "delta_active_buy_vwap_pct": _NAN,
            "delta_active_buy_vwap_log": _NAN,
            "delta_active_sell_vwap": _NAN,
            "delta_active_sell_vwap_pct": _NAN,
            "delta_active_sell_vwap_log": _NAN,
            "delta_active_none_vwap": _NAN,
            "delta_active_none_vwap_pct": _NAN,
            "delta_active_none_vwap_log": _NAN,
            "delta_active_spread_vwap": _NAN,
            "delta_active_spread_vwap_pct": _NAN,
            "delta_active_spread_vwap_log": _NAN,
            "delta_active_midpoint_vwap": _NAN,
            "delta_active_midpoint_vwap_pct": _NAN,
            "delta_active_midpoint_vwap_log": _NAN,

            # ================================================================
            # Delta Calculations - Active Weighted Midpoint Deltas
            # ================================================================
            "delta_active_mid_imbalance_weighted": _NAN,
            "delta_active_mid_imbalance_weighted_pct": _NAN,
            "delta_active_mid_imbalance_weighted_log": _NAN,
            "delta_active_mid_flow_weighted": _NAN,
            "delta_active_mid_flow_weighted_pct": _NAN,
            "delta_active_mid_flow_weighted_log": _NAN,
            "delta_active_mid_aggressor_weighted": _NAN,
            "delta_active_mid_aggressor_weighted_pct": _NAN,
            "delta_active_mid_aggressor_weighted_log": _NAN,

            # ================================================================
            # Delta Calculations - Active Price Range Deltas
            # ================================================================
            "delta_active_buy_price_min": _NAN,
            "delta_active_buy_price_min_pct": _NAN,
            "delta_active_buy_price_min_log": _NAN,
            "delta_active_buy_price_max": _NAN,
            "delta_active_buy_price_max_pct": _NAN,
            "delta_active_buy_price_max_log": _NAN,
            "delta_active_buy_price_range": _NAN,
            "delta_active_buy_price_range_pct": _NAN,
            "delta_active_buy_price_range_log": _NAN,
            "delta_active_sell_price_min": _NAN,
            "delta_active_sell_price_min_pct": _NAN,
            "delta_active_sell_price_min_log": _NAN,
            "delta_active_sell_price_max": _NAN,
            "delta_active_sell_price_max_pct": _NAN,
            "delta_active_sell_price_max_log": _NAN,
            "delta_active_sell_price_range": _NAN,
            "delta_active_sell_price_range_pct": _NAN,
            "delta_active_sell_price_range_log": _NAN,
            "delta_active_none_price_min": _NAN,
            "delta_active_none_price_min_pct": _NAN,
            "delta_active_none_price_min_log": _NAN,
            "delta_active_none_price_max": _NAN,
            "delta_active_none_price_max_pct": _NAN,
            "delta_active_none_price_max_log": _NAN,
            "delta_active_none_price_range": _NAN,
            "delta_active_none_price_range_pct": _NAN,
            "delta_active_none_price_range_log": _NAN,

            # ================================================================
            # Delta Calculations - Active Pace Deltas
            # ================================================================
            "delta_active_buy_pace": _NAN,
            "delta_active_buy_pace_pct": _NAN,
            "delta_active_buy_pace_log": _NAN,
            "delta_active_sell_pace": _NAN,
            "delta_active_sell_pace_pct": _NAN,
            "delta_active_sell_pace_log": _NAN,

            # ================================================================
            # Delta Calculations - Active Pace Log Transform Deltas
            # ================================================================
            "delta_active_buy_pace_log_base": _NAN,
            "delta_active_sell_pace_log_base": _NAN,

            # ================================================================
            # Delta Calculations - Active N-Side Inferred Volume Deltas
            # ================================================================
            "delta_active_none_inferred_buy_volume": _NAN,
            "delta_active_none_inferred_buy_volume_pct": _NAN,
            "delta_active_none_inferred_buy_volume_log": _NAN,
            "delta_active_none_inferred_sell_volume": _NAN,
            "delta_active_none_inferred_sell_volume_pct": _NAN,
            "delta_active_none_inferred_sell_volume_log": _NAN,

            # ================================================================
            # Delta Calculations - Active N-Side Inferred VWAP Deltas
            # ================================================================
            "delta_active_none_inferred_buy_vwap": _NAN,
            "delta_active_none_inferred_buy_vwap_pct": _NAN,
            "delta_active_none_inferred_buy_vwap_log": _NAN,
            "delta_active_none_inferred_sell_vwap": _NAN,
            "delta_active_none_inferred_sell_vwap_pct": _NAN,
            "delta_active_none_inferred_sell_vwap_log": _NAN,

            # ================================================================
            # Delta Calculations - Adjusted Volume Deltas
            # ================================================================
            "delta_adjusted_volume_buy": _NAN,
            "delta_adjusted_volume_buy_pct": _NAN,
            "delta_adjusted_volume_buy_log": _NAN,
            "delta_adjusted_volume_sell": _NAN,
            "delta_adjusted_volume_sell_pct": _NAN,
            "delta_adjusted_volume_sell_log": _NAN,

            # ================================================================
            # Delta Calculations - Adjusted Imbalance Deltas
            # ================================================================
            "delta_adjusted_imbalance_signed": _NAN,
            "delta_adjusted_imbalance_abs": _NAN,
            "delta_adjusted_imbalance_signed_ratio": _NAN,
            "delta_adjusted_imbalance_abs_ratio": _NAN,
            "delta_adjusted_imbalance_buy_ratio": _NAN,

            # ================================================================
            # Delta Calculations - Adjusted Link Function Transform Deltas
            # ================================================================
            "delta_adjusted_imbalance_signed_ratio_atanh": _NAN,
            "delta_adjusted_imbalance_buy_ratio_logit": _NAN,
            "delta_adjusted_imbalance_abs_ratio_logit": _NAN,

            # ================================================================
            # Delta Calculations - Adjusted Price Deltas (VWAP)
            # ================================================================
            "delta_adjusted_buy_vwap": _NAN,
            "delta_adjusted_buy_vwap_pct": _NAN,
            "delta_adjusted_buy_vwap_log": _NAN,
            "delta_adjusted_sell_vwap": _NAN,
            "delta_adjusted_sell_vwap_pct": _NAN,
            "delta_adjusted_sell_vwap_log": _NAN,
            "delta_adjusted_spread_vwap": _NAN,
            "delta_adjusted_spread_vwap_pct": _NAN,
            "delta_adjusted_spread_vwap_log": _NAN,
            "delta_adjusted_midpoint_vwap": _NAN,
            "delta_adjusted_midpoint_vwap_pct": _NAN,
            "delta_adjusted_midpoint_vwap_log": _NAN,

            # ================================================================
            # Delta Calculations - Adjusted Weighted Midpoint Deltas
            # ================================================================
            "delta_adjusted_mid_imbalance_weighted": _NAN,
            "delta_adjusted_mid_imbalance_weighted_pct": _NAN,
            "delta_adjusted_mid_imbalance_weighted_log": _NAN,
            "delta_adjusted_mid_flow_weighted": _NAN,
            "delta_adjusted_mid_flow_weighted_pct": _NAN,
            "delta_adjusted_mid_flow_weighted_log": _NAN,
            "delta_adjusted_mid_aggressor_weighted": _NAN,
            "delta_adjusted_mid_aggressor_weighted_pct": _NAN,
            "delta_adjusted_mid_aggressor_weighted_log": _NAN,

            # ================================================================
            # Delta Calculations - Passive Midprice First-Order Deltas
            # ================================================================
            "delta_passive_midprice": _NAN,
            "delta_passive_midprice_pct": _NAN,
            "delta_passive_midprice_log": _NAN,

            # ================================================================
            # Delta Calculations - Passive Midprice Second-Order Deltas
            # ================================================================
            "delta_passive_midprice_delta_price": _NAN,
            "delta_passive_midprice_delta_price_pct": _NAN,
            "delta_passive_midprice_delta_price_log": _NAN,
            "delta_passive_midprice_delta_percent": _NAN,
            "delta_passive_midprice_delta_log": _NAN,

            # ================================================================
            # Delta Calculations - Passive Normalized/CDF Deltas
            # ================================================================
            "delta_passive_midprice_delta_normalized": _NAN,
            "delta_passive_midprice_delta_cdf": _NAN,

            # ================================================================
            # Delta Calculations - Passive Volume Deltas
            # ================================================================
            "delta_passive_buy_volume": _NAN,
            "delta_passive_buy_volume_pct": _NAN,
            "delta_passive_buy_volume_log": _NAN,
            "delta_passive_sell_volume": _NAN,
            "delta_passive_sell_volume_pct": _NAN,
            "delta_passive_sell_volume_log": _NAN,

            # ================================================================
            # Delta Calculations - Passive Imbalance Deltas
            # ================================================================
            "delta_passive_imbalance_signed": _NAN,
            "delta_passive_imbalance_abs": _NAN,
            "delta_passive_imbalance_signed_ratio": _NAN,
            "delta_passive_imbalance_abs_ratio": _NAN,
            "delta_passive_imbalance_buy_ratio": _NAN,

            # ================================================================
            # Delta Calculations - Passive Link Function Transform Deltas
            # ================================================================
            "delta_passive_imbalance_signed_ratio_atanh": _NAN,
            "delta_passive_imbalance_buy_ratio_logit": _NAN,
            "delta_passive_imbalance_abs_ratio_logit": _NAN,

            # ================================================================
            # Delta Calculations - Temporal Deltas
            # ================================================================
            "delta_time_elapsed_ns": np.int64(0),         # NaN semantic (int cannot hold NaN)
            "delta_time_elapsed_ns_pct": _NAN,
            "delta_time_elapsed_ns_log": _NAN,
            "delta_pace_of_contracts_traded": _NAN,
            "delta_pace_of_contracts_traded_pct": _NAN,
            "delta_pace_of_contracts_traded_log": _NAN,

            # ================================================================
            # Delta Calculations - Temporal Log Base Deltas
            # ================================================================
            "delta_time_elapsed_ns_log_base": _NAN,
            "delta_pace_of_contracts_traded_log_base": _NAN,

            # ================================================================
            # Delta Calculations - Divergence Deltas
            # ================================================================
            "delta_divergence_buy_volume": _NAN,
            "delta_divergence_sell_volume": _NAN,
            "delta_divergence_imbalance_signed": _NAN,
            "delta_divergence_imbalance_signed_ratio": _NAN,
            "delta_divergence_buy_ratio": _NAN,
        }




    def to_pandas(self) -> pd.DataFrame:
        """

        Convert stored orders to DataFrame for inspection.
        Combines all aggressor sides, sorted by timestamp.
        
        Returns:
            DataFrame with columns:
            - ts_recv: Order timestamp
            - price: Order price
            - size: Original order size
            - size_contributed: Amount added to this bar
            - side: "A" (sell), "B" (buy), or "N" (none)
            - is_continuation: Whether order was split
            - sequence: Message sequence number
            - instrument_id: Contract identifier

        """

        # Create sell aggressor DataFrame with all order attributes
        sell_aggressor_df: pd.DataFrame = pd.DataFrame({
            "ts_recv": self._a_sell_aggressor_timestamps[:self._a_sell_aggressor_idx],
            "price": self._a_sell_aggressor_prices[:self._a_sell_aggressor_idx],
            "size_contributed": self._a_sell_aggressor_size_contributed[:self._a_sell_aggressor_idx],
            "side": "A",
            "is_continuation": self._ask_is_continuation[:self._a_sell_aggressor_idx],
            "sequence": self._a_sell_aggressor_sequences[:self._a_sell_aggressor_idx],
            "instrument_id": self._a_sell_aggressor_instrument_ids[:self._a_sell_aggressor_idx]
        })
        
        # Create buy aggressor DataFrame with all order attributes
        buy_aggressor_df: pd.DataFrame = pd.DataFrame({
            "ts_recv": self._b_buy_aggressor_timestamps[:self._b_buy_aggressor_idx],
            "price": self._b_buy_aggressor_prices[:self._b_buy_aggressor_idx],
            "size_contributed": self._b_buy_aggressor_size_contributed[:self._b_buy_aggressor_idx],
            "side": "B",
            "is_continuation": self._b_buy_aggressor_is_continuation[:self._b_buy_aggressor_idx],
            "sequence": self._b_buy_aggressor_sequences[:self._b_buy_aggressor_idx],
            "instrument_id": self._b_buy_aggressor_instrument_ids[:self._b_buy_aggressor_idx]
        })
        
        # Create none aggressor DataFrame with all order attributes
        none_aggressor_df: pd.DataFrame = pd.DataFrame({
            "ts_recv": self._n_none_aggressor_timestamps[:self._n_none_aggressor_idx],
            "price": self._n_none_aggressor_prices[:self._n_none_aggressor_idx],
            "size_contributed": self._n_none_aggressor_size_contributed[:self._n_none_aggressor_idx],
            "side": "N",
            "is_continuation": self._n_none_aggressor_is_continuation[:self._n_none_aggressor_idx],
            "sequence": self._n_none_aggressor_sequences[:self._n_none_aggressor_idx],
            "instrument_id": self._n_none_aggressor_instrument_ids[:self._n_none_aggressor_idx]
        })
        
        # Combine all sides and sort chronologically by timestamp
        combined: pd.DataFrame = pd.concat([sell_aggressor_df, buy_aggressor_df, none_aggressor_df], ignore_index=True)
        return combined.sort_values("ts_recv").reset_index(drop=True) 




    def to_dict(self) -> Dict[str, Any]:
        """

        Convert bar to dictionary using calculated statistics.
        For backward compatibility and testing.
        
        Returns:
            Dictionary containing:
            - bar_volume_size: Configured capacity
            - All raw statistics from _calculate_statistics()
            
        Note: Does not include delta statistics as this method
              doesn"t have access to previous bar data
        
        """
        # Calculate all raw statistics from arrays
        stats: Dict[str, Any] = self._calculate_statistics()
        
        # Include configuration for context
        return {
            "bar_volume_size": self.bar_volume_size,
            **stats  # Unpack all calculated statistics
        }




    def test__row_full_schema(self, full: Dict[str, Any], empty: bool, show_valid_messages: bool = False) -> None:

        source = "EMPTY DATA" if empty else "ACTUAL DATA"
        error_message = f"Full schema validation failed for << {source} >>"
        assert HelperFunctions.validate_full_schema(
            stats=full, 
            schema=VOLUMEBAR_DTYPE,
            show_valid_messages=show_valid_messages,
        ), error_message




    def test__row_statistics_schema(self, stats: Dict[str, Any], empty: bool, show_valid_messages: bool = False) -> None:

        source = "EMPTY DATA" if empty else "ACTUAL DATA"
        error_message = f"Calculations (non-delta) schema validation failed for << {source} >>"
        assert HelperFunctions.validate_statistics_schema(
            stats=stats, 
            schema=VOLUMEBAR_DTYPE,
            show_valid_messages=show_valid_messages,
        ), error_message




    def test__row_deltas_schema(self, deltas: Dict[str, Any], empty: bool, show_valid_messages: bool = False) -> None:
        
        source = "EMPTY DATA" if empty else "ACTUAL DATA"
        error_message = f"Delta schema validation failed for << {source} >>"
        assert HelperFunctions.validate_delta_schema(
            stats=deltas, 
            schema=VOLUMEBAR_DTYPE,
            show_valid_messages=show_valid_messages,
        ), error_message



