"""
Data source type definitions for the volume bar processing system.
"""


class DataSourceType:
    """Enumeration of supported data source types."""
    
    MDP2_HISTORICAL = "mdp2_historical"
    MDP3_HISTORICAL = "mdp3_historical"
    MDP3_LIVE_CME = "live_databento_cme_futures"
    LIVE_SPOT_FX = "live_spot_fx_transactions"
    
    @classmethod
    def requires_preprocessing(cls, data_type: str) -> bool:
        """Check if data type requires time-bar preprocessing."""
        return data_type in {cls.MDP2_HISTORICAL, cls.LIVE_SPOT_FX}




class AvailableDatasets:
    """Available Datasets, returns request ID"""

    FULL_YEAR_2024__CME_JPY_V: str = "GLBX-20251028-HAE9P7SP3U"
    FULL_YEAR_2023__CME_JPY_V: str = "GLBX-20251116-MJ5JKQA9A4"
    FULL_YEAR_2022__CME_JPY_V: str = "GLBX-20251115-DJHUGKSLWB"

    # 1 Day TBBO Data ( *TBBO* 6J.v 1 day "2022-11-10")
    TBBO__FULL_DAY_2022_11_10__CME_JPY_V: str = "GLBX-20251117-8KDC7XCEYW"

    # 1 Day TBBO Data ( *TRADES* 6J.v 1 day "2022-11-10") - MISTAKENLY BOUGHT
    TRADES__FULL_DAY_2022_11_10__CME_JPY_V: str = "GLBX-20251117-TWBJGCYALB"
    
    # Continuous Contract (Not Especially Useful)
    FULL_YEAR_2022__CME_JPY_C: str = "GLBX-20251027-AG9BYWQJ6L"



