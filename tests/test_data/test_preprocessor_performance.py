"""
Performance tests for data preprocessor
"""

import pytest
import pandas as pd
from pathlib import Path
import time

from src.data.preprocessor import DataPreprocessor


@pytest.mark.performance
class TestPreprocessorPerformance:
    """Test preprocessor performance meets targets"""
    
    @pytest.fixture
    def preprocessor(self):
        """Create preprocessor for real data"""
        raw_dir = Path("data/raw/minute_aggs/by_symbol")
        if not raw_dir.exists():
            pytest.skip("Real data not available")
        
        return DataPreprocessor(
            raw_data_dir=raw_dir,
            processed_data_dir=Path("data/processed"),
            cache_dir=Path("data/cache")
        )
    
    def test_process_one_year_performance(self, preprocessor):
        """Test processing 1 year of minute data completes in <1 second"""
        # Process full year of SPY data
        start_time = time.perf_counter()
        
        # Process all 12 months
        months = [f"2024_{i:02d}" for i in range(1, 13)]
        processed_df = preprocessor.process("SPY", months=months)
        
        end_time = time.perf_counter()
        processing_time = end_time - start_time
        
        # Check performance target
        assert processing_time < 1.0, f"Processing took {processing_time:.2f}s, target is <1s"
        
        # Verify data is reasonable
        assert len(processed_df) > 90000  # ~98k bars per year
        assert processed_df['close'].mean() > 0
        
        print(f"\nPerformance Results:")
        print(f"- Processed {len(processed_df)} bars in {processing_time:.3f}s")
        print(f"- Rate: {len(processed_df) / processing_time:.0f} bars/second")