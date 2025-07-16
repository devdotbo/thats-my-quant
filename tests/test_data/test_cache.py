"""
Tests for cache management system
"""

import pytest
import shutil
import json
from pathlib import Path
import time

from src.data.cache import CacheManager


class TestCacheManager:
    """Test cache management functionality"""
    
    @pytest.fixture
    def cache_dir(self, tmp_path):
        """Create temporary cache directory"""
        cache_path = tmp_path / "test_cache"
        cache_path.mkdir(exist_ok=True)
        yield cache_path
        # Cleanup handled by tmp_path
    
    @pytest.fixture
    def cache_manager(self, cache_dir):
        """Create cache manager instance"""
        # Small size limit for testing (10MB)
        return CacheManager(cache_dir=cache_dir, max_size_gb=0.01)
    
    @pytest.fixture
    def test_files(self, tmp_path):
        """Create test files of various sizes"""
        files_dir = tmp_path / "test_files"
        files_dir.mkdir(exist_ok=True)
        
        files = {}
        
        # Small file (1KB)
        small_file = files_dir / "small.txt"
        small_file.write_text("x" * 1024)
        files['small'] = small_file
        
        # Medium file (100KB)
        medium_file = files_dir / "medium.txt"
        medium_file.write_text("x" * (100 * 1024))
        files['medium'] = medium_file
        
        # Large file (1MB)
        large_file = files_dir / "large.txt"
        large_file.write_text("x" * (1024 * 1024))
        files['large'] = large_file
        
        return files
    
    def test_cache_initialization(self, cache_manager, cache_dir):
        """Test cache manager initialization"""
        assert cache_manager.cache_dir == cache_dir
        assert cache_manager.max_size_bytes == int(0.01 * 1024**3)
        assert cache_manager.metadata_file.exists()
        
        # Check initial metadata
        stats = cache_manager.get_statistics()
        assert stats['total_files'] == 0
        assert stats['total_size_bytes'] == 0
    
    def test_cache_file(self, cache_manager, test_files):
        """Test caching a file"""
        # Cache small file
        cached_path = cache_manager.cache_file(
            test_files['small'], 
            'small_key',
            category='test'
        )
        
        assert cached_path.exists()
        assert cached_path.parent.name == 'test'
        assert cached_path.name == 'small_key'
        
        # Check metadata
        stats = cache_manager.get_statistics()
        assert stats['total_files'] == 1
        assert stats['total_size_bytes'] == 1024
        assert 'test' in stats['categories']
        assert stats['categories']['test']['count'] == 1
    
    def test_get_cached_file(self, cache_manager, test_files):
        """Test retrieving cached file"""
        # Cache a file
        cache_manager.cache_file(test_files['small'], 'test_key')
        
        # Retrieve it
        cached_path = cache_manager.get_cached_file('test_key')
        assert cached_path is not None
        assert cached_path.exists()
        
        # Non-existent key
        missing = cache_manager.get_cached_file('missing_key')
        assert missing is None
    
    def test_lru_eviction(self, cache_manager, test_files):
        """Test LRU eviction when cache is full"""
        # Set very small cache size (2KB)
        cache_manager.max_size_bytes = 2 * 1024
        
        # Cache first file
        cache_manager.cache_file(test_files['small'], 'file1')
        time.sleep(0.1)  # Ensure different timestamps
        
        # Cache second file
        cache_manager.cache_file(test_files['small'], 'file2')
        
        # Both should exist
        assert cache_manager.get_cached_file('file1') is not None
        assert cache_manager.get_cached_file('file2') is not None
        
        # Cache third file - should evict file1 (oldest)
        cache_manager.cache_file(test_files['small'], 'file3')
        
        # File1 should be evicted
        assert cache_manager.get_cached_file('file1') is None
        assert cache_manager.get_cached_file('file2') is not None
        assert cache_manager.get_cached_file('file3') is not None
    
    def test_access_time_update(self, cache_manager, test_files):
        """Test that accessing files updates their LRU order"""
        # Cache files
        cache_manager.cache_file(test_files['small'], 'file1')
        time.sleep(0.1)
        cache_manager.cache_file(test_files['small'], 'file2')
        time.sleep(0.1)
        cache_manager.cache_file(test_files['small'], 'file3')
        
        # Access file1 (making it most recent)
        cache_manager.get_cached_file('file1')
        
        # Set small cache
        cache_manager.max_size_bytes = 3 * 1024
        
        # Add file4 - should evict file2 (now oldest untouched)
        cache_manager.cache_file(test_files['small'], 'file4')
        
        assert cache_manager.get_cached_file('file1') is not None  # Recently accessed
        assert cache_manager.get_cached_file('file2') is None      # Evicted
        assert cache_manager.get_cached_file('file3') is not None
        assert cache_manager.get_cached_file('file4') is not None
    
    def test_evict_file(self, cache_manager, test_files):
        """Test manual file eviction"""
        # Cache files
        cache_manager.cache_file(test_files['small'], 'test_key')
        
        # Verify it exists
        assert cache_manager.get_cached_file('test_key') is not None
        
        # Evict it
        result = cache_manager.evict_file('test_key')
        assert result is True
        
        # Verify it's gone
        assert cache_manager.get_cached_file('test_key') is None
        
        # Evicting non-existent should return False
        result = cache_manager.evict_file('test_key')
        assert result is False
    
    def test_evict_category(self, cache_manager, test_files):
        """Test evicting all files in a category"""
        # Cache files in different categories
        cache_manager.cache_file(test_files['small'], 'cat1_file1', category='cat1')
        cache_manager.cache_file(test_files['small'], 'cat1_file2', category='cat1')
        cache_manager.cache_file(test_files['small'], 'cat2_file1', category='cat2')
        
        # Evict category 1
        count = cache_manager.evict_category('cat1')
        assert count == 2
        
        # Verify cat1 files are gone, cat2 remains
        assert cache_manager.get_cached_file('cat1_file1') is None
        assert cache_manager.get_cached_file('cat1_file2') is None
        assert cache_manager.get_cached_file('cat2_file1') is not None
    
    def test_cleanup_old_files(self, cache_manager, test_files):
        """Test cleaning up old files"""
        # Cache a file
        cache_manager.cache_file(test_files['small'], 'old_file')
        
        # Manually set old access time
        with cache_manager._lock:
            old_time = (time.time() - 40 * 24 * 3600) * 1000  # 40 days ago
            cache_manager._metadata['files']['old_file']['last_access'] = \
                cache_manager._metadata['files']['old_file']['created']
            cache_manager._save_metadata()
        
        # Cache a recent file
        cache_manager.cache_file(test_files['small'], 'new_file')
        
        # Cleanup files older than 30 days
        removed = cache_manager.cleanup_old_files(days=30)
        assert removed == 1
        
        # Old file should be gone, new file remains
        assert cache_manager.get_cached_file('old_file') is None
        assert cache_manager.get_cached_file('new_file') is not None
    
    def test_clear_cache(self, cache_manager, test_files):
        """Test clearing entire cache"""
        # Cache multiple files
        cache_manager.cache_file(test_files['small'], 'file1')
        cache_manager.cache_file(test_files['medium'], 'file2')
        cache_manager.cache_file(test_files['large'], 'file3')
        
        # Clear cache
        count = cache_manager.clear_cache()
        assert count == 3
        
        # Verify all gone
        stats = cache_manager.get_statistics()
        assert stats['total_files'] == 0
        assert stats['total_size_bytes'] == 0
    
    def test_thread_safety(self, cache_manager, test_files):
        """Test thread-safe operations"""
        import threading
        import random
        
        errors = []
        
        def cache_operations():
            try:
                for i in range(10):
                    key = f"thread_file_{random.randint(1, 5)}"
                    
                    if random.random() < 0.5:
                        # Cache file
                        cache_manager.cache_file(test_files['small'], key)
                    else:
                        # Get file
                        cache_manager.get_cached_file(key)
                    
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)
        
        # Run multiple threads
        threads = []
        for _ in range(5):
            t = threading.Thread(target=cache_operations)
            threads.append(t)
            t.start()
        
        # Wait for completion
        for t in threads:
            t.join()
        
        # Should have no errors
        assert len(errors) == 0
        
        # Cache should be consistent
        stats = cache_manager.get_statistics()
        assert stats['total_files'] >= 0
        assert stats['total_size_bytes'] >= 0
    
    def test_cache_persistence(self, cache_dir, test_files):
        """Test that cache metadata persists across instances"""
        # Create first instance and cache files
        cache1 = CacheManager(cache_dir=cache_dir, max_size_gb=0.01)
        cache1.cache_file(test_files['small'], 'persist_test', category='test')
        
        stats1 = cache1.get_statistics()
        del cache1  # Destroy instance
        
        # Create new instance
        cache2 = CacheManager(cache_dir=cache_dir, max_size_gb=0.01)
        
        # Should have same statistics
        stats2 = cache2.get_statistics()
        assert stats2['total_files'] == stats1['total_files']
        assert stats2['total_size_bytes'] == stats1['total_size_bytes']
        
        # Should be able to retrieve file
        cached = cache2.get_cached_file('persist_test')
        assert cached is not None
        assert cached.exists()