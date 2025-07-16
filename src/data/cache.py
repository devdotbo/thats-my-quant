"""
Cache Management System
Manages storage with LRU eviction and 100GB limit
"""

import os
import time
import shutil
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import OrderedDict
from datetime import datetime, timedelta
import threading

from src.utils.config import get_config
from src.utils.logging import get_logger


class CacheManager:
    """
    Manages cached data files with LRU eviction and size limits.
    
    Features:
    - LRU (Least Recently Used) eviction policy
    - Configurable size limit (default 100GB)
    - Thread-safe operations
    - Automatic cleanup when limit reached
    - Cache statistics and monitoring
    """
    
    def __init__(self, cache_dir: Optional[Path] = None, max_size_gb: float = 100.0):
        """
        Initialize cache manager.
        
        Args:
            cache_dir: Directory for cache storage (default from config)
            max_size_gb: Maximum cache size in GB
        """
        self.config = get_config()
        self.logger = get_logger(__name__)
        
        # Set cache directory
        if cache_dir is None:
            cache_dir = Path(self.config.data.cache_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache settings
        self.max_size_bytes = int(max_size_gb * 1024**3)
        self.metadata_file = self.cache_dir / '.cache_metadata.json'
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize or load metadata
        self._metadata = self._load_metadata()
        
        # LRU tracking (ordered by access time)
        self._lru_order = OrderedDict()
        self._rebuild_lru_order()
        
        # Save initial metadata if new
        if not self.metadata_file.exists():
            self._save_metadata()
        
        self.logger.info(f"Initialized cache manager: {self.cache_dir} (max {max_size_gb}GB)")
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load cache metadata from disk"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading metadata: {e}")
        
        return {
            'files': {},
            'total_size': 0,
            'last_cleanup': datetime.now().isoformat()
        }
    
    def _save_metadata(self):
        """Save cache metadata to disk"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self._metadata, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving metadata: {e}")
    
    def _rebuild_lru_order(self):
        """Rebuild LRU order from metadata"""
        self._lru_order.clear()
        
        # Sort files by last access time
        files_by_access = sorted(
            self._metadata['files'].items(),
            key=lambda x: x[1].get('last_access', x[1]['created'])
        )
        
        for file_key, _ in files_by_access:
            self._lru_order[file_key] = True
    
    def _get_file_size(self, file_path: Path) -> int:
        """Get file size in bytes"""
        try:
            return file_path.stat().st_size
        except:
            return 0
    
    def _calculate_total_size(self) -> int:
        """Calculate total cache size"""
        total = 0
        for file_info in self._metadata['files'].values():
            total += file_info.get('size', 0)
        return total
    
    def cache_file(self, source_path: Path, cache_key: str, 
                   category: Optional[str] = None) -> Path:
        """
        Cache a file and return the cached path.
        
        Args:
            source_path: Path to file to cache
            cache_key: Unique key for cached file
            category: Optional category for organization
            
        Returns:
            Path to cached file
        """
        with self._lock:
            # Determine cache path
            if category:
                cache_path = self.cache_dir / category / cache_key
            else:
                cache_path = self.cache_dir / cache_key
            
            # Check if already cached
            if cache_key in self._metadata['files']:
                # Update access time
                self._metadata['files'][cache_key]['last_access'] = datetime.now().isoformat()
                self._lru_order.move_to_end(cache_key)
                self._save_metadata()
                return cache_path
            
            # Get file size
            file_size = self._get_file_size(source_path)
            
            # Check if we need to evict files
            current_size = self._calculate_total_size()
            if current_size + file_size > self.max_size_bytes:
                self._evict_files(file_size)
            
            # Copy file to cache
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, cache_path)
            
            # Update metadata
            self._metadata['files'][cache_key] = {
                'path': str(cache_path),
                'size': file_size,
                'category': category,
                'created': datetime.now().isoformat(),
                'last_access': datetime.now().isoformat()
            }
            self._lru_order[cache_key] = True
            self._metadata['total_size'] = self._calculate_total_size()
            self._save_metadata()
            
            self.logger.info(f"Cached {cache_key} ({file_size / 1024**2:.1f}MB)")
            return cache_path
    
    def get_cached_file(self, cache_key: str) -> Optional[Path]:
        """
        Get a cached file path if it exists.
        
        Args:
            cache_key: Cache key to retrieve
            
        Returns:
            Path to cached file or None if not found
        """
        with self._lock:
            if cache_key not in self._metadata['files']:
                return None
            
            file_info = self._metadata['files'][cache_key]
            cache_path = Path(file_info['path'])
            
            if not cache_path.exists():
                # File was deleted externally
                del self._metadata['files'][cache_key]
                if cache_key in self._lru_order:
                    del self._lru_order[cache_key]
                self._save_metadata()
                return None
            
            # Update access time
            file_info['last_access'] = datetime.now().isoformat()
            self._lru_order.move_to_end(cache_key)
            self._save_metadata()
            
            return cache_path
    
    def _evict_files(self, required_space: int):
        """Evict least recently used files to make space"""
        freed_space = 0
        files_to_evict = []
        
        # Find files to evict (oldest first)
        for cache_key in self._lru_order:
            if freed_space >= required_space:
                break
            
            file_info = self._metadata['files'][cache_key]
            freed_space += file_info.get('size', 0)
            files_to_evict.append(cache_key)
        
        # Evict files
        for cache_key in files_to_evict:
            self.evict_file(cache_key)
        
        self.logger.info(f"Evicted {len(files_to_evict)} files to free {freed_space / 1024**2:.1f}MB")
    
    def evict_file(self, cache_key: str) -> bool:
        """
        Evict a specific file from cache.
        
        Args:
            cache_key: Cache key to evict
            
        Returns:
            True if evicted, False if not found
        """
        with self._lock:
            if cache_key not in self._metadata['files']:
                return False
            
            file_info = self._metadata['files'][cache_key]
            cache_path = Path(file_info['path'])
            
            # Delete file
            if cache_path.exists():
                cache_path.unlink()
                
                # Remove empty directories
                try:
                    cache_path.parent.rmdir()
                except:
                    pass  # Directory not empty
            
            # Update metadata
            del self._metadata['files'][cache_key]
            if cache_key in self._lru_order:
                del self._lru_order[cache_key]
            self._metadata['total_size'] = self._calculate_total_size()
            self._save_metadata()
            
            self.logger.info(f"Evicted {cache_key}")
            return True
    
    def evict_category(self, category: str) -> int:
        """
        Evict all files in a category.
        
        Args:
            category: Category to evict
            
        Returns:
            Number of files evicted
        """
        with self._lock:
            files_to_evict = [
                cache_key for cache_key, info in self._metadata['files'].items()
                if info.get('category') == category
            ]
            
            for cache_key in files_to_evict:
                self.evict_file(cache_key)
            
            return len(files_to_evict)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_size = self._calculate_total_size()
            
            # Category breakdown
            categories = {}
            for file_info in self._metadata['files'].values():
                cat = file_info.get('category', 'uncategorized')
                if cat not in categories:
                    categories[cat] = {'count': 0, 'size': 0}
                categories[cat]['count'] += 1
                categories[cat]['size'] += file_info.get('size', 0)
            
            return {
                'total_files': len(self._metadata['files']),
                'total_size_bytes': total_size,
                'total_size_gb': total_size / 1024**3,
                'max_size_gb': self.max_size_bytes / 1024**3,
                'usage_percent': (total_size / self.max_size_bytes) * 100,
                'categories': categories,
                'last_cleanup': self._metadata.get('last_cleanup')
            }
    
    def cleanup_old_files(self, days: int = 30) -> int:
        """
        Remove files older than specified days.
        
        Args:
            days: Remove files not accessed in this many days
            
        Returns:
            Number of files removed
        """
        with self._lock:
            cutoff_date = datetime.now() - timedelta(days=days)
            files_to_remove = []
            
            for cache_key, file_info in self._metadata['files'].items():
                last_access = file_info.get('last_access', file_info['created'])
                access_date = datetime.fromisoformat(last_access)
                
                if access_date < cutoff_date:
                    files_to_remove.append(cache_key)
            
            for cache_key in files_to_remove:
                self.evict_file(cache_key)
            
            self._metadata['last_cleanup'] = datetime.now().isoformat()
            self._save_metadata()
            
            self.logger.info(f"Cleaned up {len(files_to_remove)} files older than {days} days")
            return len(files_to_remove)
    
    def clear_cache(self) -> int:
        """Clear entire cache"""
        with self._lock:
            count = len(self._metadata['files'])
            
            # Remove all cached files
            for cache_key in list(self._metadata['files'].keys()):
                self.evict_file(cache_key)
            
            self.logger.info(f"Cleared cache ({count} files)")
            return count


# Example usage and testing
if __name__ == "__main__":
    # Initialize cache
    cache = CacheManager(max_size_gb=0.001)  # 1MB for testing
    
    # Test caching files
    from pathlib import Path
    
    # Create test files
    test_dir = Path("test_cache_files")
    test_dir.mkdir(exist_ok=True)
    
    # Create a test file
    test_file = test_dir / "test_data.txt"
    test_file.write_text("Sample data " * 1000)
    
    # Cache the file
    cached_path = cache.cache_file(test_file, "test_key_1", category="test")
    print(f"Cached to: {cached_path}")
    
    # Retrieve from cache
    retrieved = cache.get_cached_file("test_key_1")
    print(f"Retrieved: {retrieved}")
    
    # Get statistics
    stats = cache.get_statistics()
    print(f"Cache stats: {json.dumps(stats, indent=2)}")
    
    # Cleanup
    shutil.rmtree(test_dir)
    cache.clear_cache()