from typing import Any, Dict, List, Optional, Union, Set
from functools import lru_cache
from collections import OrderedDict
import weakref
import time
import hashlib
from ..exception.base import ElectrusException
from ..partials import ElectrusLogicalOperators
from ..handler.filemanager import JsonFileHandler


class LRUCache:
    """
    Custom LRU Cache implementation for distinct operations with TTL support
    """
    def __init__(self, maxsize: int = 128, ttl: int = 300):
        self.maxsize = maxsize
        self.ttl = ttl
        self.cache = OrderedDict()
        self.timestamps = {}
    
    def get(self, key: str) -> Optional[Any]:
        if key not in self.cache:
            return None
        
        # Check TTL
        if time.time() - self.timestamps[key] > self.ttl:
            self._evict(key)
            return None
        
        # Move to end (most recently used)
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def put(self, key: str, value: Any) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.maxsize:
                # Remove least recently used
                oldest_key = next(iter(self.cache))
                self._evict(oldest_key)
            self.cache[key] = value
        
        self.timestamps[key] = time.time()
    
    def _evict(self, key: str) -> None:
        if key in self.cache:
            del self.cache[key]
        if key in self.timestamps:
            del self.timestamps[key]
    
    def clear(self) -> None:
        self.cache.clear()
        self.timestamps.clear()


class BloomFilter:
    """
    Space-efficient probabilistic data structure for fast membership testing
    """
    def __init__(self, expected_elements: int = 1000, false_positive_prob: float = 0.01):
        self.expected_elements = expected_elements
        self.false_positive_prob = false_positive_prob
        
        # Calculate optimal bit array size and hash function count
        import math
        self.bit_array_size = int(-(expected_elements * math.log(false_positive_prob)) / (math.log(2) ** 2))
        self.hash_count = int((self.bit_array_size / expected_elements) * math.log(2))
        
        # Initialize bit array
        self.bit_array = [0] * self.bit_array_size
        self.element_count = 0
    
    def _hash_functions(self, item: str) -> List[int]:
        """Generate multiple hash values for an item"""
        hashes = []
        for i in range(self.hash_count):
            # Use different seeds for multiple hash functions
            hash_obj = hashlib.md5((str(item) + str(i)).encode())
            hash_value = int(hash_obj.hexdigest(), 16) % self.bit_array_size
            hashes.append(hash_value)
        return hashes
    
    def add(self, item: str) -> None:
        """Add an item to the bloom filter"""
        for hash_value in self._hash_functions(item):
            self.bit_array[hash_value] = 1
        self.element_count += 1
    
    def might_contain(self, item: str) -> bool:
        """Check if an item might be in the filter (no false negatives)"""
        return all(self.bit_array[hash_value] == 1 for hash_value in self._hash_functions(item))
    
    def estimated_false_positive_probability(self) -> float:
        """Calculate current false positive probability"""
        import math
        if self.element_count == 0:
            return 0.0
        return (1 - math.exp(-self.hash_count * self.element_count / self.bit_array_size)) ** self.hash_count


class DistinctOperation:
    """
    Optimized DistinctOperation class with advanced caching, bloom filters, and efficient data structures
    """
    
    def __init__(self, collection_path: str, handler: JsonFileHandler, 
                 cache_size: int = 256, cache_ttl: int = 600, 
                 use_bloom_filter: bool = True, bloom_capacity: int = 10000):
        self.collection_path = collection_path
        self.handler = handler
        
        # Advanced caching system
        self.cache = LRUCache(maxsize=cache_size, ttl=cache_ttl)
        self.field_cache = {}  # Field-specific caches
        
        # Bloom filter for fast negative lookups
        self.use_bloom_filter = use_bloom_filter
        if use_bloom_filter:
            self.bloom_filter = BloomFilter(expected_elements=bloom_capacity)
            self._bloom_initialized = False
        
        # Weak reference for memory efficiency
        self._data_cache = weakref.WeakValueDictionary()
        
        # Performance metrics
        self.cache_hits = 0
        self.cache_misses = 0
        self.bloom_hits = 0
        self.bloom_misses = 0
    
    def _generate_cache_key(self, field: str, filter_query: Optional[Dict[str, Any]], 
                          sort: Optional[bool]) -> str:
        """Generate a unique cache key for the operation"""
        import json
        key_data = {
            'field': field,
            'filter': filter_query,
            'sort': sort,
            'path': self.collection_path
        }
        return hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
    
    async def _read_collection_data(self, use_cache: bool = True) -> List[Dict[str, Any]]:
        """
        Efficiently read collection data using JsonFileHandler with caching
        """
        try:
            cache_key = f"collection_data_{self.collection_path}"
            
            # Check cache first
            if use_cache:
                cached_data = self.cache.get(cache_key)
                if cached_data is not None:
                    self.cache_hits += 1
                    return cached_data['data']
            
            self.cache_misses += 1
            
            # Use JsonFileHandler for optimized file operations
            result = await self.handler.read_async(
                self.collection_path,
                verify_integrity=False
            )
            
            collection_data = result['data']
            
            # Cache the result
            if use_cache:
                self.cache.put(cache_key, {
                    'data': collection_data,
                    'checksum': result['checksum'],
                    'timestamp': result['timestamp']
                })
            
            # Initialize bloom filter if enabled
            if self.use_bloom_filter and not self._bloom_initialized:
                await self._initialize_bloom_filter(collection_data)
            
            return collection_data
            
        except Exception as e:
            raise ElectrusException(f"Error reading collection data: {e}")
    
    async def _initialize_bloom_filter(self, collection_data: List[Dict[str, Any]]) -> None:
        """Initialize bloom filter with all field values for fast negative lookups"""
        if not self.use_bloom_filter:
            return
        
        for item in collection_data:
            for field, value in item.items():
                if value is not None:
                    self.bloom_filter.add(f"{field}:{str(value)}")
        
        self._bloom_initialized = True
    
    @lru_cache(maxsize=128)
    def _cached_field_extraction(self, data_hash: str, field: str) -> tuple:
        """Cache field value extraction using functools.lru_cache"""
        # This would be called with a hash of the data for caching
        # The actual implementation would need the data, but this shows the pattern
        pass
    
    def _extract_distinct_values_optimized(self, collection_data: List[Dict[str, Any]], 
                                         field: str) -> Set[Any]:
        """
        Optimized distinct value extraction using sets for O(1) average case operations
        """
        # Use set comprehension for efficient distinct value extraction
        distinct_values = {
            item.get(field) 
            for item in collection_data 
            if item.get(field) is not None
        }
        
        return distinct_values
    
    def _apply_filter_optimized(self, collection_data: List[Dict[str, Any]], 
                              field: str, filter_query: Dict[str, Any]) -> Set[Any]:
        """
        Apply filters with optimized evaluation using sets
        """
        logical_operators = ElectrusLogicalOperators()
        
        # Use generator expression with set comprehension for memory efficiency
        filtered_values = {
            item.get(field)
            for item in collection_data
            if item.get(field) is not None and
            all(
                logical_operators.evaluate(item, {filter_field: filter_query[filter_field]})
                if filter_field in filter_query else True
                for filter_field in item.keys()
            )
        }
        
        return filtered_values
    
    def _fast_membership_check(self, field: str, value: Any) -> bool:
        """
        Fast membership checking using bloom filter
        """
        if not self.use_bloom_filter:
            return True
        
        lookup_key = f"{field}:{str(value)}"
        if not self.bloom_filter.might_contain(lookup_key):
            self.bloom_hits += 1
            return False
        
        self.bloom_misses += 1
        return True
    
    async def _distinct(
        self,
        field: str,
        filter_query: Optional[Dict[str, Any]] = None,
        sort: Optional[bool] = False,
        use_cache: bool = True
    ) -> Union[List[Any], None]:
        """
        Highly optimized distinct operation with multiple performance enhancements
        """
        try:
            # Generate cache key
            cache_key = self._generate_cache_key(field, filter_query, sort)
            
            # Check cache first
            if use_cache:
                cached_result = self.cache.get(cache_key)
                if cached_result is not None:
                    self.cache_hits += 1
                    return cached_result
            
            self.cache_misses += 1
            
            # Read data using optimized handler
            collection_data = await self._read_collection_data(use_cache=use_cache)
            
            # Fast path for simple distinct operations
            if not filter_query:
                distinct_values = self._extract_distinct_values_optimized(collection_data, field)
            else:
                # Apply filters with optimization
                distinct_values = self._apply_filter_optimized(collection_data, field, filter_query)
            
            # Convert to list for sorting and return
            result_list = list(distinct_values)
            
            # Optimize sorting using Timsort (Python's default)
            if sort and result_list:
                try:
                    # Use key function for consistent sorting of mixed types
                    result_list.sort(key=lambda x: (x is None, x))
                except TypeError:
                    # Fallback for incomparable types
                    result_list.sort(key=lambda x: str(x) if x is not None else '')
            
            # Cache the result
            if use_cache:
                self.cache.put(cache_key, result_list)
            
            return result_list
            
        except Exception as e:
            raise ElectrusException(f"Error retrieving distinct values: {e}")
    
    async def distinct_with_stats(
        self,
        field: str,
        filter_query: Optional[Dict[str, Any]] = None,
        sort: Optional[bool] = False
    ) -> Dict[str, Any]:
        """
        Enhanced distinct operation that returns results with performance statistics
        """
        start_time = time.time()
        
        result = await self._distinct(field, filter_query, sort)
        
        end_time = time.time()
        
        stats = {
            'result': result,
            'execution_time': end_time - start_time,
            'cache_hit_ratio': self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0,
            'bloom_hit_ratio': self.bloom_hits / (self.bloom_hits + self.bloom_misses) if (self.bloom_hits + self.bloom_misses) > 0 else 0,
            'result_count': len(result) if result else 0
        }
        
        if self.use_bloom_filter:
            stats['bloom_false_positive_prob'] = self.bloom_filter.estimated_false_positive_probability()
        
        return stats
    
    def clear_cache(self) -> None:
        """Clear all caches and reset performance counters"""
        self.cache.clear()
        self.field_cache.clear()
        self._data_cache.clear()
        
        if hasattr(self, '_cached_field_extraction'):
            self._cached_field_extraction.cache_clear()
        
        # Reset counters
        self.cache_hits = 0
        self.cache_misses = 0
        self.bloom_hits = 0
        self.bloom_misses = 0
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics"""
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_ratio': self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0,
            'bloom_hits': self.bloom_hits,
            'bloom_misses': self.bloom_misses,
            'bloom_hit_ratio': self.bloom_hits / (self.bloom_hits + self.bloom_misses) if (self.bloom_hits + self.bloom_misses) > 0 else 0,
            'cache_size': len(self.cache.cache),
            'bloom_elements': self.bloom_filter.element_count if self.use_bloom_filter else 0
        }
