from enum import Enum, IntEnum
from dataclasses import dataclass, field
import hashlib
import time
from typing import Any, Optional, Dict

class CompressionAlgorithm(Enum):
    """Available compression algorithms"""
    NONE = "none"
    ZLIB = "zlib"
    LZ4 = "lz4"
    GZIP = "gzip"


class CacheEvictionPolicy(Enum):
    """Cache eviction policies"""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In, First Out
    TTL = "ttl"  # Time To Live


class ShardingStrategy(Enum):
    """Database sharding strategies"""
    HASH = "hash"
    RANGE = "range"
    DIRECTORY = "directory"


class WALEntryType(IntEnum):
    """Write-Ahead Log entry types"""
    INSERT = 1
    UPDATE = 2
    DELETE = 3
    CHECKPOINT = 4
    COMMIT = 5
    ROLLBACK = 6


@dataclass
class WALEntry:
    """Write-Ahead Log entry"""
    entry_id: str
    entry_type: WALEntryType
    timestamp: float
    table_name: str
    data: Dict[str, Any]
    checksum: str = field(init=False)
    
    def __post_init__(self):
        self.checksum = self._calculate_checksum()
    
    def _calculate_checksum(self) -> str:
        content = f"{self.entry_id}{self.entry_type}{self.timestamp}{self.table_name}{json.dumps(self.data, sort_keys=True)}"
        return hashlib.sha256(content.encode()).hexdigest()


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    data: Any
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    created_at: float = field(default_factory=time.time)
    ttl: Optional[float] = None
    size_bytes: int = 0