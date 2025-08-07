import os
import json
import time
from enum import Enum, auto
from queue import Queue, Empty
from threading import Lock
from collections import defaultdict
from typing import Any, Dict, List, Optional, Callable, Union

from ..partials import (
    ElectrusUpdateData,
    ElectrusInsertData,
    ElectrusFindData,
    ElectrusDeleteData,
    DatabaseActionResult
)
from ..partials.indexmanager import ElectrusIndexManager
from ..utils import (
    ElectrusBulkOperation,
    ElectrusDistinctOperation,
    ElectrusAggregation
)
from .transactions import Transactions
from ..handler.filemanager import JsonFileHandler, FileVersionManager, FileLockManager
from ..exception.base import ElectrusException


class ConnectionState(Enum):
    DISCONNECTED = auto()
    CONNECTING   = auto()
    CONNECTED    = auto()
    CLOSING      = auto()


class ConnectionMetrics:
    def __init__(self):
        self.active_connections = 0
        self.operation_counts = defaultdict(int)
        self.total_latency = defaultdict(float)

    def connection_opened(self):
        self.active_connections += 1

    def connection_closed(self):
        self.active_connections -= 1

    def record(self, operation: str, latency: float):
        self.operation_counts[operation] += 1
        self.total_latency[operation] += latency

    def report(self) -> Dict[str, Any]:
        report = {"active_connections": self.active_connections}
        for op, count in self.operation_counts.items():
            avg_latency_ms = (self.total_latency[op] / count) * 1000
            report[op] = {"count": count, "avg_latency_ms": avg_latency_ms}
        return report


metrics = ConnectionMetrics()


def timed_operation(op_name: str):
    def decorator(func: Callable):
        async def wrapper(self, *args, **kwargs):
            start = time.perf_counter()
            result = await func(self, *args, **kwargs)
            elapsed = time.perf_counter() - start
            metrics.record(op_name, elapsed)
            return result
        return wrapper
    return decorator


def validate_connected(func: Callable):
    def wrapper(self, *args, **kwargs):
        if self.state is not ConnectionState.CONNECTED:
            raise ElectrusException(f"Cannot perform '{func.__name__}' when state is {self.state}")
        return func(self, *args, **kwargs)
    return wrapper


class Collection:
    def __init__(self, db_name: str, collection_name: str, db_path: str, logger) -> None:
        self.db_name = db_name
        self.collection_name = collection_name
        self.db_path = db_path
        self.logger = logger
        self.base_path = os.path.expanduser('~/.electrus')
        self.collection_dir_path = os.path.join(self.base_path, db_name, collection_name)
        self.collection_path = os.path.join(self.collection_dir_path, f'{collection_name}.json')
        self.state = ConnectionState.DISCONNECTED
        self.handler: Optional[JsonFileHandler] = None
        self.index_manager: Optional[ElectrusIndexManager] = None
        self.current_transaction: Optional[str] = None
        self.session_active = False
        self._connect()

    def _connect(self) -> None:
        if self.state is not ConnectionState.DISCONNECTED:
            raise ElectrusException(f"Cannot connect from state {self.state}")
        self.state = ConnectionState.CONNECTING
        os.makedirs(self.collection_dir_path, exist_ok=True)
        self.handler = JsonFileHandler(
            self.collection_dir_path,
            FileVersionManager(self.collection_dir_path),
            FileLockManager()
        )
        self.index_manager = ElectrusIndexManager(self)
        self._create_empty_collection_file()
        metrics.connection_opened()
        self.state = ConnectionState.CONNECTED

    def _create_empty_collection_file(self) -> None:
        if not os.path.exists(self.collection_path):
            with open(self.collection_path, 'w') as f:
                f.write(json.dumps([], indent=4))

    @validate_connected
    def transactions(self) -> Transactions:
        return Transactions(self)

    @validate_connected
    async def start_session(self) -> None:
        if self.session_active:
            raise ElectrusException("Session already active.")
        self.session_active = True

    @validate_connected
    async def end_session(self) -> None:
        if not self.session_active:
            raise ElectrusException("No active session.")
        self.session_active = False

    @validate_connected
    @timed_operation("insertOne")
    async def insertOne(self, data: Dict[str, Any], overwrite: bool = False) -> DatabaseActionResult:
        try:
            return await ElectrusInsertData(
                self.collection_path,
                self.handler,
                self.index_manager
            )._obl_one(data, overwrite)
        except Exception as e:
            raise ElectrusException(f"Error inserting data: {e}")

    @validate_connected
    @timed_operation("insertMany")
    async def insertMany(self, data_list: List[Dict[str, Any]], overwrite: bool = False) -> DatabaseActionResult:
        try:
            return await ElectrusInsertData(
                self.collection_path,
                self.handler,
                self.index_manager
            )._obl_many(data_list, overwrite)
        except Exception as e:
            raise ElectrusException(f"Error inserting multiple data: {e}")

    @validate_connected
    @timed_operation("update")
    async def update(
        self,
        filter: Dict[str, Any],
        update_data: Dict[str, Any],
        multi: bool = False,
        upsert: bool = False,
        upsert_doc: Optional[Dict[str, Any]] = None,
        return_updated_fields: Optional[List[str]] = None,
        *,
        max_workers: int = 4,
        enable_differential_logging: bool = True,
        audit_log_path: Optional[str] = None,
        pre_update_hooks: Optional[List[Callable[[Dict[str, Any]], None]]] = None,
        post_update_hooks: Optional[List[Callable[[Dict[str, Any]], None]]] = None
    ) -> DatabaseActionResult:
        return await ElectrusUpdateData(
            self.handler, max_workers = max_workers, enable_differential_logging = enable_differential_logging,
            audit_log_path = audit_log_path, pre_update_hooks = pre_update_hooks, post_update_hooks = post_update_hooks
        ).update(
            self.collection_path,
            filter,
            update_data,
            multi=multi,
            upsert=upsert,
            upsert_doc=upsert_doc,
            return_updated_fields=return_updated_fields
        )

    @validate_connected
    def find(self) -> ElectrusFindData:
        return ElectrusFindData(self.collection_path, self.handler)

    @validate_connected
    @timed_operation("count")
    async def count(self, filter_query: Dict[str, Any]) -> int:
        try:
            data = await self.handler.read_async(self.collection_path)
            return sum(1 for item in data if all(item.get(k) == v for k, v in filter_query.items()))
        except FileNotFoundError:
            raise ElectrusException(f"Collection not found.")
        except Exception as e:
            raise ElectrusException(f"Error counting documents: {e}")

    @validate_connected
    def delete(self) -> ElectrusDeleteData:
        return ElectrusDeleteData(self.handler, self.collection_path)

    @validate_connected
    @timed_operation("bulk_operation")
    async def bulk_operation(self, operations: List[Dict[str, Any]]) -> ElectrusBulkOperation:
        return await ElectrusBulkOperation(self.collection_path)._bulk_write(operations)

    @validate_connected
    @timed_operation("distinct")
    async def distinct(
        self,
        field: str,
        filter_query: Optional[Dict[str, Any]] = None,
        *,
        sort: bool = False,
        use_cache: bool = True,
        statistics: bool = True,
        use_bloom_filter: bool = True,
        bloom_capacity: int = 10_000,
        cache_size: int = 256,
        cache_ttl: int = 600,
        on_complete: Optional[Callable[[Union[List[Any], Dict[str, Any]]], None]] = None
    ) -> Union[List[Any], Dict[str, Any]]:
        builder = ElectrusDistinctOperation(
            self.collection_path,
            self.handler,
            cache_size=cache_size,
            cache_ttl=cache_ttl,
            use_bloom_filter=use_bloom_filter,
            bloom_capacity=bloom_capacity
        )
        if statistics:
            result = await builder.distinct_with_stats(field, filter_query, sort)
        else:
            result = await builder._distinct(field, filter_query, sort, use_cache)
        if on_complete:
            on_complete(result)
        return result

    @validate_connected
    def aggregation(self) -> ElectrusAggregation:
        try:
            return ElectrusAggregation(self.collection_path, self.handler)
        except Exception as e:
            raise ElectrusException(f"Error performing aggregation: {e}")

    async def close(self) -> None:
        if self.state is not ConnectionState.CONNECTED:
            raise ElectrusException(f"Cannot close from state {self.state}")
        self.state = ConnectionState.CLOSING
        metrics.connection_closed()
        self.state = ConnectionState.DISCONNECTED
        return True


class CollectionPool:
    def __init__(self, db_name: str, collection_name: str, db_path: str, logger, max_size: int = 5):
        self._pool = Queue(maxsize=max_size)
        for _ in range(max_size):
            self._pool.put(Collection(db_name, collection_name, db_path, logger))
        self._lock = Lock()

    def acquire(self, timeout: Optional[float] = None) -> Collection:
        try:
            coll = self._pool.get(timeout=timeout)
            if coll.state is not ConnectionState.CONNECTED:
                coll._connect()
            return coll
        except Empty:
            raise ElectrusException("No available connections in pool")

    def release(self, coll: Collection) -> None:
        if coll.state is not ConnectionState.CONNECTED:
            coll._connect()
        self._pool.put(coll)
