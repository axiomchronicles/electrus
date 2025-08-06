import os
import json


from typing import (
    Any,
    Dict,
    List,
    Union,
    Optional,
    Callable
)


from ..partials import (
    ElectrusUpdateData,
    ElectrusInsertData,
    ElectrusFindData,
    ElectrusDeleteData,
    DatabaseActionResult
)


from ..utils import (
    ElectrusBulkOperation,
    ElectrusDistinctOperation,
    ElectrusAggregation
)


from .transactions import Transactions
from ..handler.filemanager import JsonFileHandler, FileVersionManager, FileLockManager


from ..exception.base import ElectrusException


class Collection:
    def __init__(self, db_name: str, collection_name: str, db_path: str, logger) -> None:
        self.db_name: str = db_name
        self.collection_name: str = collection_name
        self.db_path: str = db_path
        self.logger = logger
        self.base_path: str = os.path.expanduser(f'~/.electrus')
        self.collection_dir_path: str = os.path.join(self.base_path, self.db_name, self.collection_name)
        self.collection_path: str = os.path.join(self.collection_dir_path, f'{self.collection_name}.json')
        os.makedirs(self.collection_dir_path, exist_ok=True)
        self._connected: bool = True
        self.current_database: str = self.db_name
        self._create_empty_collection_file()
        self.current_transaction: Optional[str] = None 
        self.session_active: bool = False 
        self.handler: JsonFileHandler = JsonFileHandler(self.collection_dir_path, FileVersionManager(self.collection_dir_path), FileLockManager())


    async def close(self) -> None:
        if not self._connected:
            raise ElectrusException("Not connected to any database or connection already closed.")
        
        self.current_database = None
        self._connected = False


        return True


    def _validate_connection(func):
        def wrapper(self, *args, **kwargs):
            if not self._connected:
                raise ElectrusException("Not connected to any database or connection closed.")
            return func(self, *args, **kwargs)
        return wrapper
    
    @_validate_connection
    def _create_empty_collection_file(self) -> None:
        if not os.path.exists(self.collection_path):
            with open(self.collection_path, 'w') as file:
                file.write(json.dumps([], indent=4))
        
    @_validate_connection
    def transactions(self) -> Transactions:
        return Transactions(self)
    
    @_validate_connection
    async def start_session(self) -> None:
        if self.session_active:
            raise ElectrusException("Session already active.")
        self.session_active = True


    @_validate_connection
    async def end_session(self) -> None:
        if not self.session_active:
            raise ElectrusException("No active session.")
        self.session_active = False


    @_validate_connection
    async def insertOne(self, data: Dict[str, Any], overwrite: Optional[bool] = False) -> DatabaseActionResult:
        try:
            collection_path = self.collection_path
            return await ElectrusInsertData(collection_path, self.handler)._obl_one(data, overwrite)
        except Exception as e:
            raise ElectrusException(f"Error inserting data: {e}")
        
    @_validate_connection
    async def insertMany(self, data_list: List[Dict[str, Any]], overwrite: Optional[bool] = False) :
        try:
            collection_path = self.collection_path
            return await ElectrusInsertData(collection_path, self.handler)._obl_many(data_list, overwrite)
        except Exception as e:
            raise ElectrusException(f"Error inserting multiple data: {e}")
        
    @_validate_connection
    async def update(self, filter_query: Dict[str, Any], update_data: Dict[str, Any], multi: bool = False,
    upsert: bool = False,
    upsert_doc: Dict[str, Any] | None = None,
    return_updated_fields: List[str] | None = None
    ) -> DatabaseActionResult:
        return await ElectrusUpdateData(self.handler).update(
            self.collection_path, filter_query, update_data, upsert = upsert, upsert_doc = upsert_doc, return_updated_fields = return_updated_fields, multi = multi
        )
    
    @_validate_connection
    def find(self) -> ElectrusFindData:
        return ElectrusFindData(self.collection_path, self.handler)
    
    @_validate_connection
    async def count(self, filter_query: Dict[str, Any]) -> int:
        try:
            collection_data = await self.handler.read_async(self.collection_path)
            count = sum(1 for item in collection_data if all(item.get(key) == value for key, value in filter_query.items()))
            return count
        except FileNotFoundError:
            raise ElectrusException(f"Database '{self.db_name}' or collection '{self.collection_name}' not found.")
        except Exception as e:
            raise ElectrusException(f"Error counting documents: {e}")


    @_validate_connection
    def delete(self) -> ElectrusDeleteData:
        return ElectrusDeleteData(self.handler, self.collection_path)
    
    @_validate_connection
    async def delete_many(self, filter_query: Dict[str, Any]) -> DatabaseActionResult:
        return await ElectrusDeleteData(self.handler, self.collection_path).delete(filter_query, True)
    
    @_validate_connection
    async def bulk_operation(self, operations: List[Dict[str, Any]]) -> ElectrusBulkOperation:
        return await ElectrusBulkOperation(self.collection_path)._bulk_write(operations)
    
    @_validate_connection
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
        on_complete: Optional[Callable[[Union[List[Any], Dict[str, Any]]], None]] = None,
    ) -> Union[List[Any], Dict[str, Any]]:


        builder = ElectrusDistinctOperation(
            self.collection_path,
            self.handler,
            cache_size=cache_size,
            cache_ttl=cache_ttl,
            use_bloom_filter=use_bloom_filter,
            bloom_capacity=bloom_capacity,
        )


        result: Union[List[Any], Dict[str, Any]]
        if statistics:
            result = await builder.distinct_with_stats(field, filter_query, sort)
        else:
            result = await builder._distinct(field, filter_query, sort, use_cache)


        if on_complete:
            on_complete(result)
        return result


    @_validate_connection  
    def aggregation(self):
        try:
            return ElectrusAggregation(self.collection_path, self.handler)
        except Exception as e:
            raise ElectrusException(f"Error performing aggregation: {e}")