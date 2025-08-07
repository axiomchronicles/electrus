import os
import json
import aiofiles

from typing import (
    Any,
    Dict,
    List,
    Union,
    Optional
)

from ..partials import (
    ElectrusUpdateData,
    ElectrusInsertData,
    ElectrusFindData,
    ElectrusDeleteData,
    ElectrusLogicalOperators,
    DatabaseActionResult
)

from ..utils import (
    ElectrusBulkOperation,
    ElectrusDistinctOperation,
    ElectrusDataComparator,
    ElectrusAggregation
)

from .transactions import Transactions

from ...exception.base import ElectrusException

class Collection:
    def __init__(self, db_name: str, collection_name: str) -> None:
        self.db_name: str = db_name
        self.collection_name: str = collection_name
        self.base_path: str = os.path.expanduser(f'~/.electrus')
        self.collection_dir_path: str = os.path.join(self.base_path, self.db_name, self.collection_name)
        self.collection_path: str = os.path.join(self.collection_dir_path, f'{self.collection_name}.json')
        os.makedirs(self.collection_dir_path, exist_ok=True)
        self._connected: bool = True
        self.current_database: str = self.db_name
        self._create_empty_collection_file()
        self.current_transaction: Optional[str] = None 
        self.session_active: bool = False 

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
    async def _read_collection_data(self) -> List[Dict[str, Any]]:
        try:
            if os.path.exists(self.collection_path):
                async with aiofiles.open(self.collection_path, 'r') as file:
                    return json.loads(await file.read())
        except Exception as e:
            raise ElectrusException(f"Error reading collection data: {e}")
        return []
    
    @_validate_connection
    async def _write_json(self, data: Any, file_path: str) -> None:
        try:
            async with aiofiles.open(file_path, 'w') as file:
                await file.write(json.dumps(data, indent=4))
        except Exception as e:
            raise ElectrusException(f"Error writing to file: {e}")
        
    @_validate_connection
    def transactions(self) -> Transactions:
        return Transactions(self)
    
    @_validate_connection
    async def start_session(self) -> None:
        """
        Start a session.
        """
        if self.session_active:
            raise ElectrusException("Session already active.")
        self.session_active = True

    @_validate_connection
    async def end_session(self) -> None:
        """
        End a session.
        """
        if not self.session_active:
            raise ElectrusException("No active session.")
        self.session_active = False

    @_validate_connection
    async def insert_one(self, data: Dict[str, Any], overwrite: Optional[bool] = False) -> DatabaseActionResult:
        try:
            collection_path = self.collection_path
            return await ElectrusInsertData(collection_path)._obl_one(data, overwrite)
        except Exception as e:
            raise ElectrusException(f"Error inserting data: {e}")
        
    @_validate_connection
    async def insert_many(self, data_list: List[Dict[str, Any]], overwrite: Optional[bool] = False) -> DatabaseActionResult:
        try:
            collection_path = self.collection_path
            return await ElectrusInsertData(collection_path)._obl_many(data_list, overwrite)
        except Exception as e:
            raise ElectrusException(f"Error inserting multiple data: {e}")
        
    @_validate_connection
    async def update_one(self, filter_query: Dict[str, Any], update_data: Dict[str, Any]) -> DatabaseActionResult:
        return await ElectrusUpdateData.update(self.collection_path, filter_query, update_data)
    
    @_validate_connection
    async def update_many(self, filter_query: Dict[str, Any], update_data: Dict[str, Any]) -> DatabaseActionResult:
        return await ElectrusUpdateData.update(self.collection_path, filter_query, update_data, multi=True)
    
    @_validate_connection
    async def find_one(self, filter_query: Dict[str, Any], projection: List[str] = None) -> DatabaseActionResult:
        try:
            collection_path = self.collection_path
            return await ElectrusFindData(collection_path).find_one(filter_query, projection)
        except Exception as e:
            raise ElectrusException(f"Error finding data: {e}")
        
    @_validate_connection
    def find(self) -> ElectrusFindData:
        return ElectrusFindData(self.collection_path)
    
    @_validate_connection
    async def find_many(self, filter_query: Dict[str, Any], projection: List[str] = None, sort_by: str = None, limit: int = None) -> DatabaseActionResult:
        try:
            collection_data = await self._read_collection_data()
            results = []
            operator_evaluator = ElectrusLogicalOperators()

            for item in collection_data:
                if operator_evaluator.evaluate(item, filter_query):
                    if projection:
                        result = {key: item.get(key) for key in projection}
                        results.append(result)
                    else:
                        results.append(item)
                    if limit and len(results) >= limit:
                        break

            if sort_by:
                results = sorted(results, key=lambda x: x.get(sort_by))

            return DatabaseActionResult(success = True, inserted_ids = [_dict['_id'] for _dict in results], matched_count = len(results), data = results)
        
        except FileNotFoundError:
            raise ElectrusException(f"Database '{self.db_name}' or collection '{self.collection_name}' not found.")
        except Exception as e:
            raise ElectrusException(f"Error finding data: {e}")
        
    @_validate_connection
    async def fetch_all(
        self,
        filter_query: Optional[Dict[str, Any]] = None,
        projection: Optional[List[str]] = None,
        sort_by: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> DatabaseActionResult:
        try:
            collection_data = await self._read_collection_data()
            operator_evaluator = ElectrusLogicalOperators()

            if filter_query:
                collection_data = [item for item in collection_data if operator_evaluator.evaluate(item, filter_query)]

            if projection:
                if not filter_query:
                    collection_data = [{key: item.get(key) for key in projection} for item in collection_data]
                else:
                    collection_data = [{key: doc[key] for key in projection if key in doc} for doc in collection_data]

            if sort_by:
                collection_data.sort(key=lambda x: x.get(sort_by))

            if limit:
                collection_data = collection_data[:limit]

            return DatabaseActionResult(success = True, inserted_ids = [_dict['_id'] for _dict in collection_data], matched_count = len(collection_data), data = collection_data)

        except FileNotFoundError:
            raise ElectrusException(f"Database '{self.db_name}' or collection '{self.collection_name}' not found.")
        except Exception as e:
            raise ElectrusException(f"Error fetching data: {e}")
        
    @_validate_connection
    async def count_documents(self, filter_query: Dict[str, Any]) -> int:
        try:
            collection_data = await self._read_collection_data()
            count = sum(1 for item in collection_data if all(item.get(key) == value for key, value in filter_query.items()))
            return count
        except FileNotFoundError:
            raise ElectrusException(f"Database '{self.db_name}' or collection '{self.collection_name}' not found.")
        except Exception as e:
            raise ElectrusException(f"Error counting documents: {e}")

    @_validate_connection
    async def delete_one(self, filter_query: Dict[str, Any]) -> DatabaseActionResult:
        return await ElectrusDeleteData.delete(self.collection_path, filter_query)
    
    @_validate_connection
    async def delete_many(self, filter_query: Dict[str, Any]) -> DatabaseActionResult:
        return await ElectrusDeleteData.delete(self.collection_path, filter_query, True)
    
    @_validate_connection
    async def bulk_operation(self, operations: List[Dict[str, Any]]) -> ElectrusBulkOperation:
        return await ElectrusBulkOperation(self.collection_path)._bulk_write(operations)
    
    @_validate_connection
    async def distinct(
        self,
        field: str,
        filter_query: Optional[Dict[str, Any]] = None,
        sort: Optional[bool] = False
    ) -> ElectrusDistinctOperation:
        return await ElectrusDistinctOperation(self.collection_path)._distinct(field, filter_query, sort)
    
    @_validate_connection
    async def import_data(self, file_path: str, append: bool = False) -> None:
        try:
            data_comparator = ElectrusDataComparator()
            await data_comparator.import_data(file_path, self.collection_path, append)
        except Exception as e:
            raise ElectrusException(f"Error importing data: {e}")
        
    @_validate_connection
    async def export_data(self, file_path: str) -> None:
        try:
            collection_data = await self._read_collection_data()
            data_comparator = ElectrusDataComparator()
            await data_comparator.export_data(file_path, collection_data)
        except Exception as e:
            raise ElectrusException(f"Error exporting data: {e}")

    @_validate_connection  
    async def aggregation(self, pipeline: List[Dict[str, Any]] = None) -> ElectrusAggregation:
        try:
            collection_data = await self._read_collection_data()
            aggregation = ElectrusAggregation(collection_data)
            if not pipeline:
                result = aggregation.execute(pipeline)
                return result
            else:
                return aggregation
        except Exception as e:
            raise ElectrusException(f"Error performing aggregation: {e}")
        