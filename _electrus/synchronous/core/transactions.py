import uuid
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from ...exception.base import ElectrusException

if TYPE_CHECKING:
    from .collection import Collection

class Transactions:
    def __init__(self, collection: 'Collection', parent_transaction: Optional['Transactions'] = None):
        """
        Represents a transaction for a collection.

        Args:
            collection (Collection): The collection associated with the transaction.
            parent_transaction (Optional[Transactions], optional): The parent transaction, if any. Defaults to None.
        """
        self.collection = collection
        self.transaction_id = str(uuid.uuid4())
        self.transaction_buffer = []
        self.parent_transaction = parent_transaction
        self.session = None

    def __enter__(self) -> 'Transactions':
        """
        Begin a transaction when entering the context.
        """
        self.session = self.collection.start_session()
        self.parent_transaction = self.collection.current_transaction
        self.collection.current_transaction = self
        self.begin()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """
        Commit or rollback the transaction when exiting the context.
        """
        if exc_type is None:
            try:
                self.commit()
            finally:
                self.collection.current_transaction = self.parent_transaction
                self.collection.end_session()
        else:
            self.rollback()
            self.collection.current_transaction = self.parent_transaction
            self.collection.end_session()

    def begin(self) -> None:
        """
        Begin a transaction.
        """
        self.transaction_buffer = []

    def commit(self) -> None:
        """
        Commit the transaction.
        """
        try:
            if not self.collection.session_active:
                raise ElectrusException("No active session found!.")
            
            for operation in self.transaction_buffer:
                self._execute_operation(operation)
        except ElectrusException as e:
            raise ElectrusException(f"Error committing transaction: {e}")
        finally:
            self.transaction_buffer = []

    def rollback(self) -> None:
        """
        Rollback the transaction.
        """
        self.transaction_buffer = []

    def insert_one(self, data: Dict[str, Any]) -> None:
        """
        Buffer insert operation for the transaction.

        Args:
            data (Dict[str, Any]): Data to be inserted.
        """
        self.transaction_buffer.append({'type': 'insert_one', 'data': data, "transactionId": self.transaction_id})

    def insert_many(self, data_list: List[Dict[str, Any]]) -> None:
        """
        Buffer insert_many operation for the transaction.

        Args:
            data_list (List[Dict[str, Any]]): List of data to be inserted.
        """
        self.transaction_buffer.append({'type': 'insert_many', 'data_list': data_list, "transactionId": self.transaction_id})

    def update_one(self, filter_query: Dict[str, Any], update_data: Dict[str, Any], upsert: bool = False) -> None:
        """
        Buffer update operation for the transaction.

        Args:
            filter_query (Dict[str, Any]): Filter query for documents to be updated.
            update_data (Dict[str, Any]): Data to update in the matching documents.
            upsert (bool, optional): If True, performs an upsert operation. Defaults to False.
        """
        self.transaction_buffer.append({'type': 'update_one', 'filter_query': filter_query, 'update_data': update_data, "transactionId": self.transaction_id})

    def delete_one(self, filter_query: Dict[str, Any]) -> None:
        """
        Buffer delete operation for the transaction.

        Args:
            filter_query (Dict[str, Any]): Filter query for documents to be deleted.
        """
        self.transaction_buffer.append({'type': 'delete_one', 'filter_query': filter_query, "transactionId": self.transaction_id})

    def delete_many(self, filter_query: Dict[str, Any]) -> None:
        """
        Buffer delete_many operation for the transaction.

        Args:
            filter_query (Dict[str, Any]): Filter query for documents to be deleted.
        """
        self.transaction_buffer.append({'type': 'delete_many', 'filter_query': filter_query, "transactionId": self.transaction_id})

    def find_one(self, filter_query: Dict[str, Any], projection: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        """
        Perform find_one operation within the transaction.

        Args:
            filter_query (Dict[str, Any]): Filter query for the find operation.
            projection (Optional[List[str]], optional): List of field names to include or exclude in the result. Defaults to None.

        Returns:
            Optional[Dict[str, Any]]: The found document, or None if not found.
        """
        return self.collection.find_one(filter_query, projection)

    def count_documents(self, filter_query: Dict[str, Any]) -> int:
        """
        Perform count_documents operation within the transaction.

        Args:
            filter_query (Dict[str, Any]): Filter query for the count operation.

        Returns:
            int: The count of documents matching the filter.
        """
        return self.collection.count_documents(filter_query)

    def aggregate(self, pipeline: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Perform aggregation within the transaction.

        Args:
            pipeline (List[Dict[str, Any]], optional): The aggregation pipeline. Defaults to None.

        Returns:
            List[Dict[str, Any]]: The result of the aggregation.
        """
        return self.collection.aggregation(pipeline)

    def _execute_operation(self, operation: Dict[str, Any]) -> None:
        """
        Execute individual operation within the transaction.

        Args:
            operation (Dict[str, Any]): The operation to execute.
        """
        operation_type = operation['type']
        if operation_type == 'insert_one':
            self.collection.insert_one(operation['data'])
        elif operation_type == 'insert_many':
            self.collection.insert_many(operation['data_list'])
        elif operation_type == 'update_one':
            self.collection.update_one(operation['filter_query'], operation['update_data'])
        elif operation_type == 'delete_one':
            self.collection.delete_one(operation['filter_query'])
        elif operation_type == 'delete_many':
            self.collection.delete_many(operation['filter_query'])
