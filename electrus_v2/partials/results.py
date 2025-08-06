from typing import Any, Dict, List, Union, Optional, Mapping, Sequence
from abc import ABC, abstractmethod


class DatabaseError(Exception):
    """Base exception for database operation errors."""
    
    def __init__(self, message: str, code: Optional[int] = None, 
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.code = code
        self.details = details or {}


class BaseResult(ABC):
    """Base class for all database operation results."""
    
    def __init__(self, acknowledged: bool = True, 
                 raw_result: Optional[Mapping[str, Any]] = None):
        self._acknowledged = acknowledged
        self._raw_result = raw_result or {}
    
    @property
    def acknowledged(self) -> bool:
        """Is this the result of an acknowledged write operation?"""
        return self._acknowledged
    
    @acknowledged.setter
    def acknowledged(self, value: bool) -> None:
        self._acknowledged = value
    
    @property
    def raw_result(self) -> Mapping[str, Any]:
        """The raw result document returned by the server."""
        return self._raw_result
    
    def __bool__(self) -> bool:
        """Returns True if the operation was acknowledged and successful."""
        return self.acknowledged


class InsertOneResult(BaseResult):
    """Result of an insert_one operation (MongoDB-like)."""
    
    def __init__(self, inserted_id: Any, acknowledged: bool = True,
                 raw_result: Optional[Mapping[str, Any]] = None):
        super().__init__(acknowledged, raw_result)
        self._inserted_id = inserted_id
    
    @property
    def inserted_id(self) -> Any:
        """The _id of the inserted document."""
        return self._inserted_id


class UpdateResult(BaseResult):
    """Result of update_one/update_many operations (MongoDB-like)."""
    
    def __init__(self, matched_count: int, modified_count: int,
                 upserted_id: Optional[Any] = None, acknowledged: bool = True,
                 raw_result: Optional[Mapping[str, Any]] = None):
        super().__init__(acknowledged, raw_result)
        self._matched_count = matched_count
        self._modified_count = modified_count
        self._upserted_id = upserted_id
    
    @property
    def matched_count(self) -> int:
        """The number of documents matched for this update."""
        return self._matched_count
    
    @property
    def modified_count(self) -> int:
        """The number of documents modified."""
        return self._modified_count
    
    @property
    def upserted_id(self) -> Optional[Any]:
        """The _id of the inserted document if an upsert took place."""
        return self._upserted_id
    
    @property
    def did_upsert(self) -> bool:
        """Whether an upsert took place."""
        return self._upserted_id is not None


class DeleteResult(BaseResult):
    """Result of delete_one/delete_many operations (MongoDB-like)."""
    
    def __init__(self, deleted_count: int, acknowledged: bool = True,
                 raw_result: Optional[Mapping[str, Any]] = None):
        super().__init__(acknowledged, raw_result)
        self._deleted_count = deleted_count
    
    @property
    def deleted_count(self) -> int:
        """The number of documents deleted."""
        return self._deleted_count


class DatabaseActionResult(BaseResult):
    """Enhanced universal database operation result class."""
    
    def __init__(self,
                 success: bool,
                 operation_type: str = "unknown",
                 # Insert-related
                 inserted_id: Optional[Any] = None,
                 inserted_ids: Optional[List[Any]] = None,
                 inserted_count: Optional[int] = None,
                 # Update-related
                 matched_count: int = 0,
                 modified_count: int = 0,
                 upserted_id: Optional[Any] = None,
                 upserted_count: int = 0,
                 upserted_ids: Optional[Union[List[Any], Dict[int, Any]]] = None,
                 # Delete-related
                 deleted_count: int = 0,
                 # General
                 acknowledged: bool = True,
                 raw_result: Optional[Mapping[str, Any]] = None,
                 error: Optional[DatabaseError] = None,
                 additional_data: Optional[Dict[str, Any]] = None) -> None:
        """
        Enhanced database operation result supporting multiple operation types.
        
        Args:
            success: Whether the operation was successful
            operation_type: Type of operation ("insert", "update", "delete", "bulk", etc.)
            inserted_id: Single inserted document ID
            inserted_ids: List of inserted document IDs
            inserted_count: Number of documents inserted
            matched_count: Number of documents matched for update
            modified_count: Number of documents modified
            upserted_id: ID of upserted document
            upserted_count: Number of documents upserted
            upserted_ids: Map of upserted document IDs
            deleted_count: Number of documents deleted
            acknowledged: Whether operation was acknowledged
            raw_result: Raw server response
            error: Error information if operation failed
            additional_data: Additional operation-specific data
        """
        super().__init__(acknowledged, raw_result)
        
        # Core properties
        self._success = success
        self._operation_type = operation_type
        self._error = error
        
        # Insert properties
        self._inserted_id = inserted_id
        self._inserted_ids = inserted_ids or []
        self._inserted_count = inserted_count
        
        # Update properties
        self._matched_count = matched_count
        self._modified_count = modified_count
        self._upserted_id = upserted_id
        self._upserted_count = upserted_count
        self._upserted_ids = upserted_ids or {}
        
        # Delete properties
        self._deleted_count = deleted_count
        
        # Additional data (using None to avoid mutable default)
        self._additional_data = additional_data or {}
        
        # Validation
        self._validate_parameters()
    
    def _validate_parameters(self) -> None:
        """Validate parameter consistency."""
        if not self._success and self._error is None:
            raise ValueError("Failed operations must include error information")
        
        if self._success and self._error is not None:
            raise ValueError("Successful operations should not include error information")
        
        # Auto-calculate inserted_count if not provided
        if self._inserted_count is None and self._inserted_ids:
            self._inserted_count = len(self._inserted_ids)
    
    @property
    def success(self) -> bool:
        """Whether the operation was successful."""
        return self._success
    
    @success.setter
    def success(self, value: bool) -> None:
        self._success = value
    
    @property
    def is_successful(self) -> bool:
        """Alias for success property."""
        return self.success
    
    @property
    def operation_type(self) -> str:
        """Type of database operation performed."""
        return self._operation_type
    
    @property
    def error(self) -> Optional[DatabaseError]:
        """Error information if operation failed."""
        return self._error
    
    @error.setter
    def error(self, value: Optional[DatabaseError]) -> None:
        self._error = value

    @property
    def inserted_id(self) -> Optional[Any]:
        """The _id of the inserted document (for single inserts)."""
        return self._inserted_id
    
    @property
    def inserted_ids(self) -> List[Any]:
        """List of _ids of inserted documents."""
        return self._inserted_ids
    
    @property
    def inserted_count(self) -> int:
        """Number of documents inserted."""
        return self._inserted_count or len(self._inserted_ids)
    
    @property
    def matched_count(self) -> int:
        """Number of documents matched for update."""
        return self._matched_count
    
    @property
    def modified_count(self) -> int:
        """Number of documents modified."""
        return self._modified_count
    
    @property
    def upserted_id(self) -> Optional[Any]:
        """ID of upserted document."""
        return self._upserted_id
    
    @property
    def deleted_count(self) -> int:
        """Number of documents deleted."""
        return self._deleted_count
    
    def __bool__(self) -> bool:
        """Returns True if operation was acknowledged and successful."""
        return super().__bool__() and self.success
    
    def __str__(self) -> str:
        """String representation focusing on primary result."""
        if self._operation_type == "insert":
            return str(self.inserted_id or self.inserted_ids)
        elif self._operation_type == "update":
            return f"Updated {self.modified_count} document(s)"
        elif self._operation_type == "delete":
            return f"Deleted {self.deleted_count} document(s)"
        else:
            return f"Operation result: {self.success}"
    
    def __repr__(self) -> str:
        """Detailed representation of the result."""
        return (f"DatabaseActionResult(success={self.success}, "
                f"operation='{self.operation_type}', "
                f"acknowledged={self.acknowledged})")
    
    # Factory methods for easy creation
    @classmethod
    def insert_success(cls, inserted_id: Any = None, inserted_ids: List[Any] = None,
                      raw_result: Optional[Mapping[str, Any]] = None) -> 'DatabaseActionResult':
        """Create a successful insert result."""
        return cls(
            success=True,
            operation_type="insert",
            inserted_id=inserted_id,
            inserted_ids=inserted_ids,
            raw_result=raw_result
        )
    
    @classmethod
    def failure(cls, error: DatabaseError, operation_type: str = "unknown",
               raw_result: Optional[Mapping[str, Any]] = None) -> 'DatabaseActionResult':
        """Create a failed operation result."""
        return cls(
            success=False,
            operation_type=operation_type,
            error=error,
            raw_result=raw_result
        )
