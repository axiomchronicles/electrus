from typing import Any, Dict, List, Union, Optional

class DatabaseActionResult:
    def __init__(self, success: bool, inserted_ids: Optional[Union[str, List[str]]] = None, modified_count: int = 0, upserted_id: Optional[str] = None, matched_count: int = 0, deleted_count: int = 0, data: Dict[str, Any] = {}) -> None:
        """
        Represents the result of a database operation.

        This class encapsulates the outcome of a database operation,
        including information about the success of the operation,
        the IDs of inserted/upserted documents, and counts of modified,
        matched, and deleted documents.

        Args:
            success (bool): Indicates whether the operation was successful.
            inserted_ids (Optional[Union[str, List[str]]], optional): IDs of inserted documents, if any. Defaults to None.
            modified_count (int, optional): Count of modified documents. Defaults to 0.
            upserted_id (Optional[str], optional): ID of the upserted document, if any. Defaults to None.
            matched_count (int, optional): Count of matched documents. Defaults to 0.
            deleted_count (int, optional): Count of deleted documents. Defaults to 0.
            data (Dict[str, Any], optional): Additional data associated with the result. Defaults to {}.
        """
        self._success = success
        self._inserted_ids = inserted_ids
        self._modified_count = modified_count
        self._upserted_id = upserted_id
        self._matched_count = matched_count
        self._deleted_count = deleted_count
        self._data = data

    @property
    def data(self) -> Dict[str, Any]:
        """
        Returns any additional data associated with the result.

        Returns:
            Dict[str, Any]: Additional data associated with the result.
        """
        return self._data

    @property
    def success(self) -> bool:
        """
        Checks if the operation was successful.

        Returns:
            bool: True if the operation was successful, False otherwise.
        """
        return self._success

    @property
    def object_id(self) -> Optional[Union[str, List[str]]]:
        """
        Returns the inserted ID(s) if available.

        Returns:
            Optional[Union[str, List[str]]]: The inserted ID(s) if available, None otherwise.
        """
        return self._inserted_ids

    @property
    def modified_count(self) -> int:
        """
        Returns the count of modified documents.

        Returns:
            int: The count of modified documents.
        """
        return self._modified_count

    @property
    def upserted_id(self) -> Optional[str]:
        """
        Returns the upserted ID if available.

        Returns:
            Optional[str]: The upserted ID if available, None otherwise.
        """
        return self._upserted_id

    @property
    def matched_count(self) -> int:
        """
        Returns the count of matched documents.

        Returns:
            int: The count of matched documents.
        """
        return self._matched_count

    @property
    def deleted_count(self) -> int:
        """
        Returns the count of deleted documents.

        Returns:
            int: The count of deleted documents.
        """
        return self._deleted_count

    @property
    def is_successful(self) -> bool:
        """
        Alias for success property.

        Returns:
            bool: True if the operation was successful, False otherwise.
        """
        return self.success

    @property
    def upserted_id(self) -> Optional[str]:
        """
        Alias for upserted_id property.

        Returns:
            Optional[str]: The upserted ID if available, None otherwise.
        """
        return self._upserted_id

    @property
    def matched_count(self) -> int:
        """
        Alias for matched_count property.

        Returns:
            int: The count of matched documents.
        """
        return self._matched_count

    @property
    def modified_count(self) -> int:
        """
        Alias for modified_count property.

        Returns:
            int: The count of modified documents.
        """
        return self._modified_count

    @property
    def inserted_ids(self) -> Optional[Union[str, List[str]]]:
        """
        Alias for inserted_ids property.

        Returns:
            Optional[Union[str, List[str]]]: The inserted ID(s) if available, None otherwise.
        """
        return self._inserted_ids

    @property
    def deleted_count(self) -> int:
        """
        Alias for deleted_count property.

        Returns:
            int: The count of deleted documents.
        """
        return self._deleted_count
    
    def __str__(self) -> str:
        """
        Returns a string representation of the inserted ID(s).

        Returns:
            str: String representation of the inserted ID(s).
        """
        return str(self.object_id)
    
    def __repr__(self) -> str:
        """
        Returns a string representation of the DatabaseActionResult object.

        Returns:
            str: String representation of the DatabaseActionResult object.
        """
        return "DatabaseActionResult(%s)" %(self.object_id)
