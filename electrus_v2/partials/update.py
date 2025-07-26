import json
import aiofiles

from typing import Any, Dict, Union
from ..exception.base import ElectrusException
from .results import DatabaseActionResult
from .operators import ElectrusLogicalOperators, ElectrusUpdateOperators

class UpdateData:
    @classmethod
    async def update(
        cls,
        collection_path: str,
        filter_query: Dict[str, Any],
        update_data: Dict[str, Any],
        multi: bool = False
    ) -> DatabaseActionResult:
        """
        Update documents in a collection.

        This method updates documents in a JSON file representing a collection.
        It iterates over the collection, applying the given filter to select the documents to update,
        and then modifies them according to the update data.

        Args:
            collection_path (str): The path to the collection file.
            filter_query (Dict[str, Any]): The query used to filter documents for updating.
            update_data (Dict[str, Any]): The data to be updated in the matching documents.
            multi (bool, optional): If True, updates multiple documents. Defaults to False.

        Returns:
            DatabaseActionResult: An object containing the result of the update operation.
        Raises:
            ElectrusException: If there's an error accessing or modifying the collection file.
        """
        try:
            collection_data = await cls._read_collection_data(collection_path)

            updated = False
            modified_count = 0
            updated_ids = []
            
            for item in collection_data:
                if ElectrusLogicalOperators().evaluate(item, filter_query):
                    ElectrusUpdateOperators().evaluate(item, update_data)
                    updated = True
                    modified_count += 1
                    updated_ids.append(item.get('_id'))

                    if not multi:
                        break

            if updated:
                await cls._write_collection_data(collection_path, collection_data)
                updated_ids = cls._format_updated_ids(updated_ids)
                return DatabaseActionResult(success=True, modified_count=modified_count, inserted_ids=updated_ids)
            else:
                return DatabaseActionResult(success=False)

        except FileNotFoundError:
            raise ElectrusException(f"File not found at path: {collection_path}")
        except json.JSONDecodeError as je:
            raise ElectrusException(f"Error decoding JSON: {je}")
        except Exception as e:
            raise ElectrusException(f"Error updating documents: {e}")

    @staticmethod
    async def _read_collection_data(collection_path: str) -> list[Dict[str, Any]]:
        """
        Read collection data from a file.

        This method reads the JSON data from the specified file and returns it as a list of dictionaries,
        each representing a document in the collection.

        Args:
            collection_path (str): The path to the collection file.

        Returns:
            list[Dict[str, Any]]: A list of dictionaries representing documents in the collection.
        Raises:
            ElectrusException: If there's an error reading the collection file.
        """
        async with aiofiles.open(collection_path, 'r') as file:
            return json.loads(await file.read())

    @staticmethod
    async def _write_collection_data(collection_path: str, collection_data: list[Dict[str, Any]]) -> None:
        """
        Write collection data to a file.

        This method writes the provided collection data, represented as a list of dictionaries,
        to the specified file in JSON format.

        Args:
            collection_path (str): The path to the collection file.
            collection_data (list[Dict[str, Any]]): The collection data to be written.
        Raises:
            ElectrusException: If there's an error writing the collection data to the file.
        """
        async with aiofiles.open(collection_path, 'w') as file:
            await file.write(json.dumps(collection_data, indent=4))

    @staticmethod
    def _format_updated_ids(updated_ids: list) -> Union[str, list]:
        """
        Format updated document IDs.

        This method formats the list of updated document IDs for presentation.
        If there's only one updated document, it returns the ID as a string.
        Otherwise, it returns the list of IDs.

        Args:
            updated_ids (list): A list of updated document IDs.

        Returns:
            Union[str, list]: The formatted IDs (either a single ID as a string or a list of IDs).
        """
        if len(updated_ids) == 1:
            return updated_ids[0]
        return updated_ids
