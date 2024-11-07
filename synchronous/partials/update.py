import json

from typing import Any, Dict, Union
from ...exception.base import ElectrusException
from .result import DatabaseActionResult
from .operators import ElectrusLogicalOperators, ElectrusUpdateOperators 

class UpdateData:
    @classmethod
    def update(
        cls,
        collection_path: str,
        filter_query: Dict[str, Any],
        update_data: Dict[str, Any],
        multi: bool = False
    ) -> DatabaseActionResult:
        try:
            collection_data = cls._read_collection_data(collection_path)

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
                cls._write_collection_data(collection_path, collection_data)
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
    def _read_collection_data(collection_path: str) -> list[Dict[str, Any]]:
        with open(collection_path, 'r') as file:
            return json.loads(file.read())

    @staticmethod
    def _write_collection_data(collection_path: str, collection_data: list[Dict[str, Any]]) -> None:
        with open(collection_path, 'w') as file:
            file.write(json.dumps(collection_data, indent=4))

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