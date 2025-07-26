import re
import random
import string
import logging
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union
from datetime import datetime, timedelta

from .objectId import ObjectId
from ..exception.base import ElectrusException
from .results import InsertOneResult, DatabaseActionResult, DatabaseError
from ..handler.filemanager import JsonFileHandler  

JsonValue = Union[str, int, Dict[str, Any]]
Processor = Callable[[str, JsonValue, List[Dict[str, Any]]], Any]

logger = logging.getLogger("InsertData")
logger.setLevel(logging.INFO)

class FieldOp(Enum):
    AUTO_INC    = "$auto"
    UNIQUE      = "$unique"
    TIME        = "$time"
    DATETIME    = "$datetime"
    DATE        = "$date"
    TIMESTAMP   = "$timestamp"
    DATE_ADD    = "$date_add"
    DATE_SUB    = "$date_sub"
    DATE_DIFF   = "$date_diff"
    DATE_FMT    = "$date_format"


class InsertData:
    def __init__(
        self,
        collection_file: Union[str, Any],
        json_handler: JsonFileHandler
    ) -> None:
        self._file = collection_file
        self._jf = json_handler
        self._registry: Dict[FieldOp, Processor] = {
            FieldOp.AUTO_INC: self._process_auto_inc,
            FieldOp.UNIQUE: self._process_unique_id,
            FieldOp.TIME: self._process_time_now,
            FieldOp.DATETIME: self._process_datetime,
            FieldOp.DATE: self._process_date,
            FieldOp.TIMESTAMP: self._process_timestamp,
            FieldOp.DATE_ADD: self._process_date_delta,
            FieldOp.DATE_SUB: self._process_date_delta,
            FieldOp.DATE_DIFF: self._process_date_diff,
            FieldOp.DATE_FMT: self._process_date_format,
        }

    # --- Processors ---
    async def _gen_id(self, kind: str, length: int = 10) -> str:
        if kind == "uuid":
            return str(ObjectId.generate())
        charset_map = {
            "numeric": string.digits,
            "alphanumeric": string.ascii_letters + string.digits,
            "default": string.ascii_letters + string.digits,
        }
        charset = charset_map.get(kind, charset_map["default"])
        return ''.join(random.choices(charset, k=length))

    async def _process_auto_inc(self, key: str, raw: JsonValue, coll: List[Dict[str, Any]]) -> int:
        nums = [item.get(key, 0) for item in coll if isinstance(item.get(key), int)]
        return max(nums, default=0) + 1

    async def _process_unique_id(self, key: str, raw: JsonValue, coll: List[Dict[str, Any]]) -> str:
        spec = raw if isinstance(raw, dict) else {}
        length = spec.get('length', 10)
        fmt = spec.get('format', 'default')
        return await self._gen_id(fmt, length)

    async def _process_time_now(self, key: str, raw: JsonValue, coll: List[Dict[str, Any]]) -> str:
        return datetime.now().strftime('%H:%M:%S')

    async def _process_datetime(self, key: str, raw: JsonValue, coll: List[Dict[str, Any]]) -> str:
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    async def _process_date(self, key: str, raw: JsonValue, coll: List[Dict[str, Any]]) -> str:
        return datetime.now().strftime('%Y-%m-%d')

    async def _process_timestamp(self, key: str, raw: JsonValue, coll: List[Dict[str, Any]]) -> int:
        return int(datetime.now().timestamp())

    async def _process_date_delta(self, key: str, raw: JsonValue, coll: List[Dict[str, Any]]) -> str:
        op_str, spec = next(iter(raw.items()))
        m = re.match(r'(-?\d+)([A-Za-z]+)', spec)
        if not m:
            raise ElectrusException(f"Invalid delta spec: {spec}")
        n, unit = int(m.group(1)), m.group(2).lower()
        now = datetime.now()
        try:
            delta = timedelta(**{unit.rstrip('s'): n})
        except Exception as e:
            raise ElectrusException(f"Unsupported timedelta unit '{unit}': {e}")
        result = now + delta if op_str == FieldOp.DATE_ADD.value else now - delta
        fmt = '%Y-%m-%d' if unit in ('day', 'days') else '%Y-%m-%d %H:%M:%S'
        return result.strftime(fmt)

    async def _process_date_diff(self, key: str, raw: JsonValue, coll: List[Dict[str, Any]]) -> int:
        params = raw.get(FieldOp.DATE_DIFF.value, {})
        sd, ed = params.get('start_date'), params.get('end_date')
        try:
            start = datetime.strptime(sd, '%Y-%m-%d')
            end = datetime.strptime(ed, '%Y-%m-%d')
        except Exception as e:
            raise ElectrusException(f"Invalid date for diff: {e}")
        return (end - start).days

    async def _process_date_format(self, key: str, raw: JsonValue, coll: List[Dict[str, Any]]) -> str:
        params = raw.get(FieldOp.DATE_FMT.value, {})
        dt_str = params.get('date')
        fmt = params.get('format', '%Y-%m-%d')
        try:
            dt = datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')
        except Exception as e:
            raise ElectrusException(f"Invalid datetime string for format: {e}")
        return dt.strftime(fmt)

    async def _apply_processors(self, data: Dict[str, JsonValue], coll: List[Dict[str, Any]]) -> None:
        for key, raw in list(data.items()):
            try:
                if raw == FieldOp.AUTO_INC.value:
                    data[key] = await self._registry[FieldOp.AUTO_INC](key, raw, coll)
                elif isinstance(raw, dict):
                    op = FieldOp(next(iter(raw.keys())))
                    data[key] = await self._registry[op](key, raw, coll)
                elif raw in {
                    FieldOp.TIME.value,
                    FieldOp.DATETIME.value,
                    FieldOp.DATE.value,
                    FieldOp.TIMESTAMP.value
                }:
                    op = FieldOp(raw)
                    data[key] = await self._registry[op](key, raw, coll)
            except Exception as e:
                logger.error(f"Processor failed for key '{key}': {e}")
                raise ElectrusException(f"Field '{key}' processor error: {str(e)}") from e

    async def _safe_read(self) -> dict:
        """Robustly reads collection, handles missing file gracefully."""
        try:
            read = getattr(self._jf, "read_async", None)
            if callable(read):
                existing = await read(self._file)
            else:
                existing = self._jf.read(self._file, True)

            if not isinstance(existing, dict) or "data" not in existing:
                logger.warning(f"File '{self._file}' is malformed or missing 'data' key. Resetting.")
                return {"data": []}

            return existing

        except FileNotFoundError:
            logger.warning(f"File '{self._file}' not found. Creating new empty collection.")
            return {"data": []}

        except (OSError, IOError) as io_err:
            logger.error(f"File I/O error while reading '{self._file}': {io_err}")
            raise DatabaseError(f"File read failure: {io_err}", details={"file": self._file})

        except Exception as e:
            logger.error(f"Unexpected exception during read: {e}")
            raise DatabaseError(f"General read error: {e}", details={"file": self._file})


    async def _safe_write(self, data: List[Dict[str, Any]]):
        """Robust write with sync/async handling and error reporting."""
        try:
            write = getattr(self._jf, "write_async", None)
            if callable(write):
                await write(self._file, data)
            else:
                self._jf.write(self._file, data)
        except (OSError, IOError) as io_err:
            logger.error(f"File I/O error during write '{self._file}': {io_err}")
            raise DatabaseError(f"File write failure: {io_err}", details={"file": self._file})
        except Exception as e:
            logger.error(f"Write failure: {e}")
            raise DatabaseError(f"General write error: {e}", details={"file": self._file})

    def _validate_document(self, data: dict) -> None:
        if not isinstance(data, dict):
            raise DatabaseError("Inserted object must be a dictionary", code=422)
        # Require at least one field (aside from _id), could expand with schema checks
        if not any(k != "_id" for k in data.keys()):
            raise DatabaseError("Empty document is not allowed", code=422)

    async def _update_collection_data(
        self, data: Dict[str, JsonValue], overwrite: bool = False
    ) -> DatabaseActionResult:
        """
        Insert or overwrite a document. Provides robust error handling.
        Returns a DatabaseActionResult.
        """
        try:
            self._validate_document(data)
            # Defensive copy to avoid mutating caller's dict
            insert_data = dict(data)
            existing = await self._safe_read()
            coll = existing.get("data", [])
            # Generate an _id if not present
            if "_id" not in insert_data:
                insert_data["_id"] = ObjectId.generate()
            await self._apply_processors(insert_data, coll)

            # Avoid duplicate insertion
            is_duplicate = any(doc for doc in coll if doc == insert_data)
            if is_duplicate and not overwrite:
                # No-op with acknowledgment
                return DatabaseActionResult.insert_success(
                    inserted_id=None,
                    inserted_ids=[],
                    raw_result={"warning": "duplicate"}
                )

            if overwrite:
                updated = False
                for idx, doc in enumerate(coll):
                    if doc.get("_id") == insert_data["_id"]:
                        coll[idx] = insert_data
                        updated = True
                        break
                if not updated:
                    coll.append(insert_data)
            else:
                coll.append(insert_data)

            await self._safe_write(coll)

            return DatabaseActionResult.insert_success(
                inserted_id=insert_data["_id"],
                inserted_ids=[insert_data["_id"]],
                raw_result={"operation": "insert", "success": True}
            )
        except DatabaseError as db_err:
            logger.error(f"Database error: {db_err}")
            return DatabaseActionResult.failure(
                db_err, operation_type="insert"
            )
        except ElectrusException as e:
            logger.error(f"Data validation/processor error: {e}")
            err = DatabaseError(str(e), code=422)
            return DatabaseActionResult.failure(
                err, operation_type="insert"
            )
        except Exception as e:
            logger.critical(f"Uncaught error during insert: {e}", exc_info=True)
            err = DatabaseError(f"Unexpected error: {e}")
            return DatabaseActionResult.failure(
                err, operation_type="insert"
            )

    async def _obl_one(
        self, data: Dict[str, JsonValue], overwrite: bool = False
    ) -> DatabaseActionResult:
        """Insert a single document with error reporting."""
        
        return await self._update_collection_data(data, overwrite)

    async def _obl_many(
        self, data_list: List[Dict[str, JsonValue]], overwrite: bool = False
    ) -> DatabaseActionResult:
        """
        Bulk insert/overwrite.
        Returns all inserted ids or error if the first failure occurs.
        """
        inserted_ids: List[Any] = []
        try:
            for idx, doc in enumerate(data_list):
                result = await self._update_collection_data(doc, overwrite)
                if not result.success:
                    logger.error(f"Bulk insert aborts on error (index {idx}): {result.error}")
                    # You could decide to skip failures, but better to abort on first error for data integrity
                    return result
                if result.inserted_id:
                    inserted_ids.append(result.inserted_id)
            return DatabaseActionResult.insert_success(
                inserted_ids=inserted_ids,
                raw_result={"operation": "bulk_insert", "success": True}
            )
        except Exception as e:
            logger.critical(f"Uncaught error during bulk insert: {e}", exc_info=True)
            err = DatabaseError(f"Unexpected error in bulk insert: {e}")
            return DatabaseActionResult.failure(err, operation_type="bulk_insert")
