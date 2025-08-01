import json
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from pathlib import Path
from datetime import datetime
from bisect import bisect_left, bisect_right, insort
from collections import defaultdict
import time

from .operators import ElectrusLogicalOperators
from ..exception.base import ElectrusException
from .results import DatabaseActionResult
from ..handler.filemanager import JsonFileHandler


class IndexedCollection:
    def __init__(self, docs: List[Dict[str, Any]], id_field: str = "_id"):
        self._id_field = id_field
        self.data = {doc[id_field]: doc for doc in docs}
        self._hash_index: Dict[str, Dict[Any, set]] = {}
        self._sorted_index: Dict[str, List[Tuple[Any, Any]]] = {}

    def build_hash(self, field: str):
        idx = defaultdict(set)
        for did, doc in self.data.items():
            idx[doc.get(field)].add(did)
        self._hash_index[field] = idx

    def build_sorted(self, field: str):
        arr = [(doc.get(field), did) for did, doc in self.data.items()]
        arr.sort(key=lambda x: x[0])
        self._sorted_index[field] = arr

    def lookup_eq(self, field: str, value: Any) -> set:
        if field not in self._hash_index:
            self.build_hash(field)
        return set(self._hash_index[field].get(value, set()))

    def lookup_range(self, field: str, low: Any, high: Any) -> set:
        if field not in self._sorted_index:
            self.build_sorted(field)
        arr = self._sorted_index[field]
        lo = bisect_left(arr, (low, None))
        hi = bisect_right(arr, (high, None))
        return {did for _, did in arr[lo:hi]}

    def remove(self, to_del: List[Any]):
        for did in to_del:
            self.data.pop(did, None)
        for idx in self._hash_index.values():
            for s in idx.values():
                s.difference_update(to_del)
        for field, arr in self._sorted_index.items():
            self._sorted_index[field] = [(v, d) for v, d in arr if d not in to_del]


class DeleteData:
    def __init__(self, handler: JsonFileHandler, collection: Union[str, Path]):
        self._handler = handler
        self._file = Path(collection).name
        self._path = Path(collection)
        self._filter: Dict[str, Any] = {}
        self._sort: List[Tuple[str, int]] = []
        self._limit: Optional[int] = None
        self._return_docs: bool = False
        self._soft: bool = False
        self._txn: bool = False
        self._dry: bool = False
        self._retry: int = 0
        self._batch: Optional[int] = None
        self._audit: bool = False
        self._before: Optional[Callable[[List[Dict[str, Any]]], None]] = None
        self._after: Optional[Callable[[List[Dict[str, Any]]], None]] = None
        self._evaluator = ElectrusLogicalOperators()
        self._indexed: Optional[IndexedCollection] = None

    def where(self, **conds: Any) -> "DeleteData":
        self._filter.update(conds)
        return self

    def order_by(self, *fields: str) -> "DeleteData":
        for f in fields:
            if f.startswith("-"):
                self._sort.append((f[1:], -1))
            else:
                self._sort.append((f, 1))
        return self

    def limit(self, n: int) -> "DeleteData":
        self._limit = n
        return self

    def batch(self, size: int) -> "DeleteData":
        self._batch = size
        return self

    def returning(self) -> "DeleteData":
        self._return_docs = True
        return self

    def soft(self) -> "DeleteData":
        self._soft = True
        return self

    def transaction(self) -> "DeleteData":
        self._txn = True
        return self

    def dry_run(self) -> "DeleteData":
        self._dry = True
        return self

    def retry(self, times: int) -> "DeleteData":
        self._retry = times
        return self

    def audit(self) -> "DeleteData":
        self._audit = True
        return self

    def before_delete(self, fn: Callable[[List[Dict[str, Any]]], None]) -> "DeleteData":
        self._before = fn
        return self

    def after_delete(self, fn: Callable[[List[Dict[str, Any]]], None]) -> "DeleteData":
        self._after = fn
        return self

    async def execute(self) -> DatabaseActionResult:
        attempts = 0
        while True:
            try:
                # load & index
                if self._indexed is None:
                    res = await self._handler.read_async(self._file, verify_integrity=False)
                    self._indexed = IndexedCollection(res["data"])
                coll = self._indexed

                # filter
                cand: Optional[set] = None
                for f, v in self._filter.items():
                    if not isinstance(v, dict):  # Use index-based equality lookup only for simple values
                        ids = coll.lookup_eq(f, v)
                        cand = ids if cand is None else cand & ids
                # Fallback to full scan if needed
                if cand is None:
                    cand = {did for did, doc in coll.data.items()
                            if self._evaluator.evaluate(doc, self._filter)}

                if not cand:
                    return DatabaseActionResult(
                        success=False,
                        operation_type="delete",
                        deleted_count=0,
                        error="No matching documents found for deletion"
                    )

                # sort
                if self._sort:
                    fld, dir = self._sort[0]
                    if fld not in coll._sorted_index:
                        coll.build_sorted(fld)
                    arr = coll._sorted_index[fld]
                    ordered = [did for _, did in arr if did in cand]
                    if dir < 0:
                        ordered.reverse()
                else:
                    ordered = list(cand)

                # limit & batch
                to_del = ordered
                if self._limit is not None:
                    to_del = to_del[: self._limit]
                if self._batch is not None:
                    to_del = to_del[: self._batch]

                docs = [coll.data[did] for did in to_del]

                # before hook
                if self._before:
                    self._before(docs)
                # audit log
                if self._audit:
                    with open(self._file + ".audit.log", "a") as f:
                        for d in docs:
                            f.write(f"{datetime.utcnow().isoformat()} DELETE {d}\n")

                # dry run
                if self._dry:
                    return DatabaseActionResult(
                        success=True,
                        operation_type="delete",
                        deleted_count=len(docs)
                    )

                # perform deletes
                if self._soft:
                    ts = datetime.utcnow().isoformat()
                    for d in docs:
                        coll.data[d["_id"]]["_deleted_at"] = ts
                    updated = list(coll.data.values())
                else:
                    coll.remove(to_del)
                    updated = list(coll.data.values())

                # write to file
                if self._txn:
                    tmp = self._file + ".tmp"
                    await self._handler.write_async(
                        tmp, updated, create_version=False, verify_integrity=True
                    )
                    Path(self._path.parent / tmp).replace(self._path)
                else:
                    await self._handler.write_async(
                        self._file, updated, create_version=True, verify_integrity=True
                    )

                # after hook
                if self._after:
                    self._after(docs)

                # return results
                if self._return_docs:
                    return DatabaseActionResult(
                        success=True,
                        operation_type="delete",
                        deleted_count=len(docs)
                    )

                ids = [d["_id"] for d in docs]
                return DatabaseActionResult(
                    success=True,
                    operation_type="delete",
                    deleted_count=len(docs)
                )

            except Exception as e:
                if attempts < self._retry:
                    attempts += 1
                    time.sleep(2 ** attempts)
                    continue
                raise ElectrusException(f"Delete error after {attempts} retries: {e}")