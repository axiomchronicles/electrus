import json
import uuid
import bisect
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Set, Tuple, Callable, Union
from datetime import datetime

from ..exception.base import ElectrusException
from .results import DatabaseActionResult
from ..handler.filemanager import JsonFileHandler

# --- Trie for prefix text search ---
class TrieNode:
    def __init__(self):
        self.children: Dict[str, "TrieNode"] = {}
        self.ids: Set[Any] = set()

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, text: str, doc_id: Any):
        node = self.root
        for ch in text:
            node = node.children.setdefault(ch, TrieNode())
            node.ids.add(doc_id)

    def search_prefix(self, prefix: str) -> Set[Any]:
        node = self.root
        for ch in prefix:
            node = node.children.get(ch)
            if not node:
                return set()
        return node.ids.copy()

# --- Bloom Filter using builtins and simple hash functions ---
class BloomFilter:
    def __init__(self, size: int = 10000, hash_count: int = 7):
        self.size = size
        self.hash_count = hash_count
        self.bit_array = 0  # Using integer bitmask; assumes size <= 64*int bits for demo

    def _hashes(self, item: str) -> List[int]:
        # Use built-in hash combined with offset for demo purpose (non-cryptographic)
        base_hash = hash(item)
        return [(base_hash + i * 0x9e3779b9) % self.size for i in range(self.hash_count)]

    def add(self, item: str):
        for h in self._hashes(item):
            self.bit_array |= 1 << h

    def __contains__(self, item: str) -> bool:
        return all((self.bit_array & (1 << h)) != 0 for h in self._hashes(item))

# --- Helper: Differential dict diff (recursive, built-in only) ---
def dict_diff(orig: dict, updated: dict, path="") -> dict:
    diffs = {}
    all_keys = set(orig.keys()) | set(updated.keys())
    for key in all_keys:
        orig_val = orig.get(key, None)
        updated_val = updated.get(key, None)
        current_path = f"{path}.{key}" if path else key

        if isinstance(orig_val, dict) and isinstance(updated_val, dict):
            sub_diff = dict_diff(orig_val, updated_val, current_path)
            if sub_diff:
                diffs[key] = sub_diff
        elif orig_val != updated_val:
            diffs[key] = updated_val
    return diffs

# --- Core UpdateData class ---
class UpdateData:
    def __init__(
        self,
        handler: JsonFileHandler,
        *,
        max_workers: int = 4,
        enable_differential_logging: bool = True,
        audit_log_path: Optional[str] = None,
        pre_update_hooks: Optional[List[Callable[[Dict[str, Any]], None]]] = None,
        post_update_hooks: Optional[List[Callable[[Dict[str, Any]], None]]] = None,
    ):
        self.handler = handler
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.enable_differential_logging = enable_differential_logging
        self.audit_log_path = audit_log_path
        self.pre_update_hooks = pre_update_hooks or []
        self.post_update_hooks = post_update_hooks or []

        # Indexes and caches
        self.sorted_index: List[Tuple[Any, Any]] = []  # e.g. (field_value, doc_id)
        self.id_to_doc: Dict[Any, Dict[str, Any]] = {}
        self.trie = Trie()
        self.bloom_filter = BloomFilter(size=10000, hash_count=7)

        self._subscribers: Set[asyncio.Queue] = set()  # For change streams

        # Handler for logical and update operations - assume injected or imported
        from .operators import ElectrusLogicalOperators, ElectrusUpdateOperators
        self.logic = ElectrusLogicalOperators()
        self.updater = ElectrusUpdateOperators()

    async def update(
        self,
        collection_path: str,
        filter_query: Dict[str, Any],
        update_data: Dict[str, Any],
        *,
        multi: bool = False,
        upsert: bool = False,
        upsert_doc: Optional[Dict[str, Any]] = None,
        return_updated_fields: Optional[List[str]] = None,
    ) -> DatabaseActionResult:
        try:
            docs = await self._load_and_index(collection_path)
        except ElectrusException as e:
            return self._failure_result("update", str(e))
        except Exception as e:
            return self._failure_result("update", f"Unexpected error: {e}")

        matched, modified = 0, 0
        updated_ids: List[Any] = []
        updated_docs: List[Dict[str, Any]] = []

        # Fast indexed-access prep
        id_index = self.id_to_doc
        single_id = filter_query.get('_id', None)
        early_exit = not multi and single_id is not None and single_id in id_index

        original_docs_backup: Dict[Any, Dict[str, Any]] = {}

        def snapshot_doc(doc_id: Any):
            if doc_id not in original_docs_backup:
                original_docs_backup[doc_id] = dict(self.id_to_doc[doc_id])  # shallow copy

        # Pre/Post hooks helper
        def run_hooks(doc: dict, hooks: List[Callable[[Dict[str, Any]], None]]):
            for hook in hooks:
                hook(doc)

        async def apply_updates(doc):
            run_hooks(doc, self.pre_update_hooks)
            modified_flag = self.updater.evaluate(doc, update_data)
            run_hooks(doc, self.post_update_hooks)
            return modified_flag

        updated_indexes = set()

        if early_exit:
            doc = id_index[single_id]
            snapshot_doc(single_id)
            if self.logic.evaluate(doc, filter_query):
                matched = 1
                if await asyncio.get_event_loop().run_in_executor(self.executor, lambda: self.updater.evaluate(doc, update_data)):
                    modified = 1
                    updated_ids.append(doc.get('_id'))
                    updated_docs.append(self._project_fields(doc, return_updated_fields))
        else:
            # Full or partial scan per query planner
            candidate_docs = await self._query_planner(filter_query)
            for doc in candidate_docs:
                doc_id = doc.get('_id')
                if doc_id is None:
                    continue
                if self.logic.evaluate(doc, filter_query):
                    matched += 1
                    snapshot_doc(doc_id)
                    was_modified = await asyncio.get_event_loop().run_in_executor(self.executor, lambda d=doc: self.updater.evaluate(d, update_data))
                    if was_modified:
                        modified += 1
                        updated_ids.append(doc_id)
                        updated_docs.append(self._project_fields(doc, return_updated_fields))
                    if not multi and modified > 0:
                        break

        upserted_id = None
        if matched == 0 and upsert:
            # Prepare new doc
            base_doc = dict(upsert_doc) if upsert_doc else self._project_for_upsert(filter_query)
            if '_id' not in base_doc:
                base_doc['_id'] = str(uuid.uuid4())
            # Apply updates to base_doc
            await asyncio.get_event_loop().run_in_executor(self.executor, lambda: self.updater.evaluate(base_doc, update_data))
            docs.append(base_doc)
            self._index_doc(base_doc)
            upserted_id = base_doc['_id']
            matched = 1
            modified = 1
            updated_ids.append(upserted_id)
            updated_docs.append(self._project_fields(base_doc, return_updated_fields))

        if modified or upserted_id:
            try:
                await self._write_collection(collection_path, docs)
            except Exception as e:
                return self._failure_result("update", f"Write failed: {e}")

            # Differential logging
            if self.enable_differential_logging and self.audit_log_path:
                await self._log_changes(updated_ids, original_docs_backup)

            # Notify subscribers (change streams)
            if updated_ids:
                await self._notify_change({
                    "type": "update",
                    "doc_ids": updated_ids,
                    "timestamp": datetime.utcnow().isoformat()
                })

            return DatabaseActionResult(
                success=True,
                operation_type="update",
                matched_count=matched,
                modified_count=modified,
                upserted_id=upserted_id,
                inserted_ids=updated_ids if multi else None,
                inserted_id=updated_ids[0] if not multi and updated_ids else None,
                additional_data={"updated_fields": updated_docs} if return_updated_fields else None,
            )

        return DatabaseActionResult(
            success=False,
            operation_type="update",
            matched_count=matched,
            modified_count=0,
            error=ElectrusException("No documents matched and no upsert performed"),
        )

    async def _load_and_index(self, collection_path: str) -> List[Dict[str, Any]]:
        result = await self.handler.read_async(collection_path)
        docs = result.get("data")
        if not isinstance(docs, list):
            raise ElectrusException("Collection content is not a list")

        self.id_to_doc.clear()
        self.sorted_index.clear()
        self.trie = Trie()
        self.bloom_filter = BloomFilter(size=10000, hash_count=7)

        for doc in docs:
            doc_id = doc.get('_id')
            if doc_id is not None:
                self.id_to_doc[doc_id] = doc

            # Example: indexing "age" field:
            age = doc.get("age")
            if isinstance(age, (int, float)):
                bisect.insort(self.sorted_index, (age, doc.get('_id')))

            # Text field indexing for "name" field example:
            name = doc.get("name", "")
            if isinstance(name, str) and name:
                self.bloom_filter.add(name.lower())
                for token in name.lower().split():
                    self.trie.insert(token, doc_id)

        return docs

    async def _query_planner(self, filter_query: Dict[str, Any]) -> List[Dict[str, Any]]:
        # Use indexes where appropriate:

        # Range query example with "age"
        if "age" in filter_query:
            age_filter = filter_query.get("age")
            if isinstance(age_filter, dict):
                lower = age_filter.get("$gte", float("-inf"))
                upper = age_filter.get("$lte", float("inf"))
                left_idx = bisect.bisect_left(self.sorted_index, (lower, ""))
                right_idx = bisect.bisect_right(self.sorted_index, (upper, chr(255)*10))  # max string high key
                candidate_ids = {doc_id for _, doc_id in self.sorted_index[left_idx:right_idx]}
                return [self.id_to_doc[doc_id] for doc_id in candidate_ids if doc_id in self.id_to_doc]

        # Text search example: "name" prefix
        if "name" in filter_query:
            name = filter_query.get("name", "")
            candidate_ids = self.trie.search_prefix(name.lower())
            return [self.id_to_doc[doc_id] for doc_id in candidate_ids if doc_id in self.id_to_doc]

        # No index usable, full scan fallback
        return list(self.id_to_doc.values())

    async def _write_collection(self, path: str, data: List[Dict[str, Any]]):
        await self.handler.write_async(path, data)

    async def _log_changes(self, updated_ids: List[Any], original_docs_backup: Dict[Any, Dict[str, Any]]):
        import aiofiles
        async with aiofiles.open(self.audit_log_path, "a", encoding="utf-8") as f:
            for doc_id in updated_ids:
                old_doc = original_docs_backup.get(doc_id, {})
                new_doc = self.id_to_doc.get(doc_id, {})
                diff = dict_diff(old_doc, new_doc)
                if diff:
                    timestamp = datetime.utcnow().isoformat()
                    log_entry = json.dumps({
                        "_id": doc_id,
                        "timestamp": timestamp,
                        "diff": diff
                    })
                    await f.write(log_entry + "\n")

    async def _notify_change(self, event: Dict[str, Any]):
        for queue in list(self._subscribers):
            try:
                await queue.put(event)
            except Exception:
                self._subscribers.discard(queue)

    async def subscribe_changes(self):
        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers.add(queue)
        try:
            while True:
                event = await queue.get()
                yield event
        finally:
            self._subscribers.discard(queue)

    def _index_doc(self, doc: Dict[str, Any]):
        # Add to indexes (used for upserts)
        doc_id = doc.get('_id')
        if doc_id:
            self.id_to_doc[doc_id] = doc
            age = doc.get("age")
            if isinstance(age, (int, float)):
                bisect.insort(self.sorted_index, (age, doc_id))
            name = doc.get("name", "")
            if isinstance(name, str) and name:
                self.bloom_filter.add(name.lower())
                for token in name.lower().split():
                    self.trie.insert(token, doc_id)

    def _project_for_upsert(self, filter_query: Dict[str, Any]) -> Dict[str, Any]:
        # Use equality parts of filter as base document fields for upsert
        return {k: v for k, v in filter_query.items() if not k.startswith("$") and not isinstance(v, dict)}

    def _project_fields(self, doc: Dict[str, Any], fields: Optional[List[str]]) -> Dict[str, Any]:
        if not fields:
            return dict(doc)
        return {k: doc.get(k) for k in fields}

    def _failure_result(self, op: str, message: str) -> DatabaseActionResult:
        return DatabaseActionResult(
            success=False,
            operation_type=op,
            error=ElectrusException(message),
            matched_count=0,
            modified_count=0,
        )
