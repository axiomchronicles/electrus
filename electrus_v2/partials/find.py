import os
import aiofiles
import json
import asyncio
import logging
import ast
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union, AsyncIterator, Tuple

from ..exception.base import ElectrusException
from .operators import ElectrusLogicalOperators
from .results import DatabaseActionResult
from ..handler.filemanager import JsonFileHandler


class QueryBuilder:
    """Advanced SQL-style JSON query engine with full SQL SELECT support."""

    # Supported type casters for select field casting
    _TYPE_CASTERS = {
        'int': int,
        'float': float,
        'str': str,
        'bool': lambda v: bool(int(v)) if isinstance(v, (str, int)) else bool(v),
    }

    def __init__(self, collection: Path, handler: JsonFileHandler):
        self._file: Path = collection
        self._handler: JsonFileHandler = handler

        # Query state
        self._filters: List[Callable[[Dict[str, Any]], bool]] = []
        self._projection: List[Union[str, Dict[str, Callable]]] = []
        self._exclude: set = set()
        self._sort_specs: List[Dict[str, Any]] = []
        self._offset: int = 0
        self._limit: int = 10

        # Index support
        self._indexes: Dict[str, Dict[Any, List[int]]] = {}

        # Hooks & logging
        self._pre_hooks: List[Callable[['QueryBuilder'], None]] = []
        self._post_hooks: List[Callable[['QueryBuilder', DatabaseActionResult], None]] = []
        self._debug: bool = False

        # Retry & caching
        self._retries: int = 1
        self._cached_docs: Optional[List[Dict[str, Any]]] = None


    # ------------------- Utility: Parse and Build Select Field -------------------

    def _parse_select_field(self, field_str: str) -> Tuple[str, str, Optional[str], Optional[Any]]:
        """
        Parses one select field expression into components (field, alias, cast, default).

        Supports patterns like:
            - age
            - age AS user_age
            - age:int
            - score:float=0.0
            - price:int=10 AS item_price
        """
        field_str = field_str.strip()
        regex = re.compile(
            r'^(?P<field>[^\s:=>]+)'          # field name until space or :,=,>
            r'(:(?P<cast>\w+))?'              # optional :cast
            r'(=(?P<default>[^ ]+))?'         # optional =default (non-space)
            r'(?:\s+[Aa][Ss]\s+(?P<alias>[^\s]+))?$'  # optional AS alias
        )
        match = regex.match(field_str)
        if not match:
            # Fallback: treat whole string as field name and alias
            return field_str, field_str, None, None

        gd = match.groupdict()
        raw_field = gd['field']
        cast = gd.get('cast')
        default_raw = gd.get('default')
        alias = gd.get('alias') or raw_field

        default_val = None
        if default_raw is not None:
            # Try parsing default value based on cast
            caster = self._TYPE_CASTERS.get(cast)
            if caster:
                try:
                    default_val = caster(default_raw)
                except Exception:
                    default_val = default_raw
            else:
                # Attempt bool parsing for bool cast or raw string for others
                if cast == 'bool':
                    if default_raw.lower() in ('true', '1'):
                        default_val = True
                    elif default_raw.lower() in ('false', '0'):
                        default_val = False
                    else:
                        default_val = bool(default_raw)
                else:
                    default_val = default_raw

        return raw_field, alias, cast, default_val


    # ------------------------ Fluent API Methods ------------------------

    def select(self, *fields: Union[str, Dict[str, Callable]]) -> 'QueryBuilder':
        """
        Supports:
         - '*' or no args: all fields
         - '-field': exclude field
         - 'column', 'column AS alias'
         - type casting with optional defaults e.g. 'age:int=0 AS user_age'
         - computed columns: dicts like {'full_name': lambda doc: ...}
        """
        if not fields or '*' in fields or all(isinstance(f, str) and f.startswith("-") for f in fields):
            self._projection = ['*']

        proj: List[Union[str, Dict[str, Callable]]] = []
        exclude: set = set()

        for f in fields:
            if isinstance(f, dict):
                proj.append(f)
            else:
                f = f.strip()
                if f.startswith('-'):
                    exclude.add(f[1:])
                else:
                    proj.append(f)

        parsed_proj = []
        for item in proj:
            if isinstance(item, dict):
                parsed_proj.append(item)
            else:
                raw_field, alias, cast, default_val = self._parse_select_field(item)
                if cast or default_val is not None:
                    caster = self._TYPE_CASTERS.get(cast, lambda x: x)

                    def build_fn(rf=raw_field, c=caster, d=default_val):
                        def fn(doc):
                            val = doc.get(rf, d)
                            if val is None:
                                val = d
                            try:
                                return c(val)
                            except Exception:
                                return d if d is not None else val
                        return fn

                    parsed_proj.append({alias: build_fn()})
                elif alias != raw_field:
                    parsed_proj.append(f"{raw_field} AS {alias}")
                else:
                    parsed_proj.append(raw_field)

        self._projection = parsed_proj
        self._exclude = exclude
        return self

    def where(self, **conds: Any) -> 'QueryBuilder':
        for key, val in conds.items():
            if callable(val):
                self._filters.append(lambda d, k=key, fn=val: fn(d.get(k)))
            else:
                self._filters.append(lambda d, k=key, v=val: d.get(k) == v)
        return self

    def where_expr(self, expr: str) -> 'QueryBuilder':
        code = compile(ast.parse(expr, mode='eval'), '<expr>', 'eval')
        def filter_fn(d: Dict[str, Any]) -> bool:
            try:
                return bool(eval(code, {"__builtins__": {}}, d))
            except Exception:
                return False
        self._filters.append(filter_fn)
        return self

    def order_by(
        self,
        field: str,
        descending: bool = False,
        collation: Optional[Callable[[Any], Any]] = None
    ) -> 'QueryBuilder':
        self._sort_specs.append({
            'field': field,
            'descending': descending,
            'collation': collation
        })
        return self

    def offset(self, n: int) -> 'QueryBuilder':
        self._offset = max(0, n)
        return self

    def limit(self, n: int) -> 'QueryBuilder':
        self._limit = max(1, n)
        return self

    def pre_hook(self, fn: Callable[['QueryBuilder'], None]) -> 'QueryBuilder':
        self._pre_hooks.append(fn)
        return self

    def post_hook(self, fn: Callable[['QueryBuilder', DatabaseActionResult], None]) -> 'QueryBuilder':
        self._post_hooks.append(fn)
        return self

    def create_index(self, field: str) -> 'QueryBuilder':
        sync_data = self._load_sync()
        idx: Dict[Any, List[int]] = {}
        for i, rec in enumerate(sync_data):
            idx.setdefault(rec.get(field), []).append(i)
        self._indexes[field] = idx
        if self._debug:
            logging.debug(f"Index created on '{field}', keys: {len(idx)}")
        return self

    def enable_debug(self) -> 'QueryBuilder':
        self._debug = True
        logging.basicConfig(level=logging.DEBUG)
        return self

    def retries(self, n: int) -> 'QueryBuilder':
        self._retries = max(1, n)
        return self

    def reset_cache(self) -> 'QueryBuilder':
        self._cached_docs = None
        return self


    # ------------------- Core Query Execution -------------------

    async def execute(self) -> DatabaseActionResult:
        for hook in self._pre_hooks:
            hook(self)

        # Load with retries, caching results per instance
        if self._cached_docs is not None:
            docs = self._cached_docs
            if self._debug:
                logging.debug("Using cached documents")
        else:
            for attempt in range(self._retries):
                try:
                    data = await self._handler.read_async(self._file, verify_integrity=False)
                    docs = data.get('data', [])
                    self._cached_docs = docs
                    if self._debug:
                        logging.debug(f"Loaded {len(docs)} documents from {self._file}")
                    break
                except Exception as e:
                    if attempt + 1 == self._retries:
                        raise ElectrusException(f"Failed to read after {self._retries} attempts: {e}")
                    await asyncio.sleep(0.1 * 2 ** attempt)

        # 1) Apply filters with short-circuit
        if self._filters:
            filtered = []
            for d in docs:
                if all(fn(d) for fn in self._filters):
                    filtered.append(d)
            docs = filtered

        # 2) Apply sorting, stable, handle missing keys, safe collation
        for spec in reversed(self._sort_specs):
            field = spec['field']
            desc = spec['descending']
            collate = spec['collation']

            def key_fn(x):
                val = x.get(field, None)
                try:
                    return collate(val) if collate else val
                except Exception:
                    return val

            docs.sort(key=key_fn, reverse=desc)

        # Apply projection with aliasing, casting, excludes, and computed columns
        results = []
        for rec in docs[self._offset:self._offset + self._limit]:
            if self._projection == ['*']:
                out = {k: v for k, v in rec.items() if k not in self._exclude}
            else:
                out = {}
                for fld in self._projection:
                    if isinstance(fld, dict):
                        key, fn = next(iter(fld.items()))
                        if key in self._exclude:
                            continue
                        try:
                            out[key] = fn(rec)
                        except Exception as ex:
                            if self._debug:
                                logging.warning(f"Computed '{key}' computation failed: {ex}")
                            out[key] = None
                    else:
                        if isinstance(fld, str) and ' AS ' in fld.upper():
                            parts = re.split(r'\s+AS\s+', fld, flags=re.IGNORECASE)
                            field_name, alias = parts[0], parts[1]
                        else:
                            field_name = alias = fld

                        if alias in self._exclude:
                            continue
                        if field_name in rec:
                            out[alias] = rec[field_name]

                if '_id' in rec and '_id' not in out and '_id' not in self._exclude:
                    out['_id'] = rec['_id']

            results.append(out)


        result = DatabaseActionResult(
            success=True,
            matched_count=len(results),
            raw_result=results,
            inserted_ids=[r.get('_id') for r in results]
        )

        for hook in self._post_hooks:
            hook(self, result)

        return result


    # ------------------- Streaming -------------------

    async def stream(self) -> AsyncIterator[Dict[str, Any]]:
        if self._cached_docs is not None:
            docs = self._cached_docs
        else:
            data = await self._handler.read_async(self._file, verify_integrity=False)
            docs = data.get('data', [])
            self._cached_docs = docs

        count = 0
        for rec in docs:
            if self._filters and not all(fn(rec) for fn in self._filters):
                continue
            if count < self._offset:
                count += 1
                continue
            if count >= self._offset + self._limit:
                break
            yield rec
            count += 1


    # ------------------- Sync Loader -------------------

    def _load_sync(self) -> List[Dict[str, Any]]:
        info = self._handler.read(self._file, verify_integrity=True)
        data = info.get('data', [])
        if not isinstance(data, list):
            raise ElectrusException("Collection must be a list")
        return data


    # ------------------- Aggregates -------------------

    async def count(self) -> int:
        result = await self.execute()
        return result.matched_count

    async def sum(self, field: str) -> float:
        result = await self.execute()
        total = 0.0
        for d in result.raw_result:
            try:
                val = d.get(field)
                if val is not None:
                    total += float(val)
            except (TypeError, ValueError):
                continue
        return total

    async def avg(self, field: str) -> Optional[float]:
        total = await self.sum(field)
        count = await self.count()
        return (total / count) if count else None
