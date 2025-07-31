import os
import aiofiles
import json
import asyncio
import logging
import re
import ast
import heapq
import concurrent.futures

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from collections import defaultdict, namedtuple
from datetime import datetime, timedelta, timezone

import numpy as np  # For vectorized aggregations

from ..exception.base import ElectrusException
from .results import DatabaseActionResult
from ..handler.filemanager import JsonFileHandler
from .operators import ElectrusLogicalOperators

# Named tuple for window function range
WindowFrame = namedtuple('WindowFrame', ['start', 'end'])

# ---------------------------------------------------
# Trie for prefix searches (LIKE / startsWith)
# ---------------------------------------------------

class TrieNode:
    __slots__ = ['children', 'indexes', 'is_end']
    def __init__(self):
        self.children: Dict[str, TrieNode] = {}
        self.indexes: set = set()  # record positions matching prefix
        self.is_end: bool = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str, idx: int) -> None:
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
            node.indexes.add(idx)
        node.is_end = True

    def starts_with(self, prefix: str) -> set:
        node = self.root
        for char in prefix:
            if char not in node.children:
                return set()
            node = node.children[char]
        # Return the set of indexes for all strings starting with this prefix
        return node.indexes.copy()

# ---------------------------------------------------
# AST-based safe filter compiler for .where() expressions
# ---------------------------------------------------

class ASTFilterCompiler(ast.NodeTransformer):
    allowed_nodes = {
        ast.Expression, ast.BoolOp, ast.BinOp, ast.UnaryOp,
        ast.Compare, ast.Name, ast.Load, ast.Str, ast.Num,
        ast.Call, ast.Constant, ast.Attribute, ast.Subscript,
        ast.List, ast.Tuple,
        ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
        ast.In, ast.NotIn, ast.Is, ast.IsNot, ast.And, ast.Or, ast.Not,
        ast.IfExp,  
        ast.Dict, 
        ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow
    }

    allowed_funcs = {
        'UPPER', 'LOWER', 'SUBSTR', 'ROUND', 'FLOOR', 'CEIL', 'DATE_ADD', 'DATEDIFF', 'JSON_EXTRACT'
    }

    def __init__(self, record_var_name='r'):
        super().__init__()
        self.record_var_name = record_var_name

    def visit(self, node):
        if type(node) not in self.allowed_nodes:
            raise ElectrusException(f"Disallowed syntax in filter expression: {type(node).__name__}")
        return super().visit(node)

    def visit_Name(self, node):
        if node.id in self.allowed_funcs or node.id in ('True', 'False', 'None'):
            # allowed function or constant names remain as is
            return node
        # Transform variable names into calls to self._get(record, field_name)
        return ast.Call(
            func=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='_get', ctx=ast.Load()),
            args=[
                ast.Name(id=self.record_var_name, ctx=ast.Load()),
                ast.Constant(node.id)
            ],
            keywords=[]
        )

def compile_filter_expr(expr: str, params: Dict[str, Any], query_builder: 'QueryBuilder') -> Callable[[Dict], bool]:
    # Substitute param placeholders ($param) with literal values
    for k, v in params.items():
        expr = expr.replace(f"${k}", repr(v))

    # Normalize boolean operators, equality and special terms
    expr = expr.replace("AND", "and").replace("OR", "or").replace("NOT", "not")
    expr = expr.replace("=", "==").replace("IS NULL", "is None").replace("IS NOT NULL", "is not None")
    expr = re.sub(r"\bIN\s*\(", "in (", expr, flags=re.IGNORECASE)

    try:
        tree = ast.parse(expr, mode='eval')
        compiler = ASTFilterCompiler()
        tree = compiler.visit(tree)
        ast.fix_missing_locations(tree)
        code = compile(tree, '<filter>', 'eval')
    except Exception as e:
        raise ElectrusException(f"Expression parsing failed: {expr}, error: {e}")

    def fn(record):
        ctx = dict(query_builder._SAFE_FUNCS)
        ctx.update(params)
        ctx.update({'self': query_builder})
        try:
            return bool(eval(code, {"__builtins__": {}}, {'r': record, **ctx}))
        except Exception:
            return False

    return fn

# ---------------------------------------------------
# Mongo filter wrapper for compound Mongo-style filters
# ---------------------------------------------------

class _MongoFilterWrapper:
    def __init__(self, logic: ElectrusLogicalOperators, getter: Callable[[], Dict[str, Any]]):
        self._logic = logic
        self._getter = getter

    def __call__(self, record: Dict[str, Any]) -> bool:
        return self._logic.evaluate(record, self._getter())

# ---------------------------------------------------
# Main QueryBuilder class with all enhancements
# ---------------------------------------------------

class QueryBuilder:
    """
    Enhanced SQL-style query engine for JSON data, supporting:
    - Trie and set indexes for accelerated prefix and existence filters
    - Heap-based Top-K queries
    - Parallel filtering via ThreadPoolExecutor
    - Vectorized group aggregations via numpy
    - Mini-AST compiled filters for safe and performant expression evaluation
    """

    _SAFE_FUNCS = {
        'UPPER': lambda s: s.upper() if isinstance(s, str) else s,
        'LOWER': lambda s: s.lower() if isinstance(s, str) else s,
        'SUBSTR': lambda s, start, length=None: s[start:start+length] if (isinstance(s, str) and length is not None) else (s[start:] if isinstance(s, str) else s),
        'ROUND': round,
        'FLOOR': lambda x: int(x),
        'CEIL': lambda x: int(x) + (x % 1 > 0),
        'DATE_ADD': lambda d, days: d + timedelta(days=days),
        'DATEDIFF': lambda d1, d2: (d1 - d2).days,
        'JSON_EXTRACT': lambda doc, path: QueryBuilder._json_extract(doc, path),
    }

    _TYPE_CASTERS = {
        'int': int, 'float': float, 'str': str,
        'bool': lambda v: bool(int(v)) if isinstance(v, (str, int)) else bool(v),
        'json': json.loads,
        'date': lambda s: datetime.fromisoformat(s) if isinstance(s, str) else s,
    }

    def __init__(self, path: Path, handler: JsonFileHandler):
        self._path = path
        self._handler = handler
        self.reset()

    def reset(self) -> 'QueryBuilder':
        """Reset all clauses and internal state."""
        self._filters: List[Callable[[Dict], bool]] = []
        self._selects: List[Union[str, Dict[str, Callable]]] = []
        self._excludes: set = set()
        self._joins: List[Tuple['QueryBuilder', str, str, str]] = []
        self._group_keys: List[str] = []
        self._aggs: Dict[str, Tuple[str, Callable]] = {}
        self._havings: List[Callable[[Dict], bool]] = []
        self._windows: Dict[str, Tuple[List[str], List[Dict[str, Union[str, bool]]], WindowFrame, Callable]] = {}
        self._order: List[Dict[str, Union[str, bool]]] = []
        self._offset: int = 0
        self._limit: Optional[int] = None

        # Indexes
        self._indexes: Dict[str, Dict[Any, List[int]]] = {}         # field → dict for normal indexes
        self._trie_indexes: Dict[str, Trie] = {}                    # field → Trie index
        self._set_indexes: Dict[str, set] = {}                      # field → set for fast existence

        self._snapshot: Optional[List[Dict]] = None
        self._plan: List[str] = []
        self._analysis: Dict[str, float] = {}
        self._pre_hooks: List[Callable[['QueryBuilder'], None]] = []
        self._post_hooks: List[Callable[['QueryBuilder', DatabaseActionResult], None]] = []
        self._debug: bool = False
        self._retries: int = 1
        self._mongo_filter: Dict = {}

        self._top_k: Optional[Tuple[str, int, bool]] = None     # (field, k, descending)

        if hasattr(self, "_computed_fields"):
            self._computed_fields.clear()
        else:
            self._computed_fields = []

        return self

    # ----------------- Index Creation -----------------

    def create_index(self, field: str) -> 'QueryBuilder':
        """Create a standard hash index."""
        self._plan.append(f"CREATE_INDEX {field}")
        data = self._load_sync()
        idx = defaultdict(list)
        for i, r in enumerate(data):
            key = self._get(r, field)
            idx[key].append(i)
        self._indexes[field] = idx
        return self

    def create_trie_index(self, field: str) -> 'QueryBuilder':
        """Create a Trie index on a string field."""
        self._plan.append(f"CREATE_TRIE_INDEX {field}")
        data = self._load_sync()
        trie = Trie()
        for idx, record in enumerate(data):
            val = self._get(record, field)
            if isinstance(val, str):
                trie.insert(val, idx)
        self._trie_indexes[field] = trie
        return self

    def create_set_index(self, field: str) -> 'QueryBuilder':
        """Create a set index to speed up IN queries."""
        self._plan.append(f"CREATE_SET_INDEX {field}")
        data = self._load_sync()
        vals = set()
        for record in data:
            v = self._get(record, field)
            if v is not None:
                vals.add(v)
        self._set_indexes[field] = vals
        return self

    # ----------------- Query Clauses -----------------

    def select(self, *fields: Union[str, Dict[str, Callable]]) -> 'QueryBuilder':
        self._excludes = {f[1:] for f in fields if isinstance(f, str) and f.startswith("-")}
        self._plan.append(f"SELECT {fields}")
        if "*" in fields:
            self._selects = ['*']
            return self
        sel = []
        for f in fields:
            if isinstance(f, dict):
                sel.append(f)
            elif isinstance(f, str) and f.startswith("-"):
                continue
            else:
                fld, alias, cast, defv = self._parse_select(f)
                if cast or defv is not None:
                    def mk(fld, cast, defv):
                        return lambda r: cast(r.get(fld)) if r.get(fld) is not None else defv
                    sel.append({alias: mk(fld, cast, defv)})
                else:
                    sel.append(f"{fld} AS {alias}" if alias != fld else fld)
        self._selects = sel
        return self

    def with_field(self, alias: str, fn: Callable[[Dict], Any]) -> 'QueryBuilder':
        self._plan.append(f"WITH_FIELD {alias}")
        self._computed_fields.append((alias, fn))
        return self

    def where(self, *args, **kwargs) -> 'QueryBuilder':
        """
        Add filter using Mongo-style dict or kwargs with __op syntax.
        Multiple calls combine using $and.
        """
        self._plan.append(f"WHERE args={args}, kwargs={kwargs}")
        if args:
            if len(args) != 1 or not isinstance(args[0], dict):
                raise ElectrusException("`.where()` takes either a single dict or only kwargs.")
            new_filter = args[0]
        else:
            new_filter = {}
            for key, val in kwargs.items():
                if "__" in key:
                    field, op = key.split("__", 1)
                    new_filter.setdefault(field, {})[f"${op}"] = val
                else:
                    new_filter[key] = val

        if not self._mongo_filter:
            self._mongo_filter = new_filter
        else:
            self._mongo_filter = {"$and": [self._mongo_filter, new_filter]}

        if not any(isinstance(f, _MongoFilterWrapper) for f in self._filters):
            logic = ElectrusLogicalOperators()
            self._filters.append(_MongoFilterWrapper(logic, lambda: self._mongo_filter))

        return self

    def or_where(self, *args, **kwargs) -> 'QueryBuilder':
        """
        Add filter with $or logic combining clauses.
        """
        if args:
            if len(args) != 1 or not isinstance(args[0], dict):
                raise ElectrusException("`.or_where()` takes either a single dict or only kwargs.")
            clause = args[0]
        else:
            clause = {}
            for key, val in kwargs.items():
                if "__" in key:
                    field, op = key.split("__", 1)
                    clause.setdefault(field, {})[f"${op}"] = val
                else:
                    clause[key] = val

        if not self._mongo_filter:
            self._mongo_filter = {"$or": [clause]}
        else:
            if "$or" in self._mongo_filter:
                self._mongo_filter["$or"].append(clause)
            else:
                self._mongo_filter = {"$or": [self._mongo_filter, clause]}

        if not any(isinstance(f, _MongoFilterWrapper) for f in self._filters):
            logic = ElectrusLogicalOperators()
            self._filters.append(_MongoFilterWrapper(logic, lambda: self._mongo_filter))

        return self

    def filter_expr(self, expr: str, **params) -> 'QueryBuilder':
        """Add filter from a raw expression string with parameters."""
        self._filters.append(self._compile_expr(expr, params))
        return self

    def order_by(self, field: str, descending: bool=False) -> 'QueryBuilder':
        self._plan.append(f"ORDER_BY {field} {'DESC' if descending else 'ASC'}")
        self._order.append({'field': field, 'desc': descending})
        return self

    def offset(self, n: int) -> 'QueryBuilder':
        self._plan.append(f"OFFSET {n}")
        self._offset = max(0, n)
        return self

    def limit(self, n: int) -> 'QueryBuilder':
        self._plan.append(f"LIMIT {n}")
        self._limit = max(1, n)
        return self

    def join(self, other: 'QueryBuilder', on: Tuple[str, str], how: str='inner') -> 'QueryBuilder':
        self._plan.append(f"{how.upper()} JOIN ON {on}")
        self._joins.append((other, on[0], on[1], how))
        return self

    def group_by(self, *keys: str) -> 'QueryBuilder':
        self._plan.append(f"GROUP_BY {keys}")
        self._group_keys = list(keys)
        return self

    def aggregate(self, alias: str, fn: Callable[[List[Any]], Any], field: str) -> 'QueryBuilder':
        self._plan.append(f"AGGREGATE {alias}({field})")
        self._aggs[alias] = (field, fn)
        return self

    def having(self, expr: str, **params) -> 'QueryBuilder':
        self._plan.append(f"HAVING {expr}")
        self._havings.append(self._compile_expr(expr, params))
        return self

    def window(self, alias: str, fn: Callable[[List[Any]], List[Any]], partition_by: List[str], order_by: List[Tuple[str, bool]],
               frame: WindowFrame = WindowFrame('UNBOUNDED PRECEDING','CURRENT ROW')) -> 'QueryBuilder':
        self._plan.append(f"WINDOW {alias}")
        specs = [{'field': f, 'desc': d} for f, d in order_by]
        self._windows[alias] = (partition_by, specs, frame, fn)
        return self

    # ----------------- Top-K query feature -----------------

    def top_k(self, field: str, k: int, descending: bool=True) -> 'QueryBuilder':
        """Retrieve top-k rows by a field using heapq."""
        self._plan.append(f"TOP_K {k} BY {field} DESC={descending}")
        self._top_k = (field, k, descending)
        return self

    # ----------------- Internal helpers -----------------

    def _parse_select(self, expr: str) -> Tuple[str, str, Optional[Callable], Any]:
        m = re.match(r'^(?P<f>[^\s:=>]+)(:(?P<c>\w+))?(=(?P<d>[^ ]+))?(?:\s+[Aa][Ss]\s+(?P<a>\w+))?$', expr.strip())
        if not m: return expr, expr, None, None
        gd = m.groupdict()
        f = gd['f']
        cast = self._TYPE_CASTERS.get(gd['c'])
        default = gd['d']
        if default and cast:
            default = cast(default)
        alias = gd.get('a') or f
        return f, alias, cast, default

    def _compile_expr(self, expr: str, params: Dict[str, Any]) -> Callable[[Dict], bool]:
        fn = compile_filter_expr(expr, params, self)
        self._plan.append(f"FILTER_EXPR {expr}")
        return fn

    def _load_sync(self) -> List[Dict]:
        info = self._handler.read(self._path, verify_integrity=False)
        data = info.get('data', [])
        if not isinstance(data, list):
            raise ElectrusException("Collection must be a list")
        return data

    def _get(self, doc: Dict, field: str) -> Any:
        """Get nested field."""
        return QueryBuilder._json_extract(doc, '$.' + field) if '.' in field or '[' in field else doc.get(field)

    @staticmethod
    def _json_extract(doc: Any, path: str) -> Any:
        parts = re.findall(r"\$|\.?([^.\[\]]+)|\[(\d+)\]", path)
        curr = doc
        for key, idx in parts:
            if key and isinstance(curr, dict):
                curr = curr.get(key)
            elif idx and isinstance(curr, list):
                curr = curr[int(idx)]
            else:
                return None
        return curr

    def _flatten(self, doc: Dict, parent: str = '', sep: str = '.') -> Dict[str, Any]:
        out = {}
        for k, v in doc.items():
            nk = f"{parent}{sep}{k}" if parent else k
            if isinstance(v, dict):
                out.update(self._flatten(v, nk, sep))
            else:
                out[nk] = v
        return out

    # ----------------- Parallel Filtering -----------------

    def _filter_worker(self, chunk: List[Dict], filters: List[Callable]) -> List[Dict]:
        return [rec for rec in chunk if all(f(rec) for f in filters)]

    async def _parallel_filter(self, rows: List[Dict]) -> List[Dict]:
        if not self._filters:
            return rows
        loop = asyncio.get_running_loop()
        workers = min(8, os.cpu_count() or 1)
        chunk_size = max(1, len(rows) // workers)
        chunks = [rows[i:i + chunk_size] for i in range(0, len(rows), chunk_size)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            filtered_chunks = await asyncio.gather(
                *[loop.run_in_executor(executor, self._filter_worker, chunk, self._filters) for chunk in chunks]
            )
        # Flatten filtered results
        filtered_rows = [r for chunk in filtered_chunks for r in chunk]
        return filtered_rows

    # ----------------- Execute method -----------------

    async def execute(self) -> DatabaseActionResult:
        # Pre hooks
        for h in self._pre_hooks:
            h(self)

        t0 = datetime.now(timezone.utc)

        # Load data snapshot with retries
        if self._snapshot is None:
            for i in range(self._retries):
                try:
                    info = await self._handler.read_async(self._path, verify_integrity=False)
                    self._snapshot = info.get('data', [])
                    if not isinstance(self._snapshot, list):
                        raise ElectrusException("Collection must be a list")
                    break
                except Exception as e:
                    if i + 1 == self._retries:
                        raise ElectrusException(f"Read failed: {e}")
                    await asyncio.sleep(0.1 * 2 ** i)

        rows = list(self._snapshot)
        self._analysis['load_ms'] = (datetime.now(timezone.utc) - t0).total_seconds() * 1e3

        # Use Trie indexes for prefix (LIKE) filters if detected in _mongo_filter
        prefix_filtered_ids = None
        if self._mongo_filter and isinstance(self._mongo_filter, dict):
            for key, val in self._mongo_filter.items():
                if isinstance(val, dict):
                    for op, v in val.items():
                        if op in ('$like', '$startswith'):
                            trie = self._trie_indexes.get(key)
                            if trie and isinstance(v, str):
                                matched_ids = trie.starts_with(v)
                                if prefix_filtered_ids is None:
                                    prefix_filtered_ids = matched_ids
                                else:
                                    prefix_filtered_ids &= matched_ids
        if prefix_filtered_ids is not None:
            rows = [rows[i] for i in prefix_filtered_ids]

        # Parallel filtering with filters list
        t_filter_start = datetime.now(timezone.utc)
        if self._filters:
            rows = await self._parallel_filter(rows)
        self._analysis['filter_ms'] = (datetime.now(timezone.utc) - t_filter_start).total_seconds() * 1e3

        # Joins
        t_join_start = datetime.now(timezone.utc)
        for o, lf, rf, how in self._joins:
            res = await o.find().execute()
            orows = res.raw_result
            idx = defaultdict(list)
            for r in orows:
                idx[self._get(r, rf)].append(r)

            out = []
            for r in rows:
                key = self._get(r, lf)
                matches = idx.get(key, [])
                if matches:
                    for mr in matches:
                        combined = {**r, **mr}
                        out.append(combined)
                elif how in ('left', 'outer'):
                    out.append(r)
            rows = out
        self._analysis['join_ms'] = (datetime.now(timezone.utc) - t_join_start).total_seconds() * 1e3

        # Group and Aggregations with numpy optimization
        t_agg_start = datetime.now(timezone.utc)
        if self._group_keys:
            group_map = defaultdict(list)
            for r in rows:
                key = tuple(self._get(r, k) for k in self._group_keys)
                group_map[key].append(r)

            out = []
            for kvals, group_docs in group_map.items():
                b = dict(zip(self._group_keys, kvals))
                for alias, (fld, fn) in self._aggs.items():
                    vals = [self._get(r, fld) for r in group_docs if self._get(r, fld) is not None]
                    if not vals:
                        agg_val = None
                    else:
                        try:
                            arr = np.array(vals)
                            # Common aggregations optimized with numpy
                            if fn == sum:
                                agg_val = np.sum(arr).item()
                            elif fn == min:
                                agg_val = np.min(arr).item()
                            elif fn == max:
                                agg_val = np.max(arr).item()
                            elif fn in (np.mean, 'mean', 'avg', (lambda x: sum(x)/len(x))):
                                agg_val = np.mean(arr).item()
                            else:
                                agg_val = fn(vals)
                        except Exception:
                            agg_val = fn(vals)
                    b[alias] = agg_val

                if all(h(b) for h in self._havings):
                    out.append(b)
            rows = out
        self._analysis['agg_ms'] = (datetime.now(timezone.utc) - t_agg_start).total_seconds() * 1e3

        # Window functions
        t_win_start = datetime.now(timezone.utc)
        for al, (parts, specs, frame, fn) in self._windows.items():
            grp = defaultdict(list)
            for r in rows:
                grp[tuple(self._get(r, p) for p in parts)].append(r)
            allout = []
            for bucket in grp.values():
                for s in reversed(specs):
                    bucket.sort(key=lambda x: self._get(x, s['field']), reverse=s['desc'])
                vals = fn(bucket)
                if not isinstance(vals, list):
                    vals = [vals] * len(bucket)
                for rec, val in zip(bucket, vals):
                    rec[al] = val
                allout.extend(bucket)
            rows = allout
        self._analysis['window_ms'] = (datetime.now(timezone.utc) - t_win_start).total_seconds() * 1e3

        # Computed fields
        if self._computed_fields:
            for r in rows:
                for alias, fn in self._computed_fields:
                    try:
                        r[alias] = fn(r)
                    except Exception:
                        r[alias] = None

        # Sorting by order clauses
        for o in reversed(self._order):
            rows.sort(key=lambda r: self._get(r, o['field']), reverse=o['desc'])

        # Apply Top-K selection if requested
        if self._top_k:
            field, k, descending = self._top_k
            if rows:
                rows = heapq.nlargest(k, rows, key=lambda r: self._get(r, field)) if descending else heapq.nsmallest(k, rows, key=lambda r: self._get(r, field))

        # Offset and limit slicing
        start = self._offset
        end = None if self._limit is None else self._offset + self._limit
        rows = rows[start:end]

        # Projection / selection
        t_proj_start = datetime.now(timezone.utc)
        results = []
        for r in rows:
            if self._selects == ['*'] or not self._selects:
                rec = self._flatten(r)
                for ex in self._excludes:
                    rec.pop(ex, None)
                results.append(rec)
            else:
                rec = {}
                for f in self._selects:
                    if isinstance(f, dict):
                        k, fn = next(iter(f.items()))
                        try:
                            rec[k] = fn(r)
                        except Exception:
                            rec[k] = None
                    else:
                        src, alias = (f.split(" AS ") if " AS " in f else (f, f))
                        rec[alias] = self._get(r, src)
                results.append(rec)
        self._analysis['project_ms'] = (datetime.now(timezone.utc) - t_proj_start).total_seconds() * 1e3

        # Prepare the result object
        res = DatabaseActionResult(
            success=True,
            matched_count=len(results),
            raw_result=results,
            inserted_ids=[r.get('_id') for r in results if isinstance(r, dict) and '_id' in r]
        )

        # Post hooks
        for h in self._post_hooks:
            h(self, res)

        return res

    # ----------------- Aggregation shortcuts -----------------

    async def count(self) -> int:
        return (await self.execute()).matched_count

    async def sum(self, field: str) -> float:
        return sum(self._get(r, field) or 0 for r in (await self.execute()).raw_result)

    async def avg(self, field: str) -> Optional[float]:
        c = await self.count()
        return (await self.sum(field)) / c if c else None

    async def min(self, field: str) -> Any:
        return min((self._get(r, field) for r in (await self.execute()).raw_result), default=None)

    async def max(self, field: str) -> Any:
        return max((self._get(r, field) for r in (await self.execute()).raw_result), default=None)

    # ----------------- Pre/post hooks -----------------

    def pre_hook(self, fn: Callable[['QueryBuilder'], None]) -> 'QueryBuilder':
        self._pre_hooks.append(fn)
        return self

    def post_hook(self, fn: Callable[['QueryBuilder', DatabaseActionResult], None]) -> 'QueryBuilder':
        self._post_hooks.append(fn)
        return self

    def enable_debug(self) -> 'QueryBuilder':
        logging.basicConfig(level=logging.DEBUG)
        self._debug = True
        self._plan.append("ENABLE_DEBUG")
        return self

    def retries(self, n: int) -> 'QueryBuilder':
        self._plan.append(f"RETRIES {n}")
        self._retries = max(1, n)
        return self

    def explain(self) -> List[str]:
        """Return the planned steps for explainability."""
        return list(self._plan)

    def analyze(self) -> Dict[str, float]:
        """Return timing analysis for phases in milliseconds."""
        return dict(self._analysis)

    def describe(self) -> Dict[str, Any]:
        """Return basic metadata about the dataset: path, schema, indexes, row count."""
        docs = self._load_sync()
        keys = {key for doc in docs for key in self._flatten(doc)}
        return {'path': str(self._path), 'schema': sorted(keys), 'indexes': list(self._indexes.keys()), 'rows': len(docs)}

# ---------------------------------------------------
# End of QueryBuilder implementation
# ---------------------------------------------------
