from typing import Any, Dict, List, Optional, Union, Iterable, Iterator, Tuple
from collections import defaultdict
import random
import asyncio
import math
import numpy as np

from ..exception.base import ElectrusException
from ..partials import ElectrusLogicalOperators
from ..handler.filemanager import JsonFileHandler


class Aggregation:
    def __init__(self, collection_path: str, handler: JsonFileHandler):
        self.collection_path = collection_path
        self.handler = handler
        self.data: List[Dict[str, Any]] = []
        self.logical_ops = ElectrusLogicalOperators()
        # Cache for additional collections' data by name
        self.additional_collections_data: Dict[str, List[Dict[str, Any]]] = {}

    async def _load_data(self, use_cache: bool = True) -> None:
        if use_cache and self.data:
            return
        try:
            result = await self.handler.read_async(
                self.collection_path, verify_integrity=False
            )
            self.data = result.get('data', [])
        except Exception as e:
            raise ElectrusException(f"Error loading collection data: {e}")

    def _to_list(self, data: Union[List[Dict], Dict[Any, Any], Iterable]) -> List[Dict]:
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and all(isinstance(v, dict) for v in data.values()):
            return list(data.values())
        return list(data)

    def _evaluate_expression(self, item: Dict[str, Any], expr: Any) -> Any:
        if isinstance(expr, dict) and all(not k.startswith("$") for k in expr):
            return {k: self._evaluate_expression(item, v) for k, v in expr.items()}
    
        if isinstance(expr, dict):
            for op, val in expr.items():
                if op == '$cond':
                    cond = self._evaluate_expression(item, val.get('if'))
                    return (self._evaluate_expression(item, val.get('then'))
                            if cond else self._evaluate_expression(item, val.get('else')))

                if op == '$switch':
                    for branch in val.get('branches', []):
                        if self._evaluate_expression(item, branch['case']):
                            return self._evaluate_expression(item, branch['then'])
                    return self._evaluate_expression(item, val.get('default'))

                if op == '$filter':
                    arr = self._evaluate_expression(item, val['input'])
                    if not isinstance(arr, list):
                        return []
                    result = []
                    var_name = val.get('as', 'this')
                    for elem in arr:
                        context = {**item, var_name: elem}
                        if self._evaluate_expression(context, val['cond']):
                            result.append(elem)
                    return result

                if op == '$map':
                    arr = self._evaluate_expression(item, val['input'])
                    if not isinstance(arr, list):
                        return []
                    var_name = val.get('as', 'this')
                    return [self._evaluate_expression({**item, var_name: e}, val['in']) for e in arr]

                if op == '$reduce':
                    arr = self._evaluate_expression(item, val['input'])
                    if not isinstance(arr, list):
                        return val.get('initialValue')
                    acc = self._evaluate_expression(item, val.get('initialValue'))
                    in_expr = val['in']
                    for elem in arr:
                        context = {**item, 'value': elem, 'accumulator': acc}
                        acc = self._evaluate_expression(context, in_expr)
                    return acc

                if op in ('$setUnion', '$setIntersection', '$setDifference'):
                    arr1 = self._evaluate_expression(item, val[0]) or []
                    arr2 = self._evaluate_expression(item, val[1]) or []
                    set1, set2 = set(arr1), set(arr2)
                    if op == '$setUnion':
                        return list(set1 | set2)
                    elif op == '$setIntersection':
                        return list(set1 & set2)
                    else:  # $setDifference
                        return list(set1 - set2)

                # Delegate to logical_ops for other arithmetic or logical ops
                if hasattr(self.logical_ops, 'evaluate_op'):
                    return self.logical_ops.evaluate_op(op, val, item)

                # Unknown operator
                return None

        if isinstance(expr, str) and expr.startswith('$'):
            parts = expr[1:].split('.')
            val = item
            for p in parts:
                if not isinstance(val, dict):
                    return None
                val = val.get(p)
                if val is None:
                    return None
            return val

        return expr
    
    def _group(self, data: Iterable[Dict[str, Any]], key_expr: Any,
            accs: Optional[Dict[str, Dict]] = None) -> List[Dict[str, Any]]:
        groups: Dict[Any, List[Dict[str, Any]]] = {}
        raw_keys: Dict[Any, Any] = {}

        for doc in data:
            # Step 1: Evaluate the raw key (could be a dict or scalar)
            if isinstance(key_expr, dict):
                raw_key = self._evaluate_expression(doc, key_expr)
            else:
                raw_key = doc.get(key_expr) if isinstance(key_expr, str) else key_expr

            # Step 2: Convert raw_key to a hashable version
            if isinstance(raw_key, dict):
                hashable = tuple(sorted(raw_key.items()))
                raw_keys[hashable] = raw_key
            else:
                hashable = raw_key

            # Step 3: Add the document to its group
            groups.setdefault(hashable, []).append(doc)

        # Step 4: Handle no accumulator case
        if not accs:
            return [
                {'_id': (raw_keys[k] if k in raw_keys else k), 'items': v}
                for k, v in groups.items()
            ]

        result = []
        for key, items in groups.items():
            group_doc = {'_id': key}
            for field, expr in accs.items():
                if not isinstance(expr, dict):
                    group_doc[field] = expr
                    continue
                op, operand = next(iter(expr.items()))

                # Extract numeric vals for NumPy operations
                def get_numeric_values():
                    vals = [self._evaluate_expression(i, operand) for i in items]
                    return np.array([v for v in vals if isinstance(v, (int, float))])

                if op == '$sum':
                    vals = get_numeric_values()
                    group_doc[field] = np.sum(vals) if vals.size > 0 else 0

                elif op == '$avg':
                    vals = get_numeric_values()
                    group_doc[field] = np.mean(vals) if vals.size > 0 else None

                elif op == '$min':
                    vals = get_numeric_values()
                    group_doc[field] = np.min(vals) if vals.size > 0 else None

                elif op == '$max':
                    vals = get_numeric_values()
                    group_doc[field] = np.max(vals) if vals.size > 0 else None

                elif op == '$stdDevPop':
                    vals = get_numeric_values()
                    group_doc[field] = np.std(vals, ddof=0) if vals.size > 0 else None

                elif op == '$stdDevSamp':
                    vals = get_numeric_values()
                    group_doc[field] = np.std(vals, ddof=1) if vals.size > 1 else None

                elif op == '$median':
                    vals = get_numeric_values()
                    group_doc[field] = np.median(vals) if vals.size > 0 else None

                elif op == '$push':
                    group_doc[field] = [self._evaluate_expression(i, operand) for i in items]

                elif op == '$addToSet':
                    seen = set()
                    unique = []
                    for i in items:
                        val = self._evaluate_expression(i, operand)
                        try:
                            if val not in seen:
                                seen.add(val)
                                unique.append(val)
                        except TypeError:
                            if val not in unique:
                                unique.append(val)
                    group_doc[field] = unique

                elif op == '$first':
                    group_doc[field] = self._evaluate_expression(items[0], operand) if items else None

                elif op == '$last':
                    group_doc[field] = self._evaluate_expression(items[-1], operand) if items else None

                else:
                    raise ValueError(f"Unsupported accumulator operator: {op}")
            result.append(group_doc)
        return result

    def _count(self, data: Iterable[Any]) -> int:
        return sum(1 for _ in data)

    def _project(self, data: Iterable[Dict[str, Any]],
                 projection: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        for doc in data:
            projected = {}
            for k, v in projection.items():
                if v == 1:
                    val = self._evaluate_expression(doc, f"${k}")
                    if val is not None:
                        projected[k] = val
                elif isinstance(v, dict):
                    projected[k] = self._evaluate_expression(doc, v)
            yield projected

    def _match(self, data: Iterable[Dict[str, Any]],
               filter_query: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        for doc in data:
            if self.logical_ops.evaluate(doc, filter_query):
                yield doc

    def _sort(self, data: List[Dict[str, Any]], field: str, reverse: bool = False) -> List[Dict[str, Any]]:
        def sort_key(x):
            v = self._evaluate_expression(x, field)
            return (v is None, v)
        return sorted(data, key=sort_key, reverse=reverse)

    def _limit(self, data: Iterable[Any], limit_value: int) -> List[Any]:
        return list(_take(data, limit_value))

    def _skip(self, data: Iterable[Any], skip_value: int) -> Iterator[Any]:
        return _drop(data, skip_value)

    def _sample(self, data: List[Dict[str, Any]], size: int) -> List[Dict[str, Any]]:
        if size >= len(data):
            return data.copy()
        return random.sample(data, size)

    def _unwind(self, data: Iterable[Dict[str, Any]], field: str) -> Iterator[Dict[str, Any]]:
        for doc in data:
            arr = self._evaluate_expression(doc, f"${field}")
            if isinstance(arr, list):
                for elt in arr:
                    new_doc = dict(doc)
                    new_doc[field] = elt
                    yield new_doc
            else:
                yield doc

    def _lookup(self,
                data: List[Dict[str, Any]],
                from_collection: List[Dict[str, Any]],
                local_field: str,
                foreign_field: str,
                as_field: str) -> List[Dict[str, Any]]:
        foreign_index = defaultdict(list)
        for doc in from_collection:
            key = self._evaluate_expression(doc, f"${foreign_field}")
            foreign_index[key].append(doc)

        merged = []
        for doc in data:
            local_val = self._evaluate_expression(doc, f"${local_field}")
            matches = foreign_index.get(local_val, [])
            if matches:
                for match in matches:
                    new_doc = dict(doc)
                    new_doc[as_field] = match
                    merged.append(new_doc)
            else:
                new_doc = dict(doc)
                new_doc[as_field] = None
                merged.append(new_doc)
        return merged

    def _bucket(self, data: List[Dict[str, Any]], group_by: str,
                boundaries: List[float], default: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        boundaries = sorted(boundaries)
        buckets: Dict[str, List[Dict[str, Any]]] = {str(b): [] for b in boundaries}
        if default is not None:
            buckets[default] = []
        for doc in data:
            val = self._evaluate_expression(doc, f"${group_by}")
            if not isinstance(val, (int, float)):
                if default is not None:
                    buckets[default].append(doc)
                continue
            idx = _find_bucket(boundaries, val)
            if idx is not None:
                buckets[str(boundaries[idx])].append(doc)
            elif default is not None:
                buckets[default].append(doc)
        return buckets

    def _add_fields(self, data: List[Dict[str, Any]], fields: Dict[str, Any]) -> List[Dict[str, Any]]:
        new_data = []
        for doc in data:
            new_doc = dict(doc)
            for k, expr in fields.items():
                new_doc[k] = self._evaluate_expression(new_doc, expr)
            new_data.append(new_doc)
        return new_data

    async def _facet(
        self,
        data: List[Dict[str, Any]],
        facets: Dict[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        results: Dict[str, Any] = {}
        tasks: List[Tuple[str, asyncio.Task]] = []

        for name, pipeline in facets.items():
            sub_agg = Aggregation(self.collection_path, self.handler)
            sub_agg.data = data  # share same input data
            task = asyncio.create_task(sub_agg.execute(pipeline))
            tasks.append((name, task))

        for name, task in tasks:
            results[name] = await task

        return [results]  # $facet should return a list with a single dict


    async def execute(self, pipeline: List[Dict[str, Any]],
                      additional_collections: Optional[Dict[str, Any]] = None) -> Any:
        if not self.data:
            await self._load_data()

        # Load additional collections if any using collection_path via handler
        if additional_collections:
            for name, collection_obj in additional_collections.items():
                try:
                    result = await self.handler.read_async(
                        collection_obj.collection_path,
                        verify_integrity=False
                    )
                    self.additional_collections_data[name] = result.get('data', [])
                except Exception as e:
                    raise ElectrusException(f"Error loading additional collection '{name}': {e}")

        current_data: Union[List[Dict[str, Any]], int, float, None] = self.data

        try:
            for stage in pipeline:
                for operation, value in stage.items():
                    list_required = {
                        '$sort', '$limit', '$skip', '$sample', '$unwind',
                        '$lookup', '$bucket', '$addFields', '$facet',
                        '$project', '$group'
                    }
                    if operation in list_required and not isinstance(current_data, list):
                        current_data = self._to_list(current_data)

                    if operation == '$group':
                        if isinstance(value, dict):
                            key_expr = value.get('_id')
                            accumulators = {k: v for k, v in value.items() if k != '_id'}
                            current_data = self._group(current_data, key_expr, accumulators)
                        else:
                            current_data = self._group(current_data, value)

                    elif operation == '$count':
                        current_data = self._count(current_data)

                    elif operation == '$sum':
                        current_data = sum(self._evaluate_expression(item, value) for item in current_data)

                    elif operation == '$avg':
                        values = [self._evaluate_expression(item, value) for item in current_data
                                  if isinstance(self._evaluate_expression(item, value), (int, float))]
                        current_data = (sum(values) / len(values)) if values else None

                    elif operation == '$max':
                        values = [self._evaluate_expression(item, value) for item in current_data
                                  if self._evaluate_expression(item, value) is not None]
                        current_data = max(values) if values else None

                    elif operation == '$min':
                        values = [self._evaluate_expression(item, value) for item in current_data
                                  if self._evaluate_expression(item, value) is not None]
                        current_data = min(values) if values else None

                    elif operation == '$median':
                        # Implement $median similarly using numpy
                        vals = [self._evaluate_expression(item, value) for item in current_data
                                if isinstance(self._evaluate_expression(item, value), (int, float))]
                        arr = np.array(vals)
                        current_data = np.median(arr) if arr.size > 0 else None

                    elif operation == '$project':
                        current_data = list(self._project(current_data, value))

                    elif operation == '$match':
                        current_data = list(self._match(current_data, value))

                    elif operation == '$sort':
                        field = value.get('field') if isinstance(value, dict) else None
                        reverse = value.get('reverse', False) if isinstance(value, dict) else False
                        current_data = self._sort(current_data, field, reverse)

                    elif operation == '$limit':
                        current_data = self._limit(current_data, value)

                    elif operation == '$skip':
                        current_data = list(self._skip(current_data, value))

                    elif operation == '$sample':
                        current_data = self._sample(current_data, value)

                    elif operation == '$unwind':
                        current_data = list(self._unwind(current_data, value))

                    elif operation == '$lookup':
                        from_coll_spec = value.get('from')
                        local_field = value.get('localField')
                        foreign_field = value.get('foreignField')
                        as_field = value.get('as')
                        if None in (from_coll_spec, local_field, foreign_field, as_field):
                            raise ValueError("$lookup requires 'from', 'localField', 'foreignField', and 'as' fields")

                        if isinstance(from_coll_spec, list):
                            from_collection = from_coll_spec
                        elif isinstance(from_coll_spec, str):
                            from_collection = self.additional_collections_data.get(from_coll_spec)
                            if from_collection is None:
                                raise ValueError(f"Collection named '{from_coll_spec}' not found in additional_collections for $lookup")
                        else:
                            raise ValueError("'from' field for $lookup must be a list or a string referencing additional_collections")

                        current_data = self._lookup(current_data, from_collection, local_field, foreign_field, as_field)

                    elif operation == '$bucket':
                        current_data = self._bucket(
                            current_data,
                            value.get('groupBy'),
                            value.get('boundaries', []),
                            value.get('default')
                        )

                    elif operation == '$addFields':
                        if not isinstance(value, dict):
                            raise ValueError("$addFields value must be a dict")
                        current_data = self._add_fields(current_data, value)

                    elif operation == '$facet':
                        if not isinstance(value, dict):
                            raise ValueError("$facet value must be a dict of pipelines")
                        for key, pipeline in value.items():
                            if not isinstance(pipeline, list):
                                raise ValueError(f"$facet pipeline for '{key}' must be a list")
                        current_data = await self._facet(current_data, value)


                    else:
                        raise ValueError(f"Unsupported operation: {operation}")

            return current_data

        except Exception as e:
            raise ElectrusException(f"Error executing aggregation pipeline: {e}")


# --- Utility functions ---

def _take(iterable: Iterable, n: int) -> Iterator:
    it = iter(iterable)
    for _ in range(n):
        val = next(it, None)
        if val is None:
            break
        yield val


def _drop(iterable: Iterable, n: int) -> Iterator:
    it = iter(iterable)
    for _ in range(n):
        next(it, None)
    yield from it


def _find_bucket(boundaries: List[float], value: float) -> Optional[int]:
    low, high = 0, len(boundaries) - 1
    while low <= high:
        mid = (low + high) // 2
        upper_bound = boundaries[mid + 1] if mid + 1 < len(boundaries) else float('inf')
        if boundaries[mid] <= value < upper_bound:
            return mid
        elif value < boundaries[mid]:
            high = mid - 1
        else:
            low = mid + 1
    return None
