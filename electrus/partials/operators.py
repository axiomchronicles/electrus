import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Union, Callable

from ..exception.base import ElectrusException

def _get_value(item: Dict[str, Any], path: str) -> Any:
    parts = path.split('.')
    val = item
    for p in parts:
        if not isinstance(val, dict) or p not in val:
            return None
        val = val[p]
    return val

class ElectrusLogicalOperators:
    """Evaluate MongoDB-like query operators against an item (dict)."""

    VALID_OPERATORS = {
        "$eq", "$ne", "$lt", "$lte", "$gt", "$gte",
        "$in", "$nin", "$exists", "$regex",
        "$and", "$or", "$nor", "$not",
        "$type", "$mod", "$expr",
        "$size", "$all", "$elemMatch",
        "$bitsAllSet", "$bitsAllClear", "$bitsAnySet", "$bitsAnyClear"
    }

    def __init__(self):
        self._handlers: Dict[str, Callable[[Dict, str, Any], bool]] = {
            "$eq": self._op_eq,
            "$ne": self._op_ne,
            "$lt": self._op_lt,
            "$lte": self._op_lte,
            "$gt": self._op_gt,
            "$gte": self._op_gte,
            "$in": self._op_in,
            "$nin": self._op_nin,
            "$exists": self._op_exists,
            "$regex": self._op_regex,
            "$and": self._op_and,
            "$or": self._op_or,
            "$nor": self._op_nor,
            "$not": self._op_not,
            "$type": self._op_type,
            "$mod": self._op_mod,
            "$expr": self._op_expr,
            "$size": self._op_size,
            "$all": self._op_all,
            "$elemMatch": self._op_elem_match,
            "$bitsAllSet": self._op_bits_all_set,
            "$bitsAllClear": self._op_bits_all_clear,
            "$bitsAnySet": self._op_bits_any_set,
            "$bitsAnyClear": self._op_bits_any_clear,
        }

    def evaluate(self, item: Dict[str, Any], query: Dict[str, Any]) -> bool:
        if len(query) == 1:
            top_key, top_value = next(iter(query.items()))
            if top_key in {"$or", "$and", "$nor"}:
                return self._handlers[top_key](item, top_key, top_value)
            if top_key == "$not":
                return self._handlers["$not"](item, None, top_value)
    
        for field, crit in query.items():
            if isinstance(crit, dict):
                for op, val in crit.items():
                    if op not in self.VALID_OPERATORS:
                        raise ElectrusException(f"Invalid operator: {op}")
                    if not self._handlers[op](item, field, val):
                        return False
            else:
                if _get_value(item, field) != crit:
                    return False
        return True

    def _op_eq(self, item, field, v):       return _get_value(item, field) == v
    def _op_ne(self, item, field, v):       return _get_value(item, field) != v
    def _op_lt(self, item, field, v):       return (_get_value(item, field) or float("inf")) < v
    def _op_lte(self, item, field, v):      return (_get_value(item, field) or float("inf")) <= v
    def _op_gt(self, item, field, v):       return (_get_value(item, field) or float("-inf")) > v
    def _op_gte(self, item, field, v):      return (_get_value(item, field) or float("-inf")) >= v
    def _op_in(self, item, field, v):
        val = _get_value(item, field)
        if isinstance(val, list):
            return any(elem in v for elem in val)
        return val in v

    def _op_nin(self, item, field, v):
        val = _get_value(item, field)
        if isinstance(val, list):
            return all(elem not in v for elem in val)
        return val not in v

    # Fix _op_exists to support nested fields
    def _op_exists(self, item, field, v):
        val = _get_value(item, field)
        return (val is not None) == v
    def _op_regex(self, item, field, pat):
        txt = _get_value(item, field)
        return isinstance(txt, str) and bool(re.search(pat, txt))
    def _op_and(self, item, _, subs: List[Dict]):
        return all(self.evaluate(item, sub) for sub in subs)
    def _op_or(self, item, _, subs: List[Dict]):
        return any(self.evaluate(item, sub) for sub in subs)
    def _op_nor(self, item, _, subs: List[Dict]):
        return not any(self.evaluate(item, sub) for sub in subs)
    def _op_not(self, item, field, sub: Dict):
        return not self.evaluate(item, {field: sub})
    def _op_type(self, item, field, t: str):
        val = _get_value(item, field)
        mapping = {
            "string": str, "number": (int, float),
            "array": list, "object": dict,
            "bool": bool, "null": type(None)
        }
        return isinstance(val, mapping.get(t, object))
    def _op_mod(self, item, field, args: List[int]):
        val = _get_value(item, field)
        return isinstance(val, (int, float)) and val % args[0] == args[1]
    def _op_expr(self, item, _, expr: Dict):
        # only simple binary ops supported here
        op, (left, right) = next(iter(expr.items()))
        lval = _get_value(item, left[1:]) if isinstance(left, str) and left.startswith("$") else left
        rval = _get_value(item, right[1:]) if isinstance(right, str) and right.startswith("$") else right
        ops = {
            "$gt": lambda a,b: a > b, "$lt": lambda a,b: a < b,
            "$eq": lambda a,b: a == b, "$ne": lambda a,b: a != b,
            "$gte": lambda a,b: a >= b, "$lte": lambda a,b: a <= b,
        }
        return ops[op](lval, rval)
    def _op_size(self, item, field, size: int):
        lst = _get_value(item, field)
        return isinstance(lst, list) and len(lst) == size
    def _op_all(self, item, field, vals: List[Any]):
        lst = _get_value(item, field)
        return isinstance(lst, list) and all(v in lst for v in vals)
    def _op_elem_match(self, item, field, crit: Dict):
        lst = _get_value(item, field)
        return isinstance(lst, list) and any(self.evaluate(elem, crit) for elem in lst)
    def _op_bits_all_set(self, item, field, mask: int):
        val = _get_value(item, field)
        return isinstance(val, int) and (val & mask) == mask
    def _op_bits_all_clear(self, item, field, mask: int):
        val = _get_value(item, field)
        return isinstance(val, int) and (val & mask) == 0
    def _op_bits_any_set(self, item, field, mask: int):
        val = _get_value(item, field)
        return isinstance(val, int) and (val & mask) != 0
    def _op_bits_any_clear(self, item, field, mask: int):
        val = _get_value(item, field)
        return isinstance(val, int) and (val & mask) != mask


class ElectrusUpdateOperators:
    """Apply MongoDB-like update operators to an item (dict)."""

    VALID_OPERATORS = {
        "$set", "$unset", "$inc", "$mul", "$min", "$max",
        "$currentDate", "$rename", "$upsert", "$setOnInsert",
        "$push", "$pushEach", "$pop", "$pull", "$pullAll",
        "$addToSet", "$addToSetEach", "$slice", "$sort",
        "$each", "$position", "$bit", "$bitOr", "$bitAnd",
        "$bitXor", "$pipeline"
    }

    def __init__(self):
        self._handlers: Dict[str, Callable] = {
            "$set": self._update_set,
            "$unset": self._update_unset,
            "$inc": self._update_inc,
            "$mul": self._update_mul,
            "$min": self._update_min,
            "$max": self._update_max,
            "$currentDate": self._update_current_date,
            "$rename": self._update_rename,
            "$upsert": self._update_upsert,
            "$setOnInsert": self._update_set_on_insert,
            "$push": self._update_push,
            "$pushEach": self._update_push_each,
            "$pop": self._update_pop,
            "$pull": self._update_pull,
            "$pullAll": self._update_pull_all,
            "$addToSet": self._update_add_to_set,
            "$addToSetEach": self._update_add_each,
            "$slice": self._update_slice,
            "$sort": self._update_sort,
            "$each": self._update_each,
            "$position": self._update_position,
            "$bit": self._update_bit,
            "$bitOr": lambda itm, spec: self._update_bit(itm, {k: spec for k, spec in spec.items()}),  # alias
            "$bitAnd": lambda itm, spec: self._update_bit(itm, {k: spec for k, spec in spec.items()}),
            "$bitXor": lambda itm, spec: self._update_bit(itm, {k: spec for k, spec in spec.items()}),
            "$pipeline": self._update_pipeline,
        }

    def evaluate(self, item: Dict[str, Any], doc: Dict[str, Any]) -> Dict[str, Any]:
        if not doc or not any(op in self.VALID_OPERATORS for op in doc):
            raise ElectrusException("Update document must contain valid operators.")
        for op, spec in doc.items():
            if op not in self.VALID_OPERATORS:
                raise ElectrusException(f"Invalid update operator: {op}")
            self._handlers[op](item, spec)
        return item

    def _update_set(self, item, spec: Dict[str, Any]):
        for f, v in spec.items(): item.setdefault(f.split('.')[0], None)
        for field, val in spec.items():
            parts = field.split('.'); target = item
            for p in parts[:-1]:
                target = target.setdefault(p, {})
            target[parts[-1]] = val

    def _update_unset(self, item, spec: List[str]):
        for field in spec:
            parts = field.split('.'); target = item
            for p in parts[:-1]:
                target = target.get(p, {})
            target.pop(parts[-1], None)

    def _update_inc(self, item, spec: Dict[str, Union[int,float]]):
        for f, v in spec.items():
            cur = _get_value(item, f) or 0
            self._set_nested(item, f, cur + v)

    def _update_mul(self, item, spec: Dict[str, Union[int,float]]):
        for f, v in spec.items():
            cur = _get_value(item, f) or 1
            self._set_nested(item, f, cur * v)

    def _update_min(self, item, spec: Dict[str, Union[int,float]]):
        for f, v in spec.items():
            cur = _get_value(item, f)
            if isinstance(cur, (int, float)): self._set_nested(item, f, min(cur, v))

    def _update_max(self, item, spec: Dict[str, Union[int,float]]):
        for f, v in spec.items():
            cur = _get_value(item, f)
            if isinstance(cur, (int, float)): self._set_nested(item, f, max(cur, v))

    def _update_current_date(self, item, spec: Dict[str, Dict]):
        for f, cfg in spec.items():
            if cfg.get("$type") == "date":
                self._set_nested(item, f, datetime.now(timezone.utc))
            else:
                self._set_nested(item, f, {"$type": "timestamp"})

    def _update_rename(self, item, spec: Dict[str, str]):
        for old, new in spec.items():
            val = _get_value(item, old)
            self._update_unset(item, [old])
            self._set_nested(item, new, val)

    def _update_upsert(self, item, spec: Dict[str, Any]):
        for f, v in spec.items():
            if _get_value(item, f) is None:
                self._set_nested(item, f, v)

    def _update_set_on_insert(self, item, spec: Dict[str, Any]):
        # only applies if new document insertion; treat same as $set here
        self._update_set(item, spec)

    def _update_push(self, item, spec: Dict[str, Any]):
        for f, val in spec.items():
            existing = _get_value(item, f)
            if not isinstance(existing, list):
                existing = []
            if isinstance(val, dict) and "$each" in val:
                existing.extend(val["$each"])
            else:
                existing.append(val)
            self._set_nested(item, f, existing)


    def _update_push_each(self, item, spec: Dict[str, List[Any]]):
        for f, vals in spec.items():
            lst = _get_value(item, f) or []
            if not isinstance(lst, list): raise ElectrusException(f"{f} not list")
            lst.extend(vals); self._set_nested(item, f, lst)

    def _update_pop(self, item, spec: Dict[str, int]):
        for f, v in spec.items():
            lst = _get_value(item, f)
            if isinstance(lst, list):
                lst.pop(-1 if v==1 else 0)
                self._set_nested(item, f, lst)

    def _update_pull(self, item, spec: Dict[str, Any]):
        for f, v in spec.items():
            lst = _get_value(item, f)
            if isinstance(lst, list):
                filtered = [x for x in lst if x != v]
                self._set_nested(item, f, filtered)

    def _update_pull_all(self, item, spec: Dict[str, List[Any]]):
        for f, vals in spec.items():
            lst = _get_value(item, f)
            if isinstance(lst, list):
                filtered = [x for x in lst if x not in vals]
                self._set_nested(item, f, filtered)

    def _update_add_to_set(self, item, spec: Dict[str, Any]):
        for f, v in spec.items():
            lst = _get_value(item, f) or []
            if not isinstance(lst, list): raise ElectrusException(f"{f} not list")
            if v not in lst: lst.append(v)
            self._set_nested(item, f, lst)

    def _update_add_each(self, item, spec: Dict[str, List[Any]]):
        self._update_add_to_set(item, spec)

    def _update_slice(self, item, spec: Dict[str, Union[int,Dict]]):
        for f, cfg in spec.items():
            lst = _get_value(item, f)
            if isinstance(lst, list):
                if isinstance(cfg, int):
                    sliced = lst[:cfg]
                else:
                    s = cfg.get("start",0); e = cfg.get("end"); st = cfg.get("step",1)
                    sliced = lst[s:e:st]
                self._set_nested(item, f, sliced)

    def _update_sort(self, item, spec: Dict[str, int]):
        for f, order in spec.items():
            lst = _get_value(item, f)
            if isinstance(lst, list):
                try:
                    # Ensure all elements are of the same type
                    if len(set(map(type, lst))) > 1:
                        raise ElectrusException(f"Cannot sort list with mixed types in field '{f}'")
                    sorted_lst = sorted(lst, reverse=(order == -1))
                    self._set_nested(item, f, sorted_lst)
                except TypeError as e:
                    raise ElectrusException(f"Cannot sort field '{f}': {e}")


    def _update_each(self, item, spec: Dict[str, List[Any]]):
        for f, vals in spec.items():
            lst = _get_value(item, f) or []
            if not isinstance(lst, list): raise ElectrusException(f"{f} not list")
            lst.extend(vals); self._set_nested(item, f, lst)

    def _update_position(self, item, spec: Dict[str, int]):
        for f, pos in spec.items():
            lst = _get_value(item, f)
            if isinstance(lst, list):
                lst.insert(0, pos); self._set_nested(item, f, lst)

    def _update_bit(self, item, spec: Dict[str, int]):
        for f, mask in spec.items():
            val = _get_value(item, f) or 0
            if not isinstance(val, int): raise ElectrusException(f"{f} not int")
            self._set_nested(item, f, val | mask)

    def _update_pipeline(self, item, stages: List[Dict[str, Any]]):
        for stage in stages:
            op, spec = next(iter(stage.items()))
            handler = self._handlers.get(op)
            if not handler: raise ElectrusException(f"Invalid pipeline stage: {op}")
            handler(item, spec)

    def _set_nested(self, item: Dict[str, Any], path: str, value: Any):
        parts = path.split('.')
        target = item
        for p in parts[:-1]:
            target = target.setdefault(p, {})
        target[parts[-1]] = value
