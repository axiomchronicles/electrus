from datetime import datetime
from ...exception.base import ElectrusException

import re

from typing import (
    Dict,
    Any
)

class ElectrusLogicalOperators:
    VALID_OPERATORS = {
        "$eq", "$ne", "$lt", "$gt", "$lte", "$gte", "$in", "$nin", "$exists", "$unset", "$push", "$pull", "$rename", "$regex"
    }

    def evaluate(self, item:  Dict[str, Any], query: Dict[str, Any]) -> Dict[str, Any]:
        for key, value in query.items():
            if isinstance(value, dict):
                for operator, op_value in value.items():
                    if operator not in self.VALID_OPERATORS:
                        raise ElectrusException(f"Invalid operator: {operator}")

                    if operator == "$eq":
                        if item.get(key) != op_value:
                            return False
                    elif operator == "$ne":
                        if item.get(key) == op_value:
                            return False
                    elif operator == "$lt":
                        if item.get(key) >= op_value:
                            return False
                    elif operator == "$gt":
                        if item.get(key) <= op_value:
                            return False
                    elif operator == "$lte":
                        if item.get(key) > op_value:
                            return False
                    elif operator == "$gte":
                        if item.get(key) < op_value:
                            return False
                    elif operator == "$in":
                        if item.get(key) not in op_value:
                            return False
                    elif operator == "$nin":
                        if item.get(key) in op_value:
                            return False
                    elif operator == "$exists":
                        if op_value and key not in item:
                            return False
                        if not op_value and key in item:
                            return False
                    elif operator == "$unset":
                        if key in item and isinstance(item[key], dict):
                            for unset_field in op_value:
                                if unset_field in item[key]:
                                    return False
                    elif operator == "$push":
                        if key in item and isinstance(item[key], dict):
                            for push_field, push_value in op_value.items():
                                if push_field in item[key] and isinstance(item[key][push_field], list):
                                    if push_value not in item[key][push_field]:
                                        return False
                                else:
                                    return False
                    elif operator == "$pull":
                        if key in item and isinstance(item[key], dict):
                            for pull_field, pull_value in op_value.items():
                                if pull_field in item[key] and isinstance(item[key][pull_field], list):
                                    if pull_value in item[key][pull_field]:
                                        return False
                                else:
                                    return False
                    elif operator == "$rename":
                        if key in item and isinstance(item[key], dict):
                            for rename_field, new_name in op_value.items():
                                if rename_field in item[key]:
                                    return False
                    
                    elif operator == "$regex":
                        if key not in item or not isinstance(item[key], str):
                            return False
                        pattern = re.compile(op_value)
                        if not pattern.search(item[key]):
                            return False
            else:
                if item.get(key) != value:
                    return False
        return True

class ElectrusUpdateOperators:
    VALID_OPERATORS = {
        "$set", "$unset", "$push", "$pull", "$rename", "$inc", "$mul", "$min", "$max",
        "$currentDate", "$addToSet", "$pop", "$pullAll", "$bit", "$addToSetEach", "$pullEach",
        "$unsetMany", "$pushEach", "$regex", "$slice", "$sort", "$each", "$position", "$slice", "$upsert"
    }

    def evaluate(self, item: Dict[str, Any], update_data: Dict[str, Any]) -> Dict[str, Any]:
        if not update_data or not any(op in self.VALID_OPERATORS for op in update_data):
            raise ElectrusException("Update data must contain valid operators.")

        for operator, op_value in update_data.items():
            if operator not in self.VALID_OPERATORS:
                raise ElectrusException(
                    f"Invalid update operator: '{operator}'."
                )

            if operator == "$set":
                for set_field, set_value in op_value.items():
                    if set_field in item and isinstance(item[set_field], dict):
                        item[set_field].update(set_value)
                    else:
                        item[set_field] = set_value
            elif operator == "$unset":
                for unset_field in op_value:
                    if unset_field in item and isinstance(item[unset_field], dict):
                        for field in unset_field:
                            item[unset_field].pop(field, None)
                    else:
                        item.pop(unset_field, None)
            elif operator == "$push":
                for push_field, push_value in op_value.items():
                    if push_field in item and isinstance(item[push_field], list):
                        item[push_field].append(push_value)
                    else:
                        item[push_field] = [push_value]
            elif operator == "$pull":
                for pull_field, pull_value in op_value.items():
                    if pull_field in item and isinstance(item[pull_field], list):
                        item[pull_field] = [item for item in item[pull_field] if item != pull_value]
            elif operator == "$rename":
                for rename_field, new_name in op_value.items():
                    if rename_field in item and isinstance(item[rename_field], dict):
                        item[new_name] = item.pop(rename_field)
                    else:
                        item[new_name] = item.pop(rename_field, None)
            elif operator == "$inc":
                for inc_field, inc_value in op_value.items():
                    if inc_field in item and isinstance(item[inc_field], (int, float)):
                        item[inc_field] += inc_value
                    else:
                        item[inc_field] = inc_value
            elif operator == "$mul":
                for mul_field, mul_value in op_value.items():
                    if mul_field in item and isinstance(item[mul_field], (int, float)):
                        item[mul_field] *= mul_value
            elif operator == "$min":
                for min_field, min_value in op_value.items():
                    if min_field in item and isinstance(item[min_field], (int, float)):
                        item[min_field] = min(item[min_field], min_value)
            elif operator == "$max":
                for max_field, max_value in op_value.items():
                    if max_field in item and isinstance(item[max_field], (int, float)):
                        item[max_field] = max(item[max_field], max_value)
            elif operator == "$currentDate":
                for date_field, date_type in op_value.items():
                    item[date_field] = datetime.utcnow() if date_type == { "$type": "date" } else { "$type": "timestamp" }
            elif operator == "$addToSet":
                for add_field, add_value in op_value.items():
                    if add_field in item and isinstance(item[add_field], list):
                        if add_value not in item[add_field]:
                            item[add_field].append(add_value)
                    else:
                        item[add_field] = [add_value]
            elif operator == "$pop":
                for pop_field, pop_value in op_value.items():
                    if pop_field in item and isinstance(item[pop_field], list):
                        if pop_value == 1:
                            item[pop_field].pop()
                        elif pop_value == -1:
                            item[pop_field].pop(0)
            elif operator == "$pullAll":
                for pull_field, pull_values in op_value.items():
                    if pull_field in item and isinstance(item[pull_field], list):
                        item[pull_field] = [value for value in item[pull_field] if value not in pull_values]
            elif operator == "$bit":
                for bit_field, bit_value in op_value.items():
                    if bit_field in item and isinstance(item[bit_field], (int, float)):
                        item[bit_field] = item[bit_field] | bit_value
            elif operator == "$addToSetEach":
                for add_field, add_values in op_value.items():
                    if add_field in item and isinstance(item[add_field], list):
                        for value in add_values:
                            if value not in item[add_field]:
                                item[add_field].append(value)
                    else:
                        item[add_field] = add_values
            elif operator == "$pullEach":
                for pull_field, pull_values in op_value.items():
                    if pull_field in item and isinstance(item[pull_field], list):
                        item[pull_field] = [value for value in item[pull_field] if value not in pull_values]
            elif operator == "$unsetMany":
                for unset_field in op_value:
                    if unset_field in item and isinstance(item[unset_field], dict):
                        for field in unset_field:
                            item[unset_field].pop(field, None)
                    else:
                        item.pop(unset_field, None)
            elif operator == "$pushEach":
                for push_field, push_values in op_value.items():
                    if push_field in item and isinstance(item[push_field], list):
                        item[push_field].extend(push_values)
                    else:
                        item[push_field] = push_values

            elif operator == "$regex":
                for regex_field, regex_pattern in op_value.items():
                    if regex_field in item and isinstance(item[regex_field], str):
                        pattern = re.compile(regex_pattern)
                        if pattern.search(item[regex_field]):
                            return item
                        
            elif operator == "$slice":
                return self._apply_slice_operator(item, op_value)
            elif operator == "$sort":
                return self._apply_sort_operator(item, op_value)
            elif operator == "$each":
                return self._apply_each_operator(item, op_value)
            elif operator == "$position":
                return self._apply_position_operator(item, op_value)
            elif operator == "$upsert":
                return self._apply_upsert_operator(item, op_value)
            
        return item
    
    def _apply_slice_operator(self, item, op_value) -> bool:
        for slice_field, slice_value in op_value.items():
            if slice_field in item and isinstance(item[slice_field], list):
                if isinstance(slice_value, int):
                    item[slice_field] = item[slice_field][:slice_value]
                elif isinstance(slice_value, dict):
                    start = slice_value.get("start", 0)
                    end = slice_value.get("end")
                    step = slice_value.get("step", 1)
                    item[slice_field] = item[slice_field][start:end:step]
                else:
                    return False
            else:
                return False
        return True

    def _apply_sort_operator(self, item, op_value) -> bool:
        for sort_field, sort_order in op_value.items():
            if sort_field in item and isinstance(item[sort_field], list):
                item[sort_field].sort(reverse=(sort_order == -1))
            else:
                return False
        return True

    def _apply_each_operator(self, item, op_value) -> bool:
        for each_field, each_values in op_value.items():
            if each_field in item and isinstance(item[each_field], list):
                item[each_field].extend(each_values)
            else:
                return False
        return True

    def _apply_position_operator(self, item, op_value) -> bool:
        for position_field, position_value in op_value.items():
            if position_field in item and isinstance(item[position_field], list):
                if isinstance(position_value, int):
                    item[position_field].insert(0, position_value)
                else:
                    return False
            else:
                return False
        return True

    def _apply_upsert_operator(self, item, op_value) -> bool:
        for upsert_field, upsert_value in op_value.items():
            if upsert_field not in item:
                item[upsert_field] = upsert_value
            elif isinstance(item[upsert_field], dict):
                item[upsert_field].update(upsert_value)
            else:
                return False
        return True