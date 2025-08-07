import json
import os
import aiofiles
import re
import random
import string
from typing import Any, Dict, List, Union, Optional
from datetime import datetime, timedelta

from .objectID import ObjectId
from ...exception.base import ElectrusException
from .result import DatabaseActionResult 

class InsertData:
    def __init__(self, collection_path: str) -> None:
        self.collection_path: str = collection_path

    async def _generate_unique_id(self) -> str:
        return ''.join(random.choices(string.ascii_letters + string.digits, k=10))
    
    async def _generate_numeric_id(self, length: int = 6) -> str:
        return ''.join(random.choices(string.digits, k=length))

    async def _generate_alphanumeric_id(self, length: int = 8) -> str:
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

    async def _generate_uuid(self) -> str:
        return str(ObjectId.generate())

    async def _process_auto_inc(self, data: Dict[str, Union[int, str]], collection_data: List[Dict[str, Any]]) -> None:
        auto_inc_keys: List[str] = [key for key, value in data.items() if value == '$auto']
        for key in auto_inc_keys:
            existing_ids: List[int] = [item.get(key, 0) for item in collection_data if isinstance(item.get(key), int)]
            data[key] = max(existing_ids, default=0) + 1

    async def _process_unique_id(self, data: Dict[str, Union[int, str]]) -> None:
        unique_id_keys: List[str] = [key for key, value in data.items() if isinstance(value, dict) and value.get('type') == '$unique']
        for key in unique_id_keys:
            length = data[key].get('length', 10)
            if data[key].get('format') == 'numeric':
                data[key] = await self._generate_numeric_id(length)
            elif data[key].get('format') == 'alphanumeric':
                data[key] = await self._generate_alphanumeric_id(length)
            elif data[key].get('format') == 'uuid':
                data[key] = await self._generate_uuid()
            else:
                data[key] = await self._generate_unique_id(length)

    async def _process_time_now(self, data: Dict[str, Any]) -> None:
        time_now_keys: List[str] = [key for key, value in data.items() if value == '$time']
        for key in time_now_keys:
            data[key] = datetime.now().strftime('%H:%M:%S')

    async def _process_datetime(self, data: Dict[str, Any]) -> None:
        datetime_keys: List[str] = [key for key, value in data.items() if value == '$datetime']
        for key in datetime_keys:
            data[key] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    async def _process_date(self, data: Dict[str, Any]) -> None:
        date_keys: List[str] = [key for key, value in data.items() if value == '$date']
        for key in date_keys:
            data[key] = datetime.now().strftime('%Y-%m-%d')

    async def _process_timestamp(self, data: Dict[str, Any]) -> None:
        timestamp_keys: List[str] = [key for key, value in data.items() if value == '$timestamp']
        for key in timestamp_keys:
            data[key] = int(datetime.now().timestamp())

    async def _process_date_add(self, data: Dict[str, Any]) -> None:
        date_add_keys: List[str] = [key for key, value in data.items() if isinstance(value, dict) and '$date_add' in value]
        for key in date_add_keys:
            match = re.match(r'(-?\d+)([a-zA-Z]+)', data[key]['$date_add'])
            if match:
                delta_value = int(match.group(1))
                delta_unit = match.group(2)
                current_date = datetime.now()
                if delta_unit.lower() in ['day', 'days']:
                    data[key] = (current_date + timedelta(days=delta_value)).strftime('%Y-%m-%d')
                elif delta_unit.lower() in ['hour', 'hours']:
                    data[key] = (current_date + timedelta(hours=delta_value)).strftime('%Y-%m-%d %H:%M:%S')
                elif delta_unit.lower() in ['minute', 'minutes']:
                    data[key] = (current_date + timedelta(minutes=delta_value)).strftime('%Y-%m-%d %H:%M:%S')
                elif delta_unit.lower() in ['second', 'seconds']:
                    data[key] = (current_date + timedelta(seconds=delta_value)).strftime('%Y-%m-%d %H:%M:%S')

    async def _process_date_sub(self, data: Dict[str, Any]) -> None:
        date_sub_keys: List[str] = [key for key, value in data.items() if isinstance(value, dict) and '$date_sub' in value]
        for key in date_sub_keys:
            match = re.match(r'(-?\d+)([a-zA-Z]+)', data[key]['$date_sub'])
            if match:
                delta_value = int(match.group(1))
                delta_unit = match.group(2)
                current_date = datetime.now()
                if delta_unit.lower() in ['day', 'days']:
                    data[key] = (current_date - timedelta(days=delta_value)).strftime('%Y-%m-%d')
                elif delta_unit.lower() in ['hour', 'hours']:
                    data[key] = (current_date - timedelta(hours=delta_value)).strftime('%Y-%m-%d %H:%M:%S')
                elif delta_unit.lower() in ['minute', 'minutes']:
                    data[key] = (current_date - timedelta(minutes=delta_value)).strftime('%Y-%m-%d %H:%M:%S')
                elif delta_unit.lower() in ['second', 'seconds']:
                    data[key] = (current_date - timedelta(seconds=delta_value)).strftime('%Y-%m-%d %H:%M:%S')

    async def _process_date_diff(self, data: Dict[str, Any]) -> None:
        date_diff_keys: List[str] = [key for key, value in data.items() if isinstance(value, dict) and '$date_diff' in value]
        for key in date_diff_keys:
            start_date = datetime.strptime(data[key]['$date_diff']['start_date'], '%Y-%m-%d')
            end_date = datetime.strptime(data[key]['$date_diff']['end_date'], '%Y-%m-%d')
            difference = end_date - start_date
            data[key] = difference.days

    async def _process_date_format(self, data: Dict[str, Any]) -> None:
        date_format_keys: List[str] = [key for key, value in data.items() if isinstance(value, dict) and '$date_format' in value]
        for key in date_format_keys:
            date_obj = datetime.strptime(data[key]['$date_format']['date'], '%Y-%m-%d %H:%M:%S')
            format_str = data[key]['$date_format']['format']
            data[key] = date_obj.strftime(format_str)

    async def _write_collection_data(self, collection_data: List[Dict[str, Any]]) -> None:
        async with aiofiles.open(self.collection_path, 'w') as file:
            await file.write(json.dumps(collection_data, indent=4))

    async def _read_collection_data(self) -> List[Dict[str, Any]]:
        if os.path.exists(self.collection_path):
            async with aiofiles.open(self.collection_path, 'r') as file:
                return json.loads(await file.read())
        return []

    async def _update_collection_data(self, data: Dict[str, Any], overwrite_duplicate: bool = False) -> DatabaseActionResult:
        try:
            collection_data: List[Dict[str, Any]] = await self._read_collection_data()
            data['_id'] = ObjectId.generate()
            insert_count = 0

            if overwrite_duplicate:
                index: Optional[int] = next((i for i, item in enumerate(collection_data) if item == data), None)
                if index is not None:
                    collection_data[index] = data
                else:
                    await self._process_unique_id(data)
                    await self._process_auto_inc(data, collection_data)
                    await self._process_datetime(data)
                    await self._process_date(data)
                    await self._process_timestamp(data)
                    await self._process_date_diff(data)
                    await self._process_date_format(data)
                    await self._process_date_add(data)
                    await self._process_date_sub(data)
                    await self._process_time_now(data)
                    
                    collection_data.append(data)
                    insert_count += 1
            else:
                if data not in collection_data:
                    await self._process_unique_id(data)
                    await self._process_auto_inc(data, collection_data)
                    await self._process_datetime(data)
                    await self._process_date(data)
                    await self._process_timestamp(data)
                    await self._process_date_diff(data)
                    await self._process_date_format(data)
                    await self._process_date_add(data)
                    await self._process_date_sub(data)
                    await self._process_time_now(data)

                    collection_data.append(data)
                    insert_count += 1

            await self._write_collection_data(collection_data)
            return DatabaseActionResult(success=True, modified_count=insert_count, inserted_ids=data.get('_id'))

        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise ElectrusException(f"Error handling file or JSON data: {e}")
        except Exception as e:
            raise ElectrusException(f"Error updating data: {e}")

    async def _obl_one(self, data: Dict[str, Any], overwrite_duplicate: bool = False) -> DatabaseActionResult:
        return await self._update_collection_data(data, overwrite_duplicate)

    async def _obl_many(self, data_list: List[Dict[str, Any]], overwrite_duplicate: bool = False) -> DatabaseActionResult:
        try:
            insert_count = 0
            object_ids = []
            for data in data_list:
                result = await self._update_collection_data(data, overwrite_duplicate)
                if result.success:
                    insert_count += result.modified_count
                    object_ids.append(result.object_id)

            return DatabaseActionResult(success=True, modified_count=insert_count, inserted_ids=object_ids)

        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise ElectrusException(f"Error handling file or JSON data: {e}")
        except Exception as e:
            raise ElectrusException(f"Error updating multiple data: {e}")
