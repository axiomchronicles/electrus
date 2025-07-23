import unittest
from datetime import datetime

from electrus_v2.partials.insert import InsertData, FieldOp
from electrus_v2.partials.results import DatabaseActionResult

# Dummy in-memory handler
class DummyJsonHandler:
    def __init__(self):
        self.store = {}

    async def read_async(self, file):
        return self.store.get(file, {"data": []})

    async def write_async(self, file, data):
        self.store[file] = {"data": data}


class TestInsertData(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.file = "fake.json"
        self.handler = DummyJsonHandler()
        self.inserter = InsertData(collection_file=self.file, json_handler=self.handler)

    async def test_insert_auto_inc(self):
        doc = {"count": "$auto"}
        result = await self.inserter._obl_one(doc)
        self.assertTrue(result.success)
        self.assertIsNotNone(result.inserted_id)

    async def test_insert_unique_id(self):
        doc = {
            "uid": {FieldOp.UNIQUE.value: {"length": 12, "format": "alphanumeric"}}
        }
        result = await self.inserter._obl_one(doc)
        self.assertTrue(result.success)
        data = self.handler.store[self.file]["data"]
        self.assertIn("uid", data[0])
        self.assertEqual(len(data[0]["uid"]), 12)

    async def test_insert_with_datetime(self):
        doc = {"created_at": "$datetime"}
        result = await self.inserter._obl_one(doc)
        self.assertTrue(result.success)
        data = self.handler.store[self.file]["data"]
        self.assertIn("created_at", data[0])
        self.assertIsInstance(data[0]["created_at"], str)
        self.assertIn(":", data[0]["created_at"])

    async def test_insert_duplicate(self):
        doc = {"name": "Sahur"}
        result1 = await self.inserter._obl_one(doc)
        result2 = await self.inserter._obl_one(doc)
        self.assertTrue(result2.success)
        self.assertEqual(result2.raw_result.get("warning"), "duplicate")

    async def test_bulk_insert(self):
        docs = [
            {"key": {FieldOp.UNIQUE.value: {"length": 6}}},
            {"key": {FieldOp.UNIQUE.value: {"length": 6}}}
        ]
        result = await self.inserter._obl_many(docs)
        self.assertTrue(result.success)
        self.assertEqual(len(result.inserted_ids), 2)


if __name__ == "__main__":
    unittest.main()
