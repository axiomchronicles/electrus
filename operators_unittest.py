import unittest
from datetime import datetime
from axiomelectrus.partials.operators import ElectrusLogicalOperators, ElectrusUpdateOperators


class TestElectrusLogicalOperators(unittest.TestCase):
    def setUp(self):
        self.logic = ElectrusLogicalOperators()
        self.doc = {
            "name": "Luna",
            "age": 25,
            "tags": ["ai", "ml"],
            "nested": {"status": "active", "rank": 5},
            "score": 85,
            "alive": True,
            "binary": 0b1101,
            "friends": [{"name": "Nova", "age": 22}, {"name": "Cora", "age": 20}]
        }

    def test_comparison_ops(self):
        self.assertTrue(self.logic.evaluate(self.doc, {"age": {"$eq": 25}}))
        self.assertTrue(self.logic.evaluate(self.doc, {"age": {"$ne": 30}}))
        self.assertTrue(self.logic.evaluate(self.doc, {"age": {"$lt": 30}}))
        self.assertTrue(self.logic.evaluate(self.doc, {"age": {"$lte": 25}}))
        self.assertTrue(self.logic.evaluate(self.doc, {"age": {"$gt": 20}}))
        self.assertTrue(self.logic.evaluate(self.doc, {"age": {"$gte": 25}}))

    def test_array_and_logic_ops(self):
        self.assertTrue(self.logic.evaluate(self.doc, {"tags": {"$in": ["ml"]}}))
        self.assertTrue(self.logic.evaluate(self.doc, {"tags": {"$nin": ["js"]}}))
        self.assertTrue(self.logic.evaluate(self.doc, {"tags": {"$all": ["ai", "ml"]}}))
        self.assertTrue(self.logic.evaluate(self.doc, {"tags": {"$size": 2}}))
        self.assertTrue(self.logic.evaluate(self.doc, {"friends": {"$elemMatch": {"age": {"$gt": 21}}}}))

    def test_misc_ops(self):
        self.assertTrue(self.logic.evaluate(self.doc, {"nested.status": {"$exists": True}}))
        self.assertTrue(self.logic.evaluate(self.doc, {"name": {"$regex": "^L.*"}}))
        self.assertTrue(self.logic.evaluate(self.doc, {"$and": [{"age": {"$gt": 20}}, {"score": {"$gt": 80}}]}))
        self.assertTrue(self.logic.evaluate(self.doc, {"$or": [{"name": "Luna"}, {"name": "Nova"}]}))
        self.assertTrue(self.logic.evaluate(self.doc, {"$nor": [{"name": "John"}]}))
        self.assertTrue(self.logic.evaluate(self.doc, {"age": {"$not": {"$lt": 20}}}))
        self.assertTrue(self.logic.evaluate(self.doc, {"alive": {"$type": "bool"}}))
        self.assertTrue(self.logic.evaluate(self.doc, {"score": {"$mod": [10, 5]}}))
        self.assertTrue(self.logic.evaluate(self.doc, {"$expr": {"$eq": ["$age", 25]}}))

    def test_bitwise_ops(self):
        self.assertTrue(self.logic.evaluate(self.doc, {"binary": {"$bitsAllSet": 0b0101}}))
        self.assertTrue(self.logic.evaluate(self.doc, {"binary": {"$bitsAllClear": 0b0010}}))
        self.assertTrue(self.logic.evaluate(self.doc, {"binary": {"$bitsAnySet": 0b1000}}))
        self.assertTrue(self.logic.evaluate(self.doc, {"binary": {"$bitsAnyClear": 0b0110}}))


class TestElectrusUpdateOperators(unittest.TestCase):
    def setUp(self):
        self.updater = ElectrusUpdateOperators()
        self.doc = {
            "name": "Luna",
            "age": 20,
            "tags": ["ai", "ml"],
            "stats": {"score": 50, "rank": 3},
            "binary": 0b0101
        }

    def test_update_set_unset(self):
        updated = self.updater.evaluate(self.doc.copy(), {
            "$set": {"status": "active"},
            "$unset": ["name"]
        })
        self.assertEqual(updated.get("status"), "active")
        self.assertNotIn("name", updated)

    def test_inc_mul_min_max(self):
        updated = self.updater.evaluate(self.doc.copy(), {
            "$inc": {"age": 2},
            "$mul": {"stats.score": 2},
            "$min": {"stats.rank": 2},
            "$max": {"stats.rank": 5}
        })
        self.assertEqual(updated["age"], 22)
        self.assertEqual(updated["stats"]["score"], 100)
        self.assertEqual(updated["stats"]["rank"], 5)

    def test_current_date(self):
        updated = self.updater.evaluate(self.doc.copy(), {
            "$currentDate": {"stats.lastSeen": {"$type": "date"}}
        })
        self.assertIn("lastSeen", updated["stats"])
        self.assertIsInstance(updated["stats"]["lastSeen"], datetime)

    def test_rename_upsert(self):
        updated = self.updater.evaluate(self.doc.copy(), {
            "$set": {"status": "ok"},
            "$rename": {"status": "state"},
            "$upsert": {"profile": "created"}
        })
        self.assertEqual(updated.get("state"), "ok")
        self.assertEqual(updated.get("profile"), "created")

    def test_array_ops(self):
        updated = self.updater.evaluate(self.doc.copy(), {
            "$push": {"tags": "python"},
            "$pushEach": {"tags": ["llm"]},
            "$pop": {"tags": 1},
            "$pull": {"tags": "ml"},
            "$pullAll": {"tags": ["ai"]},
            "$addToSet": {"tags": "openai"},
            "$addToSetEach": {"tags": ["openai", "gpt"]},
            "$each": {"tags": ["x"]},
            "$position": {"tags": 0},
            "$slice": {"tags": 2},
            "$sort": {"tags": 1},
        })
        self.assertIsInstance(updated["tags"], list)
        self.assertLessEqual(len(updated["tags"]), 2)

    def test_bit_pipeline(self):
        updated = self.updater.evaluate(self.doc.copy(), {
            "$bit": {"binary": 0b0010},
            "$pipeline": [{"$set": {"pipelineFlag": True}}]
        })
        self.assertTrue(updated["binary"] & 0b0010)
        self.assertTrue(updated.get("pipelineFlag", False))


if __name__ == "__main__":
    unittest.main()
