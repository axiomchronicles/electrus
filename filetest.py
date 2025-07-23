from datetime import datetime
from electrus_v2.partials.operators import ElectrusLogicalOperators, ElectrusUpdateOperators


def logical_operator_demo():
    logical = ElectrusLogicalOperators()
    doc = {
        "name": "Luna",
        "age": 25,
        "tags": ["ai", "ml"],
        "nested": {"status": "active", "rank": 5},
        "score": 85,
        "alive": True,
        "binary": 0b1101,
        "friends": [{"name": "Nova", "age": 22}, {"name": "Cora", "age": 20}]
    }

    queries = [
        {"age": {"$eq": 25}},                         # Equality
        {"age": {"$ne": 30}},                         # Not equal
        {"age": {"$lt": 30}},                         # Less than
        {"age": {"$lte": 25}},                        # Less than or equal
        {"age": {"$gt": 20}},                         # Greater than
        {"age": {"$gte": 25}},                        # Greater than or equal
        {"tags": {"$in": ["ml"]}},                    # In list
        {"tags": {"$nin": ["js"]}},                   # Not in list
        {"nested.status": {"$exists": True}},         # Field exists
        {"name": {"$regex": "^L.*"}},                 # Regex
        {"$and": [{"age": {"$gt": 20}}, {"score": {"$gt": 80}}]},  # Logical AND
        {"$or": [{"name": "Luna"}, {"name": "Nova"}]},             # Logical OR
        {"$nor": [{"name": "John"}]},                               # NOR
        {"age": {"$not": {"$lt": 20}}},              # NOT
        {"alive": {"$type": "bool"}},               # Type check
        {"score": {"$mod": [10, 5]}},                # Modulo
        {"$expr": {"$eq": ["$age", 25]}},            # Expr comparison
        {"tags": {"$size": 2}},                      # List size
        {"tags": {"$all": ["ai", "ml"]}},            # List contains all
        {"friends": {"$elemMatch": {"age": {"$gt": 21}}}},  # elemMatch
        {"binary": {"$bitsAllSet": 0b0101}},
        {"binary": {"$bitsAllClear": 0b0010}},
        {"binary": {"$bitsAnySet": 0b1000}},
        {"binary": {"$bitsAnyClear": 0b0110}},
    ]

    print("Logical Operator Results:")
    for q in queries:
        print(f"Query: {q} -> Match: {logical.evaluate(doc, q)}")
    print("-" * 50)


def update_operator_demo():
    updater = ElectrusUpdateOperators()

    doc = {
        "name": "Luna",
        "age": 20,
        "tags": ["ai", "ml"],
        "stats": {"score": 50, "rank": 3},
        "binary": 0b0101
    }

    update = {
        "$set": {"status": "active", "stats.level": 2},
        "$unset": ["name"],
        "$inc": {"age": 2},
        "$mul": {"stats.score": 2},
        "$min": {"stats.rank": 2},       # Only updates if current > 2
        "$max": {"stats.rank": 5},       # Only updates if current < 5
        "$currentDate": {"stats.lastSeen": {"$type": "date"}},
        "$rename": {"status": "state"},
        "$upsert": {"profile": "created"},
        "$setOnInsert": {"onInsert": True},
        "$push": {"tags": "python"},
        "$pushEach": {"tags": ["fastapi", "llm"]},
        "$pop": {"tags": 1},  # Remove last
        "$pull": {"tags": "ml"},
        "$pullAll": {"tags": ["ai"]},
        "$addToSet": {"tags": "openai"},
        "$addToSetEach": {"tags": ["openai", "gpt"]},
        "$slice": {"tags": 2},
        "$sort": {"tags": 1},  # Ascending
        "$each": {"tags": ["extra1", "extra2"]},
        "$position": {"tags": 0},  # Insert `0` at start
        "$bit": {"binary": 0b0010},  # Bitwise OR
        "$pipeline": [{"$set": {"pipelineResult": True}}],
    }

    print("Document Before Update:")
    print(doc)
    updated_doc = updater.evaluate(doc, update)
    print("\nDocument After Update:")
    print(updated_doc)
    print("-" * 50)


if __name__ == "__main__":
    print("=== ELECTRUS LOGICAL OPERATOR DEMO ===")
    logical_operator_demo()

    print("\n=== ELECTRUS UPDATE OPERATOR DEMO ===")
    update_operator_demo()
