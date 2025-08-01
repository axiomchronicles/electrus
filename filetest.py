import asyncio
import json
import os
from electrus_v2.core.client import Electrus

from electrus_v2.utils.distinct import DistinctOperation

client = Electrus()
database = client["MyDatabase"]
users = database["testdata"]
cities = database["cities"]

async def main():
    pipeline = [
        {"$match": {"score": {"$gte": 50}}},

        {"$lookup": {
            "from": "cities",
            "localField": "city",
            "foreignField": "_id",
            "as": "city_info"
        }},

        {"$unwind": "$city_info"},

        {"$addFields": {
            "ageGroup": {
                "$cond": {
                    "if": {"$gte": ["$age", 30]},
                    "then": "senior",
                    "else": "junior"
                }
            },
            "scoreGrade": {
                "$switch": {
                    "branches": [
                        {"case": {"$gte": ["$score", 85]}, "then": "A"},
                        {"case": {"$gte": ["$score", 70]}, "then": "B"},
                        {"case": {"$gte": ["$score", 60]}, "then": "C"},
                    ],
                    "default": "D"
                }
            },
            "cityPopulation": "$city_info.population"
        }},

        {"$group": {
            "_id": {"ageGroup": "$ageGroup", "scoreGrade": "$scoreGrade"},
            "avgAge": {"$avg": "$age"},
            "maxScore": {"$max": "$score"},
            "users": {"$push": "$name"},
            "cities": {"$addToSet": "$city"},
            "totalPopulation": {"$sum": "$cityPopulation"}
        }},

        {"$sort": {"avgAge": -1}},

        {"$facet": {
            "top_groups": [{"$limit": 2}],
            "all_groups": [{"$sort": {"_id.ageGroup": 1}}],
        }}
    ]



    aggregtion = await users.aggregation().execute(pipeline, additional_collections = {"cities": cities})
    print(aggregtion)



# --- Entry point ---
if __name__ == "__main__":
    asyncio.run(main())

