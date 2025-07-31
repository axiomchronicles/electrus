import json
import asyncio
from electrus_v2.core.client import Electrus

client = Electrus()
database = client["MyDatabase"]
users = database["users"]
orders = database["orders"]

async def main():
    result = await users.find().where(score__gte=70).count()
    print("\n✅ TEST X: count() users with score >= 70")
    print(f"Count: {result}")

if __name__ == "__main__":
    asyncio.run(main())
