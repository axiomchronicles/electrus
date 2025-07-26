from electrus_v2.core.client import Electrus

client = Electrus()
database = client["MyDatabase"]
collection = database["MyCollection"]


async def main():
    query = await collection.find().select("*", "-Date").where(id=1).execute()
    if query.acknowledged:
        print(query.raw_result)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())