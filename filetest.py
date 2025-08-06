import asyncio
from electrus_v2.core.client import Electrus

client = Electrus()
database = client["MyTestDb"]
collection = database["MyTestCollection"]

sample_data = [
    {
        "id": 1,
        "name": "Alice Johnson",
        "age": 28,
        "email": "alice@example.com",
        "address": {
            "street": "123 Maple Street",
            "city": "Springfield",
            "zip": "62704"
        },
        "interests": ["reading", "hiking", "coding"],
        "is_active": True,
        "signup_date": "2024-11-10T14:32:00Z"
    },
    {
        "id": 2,
        "name": "Bob Smith",
        "age": 34,
        "email": "bob@example.com",
        "address": {
            "street": "456 Oak Avenue",
            "city": "Shelbyville",
            "zip": "61523"
        },
        "interests": ["gaming", "travel", "cycling"],
        "is_active": False,
        "signup_date": "2025-01-22T09:15:00Z"
    },
    {
        "id": 3,
        "name": "Charlie Doe",
        "age": 22,
        "email": "charlie@example.com",
        "address": {
            "street": "789 Pine Road",
            "city": "Ogdenville",
            "zip": "60213"
        },
        "interests": ["music", "sports"],
        "is_active": True,
        "signup_date": "2025-03-05T17:45:00Z"
    },
    {
        "id": 4,
        "name": "Diana Prince",
        "age": 30,
        "email": "diana@example.com",
        "address": {
            "street": "321 Birch Blvd",
            "city": "Capitol City",
            "zip": "60606"
        },
        "interests": ["combat", "history", "justice"],
        "is_active": True,
        "signup_date": "2024-12-01T12:00:00Z"
    },
    {
        "id": 5,
        "name": "Ethan Hunt",
        "age": 40,
        "email": "ethan@example.com",
        "address": {
            "street": {
                "name": "Main 2D Street"
            },
            "city": "Metroville",
            "zip": "70001"
        },
        "interests": ["espionage", "travel", "skydiving"],
        "is_active": False,
        "signup_date": "2023-08-19T08:00:00Z"
    }
]

async def main():
    query = await collection.find().where(id = 6).execute()
    print(query.error)

if __name__ == "__main__":
    asyncio.run(main())