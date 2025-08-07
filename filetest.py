from electrus import Electrus

import os
import enum
import json
import hashlib
import typing
import dataclasses

client = Electrus()
database = client["MyTestingDatabase"]

collection = database["MyTestingCollection"]

sample_users = [
    {
        "id": 1,
        "name": "Alice Johnsons",
        "email": "alice@example.com",
        "password": "hashed_pw_1",
        "age": 28,
        "address": {
            "street": "123 Maple St",
            "city": "Springfield",
            "zip": "12345"
        },
        "roles": ["admin", "editor"],
        "is_active": True
    },
    {
        "id": 2,
        "name": "Bob Smith",
        "email": "bob@example.com",
        "password": "hashed_pw_2",
        "age": 34,
        "address": {
            "street": "456 Oak Ave",
            "city": "Riverside",
            "zip": "67890"
        },
        "roles": ["user"],
        "is_active": True
    },
    {
        "id": 3,
        "name": "Charlie Davis",
        "email": "charlie@example.com",
        "password": "hashed_pw_3",
        "age": 41,
        "address": {
            "street": "789 Pine Rd",
            "city": "Hillview",
            "zip": "24680"
        },
        "roles": ["moderator"],
        "is_active": False
    },
    {
        "id": 4,
        "name": "Dana White",
        "email": "dana@example.com",
        "password": "hashed_pw_4",
        "age": 22,
        "address": {
            "street": "101 Cedar Blvd",
            "city": "Lakeside",
            "zip": "13579"
        },
        "roles": ["user", "support"],
        "is_active": True
    }
]

"""
Helper Functions
"""

def hashPasswrod(password: str, salt: typing.Optional[str] = None) -> str:
    salt = os.urandom(10).hex() if not salt else salt
    obj = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100000)
    return "{}${}".format(salt, obj.hex())

def checkPassword(password: str, hash: str) -> bool:
    salt, hashObj = hash.split("$")
    newObj = hashPasswrod(password, salt).split("$")[1]
    return hashObj == newObj

@dataclasses.dataclass
class Users:
    id: int
    name: str
    email: str
    password: str


async def handlecollectionOperations():
    query = await collection.insertMany(data_list = sample_users, overwrite = False)
    print(query.acknowledged)

    query = await collection.find().select("*").execute()
    if query.acknowledged:
        print(json.dumps(query.raw_result, indent=2))

    query = await collection.update(
        filter = {"age": {"$gt": 30}}, multi = True,
        update_data = {"$set": {"salary": 30000}}
    )

    print((await collection.find().select("*").execute()).raw_result)

    query = await collection.delete().where(id = 1).execute()
    if query.acknowledged:
        print((await collection.find().select("*").execute()).raw_result)

if __name__ == "__main__":
    import asyncio
    asyncio.run(handlecollectionOperations())