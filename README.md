# Electrus Database

<p align="center">
  <img src="assets/electrus.png" alt="Electrus Logo"/>
</p>


Electrus is a lightweight asynchronous & synchronous database module designed for Python, providing essential functionalities for data storage and retrieval.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Examples](#examples)
- [Documantation](#documantation)
- [Support](#support)
## Overview

Electrus offers functionalities to manage collections and perform various operations such as insertion, updates, deletion, and data querying.

## Installation

To install Electrus, use the following pip command:

```bash
$ pip install electrus
```

## Getting Started

`Asynchronous`

```python
import electrus.asynchronous as electrus

client = electrus.Electrus()
database = client['mydb'] # enter you desire database
collection = database['mycollection']
```

`Synchronous`

```python
import electrus.synchronous as electrus

client = electrus.Electrus()
database = client['mydb'] # enter you desire database
collection = database['mycollection']
```

## Examples

### `Asynchronous`

### Inserting data operation

```python
# save this as main.py

import asyncio

import electrus as electrus
from electrus.exception import ElectrusException

client = electrus.Electrus()
database = client['mydb']
collection = database['mycollection']

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

```
`run the script`
```bash
$ python main.py
```
### `Synchronous`

### Inserting data operation

```python
# save this as main.py

import electrus.synchronous as electrus
from electrus.exception import ElectrusException

client = electrus.Electrus()
database = client['mydb']
collection = database['mycollection']

data = {
  "id": "auto_inc",
  "name": "Embrake | Electrus",
  "email": ["embrakeproject@gmail.com", "control@vvfin.in"],
  "role": "user"
}

try:
  query = collection.insert_one(data)
  if query:
    print("Data inserted successfully!")
except ElectrusException as e:
  print("Something went wrong {}".format(e))

```
`run the script`
```bash
$ python main.py
```

## Documantation

The complete documantation available at [http://electrus.vvfin.in](http://electrus.vvfin.in).

## Support

For any help and support feel free to contact us at `embrakeproject@gmail.com` or `control@vvfin.in`

## 🧰 Feature Roadmap

| Feature                | Status      |
| ---------------------- | ----------- |
| ✅ Atomic Write Engine  | Complete    |
| ✅ Smart Insert Logic   | Complete    |
| ✅ Modular I/O Layer    | Complete    |
| 🔄 Transaction Support | In Progress |
| 🔄 Advanced Query Ops  | In Progress |
| 🧪 Middleware Engine   | In Progress |

Have ideas? [Submit a GitHub Issue](https://github.com/axiomchronicles/electrus/issues)

---

## ❤️ Sponsor Electrus

Great open-source needs great community support.

If Electrus saves you time, sanity, or money — consider sponsoring:

[![Sponsor on GitHub](https://img.shields.io/badge/Sponsor-GitHub%20Sponsors-ff69b4?style=for-the-badge\&logo=github)](https://github.com/sponsors/axiomchronicles)

> Every donation goes toward feature development, maintenance, and coffee ☕.

---

## 🔓 License

Electrus is open-source under the **BSD License** — flexible, permissive, and production-ready.

---

## 🎨 Final Thoughts

> Electrus was crafted for those who care about code elegance, data safety, and developer happiness.

<p align="center"><strong>⚡ Electrus — Build fearlessly. Code beautifully.</strong></p>
