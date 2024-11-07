# Electrus Database

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

import electrus.asynchronous as electrus
from electrus.exception import ElectrusException

client = electrus.Electrus()
database = client['mydb']
collection = database['mycollection']

async def main():
  data = {
    "id": "auto_inc",
    "name": "Embrake | Electrus",
    "email": ["embrakeproject@gmail.com", "control@vvfin.in"],
    "role": "user"
  }

  try:
    query = await collection.insert_one(data)
    if query:
      print("Data inserted successfully!")
  except ElectrusException as e:
    print("Something went wrong {}".format(e))

if __name__ == "__main__":
  asyncio.run(main())

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
