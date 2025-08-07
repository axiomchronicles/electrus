<!-- Centered Logo -->

<p align="center">
  <img src="./assets/electrus.png" alt="Electrus Logo" width="220" />
</p>

<p align="center"><strong>A supercharged NoSQL database — modular, embeddable, and ridiculously easy to love.</strong></p>

<p align="center">
  <a href="getting-started.md">📘 Documentation</a> •
  <a href="https://github.com/axiomchronicles/electrus">💻 Source Code</a> •
  <a href="#-sponsor-electrus">❤️ Sponsor</a>
</p>

<hr/>

## 🧠 What is Electrus?

Think **MongoDB**, but lightweight. Think **in-memory**, but persistent. Think **simple**, but powerful.

**Electrus** is a Python-powered NoSQL database that behaves like a modular, embeddable data engine — clean Mongo-like syntax, no server fuss, and fully in your control.

No bulky installs. No external dependencies. Just pure Python and peace of mind.

---

## ✨ Why Developers ❤️ Electrus

✅ Intuitive, Mongo-style syntax
✅ Built-in safety: checksums, atomic writes, file locks
✅ Smart insert logic with ID generation, timestamps, and deduplication
✅ Zero-config — runs locally, anywhere Python runs
✅ Friendly for hacking, prototyping, and production alike

---

## ⚙️ Core Features

| Feature                  | Why It Rocks                                                |
| ------------------------ | ----------------------------------------------------------- |
| 🧠 Mongo-style API       | `client["MyDB"]["MyCollection"].insertOne({...})`          |
| 💾 JSON File Persistence | Saves in-memory data with atomic, async-safe JSON writes    |
| 🔐 Safe & Secure         | File locking, versioned backups, integrity verification     |
| 🧩 Modular Everything    | Swap backends, plug middleware, extend operations           |
| 🧬 InsertData Engine     | Adds metadata, handles duplicates, manages field integrity  |
| 🧪 Full Unit Test Suite  | Comes with robust coverage — file ops, logic, insert safety |
| 🛠️ Dev-First Design     | Designed for flexibility and extensibility by real devs     |

---

## 🚀 Install in One Command

```bash
pip install electrus
```

You're all set. Go build something amazing. 💻

---

## 👩‍💻 Quickstart Example

```python
from electrus import Electrus

client = Electrus()
db = client["ProjectLuna"]
collection = db["AI"]

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

Fast, clean, and safe — just like it should be.

---

## 🧰 Architecture Overview

Electrus is structured for sanity and extensibility:

* **ElectrusClient** → Initializes and manages databases
* **Database** → Organizes collections
* **Collection** → Data playground: insert, find, delete
* **InsertData** → Handles auto-fields, deduplication, and timestamps
* **FileHandler** → Low-level atomic and async-safe I/O with locking and backups
* **TransactionEngine** *(coming soon)* → Multi-op atomic transaction support
* **QueryEngine** *(coming soon)* → Advanced filtering with operators like `$in`, `$regex`

---

## 🎯 Ideal Use Cases

* Offline desktop tools with structured local data
* AI and ML experiment logging
* Prototyping without MongoDB overhead
* Game state saving systems
* Embedded device data storage
* Local-first or hybrid apps

Electrus fits where SQLite or MongoDB might feel like overkill.

---

## 🔌 Built to Be Extended

Electrus embraces developer creativity. Extend it freely:

* 🧱 Swap file handlers (YAML, SQLite, even Redis?)
* 🔐 Add encryption, validation, or compression layers
* 📊 Hook into lifecycle events with custom middleware
* 🧠 Build custom query filters or join logic

It’s your engine — mod it however you like.

---

## 🧪 Testing & Reliability

Electrus ships with a robust unit test suite. Run with:

```bash
pytest tests/
```

Test coverage includes:

* Atomic file handling and corruption recovery
* Insert field logic and auto-ID generation
* Duplicate detection and overwrite protection
* Async I/O correctness

Break stuff — safely.

---

## 📦 Dependencies

Slim and efficient. Electrus keeps its core light:

* `aiofiles` – Async file I/O
* `filelock` – Safe cross-platform locking
* `hashlib`, `datetime`, `pathlib`, `os` – Python standard library

No bloat. Just solid internals.

---

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
