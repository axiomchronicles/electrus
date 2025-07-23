<!-- Centered Logo -->
<p align="center">
  <img src="assets/logo.png" alt="Electrus Logo" width="220" />
</p>

<h1 align="center">⚡ Electrus</h1>
<p align="center"><strong>A supercharged NoSQL database — modular, embeddable, and ridiculously easy to love.</strong></p>

<p align="center">
  <a href="getting-started.md">📘 Documentation</a> •
  <a href="https://github.com/yourusername/electrus">💻 Source Code</a> •
  <a href="#-sponsor-electrus">❤️ Sponsor</a>
</p>

---

## 🧠 What is Electrus?

Think **MongoDB**, but lightweight. Think **in-memory**, but persistent. Think **simple**, but powerful.

**Electrus** is a Python-powered NoSQL database that feels like using a Python dictionary but behaves like a miniature, modular, file-safe database engine — with a clean Mongo-like syntax and zero server fuss.

It’s designed to give developers control, speed, and sanity — without the boilerplate of big DBs or the flimsiness of raw file storage.

---

## ✨ Why You'll Love It

✅ Intuitive syntax  
✅ Atomic file writes with checksums  
✅ Insert logic that feels... intelligent  
✅ No server needed — runs anywhere Python does  
✅ Built with real humans in mind (you, not machines)

---

## ⚙️ Core Features

| Feature                  | Why It Rocks                                               |
|--------------------------|------------------------------------------------------------|
| 🧠 Mongo-style API        | `client["MyDB"]["MyCollection"].insert_one({...})`         |
| 💾 JSON File Persistence | All in-memory data is safely saved with atomic writes      |
| 🔐 Safe & Secure         | File locking, backup versions, checksums, overwrite control|
| 🧩 Modular Everything     | Want to swap a file handler? Write a plugin? Go wild       |
| 🧬 InsertData Engine      | Automatically adds timestamps, prevents dupes, manages IDs |
| 🧪 Full Unit Test Suite   | Built-in tests so you don’t have to guess what's broken    |
| 🛠️ Developer-First Design| Built by devs, for devs, with ❤️ and clean architecture    |

---

## 🚀 Install Me Already!

```bash
pip install electrus
```

Now go build something awesome. 💻

---

## 👩‍💻 First Taste (Quickstart)

```python
from electrus import Electrus

client = Electrus()
db = client["ProjectLuna"]
collection = db["AI"]

collection.insert_one({
    "name": "Luna",
    "type": "AI Assistant",
    "version": "1.0"
})

print(collection.find_one({"name": "Luna"}))
```

---

## 🧰 How It Works (Under the Hood)

Electrus is built like a layered cake 🍰 — sweet, structured, and easily cut into.

- **ElectrusClient**: Your gateway to the database universe
- **Database**: A home for multiple collections
- **Collection**: Insert, read, delete — your data playground
- **InsertData**: Adds logic to data inserts like IDs and timestamps
- **FileHandler**: Handles atomic JSON reads/writes with lock + backup support
- **TransactionEngine** *(coming soon)*: ACID-inspired batch writes
- **QueryEngine** *(next up!)*: Advanced queries like `$in`, `$regex`, etc.

---

## 🧠 Use Cases

- Local apps that need lightweight storage
- Offline-first tools and dashboards
- Educational or experimental databases
- Research tools
- Game save systems
- Prototyping without the Mongo/MySQL overhead

Basically: if you need local data + structure + power = Electrus.

---

## 🎛️ Built for Modularity

Electrus is like a good toolkit — flexible, reliable, extendable.

- Swap out the file backend (e.g., SQLite, YAML, Redis).
- Add custom validation or encryption.
- Build your own middleware (e.g., logging, versioning, metrics).
- Add a real-time event hook system.
- Extend the query engine to support your own filters.

Don’t ask _“Can I?”_ — just do it.

---

## 🧪 Testing

We believe in **"don’t ship broken stuff."**

Run all tests with:

```bash
pytest tests/
```

Covers:

- File safety and locking
- InsertData logic
- Overwrite and duplicate detection
- Atomic disk I/O

Feel free to break things — but responsibly 😇

---

## 🧾 Dependencies

Minimal and lean — Electrus won’t slow your projects down.

- `aiofiles` – Async file I/O
- `filelock` – Reliable cross-platform locks
- `hashlib`, `datetime`, `pathlib` – The usual Pythonic power tools

> No huge dependency trees. No bloat. Just core libraries and good structure.

---

## 🧰 Toolkit & Internals

- ✅ Atomic Write Engine
- ✅ Custom Insert Logic
- ✅ Modular File Handler
- ✅ Transaction-ish Insert Layer
- ✅ Async-Safe JSON Ops
- 🚧 Middleware Engine *(in progress)*
- 🚧 Query Filter System *(coming soon)*

---

## ❤️ Sponsor Electrus

Open-source takes love, coffee, and community.

If Electrus made your life easier or your code more joyful — consider sponsoring the project!

[![Sponsor on GitHub](https://img.shields.io/badge/Sponsor-GitHub%20Sponsors-ff69b4?style=for-the-badge&logo=github)](https://github.com/sponsors/yourusername)

---

## 🔓 License

Electrus is released under the **BSD License** — open, flexible, and forever free.

---

## 🎨 Last Words

> Designed & crafted with ❤️ for developers who care about clean code and better tools.

---

<p align="center"><strong>⚡ Electrus — Build fearlessly. Code beautifully.</strong></p>