"""
Electrus Database Layer
~~~~~~~~~~~~~~~~~~~~~~~
Production-grade re-write of the `Database` class with modern
Python 3.9+ features, rich logging, validation, and async support.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from ..exception.base import ElectrusException
from .collection import Collection

# --------------------------------------------------------------------------- #
# Typed Result Objects
# --------------------------------------------------------------------------- #

class Operation(Enum):
    CREATE = auto()
    READ = auto()
    UPDATE = auto()
    DELETE = auto()
    RENAME = auto()

@dataclass
class Result:
    ok: bool
    op: Operation
    msg: str
    data: Optional[Dict[str, Any]] = None
    err: Optional[str] = None

# --------------------------------------------------------------------------- #
# Decorators
# --------------------------------------------------------------------------- #

def validate_collection_name(func):
    """Reject empty names, names with path separators, or >64 chars."""
    def wrapper(self, name: str, *args, **kwargs):
        if not name or "/" in name or "\\" in name or len(name) > 64:
            raise ElectrusException(
                f"Invalid collection name: '{name}'. Alphanumerics, "
                "hyphens, and underscores only."
            )
        return func(self, name, *args, **kwargs)
    return wrapper

def async_wrap(sync_fn):
    """Expose sync calls as asyncio-friendly coroutines."""
    async def run_async(*args, loop: Optional[asyncio.AbstractEventLoop] = None,
                        executor=None, **kwargs):
        _loop = loop or asyncio.get_event_loop()
        return await _loop.run_in_executor(
            executor, lambda: sync_fn(*args, **kwargs)
        )
    return run_async

# --------------------------------------------------------------------------- #
# Database Class
# --------------------------------------------------------------------------- #

class Database:
    """
    Single-database façade for Electrus.

    Parameters
    ----------
    db_name : str
        Name of the database directory residing under ~/.electrus
    base_path : Path | None
        Override storage root (DI-friendly). Default: ~/.electrus
    logger : logging.Logger | None
        Custom logger; default uses module-level logger.
    """

    JSON_SUFFIX = ".json"

    # ---------- ctor ---------- #
    def __init__(
        self,
        db_name: str,
        *,
        base_path: Optional[Union[str, Path]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.db_name = db_name
        self.base_path: Path = (
            Path(base_path).expanduser().resolve()
            if base_path
            else Path.home() / ".electrus"
        )
        self.db_path: Path = self.base_path / db_name

        self.logger = (
            logger if logger
            else self._setup_default_logger()
        )

        self._ensure_db_dir()

    # ---------- logging ---------- #
    def _setup_default_logger(self) -> logging.Logger:
        log = logging.getLogger(f"Electrus.Database.{self.db_name}")
        if not log.handlers:
            fmt = logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
            )
            h = logging.StreamHandler()
            h.setFormatter(fmt)
            h.setLevel(logging.INFO)
            log.addHandler(h)
            log.setLevel(logging.DEBUG)
        return log

    # ---------- internal FS helpers ---------- #
    def _ensure_db_dir(self) -> None:
        try:
            self.db_path.mkdir(parents=True, exist_ok=True)  # pathlib mkdir[24][29]
        except Exception as exc:
            raise ElectrusException(f"Cannot create DB dir: {exc}")

    def _collection_path(self, collection: str) -> Path:
        return self.db_path / collection

    def _json_file(self, collection: str) -> Path:
        return self._collection_path(collection) / f"{collection}{self.JSON_SUFFIX}"

    # ---------- context manager ---------- #
    @contextmanager
    def _guard(self, op: Operation):
        self.logger.debug(f"⏳ {op.name} started on DB '{self.db_name}'")
        try:
            yield
            self.logger.debug(f"✅ {op.name} finished")
        except ElectrusException:
            raise
        except OSError as exc:
            self.logger.error(f"Filesystem error: {exc}", exc_info=True)
            raise ElectrusException(str(exc))
        except Exception as exc:  # noqa: BLE001
            self.logger.error(f"Unhandled error: {exc}", exc_info=True)
            raise ElectrusException(str(exc)) from exc

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #

    # -- database-level meta --
    def info(self) -> Dict[str, Any]:
        """Return metadata (size, ctime, mtime, collection count)."""
        if not self.db_path.exists():
            raise ElectrusException(f"Database '{self.db_name}' missing")
        size = sum(f.stat().st_size for f in self.db_path.rglob("*") if f.is_file())
        stat = self.db_path.stat()
        return {
            "name": self.db_name,
            "path": str(self.db_path),
            "collections": self.list_collections(),
            "created": stat.st_ctime,
            "modified": stat.st_mtime,
            "size_bytes": size,
        }

    # -- list / existence --
    def list_collections(self) -> List[str]:
        return sorted(
            p.name for p in self.db_path.iterdir() if p.is_dir()
        )

    def collection_exists(self, name: str) -> bool:
        return self._collection_path(name).is_dir()

    def __contains__(self, name: str) -> bool:
        return self.collection_exists(name)

    def __iter__(self) -> Iterator[str]:
        return iter(self.list_collections())

    def __len__(self) -> int:
        return len(self.list_collections())

    # -- CRUD for collections --
    @validate_collection_name
    def __getitem__(self, collection_name: str) -> "Collection":
        if not self.collection_exists(collection_name):
            raise ElectrusException(
                f"Collection '{collection_name}' absent in DB '{self.db_name}'"
            )
        return Collection(
            self.db_name,
            collection_name,
            db_path=self.db_path,
            logger=self.logger,
        )

    @validate_collection_name
    def create_collection(self, collection: str, *, scaffold_json: bool = True) -> Result:
        with self._guard(Operation.CREATE):
            cpath = self._collection_path(collection)
            if cpath.exists():
                return Result(
                    ok=True,
                    op=Operation.CREATE,
                    msg=f"Collection '{collection}' already exists",
                )
            cpath.mkdir(parents=True)
            if scaffold_json:
                self._json_file(collection).write_text("[]", encoding="utf-8")
            return Result(
                ok=True,
                op=Operation.CREATE,
                msg=f"Collection '{collection}' created",
                data={"path": str(cpath)},
            )

    @validate_collection_name
    def drop_collection(self, collection: str, *, force: bool = False) -> Result:
        with self._guard(Operation.DELETE):
            cpath = self._collection_path(collection)
            if not cpath.exists():
                raise ElectrusException(f"Collection '{collection}' not found")
            if not force and any(cpath.iterdir()):
                raise ElectrusException(
                    f"Collection '{collection}' is not empty; use force=True"
                )
            shutil.rmtree(cpath)  # recursive delete[22][27][37]
            return Result(
                ok=True,
                op=Operation.DELETE,
                msg=f"Dropped collection '{collection}'",
            )

    @validate_collection_name
    def rename_collection(self, old: str, new: str) -> Result:
        with self._guard(Operation.RENAME):
            old_p = self._collection_path(old)
            new_p = self._collection_path(new)
            if not old_p.exists():
                raise ElectrusException(f"Collection '{old}' missing")
            if new_p.exists():
                raise ElectrusException(f"Target '{new}' already exists")
            old_p.rename(new_p)
            # rename JSON file if present
            old_json = self._json_file(old)
            if old_json.exists():
                old_json.rename(self._json_file(new))
            return Result(
                ok=True,
                op=Operation.RENAME,
                msg=f"Renamed '{old}' ➜ '{new}'",
            )

    # ------------------------------------------------------------------ #
    # JSON helpers
    # ------------------------------------------------------------------ #
    def _read_json(self, file_path: Path) -> List[Dict[str, Any]]:
        try:
            return json.loads(file_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:  # corrupt JSON[40]
            self.logger.warning(f"Corrupt JSON at {file_path}: {exc}")
            raise ElectrusException(f"Invalid JSON file: {file_path}") from exc

    def _write_json(self, file_path: Path, data: List[Dict[str, Any]]) -> None:
        tmp = file_path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp.replace(file_path)

    # expose async wrappers (for eg. web UI)
    async_read_json = async_wrap(_read_json)
    async_write_json = async_wrap(_write_json)

    # ------------------------------------------------------------------ #
    # Representation
    # ------------------------------------------------------------------ #
    def __repr__(self) -> str:  # pragma: no cover
        return f"<Electrus.Database name='{self.db_name}' collections={len(self)}>"

