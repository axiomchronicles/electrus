import os
import json
import hashlib
import asyncio
from typing import Any, List, Dict
from aiofiles import open as aio_open
from filelock import FileLock, Timeout
from ..exception.base import ElectrusException

# Restrict all files under this directory (prevent traversal)
_BASE_DIR = "/var/lib/electrus/data"

def _ensure_within_base(path: str) -> str:
    abs_path = os.path.abspath(path)
    if not abs_path.startswith(_BASE_DIR + os.sep):
        raise ElectrusException(f"Path {path} is outside allowed base directory")
    return abs_path

class FileHandler:
    def __init__(self, path: str, lock_timeout: float = 5.0) -> None:
        # Validate and normalize path
        self.path = _ensure_within_base(path)
        self.dir = os.path.dirname(self.path)
        os.makedirs(self.dir, exist_ok=True)

        self.temp_path = self.path + ".tmp"
        self.checksum_path = self.path + ".checksum"
        self.checksum_tmp = self.checksum_path + ".tmp"
        self.lock_path = self.path + ".lock"
        self.lock_timeout = lock_timeout

    async def read(self) -> List[Dict[str, Any]]:
        """Read JSON data after integrity check and recover if needed."""
        await self._recover_temp()
        await self._verify_integrity()
        try:
            async with aio_open(self.path, "r") as f:
                content = await f.read()
            return json.loads(content) if content else []
        except json.JSONDecodeError as e:
            raise ElectrusException(f"Invalid JSON: {e}")
        except OSError as e:
            raise ElectrusException(f"I/O Error on read: {e}")

    async def write(self, data: List[Dict[str, Any]]) -> None:
        """Atomically write JSON data and checksum under file lock."""
        text = json.dumps(data, indent=4)
        checksum = hashlib.sha256(text.encode()).hexdigest()

        lock = FileLock(self.lock_path, timeout=self.lock_timeout)
        try:
            with lock:
                # Write data temp file
                async with aio_open(self.temp_path, "w") as tf:
                    await tf.write(text)
                    await tf.flush()
                    os.fsync(tf.fileno())

                # Write checksum temp file
                async with aio_open(self.checksum_tmp, "w") as cf:
                    await cf.write(checksum)
                    await cf.flush()
                    os.fsync(cf.fileno())

                # Atomically replace files
                os.replace(self.temp_path, self.path)
                os.replace(self.checksum_tmp, self.checksum_path)

        except Timeout:
            raise ElectrusException("Timeout acquiring file lock")
        except OSError as e:
            raise ElectrusException(f"I/O Error on write: {e}")

    async def validate_checksum(self) -> str:
        """Compute and return SHA-256 of current data file."""
        try:
            h = hashlib.sha256()
            async with aio_open(self.path, "rb") as f:
                while chunk := await f.read(8192):
                    h.update(chunk)
            return h.hexdigest()
        except OSError as e:
            raise ElectrusException(f"Checksum failed: {e}")

    def exists(self) -> bool:
        return os.path.exists(self.path)

    async def _recover_temp(self) -> None:
        """If .tmp remains, recover it to main file (last-resort)."""
        if os.path.exists(self.temp_path):
            try:
                async with aio_open(self.temp_path, "rb") as tf:
                    data = await tf.read()
                async with aio_open(self.path, "wb") as f:
                    await f.write(data)
                os.remove(self.temp_path)
            except OSError as e:
                raise ElectrusException(f"Recovery failed: {e}")

    async def _verify_integrity(self) -> None:
        """Check data file SHA-256 matches stored checksum."""
        if not os.path.exists(self.path):
            return
        if not os.path.exists(self.checksum_path):
            raise ElectrusException("Missing checksum file")

        # Compute actual
        h = hashlib.sha256()
        async with aio_open(self.path, "rb") as f:
            while chunk := await f.read(8192):
                h.update(chunk)
        actual = h.hexdigest()

        # Read expected
        async with aio_open(self.checksum_path, "r") as cf:
            expected = (await cf.read()).strip()

        if actual != expected:
            raise ElectrusException("Integrity check failed: data corrupted")
