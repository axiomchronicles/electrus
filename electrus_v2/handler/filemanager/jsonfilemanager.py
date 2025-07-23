import json
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional, Union, Dict, Any
import aiofiles

from . import (
    checksumcalculator,
    atomicoperation,
    lockmanager,
    versionmanager
)


class JsonFileHandler:
    def __init__(
        self,
        base_path: Union[str, Path],
        version_manager: versionmanager.FileVersionManager = None,
        lock_manager: lockmanager.FileLockManager = None,
        checksum_algo: str = "sha256",
    ) -> None:
        self.base_path: Path = Path(base_path).expanduser().resolve()
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.lock_manager = lock_manager or lockmanager.FileLockManager()
        self.version_manager = version_manager or versionmanager.FileVersionManager(self.base_path)
        self.checksum_algo = checksum_algo

    def _resolve_path(self, file: Union[str, Path]) -> Path:
        return self.base_path / Path(file)

    def read(
        self,
        file: Union[str, Path],
        verify_integrity: bool = False,
        expected_checksum: Optional[str] = None,
    ) -> Dict[str, Any]:
        path = self._resolve_path(file)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        with self.lock_manager.acquire_lock(path, lock_type="read"):
            raw = path.read_text(encoding="utf-8")
            data = json.loads(raw)

        checksum = checksumcalculator.ChecksumCalculator.calculate_checksum(
            path, self.checksum_algo
        )
        if verify_integrity:
            if expected_checksum is None:
                raise ValueError("Expected checksum must be provided for integrity verification")
            if not checksumcalculator.ChecksumCalculator.verify_integrity(
                path, expected_checksum, self.checksum_algo
            ):
                raise RuntimeError("Integrity check failed after read")

        return {
            "data": data,
            "checksum": checksum,
            "timestamp": datetime.now().isoformat()
        }

    def write(
        self,
        file: Union[str, Path],
        data: Dict[str, Any],
        create_version: bool = True,
        verify_integrity: bool = True,
    ) -> Dict[str, Any]:
        path = self._resolve_path(file)
        path.parent.mkdir(parents=True, exist_ok=True)

        raw = json.dumps(data, indent=2)

        if create_version and path.exists() and self.version_manager:
            self.version_manager.create_version(path, self.checksum_algo)

        with self.lock_manager.acquire_lock(path, lock_type="write"):
            with atomicoperation.AtomicFileOperations.atomic_write(
                path, mode="w", encoding="utf-8"
            ) as f:
                f.write(raw)

        checksum = checksumcalculator.ChecksumCalculator.calculate_checksum(
            path, self.checksum_algo
        )
        if verify_integrity and not checksumcalculator.ChecksumCalculator.verify_integrity(
            path, checksum, self.checksum_algo
        ):
            raise RuntimeError("Integrity check failed after write")

        return {
            "file_path": str(path),
            "checksum": checksum,
            "size": path.stat().st_size,
            "timestamp": datetime.now().isoformat(),
            "versioned": create_version,
        }

    async def read_async(
        self,
        file: Union[str, Path],
        verify_integrity: bool = False,
        expected_checksum: Optional[str] = None,
    ) -> Dict[str, Any]:
        path = self._resolve_path(file)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        loop = asyncio.get_running_loop()
        lock = self.lock_manager.acquire_lock(path, lock_type="read")
        await loop.run_in_executor(None, lock.__enter__)
        try:
            async with aiofiles.open(path, "r", encoding="utf-8") as f:
                raw = await f.read()
                data = json.loads(raw)
        finally:
            await loop.run_in_executor(None, lambda: lock.__exit__(None, None, None))

        checksum = checksumcalculator.ChecksumCalculator.calculate_checksum(
            path, self.checksum_algo
        )
        if verify_integrity:
            if expected_checksum is None:
                raise ValueError("Expected checksum must be provided for integrity verification")
            if not checksumcalculator.ChecksumCalculator.verify_integrity(
                path, expected_checksum, self.checksum_algo
            ):
                raise RuntimeError("Integrity check failed after async read")

        return {
            "data": data,
            "checksum": checksum,
            "timestamp": datetime.now().isoformat()
        }

    async def write_async(
        self,
        file: Union[str, Path],
        data: Dict[str, Any],
        create_version: bool = True,
        verify_integrity: bool = True,
    ) -> Dict[str, Any]:
        path = self._resolve_path(file)
        path.parent.mkdir(parents=True, exist_ok=True)

        raw = json.dumps(data, indent=2)
        loop = asyncio.get_running_loop()

        if create_version and path.exists() and self.version_manager:
            await loop.run_in_executor(
                None,
                lambda: self.version_manager.create_version(path, self.checksum_algo)
            )

        lock = self.lock_manager.acquire_lock(path, lock_type="write")
        await loop.run_in_executor(None, lock.__enter__)
        try:
            await atomicoperation.AtomicFileOperations.atomic_write_async(
                path, raw, mode="w"
            )
        finally:
            await loop.run_in_executor(None, lambda: lock.__exit__(None, None, None))

        checksum = await checksumcalculator.ChecksumCalculator.calculate_checksum_async(
            path, self.checksum_algo
        )
        if verify_integrity and not checksumcalculator.ChecksumCalculator.verify_integrity(
            path, checksum, self.checksum_algo
        ):
            raise RuntimeError("Integrity check failed after async write")

        return {
            "file_path": str(path),
            "checksum": checksum,
            "size": path.stat().st_size,
            "timestamp": datetime.now().isoformat(),
            "versioned": create_version,
        }
