from dataclasses import dataclass
from datetime import datetime
from contextlib import contextmanager
from filelock import FileLock
from pathlib import Path
from typing import Union, Optional, Any, Dict


import os
import threading

@dataclass 
class LockInfo:
    """File lock information"""
    pid: int
    timestamp: datetime
    lock_type: str  # 'read', 'write', 'exclusive'

    def to_dict(self) -> Dict[str, Any]:
        return {
            'pid': self.pid,
            'timestamp': self.timestamp.isoformat(), 
            'lock_type': self.lock_type
        }

class FileLockManager:
    """Advanced file locking with cross-platform support"""

    def __init__(self):
        self._locks = {}
        self._lock = threading.Lock()

    @contextmanager
    def acquire_lock(self, file_path: Union[str, Path], lock_type: str = 'exclusive', timeout: float = 10.0):
        """Context manager for file locking"""
        file_path = Path(file_path)
        lock_file_path = file_path.with_suffix(file_path.suffix + '.lock')

        lock = FileLock(str(lock_file_path), timeout=timeout)

        try:
            with lock:
                lock_info = LockInfo(
                    pid=os.getpid(),
                    timestamp=datetime.now(),
                    lock_type=lock_type
                )

                with self._lock:
                    self._locks[str(file_path)] = lock_info

                yield lock_info
        finally:
            with self._lock:
                if str(file_path) in self._locks:
                    del self._locks[str(file_path)]

    def is_locked(self, file_path: Union[str, Path]) -> bool:
        """Check if file is currently locked"""
        return str(file_path) in self._locks

    def get_lock_info(self, file_path: Union[str, Path]) -> Optional[LockInfo]:
        """Get lock information for a file"""
        return self._locks.get(str(file_path))