from pathlib import Path
from typing import Union
from contextlib import contextmanager

import tempfile
import aiofiles
import os

class AtomicFileOperations:
    """Provides atomic file operations to prevent corruption"""

    @staticmethod
    @contextmanager
    def atomic_write(file_path: Union[str, Path], mode: str = 'w', **kwargs):
        """Context manager for atomic file writes"""
        file_path = Path(file_path)
        temp_file = None

        try:
            # Create temporary file in the same directory
            temp_file = tempfile.NamedTemporaryFile(
                mode=mode,
                dir=file_path.parent,
                prefix=f".{file_path.name}.tmp.",
                delete=False,
                **kwargs
            )

            yield temp_file

            # Ensure all data is written to disk
            temp_file.flush()
            os.fsync(temp_file.fileno())
            temp_file.close()

            # Atomic move (rename) to final destination
            if os.name == 'nt':  # Windows
                if file_path.exists():
                    file_path.unlink()
                os.rename(temp_file.name, file_path)
            else:  # Unix-like systems
                os.rename(temp_file.name, file_path)

        except Exception as e:
            # Clean up temporary file on error
            if temp_file and not temp_file.closed:
                temp_file.close()
            if temp_file and os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
            raise e

    @staticmethod
    async def atomic_write_async(file_path: Union[str, Path], content: Union[str, bytes], mode: str = 'w'):
        """Async atomic file write"""
        file_path = Path(file_path)
        temp_path = file_path.with_suffix(file_path.suffix + '.tmp')

        try:
            async with aiofiles.open(temp_path, mode) as f:
                await f.write(content)
                await f.flush()

            # Atomic move
            if os.name == 'nt':  # Windows
                if file_path.exists():
                    file_path.unlink()
            os.rename(temp_path, file_path)

        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            raise e