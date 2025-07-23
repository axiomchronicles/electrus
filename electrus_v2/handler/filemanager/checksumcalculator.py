import hashlib
import aiofiles

from pathlib import Path
from typing import Union

class ChecksumCalculator:
    """Utility class for calculating file checksums"""

    SUPPORTED_ALGORITHMS = ['md5', 'sha1', 'sha256', 'sha512', 'blake2b']

    @staticmethod
    def calculate_checksum(file_path: Union[str, Path], algorithm: str = 'sha256', chunk_size: int = 8192) -> str:
        """Calculate file checksum using specified algorithm"""
        if algorithm not in ChecksumCalculator.SUPPORTED_ALGORITHMS:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        hash_obj = hashlib.new(algorithm)

        with open(file_path, 'rb') as f:
            while chunk := f.read(chunk_size):
                hash_obj.update(chunk)

        return hash_obj.hexdigest()

    @staticmethod
    async def calculate_checksum_async(file_path: Union[str, Path], algorithm: str = 'sha256', chunk_size: int = 8192) -> str:
        """Async version of checksum calculation"""
        if algorithm not in ChecksumCalculator.SUPPORTED_ALGORITHMS:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        hash_obj = hashlib.new(algorithm)

        async with aiofiles.open(file_path, 'rb') as f:
            while chunk := await f.read(chunk_size):
                hash_obj.update(chunk)

        return hash_obj.hexdigest()

    @staticmethod
    def verify_integrity(file_path: Union[str, Path], expected_checksum: str, algorithm: str = 'sha256') -> bool:
        """Verify file integrity against expected checksum"""
        actual_checksum = ChecksumCalculator.calculate_checksum(file_path, algorithm)
        return actual_checksum == expected_checksum