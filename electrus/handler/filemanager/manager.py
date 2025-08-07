"""
Advanced File Handling System
============================

A comprehensive file handling system with features like:
- File locking (cross-platform)
- Integrity verification with checksums
- Non-blocking async I/O operations
- File versioning with metadata tracking
- Atomic file operations
- Advanced backup system with compression
- Thread-safe operations
- Context managers for resource management

Author: Advanced File Handling System
Created: 2025
"""

import hashlib
import asyncio
import aiofiles

from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Union

from . import (
    metadata,
    checksumcalculator,
    atomicoperation, 
    backupmanager,
    lockmanager,
    versionmanager
)


class AsyncFileHandler:
    """Async file operations for non-blocking I/O"""

    @staticmethod
    async def read_file(file_path: Union[str, Path], chunk_size: int = 8192) -> str:
        """Async file reading"""
        async with aiofiles.open(file_path, 'r') as f:
            content = await f.read()
            return content

    @staticmethod
    async def write_file(file_path: Union[str, Path], content: str, mode: str = 'w'):
        """Async file writing"""
        async with aiofiles.open(file_path, mode) as f:
            await f.write(content)
            await f.flush()

    @staticmethod
    async def copy_file(source: Union[str, Path], destination: Union[str, Path], chunk_size: int = 8192):
        """Async file copying"""
        async with aiofiles.open(source, 'rb') as src:
            async with aiofiles.open(destination, 'wb') as dst:
                while chunk := await src.read(chunk_size):
                    await dst.write(chunk)
                await dst.flush() 

    @staticmethod
    async def process_files_batch(file_operations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process multiple file operations concurrently"""
        async def process_single_operation(operation):
            try:
                op_type = operation['type']
                path = operation['path']

                if op_type == 'read':
                    content = await AsyncFileHandler.read_file(path)
                    return {'path': path, 'status': 'success', 'content_length': len(content)}
                elif op_type == 'write':
                    await AsyncFileHandler.write_file(path, operation['content'])
                    return {'path': path, 'status': 'success', 'operation': 'write'}
                elif op_type == 'checksum':
                    checksum = await checksumcalculator.ChecksumCalculator.calculate_checksum_async(path)
                    return {'path': path, 'status': 'success', 'checksum': checksum}
                else:
                    return {'path': path, 'status': 'error', 'error': 'Unknown operation'}

            except Exception as e:
                return {'path': operation.get('path', 'unknown'), 'status': 'error', 'error': str(e)}

        tasks = [process_single_operation(op) for op in file_operations]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        return [result if not isinstance(result, Exception) else 
                {'status': 'error', 'error': str(result)} for result in results]

class AdvancedFileHandler:
    """Main file handling system combining all features"""

    def __init__(self, base_path: Union[str, Path] = None):
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self.lock_manager = lockmanager.FileLockManager()
        self.version_manager = versionmanager.FileVersionManager(self.base_path)
        self.backup_manager = backupmanager.BackupManager(self.base_path / '.backups')
        self.async_handler = AsyncFileHandler()

    def secure_write(self, file_path: Union[str, Path], content: Union[str, bytes], 
                    create_version: bool = True, verify_integrity: bool = True,
                    algorithm: str = 'sha256') -> Dict[str, Any]:
        """Secure file write with locking, versioning, and integrity checking"""
        file_path = Path(file_path)

        # Create version if file exists and requested
        if create_version and file_path.exists():
            version_metadata = self.version_manager.create_version(file_path, algorithm)

        # Write file atomically with lock
        with self.lock_manager.acquire_lock(file_path, 'write'):
            with atomicoperation.AtomicFileOperations.atomic_write(file_path, 'w' if isinstance(content, str) else 'wb') as f:
                f.write(content)

        # Verify integrity if requested
        if verify_integrity:
            expected_checksum = hashlib.new(algorithm, 
                                          content.encode() if isinstance(content, str) else content).hexdigest()
            actual_checksum = checksumcalculator.ChecksumCalculator.calculate_checksum(file_path, algorithm)

            if expected_checksum != actual_checksum:
                raise RuntimeError("File integrity check failed after write")

        return {
            'file_path': str(file_path),
            'size': file_path.stat().st_size,
            'checksum': checksumcalculator.ChecksumCalculator.calculate_checksum(file_path, algorithm),
            'version_created': create_version and file_path.exists(),
            'timestamp': datetime.now().isoformat()
        }

    def secure_read(self, file_path: Union[str, Path], verify_integrity: bool = False,
                   expected_checksum: str = None, algorithm: str = 'sha256') -> Dict[str, Any]:
        """Secure file read with locking and optional integrity verification"""
        file_path = Path(file_path)

        with self.lock_manager.acquire_lock(file_path, 'read'):
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            # Read content
            with open(file_path, 'r') as f:
                content = f.read()

            # Verify integrity if requested
            if verify_integrity and expected_checksum:
                if not checksumcalculator.ChecksumCalculator.verify_integrity(file_path, expected_checksum, algorithm):
                    raise RuntimeError("File integrity verification failed")

            return {
                'content': content,
                'size': len(content),
                'checksum': checksumcalculator.ChecksumCalculator.calculate_checksum(file_path, algorithm),
                'timestamp': datetime.now().isoformat()
            }

    async def async_batch_operations(self, operations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Perform multiple file operations asynchronously"""
        return await self.async_handler.process_files_batch(operations)

    def create_backup(self, file_paths: List[Union[str, Path]], backup_name: str = None) -> Dict[str, Any]:
        """Create backup of specified files"""
        return self.backup_manager.create_backup(file_paths, backup_name)

    def list_file_versions(self, filename: str) -> List[metadata.FileMetadata]:
        """List all versions of a file"""
        return self.version_manager.list_versions(filename)

    def restore_file_version(self, filename: str, version: int, target_path: Union[str, Path]):
        """Restore a specific version of a file"""
        self.version_manager.restore_version(filename, version, target_path)
