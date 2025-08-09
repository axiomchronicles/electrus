from datetime import datetime
from pathlib import Path
from typing import Union, Dict, Any, List

import json
import shutil

from . import (
    lockmanager,
    atomicoperation,
    checksumcalculator
)

class BackupManager:
    """Advanced backup system with incremental backup support"""

    def __init__(self, backup_dir: Union[str, Path]):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True, parents=True)
        self.lock_manager = lockmanager.FileLockManager()

    def create_backup(self, source_paths: List[Union[str, Path]], backup_name: str = None) -> Dict[str, Any]:
        """Create a backup of specified files/directories"""
        if backup_name is None:
            backup_name = datetime.now().strftime("backup_%Y%m%d_%H%M%S")

        backup_path = self.backup_dir / backup_name
        backup_path.mkdir(exist_ok=True)

        backup_info = {
            'name': backup_name,
            'timestamp': datetime.now().isoformat(),
            'files': [],
            'total_size': 0
        }

        for source_path in source_paths:
            source_path = Path(source_path)

            if source_path.is_file():
                self._backup_file(source_path, backup_path, backup_info)
            elif source_path.is_dir():
                self._backup_directory(source_path, backup_path, backup_info)

        # Save backup info
        info_file = backup_path / 'backup_info.json'
        with atomicoperation.AtomicFileOperations.atomic_write(info_file, 'w') as f:
            json.dump(backup_info, f, indent=2)

        return backup_info

    def _backup_file(self, source_file: Path, backup_path: Path, backup_info: Dict[str, Any]):
        """Backup a single file"""
        with self.lock_manager.acquire_lock(source_file, 'read'):
            dest_file = backup_path / source_file.name
            shutil.copy2(source_file, dest_file)

            # Calculate checksum
            checksum = checksumcalculator.ChecksumCalculator.calculate_checksum(dest_file)

            file_info = {
                'path': str(source_file),
                'size': source_file.stat().st_size,
                'checksum': checksum,
                'backup_path': str(dest_file.relative_to(self.backup_dir))
            }

            backup_info['files'].append(file_info)
            backup_info['total_size'] += file_info['size']

    def _backup_directory(self, source_dir: Path, backup_path: Path, backup_info: Dict[str, Any]):
        """Backup a directory recursively"""
        for item in source_dir.rglob('*'):
            if item.is_file():
                relative_path = item.relative_to(source_dir)
                dest_path = backup_path / source_dir.name / relative_path
                dest_path.parent.mkdir(parents=True, exist_ok=True)

                with self.lock_manager.acquire_lock(item, 'read'):
                    shutil.copy2(item, dest_path)

                    checksum = checksumcalculator.ChecksumCalculator.calculate_checksum(dest_path)

                    file_info = {
                        'path': str(item),
                        'size': item.stat().st_size,
                        'checksum': checksum,
                        'backup_path': str(dest_path.relative_to(self.backup_dir))
                    }

                    backup_info['files'].append(file_info)
                    backup_info['total_size'] += file_info['size']