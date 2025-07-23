from pathlib import Path
from datetime import datetime
from typing import Union, List, Optional

import shutil
import json

from . import (
    checksumcalculator,
    metadata as filemetadata,
    atomicoperation
)

class FileVersionManager:
    """Manages file versions with metadata tracking"""

    def __init__(self, base_path: Union[str, Path]):
        self.base_path = Path(base_path)
        self.versions_dir = self.base_path / '.versions'
        self.metadata_file = self.versions_dir / 'metadata.json'
        self._ensure_directories()

    def _ensure_directories(self):
        """Ensure version directories exist"""
        self.versions_dir.mkdir(exist_ok=True, parents=True)

    def create_version(self, file_path: Union[str, Path], algorithm: str = 'sha256') -> filemetadata.FileMetadata:
        """Create a new version of the file"""
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Calculate checksum
        checksum = checksumcalculator.ChecksumCalculator.calculate_checksum(file_path, algorithm)

        # Get file stats
        stat_info = file_path.stat()

        # Get next version number
        current_versions = self.list_versions(file_path.name)
        next_version = max([v.version for v in current_versions], default=0) + 1

        # Create metadata
        metadata = filemetadata.FileMetadata(
            filename=file_path.name,
            size=stat_info.st_size,
            checksum=checksum,
            checksum_algorithm=algorithm,
            created_at=datetime.fromtimestamp(stat_info.st_ctime),
            modified_at=datetime.fromtimestamp(stat_info.st_mtime),
            version=next_version,
            permissions=oct(stat_info.st_mode)[-3:]
        )

        # Copy file to versions directory
        version_filename = f"{file_path.stem}_v{next_version}{file_path.suffix}"
        version_path = self.versions_dir / version_filename
        shutil.copy2(file_path, version_path)

        # Update metadata
        self._save_metadata(metadata)

        return metadata

    def list_versions(self, filename: str) -> List[filemetadata.FileMetadata]:
        """List all versions of a file"""
        all_metadata = self._load_metadata()
        return [meta for meta in all_metadata if meta.filename == filename]

    def get_version(self, filename: str, version: int) -> Optional[Path]:
        """Get path to specific version of a file"""
        file_stem = Path(filename).stem
        file_suffix = Path(filename).suffix
        version_filename = f"{file_stem}_v{version}{file_suffix}"
        version_path = self.versions_dir / version_filename

        return version_path if version_path.exists() else None

    def restore_version(self, filename: str, version: int, target_path: Union[str, Path]):
        """Restore a specific version to target location"""
        version_path = self.get_version(filename, version)
        if not version_path:
            raise FileNotFoundError(f"Version {version} of {filename} not found")

        shutil.copy2(version_path, target_path)

    def _save_metadata(self, metadata: filemetadata.FileMetadata):
        """Save metadata to file"""
        all_metadata = self._load_metadata()

        # Remove existing metadata for same file and version
        all_metadata = [
            meta for meta in all_metadata 
            if not (meta.filename == metadata.filename and meta.version == metadata.version)
        ]

        all_metadata.append(metadata)

        # Save to file
        with atomicoperation.AtomicFileOperations.atomic_write(self.metadata_file, 'w') as f:
            json.dump([meta.to_dict() for meta in all_metadata], f, indent=2)

    def _load_metadata(self) -> List[filemetadata.FileMetadata]:
        """Load metadata from file"""
        if not self.metadata_file.exists():
            return []

        try:
            with open(self.metadata_file, 'r') as f:
                data = json.load(f)
                return [filemetadata.FileMetadata.from_dict(item) for item in data]
        except (json.JSONDecodeError, KeyError):
            return []