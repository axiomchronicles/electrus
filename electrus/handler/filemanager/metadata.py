from dataclasses import dataclass, asdict

from datetime import datetime
from typing import Dict, Any

@dataclass
class FileMetadata:
    """File metadata structure"""
    filename: str
    size: int
    checksum: str
    checksum_algorithm: str
    created_at: datetime
    modified_at: datetime
    version: int
    permissions: str

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['modified_at'] = self.modified_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FileMetadata':
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['modified_at'] = datetime.fromisoformat(data['modified_at'])
        return cls(**data)