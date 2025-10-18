"""
Core data models for cof version control system.
Implements the data structures specified in the README.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, List, Any
import os
import json
import time


class StorageTier(Enum):
    """Storage tier classification for blocks."""
    HOT = "hot"
    WARM = "warm"
    COLD = "cold"


@dataclass
class Block:
    """Represents a 4KB data block with deduplication metadata."""
    hash: bytes              # BLAKE3 해시 (32바이트)
    data: bytes              # 실제 데이터 (최대 4KB)
    tier: StorageTier        # HOT, WARM, COLD
    created_commit: int      # 생성된 커밋 번호
    ref_count: int = 1       # 참조 카운트

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "hash": self.hash.hex(),
            "tier": self.tier.value,
            "created_commit": self.created_commit,
            "ref_count": self.ref_count,
            "size": len(self.data)
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], block_data: bytes) -> "Block":
        """Create from dictionary and block data."""
        return cls(
            hash=bytes.fromhex(data["hash"]),
            data=block_data,
            tier=StorageTier(data["tier"]),
            created_commit=data["created_commit"],
            ref_count=data["ref_count"]
        )


@dataclass
class Commit:
    """Represents a commit in the version control system."""
    id: bytes               # 커밋 해시 (32바이트)
    parent: Optional[bytes] # 부모 커밋 (옵션)
    tree_root: bytes        # 루트 트리 객체 (32바이트)
    timestamp: int          # Unix 타임스탬프
    author: str             # 작성자 정보
    message: str            # 커밋 메시지
    sequence: int           # 순차 번호 (aging용)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id.hex(),
            "parent": self.parent.hex() if self.parent else None,
            "tree_root": self.tree_root.hex(),
            "timestamp": self.timestamp,
            "author": self.author,
            "message": self.message,
            "sequence": self.sequence
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Commit":
        """Create from dictionary."""
        return cls(
            id=bytes.fromhex(data["id"]),
            parent=bytes.fromhex(data["parent"]) if data["parent"] else None,
            tree_root=bytes.fromhex(data["tree_root"]),
            timestamp=data["timestamp"],
            author=data["author"],
            message=data["message"],
            sequence=data["sequence"]
        )


@dataclass
class TreeEntry:
    """Represents an entry in a tree object (file or directory)."""
    name: str               # 파일/디렉토리 이름
    mode: int               # 권한
    hash: bytes             # 블롭/서브트리 해시 (32바이트)
    size: int               # 원본 파일 크기

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "mode": self.mode,
            "hash": self.hash.hex(),
            "size": self.size
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TreeEntry":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            mode=data["mode"],
            hash=bytes.fromhex(data["hash"]),
            size=data["size"]
        )


@dataclass
class Tree:
    """Represents a tree object containing file and directory entries."""
    entries: Dict[str, TreeEntry] = field(default_factory=dict)

    def add_entry(self, entry: TreeEntry) -> None:
        """Add an entry to the tree."""
        self.entries[entry.name] = entry

    def remove_entry(self, name: str) -> None:
        """Remove an entry from the tree."""
        if name in self.entries:
            del self.entries[name]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "entries": {name: entry.to_dict() for name, entry in self.entries.items()}
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Tree":
        """Create from dictionary."""
        tree = cls()
        for name, entry_data in data["entries"].items():
            tree.add_entry(TreeEntry.from_dict(entry_data))
        return tree


@dataclass
class StagedFile:
    """Represents a file in the staging area."""
    path: str
    block_hashes: List[str]
    size: int
    mode: int
    timestamp: int = field(default_factory=lambda: int(time.time()))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "path": self.path,
            "block_hashes": self.block_hashes,
            "size": self.size,
            "mode": self.mode,
            "timestamp": self.timestamp
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StagedFile":
        """Create from dictionary."""
        return cls(
            path=data["path"],
            block_hashes=data["block_hashes"],
            size=data["size"],
            mode=data["mode"],
            timestamp=data.get("timestamp", int(time.time()))
        )


@dataclass
class BlockMap:
    """Maps block hashes to their storage locations and metadata."""
    hash_to_location: Dict[str, str] = field(default_factory=dict)
    hash_to_commit: Dict[str, int] = field(default_factory=dict)
    hash_to_refcount: Dict[str, int] = field(default_factory=dict)

    def add_block(self, block_hash: str, tier: StorageTier, commit_seq: int) -> None:
        """Add a block to the map."""
        self.hash_to_location[block_hash] = tier.value
        self.hash_to_commit[block_hash] = commit_seq
        self.hash_to_refcount[block_hash] = self.hash_to_refcount.get(block_hash, 0) + 1

    def remove_reference(self, block_hash: str) -> None:
        """Remove a reference to a block."""
        if block_hash in self.hash_to_refcount:
            self.hash_to_refcount[block_hash] -= 1
            if self.hash_to_refcount[block_hash] <= 0:
                del self.hash_to_refcount[block_hash]

    def get_tier(self, block_hash: str) -> Optional[StorageTier]:
        """Get the storage tier for a block."""
        tier_str = self.hash_to_location.get(block_hash)
        return StorageTier(tier_str) if tier_str else None

    def get_commit_age(self, block_hash: str) -> Optional[int]:
        """Get the commit sequence when a block was created."""
        return self.hash_to_commit.get(block_hash)

    def is_referenced(self, block_hash: str) -> bool:
        """Check if a block is still referenced."""
        return self.hash_to_refcount.get(block_hash, 0) > 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "hash_to_location": self.hash_to_location,
            "hash_to_commit": self.hash_to_commit,
            "hash_to_refcount": self.hash_to_refcount
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BlockMap":
        """Create from dictionary."""
        block_map = cls()
        block_map.hash_to_location = data.get("hash_to_location", {})
        block_map.hash_to_commit = data.get("hash_to_commit", {})
        block_map.hash_to_refcount = data.get("hash_to_refcount", {})
        return block_map