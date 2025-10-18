"""
Block storage system with tier-based compression for cof.
Implements hot/warm/cold storage tiers with zstd compression.
"""

import os
import json
import zstandard as zstd
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from blake3 import blake3

from cof.models import Block, StorageTier, BlockMap


class BlockStorage:
    """Manages block storage across different tiers with compression."""

    def __init__(self, cof_dir: str, config: Dict):
        self.cof_dir = Path(cof_dir)
        self.config = config
        
        # Create tier directories
        self.hot_dir = self.cof_dir / "objects" / "hot"
        self.warm_dir = self.cof_dir / "objects" / "warm"
        self.cold_dir = self.cof_dir / "objects" / "cold"
        
        for tier_dir in [self.hot_dir, self.warm_dir, self.cold_dir]:
            tier_dir.mkdir(parents=True, exist_ok=True)
        
        # Compression settings
        self.warm_level = config["compression"]["warm_level"]
        self.cold_level = config["compression"]["cold_level"]
        self.warm_threshold = config["compression"]["warm_threshold"]
        self.cold_threshold = config["compression"]["cold_threshold"]
        
        # Block size
        self.block_size = config["core"]["block_size"]
        
        # Load block map
        self.block_map = self._load_block_map()

    def _load_block_map(self) -> BlockMap:
        """Load the block mapping from disk."""
        block_map_path = self.cof_dir / "index" / "block_map.json"
        if block_map_path.exists():
            with open(block_map_path, "r") as f:
                data = json.load(f)
                return BlockMap.from_dict(data)
        return BlockMap()

    def _save_block_map(self) -> None:
        """Save the block mapping to disk."""
        block_map_path = self.cof_dir / "index" / "block_map.json"
        block_map_path.parent.mkdir(parents=True, exist_ok=True)
        with open(block_map_path, "w") as f:
            json.dump(self.block_map.to_dict(), f, indent=2)

    def _get_block_path(self, block_hash: str, tier: StorageTier) -> Path:
        """Get the file path for a block in a specific tier."""
        tier_dir = {
            StorageTier.HOT: self.hot_dir,
            StorageTier.WARM: self.warm_dir,
            StorageTier.COLD: self.cold_dir
        }[tier]
        return tier_dir / block_hash

    def _compress_data(self, data: bytes, level: int) -> bytes:
        """Compress data using zstd."""
        compressor = zstd.ZstdCompressor(level=level)
        return compressor.compress(data)

    def _decompress_data(self, compressed_data: bytes) -> bytes:
        """Decompress zstd data."""
        decompressor = zstd.ZstdDecompressor()
        return decompressor.decompress(compressed_data)

    def store_block(self, data: bytes, commit_seq: int) -> str:
        """Store a block and return its hash."""
        # Calculate hash
        block_hash = blake3(data).hexdigest()
        
        # Check if block already exists
        if self.block_map.is_referenced(block_hash):
            self.block_map.add_block(block_hash, 
                                   self.block_map.get_tier(block_hash) or StorageTier.HOT, 
                                   commit_seq)
            self._save_block_map()
            return block_hash
        
        # Store in HOT tier initially
        block_path = self._get_block_path(block_hash, StorageTier.HOT)
        with open(block_path, "wb") as f:
            f.write(data)
        
        # Update block map
        self.block_map.add_block(block_hash, StorageTier.HOT, commit_seq)
        self._save_block_map()
        
        return block_hash

    def retrieve_block(self, block_hash: str) -> Optional[bytes]:
        """Retrieve a block by its hash."""
        tier = self.block_map.get_tier(block_hash)
        if not tier:
            return None
        
        block_path = self._get_block_path(block_hash, tier)
        if not block_path.exists():
            return None
        
        with open(block_path, "rb") as f:
            data = f.read()
        
        # Decompress if not in HOT tier
        if tier != StorageTier.HOT:
            data = self._decompress_data(data)
        
        return data

    def migrate_blocks(self, current_commit_seq: int) -> None:
        """Migrate blocks between tiers based on age."""
        migrated_blocks = []
        
        for block_hash, commit_age in self.block_map.hash_to_commit.items():
            current_tier = self.block_map.get_tier(block_hash)
            if not current_tier:
                continue
            
            age = current_commit_seq - commit_age
            
            # Determine target tier
            if age >= self.cold_threshold and current_tier != StorageTier.COLD:
                target_tier = StorageTier.COLD
                compression_level = self.cold_level
            elif age >= self.warm_threshold and current_tier == StorageTier.HOT:
                target_tier = StorageTier.WARM
                compression_level = self.warm_level
            else:
                continue
            
            # Perform migration
            source_path = self._get_block_path(block_hash, current_tier)
            target_path = self._get_block_path(block_hash, target_tier)
            
            if source_path.exists():
                # Read and compress data
                with open(source_path, "rb") as f:
                    data = f.read()
                
                compressed_data = self._compress_data(data, compression_level)
                
                # Write to target tier
                with open(target_path, "wb") as f:
                    f.write(compressed_data)
                
                # Remove from source tier if different
                if current_tier != target_tier:
                    source_path.unlink()
                
                # Update block map
                self.block_map.hash_to_location[block_hash] = target_tier.value
                migrated_blocks.append((block_hash, current_tier.value, target_tier.value))
        
        if migrated_blocks:
            self._save_block_map()
            print(f"Migrated {len(migrated_blocks)} blocks between tiers")

    def get_deduplication_stats(self) -> Dict:
        """Get deduplication and storage statistics."""
        stats = {
            "total_blocks": len(self.block_map.hash_to_refcount),
            "unique_blocks": len(self.block_map.hash_to_location),
            "tier_distribution": {
                "hot": 0,
                "warm": 0,
                "cold": 0
            },
            "total_size_raw": 0,
            "total_size_compressed": 0,
            "deduplication_ratio": 0.0
        }
        
        total_raw_size = 0
        total_compressed_size = 0
        
        for block_hash, tier_str in self.block_map.hash_to_location.items():
            tier = StorageTier(tier_str)
            stats["tier_distribution"][tier.value] += 1
            
            # Get block size
            block_path = self._get_block_path(block_hash, tier)
            if block_path.exists():
                compressed_size = block_path.stat().st_size
                total_compressed_size += compressed_size
                
                # Get raw size (decompress if needed)
                if tier == StorageTier.HOT:
                    raw_size = compressed_size
                else:
                    with open(block_path, "rb") as f:
                        compressed_data = f.read()
                    raw_size = len(self._decompress_data(compressed_data))
                
                total_raw_size += raw_size
        
        stats["total_size_raw"] = total_raw_size
        stats["total_size_compressed"] = total_compressed_size
        
        if total_raw_size > 0:
            stats["deduplication_ratio"] = total_raw_size / total_compressed_size
        
        # Calculate reference statistics
        total_refs = sum(self.block_map.hash_to_refcount.values())
        if stats["unique_blocks"] > 0:
            stats["avg_references"] = total_refs / stats["unique_blocks"]
        else:
            stats["avg_references"] = 0
        
        return stats

    def garbage_collect(self, current_commit_seq: int) -> None:
        """Remove unreferenced blocks older than the threshold."""
        unreachable_days = self.config["gc"]["unreachable_days"]
        unreachable_seconds = unreachable_days * 24 * 3600
        
        # Get current time for age calculation (simplified - using commit sequence)
        # In a real implementation, you'd track actual timestamps
        removed_blocks = []
        
        for block_hash in list(self.block_map.hash_to_location.keys()):
            if not self.block_map.is_referenced(block_hash):
                # Check if block is old enough to remove
                commit_age = self.block_map.get_commit_age(block_hash)
                if commit_age and (current_commit_seq - commit_age) > 100:  # Simplified age check
                    tier = self.block_map.get_tier(block_hash)
                    if tier:
                        block_path = self._get_block_path(block_hash, tier)
                        if block_path.exists():
                            block_path.unlink()
                        
                        # Remove from block map
                        del self.block_map.hash_to_location[block_hash]
                        del self.block_map.hash_to_commit[block_hash]
                        removed_blocks.append(block_hash)
        
        if removed_blocks:
            self._save_block_map()
            print(f"Garbage collected {len(removed_blocks)} unreferenced blocks")

    def process_file_blocks(self, file_path: str, commit_seq: int) -> List[str]:
        """Process a file into blocks and return block hashes."""
        block_hashes = []
        
        with open(file_path, "rb") as f:
            while True:
                block_data = f.read(self.block_size)
                if not block_data:
                    break
                
                block_hash = self.store_block(block_data, commit_seq)
                block_hashes.append(block_hash)
        
        return block_hashes

    def reconstruct_file(self, block_hashes: List[str]) -> bytes:
        """Reconstruct a file from its block hashes."""
        file_data = b""
        
        for block_hash in block_hashes:
            block_data = self.retrieve_block(block_hash)
            if block_data is None:
                raise ValueError(f"Block {block_hash} not found")
            file_data += block_data
        
        return file_data