"""
Main CLI interface for cof version control system.
"""

import os
import sys
import json
import time
import click
import toml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

from cof.models import Commit, Tree, TreeEntry, StagedFile, BlockMap
from cof.storage import BlockStorage
from cof.auth import AuthManager, ClientAuth, Permission, generate_ssh_keypair
from cof.remote import RemoteManager


COF_DIR = ".cof"


class CofRepository:
    """Main repository class for cof operations."""

    def __init__(self, path: str = "."):
        self.path = Path(path).resolve()
        self.cof_dir = self.path / COF_DIR
        self.config = self._load_config() if self._is_repo() else None
        self.storage = BlockStorage(str(self.cof_dir), self.config) if self._is_repo() and self.config else None
        self.auth_manager = AuthManager(self.cof_dir) if self._is_repo() else None
        self.client_auth = ClientAuth(self.cof_dir) if self._is_repo() else None
        self.remote_manager = RemoteManager(self) if self._is_repo() else None

    def _is_repo(self) -> bool:
        """Check if current directory is a cof repository."""
        return self.cof_dir.exists() and (self.cof_dir / "config.toml").exists()

    def _load_config(self) -> Optional[Dict]:
        """Load repository configuration."""
        if not self._is_repo():
            return None
        
        config_path = self.cof_dir / "config.toml"
        with open(config_path, "r") as f:
            return toml.load(f)

    def _get_current_branch(self) -> str:
        """Get the current branch name."""
        head_path = self.cof_dir / "HEAD"
        with open(head_path, "r") as f:
            content = f.read().strip()
        
        if content.startswith("ref: "):
            ref_path = content.split(" ", 1)[1]
            return ref_path.split("/")[-1]
        
        return "HEAD"

    def _get_branch_ref_path(self, branch: str) -> Path:
        """Get the ref file path for a branch."""
        return self.cof_dir / "refs" / "heads" / branch

    def _get_head_commit(self) -> Optional[str]:
        """Get the current HEAD commit hash."""
        branch = self._get_current_branch()
        if branch == "HEAD":
            return None
        
        ref_path = self._get_branch_ref_path(branch)
        if ref_path.exists():
            with open(ref_path, "r") as f:
                return f.read().strip()
        
        return None

    def _save_object(self, obj_data: Dict, obj_type: str) -> str:
        """Save an object and return its hash."""
        from blake3 import blake3
        
        content = json.dumps(obj_data, sort_keys=True).encode('utf-8')
        hash_hex = blake3(content).hexdigest()
        
        # Save in hot tier initially
        obj_dir = self.cof_dir / "objects" / "hot"
        obj_dir.mkdir(parents=True, exist_ok=True)
        
        obj_path = obj_dir / hash_hex
        if not obj_path.exists():
            with open(obj_path, "wb") as f:
                f.write(content)
        
        return hash_hex

    def _load_object(self, obj_hash: str) -> Optional[Dict]:
        """Load an object by its hash."""
        # Try hot tier first, then warm, then cold
        for tier in ["hot", "warm", "cold"]:
            obj_path = self.cof_dir / "objects" / tier / obj_hash
            if obj_path.exists():
                with open(obj_path, "rb") as f:
                    content = f.read()
                    
                    # Decompress if not in hot tier
                    if tier != "hot":
                        import zstandard as zstd
                        decompressor = zstd.ZstdDecompressor()
                        content = decompressor.decompress(content)
                    
                    try:
                        return json.loads(content.decode('utf-8'))
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        return None
        
        return None

    def _load_tree(self, tree_hash: str) -> Tree:
        """Load a tree object."""
        tree_data = self._load_object(tree_hash)
        if tree_data:
            return Tree.from_dict(tree_data)
        return Tree()

    def _get_next_commit_sequence(self) -> int:
        """Get the next commit sequence number."""
        # Simple implementation - in real system this would be more robust
        head_commit = self._get_head_commit()
        if head_commit:
            commit_data = self._load_object(head_commit)
            if commit_data:
                return commit_data.get("sequence", 0) + 1
        return 1

    def init(self) -> None:
        """Initialize a new cof repository."""
        if self._is_repo():
            click.echo(f"Reinitialized existing cof repository in {self.cof_dir}")
            return

        try:
            # Create directory structure
            self.cof_dir.mkdir(exist_ok=True)
            dirs = [
                "objects/hot",
                "objects/warm", 
                "objects/cold",
                "refs/heads",
                "index",
                "locks"
            ]
            
            for dir_path in dirs:
                (self.cof_dir / dir_path).mkdir(parents=True, exist_ok=True)

            # Create default configuration
            config = {
                "core": {
                    "block_size": 4096,
                    "hash_algorithm": "blake3",
                    "cache_size_mb": 256
                },
                "compression": {
                    "warm_threshold": 10,
                    "cold_threshold": 100,
                    "warm_level": 3,
                    "cold_level": 19
                },
                "network": {
                    "protocol": "udp",
                    "packet_size": 1400,
                    "timeout_ms": 5000,
                    "max_retries": 3
                },
                "gc": {
                    "auto_gc": True,
                    "unreachable_days": 30
                }
            }

            with open(self.cof_dir / "config.toml", "w") as f:
                toml.dump(config, f)

            # Initialize HEAD
            with open(self.cof_dir / "HEAD", "w") as f:
                f.write("ref: refs/heads/main")

            # Create main branch
            main_ref = self._get_branch_ref_path("main")
            main_ref.parent.mkdir(parents=True, exist_ok=True)

            click.echo(f"Initialized empty cof repository in {self.cof_dir}")

        except Exception as e:
            raise click.ClickException(f"Error initializing repository: {e}")

    def add_files(self, files: List[str]) -> None:
        """Add files to the staging area."""
        if not self._is_repo():
            raise click.ClickException("Not a cof repository. Run 'cof init' first.")

        if not self.storage or not self.config:
            raise click.ClickException("Storage system not initialized.")

        staging_area = self._load_staging_area()
        added_files = 0
        commit_seq = self._get_next_commit_sequence()

        for file_path in files:
            file_path = Path(file_path).resolve()
            
            if not file_path.exists():
                click.echo(f"Error: Path does not exist: {file_path}")
                continue

            try:
                # Get file stats
                stat = file_path.stat()
                relative_path = str(file_path.relative_to(self.path))
                
                # Process file into blocks
                block_hashes = self.storage.process_file_blocks(str(file_path), commit_seq)
                
                # Create staged file entry
                staged_file = StagedFile(
                    path=relative_path,
                    block_hashes=block_hashes,
                    size=stat.st_size,
                    mode=stat.st_mode & 0o777
                )
                
                staging_area[relative_path] = staged_file.to_dict()
                click.echo(f"Added '{relative_path}'")
                added_files += 1

            except Exception as e:
                click.echo(f"Error processing file {file_path}: {e}")

        if added_files > 0:
            self._save_staging_area(staging_area)
            click.echo(f"\nSuccessfully added {added_files} file(s) to the staging area.")

    def _load_staging_area(self) -> Dict:
        """Load the staging area."""
        staging_path = self.cof_dir / "index" / "staging.json"
        if staging_path.exists():
            with open(staging_path, "r") as f:
                try:
                    content = f.read()
                    if not content:
                        return {}
                    return json.loads(content)
                except json.JSONDecodeError:
                    return {}
        return {}

    def _save_staging_area(self, data: Dict) -> None:
        """Save the staging area."""
        staging_path = self.cof_dir / "index" / "staging.json"
        staging_path.parent.mkdir(parents=True, exist_ok=True)
        with open(staging_path, "w") as f:
            json.dump(data, f, indent=2)

    def commit(self, message: str) -> None:
        """Create a new commit."""
        if not self._is_repo():
            raise click.ClickException("Not a cof repository. Run 'cof init' first.")

        if not self.config:
            raise click.ClickException("Configuration not loaded.")

        staging_data = self._load_staging_area()
        if not staging_data:
            click.echo("Nothing to commit, staging area is empty.")
            return

        # Get parent commit
        parent_commit = self._get_head_commit()
        
        # Load current tree or create new one
        if parent_commit:
            parent_data = self._load_object(parent_commit)
            if parent_data:
                tree = self._load_tree(parent_data["tree_root"])
            else:
                tree = Tree()
        else:
            tree = Tree()

        # Update tree with staged files
        for file_path, file_data in staging_data.items():
            staged_file = StagedFile.from_dict(file_data)
            
            # Create a blob object for the file
            blob_data = {
                "type": "blob",
                "block_hashes": staged_file.block_hashes,
                "size": staged_file.size
            }
            blob_hash = self._save_object(blob_data, "blob")
            
            # Create tree entry
            tree_entry = TreeEntry(
                name=staged_file.path,
                mode=staged_file.mode,
                hash=bytes.fromhex(blob_hash),
                size=staged_file.size
            )
            tree.add_entry(tree_entry)

        # Save tree
        tree_hash = self._save_object(tree.to_dict(), "tree")

        # Create commit
        commit_seq = self._get_next_commit_sequence()
        commit_data = {
            "id": "",  # Will be set after hashing
            "parent": parent_commit,
            "tree_root": tree_hash,
            "timestamp": int(time.time()),
            "author": os.environ.get("USER", "user@example.com"),
            "message": message,
            "sequence": commit_seq
        }

        # Save commit and set its ID
        commit_hash = self._save_object(commit_data, "commit")
        commit_data["id"] = commit_hash
        
        # Update commit object with its ID
        for tier in ["hot", "warm", "cold"]:
            commit_path = self.cof_dir / "objects" / tier / commit_hash
            if commit_path.exists():
                with open(commit_path, "wb") as f:
                    content = json.dumps(commit_data, sort_keys=True).encode('utf-8')
                    f.write(content)
                break

        # Update branch reference
        branch = self._get_current_branch()
        if branch != "HEAD":
            ref_path = self._get_branch_ref_path(branch)
            with open(ref_path, "w") as f:
                f.write(commit_hash)

        # Clear staging area
        self._save_staging_area({})

        # Migrate blocks if needed
        if self.storage:
            self.storage.migrate_blocks(commit_seq)

        click.echo(f"[{branch} {commit_hash[:7]}] {message}")

    def log(self) -> None:
        """Show commit history."""
        if not self._is_repo():
            raise click.ClickException("Not a cof repository. Run 'cof init' first.")

        commit_hash = self._get_head_commit()
        if not commit_hash:
            click.echo("No commits yet.")
            return

        while commit_hash:
            commit_data = self._load_object(commit_hash)
            if not commit_data:
                click.echo(f"Error: Could not read commit {commit_hash}")
                break

            click.echo(f"commit {commit_hash}")
            click.echo(f"Author: {commit_data.get('author', 'Unknown')}")
            
            timestamp = commit_data.get('timestamp')
            if timestamp:
                dt = datetime.fromtimestamp(timestamp)
                click.echo(f"Date:   {dt.strftime('%a %b %d %H:%M:%S %Y')}")
            
            message = commit_data.get('message', '')
            click.echo(f"\n    {message.replace(chr(10), chr(10) + '    ')}")
            click.echo()

            commit_hash = commit_data.get("parent")

    def status(self) -> None:
        """Show working tree status."""
        if not self._is_repo():
            raise click.ClickException("Not a cof repository. Run 'cof init' first.")

        branch = self._get_current_branch()
        click.echo(f"On branch {branch}\n")

        # Get current tree
        head_commit = self._get_head_commit()
        if head_commit:
            commit_data = self._load_object(head_commit)
            if commit_data:
                tree = self._load_tree(commit_data["tree_root"])
            else:
                tree = Tree()
        else:
            tree = Tree()

        # Get staged files
        staged_data = self._load_staging_area()

        # Show changes to be committed
        click.echo("Changes to be committed:")
        click.echo("  (use 'cof reset <file>...' to unstage)")
        
        new_files = []
        modified_files = []
        
        for file_path, file_data in staged_data.items():
            staged_file = StagedFile.from_dict(file_data)
            
            # Check if file exists in current tree
            tree_entry = None
            for entry in tree.entries.values():
                if entry.name == file_path:
                    tree_entry = entry
                    break
            
            if tree_entry is None:
                new_files.append(file_path)
            else:
                # Compare hashes (simplified - should compare block hashes)
                modified_files.append(file_path)

        if not new_files and not modified_files:
            click.echo("\tnothing to commit")
        else:
            for file_path in sorted(new_files):
                click.echo(f"\tnew file:   {file_path}")
            for file_path in sorted(modified_files):
                click.echo(f"\tmodified:   {file_path}")

        click.echo()

        # Show untracked files
        click.echo("Untracked files:")
        click.echo("  (use 'cof add <file>...' to include in what will be commit)")

        tracked_files = set(tree.entries.keys()) | set(staged_data.keys())
        untracked_files = set()
        ignore_dirs = {COF_DIR, ".venv", "__pycache__", ".git", ".egg-info"}

        for root, dirs, files in os.walk(".", topdown=True):
            dirs[:] = [d for d in dirs if d not in ignore_dirs and not d.endswith('.egg-info')]
            for name in files:
                path = Path(root) / name
                relative_path = str(path)
                if relative_path not in tracked_files:
                    untracked_files.add(relative_path)

        if not untracked_files:
            click.echo("\tnothing to add to commit")
        else:
            for file_path in sorted(untracked_files):
                click.echo(f"\t{file_path}")

    def dedup_stats(self) -> None:
        """Show deduplication statistics."""
        if not self._is_repo():
            raise click.ClickException("Not a cof repository. Run 'cof init' first.")

        if not self.storage:
            raise click.ClickException("Storage system not initialized.")

        stats = self.storage.get_deduplication_stats()
        
        click.echo("Deduplication Statistics:")
        click.echo(f"  Total blocks: {stats['total_blocks']}")
        click.echo(f"  Unique blocks: {stats['unique_blocks']}")
        click.echo(f"  Average references: {stats.get('avg_references', 0):.2f}")
        click.echo()
        click.echo("Storage by tier:")
        for tier, count in stats['tier_distribution'].items():
            click.echo(f"  {tier.upper()}: {count} blocks")
        click.echo()
        click.echo(f"  Raw size: {stats['total_size_raw']:,} bytes")
        click.echo(f"  Compressed size: {stats['total_size_compressed']:,} bytes")
        if stats['deduplication_ratio'] > 0:
            click.echo(f"  Compression ratio: {stats['deduplication_ratio']:.2f}x")

    def create_branch(self, branch_name: str, start_point: Optional[str] = None) -> None:
        """Create a new branch."""
        if not self._is_repo():
            raise click.ClickException("Not a cof repository. Run 'cof init' first.")

        # Check if branch already exists
        ref_path = self._get_branch_ref_path(branch_name)
        if ref_path.exists():
            raise click.ClickException(f"Branch '{branch_name}' already exists.")

        # Determine starting commit
        if start_point:
            # Try to resolve start_point to a commit
            commit_hash = self._resolve_commit(start_point)
            if not commit_hash:
                raise click.ClickException(f"Invalid start point: {start_point}")
        else:
            commit_hash = self._get_head_commit()

        if not commit_hash:
            raise click.ClickException("Cannot create branch from empty repository.")

        # Create branch reference
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        with open(ref_path, "w") as f:
            f.write(commit_hash)

        click.echo(f"Created branch '{branch_name}' at {commit_hash[:7]}")

    def list_branches(self) -> None:
        """List all branches."""
        if not self._is_repo():
            raise click.ClickException("Not a cof repository. Run 'cof init' first.")

        refs_dir = self.cof_dir / "refs" / "heads"
        current_branch = self._get_current_branch()

        if refs_dir.exists():
            branches = sorted(refs_dir.iterdir())
            for branch_file in branches:
                branch_name = branch_file.name
                marker = "* " if branch_name == current_branch else "  "
                
                # Get commit hash
                with open(branch_file, "r") as f:
                    commit_hash = f.read().strip()
                
                click.echo(f"{marker}{branch_name} {commit_hash[:7]}")
        else:
            click.echo("No branches found.")

    def checkout_branch(self, branch_name: str) -> None:
        """Switch to a different branch."""
        if not self._is_repo():
            raise click.ClickException("Not a cof repository. Run 'cof init' first.")

        # Check if branch exists
        ref_path = self._get_branch_ref_path(branch_name)
        if not ref_path.exists():
            raise click.ClickException(f"Branch '{branch_name}' does not exist.")

        # Update HEAD
        head_path = self.cof_dir / "HEAD"
        with open(head_path, "w") as f:
            f.write(f"ref: refs/heads/{branch_name}")

        # Restore files from the branch
        self._restore_working_tree()

        click.echo(f"Switched to branch '{branch_name}'")

    def _resolve_commit(self, ref: str) -> Optional[str]:
        """Resolve a reference to a commit hash."""
        # Try branch name first
        ref_path = self._get_branch_ref_path(ref)
        if ref_path.exists():
            with open(ref_path, "r") as f:
                return f.read().strip()

        # Try direct commit hash
        if len(ref) >= 7:
            # Find commit by prefix
            for tier in ["hot", "warm", "cold"]:
                obj_dir = self.cof_dir / "objects" / tier
                if obj_dir.exists():
                    for obj_file in obj_dir.iterdir():
                        if obj_file.name.startswith(ref):
                            commit_data = self._load_object(obj_file.name)
                            if commit_data and "parent" in commit_data:  # It's a commit
                                return obj_file.name
        return None

    def _restore_working_tree(self) -> None:
        """Restore working tree to match the current branch."""
        if not self.storage:
            raise click.ClickException("Storage system not initialized.")

        head_commit = self._get_head_commit()
        if not head_commit:
            return

        commit_data = self._load_object(head_commit)
        if not commit_data:
            return

        tree = self._load_tree(commit_data["tree_root"])

        # Restore files from tree
        for entry in tree.entries.values():
            try:
                # Load blob data
                blob_data = self._load_object(entry.hash.hex())
                if not blob_data:
                    continue

                # Reconstruct file from blocks
                file_data = self.storage.reconstruct_file(blob_data["block_hashes"])
                
                # Write file
                file_path = self.path / entry.name
                file_path.parent.mkdir(parents=True, exist_ok=True)
                with open(file_path, "wb") as f:
                    f.write(file_data)

                # Set permissions
                file_path.chmod(entry.mode)

            except Exception as e:
                click.echo(f"Warning: Could not restore {entry.name}: {e}")

    def merge_branch(self, branch_name: str) -> None:
        """Merge another branch into the current branch."""
        if not self._is_repo():
            raise click.ClickException("Not a cof repository. Run 'cof init' first.")

        current_branch = self._get_current_branch()
        if current_branch == "HEAD":
            raise click.ClickException("Cannot merge in detached HEAD state.")

        # Get source branch commit
        source_ref_path = self._get_branch_ref_path(branch_name)
        if not source_ref_path.exists():
            raise click.ClickException(f"Branch '{branch_name}' does not exist.")

        with open(source_ref_path, "r") as f:
            source_commit = f.read().strip()

        # Get current branch commit
        current_commit = self._get_head_commit()
        if not current_commit:
            raise click.ClickException("Current branch has no commits.")

        # Find merge base (simplified - just use current commit as base)
        merge_base = current_commit

        # Load trees
        source_data = self._load_object(source_commit)
        current_data = self._load_object(current_commit)
        
        if not source_data or not current_data:
            raise click.ClickException("Could not load commit data.")

        source_tree = self._load_tree(source_data["tree_root"])
        current_tree = self._load_tree(current_data["tree_root"])

        # Merge trees (simplified - just add all files from source)
        merged_tree = Tree()
        
        # Add all current files
        for entry in current_tree.entries.values():
            merged_tree.add_entry(entry)

        # Add/overwrite with source files
        for entry in source_tree.entries.values():
            merged_tree.add_entry(entry)

        # Create merge commit
        tree_hash = self._save_object(merged_tree.to_dict(), "tree")
        
        commit_seq = self._get_next_commit_sequence()
        merge_commit_data = {
            "id": "",
            "parent": current_commit,
            "tree_root": tree_hash,
            "timestamp": int(time.time()),
            "author": os.environ.get("USER", "user@example.com"),
            "message": f"Merge branch '{branch_name}' into '{current_branch}'",
            "sequence": commit_seq,
            "merge_parent": source_commit
        }

        # Save merge commit
        merge_hash = self._save_object(merge_commit_data, "commit")
        merge_commit_data["id"] = merge_hash

        # Update commit object with its ID
        for tier in ["hot", "warm", "cold"]:
            commit_path = self.cof_dir / "objects" / tier / merge_hash
            if commit_path.exists():
                with open(commit_path, "wb") as f:
                    content = json.dumps(merge_commit_data, sort_keys=True).encode('utf-8')
                    f.write(content)
                break

        # Update branch reference
        ref_path = self._get_branch_ref_path(current_branch)
        with open(ref_path, "w") as f:
            f.write(merge_hash)

        # Restore working tree
        self._restore_working_tree()

        click.echo(f"Merge completed: {merge_hash[:7]}")

    def garbage_collect(self) -> None:
        """Run garbage collection on unreferenced blocks."""
        if not self._is_repo():
            raise click.ClickException("Not a cof repository. Run 'cof init' first.")

        if not self.storage:
            raise click.ClickException("Storage system not initialized.")

        commit_seq = self._get_next_commit_sequence()
        self.storage.garbage_collect(commit_seq)
        click.echo("Garbage collection completed.")

    def create_user(self, username: str, email: str, password: str) -> None:
        """Create a new user account."""
        if not self._is_repo():
            raise click.ClickException("Not a cof repository. Run 'cof init' first.")

        if not self.auth_manager:
            raise click.ClickException("Auth system not initialized.")

        self.auth_manager.create_user(username, email, password)

    def login_user(self, username: str, password: str) -> None:
        """Login user and store credentials."""
        if not self._is_repo():
            raise click.ClickException("Not a cof repository. Run 'cof init' first.")

        if not self.auth_manager or not self.client_auth:
            raise click.ClickException("Auth system not initialized.")

        user = self.auth_manager.authenticate_user(username, password)
        if not user:
            raise click.ClickException("Authentication failed.")

        # Create token for local use
        token = self.auth_manager.create_token(user)
        self.client_auth.store_token("local", token)
        
        click.echo(f"Logged in as '{username}'")

    def logout_user(self) -> None:
        """Logout current user."""
        if not self._is_repo():
            raise click.ClickException("Not a cof repository. Run 'cof init' first.")

        if not self.client_auth:
            raise click.ClickException("Auth system not initialized.")

        self.client_auth.remove_credentials("local")
        click.echo("Logged out")

    def whoami(self) -> None:
        """Show current user."""
        if not self._is_repo():
            raise click.ClickException("Not a cof repository. Run 'cof init' first.")

        if not self.auth_manager or not self.client_auth:
            raise click.ClickException("Auth system not initialized.")

        token = self.client_auth.get_token("local")
        if not token:
            click.echo("Not logged in")
            return

        user = self.auth_manager.validate_token(token)
        if user:
            click.echo(f"Logged in as: {user.username} ({user.email})")
            click.echo(f"Last login: {time.ctime(user.last_login) if user.last_login else 'Never'}")
        else:
            click.echo("Invalid session")

    def list_users(self) -> None:
        """List all users."""
        if not self._is_repo():
            raise click.ClickException("Not a cof repository. Run 'cof init' first.")

        if not self.auth_manager:
            raise click.ClickException("Auth system not initialized.")

        users = self.auth_manager.list_users()
        if not users:
            click.echo("No users found.")
            return

        click.echo("Users:")
        for user in users:
            status = "active" if user.is_active else "inactive"
            click.echo(f"  {user.username} ({user.email}) - {status}")

    def generate_ssh_keys(self, key_size: int = 2048) -> None:
        """Generate SSH key pair."""
        try:
            private_key, public_key = generate_ssh_keypair(key_size)
            
            # Save keys
            ssh_dir = self.cof_dir / "ssh"
            ssh_dir.mkdir(exist_ok=True)
            
            with open(ssh_dir / "id_rsa", "w") as f:
                f.write(private_key)
            os.chmod(ssh_dir / "id_rsa", 0o600)
            
            with open(ssh_dir / "id_rsa.pub", "w") as f:
                f.write(public_key)
            
            click.echo(f"Generated SSH keys:")
            click.echo(f"  Private: {ssh_dir / 'id_rsa'}")
            click.echo(f"  Public: {ssh_dir / 'id_rsa.pub'}")
            click.echo(f"\nPublic key:")
            click.echo(public_key)

        except Exception as e:
            raise click.ClickException(f"Failed to generate SSH keys: {e}")

    def show_config(self) -> None:
        """Show repository configuration."""
        if not self._is_repo():
            raise click.ClickException("Not a cof repository. Run 'cof init' first.")

        if not self.config:
            raise click.ClickException("Configuration not loaded.")

        click.echo("Repository Configuration:")
        for section_name, section_data in self.config.items():
            click.echo(f"\n[{section_name}]")
            for key, value in section_data.items():
                click.echo(f"  {key} = {value}")

    def set_config(self, key: str, value: str) -> None:
        """Set a configuration value."""
        if not self._is_repo():
            raise click.ClickException("Not a cof repository. Run 'cof init' first.")

        if not self.config:
            raise click.ClickException("Configuration not loaded.")

        # Parse key in format section.key
        if "." not in key:
            raise click.ClickException("Key must be in format 'section.key'")

        section, config_key = key.split(".", 1)
        
        # Convert value to appropriate type
        parsed_value = value
        if value.lower() in ("true", "false"):
            parsed_value = value.lower() == "true"
        elif value.isdigit():
            parsed_value = int(value)
        elif "." in value and value.replace(".", "").isdigit():
            parsed_value = float(value)

        # Update configuration
        if section not in self.config:
            self.config[section] = {}
        self.config[section][config_key] = parsed_value

        # Save configuration
        config_path = self.cof_dir / "config.toml"
        with open(config_path, "w") as f:
            toml.dump(self.config, f)

        click.echo(f"Set {key} = {parsed_value}")

    # Remote operations
    def add_remote(self, name: str, url: str) -> None:
        """Add a remote repository."""
        if not self._is_repo():
            raise click.ClickException("Not a cof repository. Run 'cof init' first.")
        
        if not self.remote_manager:
            raise click.ClickException("Remote manager not initialized.")
        
        self.remote_manager.add_remote(name, url)
    
    def remove_remote(self, name: str) -> None:
        """Remove a remote repository."""
        if not self._is_repo():
            raise click.ClickException("Not a cof repository. Run 'cof init' first.")
        
        if not self.remote_manager:
            raise click.ClickException("Remote manager not initialized.")
        
        self.remote_manager.remove_remote(name)
    
    def list_remotes(self) -> None:
        """List all remote repositories."""
        if not self._is_repo():
            raise click.ClickException("Not a cof repository. Run 'cof init' first.")
        
        if not self.remote_manager:
            raise click.ClickException("Remote manager not initialized.")
        
        remotes = self.remote_manager.list_remotes()
        if not remotes:
            click.echo("No remotes configured.")
            return
        
        click.echo("Remotes:")
        for name, remote in remotes.items():
            click.echo(f"  {name}\t{remote.url} ({remote.host}:{remote.port})")
    
    async def clone_repository(self, url: str, target_dir: str) -> None:
        """Clone a remote repository."""
        from cof.remote import RemoteOperations
        
        remote_ops = RemoteOperations(self)
        success = await remote_ops.clone_repository(url, target_dir)
        if not success:
            raise click.ClickException("Failed to clone repository.")
    
    async def push_to_remote(self, remote_name: str, branch: str) -> None:
        """Push to remote repository."""
        from cof.remote import RemoteOperations
        
        if not self._is_repo():
            raise click.ClickException("Not a cof repository. Run 'cof init' first.")
        
        remote_ops = RemoteOperations(self)
        success = await remote_ops.push_to_remote(remote_name, branch)
        if not success:
            raise click.ClickException("Failed to push to remote.")
    
    async def pull_from_remote(self, remote_name: str, branch: str) -> None:
        """Pull from remote repository."""
        from cof.remote import RemoteOperations
        
        if not self._is_repo():
            raise click.ClickException("Not a cof repository. Run 'cof init' first.")
        
        remote_ops = RemoteOperations(self)
        success = await remote_ops.pull_from_remote(remote_name, branch)
        if not success:
            raise click.ClickException("Failed to pull from remote.")


# CLI Commands
@click.group()
def cli():
    """Cof is a version control system optimized for binary files."""
    pass


@cli.command()
def init():
    """Initialize a new cof repository."""
    repo = CofRepository()
    repo.init()


@cli.command()
@click.argument("files", nargs=-1, type=click.Path(exists=True, dir_okay=False, resolve_path=True))
def add(files):
    """Add files to the staging area."""
    repo = CofRepository()
    repo.add_files(files)


@cli.command()
@click.option("-m", "--message", required=True, help="The commit message.")
def commit(message):
    """Create a new commit."""
    repo = CofRepository()
    repo.commit(message)


@cli.command()
def log():
    """Show commit history."""
    repo = CofRepository()
    repo.log()


@cli.command()
def status():
    """Show the working tree status."""
    repo = CofRepository()
    repo.status()


@cli.command()
def dedup_stats():
    """Show deduplication statistics."""
    repo = CofRepository()
    repo.dedup_stats()


@cli.command()
@click.argument("name", required=False)
@click.option("--start-point", help="Start point for the new branch.")
def branch(name, start_point):
    """Create or list branches."""
    repo = CofRepository()
    if name:
        repo.create_branch(name, start_point)
    else:
        repo.list_branches()


@cli.command()
@click.argument("branch_name")
def checkout(branch_name):
    """Switch to a different branch."""
    repo = CofRepository()
    repo.checkout_branch(branch_name)


@cli.command()
@click.argument("branch_name")
def merge(branch_name):
    """Merge another branch into the current branch."""
    repo = CofRepository()
    repo.merge_branch(branch_name)


@cli.command()
def gc():
    """Run garbage collection."""
    repo = CofRepository()
    repo.garbage_collect()


@cli.command()
def config():
    """Show repository configuration."""
    repo = CofRepository()
    repo.show_config()


@cli.command()
@click.argument("key")
@click.argument("value")
def config_set(key, value):
    """Set a configuration value."""
    repo = CofRepository()
    repo.set_config(key, value)


# Authentication commands
@cli.group()
def auth():
    """Authentication commands."""
    pass


@auth.command()
@click.argument("username")
@click.argument("email")
@click.option("--password", prompt=True, hide_input=True, confirmation_prompt=True)
def user_add(username, email, password):
    """Add a new user."""
    repo = CofRepository()
    repo.create_user(username, email, password)


@auth.command()
@click.option("--username", prompt=True)
@click.option("--password", prompt=True, hide_input=True)
def login(username, password):
    """Login to repository."""
    repo = CofRepository()
    repo.login_user(username, password)


@auth.command()
def logout():
    """Logout from repository."""
    repo = CofRepository()
    repo.logout_user()


@auth.command()
def whoami():
    """Show current user."""
    repo = CofRepository()
    repo.whoami()


@auth.command()
def users():
    """List all users."""
    repo = CofRepository()
    repo.list_users()


@auth.command()
@click.option("--key-size", default=2048, help="SSH key size in bits")
def ssh_generate(key_size):
    """Generate SSH key pair."""
    repo = CofRepository()
    repo.generate_ssh_keys(key_size)


# Remote commands
@cli.group()
def remote():
    """Remote repository commands."""
    pass


@remote.command(name="add")
@click.argument("name")
@click.argument("url")
def remote_add(name, url):
    """Add a remote repository."""
    repo = CofRepository()
    repo.add_remote(name, url)


@remote.command()
@click.argument("name")
def remove(name):
    """Remove a remote repository."""
    repo = CofRepository()
    repo.remove_remote(name)


@remote.command()
def list():
    """List all remote repositories."""
    repo = CofRepository()
    repo.list_remotes()


@cli.command()
@click.argument("url")
@click.argument("target_dir", required=False)
def clone(url, target_dir):
    """Clone a remote repository."""
    import asyncio
    
    if not target_dir:
        # Extract repository name from URL
        target_dir = url.split("/")[-1].replace(".git", "")
    
    repo = CofRepository(target_dir)
    asyncio.run(repo.clone_repository(url, target_dir))


@cli.command()
@click.argument("remote_name", default="origin")
@click.option("--branch", default="main", help="Branch to push")
def push(remote_name, branch):
    """Push to remote repository."""
    import asyncio
    
    repo = CofRepository()
    asyncio.run(repo.push_to_remote(remote_name, branch))


@cli.command()
@click.argument("remote_name", default="origin")
@click.option("--branch", default="main", help="Branch to pull")
def pull(remote_name, branch):
    """Pull from remote repository."""
    import asyncio
    
    repo = CofRepository()
    asyncio.run(repo.pull_from_remote(remote_name, branch))


if __name__ == "__main__":
    cli()