"""
Remote repository management for cof distributed version control.
Handles remote operations like clone, push, and pull.
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Set
import click

from cof.network import NetworkClient, RemoteRepository, CofProtocolError
from cof.models import Commit, Tree, TreeEntry


class RemoteManager:
    """Manages remote repositories for a cof repository."""

    def __init__(self, repository):
        self.repository = repository
        self.remotes_file = repository.cof_dir / "remotes.json"
        self.remotes = self._load_remotes()

    def _load_remotes(self) -> Dict[str, RemoteRepository]:
        """Load remote configurations."""
        if self.remotes_file.exists():
            with open(self.remotes_file, "r") as f:
                data = json.load(f)
                return {
                    name: RemoteRepository(
                        name=name,
                        url=info["url"],
                        host=info["host"],
                        port=info["port"],
                        protocol=info.get("protocol", "udp")
                    )
                    for name, info in data.items()
                }
        return {}

    def _save_remotes(self) -> None:
        """Save remote configurations."""
        data = {
            name: {
                "url": remote.url,
                "host": remote.host,
                "port": remote.port,
                "protocol": remote.protocol
            }
            for name, remote in self.remotes.items()
        }
        
        with open(self.remotes_file, "w") as f:
            json.dump(data, f, indent=2)

    def add_remote(self, name: str, url: str) -> None:
        """Add a new remote repository."""
        if name in self.remotes:
            raise click.ClickException(f"Remote '{name}' already exists.")
        
        remote = RemoteRepository.from_url(name, url)
        self.remotes[name] = remote
        self._save_remotes()
        
        click.echo(f"Added remote '{name}' ({remote.url})")

    def remove_remote(self, name: str) -> None:
        """Remove a remote repository."""
        if name not in self.remotes:
            raise click.ClickException(f"Remote '{name}' does not exist.")
        
        del self.remotes[name]
        self._save_remotes()
        
        click.echo(f"Removed remote '{name}'")

    def list_remotes(self) -> None:
        """List all remote repositories."""
        if not self.remotes:
            click.echo("No remotes configured.")
            return
        
        for name, remote in self.remotes.items():
            click.echo(f"{name}\t{remote.url}")

    def get_remote(self, name: str) -> Optional[RemoteRepository]:
        """Get a remote repository by name."""
        return self.remotes.get(name)

    def get_default_remote(self) -> Optional[RemoteRepository]:
        """Get the default remote (origin)."""
        return self.remotes.get("origin")


class RemoteOperations:
    """Handles remote operations like clone, push, and pull."""

    def __init__(self, repository):
        self.repository = repository
        self.remote_manager = RemoteManager(repository)

    async def clone_repository(self, url: str, target_dir: str) -> bool:
        """Clone a remote repository."""
        try:
            # Create remote from URL
            remote = RemoteRepository.from_url("origin", url)
            
            # Initialize target repository
            target_path = Path(target_dir)
            if target_path.exists() and any(target_path.iterdir()):
                raise click.ClickException(f"Target directory '{target_dir}' is not empty.")
            
            target_path.mkdir(parents=True, exist_ok=True)
            
            # Create temporary repository for cloning
            from .main import CofRepository
            temp_repo = CofRepository(str(target_path))
            temp_repo.init()
            
            # Connect to remote and fetch data
            async with NetworkClient(temp_repo.config) as client:
                if not await client.handshake(remote):
                    raise click.ClickException("Failed to connect to remote repository.")
                
                # Fetch refs
                refs = await client.request_refs(remote)
                if not refs:
                    raise click.ClickException("No refs found on remote repository.")
                
                # Fetch objects for main branch
                main_commit = refs.get("main")
                if not main_commit:
                    raise click.ClickException("Remote repository has no main branch.")
                
                # Fetch all objects recursively
                await self._fetch_objects_recursive(client, remote, main_commit, temp_repo)
                
                # Update local refs
                for ref_name, commit_hash in refs.items():
                    ref_path = temp_repo.cof_dir / "refs" / "heads" / ref_name
                    ref_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(ref_path, "w") as f:
                        f.write(commit_hash)
                
                # Set HEAD to main branch
                with open(temp_repo.cof_dir / "HEAD", "w") as f:
                    f.write("ref: refs/heads/main")
                
                # Restore working tree
                temp_repo._restore_working_tree()
            
            click.echo(f"Cloned repository from {url} to {target_dir}")
            return True

        except Exception as e:
            click.echo(f"Clone failed: {e}")
            return False

    async def _fetch_objects_recursive(self, client: NetworkClient, remote: RemoteRepository, 
                                     object_hash: str, repository, fetched: Optional[Set[str]] = None) -> None:
        """Recursively fetch objects from remote."""
        if fetched is None:
            fetched = set()
        
        if object_hash in fetched:
            return
        
        fetched.add(object_hash)
        
        # Fetch the object
        obj_data = await client.request_object(remote, object_hash)
        if not obj_data:
            raise click.ClickException(f"Failed to fetch object {object_hash}")
        
        # Save object locally
        obj_dict = json.loads(obj_data.decode())
        repository._save_object(obj_dict, "object")  # Will be saved in hot tier
        
        # If it's a commit, fetch its tree and parent
        if "parent" in obj_dict and obj_dict["parent"]:
            await self._fetch_objects_recursive(client, remote, obj_dict["parent"], repository, fetched)
        
        if "tree_root" in obj_dict:
            await self._fetch_tree_recursive(client, remote, obj_dict["tree_root"], repository, fetched)

    async def _fetch_tree_recursive(self, client: NetworkClient, remote: RemoteRepository,
                                  tree_hash: str, repository, fetched: Set[str]) -> None:
        """Recursively fetch tree objects."""
        if tree_hash in fetched:
            return
        
        # Fetch tree object
        obj_data = await client.request_object(remote, tree_hash)
        if not obj_data:
            raise click.ClickException(f"Failed to fetch tree {tree_hash}")
        
        fetched.add(tree_hash)
        
        # Save tree object
        tree_dict = json.loads(obj_data.decode())
        repository._save_object(tree_dict, "object")
        
        # Fetch all blob objects in the tree
        if "entries" in tree_dict:
            for entry_data in tree_dict["entries"].values():
                blob_hash = entry_data["hash"]
                if blob_hash not in fetched:
                    await self._fetch_objects_recursive(client, remote, blob_hash, repository, fetched)

    async def push_to_remote(self, remote_name: str, branch: str = "main") -> bool:
        """Push commits to remote repository."""
        try:
            remote = self.remote_manager.get_remote(remote_name)
            if not remote:
                raise click.ClickException(f"Remote '{remote_name}' not found.")
            
            # Get local branch commit
            ref_path = self.repository._get_branch_ref_path(branch)
            if not ref_path.exists():
                raise click.ClickException(f"Branch '{branch}' does not exist locally.")
            
            with open(ref_path, "r") as f:
                local_commit = f.read().strip()
            
            # Get objects to push
            objects_to_push = await self._get_objects_to_push(local_commit)
            
            if not objects_to_push:
                click.echo("Nothing to push.")
                return True
            
            # Connect to remote and push
            async with NetworkClient(self.repository.config) as client:
                if not await client.handshake(remote):
                    raise click.ClickException("Failed to connect to remote repository.")
                
                # Push objects
                success = await client.push_objects(remote, objects_to_push)
                if not success:
                    raise click.ClickException("Failed to push objects to remote.")
                
                # Update remote ref (simplified - would need proper ref negotiation)
                click.echo(f"Pushed {len(objects_to_push)} objects to {remote_name}")
                return True

        except Exception as e:
            click.echo(f"Push failed: {e}")
            return False

    async def _get_objects_to_push(self, commit_hash: str, fetched: Optional[Set[str]] = None) -> Dict[str, bytes]:
        """Get all objects that need to be pushed for a commit."""
        if fetched is None:
            fetched = set()
        
        objects = {}
        
        if commit_hash in fetched:
            return objects
        
        # Get commit object
        commit_data = self.repository._load_object(commit_hash)
        if not commit_data:
            return objects
        
        fetched.add(commit_hash)
        objects[commit_hash] = json.dumps(commit_data).encode()
        
        # Get parent commit
        if commit_data.get("parent"):
            parent_objects = await self._get_objects_to_push(commit_data["parent"], fetched)
            objects.update(parent_objects)
        
        # Get tree object
        if commit_data.get("tree_root"):
            tree_objects = await self._get_tree_objects(commit_data["tree_root"], fetched)
            objects.update(tree_objects)
        
        return objects

    async def _get_tree_objects(self, tree_hash: str, fetched: Set[str]) -> Dict[str, bytes]:
        """Get all objects in a tree."""
        objects = {}
        
        if tree_hash in fetched:
            return objects
        
        # Get tree object
        tree_data = self.repository._load_object(tree_hash)
        if not tree_data:
            return objects
        
        fetched.add(tree_hash)
        objects[tree_hash] = json.dumps(tree_data).encode()
        
        # Get all blob objects in the tree
        if "entries" in tree_data:
            for entry_data in tree_data["entries"].values():
                blob_hash = entry_data["hash"]
                if blob_hash not in fetched:
                    blob_data = self.repository._load_object(blob_hash)
                    if blob_data:
                        fetched.add(blob_hash)
                        objects[blob_hash] = json.dumps(blob_data).encode()
        
        return objects

    async def pull_from_remote(self, remote_name: str, branch: str = "main") -> bool:
        """Pull changes from remote repository."""
        try:
            remote = self.remote_manager.get_remote(remote_name)
            if not remote:
                raise click.ClickException(f"Remote '{remote_name}' not found.")
            
            # Connect to remote
            async with NetworkClient(self.repository.config) as client:
                if not await client.handshake(remote):
                    raise click.ClickException("Failed to connect to remote repository.")
                
                # Fetch remote refs
                refs = await client.request_refs(remote)
                remote_commit = refs.get(branch)
                
                if not remote_commit:
                    raise click.ClickException(f"Remote branch '{branch}' not found.")
                
                # Get local commit
                local_commit = self.repository._get_head_commit()
                
                if local_commit == remote_commit:
                    click.echo("Already up to date.")
                    return True
                
                # Fetch objects from remote
                await self._fetch_objects_recursive(client, remote, remote_commit, self.repository)
                
                # Update local ref
                ref_path = self.repository._get_branch_ref_path(branch)
                ref_path.parent.mkdir(parents=True, exist_ok=True)
                with open(ref_path, "w") as f:
                    f.write(remote_commit)
                
                # Restore working tree
                self.repository._restore_working_tree()
                
                click.echo(f"Pulled changes from {remote_name}/{branch}")
                return True

        except Exception as e:
            click.echo(f"Pull failed: {e}")
            return False

    async def start_server(self, host: str = "0.0.0.0", port: int = 7357) -> None:
        """Start the cof server."""
        from cof.network import NetworkServer
        
        server = NetworkServer(self.repository, self.repository.config)
        server.host = host
        server.port = port
        
        click.echo(f"Starting cof server on {host}:{port}")
        
        try:
            await server.start()
        except KeyboardInterrupt:
            click.echo("\nShutting down server...")
            await server.stop()