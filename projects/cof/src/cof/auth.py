"""
Authentication and authorization system for cof distributed version control.
Implements token-based auth, SSH key support, and user management.
"""

import asyncio
import hashlib
import hmac
import json
import os
import secrets
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import click
import base64

try:
    import cryptography
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.fernet import Fernet
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False
    # Create dummy classes for type hints
    class hashes:
        class SHA256:
            pass
    class serialization:
        pass
    class rsa:
        pass
    class padding:
        pass
    class PBKDF2HMAC:
        pass


class AuthMethod(Enum):
    """Authentication methods."""
    TOKEN = "token"
    SSH_KEY = "ssh_key"
    PASSWORD = "password"


class Permission(Enum):
    """Repository permissions."""
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"


@dataclass
class User:
    """User account information."""
    username: str
    email: str
    password_hash: str
    salt: str
    ssh_public_keys: List[str] = field(default_factory=list)
    permissions: Dict[str, Set[Permission]] = field(default_factory=dict)
    created_at: int = field(default_factory=lambda: int(time.time()))
    last_login: Optional[int] = None
    is_active: bool = True

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "username": self.username,
            "email": self.email,
            "password_hash": self.password_hash,
            "salt": self.salt,
            "ssh_public_keys": self.ssh_public_keys,
            "permissions": {
                repo: list(perms) for repo, perms in self.permissions.items()
            },
            "created_at": self.created_at,
            "last_login": self.last_login,
            "is_active": self.is_active
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "User":
        """Create from dictionary."""
        user = cls(
            username=data["username"],
            email=data["email"],
            password_hash=data["password_hash"],
            salt=data["salt"],
            ssh_public_keys=data.get("ssh_public_keys", []),
            created_at=data.get("created_at", int(time.time())),
            last_login=data.get("last_login"),
            is_active=data.get("is_active", True)
        )
        
        # Convert permissions back to sets
        for repo, perms in data.get("permissions", {}).items():
            user.permissions[repo] = set(Permission(p) for p in perms)
        
        return user


@dataclass
class AuthToken:
    """Authentication token."""
    token_id: str
    user_id: str
    expires_at: int
    permissions: Set[Permission]
    repository: Optional[str] = None
    created_at: int = field(default_factory=lambda: int(time.time()))

    def is_expired(self) -> bool:
        """Check if token is expired."""
        return time.time() > self.expires_at

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "token_id": self.token_id,
            "user_id": self.user_id,
            "expires_at": self.expires_at,
            "permissions": [p.value for p in self.permissions],
            "repository": self.repository,
            "created_at": self.created_at
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "AuthToken":
        """Create from dictionary."""
        return cls(
            token_id=data["token_id"],
            user_id=data["user_id"],
            expires_at=data["expires_at"],
            permissions=set(Permission(p) for p in data["permissions"]),
            repository=data.get("repository"),
            created_at=data.get("created_at", int(time.time()))
        )


class AuthManager:
    """Manages authentication and authorization."""

    def __init__(self, cof_dir: Path):
        self.cof_dir = cof_dir
        self.auth_dir = cof_dir / "auth"
        self.auth_dir.mkdir(exist_ok=True)
        
        self.users_file = self.auth_dir / "users.json"
        self.tokens_file = self.auth_dir / "tokens.json"
        self.config_file = self.auth_dir / "config.json"
        
        self.users = self._load_users()
        self.tokens = self._load_tokens()
        self.config = self._load_config()

    def _load_users(self) -> Dict[str, User]:
        """Load users from file."""
        if self.users_file.exists():
            with open(self.users_file, "r") as f:
                data = json.load(f)
                return {
                    username: User.from_dict(user_data)
                    for username, user_data in data.items()
                }
        return {}

    def _save_users(self) -> None:
        """Save users to file."""
        data = {
            username: user.to_dict()
            for username, user in self.users.items()
        }
        with open(self.users_file, "w") as f:
            json.dump(data, f, indent=2)

    def _load_tokens(self) -> Dict[str, AuthToken]:
        """Load tokens from file."""
        if self.tokens_file.exists():
            with open(self.tokens_file, "r") as f:
                data = json.load(f)
                return {
                    token_id: AuthToken.from_dict(token_data)
                    for token_id, token_data in data.items()
                }
        return {}

    def _save_tokens(self) -> None:
        """Save tokens to file."""
        data = {
            token_id: token.to_dict()
            for token_id, token in self.tokens.items()
        }
        with open(self.tokens_file, "w") as f:
            json.dump(data, f, indent=2)

    def _load_config(self) -> Dict:
        """Load auth configuration."""
        default_config = {
            "token_expiry_hours": 24,
            "max_login_attempts": 5,
            "lockout_duration_minutes": 15,
            "require_ssh_auth": False,
            "password_min_length": 8
        }
        
        if self.config_file.exists():
            with open(self.config_file, "r") as f:
                config = json.load(f)
                default_config.update(config)
        
        return default_config

    def _save_config(self) -> None:
        """Save auth configuration."""
        with open(self.config_file, "w") as f:
            json.dump(self.config, f, indent=2)

    def _hash_password(self, password: str, salt: Optional[str] = None) -> Tuple[str, str]:
        """Hash password with salt."""
        if salt is None:
            salt = secrets.token_hex(16)
        
        if CRYPTOGRAPHY_AVAILABLE:
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt.encode(),
                iterations=100000,
            )
            password_hash = base64.urlsafe_b64encode(kdf.derive(password.encode())).decode()
        else:
            # Fallback to simple hashing
            password_hash = hashlib.pbkdf2_hmac(
                'sha256', password.encode(), salt.encode(), 100000
            ).hex()
        
        return password_hash, salt

    def _verify_password(self, password: str, password_hash: str, salt: str) -> bool:
        """Verify password against hash."""
        computed_hash, _ = self._hash_password(password, salt)
        return hmac.compare_digest(computed_hash, password_hash)

    def create_user(self, username: str, email: str, password: str, 
                   permissions: Optional[Dict[str, List[Permission]]] = None) -> bool:
        """Create a new user."""
        if username in self.users:
            raise click.ClickException(f"User '{username}' already exists.")
        
        if len(password) < self.config["password_min_length"]:
            raise click.ClickException(
                f"Password must be at least {self.config['password_min_length']} characters."
            )
        
        password_hash, salt = self._hash_password(password)
        
        user = User(
            username=username,
            email=email,
            password_hash=password_hash,
            salt=salt
        )
        
        # Add permissions
        if permissions:
            for repo, perms in permissions.items():
                user.permissions[repo] = set(perms)
        
        self.users[username] = user
        self._save_users()
        
        click.echo(f"Created user '{username}'")
        return True

    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user with password."""
        user = self.users.get(username)
        if not user or not user.is_active:
            return None
        
        if self._verify_password(password, user.password_hash, user.salt):
            user.last_login = int(time.time())
            self._save_users()
            return user
        
        return None

    def create_token(self, user: User, repository: Optional[str] = None,
                    permissions: Optional[Set[Permission]] = None,
                    expires_in_hours: Optional[int] = None) -> str:
        """Create authentication token."""
        if permissions is None:
            permissions = {Permission.READ, Permission.WRITE}
        
        if expires_in_hours is None:
            expires_in_hours = self.config.get("token_expiry_hours", 24)
        
        token_id = secrets.token_urlsafe(32)
        expires_at = int(time.time()) + (expires_in_hours or 24) * 3600
        
        token = AuthToken(
            token_id=token_id,
            user_id=user.username,
            expires_at=expires_at,
            permissions=permissions,
            repository=repository
        )
        
        self.tokens[token_id] = token
        self._save_tokens()
        
        return token_id

    def validate_token(self, token_id: str, repository: Optional[str] = None,
                      required_permission: Optional[Permission] = None) -> Optional[User]:
        """Validate authentication token."""
        token = self.tokens.get(token_id)
        if not token or token.is_expired():
            return None
        
        # Check repository access
        if repository and token.repository and token.repository != repository:
            return None
        
        # Check permissions
        if required_permission and required_permission not in token.permissions:
            return None
        
        user = self.users.get(token.user_id)
        if not user or not user.is_active:
            return None
        
        return user

    def revoke_token(self, token_id: str) -> bool:
        """Revoke authentication token."""
        if token_id in self.tokens:
            del self.tokens[token_id]
            self._save_tokens()
            return True
        return False

    def add_ssh_key(self, username: str, ssh_public_key: str) -> bool:
        """Add SSH public key to user."""
        user = self.users.get(username)
        if not user:
            raise click.ClickException(f"User '{username}' not found.")
        
        # Validate SSH key format (simplified)
        if not ssh_public_key.startswith(("ssh-rsa", "ssh-ed25519", "ecdsa-sha2")):
            raise click.ClickException("Invalid SSH public key format.")
        
        if ssh_public_key not in user.ssh_public_keys:
            user.ssh_public_keys.append(ssh_public_key)
            self._save_users()
            click.echo(f"Added SSH key to user '{username}'")
        
        return True

    def remove_ssh_key(self, username: str, ssh_public_key: str) -> bool:
        """Remove SSH public key from user."""
        user = self.users.get(username)
        if not user:
            raise click.ClickException(f"User '{username}' not found.")
        
        if ssh_public_key in user.ssh_public_keys:
            user.ssh_public_keys.remove(ssh_public_key)
            self._save_users()
            click.echo(f"Removed SSH key from user '{username}'")
            return True
        
        return False

    def grant_permission(self, username: str, repository: str, permission: Permission) -> bool:
        """Grant permission to user for repository."""
        user = self.users.get(username)
        if not user:
            raise click.ClickException(f"User '{username}' not found.")
        
        if repository not in user.permissions:
            user.permissions[repository] = set()
        
        user.permissions[repository].add(permission)
        self._save_users()
        
        click.echo(f"Granted {permission.value} permission to '{username}' for repository '{repository}'")
        return True

    def revoke_permission(self, username: str, repository: str, permission: Permission) -> bool:
        """Revoke permission from user for repository."""
        user = self.users.get(username)
        if not user:
            raise click.ClickException(f"User '{username}' not found.")
        
        if repository in user.permissions and permission in user.permissions[repository]:
            user.permissions[repository].remove(permission)
            if not user.permissions[repository]:
                del user.permissions[repository]
            self._save_users()
            
            click.echo(f"Revoked {permission.value} permission from '{username}' for repository '{repository}'")
            return True
        
        return False

    def list_users(self) -> List[User]:
        """List all users."""
        return list(self.users.values())

    def get_user(self, username: str) -> Optional[User]:
        """Get user by username."""
        return self.users.get(username)

    def cleanup_expired_tokens(self) -> int:
        """Remove expired tokens."""
        expired_tokens = [
            token_id for token_id, token in self.tokens.items()
            if token.is_expired()
        ]
        
        for token_id in expired_tokens:
            del self.tokens[token_id]
        
        if expired_tokens:
            self._save_tokens()
        
        return len(expired_tokens)


class ClientAuth:
    """Client-side authentication management."""

    def __init__(self, cof_dir: Path):
        self.cof_dir = cof_dir
        self.auth_dir = cof_dir / "client_auth"
        self.auth_dir.mkdir(exist_ok=True)
        
        self.credentials_file = self.auth_dir / "credentials.json"
        self.credentials = self._load_credentials()

    def _load_credentials(self) -> Dict:
        """Load stored credentials."""
        if self.credentials_file.exists():
            with open(self.credentials_file, "r") as f:
                return json.load(f)
        return {}

    def _save_credentials(self) -> None:
        """Save credentials."""
        with open(self.credentials_file, "w") as f:
            json.dump(self.credentials, f, indent=2)

    def store_token(self, remote_url: str, token: str) -> None:
        """Store authentication token for remote."""
        self.credentials[remote_url] = {
            "type": "token",
            "value": token,
            "stored_at": int(time.time())
        }
        self._save_credentials()

    def get_token(self, remote_url: str) -> Optional[str]:
        """Get stored token for remote."""
        cred = self.credentials.get(remote_url)
        if cred and cred["type"] == "token":
            return cred["value"]
        return None

    def remove_credentials(self, remote_url: str) -> bool:
        """Remove stored credentials for remote."""
        if remote_url in self.credentials:
            del self.credentials[remote_url]
            self._save_credentials()
            return True
        return False

    def list_remotes(self) -> List[str]:
        """List all remotes with stored credentials."""
        return list(self.credentials.keys())


def generate_ssh_keypair(key_size: int = 2048) -> Tuple[str, str]:
    """Generate SSH key pair."""
    if not CRYPTOGRAPHY_AVAILABLE:
        raise click.ClickException("Cryptography library is required for SSH key generation.")
    
    # Generate private key
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=key_size
    )
    
    # Serialize private key
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.OpenSSH,
        encryption_algorithm=serialization.NoEncryption()
    ).decode()
    
    # Serialize public key
    public_key = private_key.public_key()
    public_pem = public_key.public_bytes(
        encoding=serialization.Encoding.OpenSSH,
        format=serialization.PublicFormat.OpenSSH
    ).decode()
    
    return private_pem, public_pem


def verify_ssh_signature(message: bytes, signature: bytes, public_key: str) -> bool:
    """Verify SSH signature."""
    if not CRYPTOGRAPHY_AVAILABLE:
        return False
    
    try:
        # Load public key
        key = serialization.load_ssh_public_key(public_key.encode())
        
        # Verify signature
        key.verify(
            signature,
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return True
    except Exception:
        return False