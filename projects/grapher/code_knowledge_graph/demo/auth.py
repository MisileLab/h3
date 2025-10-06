"""
Authentication module for demo application.

This module provides user authentication functionality including
user validation, password hashing, and session management.
"""

import hashlib
import secrets
from typing import Optional, Dict
from datetime import datetime, timedelta


class User:
    """User model representing a registered user."""
    
    def __init__(self, username: str, email: str, password_hash: str):
        self.username = username
        self.email = email
        self.password_hash = password_hash
        self.created_at = datetime.now()
        self.last_login = None
        self.is_active = True
    
    def __repr__(self):
        return f"User(username='{self.username}', email='{self.email}')"
    
    def update_last_login(self):
        """Update the last login timestamp."""
        self.last_login = datetime.now()
    
    def deactivate(self):
        """Deactivate the user account."""
        self.is_active = False


class Session:
    """Session model for managing user sessions."""
    
    def __init__(self, user_id: str, token: str):
        self.user_id = user_id
        self.token = token
        self.created_at = datetime.now()
        self.expires_at = datetime.now() + timedelta(hours=24)
        self.is_active = True
    
    def is_expired(self) -> bool:
        """Check if the session has expired."""
        return datetime.now() > self.expires_at
    
    def extend_session(self, hours: int = 24):
        """Extend the session expiration time."""
        self.expires_at = datetime.now() + timedelta(hours=hours)


class AuthManager:
    """Main authentication manager class."""
    
    def __init__(self):
        self.users: Dict[str, User] = {}
        self.sessions: Dict[str, Session] = {}
    
    def hash_password(self, password: str) -> str:
        """Hash a password using SHA-256 with salt."""
        salt = secrets.token_hex(16)
        password_hash = hashlib.sha256((password + salt).encode()).hexdigest()
        return f"{salt}:{password_hash}"
    
    def verify_password(self, password: str, password_hash: str) -> bool:
        """Verify a password against its hash."""
        try:
            salt, hash_value = password_hash.split(':')
            computed_hash = hashlib.sha256((password + salt).encode()).hexdigest()
            return computed_hash == hash_value
        except ValueError:
            return False
    
    def register_user(self, username: str, email: str, password: str) -> bool:
        """Register a new user."""
        if username in self.users:
            return False
        
        password_hash = self.hash_password(password)
        user = User(username, email, password_hash)
        self.users[username] = user
        return True
    
    def authenticate_user(self, username: str, password: str) -> Optional[str]:
        """Authenticate a user and return a session token."""
        user = self.users.get(username)
        if not user or not user.is_active:
            return None
        
        if not self.verify_password(password, user.password_hash):
            return None
        
        # Create session
        token = secrets.token_urlsafe(32)
        session = Session(username, token)
        self.sessions[token] = session
        
        # Update user last login
        user.update_last_login()
        
        return token
    
    def validate_session(self, token: str) -> Optional[str]:
        """Validate a session token and return the user ID."""
        session = self.sessions.get(token)
        if not session or not session.is_active or session.is_expired():
            return None
        
        return session.user_id
    
    def logout_user(self, token: str) -> bool:
        """Logout a user by invalidating their session."""
        if token in self.sessions:
            self.sessions[token].is_active = False
            return True
        return False
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get a user by their username."""
        return self.users.get(username)
    
    def change_password(self, username: str, old_password: str, new_password: str) -> bool:
        """Change a user's password."""
        user = self.users.get(username)
        if not user or not self.verify_password(old_password, user.password_hash):
            return False
        
        user.password_hash = self.hash_password(new_password)
        return True


# Global auth manager instance
auth_manager = AuthManager()


def create_default_admin() -> bool:
    """Create a default admin user for demo purposes."""
    return auth_manager.register_user("admin", "admin@example.com", "admin123")


def get_user_sessions(username: str) -> list:
    """Get all active sessions for a user."""
    user_sessions = []
    for token, session in auth_manager.sessions.items():
        if session.user_id == username and session.is_active and not session.is_expired():
            user_sessions.append({
                'token': token[:8] + '...',  # Show only first 8 chars
                'created_at': session.created_at,
                'expires_at': session.expires_at
            })
    return user_sessions


if __name__ == "__main__":
    # Demo functionality
    print("Creating default admin user...")
    if create_default_admin():
        print("Admin user created successfully!")
        
        # Test authentication
        token = auth_manager.authenticate_user("admin", "admin123")
        if token:
            print(f"Authentication successful! Token: {token[:8]}...")
            
            # Validate session
            user_id = auth_manager.validate_session(token)
            print(f"Session validated for user: {user_id}")
        else:
            print("Authentication failed!")
    else:
        print("Admin user already exists or creation failed!")