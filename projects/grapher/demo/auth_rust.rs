//! Authentication module for Rust
//! 
//! This module provides authentication functionality for Rust applications.

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

/// Authentication result enum
#[derive(Debug, PartialEq)]
pub enum AuthResult {
    Success,
    InvalidCredentials,
    UserNotFound,
    AccountLocked,
}

/// User credentials structure
#[derive(Debug, Clone)]
pub struct Credentials {
    pub username: String,
    pub password: String,
}

/// Authentication token
#[derive(Debug, Clone)]
pub struct AuthToken {
    pub token: String,
    pub expires_at: u64,
    pub user_id: u32,
}

/// Authentication service trait
pub trait AuthService {
    fn login(&self, credentials: &Credentials) -> Result<AuthToken, AuthResult>;
    fn logout(&self, token: &str) -> bool;
    fn validate_token(&self, token: &str) -> bool;
}

/// Simple authentication service implementation
pub struct SimpleAuthService {
    users: HashMap<String, String>, // username -> password
    tokens: HashMap<String, AuthToken>,
}

impl SimpleAuthService {
    /// Create a new authentication service
    pub fn new() -> Self {
        let mut users = HashMap::new();
        users.insert("admin".to_string(), "password123".to_string());
        users.insert("user".to_string(), "userpass".to_string());
        
        Self {
            users,
            tokens: HashMap::new(),
        }
    }
    
    /// Add a new user to the system
    pub fn add_user(&mut self, username: String, password: String) {
        self.users.insert(username, password);
    }
    
    /// Generate a new authentication token
    fn generate_token(&self, user_id: u32) -> AuthToken {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        AuthToken {
            token: format!("token_{}_{}", user_id, now),
            expires_at: now + 3600, // 1 hour
            user_id,
        }
    }
}

impl AuthService for SimpleAuthService {
    fn login(&self, credentials: &Credentials) -> Result<AuthToken, AuthResult> {
        if let Some(stored_password) = self.users.get(&credentials.username) {
            if stored_password == &credentials.password {
                let user_id = if credentials.username == "admin" { 1 } else { 2 };
                let token = self.generate_token(user_id);
                Ok(token)
            } else {
                Err(AuthResult::InvalidCredentials)
            }
        } else {
            Err(AuthResult::UserNotFound)
        }
    }
    
    fn logout(&self, _token: &str) -> bool {
        // In a real implementation, you would remove the token from storage
        true
    }
    
    fn validate_token(&self, token: &str) -> bool {
        if let Some(auth_token) = self.tokens.get(token) {
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs();
            auth_token.expires_at > now
        } else {
            false
        }
    }
}

/// Password hashing utility
pub struct PasswordHasher;

impl PasswordHasher {
    /// Hash a password (simplified implementation)
    pub fn hash(&self, password: &str) -> String {
        // In a real implementation, you would use a proper hashing algorithm
        format!("hashed_{}", password)
    }
    
    /// Verify a password against its hash
    pub fn verify(&self, password: &str, hash: &str) -> bool {
        hash == self.hash(password)
    }
}

/// Session manager for handling user sessions
pub struct SessionManager {
    sessions: HashMap<String, AuthToken>,
    hasher: PasswordHasher,
}

impl SessionManager {
    /// Create a new session manager
    pub fn new() -> Self {
        Self {
            sessions: HashMap::new(),
            hasher: PasswordHasher,
        }
    }
    
    /// Create a new session
    pub fn create_session(&mut self, token: AuthToken) -> String {
        let session_id = format!("session_{}", token.user_id);
        self.sessions.insert(session_id.clone(), token);
        session_id
    }
    
    /// Get session information
    pub fn get_session(&self, session_id: &str) -> Option<&AuthToken> {
        self.sessions.get(session_id)
    }
    
    /// End a session
    pub fn end_session(&mut self, session_id: &str) -> bool {
        self.sessions.remove(session_id).is_some()
    }
}

/// Authentication middleware for web applications
pub struct AuthMiddleware {
    auth_service: SimpleAuthService,
    session_manager: SessionManager,
}

impl AuthMiddleware {
    /// Create new authentication middleware
    pub fn new() -> Self {
        Self {
            auth_service: SimpleAuthService::new(),
            session_manager: SessionManager::new(),
        }
    }
    
    /// Authenticate a user and create a session
    pub fn authenticate(&mut self, credentials: &Credentials) -> Result<String, AuthResult> {
        match self.auth_service.login(credentials) {
            Ok(token) => {
                let session_id = self.session_manager.create_session(token);
                Ok(session_id)
            }
            Err(err) => Err(err)
        }
    }
    
    /// Check if a session is valid
    pub fn is_authenticated(&self, session_id: &str) -> bool {
        if let Some(token) = self.session_manager.get_session(session_id) {
            self.auth_service.validate_token(&token.token)
        } else {
            false
        }
    }
    
    /// Logout a user
    pub fn logout(&mut self, session_id: &str) -> bool {
        self.session_manager.end_session(session_id)
    }
}

/// Utility function to create test credentials
pub fn create_test_credentials(username: &str, password: &str) -> Credentials {
    Credentials {
        username: username.to_string(),
        password: password.to_string(),
    }
}

/// Utility function to validate email format (simplified)
pub fn is_valid_email(email: &str) -> bool {
    email.contains('@') && email.contains('.')
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_successful_login() {
        let auth_service = SimpleAuthService::new();
        let credentials = create_test_credentials("admin", "password123");
        
        let result = auth_service.login(&credentials);
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_invalid_credentials() {
        let auth_service = SimpleAuthService::new();
        let credentials = create_test_credentials("admin", "wrong_password");
        
        let result = auth_service.login(&credentials);
        assert_eq!(result, Err(AuthResult::InvalidCredentials));
    }
    
    #[test]
    fn test_email_validation() {
        assert!(is_valid_email("user@example.com"));
        assert!(!is_valid_email("invalid_email"));
    }
}
