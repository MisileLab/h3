//! Rust Demo Module
//! 
//! This module demonstrates various Rust language features for the code knowledge graph.

use std::collections::HashMap;
use std::fmt::Display;

/// A simple user struct with basic information
#[derive(Debug, Clone)]
pub struct User {
    pub id: u32,
    pub name: String,
    pub email: String,
    pub is_active: bool,
}

/// A trait for objects that can be displayed
pub trait Displayable {
    fn display(&self) -> String;
}

/// Implementation of Displayable for User
impl Displayable for User {
    fn display(&self) -> String {
        format!("User {}: {} ({})", self.id, self.name, self.email)
    }
}

/// A trait for authentication operations
pub trait Authenticator {
    fn authenticate(&self, credentials: &str) -> bool;
    fn logout(&self);
}

/// Implementation of Authenticator for User
impl Authenticator for User {
    fn authenticate(&self, credentials: &str) -> bool {
        // Simple authentication logic
        self.is_active && credentials.len() > 0
    }
    
    fn logout(&self) {
        println!("User {} logged out", self.name);
    }
}

/// An enum representing different user roles
#[derive(Debug, PartialEq)]
pub enum UserRole {
    Admin,
    User,
    Guest,
}

/// A struct for managing user sessions
pub struct UserSession {
    user: User,
    role: UserRole,
    created_at: std::time::SystemTime,
}

impl UserSession {
    /// Create a new user session
    pub fn new(user: User, role: UserRole) -> Self {
        Self {
            user,
            role,
            created_at: std::time::SystemTime::now(),
        }
    }
    
    /// Get the user associated with this session
    pub fn get_user(&self) -> &User {
        &self.user
    }
    
    /// Check if the user has admin privileges
    pub fn is_admin(&self) -> bool {
        self.role == UserRole::Admin
    }
    
    /// Authenticate the user with credentials
    pub fn authenticate(&self, credentials: &str) -> bool {
        self.user.authenticate(credentials)
    }
}

/// A generic container for storing items
pub struct Container<T> {
    items: Vec<T>,
    capacity: usize,
}

impl<T> Container<T> {
    /// Create a new container with specified capacity
    pub fn new(capacity: usize) -> Self {
        Self {
            items: Vec::new(),
            capacity,
        }
    }
    
    /// Add an item to the container
    pub fn add(&mut self, item: T) -> Result<(), String> {
        if self.items.len() >= self.capacity {
            return Err("Container is full".to_string());
        }
        self.items.push(item);
        Ok(())
    }
    
    /// Get the number of items in the container
    pub fn len(&self) -> usize {
        self.items.len()
    }
    
    /// Check if the container is empty
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

/// A function to create a default user
pub fn create_default_user() -> User {
    User {
        id: 1,
        name: "Default User".to_string(),
        email: "default@example.com".to_string(),
        is_active: true,
    }
}

/// A function to authenticate a user with credentials
pub fn authenticate_user(user: &User, credentials: &str) -> bool {
    user.authenticate(credentials)
}

/// A function to process user data
pub fn process_user_data(users: &[User]) -> HashMap<String, u32> {
    let mut stats = HashMap::new();
    
    for user in users {
        let count = stats.entry(user.name.clone()).or_insert(0);
        *count += 1;
    }
    
    stats
}

/// An async function for handling user operations
pub async fn handle_user_operation(user: User) -> Result<String, String> {
    // Simulate some async operation
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    
    if user.is_active {
        Ok(format!("Operation completed for user: {}", user.name))
    } else {
        Err("User is not active".to_string())
    }
}

/// A macro for creating user instances
#[macro_export]
macro_rules! create_user {
    ($id:expr, $name:expr, $email:expr) => {
        User {
            id: $id,
            name: $name.to_string(),
            email: $email.to_string(),
            is_active: true,
        }
    };
}

/// Unit tests for the module
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_user_creation() {
        let user = create_default_user();
        assert_eq!(user.id, 1);
        assert_eq!(user.name, "Default User");
    }
    
    #[test]
    fn test_authentication() {
        let user = create_default_user();
        assert!(authenticate_user(&user, "valid_credentials"));
    }
    
    #[test]
    fn test_container_operations() {
        let mut container = Container::new(5);
        assert!(container.add(1).is_ok());
        assert_eq!(container.len(), 1);
    }
}
