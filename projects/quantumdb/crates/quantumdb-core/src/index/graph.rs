//! Graph data structures for HNSW

use std::sync::Arc;
use parking_lot::RwLock;

/// Graph node representation
#[derive(Debug, Clone)]
pub struct Node {
    /// Node ID
    pub id: usize,
    /// Node level in HNSW
    pub level: usize,
    /// Compressed vector data
    pub vector: Vec<u8>,
}

/// Single layer of the HNSW graph
#[derive(Debug)]
pub struct Layer {
    /// Layer level (0 = densest)
    pub level: usize,
    /// Maximum connections per node in this layer
    pub max_connections: usize,
    /// Adjacency list: node_id -> neighbors
    pub adjacency: dashmap::DashMap<usize, Arc<RwLock<Vec<usize>>>>,
}

impl Layer {
    /// Create a new layer
    /// 
    /// # Arguments
    /// * `level` - Layer level
    /// * `max_connections` - Maximum connections per node
    pub fn new(level: usize, max_connections: usize) -> Self {
        Self {
            level,
            max_connections,
            adjacency: dashmap::DashMap::new(),
        }
    }
    
    /// Add a node to the layer
    /// 
    /// # Arguments
    /// * `id` - Node ID
    pub fn add_node(&self, id: usize) {
        if !self.adjacency.contains_key(&id) {
            self.adjacency.insert(id, Arc::new(RwLock::new(Vec::new())));
        }
    }
    
    /// Add connection between two nodes
    /// 
    /// # Arguments
    /// * `from` - Source node ID
    /// * `to` - Target node ID
    pub fn add_connection(&self, from: usize, to: usize) -> Result<(), GraphError> {
        // Ensure both nodes exist
        if !self.adjacency.contains_key(&from) {
            self.add_node(from);
        }
        if !self.adjacency.contains_key(&to) {
            self.add_node(to);
        }
        
        // Add connection from 'from' to 'to'
        {
            let from_neighbors = self.adjacency.get(&from).unwrap();
            let mut neighbors = from_neighbors.write();
            
            if neighbors.len() >= self.max_connections {
                return Err(GraphError::TooManyConnections);
            }
            
            if !neighbors.contains(&to) {
                neighbors.push(to);
            }
        }
        
        // Add connection from 'to' to 'from' (bidirectional)
        {
            let to_neighbors = self.adjacency.get(&to).unwrap();
            let mut neighbors = to_neighbors.write();
            
            if neighbors.len() < self.max_connections && !neighbors.contains(&from) {
                neighbors.push(from);
            }
        }
        
        Ok(())
    }
    
    /// Get neighbors of a node
    /// 
    /// # Arguments
    /// * `id` - Node ID
    /// 
    /// # Returns
    /// Vector of neighbor IDs
    pub fn get_neighbors(&self, id: usize) -> Vec<usize> {
        if let Some(neighbors) = self.adjacency.get(&id) {
            neighbors.read().clone()
        } else {
            Vec::new()
        }
    }
    
    /// Remove a node from the layer
    /// 
    /// # Arguments
    /// * `id` - Node ID
    pub fn remove_node(&self, id: usize) {
        // Remove from adjacency lists of neighbors
        if let Some(neighbors) = self.adjacency.get(&id) {
            let neighbor_ids = neighbors.read().clone();
            drop(neighbors);
            
            for &neighbor_id in &neighbor_ids {
                if let Some(neighbor_neighbors) = self.adjacency.get(&neighbor_id) {
                    let mut neighbor_list = neighbor_neighbors.write();
                    neighbor_list.retain(|&n| n != id);
                }
            }
        }
        
        // Remove the node itself
        self.adjacency.remove(&id);
    }
    
    /// Get the number of nodes in the layer
    pub fn node_count(&self) -> usize {
        self.adjacency.len()
    }
    
    /// Get the average degree (connections per node)
    pub fn average_degree(&self) -> f64 {
        if self.adjacency.is_empty() {
            return 0.0;
        }
        
        let total_connections: usize = self
            .adjacency
            .iter()
            .map(|entry| entry.value().read().len())
            .sum();
        
        total_connections as f64 / self.adjacency.len() as f64
    }
}

/// Graph-related errors
#[derive(Debug, thiserror::Error)]
pub enum GraphError {
    #[error("Node {0} not found")]
    NodeNotFound(usize),
    
    #[error("Too many connections")]
    TooManyConnections,
    
    #[error("Layer {0} not found")]
    LayerNotFound(usize),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_creation() {
        let layer = Layer::new(0, 16);
        assert_eq!(layer.level, 0);
        assert_eq!(layer.max_connections, 16);
        assert_eq!(layer.node_count(), 0);
    }

    #[test]
    fn test_add_node() {
        let layer = Layer::new(0, 16);
        layer.add_node(1);
        assert_eq!(layer.node_count(), 1);
        assert_eq!(layer.get_neighbors(1), Vec::<usize>::new());
    }

    #[test]
    fn test_add_connection() {
        let layer = Layer::new(0, 16);
        
        // Add connection
        layer.add_connection(1, 2).unwrap();
        
        // Check bidirectional connection
        assert_eq!(layer.get_neighbors(1), vec![2]);
        assert_eq!(layer.get_neighbors(2), vec![1]);
        
        // Test duplicate connection (should not add duplicate)
        layer.add_connection(1, 2).unwrap();
        assert_eq!(layer.get_neighbors(1), vec![2]);
        assert_eq!(layer.get_neighbors(2), vec![1]);
    }

    #[test]
    fn test_too_many_connections() {
        let layer = Layer::new(0, 2);
        
        // Add max connections
        layer.add_connection(1, 2).unwrap();
        layer.add_connection(1, 3).unwrap();
        
        // Try to add one more (should fail)
        let result = layer.add_connection(1, 4);
        assert!(matches!(result, Err(GraphError::TooManyConnections)));
    }

    #[test]
    fn test_remove_node() {
        let layer = Layer::new(0, 16);
        
        // Add connections
        layer.add_connection(1, 2).unwrap();
        layer.add_connection(1, 3).unwrap();
        
        // Remove node 1
        layer.remove_node(1);
        
        // Check that node 1 is gone and connections are removed
        assert_eq!(layer.node_count(), 2);
        assert_eq!(layer.get_neighbors(2), Vec::<usize>::new());
        assert_eq!(layer.get_neighbors(3), Vec::<usize>::new());
    }

    #[test]
    fn test_average_degree() {
        let layer = Layer::new(0, 16);
        
        // Empty layer
        assert_eq!(layer.average_degree(), 0.0);
        
        // Add some connections
        layer.add_connection(1, 2).unwrap();
        layer.add_connection(1, 3).unwrap();
        layer.add_connection(2, 3).unwrap();
        
        // Average degree: (2 + 2 + 2) / 3 = 2.0
        assert_eq!(layer.average_degree(), 2.0);
    }
}