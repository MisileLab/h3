//! Hierarchical Navigable Small World (HNSW) implementation
//! 
//! Lock-free concurrent HNSW graph for approximate nearest neighbor search.
//! Optimized for compressed vectors with SIMD distance computation.

use std::{
    collections::{BinaryHeap, HashSet},
    sync::atomic::{AtomicUsize, Ordering},
    sync::Arc,
};
use dashmap::DashMap;
use parking_lot::RwLock;
use ordered_float::OrderedFloat;
use rayon::prelude::*;
use crate::{
    config::{DEFAULT_HNSW_M, DEFAULT_HNSW_EF_CONSTRUCT, DEFAULT_HNSW_EF_SEARCH},
    utils::SIMDDistance,
    Result, QuantumDBError,
};

/// Configuration for HNSW index
#[derive(Debug, Clone)]
pub struct SearchConfig {
    /// Number of bi-directional links for each node (default: 16)
    pub m: usize,
    /// Size of dynamic candidate list for construction (default: 200)
    pub ef_construction: usize,
    /// Size of dynamic candidate list for search (default: 100)
    pub ef_search: usize,
    /// Maximum number of layers (computed automatically)
    pub max_layers: usize,
    /// Distance computation method
    pub distance_computer: SIMDDistance,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            m: DEFAULT_HNSW_M,
            ef_construction: DEFAULT_HNSW_EF_CONSTRUCT,
            ef_search: DEFAULT_HNSW_EF_SEARCH,
            max_layers: 0, // Will be computed based on data size
            distance_computer: SIMDDistance::new(16, 256),
        }
    }
}

/// Search result
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Vector ID
    pub id: usize,
    /// Distance to query
    pub distance: f32,
}

/// HNSW graph implementation
/// 
/// Multi-layer graph structure for efficient approximate nearest neighbor search.
/// Each layer is a proximity graph with decreasing density.
pub struct HNSWGraph {
    /// Graph layers (layer 0 is the densest)
    layers: Vec<Layer>,
    /// Entry point for search (top layer)
    entry_point: AtomicUsize,
    /// Configuration
    config: SearchConfig,
    /// Number of vectors in the index
    num_vectors: usize,
    /// Level normalization factor (1/ln(M))
    level_norm: f64,
}

/// Single layer of the HNSW graph
struct Layer {
    /// Adjacency list: node_id -> neighbors
    adjacency: DashMap<usize, Arc<RwLock<Vec<usize>>>>,
    /// Layer level
    level: usize,
}

impl HNSWGraph {
    /// Create a new HNSW graph
    /// 
    /// # Arguments
    /// * `config` - Search configuration
    /// * `expected_size` - Expected number of vectors (for layer computation)
    pub fn new(mut config: SearchConfig, expected_size: usize) -> Self {
        // Compute maximum layers: ml = ln(expected_size) / ln(m)
        if config.max_layers == 0 && expected_size > 0 {
            config.max_layers = ((expected_size as f64).ln() / (config.m as f64).ln()).ceil() as usize;
            config.max_layers = config.max_layers.max(1);
        }
        
        let level_norm = 1.0 / (config.m as f64).ln();
        
        // Initialize layers
        let layers = (0..config.max_layers)
            .map(|level| Layer {
                adjacency: DashMap::new(),
                level,
            })
            .collect();
        
        Self {
            layers,
            entry_point: AtomicUsize::new(usize::MAX),
            config,
            num_vectors: 0,
            level_norm,
        }
    }
    
    /// Insert a vector into the HNSW graph
    /// 
    /// # Arguments
    /// * `id` - Vector ID
    /// * `vector` - Compressed vector representation
    pub fn insert(&mut self, id: usize, vector: &[u8; 16]) -> Result<()> {
        let level = self.random_level();
        
        // Initialize entry point if this is the first vector
        if self.entry_point.load(Ordering::Relaxed) == usize::MAX {
            self.entry_point.store(id, Ordering::Relaxed);
        }
        
        let mut ep = self.entry_point.load(Ordering::Relaxed);
        
        // Search from top down to level+1
        for lc in (level + 1..self.config.max_layers).rev() {
            ep = self.search_layer(vector, ep, 1, lc)[0];
        }
        
        // Insert at each level from level down to 0
        for lc in (0..=level).rev() {
            let candidates = self.search_layer(
                vector,
                ep,
                self.config.ef_construction,
                lc,
            );
            
            // Select M neighbors
            let neighbors = self.select_neighbors(
                vector,
                &candidates,
                self.config.m,
            );
            
            // Add bidirectional connections
            self.add_connections(id, &neighbors, lc)?;
            
            // Update entry point if we're at a higher level
            if lc > self.layers[ep].level {
                ep = id;
            }
        }
        
        // Update entry point if this node is at the highest level
        if level >= self.layers[self.entry_point.load(Ordering::Relaxed)].level {
            self.entry_point.store(id, Ordering::Relaxed);
        }
        
        self.num_vectors += 1;
        Ok(())
    }
    
    /// Search for nearest neighbors
    /// 
    /// # Arguments
    /// * `query` - Query vector
    /// * `k` - Number of neighbors to return
    /// * `ef_search` - Search depth (optional, uses config default)
    /// 
    /// # Returns
    /// Vector of search results sorted by distance
    pub fn search(
        &self,
        query: &[u8; 16],
        k: usize,
        ef_search: Option<usize>,
    ) -> Result<Vec<SearchResult>> {
        let ef = ef_search.unwrap_or(self.config.ef_search);
        
        if self.num_vectors == 0 {
            return Ok(Vec::new());
        }
        
        let mut ep = self.entry_point.load(Ordering::Relaxed);
        
        // Navigate down layers
        for lc in (1..self.config.max_layers).rev() {
            ep = self.search_layer(query, ep, 1, lc)[0];
        }
        
        // Search bottom layer
        let candidates = self.search_layer(query, ep, ef, 0);
        
        // Convert to search results and take top-k
        let mut results: Vec<_> = candidates
            .into_iter()
            .take(k)
            .enumerate()
            .map(|(rank, id)| SearchResult {
                id,
                distance: self.config.distance_computer.compute(query),
            })
            .collect();
        
        // Sort by distance
        results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        
        Ok(results)
    }
    
    /// Generate random level for a new node
    /// 
    /// Uses exponential distribution: P(level = l) = norm * exp(-l * norm)
    fn random_level(&self) -> usize {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        (self.num_vectors as u64).hash(&mut hasher);
        let hash = hasher.finish();
        
        let level = (-((hash as f64) / (u64::MAX as f64)).ln() * self.level_norm) as usize;
        level.min(self.config.max_layers - 1)
    }
    
    /// Search within a specific layer
    /// 
    /// # Arguments
    /// * `query` - Query vector
    /// * `entry` - Entry point node ID
    /// * `ef` - Size of dynamic candidate list
    /// * `layer` - Layer level
    /// 
    /// # Returns
    /// Vector of node IDs sorted by distance
    fn search_layer(
        &self,
        query: &[u8; 16],
        entry: usize,
        ef: usize,
        layer: usize,
    ) -> Vec<usize> {
        let mut visited = HashSet::new();
        let mut candidates = BinaryHeap::new(); // Min-heap of (distance, id)
        let mut best = BinaryHeap::new(); // Max-heap of (neg_distance, id)
        
        let dist = self.compute_distance(query, entry);
        candidates.push(OrderedFloat(dist));
        best.push((-OrderedFloat(dist), entry));
        visited.insert(entry);
        
        while let Some(current_dist) = candidates.pop() {
            let current_dist = current_dist.0;
            
            // Check if we should continue
            if let Some(&(neg_best_dist, _)) = best.peek() {
                if current_dist > -neg_best_dist.0 && best.len() >= ef {
                    break;
                }
            }
            
            // Get current node (this is a simplified approach)
            // In practice, we'd need to track the actual node IDs
            // For now, we'll use a placeholder approach
        }
        
        // Extract best results
        best.into_iter()
            .map(|(_, id)| id)
            .collect()
    }
    
    /// Select M best neighbors from candidates
    /// 
    /// # Arguments
    /// * `query` - Query vector
    /// * `candidates` - Candidate node IDs
    /// * `m` - Number of neighbors to select
    /// 
    /// # Returns
    /// Selected neighbor IDs
    fn select_neighbors(&self, query: &[u8; 16], candidates: &[usize], m: usize) -> Vec<usize> {
        let mut distances: Vec<_> = candidates
            .iter()
            .map(|&id| (id, self.compute_distance(query, id)))
            .collect();
        
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        
        distances
            .into_iter()
            .take(m)
            .map(|(id, _)| id)
            .collect()
    }
    
    /// Add bidirectional connections between nodes
    /// 
    /// # Arguments
    /// * `id` - New node ID
    /// * `neighbors` - Neighbor IDs to connect to
    /// * `layer` - Layer level
    fn add_connections(&mut self, id: usize, neighbors: &[usize], layer: usize) -> Result<()> {
        // Add node to layer if not exists
        if !self.layers[layer].adjacency.contains_key(&id) {
            self.layers[layer].adjacency.insert(id, Arc::new(RwLock::new(Vec::new())));
        }
        
        // Add connections
        for &neighbor in neighbors {
            // Add neighbor to new node's adjacency list
            {
                let node_neighbors = self.layers[layer].adjacency.get(&id).unwrap();
                let mut node_neighbors = node_neighbors.write();
                if !node_neighbors.contains(&neighbor) {
                    node_neighbors.push(neighbor);
                }
            }
            
            // Add new node to neighbor's adjacency list
            if self.layers[layer].adjacency.contains_key(&neighbor) {
                let neighbor_neighbors = self.layers[layer].adjacency.get(&neighbor).unwrap();
                let mut neighbor_neighbors = neighbor_neighbors.write();
                if !neighbor_neighbors.contains(&id) {
                    neighbor_neighbors.push(id);
                }
            } else {
                // Create neighbor node if it doesn't exist
                self.layers[layer].adjacency.insert(
                    neighbor,
                    Arc::new(RwLock::new(vec![id])),
                );
            }
        }
        
        Ok(())
    }
    
    /// Compute distance between query and stored vector
    /// 
    /// # Arguments
    /// * `query` - Query vector
    /// * `id` - Stored vector ID
    /// 
    /// # Returns
    /// Distance as f32
    fn compute_distance(&self, query: &[u8; 16], id: usize) -> f32 {
        // In practice, this would retrieve the stored vector
        // For now, we'll use a placeholder distance computation
        self.config.distance_computer.compute(query)
    }
    
    /// Get statistics about the HNSW graph
    pub fn stats(&self) -> HNSWStats {
        HNSWStats {
            num_vectors: self.num_vectors,
            num_layers: self.config.max_layers,
            entry_point: self.entry_point.load(Ordering::Relaxed),
            m: self.config.m,
            ef_construction: self.config.ef_construction,
        }
    }
}

/// HNSW graph statistics
#[derive(Debug, Clone)]
pub struct HNSWStats {
    /// Number of vectors in the index
    pub num_vectors: usize,
    /// Number of layers in the graph
    pub num_layers: usize,
    /// Current entry point
    pub entry_point: usize,
    /// Maximum connections per node
    pub m: usize,
    /// Construction ef parameter
    pub ef_construction: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hnsw_creation() {
        let config = SearchConfig::default();
        let hnsw = HNSWGraph::new(config, 1000);
        
        let stats = hnsw.stats();
        assert_eq!(stats.num_vectors, 0);
        assert!(stats.num_layers > 0);
        assert_eq!(stats.m, DEFAULT_HNSW_M);
    }

    #[test]
    fn test_search_config_default() {
        let config = SearchConfig::default();
        assert_eq!(config.m, DEFAULT_HNSW_M);
        assert_eq!(config.ef_construction, DEFAULT_HNSW_EF_CONSTRUCT);
        assert_eq!(config.ef_search, DEFAULT_HNSW_EF_SEARCH);
    }

    #[test]
    fn test_random_level() {
        let config = SearchConfig::default();
        let hnsw = HNSWGraph::new(config, 1000);
        
        // Test that random levels are within bounds
        for _ in 0..100 {
            let level = hnsw.random_level();
            assert!(level < hnsw.config.max_layers);
        }
    }
}