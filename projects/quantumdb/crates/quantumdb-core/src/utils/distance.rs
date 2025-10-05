//! SIMD-optimized distance computation
//! 
//! Implements asymmetric distance computation with lookup tables
//! for efficient vector search in compressed space.

use std::simd::num::SimdFloat;
use std::simd::*;
use crate::config;

/// SIMD-optimized distance computation for compressed vectors
/// 
/// Uses precomputed lookup tables for asymmetric distance computation:
/// - Query: full precision (768d float32)
/// - Database: compressed (16 uint8 codes)
/// - Distance: computed via table lookup (O(1) per subvector)
#[derive(Debug, Clone)]
pub struct SIMDDistance {
    /// Precomputed lookup table: [n_subvectors × codebook_size]
    lookup_table: Vec<f32>,
    /// Number of subvectors
    n_subvectors: usize,
    /// Codebook size
    codebook_size: usize,
}

impl SIMDDistance {
    /// Create a new SIMD distance computer
    /// 
    /// # Arguments
    /// * `n_subvectors` - Number of subvectors (default: 16)
    /// * `codebook_size` - Size of each codebook (default: 256)
    pub fn new(n_subvectors: usize, codebook_size: usize) -> Self {
        Self {
            lookup_table: Vec::with_capacity(n_subvectors * codebook_size),
            n_subvectors,
            codebook_size,
        }
    }
    
    /// Build lookup table for a query
    /// 
    /// Precomputes distances from query subvectors to all codebook entries.
    /// This enables O(1) distance computation during search.
    /// 
    /// # Arguments
    /// * `query_subvectors` - Query split into subvectors [n_subvectors][subvector_dim]
    /// * `codebooks` - PQ codebooks [n_subvectors][codebook_size][subvector_dim]
    pub fn build_lookup_table(
        &mut self,
        query_subvectors: &[[f32; 16]], // Assuming 16d subvectors
        codebooks: &[[[f32; 16]; 256]], // Assuming 256 codes per subvector
    ) {
        self.lookup_table.clear();
        
        for (subvec_idx, query_subvec) in query_subvectors.iter().enumerate() {
            for code_idx in 0..self.codebook_size {
                let codebook_entry = &codebooks[subvec_idx][code_idx];
                let dist = l2_distance_simd(query_subvec, codebook_entry);
                self.lookup_table.push(dist);
            }
        }
    }
    
    /// Compute distance to a compressed vector
    /// 
    /// # Arguments
    /// * `codes` - Compressed codes [n_subvectors]
    /// 
    /// # Returns
    /// L2 distance as f32
    #[inline]
    pub fn compute(&self, codes: &[u8]) -> f32 {
        let mut sum = 0.0f32;
        
        for (i, &code) in codes.iter().enumerate() {
            let table_idx = i * self.codebook_size + code as usize;
            sum += self.lookup_table[table_idx];
        }
        
        sum
    }
    
    /// Compute distances for multiple vectors in parallel
    /// 
    /// # Arguments
    /// * `codes_batch` - Batch of compressed codes [batch_size][n_subvectors]
    /// 
    /// # Returns
    /// Vector of distances
    pub fn compute_batch(&self, codes_batch: &[[u8; 16]]) -> Vec<f32> {
        use rayon::prelude::*;
        
        codes_batch
            .par_iter()
            .map(|codes| self.compute(codes))
            .collect()
    }
    
    /// Compute distance between two compressed vectors
    /// 
    /// # Arguments
    /// * `codes1` - First compressed codes
    /// * `codes2` - Second compressed codes
    /// 
    /// # Returns
    /// L2 distance between compressed representations
    pub fn compute_compressed_distance(&self, codes1: &[u8], codes2: &[u8]) -> f32 {
        codes1
            .iter()
            .zip(codes2.iter())
            .map(|(&c1, &c2)| {
                let idx1 = c1 as usize;
                let idx2 = c2 as usize;
                // Simple L2 distance in code space
                (idx1 as f32 - idx2 as f32).powi(2)
            })
            .sum()
    }
}

/// SIMD-optimized L2 distance computation
/// 
/// # Arguments
/// * `a` - First vector (must be same length as b)
/// * `b` - Second vector (must be same length as a)
/// 
/// # Returns
/// L2 distance as f32
#[inline]
fn l2_distance_simd(a: &[f32; 16], b: &[f32; 16]) -> f32 {
    let a_simd = f32x16::from_array(*a);
    let b_simd = f32x16::from_array(*b);
    let diff = a_simd - b_simd;
    let squared = diff * diff;
    squared.reduce_sum()
}

/// Standard L2 distance computation (fallback)
/// 
/// # Arguments
/// * `a` - First vector
/// * `b` - Second vector
/// 
/// # Returns
/// L2 distance as f32
pub fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum()
}

/// Cosine similarity computation
/// 
/// # Arguments
/// * `a` - First vector
/// * `b` - Second vector
/// 
/// # Returns
/// Cosine similarity as f32 (range: [-1, 1])
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
}

/// Inner product computation
/// 
/// # Arguments
/// * `a` - First vector
/// * `b` - Second vector
/// 
/// # Returns
/// Inner product as f32
pub fn inner_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l2_distance_simd() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];
        let b = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0];
        
        let simd_dist = l2_distance_simd(&a, &b);
        let standard_dist = l2_distance(&a, &b);
        
        assert!((simd_dist - standard_dist).abs() < 1e-6);
    }

    #[test]
    fn test_simd_distance() {
        let mut simd_dist = SIMDDistance::new(16, 256);
        
        // Mock query subvectors and codebooks
        let query_subvectors: Vec<[f32; 16]> = (0..16)
            .map(|i| [i as f32; 16])
            .collect();
        
        let codebooks: Vec<[[f32; 16]; 256]> = (0..16)
            .map(|_| {
                (0..256)
                    .map(|j| [j as f32; 16])
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap()
            })
            .collect();
        
        simd_dist.build_lookup_table(&query_subvectors, &codebooks);
        
        // Test distance computation
        let codes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let distance = simd_dist.compute(&codes);
        
        assert!(distance >= 0.0);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = [1.0, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);
        
        let c = [0.0, 1.0, 0.0];
        assert!((cosine_similarity(&a, &c) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_inner_product() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        assert_eq!(inner_product(&a, &b), 32.0);
    }
}