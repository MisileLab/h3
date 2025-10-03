//! Distance computation utilities for HNSW

use crate::utils::SIMDDistance;

/// Distance metrics for vector comparison
#[derive(Debug, Clone, Copy)]
pub enum DistanceMetric {
    /// Euclidean distance (L2)
    L2,
    /// Inner product
    InnerProduct,
    /// Cosine similarity
    Cosine,
}

/// Distance computer for HNSW search
pub struct HNSWDistance {
    /// SIMD distance computer
    simd_distance: SIMDDistance,
    /// Distance metric
    metric: DistanceMetric,
}

impl HNSWDistance {
    /// Create a new distance computer
    /// 
    /// # Arguments
    /// * `metric` - Distance metric to use
    /// * `n_subvectors` - Number of subvectors
    /// * `codebook_size` - Codebook size
    pub fn new(metric: DistanceMetric, n_subvectors: usize, codebook_size: usize) -> Self {
        Self {
            simd_distance: SIMDDistance::new(n_subvectors, codebook_size),
            metric,
        }
    }
    
    /// Compute distance between two compressed vectors
    /// 
    /// # Arguments
    /// * `a` - First compressed vector
    /// * `b` - Second compressed vector
    /// 
    /// # Returns
    /// Distance according to the configured metric
    pub fn compute(&self, a: &[u8], b: &[u8]) -> f32 {
        match self.metric {
            DistanceMetric::L2 => self.simd_distance.compute_compressed_distance(a, b),
            DistanceMetric::InnerProduct => {
                // For compressed vectors, use a simple inner product approximation
                a.iter().zip(b.iter()).map(|(x, y)| (*x as f32) * (*y as f32)).sum()
            }
            DistanceMetric::Cosine => {
                // Cosine similarity for compressed vectors
                let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| (*x as f32) * (*y as f32)).sum();
                let norm_a: f32 = a.iter().map(|x| (*x as f32) * (*x as f32)).sum::<f32>().sqrt();
                let norm_b: f32 = b.iter().map(|x| (*x as f32) * (*x as f32)).sum::<f32>().sqrt();
                
                if norm_a == 0.0 || norm_b == 0.0 {
                    0.0
                } else {
                    dot_product / (norm_a * norm_b)
                }
            }
        }
    }
    
    /// Get the distance metric
    pub fn metric(&self) -> DistanceMetric {
        self.metric
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distance_creation() {
        let distance = HNSWDistance::new(DistanceMetric::L2, 16, 256);
        assert!(matches!(distance.metric(), DistanceMetric::L2));
    }

    #[test]
    fn test_l2_distance() {
        let distance = HNSWDistance::new(DistanceMetric::L2, 16, 256);
        
        let a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let b = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        
        let dist = distance.compute(&a, &b);
        assert!(dist >= 0.0);
    }

    #[test]
    fn test_inner_product() {
        let distance = HNSWDistance::new(DistanceMetric::InnerProduct, 16, 256);
        
        let a = [1, 2, 3, 4];
        let b = [2, 3, 4, 5];
        
        let ip = distance.compute(&a, &b);
        assert_eq!(ip, 1.0*2.0 + 2.0*3.0 + 3.0*4.0 + 4.0*5.0);
    }

    #[test]
    fn test_cosine_similarity() {
        let distance = HNSWDistance::new(DistanceMetric::Cosine, 16, 256);
        
        let a = [1, 0, 0];
        let b = [1, 0, 0];
        
        let sim = distance.compute(&a, &b);
        assert!((sim - 1.0).abs() < 1e-6);
        
        let c = [0, 1, 0];
        let sim = distance.compute(&a, &c);
        assert!((sim - 0.0).abs() < 1e-6);
    }
}