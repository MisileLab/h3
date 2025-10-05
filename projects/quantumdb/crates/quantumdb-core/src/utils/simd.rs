#![feature(portable_simd)]
//! SIMD utilities and vectorized operations

use std::simd::num::SimdFloat;
use std::simd::*;

/// Vectorized operations for batch processing
pub struct SIMDUtils;

impl SIMDUtils {
    /// Vectorized sum of f32 array
    /// 
    /// # Arguments
    /// * `data` - Input array
    /// 
    /// # Returns
    /// Sum as f32
    pub fn sum_f32(data: &[f32]) -> f32 {
        let chunks = data.chunks_exact(16);
        let remainder = chunks.remainder();
        
        let mut sum = f32x16::splat(0.0);
        for chunk in chunks {
            let vec = f32x16::from_slice(chunk);
            sum += vec;
        }
        
        let mut total = sum.reduce_sum();
        
        // Handle remainder
        for &x in remainder {
            total += x;
        }
        
        total
    }
    
    /// Vectorized L2 norm computation
    /// 
    /// # Arguments
    /// * `data` - Input array
    /// 
    /// # Returns
    /// L2 norm as f32
    pub fn l2_norm(data: &[f32]) -> f32 {
        let chunks = data.chunks_exact(16);
        let remainder = chunks.remainder();
        
        let mut sum_sq = f32x16::splat(0.0);
        for chunk in chunks {
            let vec = f32x16::from_slice(chunk);
            sum_sq += vec * vec;
        }
        
        let mut total = sum_sq.reduce_sum();
        
        // Handle remainder
        for &x in remainder {
            total += x * x;
        }
        
        total.sqrt()
    }
    
    /// Vectorized dot product
    /// 
    /// # Arguments
    /// * `a` - First vector
    /// * `b` - Second vector (must be same length as a)
    /// 
    /// # Returns
    /// Dot product as f32
    pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        
        let chunks_a = a.chunks_exact(16);
        let chunks_b = b.chunks_exact(16);
        let remainder_a = chunks_a.remainder();
        let remainder_b = chunks_b.remainder();
        
        let mut sum = f32x16::splat(0.0);
        for (chunk_a, chunk_b) in chunks_a.zip(chunks_b) {
            let vec_a = f32x16::from_slice(chunk_a);
            let vec_b = f32x16::from_slice(chunk_b);
            sum += vec_a * vec_b;
        }
        
        let mut total = sum.reduce_sum();
        
        // Handle remainder
        for (&x, &y) in remainder_a.iter().zip(remainder_b.iter()) {
            total += x * y;
        }
        
        total
    }
    
    /// Vectorized cosine similarity
    /// 
    /// # Arguments
    /// * `a` - First vector
    /// * `b` - Second vector (must be same length as a)
    /// 
    /// # Returns
    /// Cosine similarity as f32
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot = Self::dot_product(a, b);
        let norm_a = Self::l2_norm(a);
        let norm_b = Self::l2_norm(b);
        
        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }
    
    /// Vectorized array scaling
    /// 
    /// # Arguments
    /// * `data` - Input array
    /// * `scale` - Scale factor
    /// * `output` - Output array (must be same length as input)
    pub fn scale(data: &[f32], scale: f32, output: &mut [f32]) {
        assert_eq!(data.len(), output.len());
        
        let scale_vec = f32x16::splat(scale);
        let chunks = data.chunks_exact(16);
        let remainder = chunks.remainder();
        
        let mut output_chunks = output.chunks_exact_mut(16);
        
        for (chunk, out_chunk) in chunks.zip(&mut output_chunks) {
            let vec = f32x16::from_slice(chunk);
            let scaled = vec * scale_vec;
            scaled.copy_to_slice(out_chunk);
        }
        
        // Handle remainder
        let output_remainder = output_chunks.into_remainder();
        for (i, &x) in remainder.iter().enumerate() {
            output_remainder[i] = x * scale;
        }
    }
    
    /// Vectorized array addition
    /// 
    /// # Arguments
    /// * `a` - First array
    /// * `b` - Second array (must be same length as a)
    /// * `output` - Output array (must be same length as inputs)
    pub fn add(a: &[f32], b: &[f32], output: &mut [f32]) {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), output.len());
        
        let chunks_a = a.chunks_exact(16);
        let chunks_b = b.chunks_exact(16);
        let remainder_a = chunks_a.remainder();
        let remainder_b = chunks_b.remainder();
        
        let mut output_chunks = output.chunks_exact_mut(16);
        
        for ((chunk_a, chunk_b), out_chunk) in chunks_a.zip(chunks_b).zip(&mut output_chunks) {
            let vec_a = f32x16::from_slice(chunk_a);
            let vec_b = f32x16::from_slice(chunk_b);
            let sum = vec_a + vec_b;
            sum.copy_to_slice(out_chunk);
        }
        
        // Handle remainder
        let output_remainder = output_chunks.into_remainder();
        for (i, (&x, &y)) in remainder_a.iter().zip(remainder_b.iter()).enumerate() {
            output_remainder[i] = x + y;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sum_f32() {
        let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let sum = SIMDUtils::sum_f32(&data);
        
        let expected: f32 = (0..100).map(|i| i as f32).sum();
        assert!((sum - expected).abs() < 1e-6);
    }

    #[test]
    fn test_l2_norm() {
        let data = vec![3.0, 4.0];
        let norm = SIMDUtils::l2_norm(&data);
        assert!((norm - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let dot = SIMDUtils::dot_product(&a, &b);
        assert_eq!(dot, 32.0);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let sim = SIMDUtils::cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-6);
        
        let c = vec![0.0, 1.0, 0.0];
        let sim = SIMDUtils::cosine_similarity(&a, &c);
        assert!((sim - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_scale() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0; 4];
        SIMDUtils::scale(&data, 2.0, &mut output);
        assert_eq!(output, vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_add() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let mut output = vec![0.0; 4];
        SIMDUtils::add(&a, &b, &mut output);
        assert_eq!(output, vec![6.0, 8.0, 10.0, 12.0]);
    }
}