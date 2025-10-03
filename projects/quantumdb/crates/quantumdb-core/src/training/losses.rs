//! Loss functions for training

use burn::{
    tensor::{backend::Backend, Tensor},
};

/// Loss components for compression training
#[derive(Debug)]
pub struct CompressionLoss<B: Backend> {
    /// Reconstruction loss
    pub reconstruction: Tensor<B, 1>,
    /// Codebook diversity loss
    pub diversity: Tensor<B, 1>,
    /// Commitment loss
    pub commitment: Tensor<B, 1>,
    /// Local structure preservation loss
    pub local_structure: Tensor<B, 1>,
    /// Total weighted loss
    pub total: Tensor<B, 1>,
}

impl<B: Backend> CompressionLoss<B> {
    /// Compute all loss components
    /// 
    /// # Arguments
    /// * `input` - Original embeddings
    /// * `reconstructed` - Reconstructed embeddings
    /// * `soft_codes` - Soft quantization codes
    /// * `hard_codes` - Hard quantization codes
    /// * `weights` - Loss weights
    pub fn compute(
        input: &Tensor<B, 2>,
        reconstructed: &Tensor<B, 2>,
        soft_codes: &Tensor<B, 3>,
        hard_codes: &Tensor<B, 2>,
        weights: &LossWeights,
    ) -> Self {
        // Reconstruction loss
        let reconstruction = Self::reconstruction_loss(input, reconstructed);
        
        // Codebook diversity loss
        let diversity = Self::diversity_loss(soft_codes);
        
        // Commitment loss
        let commitment = Self::commitment_loss(input, reconstructed);
        
        // Local structure preservation loss
        let local_structure = Self::local_structure_loss(input, hard_codes);
        
        // Weighted total loss
        let total = reconstruction.clone() * weights.reconstruction
            + diversity.clone() * weights.diversity
            + commitment.clone() * weights.commitment
            + local_structure.clone() * weights.local_structure;
        
        Self {
            reconstruction,
            diversity,
            commitment,
            local_structure,
            total,
        }
    }
    
    /// Reconstruction loss: ||x - decode(encode(x))||²
    fn reconstruction_loss(input: &Tensor<B, 2>, reconstructed: &Tensor<B, 2>) -> Tensor<B, 1> {
        (input.clone() - reconstructed.clone())
            .pow_scalar(2.0)
            .mean()
    }
    
    /// Codebook diversity loss (entropy regularization)
    fn diversity_loss(soft_codes: &Tensor<B, 3>) -> Tensor<B, 1> {
        let mean_codes = soft_codes.clone().mean_dim(0);
        let entropy = -(mean_codes.clone() * (mean_codes + 1e-10).log()).sum();
        -entropy // Maximize entropy = minimize negative entropy
    }
    
    /// Commitment loss (VQ-VAE style)
    fn commitment_loss(input: &Tensor<B, 2>, reconstructed: &Tensor<B, 2>) -> Tensor<B, 1> {
        (input.clone() - reconstructed.clone())
            .pow_scalar(2.0)
            .mean()
    }
    
    /// Local structure preservation loss
    fn local_structure_loss(input: &Tensor<B, 2>, hard_codes: &Tensor<B, 2>) -> Tensor<B, 1> {
        // Simplified triplet loss for local structure preservation
        let batch_size = input.dims()[0];
        
        if batch_size < 3 {
            return Tensor::from_floats([0.0], &input.device());
        }
        
        // Create anchor, positive, negative samples
        let anchor = input.clone().slice([0..batch_size-2]);
        let positive = input.clone().slice([1..batch_size-1]);
        let negative = input.clone().slice([2..batch_size]);
        
        let anchor_codes = hard_codes.clone().slice([0..batch_size-2]);
        let positive_codes = hard_codes.clone().slice([1..batch_size-1]);
        let negative_codes = hard_codes.clone().slice([2..batch_size]);
        
        // Compute distances in code space
        let pos_dist = Self::code_distance(&anchor_codes, &positive_codes);
        let neg_dist = Self::code_distance(&anchor_codes, &negative_codes);
        
        // Triplet loss: max(0, pos_dist - neg_dist + margin)
        let margin = Tensor::from_floats([1.0], &input.device());
        (pos_dist - neg_dist + margin).clamp_min(0.0).mean()
    }
    
    /// Compute distance between code sequences
    fn code_distance(codes1: &Tensor<B, 2>, codes2: &Tensor<B, 2>) -> Tensor<B, 1> {
        (codes1.clone() - codes2.clone())
            .pow_scalar(2.0)
            .sum_dim(1)
            .mean()
    }
}

/// Loss weights for different components
#[derive(Debug, Clone)]
pub struct LossWeights {
    /// Reconstruction loss weight
    pub reconstruction: f32,
    /// Diversity loss weight
    pub diversity: f32,
    /// Commitment loss weight
    pub commitment: f32,
    /// Local structure loss weight
    pub local_structure: f32,
}

impl Default for LossWeights {
    fn default() -> Self {
        Self {
            reconstruction: 0.3,
            diversity: 0.1,
            commitment: 0.1,
            local_structure: 0.3,
        }
    }
}

impl LossWeights {
    /// Create new loss weights
    pub fn new(
        reconstruction: f32,
        diversity: f32,
        commitment: f32,
        local_structure: f32,
    ) -> Self {
        Self {
            reconstruction,
            diversity,
            commitment,
            local_structure,
        }
    }
    
    /// Validate that weights sum to approximately 1.0
    pub fn validate(&self) -> bool {
        let total = self.reconstruction + self.diversity + self.commitment + self.local_structure;
        (total - 1.0).abs() < 0.1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::tensor::TensorData;

    type Backend = NdArray;

    #[test]
    fn test_loss_weights_default() {
        let weights = LossWeights::default();
        
        assert_eq!(weights.reconstruction, 0.3);
        assert_eq!(weights.diversity, 0.1);
        assert_eq!(weights.commitment, 0.1);
        assert_eq!(weights.local_structure, 0.3);
        assert!(weights.validate());
    }

    #[test]
    fn test_loss_weights_validation() {
        let valid_weights = LossWeights::new(0.4, 0.1, 0.1, 0.4);
        assert!(valid_weights.validate());
        
        let invalid_weights = LossWeights::new(0.8, 0.1, 0.1, 0.4);
        assert!(!invalid_weights.validate());
    }

    #[test]
    fn test_reconstruction_loss() {
        let device = Backend::Device::default();
        
        let input_data = TensorData::random::<f32, _>(
            &[2, 4],
            burn::tensor::Distribution::Uniform(0.0, 1.0)
        );
        let input = Tensor::<Backend, 2>::from_data(input_data, &device);
        
        let reconstructed = input.clone() + 0.1;
        let loss = CompressionLoss::reconstruction_loss(&input, &reconstructed);
        
        assert_eq!(loss.dims(), &[]);
        assert!(loss.into_scalar() > 0.0);
    }
}