//! Evaluation metrics for vector search
//! 
//! Implements standard information retrieval metrics for evaluating
//! search quality including Recall@K, nDCG@K, and MRR.

use std::collections::HashSet;
use ordered_float::OrderedFloat;

/// Evaluation metrics for search results
pub struct Metrics {
    /// Recall@K values for different K
    pub recall_at_k: Vec<f32>,
    /// nDCG@K values for different K  
    pub ndcg_at_k: Vec<f32>,
    /// Mean Reciprocal Rank
    pub mrr: f32,
    /// Queries Per Second
    pub qps: f32,
    /// Average latency in milliseconds
    pub avg_latency_ms: f32,
    /// Memory usage in GB
    pub memory_gb: f32,
}

impl Metrics {
    /// Create empty metrics
    pub fn new() -> Self {
        Self {
            recall_at_k: Vec::new(),
            ndcg_at_k: Vec::new(),
            mrr: 0.0,
            qps: 0.0,
            avg_latency_ms: 0.0,
            memory_gb: 0.0,
        }
    }
    
    /// Compute recall@K for a single query
    /// 
    /// # Arguments
    /// * `predicted` - Predicted result IDs (ordered by relevance)
    /// * `ground_truth` - Ground truth relevant IDs
    /// * `k` - Cut-off position
    /// 
    /// # Returns
    /// Recall@K as f32
    pub fn recall_at_k(
        predicted: &[usize], 
        ground_truth: &HashSet<usize>, 
        k: usize
    ) -> f32 {
        if ground_truth.is_empty() {
            return 0.0;
        }
        
        let k = k.min(predicted.len());
        let relevant_found = predicted[..k]
            .iter()
            .filter(|&id| ground_truth.contains(id))
            .count();
        
        relevant_found as f32 / ground_truth.len() as f32
    }
    
    /// Compute nDCG@K for a single query
    /// 
    /// # Arguments
    /// * `predicted` - Predicted result IDs (ordered by relevance)
    /// * `relevance_scores` - Map from ID to relevance score
    /// * `k` - Cut-off position
    /// 
    /// # Returns
    /// nDCG@K as f32
    pub fn ndcg_at_k(
        predicted: &[usize],
        relevance_scores: &std::collections::HashMap<usize, f32>,
        k: usize,
    ) -> f32 {
        let k = k.min(predicted.len());
        
        // Compute DCG@K
        let dcg = predicted[..k]
            .iter()
            .enumerate()
            .map(|(i, &id)| {
                let relevance = relevance_scores.get(&id).unwrap_or(&0.0);
                relevance / ((i + 1) as f32).log2()
            })
            .sum::<f32>();
        
        // Compute IDCG@K (ideal DCG)
        let mut ideal_relevance: Vec<_> = relevance_scores
            .values()
            .copied()
            .filter(|&r| r > 0.0)
            .collect();
        ideal_relevance.sort_by(|a, b| b.partial_cmp(a).unwrap());
        
        if ideal_relevance.is_empty() {
            return 0.0;
        }
        
        let idcg = ideal_relevance
            .iter()
            .take(k)
            .enumerate()
            .map(|(i, &relevance)| relevance / ((i + 1) as f32).log2())
            .sum::<f32>();
        
        if idcg == 0.0 {
            0.0
        } else {
            dcg / idcg
        }
    }
    
    /// Compute Mean Reciprocal Rank (MRR)
    /// 
    /// # Arguments
    /// * `predicted` - Predicted result IDs (ordered by relevance)
    /// * `ground_truth` - Ground truth relevant IDs
    /// 
    /// # Returns
    /// MRR as f32
    pub fn reciprocal_rank(
        predicted: &[usize],
        ground_truth: &HashSet<usize>,
    ) -> f32 {
        for (i, &id) in predicted.iter().enumerate() {
            if ground_truth.contains(&id) {
                return 1.0 / (i + 1) as f32;
            }
        }
        0.0
    }
    
    /// Compute metrics for multiple queries
    /// 
    /// # Arguments
    /// * `results` - Search results for each query
    /// * `ground_truth` - Ground truth relevance for each query
    /// * `k_values` - K values to evaluate
    /// * `query_times` - Query execution times in seconds
    /// 
    /// # Returns
    /// Computed metrics
    pub fn compute(
        results: &[Vec<usize>],
        ground_truth: &[HashSet<usize>],
        k_values: &[usize],
        query_times: &[f64],
    ) -> Self {
        assert_eq!(results.len(), ground_truth.len());
        assert_eq!(results.len(), query_times.len());
        
        let num_queries = results.len();
        let mut metrics = Self::new();
        
        // Compute Recall@K and nDCG@K for each K
        for &k in k_values {
            let mut total_recall = 0.0;
            let mut total_ndcg = 0.0;
            
            for (result, gt) in results.iter().zip(ground_truth.iter()) {
                total_recall += Self::recall_at_k(result, gt, k);
                total_ndcg += Self::ndcg_at_k(result, &std::collections::HashMap::new(), k);
            }
            
            metrics.recall_at_k.push(total_recall / num_queries as f32);
            metrics.ndcg_at_k.push(total_ndcg / num_queries as f32);
        }
        
        // Compute MRR
        let mut total_rr = 0.0;
        for (result, gt) in results.iter().zip(ground_truth.iter()) {
            total_rr += Self::reciprocal_rank(result, gt);
        }
        metrics.mrr = total_rr / num_queries as f32;
        
        // Compute QPS and latency
        let total_time: f64 = query_times.iter().sum();
        metrics.qps = num_queries as f32 / total_time as f32;
        metrics.avg_latency_ms = (total_time / num_queries as f64 * 1000.0) as f32;
        
        metrics
    }
    
    /// Print metrics summary
    pub fn print_summary(&self, k_values: &[usize]) {
        println!("=== Search Performance Metrics ===");
        
        for (i, &k) in k_values.iter().enumerate() {
            if i < self.recall_at_k.len() {
                println!("Recall@{}: {:.4}", k, self.recall_at_k[i]);
            }
            if i < self.ndcg_at_k.len() {
                println!("nDCG@{}: {:.4}", k, self.ndcg_at_k[i]);
            }
        }
        
        println!("MRR: {:.4}", self.mrr);
        println!("QPS: {:.2}", self.qps);
        println!("Avg Latency: {:.2} ms", self.avg_latency_ms);
        println!("Memory Usage: {:.2} GB", self.memory_gb);
    }
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{HashSet, HashMap};

    #[test]
    fn test_recall_at_k() {
        let predicted = vec![1, 2, 3, 4, 5];
        let ground_truth: HashSet<usize> = [2, 4, 6].iter().cloned().collect();
        
        // Recall@1: 0/3 = 0.0
        assert_eq!(Metrics::recall_at_k(&predicted, &ground_truth, 1), 0.0);
        
        // Recall@3: 2/3 = 0.667
        let recall = Metrics::recall_at_k(&predicted, &ground_truth, 3);
        assert!((recall - 0.6667).abs() < 0.001);
        
        // Recall@5: 2/3 = 0.667
        let recall = Metrics::recall_at_k(&predicted, &ground_truth, 5);
        assert!((recall - 0.6667).abs() < 0.001);
    }

    #[test]
    fn test_reciprocal_rank() {
        let predicted = vec![1, 2, 3, 4, 5];
        let ground_truth: HashSet<usize> = [2, 4, 6].iter().cloned().collect();
        
        // First relevant at position 2 (1-indexed): 1/2 = 0.5
        let rr = Metrics::reciprocal_rank(&predicted, &ground_truth);
        assert_eq!(rr, 0.5);
        
        // No relevant results
        let empty_gt: HashSet<usize> = HashSet::new();
        let rr = Metrics::reciprocal_rank(&predicted, &empty_gt);
        assert_eq!(rr, 0.0);
    }

    #[test]
    fn test_ndcg_at_k() {
        let predicted = vec![1, 2, 3, 4, 5];
        let mut relevance_scores = HashMap::new();
        relevance_scores.insert(1, 3.0);
        relevance_scores.insert(2, 2.0);
        relevance_scores.insert(3, 1.0);
        
        let ndcg = Metrics::ndcg_at_k(&predicted, &relevance_scores, 3);
        assert!(ndcg > 0.0 && ndcg <= 1.0);
    }

    #[test]
    fn test_compute_metrics() {
        let results = vec![
            vec![1, 2, 3, 4, 5],
            vec![2, 1, 4, 3, 5],
        ];
        let ground_truth = vec![
            [1, 3].iter().cloned().collect(),
            [2, 4].iter().cloned().collect(),
        ];
        let query_times = vec![0.1, 0.2];
        let k_values = vec![1, 3, 5];
        
        let metrics = Metrics::compute(&results, &ground_truth, &k_values, &query_times);
        
        assert_eq!(metrics.recall_at_k.len(), 3);
        assert_eq!(metrics.ndcg_at_k.len(), 3);
        assert!(metrics.mrr > 0.0);
        assert!(metrics.qps > 0.0);
        assert!(metrics.avg_latency_ms > 0.0);
    }
}