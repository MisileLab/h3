

## 1. Project Overview

### 1.1 Problem Statement

**현재 벡터 검색의 문제점:**

1. **메모리 비용 폭발**
   - 10억 벡터 (768d, float32) = 2.86 TB
   - 클라우드 벡터 DB 비용: 월 $5,000+

2. **검색 속도 한계**
   - 고차원 벡터의 거리 계산 비용
   - 인덱스 크기 증가로 캐시 미스 증가

3. **기존 압축 방법의 한계**
   - Scalar quantization: 정확도 손실 큼
   - Standard PQ: 고정된 codebook, task-agnostic
   - Binary quantization: 너무 aggressive한 압축

### 1.2 Solution: QuantumDB

**학습 가능한 신경망 기반 압축:**

```
Input: 768d float32 embedding (3,072 bytes)
         ↓
Neural Encoder: 768d → 256d (learned projection)
         ↓
Subvector Split: 256d → 16 × 16d
         ↓
Learnable PQ: 16 codebooks × 256 codes
         ↓
Output: 16 × uint8 codes (16 bytes)

Compression Ratio: 192×
```

**핵심 혁신:**
- End-to-end 학습으로 검색 품질 직접 최적화
- Unsupervised loss로 레이블 불필요
- 자동 하이퍼파라미터 탐색으로 수동 튜닝 제거

---

## 2. Technical Architecture

### 2.1 System Design

```
┌─────────────────────────────────────────────────────────┐
│                     QuantumDB                            │
├─────────────────────────────────────────────────────────┤
│  Compression Layer                                       │
│  ├─ Neural Encoder (Learnable projection)              │
│  ├─ Learnable Codebooks (16 × 256 × 16d)               │
│  └─ Gumbel-Softmax Quantization                        │
├─────────────────────────────────────────────────────────┤
│  Storage Layer                                           │
│  ├─ Compressed codes (uint8 arrays)                    │
│  ├─ Codebook parameters (small, shared)                │
│  └─ Metadata index                                      │
├─────────────────────────────────────────────────────────┤
│  Graph Index Layer (HNSW)                               │
│  ├─ Hierarchical Navigable Small World Graph           │
│  ├─ Built on compressed space (16-byte codes)          │
│  ├─ Optimized for quantized distance computation       │
│  └─ Parameters: M=16, ef_construct=200                 │
├─────────────────────────────────────────────────────────┤
│  Search Layer                                            │
│  ├─ Asymmetric Distance Computation (ADC)              │
│  ├─ HNSW traversal with lookup tables                  │
│  ├─ Beam search (ef_search tunable)                    │
│  └─ Re-ranking with full precision (optional)          │
└─────────────────────────────────────────────────────────┘
```

### 2.2 Neural Compression Module

**Architecture:**

```python
class QuantumCompressor(nn.Module):
    """
    768d → 256d → 16×16d → 16×uint8
    """
    components:
        - encoder: MLP(768 → 256)
        - codebooks: Parameter(16, 256, 16)
        - quantizer: GumbelSoftmax
```

**Loss Function:**

```
L_total = 0.3 × L_reconstruction
        + 0.2 × L_cluster_quality
        + 0.1 × L_codebook_diversity
        + 0.1 × L_commitment
        + 0.3 × L_local_structure
```

**각 Loss 상세:**

1. **Reconstruction Loss**: `||x - decode(encode(x))||²`
   - 기본 정보 보존

2. **Cluster Quality Loss**: Within-cluster variance 최소화
   - Codebook assignment의 compactness

3. **Codebook Diversity Loss**: Entropy regularization
   - 모든 코드북 엔트리 균등 사용

4. **Commitment Loss**: VQ-VAE 스타일
   - 인코더 출력과 선택된 벡터 정렬

5. **Local Structure Preservation**: k-NN graph 보존
   - Triplet loss로 이웃 관계 유지
   - HNSW 그래프 품질 최적화를 위한 구조 보존

### 2.4 HNSW Integration

**Hierarchical Navigable Small World (HNSW) 개요:**

HNSW는 근사 최근접 이웃(ANN) 검색을 위한 그래프 기반 인덱스로, 계층적 구조를 통해 빠른 탐색을 가능하게 합니다.

**QuantumDB의 HNSW 최적화:**

```
Layer 2 (희소)    •
                 ╱ ╲
Layer 1 (중간)  •───•───•
               ╱│╲ │ ╱│╲
Layer 0 (밀집) •─•─•─•─•─•─•─•
              [All vectors]

특징:
- Compressed 16-byte codes로 그래프 구성
- Greedy search로 logarithmic 복잡도
- M=16: 각 노드당 최대 연결 수
- ef_construct=200: 빌드 시 탐색 깊이
```

**Compressed Space에서의 HNSW 구축:**

```python
1. 압축된 벡터로 distance 계산
   - Asymmetric Distance: query(full) vs db(compressed)
   - Lookup table 활용 (16 × 256 table)
   - SIMD 최적화로 병렬 계산

2. HNSW 파라미터 최적화
   - M: 16 (메모리-속도 균형)
   - ef_construct: 200 (높은 품질)
   - max_layers: log₂(N) (자동 결정)

3. 점진적 인덱스 빌드
   - Batch insertion (10K vectors/batch)
   - 메모리 효율적 구축
   - Progress tracking
```

**검색 알고리즘:**

```
Query Processing Pipeline:
1. Query embedding (768d, full precision)
2. Compute lookup table
   - 16 subqueries × 256 codebook entries
   - Precompute all distances: O(16×256×16)
3. HNSW traversal
   - Start from entry point (top layer)
   - Greedy search down layers
   - ef_search candidates maintained
4. Distance computation
   - Table lookup: O(1) per subvector
   - Total: O(16) per vector
5. Return top-k results

Time Complexity:
- HNSW: O(log N) graph hops
- Distance: O(M) per code (M=16)
- Total: O(M log N) = O(16 log N)
```

**HNSW vs Flat Search:**

| Method | Time Complexity | Recall@10 | QPS (1M vecs) |
|--------|----------------|-----------|---------------|
| Flat (exhaustive) | O(N×M) | 100% | ~100 |
| HNSW (ef=50) | O(M log N) | 99% | ~5,000 |
| HNSW (ef=100) | O(M log N) | 99.5% | ~3,000 |
| HNSW (ef=200) | O(M log N) | 99.8% | ~1,500 |

**QuantumDB 최적화 포인트:**

1. **압축 공간에서 HNSW 직접 구축**
   - 192배 작은 메모리 → 더 많은 벡터 RAM에 캐싱
   - 16-byte 비교 → CPU 캐시 효율성 극대화

2. **Asymmetric Distance Computation (ADC)**
   - Query는 full precision 유지
   - DB는 compressed
   - 정확도 손실 최소화

3. **SIMD Vectorization**
   - AVX-512로 16개 subvector 병렬 처리
   - Lookup table은 cache-aligned

4. **Dynamic ef_search**
   - Latency budget에 따라 자동 조정
   - Adaptive search depth

**HNSW 학습 통합:**

```python
# Loss에 HNSW 품질 반영
L_hnsw_quality = recall_loss_on_validation_queries

# Auto-tuning에서 HNSW 파라미터도 최적화
search_space['hnsw_M'] = [12, 16, 24]
search_space['hnsw_ef_construct'] = [100, 200, 400]
```

### 2.3 Auto-Tuning System

**3-Stage Progressive Search:**

```
Stage 1: Random Search (12시간)
├─ Search space: 300+ configurations
├─ Sample: 30 random configs
├─ Budget: 3 epochs × 10% data
├─ Metric: Reconstruction + Speed
└─ Output: Top 15 configs

Stage 2: Bayesian Optimization (18시간)
├─ Method: Optuna TPE sampler
├─ Sample: 15 trials
├─ Budget: 10 epochs × 50% data
├─ Pruning: MedianPruner at epoch 5
└─ Output: Top 5 configs

Stage 3: Fine-tuning (18시간)
├─ Configs: Top 5 from Stage 2
├─ Budget: 20 epochs × 100% data
├─ Validation: MS MARCO dev (6,980 queries)
└─ Output: Best model + hyperparameters

Total Time: 48 hours on RTX 4090
```

**Search Space:**

| Parameter | Range | Description |
|-----------|-------|-------------|
| target_dim | [192, 256, 320] | Compressed dimension |
| n_subvectors | [12, 16, 24] | PQ segments |
| codebook_size | 256 | Fixed (uint8) |
| learning_rate | [3e-4, 5e-4, 1e-3] | Adam LR |
| batch_size | [256, 512] | Training batch |
| α (recon) | [0.3, 0.4] | Reconstruction weight |
| ε (local) | [0.15, 0.25] | Structure weight |
| k_neighbors | [10, 20] | k-NN graph size |
| **hnsw_M** | **[12, 16, 24]** | **HNSW connections** |
| **hnsw_ef_construct** | **[100, 200, 400]** | **HNSW build quality** |

**Objective Function:**

```python
score = 0.4 × reconstruction_quality 
      + 0.1 × compression_ratio
      + 0.4 × search_speed
      - 0.1 × memory_penalty
```

---

## 3. Implementation Plan

### 3.1 Development Stack

**Unified Rust Stack (Training + Production):**

**Training Framework:**
- **Burn 0.14+** (Deep learning framework)
  - Full training support with autodiff
  - GPU acceleration (CUDA/ROCm/Metal)
  - Distributed training (planned)
  - Memory-efficient gradient computation
- **Optuna (via PyO3 bridge)** - Hyperparameter tuning
  - Or pure Rust alternative: `hyperopt-rs`
- **WandB (via REST API)** - Experiment tracking
  - Or Rust alternative: Custom telemetry

**Model Serialization:**
- **SafeTensors** - Zero-copy throughout entire pipeline
  - Training checkpoints
  - Model weights
  - Codebooks

**Infrastructure:**
- **Tokio** (Async runtime)
- **Rayon** (Data parallelism)
- **ndarray** (NumPy-like arrays)
- **SIMD optimizations** (portable-simd)

**Data Processing:**
- **Arrow/Parquet** (Fast columnar data)
- **tokenizers-rs** (Hugging Face tokenizers)
- **hf-hub** (Download models/datasets)

**Production:**
- **gRPC** (API server)
- **Memory-mapped I/O** (memmap2)
- **Custom HNSW** (Lock-free concurrent)

**Benchmarking:**
- Qdrant (Rust-based comparison)
- Faiss (via FFI)
- Criterion.rs (Micro-benchmarking)

### 3.2 Pure Rust Architecture: Training to Production

**Why Full Rust Stack?**

1. **Single Language**: No Python/Rust 경계 없음
2. **Type Safety**: 컴파일 타임 에러 검출
3. **Performance**: 학습부터 추론까지 최고 속도
4. **Memory Efficiency**: Ownership system으로 메모리 누수 방지
5. **Reproducibility**: 동일 환경, 동일 결과
6. **Deployment**: 단일 바이너리 배포

**Unified Workflow:**

```
┌─────────────────────────────────────────────────────────┐
│ Phase 1: Data Preparation (Rust)                        │
├─────────────────────────────────────────────────────────┤
│ Load MS MARCO (Arrow/Parquet)                           │
│  ├─ Streaming data loader                               │
│  ├─ Tokenization (tokenizers-rs)                        │
│  └─ Embedding generation (Burn + sentence-transformers) │
│                                                           │
│ Export: SafeTensors embeddings (memory-mapped)          │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ Phase 2: Training (Burn)                                │
├─────────────────────────────────────────────────────────┤
│ Model Definition                                         │
│  ├─ NeuralEncoder (Burn Module)                        │
│  ├─ LearnablePQ (Burn Module)                          │
│  └─ Full model with autodiff                            │
│                                                           │
│ Training Loop                                            │
│  ├─ Forward pass (Burn tensors)                        │
│  ├─ Loss computation (5 loss components)                │
│  ├─ Backward pass (autodiff)                            │
│  └─ Optimizer step (AdamW in Burn)                     │
│                                                           │
│ Auto-tuning (hyperopt-rs or Optuna via FFI)            │
│  ├─ 3-stage optimization                                │
│  ├─ Parallel trials (Rayon)                            │
│  └─ Validation on MS MARCO dev                          │
│                                                           │
│ Checkpointing: SafeTensors (every epoch)                │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ Phase 3: Index Building (Pure Rust)                     │
├─────────────────────────────────────────────────────────┤
│ Load trained model (SafeTensors → Burn)                │
│                                                           │
│ Compress all vectors                                     │
│  ├─ Burn inference (no_grad mode)                      │
│  ├─ Batch processing (Rayon parallel)                  │
│  └─ 16-byte codes output                                │
│                                                           │
│ Build HNSW index                                         │
│  ├─ Graph construction (lock-free)                     │
│  ├─ Parallel insertion (Rayon)                         │
│  └─ Memory-mapped persistence                           │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ Phase 4: Serving (Pure Rust)                            │
├─────────────────────────────────────────────────────────┤
│ Load from disk (zero-copy mmap)                         │
│                                                           │
│ Query Processing                                         │
│  ├─ Burn inference (encoder)                           │
│  ├─ SIMD distance computation                           │
│  ├─ HNSW traversal (concurrent)                        │
│  └─ gRPC/HTTP response                                  │
└─────────────────────────────────────────────────────────┘
```

**Burn Training Advantages:**

```rust
// Type-safe tensors (dimensions checked at compile time!)
use burn::prelude::*;

// Define model with compile-time guarantees
#[derive(Module, Debug)]
struct LearnablePQ<B: Backend> {
    // Codebooks: [n_subvectors, codebook_size, subdim]
    codebooks: Param<Tensor<B, 3>>,
    temperature: f32,
}

impl<B: Backend> LearnablePQ<B> {
    // Forward pass with autodiff
    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let batch_size = x.dims()[0];
        let n_sub = self.codebooks.dims()[0];
        
        // Split into subvectors (type-safe reshape)
        let subvectors = x.reshape([batch_size, n_sub, -1]);
        
        // Compute distances to codebooks
        let distances = self.compute_distances(&subvectors);
        
        // Gumbel-Softmax (differentiable!)
        let soft_codes = gumbel_softmax(distances, self.temperature);
        
        // Reconstruct
        let reconstructed = self.reconstruct(soft_codes);
        
        reconstructed // Gradient flows automatically!
    }
}

// Training is just as easy as PyTorch
fn train_epoch<B: Backend>(
    model: &mut LearnablePQ<B>,
    optimizer: &mut AdamW<B>,
    dataloader: &DataLoader,
) -> f32 {
    let mut total_loss = 0.0;
    
    for batch in dataloader.iter() {
        // Forward
        let output = model.forward(batch.embeddings.clone());
        
        // Loss
        let loss = combined_loss(
            &batch.embeddings,
            &output,
            model,
        );
        
        // Backward (automatic!)
        let gradients = loss.backward();
        
        // Update
        optimizer.step(model, gradients);
        
        total_loss += loss.into_scalar();
    }
    
    total_loss / dataloader.len() as f32
}
```

**SafeTensors Throughout:**

```rust
// Training checkpoint
model.save_safetensors("checkpoint_epoch_10.safetensors")?;

// Resume training
let model = LearnablePQ::load_safetensors(
    "checkpoint_epoch_10.safetensors"
)?;

// Production (same format!)
let model = LearnablePQ::load_safetensors(
    "final_model.safetensors"
)?;
```

**Training Data:**
- **MS MARCO Passage**: 8.8M passages
- **Preprocessing**: Pre-compute embeddings with all-mpnet-base-v2
- **Storage**: Memory-mapped .npy arrays (zero-copy)

**Validation Data:**
- **MS MARCO Dev**: 6,980 queries (전체 사용)
- **Purpose**: Hyperparameter selection via Recall@10

**Test Data:**
- **BEIR 13 datasets**: Cross-domain generalization
- **Wikipedia 21M**: Large-scale stress test

**Memory Strategy:**
```
Total embeddings: 8.8M × 768 × 4 bytes = 27 GB
→ Memory-map (no full load)
→ DataLoader with prefetch (2 batches)
→ Pin memory for GPU transfer
```

### 3.4 Timeline (8 Weeks - Extended for Rust)

**Week 1: Infrastructure Setup**
- Day 1-2: Qdrant 설치 및 baseline 측정
- Day 3-4: MS MARCO 다운로드 및 임베딩 생성
- Day 5-7: 데이터 파이프라인 구축
- **Deliverable**: Baseline metrics (QPS, Recall@10)

**Week 2: Core Implementation**
- Day 1-3: LearnablePQ 모듈 구현
- Day 4-5: Loss functions 구현
- Day 6-7: Training loop 및 validation
- **Deliverable**: 첫 모델 학습 완료

**Week 3: Auto-Tuning + HNSW**
- Day 1-2: Optuna integration
- Day 3-5: Stage 1-2 실행 (30시간)
- Day 6-7: HNSW integration 및 최적화
- **Deliverable**: Best hyperparameters + HNSW index

**Week 4: Training & Optimization**
- Day 1-3: Stage 3 fine-tuning (18시간)
- Day 4-5: Model export to SafeTensors
- Day 6-7: Rust project setup, Burn integration
- **Deliverable**: PyTorch model + SafeTensors export

**Week 5: Rust Database Core**
- Day 1-3: Burn model loading, inference pipeline
- Day 4-5: HNSW implementation in Rust
- Day 6-7: SIMD distance computation
- **Deliverable**: Working Rust indexing

**Week 6: Rust API & Optimization**
- Day 1-3: gRPC/HTTP API server (Tokio)
- Day 4-5: Memory-mapped storage, persistence
- Day 6-7: Concurrency optimization
- **Deliverable**: Production-ready database

**Week 7: Comprehensive Benchmarking**
- Day 1-3: Qdrant 5가지 설정 벤치마크
- Day 4-5: BEIR 13 datasets 평가
- Day 6-7: 결과 분석 및 시각화
- **Deliverable**: 성능 비교 리포트

**Week 8: Documentation & Release**
- Day 1-3: API 문서 (Rust docs, Python bindings)
- Day 4-5: README, examples, tests
- Day 6-7: Cargo publish, PyPI release, blog post
- **Deliverable**: 오픈소스 공개 (Rust crate + Python wheel)

---

## 4. Evaluation Metrics

### 4.1 Benchmark Configurations

**Comparison Targets:**

| System | Configuration | Compression | Index Type |
|--------|--------------|-------------|------------|
| Qdrant Baseline | float32, HNSW | 1× | HNSW (M=16, ef=100) |
| Qdrant + Scalar Quant | int8, HNSW | 4× | HNSW (M=16, ef=100) |
| Qdrant + Binary Quant | 1-bit, HNSW | 32× | HNSW (M=16, ef=100) |
| Qdrant + PQ | m=16, nbits=8 | 32× | HNSW (M=16, ef=100) |
| Faiss + IVF-PQ | Standard PQ | 32× | IVF (nlist=4096) |
| Faiss + HNSW-PQ | K-means PQ | 32× | HNSW (M=16) |
| **QuantumDB** | Neural + Learnable PQ | **192×** | **HNSW (M=16, ef=200)** |

### 4.2 Performance Metrics

**Accuracy:**
- Recall@1, @10, @100
- nDCG@10
- MRR (Mean Reciprocal Rank)

**Speed:**
- QPS (Queries Per Second) at various ef_search
- Latency: p50, p95, p99
- Index build time (HNSW construction)
- Graph traversal efficiency (avg hops)

**Efficiency:**
- Memory footprint (GB)
- Compression ratio
- Storage cost ($/1M vectors)

**Trade-off:**
- Pareto curve: Recall@10 vs QPS
- Operating point: Recall@10 ≥ 0.95

### 4.3 Success Criteria

**Must-Have:**
- ✅ Recall@10 ≥ 0.95 (Qdrant PQ 대비 +3%)
- ✅ QPS 1.5-2× improvement (동일 recall)
- ✅ 메모리 100× 이상 압축
- ✅ Auto-tuning 완전 자동화

**Nice-to-Have:**
- ✅ BEIR 평균 성능 향상
- ✅ 100M 벡터 인덱싱 < 10분
- ✅ Training time < 24시간 (single GPU)

---

## 5. Project Structure

### 5.1 Repository Layout

```
quantumdb/
├── Cargo.toml              # Workspace root
├── crates/
│   ├── quantumdb-core/     # Core library
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── models/
│   │       │   ├── encoder.rs      # Neural encoder (Burn)
│   │       │   ├── quantizer.rs    # Learnable PQ (Burn)
│   │       │   └── compressor.rs   # End-to-end model
│   │       ├── training/
│   │       │   ├── losses.rs       # 5 loss functions
│   │       │   ├── trainer.rs      # Training loop (Burn)
│   │       │   ├── optimizer.rs    # AdamW wrapper
│   │       │   └── autotuning.rs   # Hyperparameter search
│   │       ├── data/
│   │       │   ├── loader.rs       # Arrow/Parquet loader
│   │       │   ├── embeddings.rs   # Embedding generation
│   │       │   └── augmentation.rs # Data augmentation
│   │       ├── index/
│   │       │   ├── hnsw.rs         # HNSW implementation
│   │       │   ├── distance.rs     # SIMD distance
│   │       │   └── graph.rs        # Graph structure
│   │       ├── storage/
│   │       │   ├── mmap.rs         # Memory-mapped storage
│   │       │   └── persistence.rs  # SafeTensors I/O
│   │       └── utils/
│   │           ├── metrics.rs      # Evaluation metrics
│   │           └── simd.rs         # SIMD helpers
│   │
│   ├── quantumdb-server/   # gRPC/HTTP server
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── main.rs
│   │       ├── grpc.rs
│   │       ├── http.rs
│   │       └── proto/
│   │
│   ├── quantumdb-cli/      # CLI tool
│   │   ├── Cargo.toml
│   │   └── src/
│   │       └── main.rs
│   │
│   └── quantumdb-python/   # PyO3 bindings (optional)
│       ├── Cargo.toml
│       ├── pyproject.toml
│       └── src/
│           └── lib.rs
│
├── benches/                # Criterion benchmarks
│   ├── training.rs
│   ├── inference.rs
│   └── search.rs
│
├── examples/               # Example code
│   ├── train.rs            # Training example
│   ├── serve.rs            # Server example
│   └── search.rs           # Search example
│
├── tests/                  # Integration tests
│   ├── model_tests.rs
│   ├── index_tests.rs
│   └── e2e_tests.rs
│
├── scripts/                # Build/deploy scripts
│   └── download_data.sh
│
├── models/                 # Trained artifacts
│   ├── encoder.safetensors
│   ├── codebooks.safetensors
│   └── config.json
│
├── docs/
│   ├── book/               # mdBook documentation
│   └── api/                # Generated docs
│
├── docker/
│   └── Dockerfile
│
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── release.yml
│
├── README.md
├── LICENSE                 # MIT
└── rust-toolchain.toml     # Rust version pinning
```

### 5.2 API Design

**Rust Training API:**

```rust
use quantumdb_core::prelude::*;

// Create trainer
let config = TrainingConfig {
    target_dim: 256,
    n_subvectors: 16,
    codebook_size: 256,
    learning_rate: 5e-4,
    batch_size: 512,
    epochs: 25,
    auto_tune: true, // Enable hyperparameter search
};

let mut trainer = Trainer::new(config);

// Load data (streaming from Arrow files)
let train_data = DataLoader::from_parquet(
    "ms_marco_embeddings.parquet",
    batch_size: 512,
)?;

let val_data = DataLoader::from_parquet(
    "ms_marco_dev.parquet",
    batch_size: 256,
)?;

// Train model
trainer.fit(train_data, val_data)?;

// Save to SafeTensors
trainer.save_checkpoint("models/final_model.safetensors")?;
```

**Rust Inference API:**

```rust
use quantumdb_core::QuantumDB;

// Load trained model
let db = QuantumDB::from_safetensors(
    "models/final_model.safetensors"
)?;

// Add vectors (compress + index)
db.add_batch(&ids, &embeddings)?;

// Build HNSW index
db.build_index()?;

// Search
let results = db.search(
    &query_embedding,
    top_k: 10,
    ef_search: 100,
)?;

// Save index
db.save("index.qdb")?;

// Load existing index (zero-copy mmap)
let db = QuantumDB::load("index.qdb")?;
```

**Rust CLI:**

```bash
# Train from scratch
quantumdb train \
    --data ms_marco_embeddings.parquet \
    --output models/model.safetensors \
    --auto-tune \
    --epochs 25 \
    --gpu 0

# Build index
quantumdb build \
    --model models/model.safetensors \
    --embeddings vectors.parquet \
    --output index.qdb \
    --hnsw-m 16

# Serve
quantumdb serve \
    --index index.qdb \
    --port 6333 \
    --workers 8

# Benchmark
quantumdb benchmark \
    --index index.qdb \
    --queries test_queries.parquet \
    --ground-truth relevance.json
```

**Python Bindings (PyO3 - Optional):**

```python
import quantumdb

# Training (calls Rust)
trainer = quantumdb.Trainer(
    target_dim=256,
    auto_tune=True,
)

trainer.fit("ms_marco_embeddings.parquet")
trainer.save("model.safetensors")

# Inference (Rust backend, zero-copy)
db = quantumdb.QuantumDB.load("model.safetensors")

# NumPy arrays (zero-copy via PyO3)
db.add(embeddings_np)  

# Multi-threaded search (releases GIL)
results = db.search(query_np, top_k=10)
```

**gRPC API:**

```protobuf
service QuantumDBService {
  rpc Add(AddRequest) returns (AddResponse);
  rpc Search(SearchRequest) returns (SearchResponse);
  rpc BatchSearch(stream SearchRequest) returns (stream SearchResponse);
  rpc BuildIndex(BuildIndexRequest) returns (BuildIndexResponse);
}

message SearchRequest {
  repeated float embedding = 1;
  int32 top_k = 2;
  int32 ef_search = 3;
}
```",
    config
)?;

// Add vectors
db.add_batch(&ids, &embeddings)?;

// Build HNSW index
db.build_index()?;

// Search
let results = db.search(&query_embedding, 10)?;

// Save index
db.save("my_index.qdb")?;
```

**Rust CLI:**

```bash
# Build index from embeddings
quantumdb build \
    --encoder models/encoder.safetensors \
    --codebooks models/codebooks.safetensors \
    --input embeddings.npy \
    --output index.qdb

# Serve gRPC API
quantumdb serve \
    --index index.qdb \
    --port 6333 \
    --workers 8

# Query
quantumdb query \
    --index index.qdb \
    --text "quantum computing" \
    --top-k 10
```

**Python Bindings (PyO3):**

```python
# Python wrapper around Rust database
from quantumdb import QuantumDB

# Same interface as training, but uses Rust backend
db = QuantumDB.from_safetensors(
    encoder="models/encoder.safetensors",
    codebooks="models/codebooks.safetensors"
)

# Zero-copy numpy arrays
db.add(embeddings)  # np.ndarray

# Multi-threaded search (releases GIL)
results = db.search(query, top_k=10)
```

**gRPC API (proto):**

```protobuf
service QuantumDBService {
  rpc Add(AddRequest) returns (AddResponse);
  rpc Search(SearchRequest) returns (SearchResponse);
  rpc BatchSearch(BatchSearchRequest) returns (stream SearchResponse);
}

message SearchRequest {
  repeated float embedding = 1;
  int32 top_k = 2;
  int32 ef_search = 3;
}

message SearchResponse {
  repeated SearchResult results = 1;
}

message SearchResult {
  string id = 1;
  float score = 2;
}
```

---

## 6. Resource Requirements

### 6.1 Hardware

**Development:**
- GPU: RTX 4090 (24GB VRAM)
- CPU: 16 cores recommended
- RAM: 64GB
- Storage: 500GB SSD

**Production (예상):**
- 100M vectors: ~1.5GB (compressed codes) + ~400MB (HNSW graph)
- Query latency: <5ms (ef_search=50)
- Throughput: 10,000+ QPS (Rust multi-threaded)
- HNSW build time: ~15 minutes (100M vectors, Rust parallel)

**Rust Performance Benefits:**
- 2-5× faster than Python inference
- Zero-copy with SafeTensors
- Lock-free concurrent search
- SIMD auto-vectorization

### 6.2 Cost Estimate

**Development Phase (8주):**
- Cloud GPU (RTX 4090 equivalent): $1.50/hr × 240hr = $360
- Storage (S3): ~$50
- Compute (CPU): ~$100
- **Total: ~$510**

**Production (vs Qdrant Cloud):**

| Scale | Qdrant Cloud | QuantumDB (Rust, Self-hosted) | Savings |
|-------|--------------|-------------------------------|---------|
| 10M vectors | $500/month | $30/month (CPU-only server) | 94% |
| 100M vectors | $3,000/month | $150/month | 95% |
| 1B vectors | $15,000/month | $800/month | 95% |

*QuantumDB Rust는 CPU만으로도 고성능 (SIMD + multi-threading)*

---

## 7. Risk Analysis

### 7.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Auto-tuning 실패 | Low | High | Stage-wise validation, manual fallback |
| 정확도 손실 > 5% | Medium | High | Stricter loss weights, codebook size 증가 |
| 속도 목표 미달 | Low | Medium | Rust SIMD 최적화, rayon parallelism |
| RTX 메모리 부족 | Low | Medium | Gradient checkpointing, smaller batch |
| HNSW 품질 저하 | Medium | Medium | Higher ef_construct, graph refinement |
| PyTorch → Burn 변환 | Medium | High | Thorough testing, numerical validation |
| Rust 개발 복잡도 | Medium | Medium | Incremental development, Python prototype first |

### 7.2 Project Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| 일정 지연 | Medium | Medium | 주간 마일스톤, agile sprint |
| Scope creep | Medium | Low | MVP 우선, feature freeze week 5 |
| 재현성 문제 | Low | High | Docker, fixed seeds, CI/CD |

---

## 8. Future Work

### 8.1 Phase 2 Extensions (Post-Launch)

**Multi-Backend Support (Burn):**
- CUDA (NVIDIA)
- ROCm (AMD)
- Metal (Apple Silicon)
- WebGPU (Browser)

**Multi-Modal Support:**
- 이미지 임베딩 (CLIP, ViT)
- 오디오 임베딩 (Whisper)
- 통합 멀티모달 검색

**Advanced Features:**
- Incremental learning (new data → update codebooks)
- Distributed training (multi-GPU) - PyTorch DDP
- Sparse embeddings 지원
- Dynamic HNSW (online insertion/deletion)
- Graph compression (reduce M adaptively)

**Production Tools:**
- Kubernetes operator (Rust binary)
- Grafana dashboard (Prometheus metrics)
- Auto-scaling policies
- Rust Docker images (minimal size)

### 8.2 Research Directions

- **Adaptive quantization**: Query-dependent precision
- **Neural architecture search**: AutoML for encoder
- **Federated learning**: Privacy-preserving compression
- **Cross-lingual**: Multilingual embedding 최적화
- **Burn-native training**: Replace PyTorch completely (long-term)
- **Quantization-aware inference**: INT8/FP16 Burn kernels

---

## 9. Team & Responsibilities

### 9.1 Roles (1인 프로젝트 가정)

**Week 1-3: ML Engineer**
- 데이터 파이프라인
- PyTorch 모델 구현
- Auto-tuning

**Week 4: ML/Systems Engineer**
- 모델 최적화
- SafeTensors export
- Rust 환경 구축

**Week 5-6: Systems Engineer (Rust)**
- Burn integration
- HNSW implementation
- API 개발

**Week 7: Performance Engineer**
- 벤치마킹
- 최적화

**Week 8: Developer Advocate**
- 문서화
- 오픈소스 준비

### 9.2 Community Engagement

**Launch Strategy:**
- GitHub: Star 목표 100+ (첫 달)
- HackerNews: Show HN 포스트
- Reddit: r/MachineLearning
- Twitter/X: Thread with benchmarks
- Medium: Technical deep-dive

---

## 10. Conclusion

## 10. Conclusion

**QuantumDB**는 AI 기반 압축과 Rust의 성능을 결합하여 차세대 벡터 데이터베이스를 구현합니다. Python으로 학습하고 Rust로 서빙하는 하이브리드 아키텍처로 개발 속도와 운영 성능을 동시에 확보합니다.

**핵심 가치:**
1. **192배 압축** → 클라우드 비용 95% 절감
2. **Rust 성능** → 2-5× 빠른 추론, CPU 만으로 고성능
3. **Zero-copy SafeTensors** → 메모리 효율적 모델 로딩
4. **Zero Label 학습** → 도메인 제약 없음
5. **완전 자동화** → 48시간이면 production-ready
6. **Cross-platform** → Burn으로 CUDA/ROCm/Metal 지원

**기술적 혁신:**
- **Learnable PQ**: End-to-end 최적화로 검색 품질 향상
- **Unsupervised Learning**: 레이블 없이 구조 보존
- **Auto-tuning**: Optuna로 하이퍼파라미터 자동 탐색
- **Rust HNSW**: Lock-free concurrent search
- **SIMD Distance**: portable-simd로 벡터화
- **PyTorch → Burn**: SafeTensors 기반 무손실 변환

**Impact:**
- Vector DB 사용 기업의 인프라 비용 95% 절감
- CPU 서버로도 고성능 벡터 검색 가능
- Rust 생태계에 ML inference 레퍼런스 제공
- 학술 연구: Learnable quantization + HNSW 벤치마크

**Why Rust?**
- **Memory Safety**: 메모리 버그 원천 차단
- **Zero-cost Abstractions**: 추상화 비용 없음
- **Fearless Concurrency**: 안전한 병렬 처리
- **Cross-compilation**: ARM, x86, WASM 모두 지원
- **Small Binaries**: Docker 이미지 크기 최소화

**Why Burn?**
- **Rust-native**: PyTorch보다 빠른 추론
- **Backend-agnostic**: GPU 벤더 독립적
- **Type-safe**: 컴파일 타임 차원 검사
- **No Python runtime**: 배포 단순화

**Why SafeTensors?**
- **Zero-copy**: mmap으로 즉시 로딩
- **Safe**: Pickle 취약점 없음
- **Fast**: 직렬화 오버헤드 최소
- **Interoperable**: PyTorch/JAX/TF 모두 지원

8주 후, 세상에서 가장 빠르고 효율적인 오픈소스 벡터 DB를 Rust로 선보이겠습니다.

---

## 11. Technical Deep Dive

### 11.1 Burn Framework Integration

**Burn Tensor Operations:**

```rust
use burn::prelude::*;
use burn::tensor::Tensor;
use burn::backend::Wgpu;

// Define encoder architecture
#[derive(Module, Debug)]
pub struct NeuralEncoder<B: Backend> {
    linear1: nn::Linear<B>,
    linear2: nn::Linear<B>,
    dropout: nn::Dropout,
}

impl<B: Backend> NeuralEncoder<B> {
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.linear1.forward(input);
        let x = burn::tensor::activation::relu(x);
        let x = self.dropout.forward(x);
        self.linear2.forward(x)
    }
}

// Load from SafeTensors
impl<B: Backend> NeuralEncoder<B> {
    pub fn from_safetensors(path: &str) -> Result<Self, Error> {
        let file = std::fs::read(path)?;
        let tensors = SafeTensors::deserialize(&file)?;
        
        // Extract weights
        let w1 = tensors.tensor("linear1.weight")?;
        let b1 = tensors.tensor("linear1.bias")?;
        let w2 = tensors.tensor("linear2.weight")?;
        let b2 = tensors.tensor("linear2.bias")?;
        
        // Convert to Burn tensors
        let linear1 = nn::LinearConfig::new(768, 256)
            .with_weights(Tensor::from_data(w1))
            .with_bias(Tensor::from_data(b1))
            .init();
            
        let linear2 = nn::LinearConfig::new(256, 256)
            .with_weights(Tensor::from_data(w2))
            .with_bias(Tensor::from_data(b2))
            .init();
        
        Ok(Self {
            linear1,
            linear2,
            dropout: nn::DropoutConfig::new(0.0).init(),
        })
    }
}
```

**Backend Selection:**

```rust
// CPU backend (default)
type Backend = burn::backend::NdArray;

// GPU backend (CUDA)
type Backend = burn::backend::Cuda;

// GPU backend (AMD)
type Backend = burn::backend::ROCm;

// Apple Silicon
type Backend = burn::backend::Metal;

// WebGPU (browser/WASM)
type Backend = burn::backend::Wgpu;
```

### 11.2 SIMD Distance Computation

**Asymmetric Distance with portable-simd:**

```rust
use std::simd::*;

pub struct SIMDDistance {
    lookup_table: Vec<f32>, // 16 × 256 precomputed distances
}

impl SIMDDistance {
    // Precompute distances for query
    pub fn build_lookup_table(
        &mut self,
        query_subvectors: &[[f32; 16]],
        codebooks: &[[[f32; 16]; 256]],
    ) {
        for (subvec, codebook) in query_subvectors.iter()
            .zip(codebooks.iter()) 
        {
            for code in codebook {
                let dist = l2_distance_simd(subvec, code);
                self.lookup_table.push(dist);
            }
        }
    }
    
    // Compute distance to compressed vector (16 codes)
    #[inline]
    pub fn compute(&self, codes: &[u8; 16]) -> f32 {
        let mut sum = 0.0f32;
        for (i, &code) in codes.iter().enumerate() {
            sum += self.lookup_table[i * 256 + code as usize];
        }
        sum
    }
    
    // Batch version with SIMD
    pub fn compute_batch(&self, codes: &[[u8; 16]]) -> Vec<f32> {
        codes.par_iter() // Rayon parallel
            .map(|c| self.compute(c))
            .collect()
    }
}

#[inline]
fn l2_distance_simd(a: &[f32; 16], b: &[f32; 16]) -> f32 {
    let a_simd = f32x16::from_array(*a);
    let b_simd = f32x16::from_array(*b);
    let diff = a_simd - b_simd;
    let squared = diff * diff;
    squared.reduce_sum()
}
```

### 11.3 HNSW Graph Implementation

**Lock-free Concurrent HNSW:**

```rust
use std::sync::Arc;
use parking_lot::RwLock;
use crossbeam::queue::SegQueue;

pub struct HNSWGraph {
    // Layered graph structure
    layers: Vec<Layer>,
    entry_point: AtomicUsize,
    max_layers: usize,
    m: usize,  // Max connections
    ef_construction: usize,
}

struct Layer {
    // Node ID → neighbors (lock-free reads)
    adjacency: DashMap<usize, Arc<RwLock<Vec<usize>>>>,
}

impl HNSWGraph {
    pub fn insert(&mut self, id: usize, vector: &[u8; 16]) {
        let level = self.random_level();
        
        // Start from top layer
        let mut ep = self.entry_point.load(Ordering::Relaxed);
        
        for lc in (level+1..self.max_layers).rev() {
            // Greedy search to find closest
            ep = self.search_layer(vector, ep, 1, lc)[0];
        }
        
        // Insert at each layer
        for lc in (0..=level).rev() {
            let candidates = self.search_layer(
                vector, 
                ep, 
                self.ef_construction, 
                lc
            );
            
            // Select M neighbors
            let neighbors = self.select_neighbors(
                vector, 
                candidates, 
                self.m
            );
            
            // Add bidirectional links
            self.add_connections(id, neighbors, lc);
            
            ep = neighbors[0];
        }
    }
    
    pub fn search(
        &self,
        query: &[u8; 16],
        k: usize,
        ef_search: usize,
    ) -> Vec<usize> {
        let mut ep = self.entry_point.load(Ordering::Relaxed);
        
        // Navigate down layers
        for lc in (1..self.max_layers).rev() {
            ep = self.search_layer(query, ep, 1, lc)[0];
        }
        
        // Search bottom layer
        let candidates = self.search_layer(
            query, 
            ep, 
            ef_search, 
            0
        );
        
        candidates.into_iter().take(k).collect()
    }
    
    fn search_layer(
        &self,
        query: &[u8; 16],
        entry: usize,
        num_closest: usize,
        layer: usize,
    ) -> Vec<usize> {
        let mut visited = HashSet::new();
        let mut candidates = BinaryHeap::new(); // Min-heap
        let mut best = BinaryHeap::new(); // Max-heap
        
        let dist = self.distance(query, entry);
        candidates.push(Reverse((OrderedFloat(dist), entry)));
        best.push((OrderedFloat(dist), entry));
        visited.insert(entry);
        
        while let Some(Reverse((d_c, c))) = candidates.pop() {
            if d_c > best.peek().unwrap().0 {
                break;
            }
            
            // Check neighbors
            if let Some(neighbors) = self.layers[layer]
                .adjacency.get(&c) 
            {
                for &e in neighbors.read().iter() {
                    if visited.insert(e) {
                        let d_e = self.distance(query, e);
                        
                        if d_e < best.peek().unwrap().0 
                            || best.len() < num_closest 
                        {
                            candidates.push(
                                Reverse((OrderedFloat(d_e), e))
                            );
                            best.push((OrderedFloat(d_e), e));
                            
                            if best.len() > num_closest {
                                best.pop();
                            }
                        }
                    }
                }
            }
        }
        
        best.into_sorted_vec()
            .into_iter()
            .map(|(_, id)| id)
            .collect()
    }
}
```

### 11.4 Memory-Mapped Persistence

**Zero-copy Save/Load:**

```rust
use memmap2::{Mmap, MmapMut};
use std::fs::OpenOptions;

pub struct QuantumDBStorage {
    // Memory-mapped files
    codes_mmap: Mmap,
    graph_mmap: Mmap,
    metadata: IndexMetadata,
}

#[repr(C)]
struct IndexMetadata {
    num_vectors: u64,
    dimension: u32,
    num_subvectors: u32,
    hnsw_m: u32,
    max_layers: u32,
}

impl QuantumDBStorage {
    pub fn save(
        &self,
        path: &str,
        codes: &[[u8; 16]],
        graph: &HNSWGraph,
    ) -> Result<(), Error> {
        // Create memory-mapped file
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(format!("{}/codes.bin", path))?;
        
        file.set_len((codes.len() * 16) as u64)?;
        
        let mut mmap = unsafe { MmapMut::map_mut(&file)? };
        
        // Write codes (zero-copy)
        unsafe {
            let ptr = mmap.as_mut_ptr() as *mut [u8; 16];
            std::ptr::copy_nonoverlapping(
                codes.as_ptr(),
                ptr,
                codes.len(),
            );
        }
        
        mmap.flush()?;
        
        // Save graph structure
        self.save_graph(path, graph)?;
        
        Ok(())
    }
    
    pub fn load(path: &str) -> Result<Self, Error> {
        // Memory-map codes (zero-copy load!)
        let file = OpenOptions::new()
            .read(true)
            .open(format!("{}/codes.bin", path))?;
        
        let codes_mmap = unsafe { Mmap::map(&file)? };
        
        // Load graph
        let graph_mmap = self.load_graph(path)?;
        
        Ok(Self {
            codes_mmap,
            graph_mmap,
            metadata: self.load_metadata(path)?,
        })
    }
    
    // Access codes without copying
    pub fn get_code(&self, id: usize) -> &[u8; 16] {
        unsafe {
            let ptr = self.codes_mmap.as_ptr() as *const [u8; 16];
            &*ptr.add(id)
        }
    }
}
```

### 11.5 gRPC API Server

**High-performance async server:**

```rust
use tonic::{transport::Server, Request, Response, Status};
use tokio::sync::RwLock;

pub struct QuantumDBService {
    db: Arc<RwLock<QuantumDB>>,
    distance_computer: Arc<SIMDDistance>,
}

#[tonic::async_trait]
impl quantumdb_proto::quantum_db_service_server::QuantumDbService 
    for QuantumDBService 
{
    async fn search(
        &self,
        request: Request<SearchRequest>,
    ) -> Result<Response<SearchResponse>, Status> {
        let req = request.into_inner();
        
        // Parse embedding (no copy if possible)
        let embedding = req.embedding;
        
        // Compress query
        let compressed = self.db.read().await
            .compress(&embedding)
            .map_err(|e| Status::internal(e.to_string()))?;
        
        // Search (releases lock)
        let results = self.db.read().await
            .search(&compressed, req.top_k as usize, req.ef_search as usize)
            .map_err(|e| Status::internal(e.to_string()))?;
        
        // Build response
        let search_results = results.into_iter()
            .map(|(id, score)| SearchResult {
                id: id.to_string(),
                score,
            })
            .collect();
        
        Ok(Response::new(SearchResponse {
            results: search_results,
        }))
    }
    
    async fn batch_search(
        &self,
        request: Request<BatchSearchRequest>,
    ) -> Result<Response<Self::BatchSearchStream>, Status> {
        // Stream responses for large batches
        let (tx, rx) = mpsc::channel(100);
        
        tokio::spawn(async move {
            // Process in parallel with rayon
            // Send results as they complete
        });
        
        Ok(Response::new(ReceiverStream::new(rx)))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr = "[::1]:6333".parse()?;
    
    let service = QuantumDBService::new("index.qdb")?;
    
    Server::builder()
        .add_service(
            quantumdb_proto::quantum_db_service_server::QuantumDbServiceServer::new(service)
        )
        .serve(addr)
        .await?;
    
    Ok(())
}
```

---

## 12. Deployment Guide

### 12.1 Docker Deployment

**Minimal Rust image:**

```dockerfile
# Multi-stage build
FROM rust:1.75 as builder

WORKDIR /app
COPY . .

# Build with optimizations
RUN cargo build --release --bin quantumdb

# Minimal runtime image
FROM debian:bookworm-slim

# Install only runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/quantumdb /usr/local/bin/

# Copy models
COPY models/ /models/

EXPOSE 6333

CMD ["quantumdb", "serve", \
     "--index", "/data/index.qdb", \
     "--port", "6333"]
```

**Docker Compose:**

```yaml
version: '3.8'

services:
  quantumdb:
    image: quantumdb:latest
    ports:
      - "6333:6333"
    volumes:
      - ./models:/models:ro
      - ./data:/data
    environment:
      - RUST_LOG=info
      - QUANTUMDB_WORKERS=8
    deploy:
      resources:
        limits:
          cpus: '8'
          memory: 16G
```

**Image size**: ~50MB (vs Python: ~1GB+)

### 12.2 Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: quantumdb
spec:
  replicas: 3
  selector:
    matchLabels:
      app: quantumdb
  template:
    metadata:
      labels:
        app: quantumdb
    spec:
      containers:
      - name: quantumdb
        image: quantumdb:latest
        ports:
        - containerPort: 6333
        resources:
          requests:
            memory: "8Gi"
            cpu: "4"
          limits:
            memory: "16Gi"
            cpu: "8"
        volumeMounts:
        - name: index
          mountPath: /data
          readOnly: true
        livenessProbe:
          grpc:
            port: 6333
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          grpc:
            port: 6333
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: index
        persistentVolumeClaim:
          claimName: quantumdb-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: quantumdb-service
spec:
  selector:
    app: quantumdb
  ports:
  - protocol: TCP
    port: 6333
    targetPort: 6333
  type: LoadBalancer
```

### 12.3 Performance Tuning

**Rust Compiler Optimizations:**

```toml
[profile.release]
opt-level = 3
lto = "fat"           # Link-time optimization
codegen-units = 1     # Better optimization
panic = "abort"       # Smaller binary
strip = true          # Remove symbols

[profile.release.package."*"]
opt-level = 3

# Platform-specific
[target.x86_64-unknown-linux-gnu]
rustflags = [
    "-C", "target-cpu=native",  # Use all CPU features
    "-C", "target-feature=+avx2,+fma",
]
```

**Runtime Configuration:**

```rust
// Set thread pool size
rayon::ThreadPoolBuilder::new()
    .num_threads(num_cpus::get())
    .build_global()
    .unwrap();

// Configure allocator
#[global_allocator]
static ALLOC: jemallocator::Jemalloc = jemallocator::Jemalloc;

// Memory mapping advice
mmap.advise(memmap2::Advice::Sequential)?;
mmap.advise(memmap2::Advice::WillNeed)?;
```

---

## Appendix

### A. References

**Product Quantization:**
- Jégou et al. (2011): Product Quantization for Nearest Neighbor Search
- Ge et al. (2013): Optimized Product Quantization

**Learnable Quantization:**
- Klein & Wolf (2019): JPQ: Joint Product Quantization
- Martinez et al. (2018): LSQ: Locally-Sensitive Quantization

**HNSW:**
- Malkov & Yashunin (2018): Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs
- HNSWlib: https://github.com/nmslib/hnswlib

**Rust ML:**
- Burn: https://github.com/tracel-ai/burn
- SafeTensors: https://github.com/huggingface/safetensors
- Candle: https://github.com/huggingface/candle (alternative)

**Vector Databases:**
- Qdrant: https://qdrant.tech
- Faiss: https://github.com/facebookresearch/faiss

### B. Datasets

- MS MARCO: https://microsoft.github.io/msmarco/
- BEIR: https://github.com/beir-cellar/beir

### C. Benchmarking Tools

- Criterion.rs: https://github.com/bheisler/criterion.rs
- pprof: Profiling for Rust
- cargo-flamegraph: Flame graph generation

### D. Contact & Resources

- GitHub: https://github.com/[username]/quantumdb
- Crates.io: https://crates.io/crates/quantumdb
- PyPI: https://pypi.org/project/quantumdb
- Documentation: https://docs.rs/quantumdb
- Email: [your-email]
- Discord: [community-link]

---

## Appendix

### A. References

**Product Quantization:**
- Jégou et al. (2011): Product Quantization for Nearest Neighbor Search
- Ge et al. (2013): Optimized Product Quantization

**Learnable Quantization:**
- Klein & Wolf (2019): JPQ: Joint Product Quantization
- Martinez et al. (2018): LSQ: Locally-Sensitive Quantization

**HNSW:**
- Malkov & Yashunin (2018): Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs
- HNSWlib: https://github.com/nmslib/hnswlib

**Vector Databases:**
- Qdrant: https://qdrant.tech
- Faiss: https://github.com/facebookresearch/faiss

### B. Datasets

- MS MARCO: https://microsoft.github.io/msmarco/
- BEIR: https://github.com/beir-cellar/beir

### C. Contact

- GitHub: https://github.com/[username]/quantumdb
- Email: [your-email]
- Documentation: https://quantumdb.readthedocs.io
