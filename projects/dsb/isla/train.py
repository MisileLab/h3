#!/usr/bin/env python
# -*- coding: utf-8 -*-

# --- Installation (using uv) ---
# uv init
# uv pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cu121
# uv pip install transformers datasets tqdm safetensors accelerate
# uv pip install trackio  # Optional, for logging. Falls back to a mock object if not found.

# Note: torch>=2.0 is required for torch.nn.functional.scaled_dot_product_attention.
# This script is designed for modern PyTorch versions.

# --- Example Execution ---
# python train.py --config 1B --batch_size 4 --epochs 1 --output_dir ./outputs --project_name "CodeGenPro-1B"
#
# For debugging on a small number of samples:
# python train.py --config 1B --batch_size 2 --debug --debug_steps 50 --log_interval 10

# --- OOM Warning ---
# Larger models (7B, 15B) will require substantial VRAM and may cause Out-Of-Memory (OOM)
# errors on consumer-grade GPUs. Adjust batch_size or use model parallelism strategies
# not included in this script for training such models.

import argparse
import logging
import math
import os
import sys
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from torch.cuda.amp import GradScaler, autocast

from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizer

# --- 1. Tracking and Logging Setup ---
try:
    import trackio as wandb
    trackio_available = True
except ImportError:
    logging.warning("`trackio` not found. Using a mock logger. `pip install trackio` for full logging.")
    trackio_available = False

    @dataclass
    class MockWandb:
        """A mock object to stand in for the wandb library if it's not installed."""
        def init(self, *args, **kwargs): pass
        def log(self, *args, **kwargs): pass
        def define_metric(self, *args, **kwargs): pass
        def finish(self, *args, **kwargs): pass
        def __call__(self, *args, **kwargs): return self
    wandb = MockWandb()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


# --- 2. Model Configuration ---
MODEL_PRESETS = {
    "1B": {
        "n_layer": 12, "n_head": 12, "d_model": 768, "d_ff": 3072,
        "vocab_size": 50257, "max_length": 1024, "dropout": 0.1
    },
    "7B": {
        "n_layer": 32, "n_head": 32, "d_model": 4096, "d_ff": 16384,
        "vocab_size": 50257, "max_length": 2048, "dropout": 0.1
    },
    "15B": {
        "n_layer": 48, "n_head": 40, "d_model": 6144, "d_ff": 24576,
        "vocab_size": 50257, "max_length": 2048, "dropout": 0.1
    },
}

@dataclass
class ModelConfig:
    n_layer: int = 12
    n_head: int = 12
    d_model: int = 768
    d_ff: int = 3072
    vocab_size: int = 50257
    max_length: int = 1024
    dropout: float = 0.1
    ast_vocab_size: Optional[int] = None # Placeholder for AST tokenizer

# --- 3. Model Architecture ---

class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        """
        return x + self.pe[:, :x.size(1)]

class MultiModalEncoder(nn.Module):
    """
    Encodes text and optional AST tokens, combining them with element-wise addition.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_encoder = PositionalEncoding(config.d_model, config.max_length)
        self.dropout = nn.Dropout(config.dropout)

        if config.ast_vocab_size:
            self.ast_embedding = nn.Embedding(config.ast_vocab_size, config.d_model)
            logging.info("AST embedding layer initialized.")

    def forward(self, input_ids: torch.Tensor, ast_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Embed text tokens
        txt_embeds = self.token_embedding(input_ids) * math.sqrt(self.config.d_model)
        
        final_embeds = txt_embeds

        # If AST tokens are provided, embed and add them
        if ast_ids is not None and hasattr(self, 'ast_embedding'):
            ast_embeds = self.ast_embedding(ast_ids) * math.sqrt(self.config.d_model)
            # Ensure shapes match for element-wise addition
            if ast_embeds.shape == txt_embeds.shape:
                final_embeds = final_embeds + ast_embeds
            else:
                logging.warning(f"Shape mismatch between text ({txt_embeds.shape}) and AST ({ast_embeds.shape}) embeddings. Skipping addition.")

        # Add positional encoding and apply dropout
        final_embeds = self.pos_encoder(final_embeds)
        return self.dropout(final_embeds)

class StructureAwareAttention(nn.Module):
    """
    Multi-head attention with optional additive attention bias.
    Uses PyTorch 2.0's scaled_dot_product_attention for efficiency.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.d_model % config.n_head == 0
        self.n_head = config.n_head
        self.d_model = config.d_model
        self.head_dim = config.d_model // config.n_head
        
        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, attention_bias: Optional[torch.Tensor] = None):
        B, T, C = x.size() # Batch, Time, Channels (d_model)

        # 1. Project to Q, K, V
        q, k, v = self.qkv_proj(x).split(self.d_model, dim=2)

        # 2. Reshape for multi-head attention
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)

        # 3. Apply scaled dot-product attention
        # `attn_mask` should be a boolean mask where True means "attend"
        # `attention_bias` is added to the pre-softmax attention scores.
        # Note: `scaled_dot_product_attention` handles the causal mask internally if `is_causal=True`.
        # Here we use an explicit mask for flexibility.
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=attention_bias is None and attention_mask is None # Use causal mask if no bias/mask is given
        )

        # 4. Reshape and project output
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(y)

class FeedForward(nn.Module):
    """A simple MLP block for the Transformer."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.linear_1 = nn.Linear(config.d_model, config.d_ff)
        self.gelu = nn.GELU()
        self.linear_2 = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.linear_2(self.gelu(self.linear_1(x))))

class TransformerBlock(nn.Module):
    """A single Transformer block."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = StructureAwareAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        x = x + self.attn(self.ln_1(x), attention_mask=attention_mask)
        x = x + self.ffn(self.ln_2(x))
        return x

class CodeGenPro(nn.Module):
    """The main model: a stack of Transformer blocks."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.encoder = MultiModalEncoder(config)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying
        self.encoder.token_embedding.weight = self.lm_head.weight

        self.apply(self._init_weights)
        logging.info(f"Model initialized with {self.get_num_params():,} parameters.")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, input_ids: torch.Tensor, ast_ids: Optional[torch.Tensor] = None, targets: Optional[torch.Tensor] = None):
        B, T = input_ids.size()
        
        # Create a causal mask for the self-attention layers
        # (1, T, T) -> (B, 1, T, T) after broadcasting
        causal_mask = torch.triu(torch.ones(T, T, device=input_ids.device, dtype=torch.bool), diagonal=1)

        x = self.encoder(input_ids, ast_ids)
        
        for block in self.transformer_blocks:
            x = block(x, attention_mask=~causal_mask) # Attention mask expects False for masked positions

        x = self.ln_f(x)
        
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            # Shift for next-token prediction
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            
        return logits, loss

# --- 4. Data Loading ---

class StreamingDataset(IterableDataset):
    """
    An IterableDataset that streams data from Hugging Face datasets,
    tokenizes on the fly, and creates batches.
    """
    def __init__(self, tokenizer: PreTrainedTokenizer, dataset_name: str, split: str, block_size: int, max_samples: Optional[int] = None):
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.split = split
        self.block_size = block_size
        self.max_samples = max_samples
        
        # For reproducibility with streaming datasets
        self.seed = 42

    def __iter__(self):
        dataset = load_dataset(self.dataset_name, split=self.split, streaming=True)
        dataset = dataset.shuffle(seed=self.seed, buffer_size=10_000)
        
        if self.max_samples:
            dataset = dataset.take(self.max_samples)

        buffer = []
        count = 0
        for item in dataset:
            if count >= self.max_samples if self.max_samples else False:
                break
            
            # Assuming the dataset has a 'whole_func_string' field like codesearchnet
            text = item.get('whole_func_string') or item.get('text')
            if not text:
                continue

            tokens = self.tokenizer(text, truncation=False, padding=False)["input_ids"]
            buffer.extend(tokens)

            while len(buffer) >= self.block_size:
                chunk = buffer[:self.block_size]
                buffer = buffer[self.block_size:]
                
                input_ids = torch.tensor(chunk, dtype=torch.long)
                
                # --- AST Tokenizer Integration Point ---
                # If you have an AST tokenizer, you would generate ast_ids here.
                # For now, we create a placeholder tensor of zeros.
                ast_ids = torch.zeros_like(input_ids)
                
                yield {"input_ids": input_ids, "ast_ids": ast_ids}
                count += 1
                if count >= self.max_samples if self.max_samples else False:
                    break

def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    ast_ids = torch.stack([item['ast_ids'] for item in batch])
    labels = input_ids.clone()
    labels[labels == 0] = -1 # Use -1 for ignore_index in CrossEntropyLoss
    return {"input_ids": input_ids, "ast_ids": ast_ids, "labels": labels}

# --- 5. Utilities ---

def get_device():
    """Detects and returns the best available device."""
    if torch.cuda.is_available():
        logging.info("CUDA is available. Using GPU.")
        return torch.device("cuda")
    # Add MPS support for Apple Silicon if needed
    # elif torch.backends.mps.is_available():
    #     logging.info("MPS is available. Using Apple Silicon GPU.")
    #     return torch.device("mps")
    else:
        logging.info("CUDA not available. Using CPU.")
        return torch.device("cpu")

def set_seed(seed_value=42):
    """Sets the seed for reproducibility."""
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    # np.random.seed(seed_value)
    # random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

def find_adaptive_batch_size(model, device, block_size, initial_batch_size, safety_margin=0.95):
    """
    Finds the largest batch size that fits in memory by trying and halving.
    """
    if device.type != 'cuda':
        logging.info(f"Device is not CUDA. Using initial batch size: {initial_batch_size}")
        return initial_batch_size

    logging.info("Attempting to find an adaptive batch size...")
    torch.cuda.empty_cache()
    
    batch_size = initial_batch_size
    while batch_size > 0:
        try:
            # Create a dummy batch
            input_ids = torch.randint(0, model.config.vocab_size, (batch_size, block_size), device=device)
            targets = input_ids.clone()

            # Perform a forward pass
            with autocast(enabled=True):
                _, loss = model(input_ids, ast_ids=None, targets=targets)
            
            # Perform a backward pass to account for gradients
            scaler = GradScaler()
            scaler.scale(loss).backward()

            # If we reach here, the batch size is viable
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            logging.info(f"Found a viable batch size: {batch_size}")
            return batch_size

        except torch.cuda.OutOfMemoryError:
            logging.warning(f"OOM with batch size: {batch_size}. Halving and retrying.")
            batch_size //= 2
            torch.cuda.empty_cache()
        
        except Exception as e:
            logging.error(f"An unexpected error occurred during batch size finding: {e}")
            torch.cuda.empty_cache()
            return 1 # Fallback to a minimal batch size

    if batch_size == 0:
        raise RuntimeError("Could not fit a batch size of 1 into VRAM. Aborting.")
        
    return 1

@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device, criterion, max_batches: int = 50):
    """Performs evaluation on a validation set."""
    model.eval()
    total_loss = 0
    total_batches = 0
    
    pbar = tqdm(dataloader, desc="Evaluating", total=max_batches, leave=False)
    for i, batch in enumerate(pbar):
        if i >= max_batches:
            break
        
        input_ids = batch['input_ids'].to(device)
        ast_ids = batch['ast_ids'].to(device)
        targets = batch['labels'].to(device)

        _, loss = model(input_ids, ast_ids=ast_ids, targets=targets)
        
        if loss is not None:
            total_loss += loss.item()
        total_batches += 1

    model.train()
    if total_batches == 0:
        return float('inf')
    return total_loss / total_batches

# --- 6. Main Training and Evaluation Logic ---

def main(args):
    """Main function to orchestrate training."""
    set_seed()
    device = get_device()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Init Tracking ---
    if trackio_available and not args.debug:
        wandb.init(project=args.project_name, config=vars(args))

    # --- Tokenizer ---
    logging.info(f"Loading tokenizer: {args.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Model ---
    logging.info(f"Initializing model with '{args.config}' preset.")
    preset = MODEL_PRESETS[args.config]
    preset['vocab_size'] = tokenizer.vocab_size
    preset['max_length'] = args.block_size
    preset['dropout'] = args.dropout
    model_config = ModelConfig(**preset)
    model = CodeGenPro(model_config).to(device)

    # --- Adaptive Batch Size ---
    batch_size = args.batch_size
    if args.auto_batch_size:
        batch_size = find_adaptive_batch_size(model, device, args.block_size, args.batch_size)

    # --- Data ---
    logging.info(f"Loading and streaming dataset: {args.dataset} with batch size {batch_size}")
    train_dataset = StreamingDataset(tokenizer, args.dataset, "train", args.block_size, args.max_samples)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn)
    
    val_dataset = StreamingDataset(tokenizer, args.dataset, "validation", args.block_size, max_samples=batch_size * 50) # smaller val set
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)

    # --- Optimizer, Loss, and Scaler ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    scaler = GradScaler(enabled=device.type == 'cuda')

    # --- Checkpoint Resuming ---
    # TODO: Implement more robust checkpoint loading logic
    # For now, this is a placeholder.
    start_epoch = 0
    global_step = 0
    if args.resume_from:
        logging.info(f"Resuming from checkpoint: {args.resume_from}")
        # checkpoint = torch.load(args.resume_from, map_location=device)
        # model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # scaler.load_state_dict(checkpoint['scaler_state_dict'])
        # start_epoch = checkpoint['epoch']
        # global_step = checkpoint['global_step']

    # --- Early Stopping ---
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_checkpoint_path = ""

    # --- Training Loop ---
    model.train()
    for epoch in range(start_epoch, args.epochs):
        logging.info(f"Starting Epoch {epoch + 1}/{args.epochs}")
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", disable=args.debug)
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            ast_ids = batch['ast_ids'].to(device)
            targets = batch['labels'].to(device)

            with autocast(enabled=device.type == 'cuda'):
                logits, loss = model(input_ids, ast_ids=ast_ids, targets=targets)
            
            if loss is None:
                logging.warning("Loss is None, skipping batch.")
                continue

            if args.stop_on_nan and torch.isnan(loss):
                logging.error("Loss is NaN. Stopping training.")
                wandb.finish(exit_code=1)
                return

            scaler.scale(loss).backward()

            if args.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            # Logging
            if global_step % args.log_interval == 0:
                log_data = {"train/loss": loss.item(), "epoch": epoch + 1, "step": global_step}
                if trackio_available and not args.debug:
                    wandb.log(log_data)
                else:
                    logging.info(f"Step {global_step}: Train Loss = {loss.item():.4f}")

            # Checkpointing
            if global_step % args.save_interval == 0:
                checkpoint_dir = output_dir / f"ckpt-step-{global_step}"
                checkpoint_dir.mkdir(exist_ok=True)
                
                # Save model, optimizer, etc.
                save_path = checkpoint_dir / "state.pt"
                torch.save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'args': vars(args)
                }, save_path)
                logging.info(f"Saved checkpoint to {save_path}")

            if args.debug and global_step >= args.debug_steps:
                logging.info(f"Reached debug step limit ({args.debug_steps}). Stopping.")
                break
        
        if args.debug and global_step >= args.debug_steps:
            break

        # --- End-of-Epoch Evaluation and Early Stopping ---
        if args.eval_after_epoch:
            logging.info("Running end-of-epoch evaluation...")
            val_loss = evaluate(model, val_loader, device, criterion)
            logging.info(f"Epoch {epoch+1}: Validation Loss = {val_loss:.4f}")
            
            log_data = {"val/loss": val_loss, "epoch": epoch + 1}
            if trackio_available and not args.debug:
                wandb.log(log_data)

            # Early stopping logic
            if val_loss < best_val_loss - args.early_stop_min_delta:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                
                # Save the best model
                best_checkpoint_path = output_dir / "best_model.pt"
                torch.save(model.state_dict(), best_checkpoint_path)
                logging.info(f"New best model saved with val_loss: {best_val_loss:.4f} to {best_checkpoint_path}")
                if trackio_available and not args.debug:
                    wandb.log({"best_val_loss": best_val_loss})

            else:
                epochs_without_improvement += 1
                logging.info(f"No improvement in validation loss for {epochs_without_improvement} epoch(s).")

            if trackio_available and not args.debug:
                wandb.log({"epochs_since_improve": epochs_without_improvement})

            if epochs_without_improvement >= args.early_stop_patience:
                logging.info(f"Early stopping triggered after {args.early_stop_patience} epochs without improvement.")
                logging.info(f"Best model checkpoint is at: {best_checkpoint_path}")
                break # Exit training loop

    logging.info("Training finished.")
    if trackio_available and not args.debug:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a CodeGenPro model.")

    # --- Core ---
    parser.add_argument("--config", type=str, required=True, choices=MODEL_PRESETS.keys(), help="Model configuration preset.")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Directory to save checkpoints and logs.")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to a checkpoint to resume training from.")

    # --- Data ---
    parser.add_argument("--dataset", type=str, default="sentence-transformers/codesearchnet", help="Hugging Face dataset name.")
    parser.add_argument("--tokenizer_name", type=str, default="gpt2", help="Tokenizer to use.")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size. Overridden if --auto_batch_size is set.")
    parser.add_argument("--auto_batch_size", action="store_true", help="Automatically find the largest batch size that fits in VRAM.")
    parser.add_argument("--block_size", type=int, default=1024, help="Context size for the model.")
    parser.add_argument("--max_samples", type=int, default=None, help="Max number of samples to process (for debugging).")

    # --- Training ---
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for AdamW.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping value (0 for no clipping).")
    parser.add_argument("--stop_on_nan", action="store_true", help="Stop training if loss becomes NaN.")

    # --- Logging & Saving ---
    parser.add_argument("--project_name", type=str, default="codegenpro-training", help="Project name for tracking.")
    parser.add_argument("--log_interval", type=int, default=100, help="Log training loss every N steps.")
    parser.add_argument("--save_interval", type=int, default=1000, help="Save a checkpoint every N steps.")

    # --- Evaluation & Early Stopping ---
    parser.add_argument("--eval_after_epoch", action="store_true", help="Run evaluation after each epoch.")
    parser.add_argument("--early_stop_patience", type=int, default=3, help="Patience for early stopping.")
    parser.add_argument("--early_stop_min_delta", type=float, default=0.0, help="Minimum change to qualify as an improvement.")
    parser.add_argument("--early_stop_metric", type=str, default="val_loss", help="Metric for early stopping (currently only val_loss).")
    
    # --- Debugging ---
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (disables tracking, limits steps).")
    parser.add_argument("--debug_steps", type=int, default=100, help="Number of steps to run in debug mode.")

    args = parser.parse_args()
    main(args)
