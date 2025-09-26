'''
Training script for the code generation model.
'''
import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from safetensors.torch import save_file

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.tokenizer import get_tokenizer
from src.dataset import CodeDataset
from src.model import CodeGenerationModel

def train(args):
    '''Main training loop for the code generation model.'''
    # --- 1. Configuration ---
    tokenizer = get_tokenizer()
    vocab_size = tokenizer.n_vocab
    
    # --- 2. Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    print(f"Using device: {device}")

    # --- 3. Data ---
    train_dataset = CodeDataset(data_path="data/train.parquet", tokenizer=tokenizer, max_length=args.max_seq_length)
    loader_batch_size = max(1, args.batch_size // args.accumulation_steps)
    train_loader = DataLoader(train_dataset, batch_size=loader_batch_size, shuffle=True)

    # --- 4. Model ---
    model = CodeGenerationModel(
        vocab_size=vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_seq_length=args.max_seq_length
    ).to(device)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters.")

    # --- 5. Training ---
    lm_criterion = nn.CrossEntropyLoss(ignore_index=-100)
    contrastive_criterion = nn.CosineEmbeddingLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.01)
    
    # LR Scheduler
    warmup_scheduler = LinearLR(optimizer, start_factor=0.001, end_factor=1.0, total_iters=args.warmup_steps)
    total_steps = len(train_loader) * args.epochs
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - args.warmup_steps, eta_min=args.learning_rate * 0.1)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[args.warmup_steps])

    model.train()
    optimizer.zero_grad()

    print("Starting training...")
    for epoch in range(args.epochs):
        total_epoch_loss = 0
        num_batches = 0
        for i, batch in enumerate(train_loader):
            if args.limit_train_batches and i >= args.limit_train_batches:
                break

            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            split_point = int(input_ids.size(1) * 0.3)
            src = input_ids[:, :split_point]
            tgt = input_ids[:, split_point:]
            tgt_labels = labels[:, split_point:]

            if src.size(1) == 0 or tgt.size(1) <= 1:
                continue

            output, src_embedding, code_embedding = model(src=src, tgt=tgt[:, :-1])
            
            loss_lm = lm_criterion(output.reshape(-1, vocab_size), tgt_labels[:, 1:].reshape(-1))
            
            target = torch.ones(src_embedding.size(0)).to(device)
            loss_align = contrastive_criterion(src_embedding, code_embedding, target)
            
            loss = loss_lm + args.contrastive_loss_weight * loss_align
            
            loss = loss / args.accumulation_steps
            loss.backward()

            if (i + 1) % args.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_epoch_loss += loss.item() * args.accumulation_steps
            num_batches = i + 1

            if (i + 1) % (10 * args.accumulation_steps) == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}], Step [{i+1}/{len(train_loader)}], LR: {scheduler.get_last_lr()[0]:.6f}, Total Loss: {loss.item() * args.accumulation_steps:.4f}, LM Loss: {loss_lm.item():.4f}, Align Loss: {loss_align.item():.4f}")

        avg_epoch_loss = total_epoch_loss / num_batches if num_batches > 0 else 0
        print(f"Epoch {epoch+1} finished. Average Loss: {avg_epoch_loss:.4f}")

    print("Training complete.")
    
    # Save the final model state
    print(f"Saving model to {args.save_path}")
    save_file(model.state_dict(), args.save_path)
    print("Model saved.")

def main():
    parser = argparse.ArgumentParser(description="Train the code generation model.")
    
    # Model Hyperparameters
    parser.add_argument("--d_model", type=int, default=1024, help="Model dimension")
    parser.add_argument("--nhead", type=int, default=16, help="Number of attention heads")
    parser.add_argument("--num_encoder_layers", type=int, default=6, help="Number of encoder layers")
    parser.add_argument("--num_decoder_layers", type=int, default=6, help="Number of decoder layers")
    parser.add_argument("--dim_feedforward", type=int, default=2048, help="Dimension of feedforward network")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

    # Training Hyperparameters
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Effective batch size")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=10000, help="Number of warmup steps")
    parser.add_argument("--contrastive_loss_weight", type=float, default=0.1, help="Weight for the contrastive loss")
    parser.add_argument("--save_path", type=str, default="codegen_model.safetensors", help="Path to save the trained model")
    
    # Testing & Debugging
    parser.add_argument("--limit_train_batches", type=int, default=None, help="Limit the number of batches for a quick test run")
    parser.add_argument("--force_cpu", action="store_true", help="Force use of CPU even if CUDA is available")

    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main()