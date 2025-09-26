'''
Evaluation script for the code generation model.
'''
import os
import sys
import argparse
import math
import torch
from torch.utils.data import DataLoader
from torchtext.data.metrics import bleu_score
from safetensors.torch import load_file

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.tokenizer import get_tokenizer
from src.dataset import CodeDataset
from src.model import CodeGenerationModel

def greedy_decode(model, src, tokenizer, device, max_len=512):
    '''
    Generates a sequence from the model using greedy decoding.
    '''
    model.eval()
    
    start_token = tokenizer.encode("<|startofcode|>").pop(0)
    end_token = tokenizer.encode("<|endofcode|>").pop(0)

    src = src.to(device)
    if src.dim() == 1:
        src = src.unsqueeze(0)

    with torch.no_grad():
        src_emb = model.embedding(src) * math.sqrt(model.d_model)
        src_emb = model.pos_encoder(src_emb)
        memory = model.transformer.encoder(src_emb)
        
        tgt = torch.ones(1, 1).fill_(start_token).long().to(device)
        
        for _ in range(max_len - 1):
            memory = memory.to(device)
            tgt_mask = model.transformer.generate_square_subsequent_mask(tgt.size(1)).to(device)
            
            out = model.transformer.decoder(model.pos_encoder(model.embedding(tgt) * math.sqrt(model.d_model)), memory, tgt_mask)
            prob = model.fc_out(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            
            if next_word.item() == end_token:
                break
                
            tgt = torch.cat([tgt, torch.ones(1, 1).type_as(src.data).fill_(next_word.item())], dim=1)
            
    return tgt

def evaluate(args):
    '''Main evaluation loop for the model.'''
    tokenizer = get_tokenizer()
    vocab_size = tokenizer.n_vocab
    
    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    print(f"Using device: {device}")

    eval_dataset = CodeDataset(data_path="data/validation.parquet", tokenizer=tokenizer, max_length=args.max_seq_length)
    eval_loader = DataLoader(eval_dataset, batch_size=1)

    model = CodeGenerationModel(
        vocab_size=vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        max_seq_length=args.max_seq_length
    ).to(device)

    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Model loaded from {args.model_path}")
    else:
        print(f"Warning: Model weights not found at {args.model_path}. Using a randomly initialized model for evaluation.")

    model.eval()
    print("Starting evaluation...")
    
    total_bleu = 0
    num_batches = 0

    with torch.no_grad():
        for i, batch in enumerate(eval_loader):
            if args.limit_eval_batches and i >= args.limit_eval_batches:
                break

            input_ids = batch['input_ids'].to(device)
            
            split_point = int(input_ids.size(1) * 0.3)
            src = input_ids[:, :split_point]
            reference_ids = batch['input_ids'].squeeze().tolist()

            if src.size(1) == 0:
                continue

            generated_ids = greedy_decode(model, src, tokenizer, device, max_len=args.max_gen_length).squeeze().tolist()
            
            candidate_corpus = [str(tok) for tok in tokenizer.decode(generated_ids).split()]
            reference_corpus = [[str(tok) for tok in tokenizer.decode(reference_ids).split()]]
            
            bleu = bleu_score([candidate_corpus], reference_corpus)
            total_bleu += bleu
            num_batches = i + 1
            
            if i % 10 == 0:
                print(f"\n--- Example {i} ---")
                print(f"  Source:    {tokenizer.decode(src.squeeze().tolist())}")
                print(f"  Generated: {tokenizer.decode(generated_ids)}")
                print(f"  Reference: {tokenizer.decode(reference_ids)}")
                print(f"  BLEU: {bleu:.4f}")

    avg_bleu = total_bleu / num_batches if num_batches > 0 else 0
    print(f"\nEvaluation complete. Average BLEU score: {avg_bleu:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate the code generation model.")
    
    # Model Hyperparameters
    parser.add_argument("--d_model", type=int, default=1024, help="Model dimension")
    parser.add_argument("--nhead", type=int, default=16, help="Number of attention heads")
    parser.add_argument("--num_encoder_layers", type=int, default=6, help="Number of encoder layers")
    parser.add_argument("--num_decoder_layers", type=int, default=6, help="Number of decoder layers")
    parser.add_argument("--dim_feedforward", type=int, default=2048, help="Dimension of feedforward network")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length")
    
    # Evaluation config
    parser.add_argument("--model_path", type=str, default="codegen_model.pth", help="Path to the trained model weights")
    parser.add_argument("--max_gen_length", type=int, default=512, help="Maximum length of the generated sequence")
    parser.add_argument("--limit_eval_batches", type=int, default=None, help="Limit the number of batches for a quick test run")
    parser.add_argument("--force_cpu", action="store_true", help="Force use of CPU even if CUDA is available")

    args = parser.parse_args()
    evaluate(args)

if __name__ == "__main__":
    main()
