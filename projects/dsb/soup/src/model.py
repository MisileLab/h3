'''
The Transformer model for code generation.
'''
import math
import torch
import torch.nn as nn
from torch.nn import Transformer

class PositionalEncoding(nn.Module):
    '''Injects positional information into the input embeddings.'''
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        '''
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class CodeGenerationModel(nn.Module):
    '''
    An Encoder-Decoder Transformer model for code generation.
    '''
    def __init__(self, vocab_size, d_model=1024, nhead=16, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, max_seq_length=2048):
        super(CodeGenerationModel, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=max_seq_length)
        
        self.transformer = Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True # Ensures input/output tensors are (batch, seq, feature)
        )
        
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt, src_padding_mask=None, tgt_padding_mask=None, memory_key_padding_mask=None):
        # Embed and add positional encoding
        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb)
        
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoder(tgt_emb)

        # Generate causal mask for the decoder
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)

        # Encoder pass
        memory = self.transformer.encoder(
            src_emb, 
            src_key_padding_mask=src_padding_mask
        )
        
        # Decoder pass
        decoder_output = self.transformer.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        
        # Final output for language modeling
        output = self.fc_out(decoder_output)
        
        # Pool embeddings for contrastive loss
        # We use the mean of the token embeddings as the sentence embedding.
        src_embedding = memory.mean(dim=1)
        code_embedding = decoder_output.mean(dim=1)
        
        return output, src_embedding, code_embedding

if __name__ == '__main__':
    # This block is for demonstration and testing purposes.
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from tokenizer import get_tokenizer

    # Initialize tokenizer to get vocab size
    tokenizer = get_tokenizer()
    vocab_size = tokenizer.n_vocab

    # Hyperparameters from the project plan
    model = CodeGenerationModel(
        vocab_size=vocab_size,
        d_model=1024,
        nhead=16,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048, # As per plan
        max_seq_length=2048
    )

    print(f"Model created with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters.")

    # Create dummy input tensors
    src = torch.randint(0, vocab_size, (2, 15)) 
    tgt = torch.randint(0, vocab_size, (2, 25))

    # Get model output
    output, src_emb, code_emb = model(src, tgt)
    
    print(f"Source tensor shape: {src.shape}")
    print(f"Target tensor shape: {tgt.shape}")
    print(f"Model output shape: {output.shape}")
    print(f"Source embedding shape: {src_emb.shape}")
    print(f"Code embedding shape: {code_emb.shape}")