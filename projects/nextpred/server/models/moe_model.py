"""
Mixture of Experts (MoE) Model for Next Action Prediction
Implements a shared encoder with specialized expert networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import math
import logging

logger = logging.getLogger(__name__)


class NextActionMoE(nn.Module):
    """
    Mixture of Experts model for next action prediction
    
    Architecture:
    - Shared Transformer Encoder (4 layers, 512 dim, 8 heads)
    - Router Network: Decides action type [tab, search, scroll]
    - 3 Expert Networks:
      1. TabSwitchExpert: Predicts tab index (regression)
      2. SearchQueryGenerator: Generates search text (seq2seq)
      3. ScrollPositionPredictor: Predicts scroll % (regression)
    """
    
    def __init__(
        self,
        vocab_size: int = 10000,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_length: int = 128
    ):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        
        # Input embedding layers
        self.temporal_embedding = TemporalFeatureEncoder(d_model)
        self.url_embedding = URLFeatureEncoder(vocab_size, d_model)
        self.behavior_embedding = BehaviorFeatureEncoder(d_model)
        
        # Feature fusion
        self.feature_fusion = nn.Linear(d_model * 3, d_model)
        
        # Shared transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.shared_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_encoder_layers
        )
        
        # Router network
        self.router = RouterNetwork(d_model)
        
        # Expert networks
        self.tab_expert = TabSwitchExpert(d_model)
        self.search_expert = SearchQueryExpert(d_model, vocab_size)
        self.scroll_expert = ScrollPositionExpert(d_model)
        
        # Output processing
        self.output_combiner = OutputCombiner(d_model)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, 0, 0.1)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
    
    def forward(
        self, 
        temporal_features: torch.Tensor,
        url_features: torch.Tensor,
        behavior_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the MoE model
        
        Args:
            temporal_features: Temporal features [batch_size, temporal_dim]
            url_features: URL features [batch_size, url_seq_len]
            behavior_features: Behavior features [batch_size, behavior_dim]
            attention_mask: Attention mask for transformer
            
        Returns:
            Dictionary containing:
            - router_weights: Router probabilities [batch_size, 3]
            - tab_predictions: Tab switch predictions [batch_size, max_tabs]
            - search_predictions: Search query predictions [batch_size, seq_len, vocab_size]
            - scroll_predictions: Scroll position predictions [batch_size, 1]
        """
        batch_size = temporal_features.size(0)
        
        # Encode different feature types
        temporal_encoded = self.temporal_embedding(temporal_features)  # [batch_size, d_model]
        url_encoded = self.url_embedding(url_features)  # [batch_size, seq_len, d_model]
        behavior_encoded = self.behavior_embedding(behavior_features)  # [batch_size, d_model]
        
        # Fuse features
        # Expand temporal and behavior to match URL sequence length
        seq_len = url_encoded.size(1)
        temporal_expanded = temporal_encoded.unsqueeze(1).expand(-1, seq_len, -1)
        behavior_expanded = behavior_encoded.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Concatenate and fuse
        fused_features = torch.cat([
            temporal_expanded, 
            url_encoded, 
            behavior_expanded
        ], dim=-1)  # [batch_size, seq_len, d_model * 3]
        
        fused_features = self.feature_fusion(fused_features)  # [batch_size, seq_len, d_model]
        
        # Apply shared encoder
        if attention_mask is not None:
            # Convert attention mask to the right format
            attention_mask = attention_mask.bool()
            # Invert mask for transformer (True means keep)
            attention_mask = ~attention_mask
        
        encoded_features = self.shared_encoder(
            fused_features, 
            src_key_padding_mask=attention_mask
        )  # [batch_size, seq_len, d_model]
        
        # Global pooling (use mean of sequence)
        pooled_features = encoded_features.mean(dim=1)  # [batch_size, d_model]
        
        # Router decision
        router_weights = self.router(pooled_features)  # [batch_size, 3]
        
        # Expert predictions
        tab_predictions = self.tab_expert(pooled_features)  # [batch_size, max_tabs]
        search_predictions = self.search_expert(pooled_features)  # [batch_size, seq_len, vocab_size]
        scroll_predictions = self.scroll_expert(pooled_features)  # [batch_size, 1]
        
        # Combine outputs using router weights
        final_outputs = self.output_combiner(
            router_weights,
            tab_predictions,
            search_predictions,
            scroll_predictions
        )
        
        return {
            'router_weights': router_weights,
            'tab_predictions': tab_predictions,
            'search_predictions': search_predictions,
            'scroll_predictions': scroll_predictions,
            'final_outputs': final_outputs
        }


class TemporalFeatureEncoder(nn.Module):
    """Encoder for temporal features (time, day, etc.)"""
    
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
        # Temporal features: hour, minute, day_of_week, is_weekend, time_on_page, etc.
        self.temporal_dim = 10
        
        self.linear = nn.Linear(self.temporal_dim, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, temporal_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            temporal_features: [batch_size, temporal_dim]
        Returns:
            encoded: [batch_size, d_model]
        """
        # Ensure correct size
        if temporal_features.size(-1) != self.temporal_dim:
            # Pad or truncate as needed
            if temporal_features.size(-1) > self.temporal_dim:
                temporal_features = temporal_features[:, :self.temporal_dim]
            else:
                padding = torch.zeros(
                    temporal_features.size(0), 
                    self.temporal_dim - temporal_features.size(-1),
                    device=temporal_features.device
                )
                temporal_features = torch.cat([temporal_features, padding], dim=-1)
        
        encoded = self.linear(temporal_features)
        encoded = self.layer_norm(encoded)
        encoded = F.relu(encoded)
        encoded = self.dropout(encoded)
        
        return encoded


class URLFeatureEncoder(nn.Module):
    """Encoder for URL and domain features"""
    
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.d_model = d_model
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, url_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            url_features: [batch_size, seq_len] (tokenized URLs)
        Returns:
            encoded: [batch_size, seq_len, d_model]
        """
        # Embed tokens
        embedded = self.embedding(url_features)
        embedded = self.positional_encoding(embedded)
        embedded = self.layer_norm(embedded)
        embedded = self.dropout(embedded)
        
        return embedded


class BehaviorFeatureEncoder(nn.Module):
    """Encoder for user behavior features"""
    
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
        # Behavior features: tab_switch_freq, search_freq, scroll_freq, etc.
        self.behavior_dim = 15
        
        self.linear = nn.Linear(self.behavior_dim, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, behavior_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            behavior_features: [batch_size, behavior_dim]
        Returns:
            encoded: [batch_size, d_model]
        """
        # Ensure correct size
        if behavior_features.size(-1) != self.behavior_dim:
            if behavior_features.size(-1) > self.behavior_dim:
                behavior_features = behavior_features[:, :self.behavior_dim]
            else:
                padding = torch.zeros(
                    behavior_features.size(0), 
                    self.behavior_dim - behavior_features.size(-1),
                    device=behavior_features.device
                )
                behavior_features = torch.cat([behavior_features, padding], dim=-1)
        
        encoded = self.linear(behavior_features)
        encoded = self.layer_norm(encoded)
        encoded = F.relu(encoded)
        encoded = self.dropout(encoded)
        
        return encoded


class RouterNetwork(nn.Module):
    """Router network for expert selection"""
    
    def __init__(self, d_model: int, num_experts: int = 3):
        super().__init__()
        self.num_experts = num_experts
        
        self.router = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_experts)
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [batch_size, d_model]
        Returns:
            router_weights: [batch_size, num_experts] (softmax probabilities)
        """
        logits = self.router(features)
        router_weights = F.softmax(logits, dim=-1)
        return router_weights


class TabSwitchExpert(nn.Module):
    """Expert network for tab switch predictions"""
    
    def __init__(self, d_model: int, max_tabs: int = 20):
        super().__init__()
        self.max_tabs = max_tabs
        
        self.predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 4, max_tabs)
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [batch_size, d_model]
        Returns:
            predictions: [batch_size, max_tabs] (logits for each tab)
        """
        predictions = self.predictor(features)
        return predictions


class SearchQueryExpert(nn.Module):
    """Expert network for search query generation"""
    
    def __init__(self, d_model: int, vocab_size: int, max_seq_length: int = 32):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        
        # Decoder for sequence generation
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=d_model * 2,
            dropout=0.1,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=3)
        
        self.output_projection = nn.Linear(d_model, vocab_size)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [batch_size, d_model] (encoded context)
        Returns:
            predictions: [batch_size, max_seq_length, vocab_size]
        """
        batch_size = features.size(0)
        
        # Create target sequence (start with SOS token)
        # For simplicity, generate a fixed-length sequence
        target_seq = torch.zeros(
            batch_size, self.max_seq_length, 
            dtype=torch.long, device=features.device
        )
        
        # Embed target sequence
        target_embedded = self.embedding(target_seq)
        target_embedded = self.positional_encoding(target_embedded)
        
        # Use features as memory for decoder
        memory = features.unsqueeze(1)  # [batch_size, 1, d_model]
        
        # Decode
        decoded = self.decoder(
            target_embedded, 
            memory
        )  # [batch_size, max_seq_length, d_model]
        
        # Project to vocabulary
        predictions = self.output_projection(decoded)
        
        return predictions


class ScrollPositionExpert(nn.Module):
    """Expert network for scroll position prediction"""
    
    def __init__(self, d_model: int):
        super().__init__()
        
        self.predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()  # Output between 0 and 1 (scroll percentage)
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [batch_size, d_model]
        Returns:
            predictions: [batch_size, 1] (scroll position 0-1)
        """
        predictions = self.predictor(features)
        return predictions


class OutputCombiner(nn.Module):
    """Combine expert outputs using router weights"""
    
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
    
    def forward(
        self,
        router_weights: torch.Tensor,
        tab_predictions: torch.Tensor,
        search_predictions: torch.Tensor,
        scroll_predictions: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Combine expert outputs using router weights
        
        Args:
            router_weights: [batch_size, 3]
            tab_predictions: [batch_size, max_tabs]
            search_predictions: [batch_size, seq_len, vocab_size]
            scroll_predictions: [batch_size, 1]
        
        Returns:
            Combined outputs with router weighting applied
        """
        # Apply router weights to expert outputs
        weighted_tab = router_weights[:, 0:1] * tab_predictions
        weighted_search = router_weights[:, 1:2] * search_predictions
        weighted_scroll = router_weights[:, 2:3] * scroll_predictions
        
        return {
            'weighted_tab': weighted_tab,
            'weighted_search': weighted_search,
            'weighted_scroll': weighted_scroll,
            'router_weights': router_weights
        }


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            x with positional encoding added
        """
        seq_len = x.size(1)
        x = x + self.pe[:seq_len, :].transpose(0, 1)
        return x


def create_model(config: Optional[Dict] = None) -> NextActionMoE:
    """Create and return a NextActionMoE model"""
    if config is None:
        config = {
            'vocab_size': 10000,
            'd_model': 512,
            'nhead': 8,
            'num_encoder_layers': 4,
            'dim_feedforward': 2048,
            'dropout': 0.1,
            'max_seq_length': 128
        }
    
    model = NextActionMoE(**config)
    logger.info(f"Created NextActionMoE model with config: {config}")
    
    return model


if __name__ == "__main__":
    # Test the model
    model = create_model()
    
    # Create dummy inputs
    batch_size = 2
    temporal_features = torch.randn(batch_size, 10)
    url_features = torch.randint(0, 1000, (batch_size, 32))
    behavior_features = torch.randn(batch_size, 15)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(temporal_features, url_features, behavior_features)
    
    print("Model outputs:")
    for key, value in outputs.items():
        print(f"{key}: {value.shape}")
    
    print(f"Router weights: {outputs['router_weights']}")