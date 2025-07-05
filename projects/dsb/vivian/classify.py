import marimo

__generated_with = "0.14.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import ElectraModel, AutoTokenizer
    from torch.utils.data import DataLoader, Dataset
    from torch.optim import AdamW
    import altair as alt
    import polars as pl

    class FocalLoss(nn.Module):
        """Focal Loss for handling class imbalance - supports both binary and multi-class"""
        def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma
            self.reduction = reduction

        def forward(self, inputs, targets):
            # Handle binary classification (single output) vs multi-class
            if inputs.dim() == 2 and inputs.size(1) == 1:
                # Binary classification: inputs shape [batch_size, 1]
                inputs = inputs.squeeze(-1)  # [batch_size]
                bce_loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction='none')
                pt = torch.exp(-bce_loss)
                focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
            else:
                # Multi-class classification: inputs shape [batch_size, num_classes]
                ce_loss = F.cross_entropy(inputs, targets, reduction='none')
                pt = torch.exp(-ce_loss)
                focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

            if self.reduction == 'mean':
                return focal_loss.mean()
            elif self.reduction == 'sum':
                return focal_loss.sum()
            else:
                return focal_loss

    class SpamUserClassifier(nn.Module):
        def __init__(self, pretrained_model_name="beomi/kcelectra-base"):
            super().__init__()

            self.encoder = ElectraModel.from_pretrained(pretrained_model_name)

            # Freeze first 2 layers for imbalanced dataset scenario
            for i, layer in enumerate(self.encoder.encoder.layer):
                if i < 2:
                    for param in layer.parameters():
                        param.requires_grad = False

            # Classification network optimized for imbalanced datasets
            # Changed input dimension from 768 to 1536 (CLS + mean pooling)
            self.dense1 = nn.Linear(1536, 512)
            self.layernorm1 = nn.LayerNorm(512)
            self.gelu1 = nn.GELU()
            self.dropout1 = nn.Dropout(0.4)

            self.dense2 = nn.Linear(512, 256)
            self.layernorm2 = nn.LayerNorm(256)
            self.gelu2 = nn.GELU()
            self.dropout2 = nn.Dropout(0.3)

            self.output_layer = nn.Linear(256, 1)
            self.sigmoid = nn.Sigmoid()

            # Initialize weights properly
            self._init_weights()

        def _init_weights(self):
            """Initialize weights using Xavier/Glorot initialization"""
            for module in [self.dense1, self.dense2, self.output_layer]:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        def forward(self, input_ids, attention_mask=None, token_type_ids=None, return_logits=False):
            # Get encoder outputs
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                output_attentions=True
            )

            # CLS token representation
            cls_output = outputs.last_hidden_state[:, 0, :]  # [batch, 768]

            # Mean pooling with proper attention masking
            token_embeddings = outputs.last_hidden_state  # [batch, seq_len, 768]
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            mean_pooled = sum_embeddings / sum_mask  # [batch, 768]

            # Concatenate CLS + mean pooling
            combined_output = torch.cat([cls_output, mean_pooled], dim=1)  # [batch, 1536]

            # Pass through classification network
            x = self.dense1(combined_output)
            x = self.layernorm1(x)
            x = self.gelu1(x)
            x = self.dropout1(x)

            x = self.dense2(x)
            x = self.layernorm2(x)
            x = self.gelu2(x)
            x = self.dropout2(x)

            logits = self.output_layer(x)

            if return_logits:
                return logits
            else:
                # Apply sigmoid and return probabilities or predictions
                probs = self.sigmoid(logits)
                # Return class predictions: 0 (not bot) or 1 (bot)
                return (probs > 0.5).long().squeeze(-1)

        def get_attention_weights(self, input_ids, attention_mask=None, token_type_ids=None):
            """Extract attention weights for interpretability"""
            with torch.no_grad():
                outputs = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    output_attentions=True
                )
                # Return attention weights from last layer
                return outputs.attentions[-1]
    return AutoTokenizer, DataLoader, Dataset, SpamUserClassifier, pl, torch


@app.cell
def _(pl):
    comments = pl.read_avro("comments.avro")
    return (comments,)


@app.cell
def _(AutoTokenizer, Dataset):
    tokenizer = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base")

    # dataset wrapper
    class YTBotDataset(Dataset):
        def __init__(self, ds, tokenizer, max_length=128):
            self.texts = ds["content"].to_list()
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            text = self.texts[idx]
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )
            item = {k: v.squeeze(0) for k, v in encoding.items()}
            return item
    return YTBotDataset, tokenizer


@app.cell
def _(
    DataLoader,
    SpamUserClassifier,
    YTBotDataset,
    comments,
    tokenizer,
    torch,
):
    data_loader = DataLoader(YTBotDataset(comments, tokenizer), batch_size=148)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpamUserClassifier().to(device)
    return data_loader, device, model


@app.cell
def _(comments, data_loader, device, mo, model, pl, torch):
    # Evaluate model and save predictions
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in mo.status.progress_bar(data_loader, show_eta=True, show_rate=True):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
        
            # Get predictions (returns 0 or 1 for each sample)
            batch_predictions = model(input_ids, attention_mask)
            predictions.extend(batch_predictions.cpu().numpy().tolist())

    # Add predictions to the original dataset
    processed_comments = comments.with_columns(
        pl.Series("is_bot_comment", predictions, dtype=pl.Int32)
    )

    # Save to processed.avro
    processed_comments.write_avro("processed.avro")

    print(f"Processed {len(predictions)} comments")
    print(f"Bot comments detected: {sum(predictions)}")
    print(f"Regular comments: {len(predictions) - sum(predictions)}")
    print("Results saved to processed.avro")
    return


if __name__ == "__main__":
    app.run()
