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
            inputs = inputs.squeeze(-1)  # [batch_size]
            bce_loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction='none')
            pt = torch.exp(-bce_loss)
            focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

            if self.reduction == 'mean':
                return focal_loss.mean()
            elif self.reduction == 'sum':
                return focal_loss.sum()
            else:
                return focal_loss

    class SpamUserClassificationLayer(nn.Module):
        def __init__(self, encoder: ElectraModel):
            super().__init__()

            self.encoder = encoder

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

            # Initialize weights properly
            self._init_weights()

        def _init_weights(self):
            """Initialize weights using Xavier/Glorot initialization"""
            for module in [self.dense1, self.dense2]:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        def forward(self, input_ids, attention_mask=None, token_type_ids=None):
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

            return x

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

    class SpamUserClassifier(nn.Module):
        def __init__(self, pretrained_model_name="beomi/kcelectra-base"):
            super().__init__()

            self.encoder = ElectraModel.from_pretrained(pretrained_model_name)

            # Freeze first 2 layers for imbalanced dataset scenario
            for i, layer in enumerate(self.encoder.encoder.layer):
                if i < 2:
                    for param in layer.parameters():
                        param.requires_grad = False

            self.nameLayer = SpamUserClassificationLayer(self.encoder)
            self.contentLayer = SpamUserClassificationLayer(self.encoder)

            self.dense = nn.Linear(512, 256)
            self.layernorm = nn.LayerNorm(256)
            self.gelu = nn.GELU()
            self.dropout = nn.Dropout(0.3)

            self.output_layer = nn.Linear(256, 1)
            self.sigmoid = nn.Sigmoid()

            # Initialize weights properly
            self._init_weights()

        def _init_weights(self):
            """Initialize weights using Xavier/Glorot initialization"""
            nn.init.xavier_uniform_(self.dense.weight)
            if self.dense.bias is not None:
                nn.init.constant_(self.dense.bias, 0)

        # def forward(self, input_ids, attention_mask=None, token_type_ids=None, return_logits=False):
        def forward(self, name_input_ids, content_input_ids, name_attention_mask=None, name_token_type_ids=None,
                   content_attention_mask=None, content_token_type_ids=None, return_logits=False):

            namePrediction = self.nameLayer(name_input_ids, name_attention_mask, name_token_type_ids)
            contentPrediction = self.contentLayer(content_input_ids, content_attention_mask, content_token_type_ids)

            # Pass through classification network
            x = self.dense(torch.cat([namePrediction, contentPrediction], dim=1))
            x = self.layernorm(x)
            x = self.gelu(x)
            x = self.dropout(x)

            logits = self.output_layer(x)

            if return_logits:
                return logits
            else:
                # Apply sigmoid and return probabilities or predictions
                probs = self.sigmoid(logits)
                # Return class predictions: 0 (not bot) or 1 (bot)
                return (probs > 0.5).long().squeeze(-1)
    return (
        AdamW,
        AutoTokenizer,
        DataLoader,
        Dataset,
        FocalLoss,
        SpamUserClassifier,
        alt,
        pl,
        torch,
    )


@app.cell
def _():
    from datasets import load_dataset

    dataset_name = "misilelab/youtube-bot-comments"

    train_dataset = load_dataset(dataset_name, split="train").with_format("polars")[:]
    valid_dataset = load_dataset(dataset_name, split="validation").with_format("polars")[:]
    test_dataset  = load_dataset(dataset_name, split="test").with_format("polars")[:]
    return test_dataset, train_dataset, valid_dataset


@app.cell
def _(
    AdamW,
    AutoTokenizer,
    DataLoader,
    Dataset,
    FocalLoss,
    SpamUserClassifier,
    alt,
    mo,
    pl,
    test_dataset,
    torch,
    train_dataset,
    valid_dataset,
):
    # prepare tokenizer
    tokenizer = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base")

    # dataset wrapper
    class YTBotDataset(Dataset):
        def __init__(self, ds, tokenizer, max_length=128):
            self.author_names = ds["author_name"].to_list()
            self.contents = ds["content"].to_list()
            self.labels = [int(x) for x in ds["is_bot_comment"].to_list()]
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            author_name = self.author_names[idx]
            content = self.contents[idx]

            # Tokenize author name
            name_encoding = self.tokenizer(
                author_name,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )

            # Tokenize content
            content_encoding = self.tokenizer(
                content,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )

            item = {}
            # Add name encodings with prefix
            for k, v in name_encoding.items():
                item[f"name_{k}"] = v.squeeze(0)

            # Add content encodings with prefix
            for k, v in content_encoding.items():
                item[f"content_{k}"] = v.squeeze(0)

            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item

    # create datasets and loaders
    train_ds = YTBotDataset(train_dataset, tokenizer)
    valid_ds = YTBotDataset(valid_dataset, tokenizer)
    test_ds  = YTBotDataset(test_dataset, tokenizer)

    batch_size = 74
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size)
    test_loader  = DataLoader(test_ds, batch_size=batch_size)

    # Initialize model and move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpamUserClassifier().to(device)

    # Create Focal Loss for imbalanced datasets
    criterion = FocalLoss(alpha=1.0, gamma=2.0)

    # optimizer - using different learning rates for frozen and unfrozen layers
    optimizer = AdamW([
        {'params': [p for n, p in model.named_parameters() if 'encoder' in n and p.requires_grad], 'lr': 1e-5},
        {'params': [p for n, p in model.named_parameters() if 'encoder' not in n], 'lr': 2e-5}
    ], weight_decay=0.01)

    # training setup
    num_epochs = 100
    patience = 5
    best_valid_acc = 0.0
    no_improve_epochs = 0

    # Initialize training history
    training_history = {
        'epochs': [],
        'train_losses': [],
        'valid_losses': [],
        'valid_accuracies': []
    }

    # create a top-level progress bar for all epochs
    for epoch in (progress_bar := mo.status.progress_bar(range(1, num_epochs + 1), show_eta=True, show_rate=True)):
        # training
        progress_bar.completion_title = f"epoch {epoch}"
        model.train()
        running_loss = 0.0

        for i, batch in enumerate(mo.status.progress_bar(
            train_loader,
            subtitle=f"Training Epoch {epoch}",
            show_eta=True,
            show_rate=True,
            remove_on_exit=True
        )):
            # Extract name inputs
            name_input_ids = batch["name_input_ids"].to(device)
            name_attention_mask = batch["name_attention_mask"].to(device)
            name_token_type_ids = batch.get("name_token_type_ids", None)
            if name_token_type_ids is not None:
                name_token_type_ids = name_token_type_ids.to(device)

            # Extract content inputs
            content_input_ids = batch["content_input_ids"].to(device)
            content_attention_mask = batch["content_attention_mask"].to(device)
            content_token_type_ids = batch.get("content_token_type_ids", None)
            if content_token_type_ids is not None:
                content_token_type_ids = content_token_type_ids.to(device)

            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            # Get logits for training (not probabilities)
            logits = model(
                name_input_ids=name_input_ids,
                content_input_ids=content_input_ids,
                name_attention_mask=name_attention_mask,
                name_token_type_ids=name_token_type_ids,
                content_attention_mask=content_attention_mask,
                content_token_type_ids=content_token_type_ids,
                return_logits=True
            )

            # Use Focal Loss directly (now supports binary classification)
            loss = criterion(logits, labels)
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        # validation
        model.eval()
        correct, total = 0, 0
        valid_running_loss = 0.0

        for i, batch in enumerate(mo.status.progress_bar(
            valid_loader,
            subtitle=f"Validating Epoch {epoch}",
            show_eta=True,
            show_rate=True,
            remove_on_exit=True
        )):
            with torch.no_grad():
                # Extract name inputs
                name_input_ids = batch["name_input_ids"].to(device)
                name_attention_mask = batch["name_attention_mask"].to(device)
                name_token_type_ids = batch.get("name_token_type_ids", None)
                if name_token_type_ids is not None:
                    name_token_type_ids = name_token_type_ids.to(device)

                # Extract content inputs
                content_input_ids = batch["content_input_ids"].to(device)
                content_attention_mask = batch["content_attention_mask"].to(device)
                content_token_type_ids = batch.get("content_token_type_ids", None)
                if content_token_type_ids is not None:
                    content_token_type_ids = content_token_type_ids.to(device)

                labels = batch["labels"].to(device)

                # Get logits for loss calculation
                logits = model(
                    name_input_ids=name_input_ids,
                    content_input_ids=content_input_ids,
                    name_attention_mask=name_attention_mask,
                    name_token_type_ids=name_token_type_ids,
                    content_attention_mask=content_attention_mask,
                    content_token_type_ids=content_token_type_ids,
                    return_logits=True
                )

                # Calculate validation loss using Focal Loss
                loss = criterion(logits, labels)
                valid_running_loss += loss.item()

                # Get predictions from logits (use model's sigmoid)
                preds = model(
                    name_input_ids=name_input_ids,
                    content_input_ids=content_input_ids,
                    name_attention_mask=name_attention_mask,
                    name_token_type_ids=name_token_type_ids,
                    content_attention_mask=content_attention_mask,
                    content_token_type_ids=content_token_type_ids,
                    return_logits=False
                )
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        valid_acc = correct / total
        avg_valid_loss = valid_running_loss / len(valid_loader)

        # Store training history
        training_history['epochs'].append(epoch)
        training_history['train_losses'].append(avg_train_loss)
        training_history['valid_losses'].append(avg_valid_loss)
        training_history['valid_accuracies'].append(valid_acc)

        print(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}, Valid Acc: {valid_acc:.4f}")

        # early stopping check
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            no_improve_epochs = 0
            # Save best model
            torch.save(model.state_dict(), 'model.pth')
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(f"Early stopping triggered after {epoch} epochs")
                break

    print(f"Training completed. Best validation accuracy: {best_valid_acc:.4f}")

    # Create and display final training chart
    if training_history['epochs']:
        epochs = training_history['epochs']
        train_losses = training_history['train_losses']
        valid_losses = training_history['valid_losses']

        _df = pl.DataFrame({
            'epoch': epochs * 2,
            'loss': train_losses + valid_losses,
            'type': ['Train Loss'] * len(train_losses) + ['Validation Loss'] * len(valid_losses)
        })

        final_chart = alt.Chart(_df).mark_line(point=True).encode(
            x=alt.X('epoch:Q', title='Epoch'),
            y=alt.Y('loss:Q', title='Loss'),
            color=alt.Color('type:N', 
                           scale=alt.Scale(domain=['Train Loss', 'Validation Loss'], 
                                           range=['firebrick', 'royalblue'])),
            tooltip=['epoch:Q', 'loss:Q', 'type:N']
        ).properties(
            title='Training and Validation Loss Over Epochs (Focal Loss)',
            width=700,
            height=400
        ).interactive()

    # Test the final model
    print("\nEvaluating on test set...")
    model.eval()
    test_correct, test_total = 0, 0
    test_predictions = []
    test_labels = []

    for batch in mo.status.progress_bar(test_loader, subtitle="Testing", show_eta=True, remove_on_exit=True):
        with torch.no_grad():
            # Extract name inputs
            name_input_ids = batch["name_input_ids"].to(device)
            name_attention_mask = batch["name_attention_mask"].to(device)
            name_token_type_ids = batch.get("name_token_type_ids", None)
            if name_token_type_ids is not None:
                name_token_type_ids = name_token_type_ids.to(device)

            # Extract content inputs
            content_input_ids = batch["content_input_ids"].to(device)
            content_attention_mask = batch["content_attention_mask"].to(device)
            content_token_type_ids = batch.get("content_token_type_ids", None)
            if content_token_type_ids is not None:
                content_token_type_ids = content_token_type_ids.to(device)

            labels = batch["labels"].to(device)

            # Get logits and convert to predictions
            logits = model(
                name_input_ids=name_input_ids,
                content_input_ids=content_input_ids,
                name_attention_mask=name_attention_mask,
                name_token_type_ids=name_token_type_ids,
                content_attention_mask=content_attention_mask,
                content_token_type_ids=content_token_type_ids,
                return_logits=True
            )

            # Get predictions using model's sigmoid
            preds = model(
                name_input_ids=name_input_ids,
                content_input_ids=content_input_ids,
                name_attention_mask=name_attention_mask,
                name_token_type_ids=name_token_type_ids,
                content_attention_mask=content_attention_mask,
                content_token_type_ids=content_token_type_ids,
                return_logits=False
            )

            test_correct += (preds == labels).sum().item()
            test_total += labels.size(0)

            # Store for detailed analysis
            test_predictions.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    test_acc = test_correct / test_total
    print(f"Test Accuracy: {test_acc:.4f}")
    return (
        device,
        final_chart,
        model,
        test_labels,
        test_loader,
        test_predictions,
        tokenizer,
    )


@app.cell
def _(final_chart):
    final_chart
    return


@app.cell
def _():
    model_save_path = "model.pth"
    return (model_save_path,)


@app.cell
def _(model, model_save_path, torch):
    # Save the trained model's state_dict
    torch.save(model.state_dict(), model_save_path)
    model_save_path
    return


@app.cell
def _(mo):
    mo.md(r"""# Test""")
    return


@app.cell
def _(test_labels, test_predictions):
    # Calculate additional metrics for imbalanced dataset evaluation
    from sklearn.metrics import classification_report, confusion_matrix
    print("\nDetailed Classification Report:")
    print(classification_report(test_labels, test_predictions, target_names=['Normal', 'Spam']))
    print("\nConfusion Matrix:")
    print(confusion_matrix(test_labels, test_predictions))
    return


@app.cell
def _(device, mo, model, test_loader, torch):
    def _():
        import seaborn as sns
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix, accuracy_score

        # Load saved model state
        model.load_state_dict(torch.load('model.pth', map_location=device))
        model.to(device)
        model.eval()

        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch in mo.status.progress_bar(
                test_loader,
                title="Computing Confusion Matrix",
                show_eta=True,
                show_rate=True
            ):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                token_type_ids = batch.get("token_type_ids", None)
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(device)
                labels = batch["labels"].to(device)

                # Get raw logits for probability calculation
                logits = model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    return_logits=True
                )

                # Calculate probabilities using model's sigmoid
                probs = torch.sigmoid(logits.squeeze(-1))  # Bot probability

                # Use custom threshold (0.8 for high precision)
                preds = (probs > 0.8).long()

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        # Compute metrics
        cm = confusion_matrix(all_labels, all_preds)
        acc = accuracy_score(all_labels, all_preds)
        print(f"Accuracy with 0.8 threshold: {acc * 100:.2f}%")

        # Print additional metrics
        tn, fp, fn, tp = cm.ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print(f"Precision: {precision * 100:.2f}%")
        print(f"Recall: {recall * 100:.2f}%")
        print(f"F1-Score: {f1 * 100:.2f}%")
        print(f"True Negatives: {tn}, False Positives: {fp}")
        print(f"False Negatives: {fn}, True Positives: {tp}")

        # Plot heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Normal User', 'Bot/Spam'],
            yticklabels=['Normal User', 'Bot/Spam'],
            cbar_kws={'label': 'Count'}
        )
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix - Spam Detection (Threshold: 0.8)')
        plt.tight_layout()

        # Add text annotations for better interpretation
        plt.text(0.5, -0.1, f'Accuracy: {acc:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}', 
                 ha='center', va='top', transform=plt.gca().transAxes, fontsize=10)

        # Optional: Also create a probability distribution plot
        def plot_probability_distribution():
            # Separate probabilities by true labels
            normal_probs = [prob for prob, label in zip(all_probs, all_labels) if label == 0]
            spam_probs = [prob for prob, label in zip(all_probs, all_labels) if label == 1]

            plt.figure(figsize=(10, 6))
            plt.hist(normal_probs, bins=50, alpha=0.7, label='Normal Users', color='blue', density=True)
            plt.hist(spam_probs, bins=50, alpha=0.7, label='Bot/Spam', color='red', density=True)
            plt.axvline(x=0.5, color='green', linestyle='--', label='Default Threshold (0.5)')
            plt.axvline(x=0.8, color='orange', linestyle='--', label='Current Threshold (0.8)')
            plt.xlabel('Predicted Bot Probability')
            plt.ylabel('Density')
            plt.title('Distribution of Predicted Probabilities by True Label')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            return plt.gca()

        # Uncomment to see probability distribution
        return (plot_probability_distribution(), plt.gca())

    _()
    return


@app.cell
def _(mo):
    mo.md(r"""# Evaluation""")
    return


@app.cell
def _(device, model, tokenizer, torch):
    def _():
       # Evaluate a single user input comment
       model.eval()
       comment = input("Enter a YouTube comment to evaluate: ")

       # Tokenize and prepare tensors
       encoding = tokenizer(
           comment,
           truncation=True,
           padding="max_length",
           max_length=128,
           return_tensors="pt"
       )
       input_ids = encoding["input_ids"].to(device)
       attention_mask = encoding["attention_mask"].to(device)

       # Get prediction using model's built-in sigmoid and threshold
       with torch.no_grad():
           prediction = model(
               input_ids=input_ids, 
               attention_mask=attention_mask, 
               return_logits=False
           )
           is_bot = prediction.item()

       # Print result
       result = "Bot" if is_bot == 1 else "Not Bot"
       return print(f"Prediction: {result}")

    _()
    return


if __name__ == "__main__":
    app.run()
