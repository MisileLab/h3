import marimo

__generated_with = "0.14.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import torch
    import torch.nn as nn
    from transformers import ElectraModel

    class SpamUserClassifier(nn.Module):
        def __init__(self, pretrained_model_name="beomi/kcelectra-base"):
            super().__init__()
            self.encoder = ElectraModel.from_pretrained(pretrained_model_name)
            # 분류 네트워크
            self.dense1 = nn.Linear(768, 256)
            self.relu = nn.ReLU()
            self.dropout1 = nn.Dropout(0.3)
            self.dense2 = nn.Linear(256, 64)
            self.dropout2 = nn.Dropout(0.2)
            self.output_layer = nn.Linear(64, 2)
            self.softmax = nn.Softmax(dim=1)

        def forward(self, input_ids, attention_mask=None, token_type_ids=None):
            # kcElectra CLS 토큰 추출
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            cls_output = outputs.last_hidden_state[:, 0, :]  # [batch, 768]
            # 분류 네트워크
            x = self.dense1(cls_output)
            x = self.relu(x)
            x = self.dropout1(x)
            x = self.dense2(x)
            x = self.relu(x)
            x = self.dropout2(x)
            logits = self.output_layer(x)
            probs = self.softmax(logits)
            return probs

    model = SpamUserClassifier().to('cuda')
    return model, torch


@app.cell
def _():
    from datasets import load_dataset

    dataset_name = "misilelab/youtube-bot-comments"

    train_dataset = load_dataset(dataset_name, split="train").with_format("polars")[:]
    valid_dataset = load_dataset(dataset_name, split="validation").with_format("polars")[:]
    test_dataset  = load_dataset(dataset_name, split="test").with_format("polars")[:]
    return test_dataset, train_dataset, valid_dataset


@app.cell
def _(model, test_dataset, torch, train_dataset, valid_dataset):
    from transformers import ElectraTokenizerFast, AdamW
    from torch.utils.data import DataLoader, Dataset
    import torch.nn.functional as F
    from tqdm.auto import tqdm

    # prepare tokenizer
    tokenizer = ElectraTokenizerFast.from_pretrained("beomi/kcelectra-base")

    # dataset wrapper
    class YTBotDataset(Dataset):
        def __init__(self, ds, tokenizer, max_length=128):
            self.texts = ds["content"].to_list()
            self.labels = [int(x) for x in ds["is_bot_comment"].to_list()]
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
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item

    # create datasets and loaders
    train_ds = YTBotDataset(train_dataset, tokenizer)
    valid_ds = YTBotDataset(valid_dataset, tokenizer)
    test_ds  = YTBotDataset(test_dataset, tokenizer)

    batch_size = 16
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size)
    test_loader  = DataLoader(test_ds, batch_size=batch_size)

    # optimizer
    optimizer = AdamW(model.parameters(), lr=2e-5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    num_epochs = 3
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            probs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = F.nll_loss(torch.log(probs), labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        avg_train_loss = running_loss / len(train_loader)

        # validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in valid_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                probs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = probs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        valid_acc = correct / total

        print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Valid Acc: {valid_acc:.4f}")

    # final test evaluation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            probs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = probs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    test_acc = correct / total
    print(f"Test Accuracy: {test_acc:.4f}")

    return


if __name__ == "__main__":
    app.run()
