import os
import torch
from torch.utils.data import DataLoader
from transformers import (
  AutoTokenizer,
  DataCollatorForLanguageModeling
)
from datasets import load_dataset
from typing import Dict
from tqdm import tqdm

from gemma_with_tensor import GemmaWithTensorLogic


class CodeDataset:
  """Dataset wrapper using correct column names from code-search-net"""

  def __init__(
    self,
    tokenizer: AutoTokenizer,
    dataset_name: str = "code-search-net/code_search_net",
    subset: str = "python",
    max_length: int = 512
  ):
    self.tokenizer = tokenizer
    self.max_length = max_length
    
    # Load dataset
    self.dataset = load_dataset(dataset_name, subset, split="train[:10%]")
    
    # Filter using correct column name 'code'
    self.dataset = self.dataset.filter(lambda x: x['code'].strip() != '')

  def __len__(self) -> int:
    return len(self.dataset)

  def __getitem__(self, idx: int) -> Dict[str, str]:
    """Return raw text for data collator to process"""
    return {
      "text": self.dataset[idx]['code']
    }


class LogicEvaluator:
  """Evaluates logical consistency of generated code"""

  def __init__(self, model: GemmaWithTensorLogic):
    self.model = model

  def evaluate_consistency(self, tensor_outputs: Dict[str, torch.Tensor]) -> float:
    """
    Calculate logic consistency score from tensor module outputs
    
    Returns:
        Score between 0-1 where higher is better
    """
    # Weighted average of tensor components
    var_dep = tensor_outputs['var_dependency'].mean().item() * 0.2
    type_cons = tensor_outputs['type_consistency'].mean().item() * 0.3
    scope_val = tensor_outputs['scope_validity'].mean().item() * 0.25
    logic_flow = tensor_outputs['logic_flow'].mean().item() * 0.25
    
    return (var_dep + type_cons + scope_val + logic_flow)


def train_model(
  model: GemmaWithTensorLogic,
  train_loader: DataLoader,
  optimizer: torch.optim.Optimizer,
  device: torch.device,
  num_epochs: int = 3,
  gradient_accumulation_steps: int = 4
) -> None:
  """Train the tensor-enhanced model"""
  model.train()
  evaluator = LogicEvaluator(model)

  for epoch in range(num_epochs):
    total_loss = 0.0
    total_consistency = 0.0
    steps = 0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

    for step, batch in enumerate(progress_bar):
      batch = {k: v.to(device) for k, v in batch.items()}

      # Forward pass
      outputs = model(**batch)
      loss = outputs['loss']
      tensor_outputs = outputs['tensor_outputs']

      # Skip if loss is None (shouldn't happen in training)
      if loss is None:
        continue

      # Calculate logic consistency
      consistency_score = evaluator.evaluate_consistency(tensor_outputs)

      # Backward pass
      loss = loss / gradient_accumulation_steps
      loss.backward()

      # Gradient accumulation
      if (step + 1) % gradient_accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
        
        # Track metrics
        total_loss += loss.item() * gradient_accumulation_steps
        total_consistency += consistency_score
        steps += 1

        # Update progress bar
        if steps > 0:
          progress_bar.set_postfix({
            'loss': total_loss / steps,
            'consistency': total_consistency / steps
          })

    # Save checkpoint
    checkpoint_path = f"checkpoints/epoch_{epoch+1}"
    os.makedirs(checkpoint_path, exist_ok=True)
    torch.save(model.state_dict(), f"{checkpoint_path}/model.pt")
    print(f"Saved checkpoint to {checkpoint_path}")


def main() -> None:
  """Main training pipeline"""
  # Configuration
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model_name = "google/gemma-3-270m"
  batch_size = 2
  max_length = 512

  # Create checkpoints directory
  os.makedirs("checkpoints", exist_ok=True)

  # Initialize model and tokenizer
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  model = GemmaWithTensorLogic(model_name).to(device)

  # Prepare dataset
  dataset = CodeDataset(tokenizer, max_length=max_length)
  
  # Setup data collator for causal language modeling
  data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # Causal language modeling
    pad_to_multiple_of=8
  )

  train_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=data_collator
  )

  # Setup optimizer
  optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

  # Train model
  train_model(model, train_loader, optimizer, device)

  print("\nTraining completed successfully! ✨")

if __name__ == "__main__":
  main()
