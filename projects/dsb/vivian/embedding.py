#!pip install -U transformers sentence-transformers tqdm pydantic polars accelerate
#!apt install zstd
#!zstd -d --rm embedding_data.avro.zst
import gc
from contextlib import suppress

from polars import DataFrame, concat, read_avro
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
# Configuration - Set your VRAM amount here (in GB)
# Common GPU VRAM amounts:
# RTX 4090: 24GB, RTX 4080: 16GB, RTX 4070 Ti: 12GB, RTX 4060 Ti: 8GB
# RTX 3080: 10GB, RTX 3070: 8GB, RTX 3060: 6GB, RTX 3050: 4GB
VRAM_GB = 12  # Change this to match your GPU's VRAM

# Model requirements for Qwen3-Embedding-4B
MODEL_BASE_MEMORY_GB = 8.5  # Model size + framework overhead
MINIMUM_VRAM_GB = 10  # Minimum VRAM required to run the model
VRAM_DIVISOR = 0.5  # Constant to calculate batch size from available VRAM

processed = read_avro("embedding_data.avro")
df = DataFrame()

class Embedding(BaseModel):
  parent_comment_author: list[float]
  parent_comment_content: list[float]
  comment_author: list[float]
  comment_content: list[float]
  is_bot_comment: int

def get_optimal_batch_size(vram_gb: float) -> int:
  """Determine optimal batch size based on specified VRAM using division formula"""
  print(f"Configured VRAM: {vram_gb} GB")
  
  # Check minimum VRAM requirement
  if vram_gb < MINIMUM_VRAM_GB:
    print(f"ERROR: Insufficient VRAM! Qwen3-Embedding-4B requires at least {MINIMUM_VRAM_GB}GB VRAM")
    print(f"Your configured VRAM: {vram_gb}GB")
    print("Please upgrade your GPU or use a smaller model like Qwen3-Embedding-0.6B")
    return 1  # Return minimal batch size but warn user
  
  # Calculate available VRAM after model loading
  available_vram = vram_gb - MODEL_BASE_MEMORY_GB
  print(f"Available VRAM for batch processing: {available_vram:.1f}GB")
  
  # Calculate batch size using division formula
  # batch_size = available_vram / VRAM_DIVISOR
  calculated_batch_size = int(available_vram / VRAM_DIVISOR)
  
  # Ensure minimum batch size of 1 and maximum reasonable limit
  batch_size = max(1, min(calculated_batch_size, 512))
  
  estimated_usage = MODEL_BASE_MEMORY_GB + (batch_size * VRAM_DIVISOR)
  
  print(f"Calculated batch size: {batch_size}")
  print(f"Estimated total VRAM usage: {estimated_usage:.1f}GB")
  print(f"VRAM utilization: {(estimated_usage / vram_gb) * 100:.1f}%")
  
  return batch_size

model = SentenceTransformer("Qwen/Qwen3-Embedding-4B")

def append_batch(df: DataFrame, embeddings_batch: list[Embedding]) -> DataFrame:
  """Append a batch of embeddings to the dataframe"""
  if not embeddings_batch:
    return df
  
  batch_data = [emb.model_dump() for emb in embeddings_batch]
  batch_df = DataFrame(batch_data)
  return concat([df, batch_df], how="vertical", rechunk=True)

def process_batch(batch_data: list[dict[str, str]], model: SentenceTransformer) -> list[Embedding]:
  """Process a batch of data and return embeddings"""
  if not batch_data:
    return []
  
  # Prepare all texts for batch encoding
  all_texts = []
  for item in batch_data:
    all_texts.extend([ # pyright: ignore[reportUnknownMemberType]
      item["parent_comment_author"],
      item["parent_comment_content"], 
      item["comment_author"],
      item["comment_content"]
    ])
  
  # Encode all texts in one batch
  all_embeddings = model.encode(all_texts, show_progress_bar=False)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
  
  # Split embeddings back into groups of 4 for each item
  embeddings_batch = []
  for i, item in enumerate(batch_data):
    start_idx = i * 4
    is_bot = 1 if item["is_bot_comment"] else 0
    
    embeddings_batch.append(Embedding( # pyright: ignore[reportUnknownMemberType]
      parent_comment_author=all_embeddings[start_idx].tolist(),     # pyright: ignore[reportAny]
      parent_comment_content=all_embeddings[start_idx + 1].tolist(), # pyright: ignore[reportAny]
      comment_author=all_embeddings[start_idx + 2].tolist(),        # pyright: ignore[reportAny]
      comment_content=all_embeddings[start_idx + 3].tolist(),       # pyright: ignore[reportAny]
      is_bot_comment=is_bot
    ))
  
  return embeddings_batch # pyright: ignore[reportUnknownVariableType]

# Configuration - Batch size based on VRAM
BATCH_SIZE = get_optimal_batch_size(VRAM_GB)
SAVE_INTERVAL = 1000  # Save every N batches to avoid data loss

# Clear memory before starting
_ = gc.collect()

# Convert to list for batch processing
data_list = processed.to_dicts()
total_batches = (len(data_list) + BATCH_SIZE - 1) // BATCH_SIZE

print(f"Processing {len(data_list)} items in {total_batches} batches of size {BATCH_SIZE}")

with suppress(KeyboardInterrupt):
# Process in batches with memory management
  for batch_idx in tqdm(range(0, len(data_list), BATCH_SIZE), desc="Processing batches"):
    batch_data = data_list[batch_idx:batch_idx + BATCH_SIZE]
    
    try:
      # Process the batch
      embeddings_batch = process_batch(batch_data, model)
      
      # Append to dataframe
      df = append_batch(df, embeddings_batch)
      
      # Clear memory periodically
      if (batch_idx // BATCH_SIZE + 1) % 10 == 0:
        _ = gc.collect()
      
      # Periodic saving to avoid data loss
      if (batch_idx // BATCH_SIZE + 1) % SAVE_INTERVAL == 0:
        print(f"Saving checkpoint at batch {batch_idx // BATCH_SIZE + 1}")
        df.write_avro(f"embedding_checkpoint_{batch_idx // BATCH_SIZE + 1}.avro")
        
    except Exception as e:
      print(f"Error processing batch {batch_idx // BATCH_SIZE + 1}: {e}")
      # Try with smaller batch if memory error
      if "memory" in str(e).lower() or "cuda" in str(e).lower():
        print("Memory error detected, trying smaller batch size")
        BATCH_SIZE = max(4, BATCH_SIZE - 1) # pyright: ignore[reportConstantRedefinition]
        print(f"New batch size: {BATCH_SIZE}")
        _ = gc.collect()
        
        # Retry current batch with smaller size
        batch_data = data_list[batch_idx:batch_idx + BATCH_SIZE]
        try:
          embeddings_batch = process_batch(batch_data, model)
          df = append_batch(df, embeddings_batch)
        except Exception as retry_e:
          print(f"Retry failed: {retry_e}")
          continue
      else:
        continue

# Final save
df.write_avro("embedding.avro")
print(f"Processing complete! Final dataframe shape: {df.shape}")
