import polars as pl
import json
from typing import Dict, Any

def extract_law_text(row: Dict[str, Any]) -> str:
    """Extract text from law row."""
    return row.get('본문', '')

def extract_precedent_text(row: Dict[str, Any]) -> str:
    """Extract text from precedent row. Adjust based on actual schema."""
    # Example: assuming 'precService' contains the main content
    # Print row.keys() to inspect if needed
    text_parts = []
    if 'precService' in row:
        prec = row['precService']
        if isinstance(prec, dict):
            # Common fields: '판례요지', '판결주요내용', '판례본문' etc.
            for field in ['판례요지', '판결주요내용', '판례본문', '본문']:
                if field in prec:
                    text_parts.append(prec[field])
        elif isinstance(prec, list):
            for p in prec:
                if isinstance(p, dict):
                    for field in ['판례요지', '판결주요내용', '판례본문', '본문']:
                        if field in p:
                            text_parts.append(p[field])
    # Fallback to string representation if no text found
    if not text_parts:
        text_parts = [str(row)]
    return '\n\n'.join(text_parts)

# Load parquet files
df_laws = pl.read_parquet("korean_labor_laws.parquet")
df_precedents = pl.read_parquet("korean_labor_precedents_with_content.parquet")

examples = []

# Process laws
for row_dict in df_laws.iter_rows(named=True):
    title = row_dict.get('법령명', row_dict.get('법령명한', 'Unknown'))
    body = extract_law_text(row_dict)
    if body and len(body) > 50 and '오류' not in body and '없음' not in body:
        example = {
            "contents": [
                {"role": "user", "parts": [{"text": f"한국 노동법 '{title}'의 전체 내용을 제공해주세요."}]},
                {"role": "model", "parts": [{"text": body}]}
            ]
        }
        examples.append(example)

# Process precedents
for row_dict in df_precedents.iter_rows(named=True):
    title = row_dict.get('판례명', row_dict.get('사건번호', row_dict.get('판례일련번호', 'Unknown')))
    body = extract_precedent_text(row_dict)
    if body and len(body) > 50:
        example = {
            "contents": [
                {"role": "user", "parts": [{"text": f"한국 노동 판례 '{title}'의 전체 내용을 제공해주세요."}]},
                {"role": "model", "parts": [{"text": body}]}
            ]
        }
        examples.append(example)

# Save to JSONL for Gemini fine-tuning
output_file = "fine_tune_dataset.jsonl"
with open(output_file, "w", encoding="utf-8") as f:
    for ex in examples:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

print(f"Created fine-tuning dataset with {len(examples)} examples in {output_file}")
print("Upload this JSONL to a GCS bucket and use Vertex AI to create a tuning job for Gemini.")