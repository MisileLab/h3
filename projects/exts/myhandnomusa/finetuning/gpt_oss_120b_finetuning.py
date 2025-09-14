# gpt-oss-120b-finetuning.py
import marimo as mo

# Cell 1: Imports
import marimo as mo
import os
import torch
import pandas as pd
import polars as pl
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import Dataset
import httpx
import asyncio
import json
from tqdm.asyncio import tqdm_asyncio
from dotenv import load_dotenv
import psutil
import GPUtil

# Cell 2: Notebook Title
mo.md(r"""
# 🔬 gpt-oss-120b Fine-Tuning Notebook

This notebook fine-tunes a large language model on Korean legal data.

**Features:**
- **VRAM Check:** Ensures sufficient GPU memory before starting.
- **Resource Monitoring:** Live tracking of CPU, RAM, and GPU usage.
- **Reactive Data Fetching:** Buttons to fetch law and precedent data.
- **Efficient Training:** Uses `transformers`, `pytorch`, and `safetensors`.
""")

# Cell 3: API Key and Configuration
_ = load_dotenv()

mo.md(r"""
## ⚙️ Configuration
Enter your API key from the [National Law Information Center](https://www.law.go.kr/DRF/main.do).
""")

API_KEY = mo.ui.text(
    value=os.environ.get("LAW_API_KEY", ""),
    label="API Key",
    kind="password"
)
API_KEY

# Cell 4: GPU VRAM Check
mo.md(r"""
## 📊 GPU VRAM Check
We need significant GPU VRAM for this model. Let's check the available resources.
We are checking for at least 80GB of VRAM.
""")

MIN_VRAM_GB = 80

def check_gpu_vram():
    if not torch.cuda.is_available():
        return mo.md("🔴 **Error:** No GPU detected. A GPU is required for fine-tuning.")

    gpu_properties = torch.cuda.get_device_properties(0)
    vram_gb = gpu_properties.total_memory / (1024**3)
    
    vram_info = mo.md(f"✅ **GPU:** `{gpu_properties.name}`

✅ **Available VRAM:** `{vram_gb:.2f} GB`")
    
    if vram_gb < MIN_VRAM_GB:
        mo.stop(True, mo.md(f"🔴 **Error:** Insufficient GPU VRAM. Required: `{MIN_VRAM_GB} GB`, but only `{vram_gb:.2f} GB` is available."))
    
    return vram_info

check_gpu_vram()

# Cell 5: trackio - System Resource Monitor
mo.md(r"""
## 📈 System Resource Monitor (trackio)
Live monitoring of system resources during the process.
""")

@mo.cache
def get_system_stats():
    cpu_percent = psutil.cpu_percent(interval=1)
    ram_percent = psutil.virtual_memory().percent
    
    gpus = GPUtil.getGPUs()
    if gpus:
        gpu = gpus[0]
        gpu_load = f"{gpu.load*100:.1f}%"
        gpu_vram = f"{gpu.memoryUsed / gpu.memoryTotal * 100:.1f}% ({gpu.memoryUsed}MB / {gpu.memoryTotal}MB)"
    else:
        gpu_load = "N/A"
        gpu_vram = "N/A"
        
    return cpu_percent, ram_percent, gpu_load, gpu_vram

def create_stat_card(title, value, color):
    return mo.Html(f"""
    <div style="border: 1px solid {color}; border-radius: 5px; padding: 10px; text-align: center;">
        <h3 style="margin: 0;">{title}</h3>
        <p style="font-size: 24px; margin: 5px 0;">{value}</p>
    </div>
    """)

cpu, ram, gpu_load, gpu_vram_usage = get_system_stats()

trackio_dashboard = mo.hstack([
    create_stat_card("CPU Usage", f"{cpu}%", "#4CAF50"),
    create_stat_card("RAM Usage", f"{ram}%", "#2196F3"),
    create_stat_card("GPU Load", gpu_load, "#FF9800"),
    create_stat_card("GPU VRAM", gpu_vram_usage, "#f44336"),
])
trackio_dashboard

# Cell 6: Data Fetching - API Functions
# Functions from get_laws_api.py and get_precedents_api.py
# These are defined here to be used by the data fetching cells.

# --- Law API Functions ---
LAW_LIST_URL = "http://www.law.go.kr/DRF/lawSearch.do"
LAW_DETAIL_URL = "http://www.law.go.kr/DRF/lawService.do"

def parse_law_list(json_data):
    try:
        data = json.loads(json_data)
        if "LawSearch" in data:
            law_search_data = data["LawSearch"]
            items = law_search_data.get("law", [])
            total_count = int(law_search_data.get("totalCnt", "0"))
            return items, total_count
        elif "law" in data:
            items = data["law"]
            total_count = int(data.get("totalCnt", "0"))
            return items, total_count
        else: return [], 0
    except json.JSONDecodeError: return [], 0

def parse_law_detail(json_data):
    try:
        data = json.loads(json_data)
        law_data = data.get("법령")
        if not law_data: return "법령 상세 데이터 없음"
        texts = []
        reason_info = law_data.get("제개정이유")
        if isinstance(reason_info, dict) and "제개정이유내용" in reason_info:
            texts.append(f"[제개정 이유]\n{reason_info['제개정이유내용']}\n")
        articles_info = law_data.get("조문")
        if isinstance(articles_info, dict) and "조문단위" in articles_info:
            article_units = articles_info["조문단위"]
            if isinstance(article_units, dict): article_units = [article_units]
            if isinstance(article_units, list):
                for article in article_units:
                    if isinstance(article, dict):
                        parts = [article.get("조문제목", ""), article.get("조문내용", "")]
                        texts.append("\n".join(filter(None, parts)))
        return "\n".join(texts) if texts else "상세 내용 없음"
    except json.JSONDecodeError: return "상세 내용 파싱 오류"

async def fetch_law_details(client, law_item, pbar):
    law_id = law_item.get("법령ID")
    if not law_id:
        pbar.update(1)
        law_item['본문'] = '법령ID 없음'
        return law_item
    params = {"OC": API_KEY.value, "target": "law", "ID": law_id, "type": "JSON"}
    try:
        response = await client.get(LAW_DETAIL_URL, params=params, timeout=30)
        response.raise_for_status()
        law_item['본문'] = parse_law_detail(response.text)
    except httpx.RequestError as e:
        law_item['본문'] = f"상세 정보 요청 실패: {e}"
    finally:
        pbar.update(1)
        return law_item

# --- Precedent API Functions ---
PRECEDENT_LIST_URL = "http://www.law.go.kr/DRF/lawSearch.do"
PRECEDENT_DETAIL_URL = "http://www.law.go.kr/DRF/lawService.do"

def parse_precedent_list(json_data):
    try:
        data = json.loads(json_data)
        for key in ["PrecSearch", "lawSearch"]:
            if key in data and "prec" in data[key]:
                search_data = data[key]
                return search_data.get("prec", []), int(search_data.get("totalCnt", "0"))
        if "prec" in data:
            return data["prec"], int(data.get("totalCnt", "0"))
        return [], 0
    except json.JSONDecodeError: return [], 0

def parse_precedent_detail(json_data):
    try:
        return json.loads(json_data)
    except json.JSONDecodeError: return None

async def fetch_precedent_detail(client, precedent_id):
    params = {"OC": API_KEY.value, "target": "prec", "ID": precedent_id, "type": "JSON"}
    try:
        response = await client.get(PRECEDENT_DETAIL_URL, params=params, timeout=30)
        response.raise_for_status()
        return parse_precedent_detail(response.text)
    except Exception: return None

# Cell 7: Fetch Law Data
mo.md(r"""
### 1. Fetch Korean Labor Laws
Click the button to fetch law data using the API. The data will be saved to `korean_labor_laws.parquet`.
""")

fetch_laws_button = mo.ui.button(label="Fetch Law Data")
fetch_laws_button

# Cell 8: Law Data Fetching Logic
@mo.cache
async def get_law_data():
    if fetch_laws_button.value:
        all_laws = []
        page = 1
        total_count = 0
        async with httpx.AsyncClient() as client:
            while True:
                params = {"OC": API_KEY.value, "target": "law", "query": "노동", "display": 100, "page": page, "type": "JSON"}
                response = await client.get(LAW_LIST_URL, params=params)
                laws, current_total = parse_law_list(response.text)
                if page == 1: total_count = current_total
                if not laws: break
                all_laws.extend(laws)
                if len(all_laws) >= total_count: break
                page += 1
            
            progress_bar = tqdm_asyncio(total=len(all_laws), desc="Fetching Law Details")
            tasks = [fetch_law_details(client, item, progress_bar) for item in all_laws]
            results = await asyncio.gather(*tasks)
            
            df = pl.DataFrame(results)
            df.write_parquet("korean_labor_laws.parquet")
            return df
    return None

law_df = get_law_data()
law_df

# Cell 9: Fetch Precedent Data
mo.md(r"""
### 2. Fetch Korean Labor Precedents
Click the button to fetch precedent data. This will be saved to `korean_labor_precedents.parquet`.
""")

fetch_precedents_button = mo.ui.button(label="Fetch Precedent Data")
fetch_precedents_button

# Cell 10: Precedent Data Fetching Logic
@mo.cache
async def get_precedent_data():
    if fetch_precedents_button.value:
        all_precedents_list = []
        page = 1
        total_count = 0
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, read=60.0)) as client:
            while True:
                params = {"OC": API_KEY.value, "target": "prec", "query": "노동", "display": 100, "page": page, "type": "JSON"}
                response = await client.get(PRECEDENT_LIST_URL, params=params)
                precedents, current_total = parse_precedent_list(response.text)
                if page == 1: total_count = current_total
                if not precedents: break
                all_precedents_list.extend(precedents)
                if len(all_precedents_list) >= total_count: break
                page += 1

            detailed_precedents = []
            tasks = [fetch_precedent_detail(client, p['판례일련번호']) for p in all_precedents_list]
            results = await tqdm_asyncio.gather(tasks, desc="Fetching Precedent Details")
            
            for res in results:
                if res:
                    content = res.get("PrecService") or res.get("precService")
                    if content:
                        if isinstance(content, list): detailed_precedents.extend(content)
                        else: detailed_precedents.append(content)
            
            df = pl.DataFrame(detailed_precedents)
            df = df.select([pl.all().cast(pl.Utf8)])
            df.write_parquet("korean_labor_precedents.parquet")
            return df
    return None

precedent_df = get_precedent_data()
precedent_df

# Cell 11: Load and Prepare Dataset
mo.md(r"""
## 📚 Data Preparation
Load the fetched data and prepare it for training.
""")

def load_and_prepare_data():
    try:
        laws = pl.read_parquet("korean_labor_laws.parquet")
        precedents = pl.read_parquet("korean_labor_precedents.parquet")

        law_texts = laws.select(pl.col("본문").alias("text"))
        
        # Extract relevant text fields from precedents
        precedent_texts = precedents.select(
            pl.concat_str([
                pl.col("사건명"),
                pl.col("선고일자"),
                pl.col("판시사항"),
                pl.col("판결요지"),
                pl.col("참조조문"),
                pl.col("참조판례"),
                pl.col("판례내용")
            ], separator="\n\n").alias("text")
        )
        
        combined_df = pl.concat([law_texts, precedent_texts])
        combined_df = combined_df.filter(pl.col("text").is_not_null() & (pl.col("text") != ""))
        
        # Convert to Hugging Face Dataset
        hf_dataset = Dataset.from_pandas(combined_df.to_pandas())
        return hf_dataset

    except Exception as e:
        return mo.md(f"🔴 **Error:** Could not load or process data files. Have you fetched them? Error: {e}")

dataset = load_and_prepare_data()
dataset

# Cell 12: Load Model and Tokenizer
mo.md(r"""
## 🤖 Load Model & Tokenizer
Loading the `gpt-oss-120b` model. We'll use a placeholder model for this example.
We use `EleutherAI/polyglot-ko-12.8b` as a stand-in for the fictional `gpt-oss-120b`.
""")

model_name = "EleutherAI/polyglot-ko-12.8b" # Placeholder for gpt-oss-120b

@mo.cache
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer()

mo.md(f"✅ Model and tokenizer for `{model_name}` loaded successfully.")

# Cell 13: Tokenize Dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

mo.md("✅ Dataset tokenized and ready for training.")

# Cell 14: Fine-Tuning Configuration
mo.md(r"""
## 🏋️ Fine-Tuning
Configure and start the fine-tuning process.
""")

training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=4, # Adjust based on VRAM
    gradient_accumulation_steps=8, # Adjust based on VRAM
    save_steps=10_000,
    save_total_limit=2,
    learning_rate=2e-5,
    bf16=True, # Requires Ampere or newer GPU
    logging_dir='./logs',
    logging_steps=100,
    report_to="none", # "wandb" or "tensorboard"
    save_safetensors=True, # Use safetensors
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

start_training_button = mo.ui.button(label="Start Fine-Tuning")
start_training_button

# Cell 15: Run Training
if start_training_button.value:
    mo.md("🚀 Starting training... This may take a long time.")
    trainer.train()
    mo.md("✅ Training complete! Model saved to `./results`.")
    
    # Save final model with safetensors
    final_path = "./results/final_model"
    trainer.save_model(final_path)
    
    # Verify safetensors file exists
    safetensor_file = os.path.join(final_path, "model.safetensors")
    if os.path.exists(safetensor_file):
        mo.md(f"✅ Model successfully saved in safetensors format at `{safetensor_file}`")
    else:
        mo.md(f"🔴 **Warning:** Could not find `model.safetensors` file.")


# Cell 16: Inference
mo.md(r"""
## 🧪 Inference
Test the fine-tuned model.
""")

prompt_text = mo.ui.text_area(label="Enter a prompt:")
prompt_text

# Cell 17: Generate Text
if prompt_text.value:
    inputs = tokenizer(prompt_text.value, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_length=150, num_return_sequences=1)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    mo.md(f"**Generated Text:**\n\n{generated_text}")
