import os
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List
from openai import OpenAI
from fastapi.middleware.cors import CORSMiddleware

# --- OpenRouter and Gemini Flash ---
api_key = os.getenv("OPENROUTER_API_KEY")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
    default_headers={
        "HTTP-Referer": "http://localhost:3000",
        "X-Title": "MyHandNomusa",
    },
)

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models for Structured Output ---
class ClauseAnalysis(BaseModel):
    risk: str = Field(..., description="Risk level, e.g., '위험', '주의', '안전'")
    original_content: str = Field(..., description="The original clause content from the contract.")
    law_content: str = Field(..., description="The relevant content of the law.")
    law_line: str = Field(..., description="The specific article and clause of the law.")
    original_line: str = Field(..., description="The line number or identifier of the original clause.")
    reasoning: str = Field(..., description="The AI's reasoning for the analysis.")

class AnalysisReport(BaseModel):
    analyses: List[ClauseAnalysis]

class Contract(BaseModel):
    text: str

# --- System Prompt for Structured JSON Output ---
SYSTEM_PROMPT = """
You are an expert in Korean labor law. Analyze the provided labor contract text.
Identify clauses with potential legal risks. For each identified clause, provide a detailed analysis.
"""

@app.post("/api/analyze", response_model=AnalysisReport)
async def analyze_contract(contract: Contract):
    """
    Analyzes a labor contract and returns a structured report of potential risks.
    """
    if not api_key:
        return {"error": "OPENROUTER_API_KEY environment variable not set."}

    try:
        completion = client.beta.chat.completions.parse(
            model="google/gemini-2.5-flash",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": contract.text},
            ],
            response_format=AnalysisReport,
        )

        message = completion.choices[0].message
        if message.parsed:
            return message.parsed
        else:
            return {"error": message.refusal}

    except Exception as e:
        return {"error": str(e)}

@app.get("/")
def read_root():
    return {"message": "MyHandNomusa API is running."}
