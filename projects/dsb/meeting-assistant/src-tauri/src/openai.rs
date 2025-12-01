use reqwest::{multipart, Client};
use serde::{Deserialize, Serialize};
use std::env;
use std::time::Duration;

const OPENAI_API_BASE: &str = "https://api.openai.com/v1";
const MAX_RETRIES: u32 = 2;
const REQUEST_TIMEOUT_SECS: u64 = 30;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionResult {
    pub text: String,
    pub timestamp: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionItem {
    pub text: String,
    #[serde(rename = "isRelevantToUser")]
    pub is_relevant_to_user: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeetingSummary {
    pub summary: Vec<String>,
    pub decisions: Vec<String>,
    pub action_items: Vec<ActionItem>,
}

impl Default for MeetingSummary {
    fn default() -> Self {
        Self {
            summary: Vec::new(),
            decisions: Vec::new(),
            action_items: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubtitleEntry {
    pub id: String,
    pub timestamp: i64,
    pub speaker: Option<String>,
    pub text_en: String,
    pub text_ko: String,
}

// OpenAI API response structures
#[derive(Debug, Deserialize)]
struct WhisperResponse {
    text: String,
}

#[derive(Debug, Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,
    temperature: f32,
    max_tokens: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    message: ChatMessageResponse,
}

#[derive(Debug, Deserialize)]
struct ChatMessageResponse {
    content: String,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<ChatChoice>,
}

pub struct OpenAIClient {
    client: Client,
    api_key: String,
}

impl OpenAIClient {
    pub fn new(api_key: String) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(REQUEST_TIMEOUT_SECS))
            .build()
            .expect("Failed to create HTTP client");

        Self { client, api_key }
    }

    pub fn from_env() -> Result<Self, String> {
        let api_key = env::var("OPENAI_API_KEY")
            .map_err(|_| "OPENAI_API_KEY environment variable not set".to_string())?;
        Ok(Self::new(api_key))
    }

    /// Transcribe audio chunk to English text using Whisper API
    pub async fn transcribe_audio_chunk(
        &self,
        audio_bytes: Vec<u8>,
    ) -> Result<TranscriptionResult, String> {
        let timestamp = chrono::Utc::now().timestamp_millis();

        for attempt in 0..MAX_RETRIES {
            match self.do_transcribe(&audio_bytes).await {
                Ok(text) => {
                    return Ok(TranscriptionResult { text, timestamp });
                }
                Err(e) => {
                    if attempt < MAX_RETRIES - 1 {
                        eprintln!(
                            "Transcription attempt {} failed: {}, retrying...",
                            attempt + 1,
                            e
                        );
                        tokio::time::sleep(Duration::from_millis(500)).await;
                    } else {
                        return Err(format!("Transcription failed after {} attempts: {}", MAX_RETRIES, e));
                    }
                }
            }
        }

        Err("Transcription failed".to_string())
    }

    async fn do_transcribe(&self, audio_bytes: &[u8]) -> Result<String, String> {
        let part = multipart::Part::bytes(audio_bytes.to_vec())
            .file_name("audio.wav")
            .mime_str("audio/wav")
            .map_err(|e| format!("Failed to create multipart: {}", e))?;

        let form = multipart::Form::new()
            .text("model", "whisper-1")
            .text("language", "en")
            .text("response_format", "json")
            .part("file", part);

        let response = self
            .client
            .post(format!("{}/audio/transcriptions", OPENAI_API_BASE))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .multipart(form)
            .send()
            .await
            .map_err(|e| format!("Request failed: {}", e))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(format!("API error {}: {}", status, body));
        }

        let whisper_response: WhisperResponse = response
            .json()
            .await
            .map_err(|e| format!("Failed to parse response: {}", e))?;

        Ok(whisper_response.text)
    }

    /// Translate English text to Korean
    pub async fn translate_text_en_to_ko(&self, text_en: &str) -> Result<String, String> {
        if text_en.trim().is_empty() {
            return Ok(String::new());
        }

        for attempt in 0..MAX_RETRIES {
            match self.do_translate(text_en).await {
                Ok(text) => return Ok(text),
                Err(e) => {
                    if attempt < MAX_RETRIES - 1 {
                        eprintln!(
                            "Translation attempt {} failed: {}, retrying...",
                            attempt + 1,
                            e
                        );
                        tokio::time::sleep(Duration::from_millis(500)).await;
                    } else {
                        return Err(format!("Translation failed after {} attempts: {}", MAX_RETRIES, e));
                    }
                }
            }
        }

        Err("Translation failed".to_string())
    }

    async fn do_translate(&self, text_en: &str) -> Result<String, String> {
        let request = ChatCompletionRequest {
            model: "gpt-4o-mini".to_string(),
            messages: vec![
                ChatMessage {
                    role: "system".to_string(),
                    content: "You are a professional translator. Translate the following English text to natural Korean. Only output the translation, nothing else. Preserve the original meaning and tone.".to_string(),
                },
                ChatMessage {
                    role: "user".to_string(),
                    content: text_en.to_string(),
                },
            ],
            temperature: 0.3,
            max_tokens: Some(1000),
        };

        let response = self
            .client
            .post(format!("{}/chat/completions", OPENAI_API_BASE))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| format!("Request failed: {}", e))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(format!("API error {}: {}", status, body));
        }

        let chat_response: ChatCompletionResponse = response
            .json()
            .await
            .map_err(|e| format!("Failed to parse response: {}", e))?;

        chat_response
            .choices
            .first()
            .map(|c| c.message.content.clone())
            .ok_or_else(|| "No response from API".to_string())
    }

    /// Summarize meeting state from transcript logs
    pub async fn summarize_meeting_state(
        &self,
        logs: &[SubtitleEntry],
        user_keywords: &[String],
    ) -> Result<MeetingSummary, String> {
        if logs.is_empty() {
            return Ok(MeetingSummary::default());
        }

        // Build transcript text
        let transcript: String = logs
            .iter()
            .map(|entry| {
                let speaker = entry.speaker.as_deref().unwrap_or("Speaker");
                format!("[{}]: {}", speaker, entry.text_en)
            })
            .collect::<Vec<_>>()
            .join("\n");

        let keywords_str = if user_keywords.is_empty() {
            "None specified".to_string()
        } else {
            user_keywords.join(", ")
        };

        for attempt in 0..MAX_RETRIES {
            match self.do_summarize(&transcript, &keywords_str).await {
                Ok(summary) => return Ok(summary),
                Err(e) => {
                    if attempt < MAX_RETRIES - 1 {
                        eprintln!(
                            "Summarization attempt {} failed: {}, retrying...",
                            attempt + 1,
                            e
                        );
                        tokio::time::sleep(Duration::from_millis(500)).await;
                    } else {
                        return Err(format!("Summarization failed after {} attempts: {}", MAX_RETRIES, e));
                    }
                }
            }
        }

        Err("Summarization failed".to_string())
    }

    async fn do_summarize(&self, transcript: &str, keywords: &str) -> Result<MeetingSummary, String> {
        let system_prompt = format!(
            r#"You are a meeting summarizer. Analyze the following meeting transcript and extract:
1. Summary: Key discussion points (2-5 bullet points)
2. Decisions: Any decisions made during the meeting
3. Action Items: Tasks or follow-ups mentioned

User's keywords to watch for: {}

For action items, mark "isRelevantToUser" as true if the item mentions or relates to any of the user's keywords.

Respond ONLY with valid JSON in this exact format:
{{
  "summary": ["point 1", "point 2"],
  "decisions": ["decision 1", "decision 2"],
  "action_items": [
    {{"text": "action item text", "isRelevantToUser": false}},
    {{"text": "action item text", "isRelevantToUser": true}}
  ]
}}

If there are no items for a category, use an empty array."#,
            keywords
        );

        let request = ChatCompletionRequest {
            model: "gpt-4o-mini".to_string(),
            messages: vec![
                ChatMessage {
                    role: "system".to_string(),
                    content: system_prompt,
                },
                ChatMessage {
                    role: "user".to_string(),
                    content: format!("Meeting transcript:\n\n{}", transcript),
                },
            ],
            temperature: 0.3,
            max_tokens: Some(2000),
        };

        let response = self
            .client
            .post(format!("{}/chat/completions", OPENAI_API_BASE))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| format!("Request failed: {}", e))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(format!("API error {}: {}", status, body));
        }

        let chat_response: ChatCompletionResponse = response
            .json()
            .await
            .map_err(|e| format!("Failed to parse response: {}", e))?;

        let content = chat_response
            .choices
            .first()
            .map(|c| c.message.content.clone())
            .ok_or_else(|| "No response from API".to_string())?;

        // Parse the JSON response
        serde_json::from_str(&content)
            .map_err(|e| format!("Failed to parse summary JSON: {} - Content: {}", e, content))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_meeting_summary_default() {
        let summary = MeetingSummary::default();
        assert!(summary.summary.is_empty());
        assert!(summary.decisions.is_empty());
        assert!(summary.action_items.is_empty());
    }
}
