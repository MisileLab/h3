mod audio;
mod openai;

use audio::AudioCapture;
use openai::{ActionItem, MeetingSummary, OpenAIClient, SubtitleEntry, TranscriptionResult};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tauri::{AppHandle, Emitter, Manager, State};
use tokio::sync::{mpsc, Mutex, RwLock};

// Application state
pub struct AppState {
    openai_client: RwLock<Option<OpenAIClient>>,
    audio_capture: Mutex<Option<AudioCapture>>,
    subtitle_logs: RwLock<Vec<SubtitleEntry>>,
    user_keywords: RwLock<Vec<String>>,
    is_session_active: RwLock<bool>,
    session_start_time: RwLock<Option<i64>>,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            openai_client: RwLock::new(None),
            audio_capture: Mutex::new(None),
            subtitle_logs: RwLock::new(Vec::new()),
            user_keywords: RwLock::new(Vec::new()),
            is_session_active: RwLock::new(false),
            session_start_time: RwLock::new(None),
        }
    }
}

// Event payloads
#[derive(Clone, Serialize)]
struct SubtitleEvent {
    subtitle: SubtitleEntry,
}

#[derive(Clone, Serialize)]
struct SummaryEvent {
    summary: MeetingSummary,
}

#[derive(Clone, Serialize)]
struct ErrorEvent {
    message: String,
}

#[derive(Clone, Serialize)]
struct SessionStatusEvent {
    is_active: bool,
    start_time: Option<i64>,
}

// Settings structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserSettings {
    pub username: String,
    pub keywords: Vec<String>,
    pub api_key: String,
}

impl Default for UserSettings {
    fn default() -> Self {
        Self {
            username: String::new(),
            keywords: Vec::new(),
            api_key: String::new(),
        }
    }
}

// Commands

#[tauri::command]
async fn set_api_key(api_key: String, state: State<'_, Arc<AppState>>) -> Result<(), String> {
    if api_key.trim().is_empty() {
        return Err("API key cannot be empty".to_string());
    }

    let client = OpenAIClient::new(api_key);
    let mut client_guard = state.openai_client.write().await;
    *client_guard = Some(client);

    Ok(())
}

#[tauri::command]
async fn set_user_keywords(
    keywords: Vec<String>,
    state: State<'_, Arc<AppState>>,
) -> Result<(), String> {
    let mut keywords_guard = state.user_keywords.write().await;
    *keywords_guard = keywords;
    Ok(())
}

#[tauri::command]
async fn get_user_keywords(state: State<'_, Arc<AppState>>) -> Result<Vec<String>, String> {
    let keywords = state.user_keywords.read().await;
    Ok(keywords.clone())
}

#[tauri::command]
async fn start_session(
    app: AppHandle,
    state: State<'_, Arc<AppState>>,
) -> Result<(), String> {
    // Check if API key is set
    {
        let client = state.openai_client.read().await;
        if client.is_none() {
            return Err("Please set your OpenAI API key in settings first".to_string());
        }
    }

    // Check if already active
    {
        let is_active = state.is_session_active.read().await;
        if *is_active {
            return Err("Session already active".to_string());
        }
    }

    // Clear previous logs
    {
        let mut logs = state.subtitle_logs.write().await;
        logs.clear();
    }

    // Set session as active
    {
        let mut is_active = state.is_session_active.write().await;
        *is_active = true;
    }

    // Set start time
    let start_time = chrono::Utc::now().timestamp_millis();
    {
        let mut session_start = state.session_start_time.write().await;
        *session_start = Some(start_time);
    }

    // Create audio capture
    let mut audio_capture = AudioCapture::new()?;

    // Create channel for audio chunks
    let (audio_tx, mut audio_rx) = mpsc::unbounded_channel::<Vec<u8>>();

    // Start audio capture
    audio_capture.start_capture(audio_tx)?;

    // Store audio capture in state
    {
        let mut capture_guard = state.audio_capture.lock().await;
        *capture_guard = Some(audio_capture);
    }

    // Clone state for async task
    let state_clone = Arc::clone(&state.inner());
    let app_clone = app.clone();

    // Spawn audio processing task
    tokio::spawn(async move {
        let mut last_summary_time = std::time::Instant::now();
        let summary_interval = std::time::Duration::from_secs(20);

        while let Some(audio_data) = audio_rx.recv().await {
            // Check if session is still active
            let is_active = {
                let guard = state_clone.is_session_active.read().await;
                *guard
            };

            if !is_active {
                break;
            }

            // Get OpenAI client
            let client = {
                let guard = state_clone.openai_client.read().await;
                guard.clone()
            };

            if let Some(client) = client {
                // Transcribe audio
                match client.transcribe_audio_chunk(audio_data).await {
                    Ok(transcription) => {
                        if transcription.text.trim().is_empty() {
                            continue;
                        }

                        // Translate to Korean
                        let text_ko = match client.translate_text_en_to_ko(&transcription.text).await {
                            Ok(ko) => ko,
                            Err(e) => {
                                eprintln!("Translation error: {}", e);
                                String::new()
                            }
                        };

                        // Create subtitle entry
                        let entry = SubtitleEntry {
                            id: uuid::Uuid::new_v4().to_string(),
                            timestamp: transcription.timestamp,
                            speaker: None,
                            text_en: transcription.text,
                            text_ko,
                        };

                        // Store in logs
                        {
                            let mut logs = state_clone.subtitle_logs.write().await;
                            logs.push(entry.clone());
                        }

                        // Emit subtitle event to frontend
                        let _ = app_clone.emit("subtitle", SubtitleEvent { subtitle: entry });

                        // Check if we should update summary
                        if last_summary_time.elapsed() >= summary_interval {
                            last_summary_time = std::time::Instant::now();

                            let logs = {
                                let guard = state_clone.subtitle_logs.read().await;
                                guard.clone()
                            };

                            let keywords = {
                                let guard = state_clone.user_keywords.read().await;
                                guard.clone()
                            };

                            match client.summarize_meeting_state(&logs, &keywords).await {
                                Ok(summary) => {
                                    let _ = app_clone.emit("summary", SummaryEvent { summary });
                                }
                                Err(e) => {
                                    eprintln!("Summary error: {}", e);
                                }
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("Transcription error: {}", e);
                        let _ = app_clone.emit("error", ErrorEvent { message: e });
                    }
                }
            }
        }
    });

    // Emit session status
    app.emit(
        "session-status",
        SessionStatusEvent {
            is_active: true,
            start_time: Some(start_time),
        },
    )
    .map_err(|e| e.to_string())?;

    Ok(())
}

#[tauri::command]
async fn stop_session(
    app: AppHandle,
    state: State<'_, Arc<AppState>>,
) -> Result<MeetingSummary, String> {
    // Set session as inactive
    {
        let mut is_active = state.is_session_active.write().await;
        *is_active = false;
    }

    // Stop audio capture
    {
        let mut capture_guard = state.audio_capture.lock().await;
        if let Some(ref mut capture) = *capture_guard {
            capture.stop_capture();
        }
        *capture_guard = None;
    }

    // Generate final summary
    let final_summary = {
        let client_guard = state.openai_client.read().await;
        let logs = state.subtitle_logs.read().await;
        let keywords = state.user_keywords.read().await;

        if let Some(ref client) = *client_guard {
            client
                .summarize_meeting_state(&logs, &keywords)
                .await
                .unwrap_or_default()
        } else {
            MeetingSummary::default()
        }
    };

    // Emit session status
    app.emit(
        "session-status",
        SessionStatusEvent {
            is_active: false,
            start_time: None,
        },
    )
    .map_err(|e| e.to_string())?;

    Ok(final_summary)
}

#[tauri::command]
async fn get_session_status(
    state: State<'_, Arc<AppState>>,
) -> Result<SessionStatusEvent, String> {
    let is_active = state.is_session_active.read().await;
    let start_time = state.session_start_time.read().await;

    Ok(SessionStatusEvent {
        is_active: *is_active,
        start_time: *start_time,
    })
}

#[tauri::command]
async fn get_subtitle_logs(state: State<'_, Arc<AppState>>) -> Result<Vec<SubtitleEntry>, String> {
    let logs = state.subtitle_logs.read().await;
    Ok(logs.clone())
}

#[tauri::command]
async fn get_current_summary(state: State<'_, Arc<AppState>>) -> Result<MeetingSummary, String> {
    let client_guard = state.openai_client.read().await;
    let logs = state.subtitle_logs.read().await;
    let keywords = state.user_keywords.read().await;

    if let Some(ref client) = *client_guard {
        client
            .summarize_meeting_state(&logs, &keywords)
            .await
    } else {
        Ok(MeetingSummary::default())
    }
}

#[tauri::command]
fn list_audio_devices() -> Vec<String> {
    audio::list_input_devices()
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // Load environment variables from .env file if it exists
    let _ = dotenvy::dotenv();

    let app_state = Arc::new(AppState::default());

    // Try to initialize OpenAI client from environment
    if let Ok(api_key) = std::env::var("OPENAI_API_KEY") {
        let client = OpenAIClient::new(api_key);
        let state_clone = Arc::clone(&app_state);
        tokio::runtime::Runtime::new().unwrap().block_on(async {
            let mut client_guard = state_clone.openai_client.write().await;
            *client_guard = Some(client);
        });
    }

    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_store::Builder::new().build())
        .manage(app_state)
        .invoke_handler(tauri::generate_handler![
            set_api_key,
            set_user_keywords,
            get_user_keywords,
            start_session,
            stop_session,
            get_session_status,
            get_subtitle_logs,
            get_current_summary,
            list_audio_devices,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
