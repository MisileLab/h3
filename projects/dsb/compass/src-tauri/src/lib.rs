mod keyboard_tracker;

use keyboard_tracker::{KeyboardTracker, TypingStats};
use std::sync::Arc;
use tauri::{State, AppHandle, Manager};
use tokio::sync::Mutex;

// Global state for the keyboard tracker
type TrackerState = Arc<Mutex<Option<KeyboardTracker>>>;

#[tauri::command]
async fn init_tracker(
    app: AppHandle, 
    tracker_state: State<'_, TrackerState>
) -> Result<String, String> {
    // Use Tauri's app data directory for session storage
    let app_data_dir = app.path().app_data_dir().map_err(|e| {
        println!("Error getting app data directory: {}", e);
        e.to_string()
    })?;
    
    let sessions_dir = app_data_dir.join("sessions");
    
    // Debug: print the path being used
    println!("Sessions directory: {:?}", sessions_dir);
    
    let tracker = KeyboardTracker::new(sessions_dir).await.map_err(|e| {
        println!("Error creating tracker: {}", e);
        e.to_string()
    })?;
    let mut state = tracker_state.lock().await;
    *state = Some(tracker);
    Ok("Keyboard tracker initialized successfully with Polars + Parquet storage".to_string())
}

#[tauri::command]
async fn start_tracking(tracker_state: State<'_, TrackerState>) -> Result<String, String> {
    let mut state = tracker_state.lock().await;
    if let Some(tracker) = state.as_mut() {
        tracker.start_tracking().await.map_err(|e| e.to_string())?;
        Ok("Keyboard tracking started".to_string())
    } else {
        Err("Tracker not initialized".to_string())
    }
}

#[tauri::command]
async fn stop_tracking(tracker_state: State<'_, TrackerState>) -> Result<String, String> {
    let mut state = tracker_state.lock().await;
    if let Some(tracker) = state.as_mut() {
        tracker.stop_tracking().await;
        Ok("Keyboard tracking stopped".to_string())
    } else {
        Err("Tracker not initialized".to_string())
    }
}

#[tauri::command]
async fn get_typing_stats(tracker_state: State<'_, TrackerState>) -> Result<TypingStats, String> {
    let state = tracker_state.lock().await;
    if let Some(tracker) = state.as_ref() {
        tracker.get_typing_stats().await.map_err(|e| e.to_string())
    } else {
        Err("Tracker not initialized".to_string())
    }
}

#[tauri::command]
async fn clear_stats(tracker_state: State<'_, TrackerState>) -> Result<String, String> {
    let mut state = tracker_state.lock().await;
    if let Some(tracker) = state.as_mut() {
        tracker.clear_stats().await.map_err(|e| e.to_string())?;
        Ok("Statistics cleared and new session started".to_string())
    } else {
        Err("Tracker not initialized".to_string())
    }
}

#[tauri::command]
async fn list_sessions(tracker_state: State<'_, TrackerState>) -> Result<Vec<String>, String> {
    let state = tracker_state.lock().await;
    if let Some(tracker) = state.as_ref() {
        let sessions = tracker.list_sessions().map_err(|e| e.to_string())?;
        let session_names: Vec<String> = sessions
            .into_iter()
            .filter_map(|path| {
                path.file_name()?.to_str().map(|s| s.to_string())
            })
            .collect();
        Ok(session_names)
    } else {
        Err("Tracker not initialized".to_string())
    }
}

#[tauri::command]
async fn load_session(
    session_name: String,
    tracker_state: State<'_, TrackerState>
) -> Result<TypingStats, String> {
    let state = tracker_state.lock().await;
    if let Some(tracker) = state.as_ref() {
        let sessions = tracker.list_sessions().map_err(|e| e.to_string())?;
        
        // Find the session file by name
        let session_path = sessions
            .into_iter()
            .find(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .map_or(false, |name| name == session_name)
            })
            .ok_or("Session file not found")?;
            
        tracker.load_session_data(&session_path).await.map_err(|e| e.to_string())
    } else {
        Err("Tracker not initialized".to_string())
    }
}

#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! You've been greeted from Rust!", name)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_fs::init())
        .manage(TrackerState::default())
        .invoke_handler(tauri::generate_handler![
            greet,
            init_tracker,
            start_tracking,
            stop_tracking,
            get_typing_stats,
            clear_stats,
            list_sessions,
            load_session
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
