use chrono::{DateTime, Utc};
use rdev::{listen, Event, EventType};
use serde::{Deserialize, Serialize};
use polars::prelude::*;
use std::path::PathBuf;
use std::sync::Arc;
use std::thread;
use tokio::sync::{Mutex, mpsc};
use tokio::time::{interval, Duration};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyStats {
    pub key: String,
    pub count: i64,
    pub last_pressed: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypingStats {
    pub total_keys: i64,
    pub words_per_minute: f64,
    pub keys_per_hour: f64,
    pub most_pressed_keys: Vec<KeyStats>,
    pub daily_stats: Vec<DailyStats>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DailyStats {
    pub date: String,
    pub key_count: i64,
    pub session_time: i64, // in minutes
}

#[derive(Debug, Clone)]
struct KeyEvent {
    key_name: String,
    timestamp: DateTime<Utc>,
}

pub struct KeyboardTracker {
    data: Arc<Mutex<DataFrame>>,
    data_dir: PathBuf,
    current_session_path: PathBuf,
    session_start_time: DateTime<Utc>,
    is_running: Arc<Mutex<bool>>,
    event_sender: Option<mpsc::UnboundedSender<KeyEvent>>,
    save_handle: Option<tokio::task::JoinHandle<()>>,
}

impl KeyboardTracker {
    pub async fn new(data_dir: PathBuf) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let session_start_time = Utc::now();
        let current_session_path = Self::generate_session_path(&data_dir, session_start_time);
        let data = Self::load_or_create_dataframe(&current_session_path).await?;
        
        let tracker = Self {
            data: Arc::new(Mutex::new(data)),
            data_dir,
            current_session_path,
            session_start_time,
            is_running: Arc::new(Mutex::new(false)),
            event_sender: None,
            save_handle: None,
        };
        
        Ok(tracker)
    }

    fn generate_session_path(data_dir: &PathBuf, start_time: DateTime<Utc>) -> PathBuf {
        let filename = format!("keyboard_stats_{}.parquet", start_time.format("%Y%m%d_%H%M%S"));
        data_dir.join(filename)
    }

    async fn load_or_create_dataframe(data_path: &PathBuf) -> PolarsResult<DataFrame> {
        // Create the directory if it doesn't exist
        if let Some(parent) = data_path.parent() {
            std::fs::create_dir_all(parent).unwrap_or_else(|e| {
                eprintln!("Failed to create directory: {}", e);
            });
        }
        
        // Try to load existing parquet file
        if data_path.exists() {
            println!("Loading existing data from: {:?}", data_path);
            match std::fs::File::open(data_path) {
                Ok(file) => {
                    ParquetReader::new(file).finish()
                },
                Err(e) => {
                    eprintln!("Error opening parquet file: {}", e);
                    // Create empty dataframe with the correct schema
                    df! {
                        "key_name" => Vec::<String>::new(),
                        "timestamp" => Vec::<i64>::new(),  // Use i64 for timestamp microseconds
                        "date" => Vec::<String>::new(),     // Use String for date
                    }
                }
            }
        } else {
            println!("Creating new dataframe at: {:?}", data_path);
            // Create empty dataframe with the correct schema
            df! {
                "key_name" => Vec::<String>::new(),
                "timestamp" => Vec::<i64>::new(),  // Use i64 for timestamp microseconds
                "date" => Vec::<String>::new(),     // Use String for date
            }
        }
    }


    async fn save_to_parquet(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let data_guard = self.data.lock().await;
        let mut file = std::fs::File::create(&self.current_session_path)?;
        
        ParquetWriter::new(&mut file)
            .finish(&mut data_guard.clone())?;
            
        println!("Data saved to parquet file: {:?}", self.current_session_path);
        Ok(())
    }

    fn start_auto_save_task(&mut self) {
        let data_clone = self.data.clone();
        let session_path = self.current_session_path.clone();
        let is_running = self.is_running.clone();
        
        let handle = tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(60)); // 1분마다 저장
            
            loop {
                interval.tick().await;
                
                // 실행 중인지 확인
                if !*is_running.lock().await {
                    break;
                }
                
                // Parquet 파일에 저장
                let data_guard = data_clone.lock().await;
                if data_guard.height() > 0 {
                    let mut file = match std::fs::File::create(&session_path) {
                        Ok(file) => file,
                        Err(e) => {
                            eprintln!("Failed to create parquet file: {}", e);
                            continue;
                        }
                    };
                    
                    if let Err(e) = ParquetWriter::new(&mut file).finish(&mut data_guard.clone()) {
                        eprintln!("Failed to save parquet file: {}", e);
                    } else {
                        println!("Auto-saved data to parquet file: {:?}", session_path);
                    }
                }
            }
        });
        
        self.save_handle = Some(handle);
    }

    pub async fn start_tracking(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Set running state
        *self.is_running.lock().await = true;
        
        // Start auto-save task (1분마다 저장)
        self.start_auto_save_task();
        
        // Create channel for key events
        let (tx, mut rx) = mpsc::unbounded_channel::<KeyEvent>();
        self.event_sender = Some(tx.clone());
        
        // Clone data for the event handler task
        let data_clone = self.data.clone();
        let is_running_clone = self.is_running.clone();
        
        // Spawn async task to handle key events
        tokio::spawn(async move {
            while let Some(key_event) = rx.recv().await {
                // Check if we should still be running
                if !*is_running_clone.lock().await {
                    break;
                }
                
                // Add key event to DataFrame
                {
                    let mut data_guard = data_clone.lock().await;
                    
                    let new_row = match df! {
                        "key_name" => [key_event.key_name],
                        "timestamp" => [key_event.timestamp.timestamp_micros()],
                        "date" => [key_event.timestamp.date_naive().to_string()],
                    } {
                        Ok(df) => df,
                        Err(e) => {
                            eprintln!("Error creating DataFrame row: {}", e);
                            continue;
                        }
                    };
                    
                    match data_guard.clone().vstack(&new_row) {
                        Ok(new_data) => *data_guard = new_data,
                        Err(e) => eprintln!("Error adding key event to DataFrame: {}", e),
                    }
                }
            }
        });
        
        // Spawn keyboard listener in a separate thread
        let tx_for_thread = tx;
        let is_running_for_thread = self.is_running.clone();
        
        thread::spawn(move || {
            if let Err(error) = listen(move |event| {
                // Check if we should still be running (non-blocking check)
                let running = match is_running_for_thread.try_lock() {
                    Ok(guard) => *guard,
                    Err(_) => true, // If we can't lock, assume we're still running
                };
                
                if !running {
                    return;
                }
                
                if let Event { event_type: EventType::KeyPress(key), .. } = event {
                    let key_name = format!("{:?}", key);
                    let timestamp = Utc::now();
                    
                    let key_event = KeyEvent {
                        key_name,
                        timestamp,
                    };
                    
                    // Send the event through the channel (ignore send errors)
                    let _ = tx_for_thread.send(key_event);
                }
            }) {
                eprintln!("Error starting keyboard listener: {:?}", error);
            }
        });
        
        Ok(())
    }

    pub async fn stop_tracking(&mut self) {
            *self.is_running.lock().await = false;
        
        // Drop the sender to signal the channel to close
        self.event_sender = None;
        
        // Cancel the auto-save task
        if let Some(handle) = self.save_handle.take() {
            handle.abort();
        }
        
        // Final save before stopping
        if let Err(e) = self.save_to_parquet().await {
            eprintln!("Error saving final data: {}", e);
        }
    }

    pub async fn get_typing_stats(&self) -> Result<TypingStats, Box<dyn std::error::Error + Send + Sync>> {
        let data_guard = self.data.lock().await;
        let df = &*data_guard;
        
        // Debug: DataFrame 정보 출력
        println!("DataFrame schema: {:?}", df.schema());
        println!("DataFrame shape: {} rows, {} columns", df.height(), df.width());
        
        // Use the extracted stats calculation method
        self.calculate_stats_from_dataframe(df).await
    }

    pub async fn clear_stats(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Final save of current session before clearing
        if let Err(e) = self.save_to_parquet().await {
            eprintln!("Error saving final session data: {}", e);
        }
        
        // Clear the in-memory DataFrame
        let mut data_guard = self.data.lock().await;
        *data_guard = df! {
            "key_name" => Vec::<String>::new(),
            "timestamp" => Vec::<i64>::new(),
            "date" => Vec::<String>::new(),
        }?;
        
        // Start a new session with new timestamp
        self.session_start_time = Utc::now();
        self.current_session_path = Self::generate_session_path(&self.data_dir, self.session_start_time);
        
        println!("Started new session: {:?}", self.current_session_path);
        Ok(())
    }

    // List all historical session files
    pub fn list_sessions(&self) -> Result<Vec<PathBuf>, Box<dyn std::error::Error + Send + Sync>> {
        let mut sessions = Vec::new();
        
        if self.data_dir.exists() {
            for entry in std::fs::read_dir(&self.data_dir)? {
                let entry = entry?;
                let path = entry.path();
                if path.is_file() && 
                   path.extension().map_or(false, |ext| ext == "parquet") &&
                   path.file_name().map_or(false, |name| name.to_string_lossy().starts_with("keyboard_stats_")) {
                    sessions.push(path);
                }
            }
        }
        
        // Sort by modification time (newest first)
        sessions.sort_by(|a, b| {
            let a_metadata = std::fs::metadata(a).unwrap_or_else(|_| std::fs::metadata(".").unwrap());
            let b_metadata = std::fs::metadata(b).unwrap_or_else(|_| std::fs::metadata(".").unwrap());
            b_metadata.modified().unwrap_or(std::time::UNIX_EPOCH)
                .cmp(&a_metadata.modified().unwrap_or(std::time::UNIX_EPOCH))
        });
        
        Ok(sessions)
    }

    // Load data from a specific session file
    pub async fn load_session_data(&self, session_path: &PathBuf) -> Result<TypingStats, Box<dyn std::error::Error + Send + Sync>> {
        if !session_path.exists() {
            return Err("Session file does not exist".into());
        }
        
        let file = std::fs::File::open(session_path)?;
        let df = ParquetReader::new(file).finish()?;
        
        // Use the same stats calculation logic but with loaded data
        self.calculate_stats_from_dataframe(&df).await
    }
    
    // Extract stats calculation logic for reuse
    async fn calculate_stats_from_dataframe(&self, df: &DataFrame) -> Result<TypingStats, Box<dyn std::error::Error + Send + Sync>> {
        if df.height() == 0 {
            return Ok(TypingStats {
                total_keys: 0,
                words_per_minute: 0.0,
                keys_per_hour: 0.0,
                most_pressed_keys: vec![],
                daily_stats: vec![],
            });
        }

        // Get total key count
        let total_keys = df.height() as i64;

        // Get most pressed keys (top 10)
        let key_stats_df = df
            .clone()
            .lazy()
            .group_by([col("key_name")])
            .agg([
                col("key_name").count().alias("count"),
                col("timestamp").max().alias("last_pressed"),
            ])
            .sort(
                ["count"],
                SortMultipleOptions::default().with_order_descending(true),
            )
            .limit(10)
            .collect()?;

        let mut most_pressed_keys = Vec::new();
        
        if key_stats_df.height() > 0 {
            let key_names = key_stats_df.column("key_name")?.str()?;
            let counts_u32 = key_stats_df.column("count")?.u32()?;
            let last_pressed = key_stats_df.column("last_pressed")?.i64()?;
            
            for i in 0..key_stats_df.height() {
                if let (Some(key), Some(count_u32), Some(timestamp_us)) = (
                    key_names.get(i),
                    counts_u32.get(i),
                    last_pressed.get(i),
                ) {
                    let timestamp = DateTime::<Utc>::from_timestamp_micros(timestamp_us)
                        .unwrap_or_else(|| Utc::now());
                    
                    most_pressed_keys.push(KeyStats {
                        key: key.to_string(),
                        count: count_u32 as i64,
                        last_pressed: timestamp,
                    });
                }
            }
        }

        // Get daily stats
        let daily_stats_df = df
            .clone()
            .lazy()
            .group_by([col("date")])
            .agg([
                col("key_name").count().alias("key_count"),
            ])
            .sort(
                ["date"],
                SortMultipleOptions::default().with_order_descending(true),
            )
            .limit(7)
            .collect()?;

        let mut daily_stats = Vec::new();
        
        if daily_stats_df.height() > 0 {
            let dates = daily_stats_df.column("date")?.str()?;
            let key_counts_u32 = daily_stats_df.column("key_count")?.u32()?;
            
            for i in 0..daily_stats_df.height() {
                if let (Some(date), Some(key_count_u32)) = (
                    dates.get(i),
                    key_counts_u32.get(i),
                ) {
                    daily_stats.push(DailyStats {
                        date: date.to_string(),
                        key_count: key_count_u32 as i64,
                        session_time: 0i64,
                    });
                }
            }
        }

        // Calculate WPM and keys per hour (last hour)
        let one_hour_ago = Utc::now() - chrono::Duration::hours(1);
        
        let hour_df = df
            .clone()
            .lazy()
            .filter(col("timestamp").gt(lit(one_hour_ago.timestamp_micros())))
            .collect()?;
            
        let hour_keys = hour_df.height() as f64;
        let words_per_minute = hour_keys / 5.0;
        let keys_per_hour = hour_keys;

        Ok(TypingStats {
            total_keys,
            words_per_minute,
            keys_per_hour,
            most_pressed_keys,
            daily_stats,
        })
    }
}
