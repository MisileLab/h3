import { createSignal, onMount, For, Show } from "solid-js";
import { invoke } from "@tauri-apps/api/core";
import "./App.css";

interface KeyStats {
  key: string;
  count: number;
  last_pressed: string;
}

interface DailyStats {
  date: string;
  key_count: number;
  session_time: number;
}

interface TypingStats {
  total_keys: number;
  words_per_minute: number;
  keys_per_hour: number;
  most_pressed_keys: KeyStats[];
  daily_stats: DailyStats[];
}

function App() {
  const [isTracking, setIsTracking] = createSignal(false);
  const [isInitialized, setIsInitialized] = createSignal(false);
  const [stats, setStats] = createSignal<TypingStats | null>(null);
  const [status, setStatus] = createSignal("Ready to initialize");
  const [error, setError] = createSignal("");

  onMount(async () => {
    await initializeTracker();
    // Auto-refresh stats every 5 seconds
    setInterval(async () => {
      if (isInitialized()) {
        await refreshStats();
      }
    }, 5000);
  });

  async function initializeTracker() {
    try {
      setError("");
      setStatus("Initializing tracker...");
      const result = await invoke<string>("init_tracker");
      setStatus(result);
      setIsInitialized(true);
      await refreshStats();
    } catch (err) {
      setError(`Failed to initialize: ${err}`);
      setStatus("Failed to initialize");
    }
  }

  async function startTracking() {
    try {
      setError("");
      const result = await invoke<string>("start_tracking");
      setStatus(result);
      setIsTracking(true);
    } catch (err) {
      setError(`Failed to start tracking: ${err}`);
    }
  }

  async function stopTracking() {
    try {
      setError("");
      const result = await invoke<string>("stop_tracking");
      setStatus(result);
      setIsTracking(false);
    } catch (err) {
      setError(`Failed to stop tracking: ${err}`);
    }
  }

  async function refreshStats() {
    try {
      setError("");
      const result = await invoke<TypingStats>("get_typing_stats");
      setStats(result);
    } catch (err) {
      setError(`Failed to get stats: ${err}`);
    }
  }

  async function clearStats() {
    try {
      setError("");
      const result = await invoke<string>("clear_stats");
      setStatus(result);
      await refreshStats();
    } catch (err) {
      setError(`Failed to clear stats: ${err}`);
    }
  }

  function formatKey(key: string) {
    // Remove "Key" prefix and clean up key names
    return key.replace(/^Key/, "").replace(/([A-Z])/g, " $1").trim();
  }

  function formatDate(dateStr: string) {
    const date = new Date(dateStr);
    return date.toLocaleDateString();
  }

  return (
    <main class="container">
      <header class="header">
        <h1>🖮 Keyboard Tracker</h1>
        <p class="subtitle">Monitor your typing patterns and productivity</p>
      </header>

      <div class="status-bar">
        <div class={`status ${error() ? "error" : ""}`}>
          {error() || status()}
        </div>
        <div class={`tracking-indicator ${isTracking() ? "active" : ""}`}>
          {isTracking() ? "🔴 Tracking" : "⚪ Stopped"}
        </div>
      </div>

      <div class="control-panel">
        <Show when={!isInitialized()}>
          <button onClick={initializeTracker} class="btn primary">
            Initialize Tracker
          </button>
        </Show>
        
        <Show when={isInitialized()}>
          <Show when={!isTracking()}>
            <button onClick={startTracking} class="btn success">
              Start Tracking
            </button>
          </Show>
          
          <Show when={isTracking()}>
            <button onClick={stopTracking} class="btn warning">
              Stop Tracking
            </button>
          </Show>

          <button onClick={refreshStats} class="btn secondary">
            Refresh Stats
          </button>
          
          <button onClick={clearStats} class="btn danger">
            Clear Stats
          </button>
        </Show>
      </div>

      <Show when={stats()}>
        <div class="stats-grid">
          <div class="stat-card">
            <h3>Total Keys Pressed</h3>
            <div class="stat-value">{stats()?.total_keys.toLocaleString()}</div>
          </div>

          <div class="stat-card">
            <h3>Words Per Minute</h3>
            <div class="stat-value">{stats()?.words_per_minute.toFixed(1)}</div>
          </div>

          <div class="stat-card">
            <h3>Keys Per Hour</h3>
            <div class="stat-value">{stats()?.keys_per_hour.toFixed(0)}</div>
          </div>
        </div>

        <div class="charts-container">
          <div class="chart-section">
            <h3>Most Pressed Keys</h3>
            <div class="key-list">
              <For each={stats()?.most_pressed_keys}>
                {(keyData) => (
                  <div class="key-item">
                    <div class="key-name">{formatKey(keyData.key)}</div>
                    <div class="key-count">{keyData.count}</div>
                    <div class="key-bar">
                      <div 
                        class="key-bar-fill" 
                        style={`width: ${Math.min(100, (keyData.count / (stats()?.most_pressed_keys[0]?.count || 1)) * 100)}%`}
                      ></div>
                    </div>
                  </div>
                )}
              </For>
            </div>
          </div>

          <div class="chart-section">
            <h3>Daily Activity</h3>
            <div class="daily-stats">
              <For each={stats()?.daily_stats}>
                {(dayData) => (
                  <div class="day-item">
                    <div class="day-date">{formatDate(dayData.date)}</div>
                    <div class="day-count">{dayData.key_count} keys</div>
                    <div class="day-bar">
                      <div 
                        class="day-bar-fill"
                        style={`width: ${Math.min(100, (dayData.key_count / Math.max(...(stats()?.daily_stats.map(d => d.key_count) || [1]))) * 100)}%`}
                      ></div>
                    </div>
                  </div>
                )}
              </For>
            </div>
          </div>
        </div>
      </Show>

      <Show when={!stats() && isInitialized()}>
        <div class="empty-state">
          <p>No data available yet. Start tracking to see your keyboard statistics!</p>
        </div>
      </Show>
    </main>
  );
}

export default App;
