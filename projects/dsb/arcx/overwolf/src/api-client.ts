/**
 * API client for communicating with ArcX backend
 */

const API_BASE_URL = "http://127.0.0.1:8765";

export interface EVResponse {
  ev_stay: number;
  ev_extract: number;
  delta_ev: number;
  recommendation: "stay" | "extract" | "neutral";
  confidence: number;
  message: string;
  color: "green" | "yellow" | "red";
  timestamp: number;
  risk_profile: string;
}

export interface SystemStatus {
  is_capturing: boolean;
  is_model_loaded: boolean;
  buffer_frames: number;
  device_backend: string;
  fps: number;
  inference_time_ms: number;
}

export interface RunStartResponse {
  status: string;
  run_id: string;
}

export class APIClient {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  private async request<T>(endpoint: string, options?: RequestInit): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;

    try {
      const response = await fetch(url, {
        ...options,
        headers: {
          "Content-Type": "application/json",
          ...options?.headers,
        },
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(
          `API request failed: ${response.status} ${response.statusText} - ${
            errorData.detail || ""
          }`
        );
      }

      return await response.json();
    } catch (error) {
      console.error(`API request failed: ${endpoint}`, error);
      throw error;
    }
  }

  async getEV(): Promise<EVResponse> {
    return this.request<EVResponse>("/ev");
  }

  async getStatus(): Promise<SystemStatus> {
    return this.request<SystemStatus>("/status");
  }

  async updateConfig(riskProfile: "safe" | "neutral" | "aggressive"): Promise<void> {
    await this.request("/config", {
      method: "POST",
      body: JSON.stringify({ risk_profile: riskProfile }),
    });
  }

  async submitFeedback(
    runId: string,
    decisionIdx: number,
    rating: "bad" | "good"
  ): Promise<void> {
    await this.request("/feedback", {
      method: "POST",
      body: JSON.stringify({
        run_id: runId,
        decision_idx: decisionIdx,
        timestamp: Date.now() / 1000,
        rating,
        context: {},
      }),
    });
  }

  async startRun(mapId: string = "unknown"): Promise<RunStartResponse> {
    return this.request<RunStartResponse>("/run/start", {
      method: "POST",
      body: JSON.stringify({
        map_id: mapId,
        metadata: {},
      }),
    });
  }

  async endRun(
    runId: string,
    finalLootValue: number,
    totalTimeSec: number,
    success: boolean,
    actionTaken: "stay" | "extract"
  ): Promise<void> {
    await this.request("/run/end", {
      method: "POST",
      body: JSON.stringify({
        run_id: runId,
        final_loot_value: finalLootValue,
        total_time_sec: totalTimeSec,
        success,
        action_taken: actionTaken,
      }),
    });
  }

  async checkConnection(): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseUrl}/`);
      return response.ok;
    } catch {
      return false;
    }
  }
}

export const apiClient = new APIClient();
