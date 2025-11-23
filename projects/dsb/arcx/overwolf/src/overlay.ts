/**
 * Main overlay logic for ArcX
 */

import { apiClient, EVResponse, SystemStatus } from "./api-client";

class OverlayApp {
  private currentRunId: string | null = null;
  private isConnected: boolean = false;
  private updateInterval: number = 500; // ms
  private updateTimer: number | null = null;

  // UI elements
  private evStayElement: HTMLElement | null = null;
  private evExtractElement: HTMLElement | null = null;
  private deltaElement: HTMLElement | null = null;
  private recommendationElement: HTMLElement | null = null;
  private statusElement: HTMLElement | null = null;
  private riskSlider: HTMLInputElement | null = null;

  constructor() {
    this.init();
  }

  private async init() {
    console.log("ArcX Overlay initializing...");

    // Get UI elements
    this.evStayElement = document.getElementById("ev-stay");
    this.evExtractElement = document.getElementById("ev-extract");
    this.deltaElement = document.getElementById("delta-ev");
    this.recommendationElement = document.getElementById("recommendation");
    this.statusElement = document.getElementById("status");
    this.riskSlider = document.getElementById("risk-slider") as HTMLInputElement;

    // Setup event listeners
    this.setupEventListeners();

    // Check backend connection
    await this.checkConnection();

    // Start update loop
    this.startUpdateLoop();
  }

  private setupEventListeners() {
    // Risk profile slider
    if (this.riskSlider) {
      this.riskSlider.addEventListener("change", async (e) => {
        const value = parseInt((e.target as HTMLInputElement).value);
        const profiles = ["safe", "neutral", "aggressive"] as const;
        const profile = profiles[value];

        try {
          await apiClient.updateConfig(profile);
          console.log(`Risk profile updated to: ${profile}`);
        } catch (error) {
          console.error("Failed to update risk profile:", error);
        }
      });
    }

    // Feedback buttons
    const goodBtn = document.getElementById("feedback-good");
    const badBtn = document.getElementById("feedback-bad");

    if (goodBtn) {
      goodBtn.addEventListener("click", () => this.submitFeedback("good"));
    }

    if (badBtn) {
      badBtn.addEventListener("click", () => this.submitFeedback("bad"));
    }

    // Start run button
    const startBtn = document.getElementById("start-run");
    if (startBtn) {
      startBtn.addEventListener("click", () => this.startRun());
    }

    // End run button
    const endBtn = document.getElementById("end-run");
    if (endBtn) {
      endBtn.addEventListener("click", () => this.endRun());
    }
  }

  private async checkConnection(): Promise<void> {
    try {
      this.isConnected = await apiClient.checkConnection();
      console.log(`Backend connection: ${this.isConnected ? "OK" : "FAILED"}`);
      this.updateConnectionStatus();
    } catch (error) {
      console.error("Connection check failed:", error);
      this.isConnected = false;
      this.updateConnectionStatus();
    }
  }

  private updateConnectionStatus() {
    if (this.statusElement) {
      if (this.isConnected) {
        this.statusElement.textContent = "Connected";
        this.statusElement.className = "status connected";
      } else {
        this.statusElement.textContent = "Disconnected";
        this.statusElement.className = "status disconnected";
      }
    }
  }

  private startUpdateLoop() {
    if (this.updateTimer !== null) {
      return;
    }

    this.updateTimer = window.setInterval(() => {
      this.update();
    }, this.updateInterval);

    console.log("Update loop started");
  }

  private stopUpdateLoop() {
    if (this.updateTimer !== null) {
      clearInterval(this.updateTimer);
      this.updateTimer = null;
    }
  }

  private async update() {
    if (!this.isConnected) {
      await this.checkConnection();
      return;
    }

    try {
      // Get EV prediction
      const ev = await apiClient.getEV();
      this.updateUI(ev);
    } catch (error) {
      // Silently handle errors during prediction (buffer might be filling)
      // Only log if it's not a "buffer filling" error
      if (error instanceof Error && !error.message.includes("buffer filling")) {
        console.error("Failed to get EV:", error);
      }
    }
  }

  private updateUI(ev: EVResponse) {
    // Update EV values
    if (this.evStayElement) {
      this.evStayElement.textContent = ev.ev_stay.toFixed(1);
    }

    if (this.evExtractElement) {
      this.evExtractElement.textContent = ev.ev_extract.toFixed(1);
    }

    if (this.deltaElement) {
      const deltaText = ev.delta_ev > 0 ? `+${ev.delta_ev.toFixed(1)}` : ev.delta_ev.toFixed(1);
      this.deltaElement.textContent = deltaText;
      this.deltaElement.className = `delta ${ev.color}`;
    }

    // Update recommendation
    if (this.recommendationElement) {
      this.recommendationElement.textContent = ev.message;
      this.recommendationElement.className = `recommendation ${ev.color}`;
    }
  }

  private async submitFeedback(rating: "good" | "bad") {
    if (!this.currentRunId) {
      console.warn("No active run, cannot submit feedback");
      return;
    }

    try {
      await apiClient.submitFeedback(this.currentRunId, 0, rating);
      console.log(`Feedback submitted: ${rating}`);

      // Visual feedback
      const feedbackElement = document.getElementById("feedback-message");
      if (feedbackElement) {
        feedbackElement.textContent = "피드백 전송됨!";
        feedbackElement.style.display = "block";
        setTimeout(() => {
          feedbackElement.style.display = "none";
        }, 2000);
      }
    } catch (error) {
      console.error("Failed to submit feedback:", error);
    }
  }

  private async startRun() {
    try {
      const response = await apiClient.startRun();
      this.currentRunId = response.run_id;
      console.log(`Run started: ${this.currentRunId}`);

      // Update UI
      const runIdElement = document.getElementById("run-id");
      if (runIdElement) {
        runIdElement.textContent = this.currentRunId.substring(0, 8);
      }
    } catch (error) {
      console.error("Failed to start run:", error);
    }
  }

  private async endRun() {
    if (!this.currentRunId) {
      console.warn("No active run");
      return;
    }

    // TODO: Get actual values from game
    const finalLootValue = 0; // Placeholder
    const totalTimeSec = 0; // Placeholder
    const success = true; // Placeholder
    const actionTaken = "extract"; // Placeholder

    try {
      await apiClient.endRun(this.currentRunId, finalLootValue, totalTimeSec, success, actionTaken);
      console.log("Run ended");
      this.currentRunId = null;

      // Update UI
      const runIdElement = document.getElementById("run-id");
      if (runIdElement) {
        runIdElement.textContent = "-";
      }
    } catch (error) {
      console.error("Failed to end run:", error);
    }
  }
}

// Initialize when DOM is ready
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", () => {
    new OverlayApp();
  });
} else {
  new OverlayApp();
}
