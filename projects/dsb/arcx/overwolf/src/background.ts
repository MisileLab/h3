/**
 * Background script for ArcX Overwolf app
 *
 * Manages window lifecycle and game events
 */

const OVERLAY_WINDOW_NAME = "overlay";

// Open overlay window when game starts
function openOverlay() {
  overwolf.windows.obtainDeclaredWindow(OVERLAY_WINDOW_NAME, (result) => {
    if (result.success) {
      overwolf.windows.restore(result.window.id, () => {
        console.log("Overlay window opened");
      });
    }
  });
}

// Close overlay window
function closeOverlay() {
  overwolf.windows.obtainDeclaredWindow(OVERLAY_WINDOW_NAME, (result) => {
    if (result.success) {
      overwolf.windows.close(result.window.id, () => {
        console.log("Overlay window closed");
      });
    }
  });
}

// Listen for game launch events
overwolf.games.onGameLaunched.addListener((info) => {
  console.log("Game launched:", info);
  openOverlay();
});

// Listen for game info updates
overwolf.games.onGameInfoUpdated.addListener((info) => {
  console.log("Game info updated:", info);
});

console.log("ArcX background script initialized");
