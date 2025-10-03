//! Progress reporting utilities

use indicatif::{ProgressBar, ProgressStyle};
use std::time::Duration;

/// Create a new progress bar
pub fn create_progress_bar(total: u64) -> ProgressBar {
    let pb = ProgressBar::new(total);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
            .unwrap()
            .progress_chars("#>-")
    );
    pb.set_elapsed(Duration::from_secs(0));
    pb
}

/// Create a spinner for indeterminate progress
pub fn create_spinner() -> ProgressBar {
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} {msg}")
            .unwrap()
    );
    pb.enable_steady_tick(Duration::from_millis(100));
    pb
}