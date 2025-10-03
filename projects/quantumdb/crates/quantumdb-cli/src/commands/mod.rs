//! CLI command implementations

pub mod train;
pub mod build;
pub mod serve;
pub mod query;
pub mod benchmark;
pub mod stats;

pub use train::*;
pub use build::*;
pub use serve::*;
pub use query::*;
pub use benchmark::*;
pub use stats::*;