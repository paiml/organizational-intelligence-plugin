// Organizational Intelligence Plugin - Library
// Toyota Way: Start simple, evolve based on evidence

pub mod config;
pub mod error;
pub mod observability;
pub mod perf;

pub mod analyzer;
pub mod classifier;
pub mod cli;
pub mod cli_handlers;
pub mod git;
pub mod github;
pub mod nlp;
pub mod pmat;
pub mod pr_reviewer;
pub mod report;
pub mod summarizer;
pub mod training;

// GPU acceleration modules
pub mod correlation;
pub mod features;
pub mod gpu_store;
pub mod imbalance;
pub mod ml;
pub mod query;
pub mod sliding_window;
pub mod storage;

// GPU compute (Phase 2) - requires `gpu` feature flag
#[cfg(feature = "gpu")]
pub mod gpu_correlation;

// Re-export main types for convenience
pub use cli::{Cli, Commands};
