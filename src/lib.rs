// Organizational Intelligence Plugin - Library
// Toyota Way: Start simple, evolve based on evidence

pub mod analyzer;
pub mod classifier;
pub mod cli;
pub mod git;
pub mod github;
pub mod pmat;
pub mod pr_reviewer;
pub mod report;
pub mod summarizer;

// GPU acceleration modules
pub mod correlation;
pub mod features;
pub mod gpu_store;
pub mod query;
pub mod storage;

// Re-export main types for convenience
pub use cli::{Cli, Commands};
