// Organizational Intelligence Plugin - Library
// Toyota Way: Start simple, evolve based on evidence

pub mod cli;
pub mod github;
pub mod report;

// Re-export main types for convenience
pub use cli::{Cli, Commands};
