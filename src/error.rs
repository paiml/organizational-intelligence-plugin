//! Error Handling for OIP
//!
//! PROD-002: Centralized error types with context and recovery hints
//! Uses thiserror for ergonomic error definitions

use thiserror::Error;

/// Main error type for OIP operations
#[derive(Error, Debug)]
pub enum OipError {
    // ===== Data Errors =====
    #[error("No data available: {context}")]
    NoData { context: String },

    #[error("Invalid data format: {message}")]
    InvalidData { message: String },

    #[error("Data validation failed: {field} - {reason}")]
    ValidationError { field: String, reason: String },

    // ===== GitHub/Git Errors =====
    #[error("GitHub API error: {message}")]
    GitHubError { message: String },

    #[error("Repository not found: {repo}")]
    RepoNotFound { repo: String },

    #[error("Git operation failed: {operation} - {reason}")]
    GitError { operation: String, reason: String },

    #[error("Authentication required: {message}")]
    AuthRequired { message: String },

    // ===== ML/Compute Errors =====
    #[error("Model not trained: call train() before predict()")]
    ModelNotTrained,

    #[error("Insufficient data for {operation}: need {required}, got {actual}")]
    InsufficientData {
        operation: String,
        required: usize,
        actual: usize,
    },

    #[error("Computation failed: {operation} - {reason}")]
    ComputeError { operation: String, reason: String },

    #[error("GPU not available: {reason}")]
    GpuUnavailable { reason: String },

    // ===== Storage Errors =====
    #[error("Storage error: {operation} - {reason}")]
    StorageError { operation: String, reason: String },

    #[error("File not found: {path}")]
    FileNotFound { path: String },

    #[error("IO error: {context}")]
    IoError {
        context: String,
        #[source]
        source: std::io::Error,
    },

    // ===== Configuration Errors =====
    #[error("Configuration error: {message}")]
    ConfigError { message: String },

    #[error("Invalid argument: {arg} - {reason}")]
    InvalidArgument { arg: String, reason: String },

    // ===== Generic Errors =====
    #[error("Operation failed: {message}")]
    OperationFailed { message: String },

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

impl OipError {
    // ===== Constructors =====

    pub fn no_data(context: impl Into<String>) -> Self {
        Self::NoData {
            context: context.into(),
        }
    }

    pub fn invalid_data(message: impl Into<String>) -> Self {
        Self::InvalidData {
            message: message.into(),
        }
    }

    pub fn validation(field: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::ValidationError {
            field: field.into(),
            reason: reason.into(),
        }
    }

    pub fn github(message: impl Into<String>) -> Self {
        Self::GitHubError {
            message: message.into(),
        }
    }

    pub fn repo_not_found(repo: impl Into<String>) -> Self {
        Self::RepoNotFound { repo: repo.into() }
    }

    pub fn git(operation: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::GitError {
            operation: operation.into(),
            reason: reason.into(),
        }
    }

    pub fn auth_required(message: impl Into<String>) -> Self {
        Self::AuthRequired {
            message: message.into(),
        }
    }

    pub fn insufficient_data(operation: impl Into<String>, required: usize, actual: usize) -> Self {
        Self::InsufficientData {
            operation: operation.into(),
            required,
            actual,
        }
    }

    pub fn compute(operation: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::ComputeError {
            operation: operation.into(),
            reason: reason.into(),
        }
    }

    pub fn gpu_unavailable(reason: impl Into<String>) -> Self {
        Self::GpuUnavailable {
            reason: reason.into(),
        }
    }

    pub fn storage(operation: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::StorageError {
            operation: operation.into(),
            reason: reason.into(),
        }
    }

    pub fn file_not_found(path: impl Into<String>) -> Self {
        Self::FileNotFound { path: path.into() }
    }

    pub fn io(context: impl Into<String>, source: std::io::Error) -> Self {
        Self::IoError {
            context: context.into(),
            source,
        }
    }

    pub fn config(message: impl Into<String>) -> Self {
        Self::ConfigError {
            message: message.into(),
        }
    }

    pub fn invalid_arg(arg: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::InvalidArgument {
            arg: arg.into(),
            reason: reason.into(),
        }
    }

    pub fn failed(message: impl Into<String>) -> Self {
        Self::OperationFailed {
            message: message.into(),
        }
    }

    // ===== Recovery Hints =====

    /// Get a user-friendly recovery hint for this error
    pub fn recovery_hint(&self) -> Option<&'static str> {
        match self {
            Self::NoData { .. } => Some("Try analyzing a repository first with 'oip-gpu analyze'"),
            Self::RepoNotFound { .. } => {
                Some("Check the repository name format (owner/repo) and ensure it exists")
            }
            Self::AuthRequired { .. } => Some("Set GITHUB_TOKEN environment variable"),
            Self::ModelNotTrained => Some("Train the model first with predictor.train(features)"),
            Self::InsufficientData { .. } => Some("Provide more training data or reduce k value"),
            Self::GpuUnavailable { .. } => {
                Some("Use --backend simd for CPU fallback, or install GPU drivers")
            }
            Self::FileNotFound { .. } => Some("Check the file path and ensure it exists"),
            Self::ConfigError { .. } => Some("Check configuration file syntax (YAML/TOML)"),
            Self::InvalidArgument { .. } => Some("Run with --help to see valid arguments"),
            _ => None,
        }
    }

    /// Check if this error is recoverable
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            Self::NoData { .. }
                | Self::RepoNotFound { .. }
                | Self::AuthRequired { .. }
                | Self::ModelNotTrained
                | Self::InsufficientData { .. }
                | Self::GpuUnavailable { .. }
                | Self::FileNotFound { .. }
                | Self::ConfigError { .. }
                | Self::InvalidArgument { .. }
        )
    }

    /// Get error category for logging/metrics
    pub fn category(&self) -> &'static str {
        match self {
            Self::NoData { .. } | Self::InvalidData { .. } | Self::ValidationError { .. } => "data",
            Self::GitHubError { .. }
            | Self::RepoNotFound { .. }
            | Self::GitError { .. }
            | Self::AuthRequired { .. } => "git",
            Self::ModelNotTrained
            | Self::InsufficientData { .. }
            | Self::ComputeError { .. }
            | Self::GpuUnavailable { .. } => "compute",
            Self::StorageError { .. } | Self::FileNotFound { .. } | Self::IoError { .. } => {
                "storage"
            }
            Self::ConfigError { .. } | Self::InvalidArgument { .. } => "config",
            Self::OperationFailed { .. } | Self::Other(_) => "other",
        }
    }
}

/// Result type alias for OIP operations
pub type OipResult<T> = Result<T, OipError>;

/// Extension trait for adding context to errors
pub trait ResultExt<T> {
    /// Add context to an error
    fn context(self, context: impl Into<String>) -> OipResult<T>;

    /// Add context with a closure (lazy evaluation)
    fn with_context<F, S>(self, f: F) -> OipResult<T>
    where
        F: FnOnce() -> S,
        S: Into<String>;
}

impl<T, E: Into<OipError>> ResultExt<T> for Result<T, E> {
    fn context(self, context: impl Into<String>) -> OipResult<T> {
        self.map_err(|e| {
            let inner = e.into();
            OipError::OperationFailed {
                message: format!("{}: {}", context.into(), inner),
            }
        })
    }

    fn with_context<F, S>(self, f: F) -> OipResult<T>
    where
        F: FnOnce() -> S,
        S: Into<String>,
    {
        self.map_err(|e| {
            let inner = e.into();
            OipError::OperationFailed {
                message: format!("{}: {}", f().into(), inner),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = OipError::no_data("empty feature store");
        assert!(err.to_string().contains("No data available"));
        assert!(err.to_string().contains("empty feature store"));
    }

    #[test]
    fn test_error_recovery_hint() {
        let err = OipError::ModelNotTrained;
        assert!(err.recovery_hint().is_some());
        assert!(err.recovery_hint().unwrap().contains("train"));
    }

    #[test]
    fn test_error_is_recoverable() {
        assert!(OipError::ModelNotTrained.is_recoverable());
        assert!(OipError::repo_not_found("test/repo").is_recoverable());
        assert!(!OipError::failed("unknown").is_recoverable());
    }

    #[test]
    fn test_error_category() {
        assert_eq!(OipError::ModelNotTrained.category(), "compute");
        assert_eq!(OipError::repo_not_found("test").category(), "git");
        assert_eq!(OipError::no_data("test").category(), "data");
    }

    #[test]
    fn test_insufficient_data_error() {
        let err = OipError::insufficient_data("k-means clustering", 10, 5);
        assert!(err.to_string().contains("10"));
        assert!(err.to_string().contains("5"));
        assert!(err.is_recoverable());
    }

    #[test]
    fn test_validation_error() {
        let err = OipError::validation("category", "must be 0-9");
        assert!(err.to_string().contains("category"));
        assert!(err.to_string().contains("must be 0-9"));
    }

    #[test]
    fn test_result_context() {
        let result: Result<(), OipError> = Err(OipError::no_data("test"));
        let with_context = result.context("during analysis");
        assert!(with_context.is_err());
        assert!(with_context.unwrap_err().to_string().contains("analysis"));
    }
}
