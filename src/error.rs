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

    #[test]
    fn test_result_with_context() {
        let result: Result<(), OipError> = Err(OipError::no_data("test"));
        let with_context = result.with_context(|| "lazy context");
        assert!(with_context.is_err());
        assert!(with_context.unwrap_err().to_string().contains("lazy"));
    }

    #[test]
    fn test_invalid_data_constructor() {
        let err = OipError::invalid_data("malformed JSON");
        assert!(err.to_string().contains("Invalid data format"));
        assert_eq!(err.category(), "data");
    }

    #[test]
    fn test_github_error_constructor() {
        let err = OipError::github("rate limit exceeded");
        assert!(err.to_string().contains("GitHub API error"));
        assert_eq!(err.category(), "git");
    }

    #[test]
    fn test_git_error_constructor() {
        let err = OipError::git("clone", "network timeout");
        assert!(err.to_string().contains("Git operation failed"));
        assert!(err.to_string().contains("clone"));
        assert_eq!(err.category(), "git");
    }

    #[test]
    fn test_auth_required_constructor() {
        let err = OipError::auth_required("GitHub API requires token");
        assert!(err.to_string().contains("Authentication required"));
        assert!(err.recovery_hint().is_some());
        assert!(err.recovery_hint().unwrap().contains("GITHUB_TOKEN"));
        assert!(err.is_recoverable());
    }

    #[test]
    fn test_compute_error_constructor() {
        let err = OipError::compute("correlation", "division by zero");
        assert!(err.to_string().contains("Computation failed"));
        assert_eq!(err.category(), "compute");
    }

    #[test]
    fn test_gpu_unavailable_constructor() {
        let err = OipError::gpu_unavailable("no Vulkan driver");
        assert!(err.to_string().contains("GPU not available"));
        assert!(err.recovery_hint().unwrap().contains("simd"));
        assert!(err.is_recoverable());
    }

    #[test]
    fn test_storage_error_constructor() {
        let err = OipError::storage("save", "disk full");
        assert!(err.to_string().contains("Storage error"));
        assert_eq!(err.category(), "storage");
    }

    #[test]
    fn test_file_not_found_constructor() {
        let err = OipError::file_not_found("/tmp/missing.db");
        assert!(err.to_string().contains("File not found"));
        assert!(err.recovery_hint().is_some());
        assert!(err.is_recoverable());
    }

    #[test]
    fn test_io_error_constructor() {
        let io_err = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "access denied");
        let err = OipError::io("reading file", io_err);
        assert!(err.to_string().contains("IO error"));
        assert_eq!(err.category(), "storage");
    }

    #[test]
    fn test_config_error_constructor() {
        let err = OipError::config("invalid YAML syntax");
        assert!(err.to_string().contains("Configuration error"));
        assert!(err.recovery_hint().unwrap().contains("YAML"));
        assert!(err.is_recoverable());
    }

    #[test]
    fn test_invalid_arg_constructor() {
        let err = OipError::invalid_arg("--backend", "must be simd or gpu");
        assert!(err.to_string().contains("Invalid argument"));
        assert!(err.recovery_hint().unwrap().contains("--help"));
        assert!(err.is_recoverable());
    }

    #[test]
    fn test_failed_constructor() {
        let err = OipError::failed("network unreachable");
        assert!(err.to_string().contains("Operation failed"));
        assert!(!err.is_recoverable());
        assert_eq!(err.category(), "other");
    }

    #[test]
    fn test_model_not_trained_recovery() {
        let err = OipError::ModelNotTrained;
        assert!(err.recovery_hint().unwrap().contains("train"));
        assert!(err.is_recoverable());
        assert_eq!(err.category(), "compute");
    }

    #[test]
    fn test_repo_not_found_recovery() {
        let err = OipError::repo_not_found("invalid/repo");
        assert!(err.recovery_hint().unwrap().contains("owner/repo"));
        assert!(err.is_recoverable());
    }

    #[test]
    fn test_no_data_recovery() {
        let err = OipError::no_data("empty store");
        assert!(err.recovery_hint().unwrap().contains("analyze"));
        assert!(err.is_recoverable());
    }

    #[test]
    fn test_non_recoverable_errors() {
        assert!(!OipError::invalid_data("test").is_recoverable());
        assert!(!OipError::github("test").is_recoverable());
        assert!(!OipError::git("op", "reason").is_recoverable());
        assert!(!OipError::compute("op", "reason").is_recoverable());
        assert!(!OipError::storage("op", "reason").is_recoverable());
    }

    #[test]
    fn test_category_assignments() {
        // Data errors
        assert_eq!(OipError::invalid_data("test").category(), "data");
        assert_eq!(OipError::validation("f", "r").category(), "data");

        // Git errors
        assert_eq!(OipError::github("test").category(), "git");
        assert_eq!(OipError::git("o", "r").category(), "git");
        assert_eq!(OipError::auth_required("test").category(), "git");

        // Compute errors
        assert_eq!(OipError::compute("o", "r").category(), "compute");

        // Storage errors
        assert_eq!(OipError::storage("o", "r").category(), "storage");
        let io = std::io::Error::other("test");
        assert_eq!(OipError::io("ctx", io).category(), "storage");

        // Config errors
        assert_eq!(OipError::config("test").category(), "config");
        assert_eq!(OipError::invalid_arg("a", "r").category(), "config");
    }
}
