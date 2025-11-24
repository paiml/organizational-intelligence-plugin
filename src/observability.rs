//! Observability: Logging, Tracing, and Metrics
//!
//! PROD-003: Production-ready observability using tracing
//! Provides structured logging, span tracing, and performance metrics

use tracing::{debug, error, info, instrument, trace, warn};
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

/// Initialize the tracing subscriber
///
/// # Arguments
/// * `verbose` - Enable debug-level logging
/// * `_json` - Reserved for JSON output (requires tracing-subscriber json feature)
pub fn init_tracing(verbose: bool, _json: bool) {
    let filter = if verbose {
        EnvFilter::new("debug")
    } else {
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"))
    };

    // Note: JSON output requires tracing-subscriber with "json" feature
    // For now, always use compact format
    tracing_subscriber::registry()
        .with(filter)
        .with(fmt::layer().compact())
        .init();
}

/// Initialize tracing with custom filter
pub fn init_with_filter(filter: &str) {
    let filter = EnvFilter::new(filter);
    tracing_subscriber::registry()
        .with(filter)
        .with(fmt::layer().compact())
        .init();
}

/// Span for tracking analysis operations
#[derive(Debug)]
pub struct AnalysisSpan {
    pub repo: String,
    pub operation: String,
}

impl AnalysisSpan {
    pub fn new(repo: impl Into<String>, operation: impl Into<String>) -> Self {
        Self {
            repo: repo.into(),
            operation: operation.into(),
        }
    }
}

/// Log levels for different operations
pub struct LogOps;

impl LogOps {
    /// Log start of analysis
    #[instrument(skip_all, fields(repo = %repo, commits = commits))]
    pub fn analysis_start(repo: &str, commits: usize) {
        info!("Starting repository analysis");
    }

    /// Log analysis completion
    #[instrument(skip_all, fields(repo = %repo, patterns = patterns, duration_ms = duration_ms))]
    pub fn analysis_complete(repo: &str, patterns: usize, duration_ms: u64) {
        info!("Analysis complete");
    }

    /// Log feature extraction
    #[instrument(skip_all, fields(count = count))]
    pub fn features_extracted(count: usize) {
        debug!("Features extracted");
    }

    /// Log correlation computation
    #[instrument(skip_all, fields(size = size, backend = %backend))]
    pub fn correlation_start(size: usize, backend: &str) {
        debug!("Computing correlation");
    }

    /// Log correlation result
    #[instrument(skip_all, fields(result = %format!("{:.4}", result), duration_us = duration_us))]
    pub fn correlation_complete(result: f32, duration_us: u64) {
        trace!("Correlation computed");
    }

    /// Log ML training
    #[instrument(skip_all, fields(model = %model, samples = samples))]
    pub fn training_start(model: &str, samples: usize) {
        info!("Training model");
    }

    /// Log ML training complete
    #[instrument(skip_all, fields(model = %model, accuracy = %format!("{:.2}%", accuracy * 100.0)))]
    pub fn training_complete(model: &str, accuracy: f32) {
        info!("Model training complete");
    }

    /// Log prediction
    #[instrument(skip_all, fields(model = %model))]
    pub fn prediction(model: &str, category: u8) {
        debug!(category = category, "Prediction made");
    }

    /// Log clustering
    #[instrument(skip_all, fields(k = k, samples = samples))]
    pub fn clustering_start(k: usize, samples: usize) {
        debug!("Starting clustering");
    }

    /// Log clustering complete
    #[instrument(skip_all, fields(k = k, inertia = %format!("{:.2}", inertia)))]
    pub fn clustering_complete(k: usize, inertia: f32) {
        debug!("Clustering complete");
    }

    /// Log storage operation
    #[instrument(skip_all, fields(operation = %operation, count = count))]
    pub fn storage_op(operation: &str, count: usize) {
        trace!("Storage operation");
    }

    /// Log GPU operation
    #[instrument(skip_all, fields(operation = %operation, backend = %backend))]
    pub fn gpu_op(operation: &str, backend: &str) {
        debug!("GPU operation");
    }

    /// Log error with context
    pub fn error_with_context(error: &impl std::fmt::Display, context: &str) {
        error!(context = context, error = %error, "Operation failed");
    }

    /// Log warning
    pub fn warning(message: &str, context: &str) {
        warn!(context = context, "{}", message);
    }

    /// Log performance metric
    #[instrument(skip_all, fields(operation = %operation, duration_ms = duration_ms, throughput = throughput))]
    pub fn performance(operation: &str, duration_ms: u64, throughput: Option<f64>) {
        if let Some(tp) = throughput {
            info!(throughput_per_sec = %format!("{:.2}", tp), "Performance metric");
        } else {
            info!("Performance metric");
        }
    }
}

/// Timer for measuring operation duration
pub struct Timer {
    start: std::time::Instant,
    operation: String,
}

impl Timer {
    pub fn new(operation: impl Into<String>) -> Self {
        Self {
            start: std::time::Instant::now(),
            operation: operation.into(),
        }
    }

    pub fn elapsed_ms(&self) -> u64 {
        self.start.elapsed().as_millis() as u64
    }

    pub fn elapsed_us(&self) -> u64 {
        self.start.elapsed().as_micros() as u64
    }

    pub fn log_completion(&self) {
        let duration = self.elapsed_ms();
        debug!(
            operation = %self.operation,
            duration_ms = duration,
            "Operation completed"
        );
    }
}

impl Drop for Timer {
    fn drop(&mut self) {
        // Optionally log on drop
    }
}

/// Metrics collector for aggregating statistics
#[derive(Debug, Default)]
pub struct Metrics {
    pub analyses_count: u64,
    pub features_extracted: u64,
    pub correlations_computed: u64,
    pub predictions_made: u64,
    pub errors_count: u64,
    pub total_duration_ms: u64,
}

impl Metrics {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_analysis(&mut self, duration_ms: u64) {
        self.analyses_count += 1;
        self.total_duration_ms += duration_ms;
    }

    pub fn record_features(&mut self, count: u64) {
        self.features_extracted += count;
    }

    pub fn record_correlation(&mut self) {
        self.correlations_computed += 1;
    }

    pub fn record_prediction(&mut self) {
        self.predictions_made += 1;
    }

    pub fn record_error(&mut self) {
        self.errors_count += 1;
    }

    pub fn summary(&self) -> String {
        format!(
            "Metrics: analyses={}, features={}, correlations={}, predictions={}, errors={}, total_time={}ms",
            self.analyses_count,
            self.features_extracted,
            self.correlations_computed,
            self.predictions_made,
            self.errors_count,
            self.total_duration_ms
        )
    }

    pub fn log_summary(&self) {
        info!(
            analyses = self.analyses_count,
            features = self.features_extracted,
            correlations = self.correlations_computed,
            predictions = self.predictions_made,
            errors = self.errors_count,
            total_duration_ms = self.total_duration_ms,
            "Session metrics"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timer_creation() {
        let timer = Timer::new("test_operation");
        assert_eq!(timer.operation, "test_operation");
    }

    #[test]
    fn test_timer_elapsed() {
        let timer = Timer::new("test");
        std::thread::sleep(std::time::Duration::from_millis(10));
        assert!(timer.elapsed_ms() >= 10);
    }

    #[test]
    fn test_metrics_default() {
        let metrics = Metrics::new();
        assert_eq!(metrics.analyses_count, 0);
        assert_eq!(metrics.errors_count, 0);
    }

    #[test]
    fn test_metrics_recording() {
        let mut metrics = Metrics::new();
        metrics.record_analysis(100);
        metrics.record_features(50);
        metrics.record_correlation();
        metrics.record_prediction();
        metrics.record_error();

        assert_eq!(metrics.analyses_count, 1);
        assert_eq!(metrics.features_extracted, 50);
        assert_eq!(metrics.correlations_computed, 1);
        assert_eq!(metrics.predictions_made, 1);
        assert_eq!(metrics.errors_count, 1);
        assert_eq!(metrics.total_duration_ms, 100);
    }

    #[test]
    fn test_metrics_summary() {
        let mut metrics = Metrics::new();
        metrics.record_analysis(50);
        metrics.record_features(100);

        let summary = metrics.summary();
        assert!(summary.contains("analyses=1"));
        assert!(summary.contains("features=100"));
    }

    #[test]
    fn test_analysis_span() {
        let span = AnalysisSpan::new("owner/repo", "analyze");
        assert_eq!(span.repo, "owner/repo");
        assert_eq!(span.operation, "analyze");
    }
}
