//! Configuration Management
//!
//! PROD-004: Centralized configuration with file and environment support
//! Supports YAML files with environment variable overrides

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Main configuration structure
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    /// Analysis settings
    pub analysis: AnalysisConfig,

    /// ML model settings
    pub ml: MlConfig,

    /// Storage settings
    pub storage: StorageConfig,

    /// GPU/compute settings
    pub compute: ComputeConfig,

    /// Logging settings
    pub logging: LoggingConfig,
}

/// Analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AnalysisConfig {
    /// Maximum commits to analyze per repository
    pub max_commits: usize,

    /// Number of parallel workers
    pub workers: usize,

    /// Cache directory for cloned repos
    pub cache_dir: String,

    /// Include merge commits
    pub include_merges: bool,
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            max_commits: 1000,
            workers: num_cpus::get().max(1),
            cache_dir: ".oip-cache".to_string(),
            include_merges: false,
        }
    }
}

/// ML model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct MlConfig {
    /// Number of trees for Random Forest
    pub n_trees: usize,

    /// Maximum tree depth
    pub max_depth: usize,

    /// Number of clusters for K-means
    pub k_clusters: usize,

    /// K-means max iterations
    pub max_iterations: usize,

    /// SMOTE k-neighbors
    pub smote_k: usize,

    /// Target minority ratio for SMOTE
    pub smote_ratio: f32,
}

impl Default for MlConfig {
    fn default() -> Self {
        Self {
            n_trees: 100,
            max_depth: 10,
            k_clusters: 5,
            max_iterations: 100,
            smote_k: 5,
            smote_ratio: 0.5,
        }
    }
}

/// Storage configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct StorageConfig {
    /// Default output file
    pub default_output: String,

    /// Enable compression
    pub compress: bool,

    /// Batch size for bulk operations
    pub batch_size: usize,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            default_output: "oip-gpu.db".to_string(),
            compress: true,
            batch_size: 1000,
        }
    }
}

/// Compute/GPU configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ComputeConfig {
    /// Preferred backend: "auto", "gpu", "simd", "cpu"
    pub backend: String,

    /// GPU workgroup size
    pub workgroup_size: usize,

    /// Enable GPU if available
    pub gpu_enabled: bool,
}

impl Default for ComputeConfig {
    fn default() -> Self {
        Self {
            backend: "auto".to_string(),
            workgroup_size: 256,
            gpu_enabled: true,
        }
    }
}

/// Logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LoggingConfig {
    /// Log level: "trace", "debug", "info", "warn", "error"
    pub level: String,

    /// Enable JSON output
    pub json: bool,

    /// Log file path (optional)
    pub file: Option<String>,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
            json: false,
            file: None,
        }
    }
}

impl Config {
    /// Load configuration from file
    pub fn from_file(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Config = serde_yaml::from_str(&content)?;
        Ok(config)
    }

    /// Load configuration with environment overrides
    pub fn load(path: Option<&Path>) -> Result<Self> {
        let mut config = if let Some(p) = path {
            if p.exists() {
                Self::from_file(p)?
            } else {
                Self::default()
            }
        } else {
            // Try default locations
            let default_paths = [".oip.yaml", ".oip.yml", "oip.yaml", "oip.yml"];
            let mut found = None;
            for p in &default_paths {
                let path = Path::new(p);
                if path.exists() {
                    found = Some(Self::from_file(path)?);
                    break;
                }
            }
            found.unwrap_or_default()
        };

        // Apply environment overrides
        config.apply_env_overrides();

        Ok(config)
    }

    /// Apply environment variable overrides
    fn apply_env_overrides(&mut self) {
        // Analysis
        if let Ok(val) = std::env::var("OIP_MAX_COMMITS") {
            if let Ok(n) = val.parse() {
                self.analysis.max_commits = n;
            }
        }
        if let Ok(val) = std::env::var("OIP_WORKERS") {
            if let Ok(n) = val.parse() {
                self.analysis.workers = n;
            }
        }
        if let Ok(val) = std::env::var("OIP_CACHE_DIR") {
            self.analysis.cache_dir = val;
        }

        // ML
        if let Ok(val) = std::env::var("OIP_K_CLUSTERS") {
            if let Ok(n) = val.parse() {
                self.ml.k_clusters = n;
            }
        }

        // Compute
        if let Ok(val) = std::env::var("OIP_BACKEND") {
            self.compute.backend = val;
        }
        if let Ok(val) = std::env::var("OIP_GPU_ENABLED") {
            self.compute.gpu_enabled = val == "1" || val.to_lowercase() == "true";
        }

        // Logging
        if let Ok(val) = std::env::var("OIP_LOG_LEVEL") {
            self.logging.level = val;
        }
        if let Ok(val) = std::env::var("OIP_LOG_JSON") {
            self.logging.json = val == "1" || val.to_lowercase() == "true";
        }
    }

    /// Save configuration to file
    pub fn save(&self, path: &Path) -> Result<()> {
        let content = serde_yaml::to_string(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        if self.analysis.max_commits == 0 {
            anyhow::bail!("max_commits must be > 0");
        }
        if self.analysis.workers == 0 {
            anyhow::bail!("workers must be > 0");
        }
        if self.ml.k_clusters == 0 {
            anyhow::bail!("k_clusters must be > 0");
        }
        if self.ml.smote_ratio <= 0.0 || self.ml.smote_ratio > 1.0 {
            anyhow::bail!("smote_ratio must be in (0, 1]");
        }
        Ok(())
    }

    /// Generate example configuration
    pub fn example_yaml() -> String {
        let config = Config::default();
        serde_yaml::to_string(&config).unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.analysis.max_commits, 1000);
        assert_eq!(config.ml.k_clusters, 5);
        assert_eq!(config.compute.backend, "auto");
    }

    #[test]
    fn test_config_validation() {
        let config = Config::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_invalid_config() {
        let mut config = Config::default();
        config.analysis.max_commits = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_save_load() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("test-config.yaml");

        let config = Config::default();
        config.save(&config_path).unwrap();

        let loaded = Config::from_file(&config_path).unwrap();
        assert_eq!(loaded.analysis.max_commits, config.analysis.max_commits);
        assert_eq!(loaded.ml.k_clusters, config.ml.k_clusters);
    }

    #[test]
    fn test_example_yaml() {
        let yaml = Config::example_yaml();
        assert!(yaml.contains("analysis"));
        assert!(yaml.contains("ml"));
        assert!(yaml.contains("compute"));
    }

    #[test]
    fn test_load_missing_file() {
        // Clean up any env vars from other tests
        std::env::remove_var("OIP_MAX_COMMITS");
        std::env::remove_var("OIP_GPU_ENABLED");

        let config = Config::load(Some(Path::new("nonexistent.yaml"))).unwrap();
        // Should return defaults when file doesn't exist
        assert_eq!(config.analysis.max_commits, 1000);
    }

    #[test]
    fn test_load_no_path_no_defaults() {
        // Clean up any env vars from other tests
        std::env::remove_var("OIP_MAX_COMMITS");
        std::env::remove_var("OIP_GPU_ENABLED");

        // Load with no path and no default files present
        let config = Config::load(None).unwrap();
        assert_eq!(config.analysis.max_commits, 1000); // Should use defaults
    }

    #[test]
    fn test_env_overrides_max_commits() {
        std::env::set_var("OIP_MAX_COMMITS", "500");
        let mut config = Config::default();
        config.apply_env_overrides();
        assert_eq!(config.analysis.max_commits, 500);
        std::env::remove_var("OIP_MAX_COMMITS");
    }

    #[test]
    fn test_env_overrides_workers() {
        std::env::set_var("OIP_WORKERS", "8");
        let mut config = Config::default();
        config.apply_env_overrides();
        assert_eq!(config.analysis.workers, 8);
        std::env::remove_var("OIP_WORKERS");
    }

    #[test]
    fn test_env_overrides_cache_dir() {
        std::env::set_var("OIP_CACHE_DIR", "/tmp/custom-cache");
        let mut config = Config::default();
        config.apply_env_overrides();
        assert_eq!(config.analysis.cache_dir, "/tmp/custom-cache");
        std::env::remove_var("OIP_CACHE_DIR");
    }

    #[test]
    fn test_env_overrides_k_clusters() {
        std::env::set_var("OIP_K_CLUSTERS", "10");
        let mut config = Config::default();
        config.apply_env_overrides();
        assert_eq!(config.ml.k_clusters, 10);
        std::env::remove_var("OIP_K_CLUSTERS");
    }

    #[test]
    fn test_env_overrides_backend() {
        std::env::set_var("OIP_BACKEND", "simd");
        let mut config = Config::default();
        config.apply_env_overrides();
        assert_eq!(config.compute.backend, "simd");
        std::env::remove_var("OIP_BACKEND");
    }

    #[test]
    fn test_env_overrides_gpu_enabled_true() {
        std::env::set_var("OIP_GPU_ENABLED", "true");
        let mut config = Config::default();
        config.apply_env_overrides();
        assert!(config.compute.gpu_enabled);
        std::env::remove_var("OIP_GPU_ENABLED");
    }

    #[test]
    fn test_env_overrides_gpu_enabled_1() {
        std::env::set_var("OIP_GPU_ENABLED", "1");
        let mut config = Config::default();
        config.apply_env_overrides();
        assert!(config.compute.gpu_enabled);
        std::env::remove_var("OIP_GPU_ENABLED");
    }

    #[test]
    fn test_env_overrides_gpu_enabled_false() {
        std::env::set_var("OIP_GPU_ENABLED", "false");
        let mut config = Config::default();
        config.compute.gpu_enabled = true; // Start with true
        config.apply_env_overrides();
        assert!(!config.compute.gpu_enabled);
        std::env::remove_var("OIP_GPU_ENABLED");
    }

    #[test]
    fn test_env_overrides_log_level() {
        std::env::set_var("OIP_LOG_LEVEL", "debug");
        let mut config = Config::default();
        config.apply_env_overrides();
        assert_eq!(config.logging.level, "debug");
        std::env::remove_var("OIP_LOG_LEVEL");
    }

    #[test]
    fn test_env_overrides_log_json() {
        std::env::set_var("OIP_LOG_JSON", "1");
        let mut config = Config::default();
        config.apply_env_overrides();
        assert!(config.logging.json);
        std::env::remove_var("OIP_LOG_JSON");
    }

    #[test]
    fn test_validation_workers_zero() {
        let mut config = Config::default();
        config.analysis.workers = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validation_k_clusters_zero() {
        let mut config = Config::default();
        config.ml.k_clusters = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validation_smote_ratio_zero() {
        let mut config = Config::default();
        config.ml.smote_ratio = 0.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validation_smote_ratio_over_one() {
        let mut config = Config::default();
        config.ml.smote_ratio = 1.5;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validation_smote_ratio_exactly_one() {
        let mut config = Config::default();
        config.ml.smote_ratio = 1.0;
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_analysis_config_defaults() {
        let config = AnalysisConfig::default();
        assert_eq!(config.max_commits, 1000);
        assert!(config.workers > 0); // At least 1
        assert_eq!(config.cache_dir, ".oip-cache");
        assert!(!config.include_merges);
    }

    #[test]
    fn test_ml_config_defaults() {
        let config = MlConfig::default();
        assert_eq!(config.n_trees, 100);
        assert_eq!(config.max_depth, 10);
        assert_eq!(config.k_clusters, 5);
        assert_eq!(config.max_iterations, 100);
        assert_eq!(config.smote_k, 5);
        assert_eq!(config.smote_ratio, 0.5);
    }

    #[test]
    fn test_storage_config_defaults() {
        let config = StorageConfig::default();
        assert_eq!(config.default_output, "oip-gpu.db");
        assert!(config.compress);
        assert_eq!(config.batch_size, 1000);
    }

    #[test]
    fn test_compute_config_defaults() {
        let config = ComputeConfig::default();
        assert_eq!(config.backend, "auto");
        assert_eq!(config.workgroup_size, 256);
        assert!(config.gpu_enabled);
    }

    #[test]
    fn test_logging_config_defaults() {
        let config = LoggingConfig::default();
        assert_eq!(config.level, "info");
        assert!(!config.json);
        assert!(config.file.is_none());
    }

    #[test]
    fn test_config_serialization() {
        let config = Config::default();
        let yaml = serde_yaml::to_string(&config).unwrap();
        let deserialized: Config = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(
            config.analysis.max_commits,
            deserialized.analysis.max_commits
        );
        assert_eq!(config.ml.k_clusters, deserialized.ml.k_clusters);
    }

    #[test]
    fn test_invalid_env_value_ignored() {
        // Clean up any env vars from other tests
        std::env::remove_var("OIP_GPU_ENABLED");

        std::env::set_var("OIP_MAX_COMMITS", "not-a-number");
        let config = Config::load(None).unwrap();
        assert_eq!(config.analysis.max_commits, 1000); // Should use default
        std::env::remove_var("OIP_MAX_COMMITS");
    }
}
