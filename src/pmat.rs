// pmat integration module
// Toyota Way: Integrate existing quality tools rather than reinventing
// Phase 1.5: Add TDG, SATD, and complexity metrics to defect analysis

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::process::Command;
use tracing::{debug, info, warn};

/// TDG analysis result for a file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileTdgScore {
    pub path: String,
    pub score: f32,
    pub grade: String,
}

/// Aggregated TDG results for a repository
#[derive(Debug, Clone)]
pub struct TdgAnalysis {
    /// Map of file path to TDG score
    pub file_scores: HashMap<String, f32>,
    /// Average TDG score across all files
    pub average_score: f32,
    /// Maximum TDG score (worst file)
    pub max_score: f32,
}

/// pmat integration wrapper
pub struct PmatIntegration;

impl PmatIntegration {
    /// Run pmat TDG analysis on a repository
    ///
    /// # Arguments
    /// * `repo_path` - Path to the repository to analyze
    ///
    /// # Returns
    /// * `Ok(TdgAnalysis)` with TDG scores
    /// * `Err` if pmat is not available or analysis fails
    ///
    /// # Examples
    /// ```no_run
    /// use organizational_intelligence_plugin::pmat::PmatIntegration;
    /// use std::path::PathBuf;
    ///
    /// let analysis = PmatIntegration::analyze_tdg(&PathBuf::from("/tmp/repo")).unwrap();
    /// println!("Average TDG: {}", analysis.average_score);
    /// ```
    pub fn analyze_tdg<P: AsRef<Path>>(repo_path: P) -> Result<TdgAnalysis> {
        let path = repo_path.as_ref();
        info!("Running pmat TDG analysis on {:?}", path);

        // Check if pmat is available
        if !Self::is_pmat_available() {
            warn!("pmat command not found - TDG analysis unavailable");
            return Err(anyhow!("pmat command not available in PATH"));
        }

        // Run pmat analyze tdg --path {repo_path} --format json
        let output = Command::new("pmat")
            .args(["analyze", "tdg", "--path"])
            .arg(path)
            .args(["--format", "json"])
            .output()
            .map_err(|e| anyhow!("Failed to execute pmat: {}", e))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow!("pmat tdg failed: {}", stderr));
        }

        // Parse JSON output
        let stdout = String::from_utf8_lossy(&output.stdout);
        debug!("pmat output: {}", stdout);

        Self::parse_tdg_output(&stdout)
    }

    /// Check if pmat command is available
    fn is_pmat_available() -> bool {
        Command::new("pmat")
            .arg("--version")
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false)
    }

    /// Parse pmat TDG JSON output
    fn parse_tdg_output(json_output: &str) -> Result<TdgAnalysis> {
        // pmat outputs JSON with file scores
        // Expected format: {"files": [{"path": "src/main.rs", "score": 95.0, "grade": "A+"}]}

        #[derive(Deserialize)]
        struct PmatOutput {
            files: Vec<FileTdgScore>,
            #[serde(default)]
            average_score: Option<f32>,
        }

        let parsed: PmatOutput = serde_json::from_str(json_output)
            .map_err(|e| anyhow!("Failed to parse pmat JSON: {}", e))?;

        let mut file_scores = HashMap::new();
        let mut total_score = 0.0_f32;
        let mut max_score = 0.0_f32;

        for file in &parsed.files {
            file_scores.insert(file.path.clone(), file.score);
            total_score += file.score;
            max_score = max_score.max(file.score);
        }

        let average_score = if parsed.files.is_empty() {
            0.0
        } else {
            parsed
                .average_score
                .unwrap_or(total_score / parsed.files.len() as f32)
        };

        Ok(TdgAnalysis {
            file_scores,
            average_score,
            max_score,
        })
    }

    /// Get TDG score for a specific file
    ///
    /// # Arguments
    /// * `analysis` - TDG analysis result
    /// * `file_path` - Path to look up
    ///
    /// # Returns
    /// * `Some(score)` if file was analyzed
    /// * `None` if file not found
    pub fn get_file_score(analysis: &TdgAnalysis, file_path: &str) -> Option<f32> {
        analysis.file_scores.get(file_path).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_tdg_output() {
        let json = r#"{
            "files": [
                {"path": "src/main.rs", "score": 95.0, "grade": "A+"},
                {"path": "src/lib.rs", "score": 88.0, "grade": "A"}
            ],
            "average_score": 91.5
        }"#;

        let result = PmatIntegration::parse_tdg_output(json).unwrap();

        assert_eq!(result.average_score, 91.5);
        assert_eq!(result.max_score, 95.0);
        assert_eq!(result.file_scores.len(), 2);
        assert_eq!(result.file_scores.get("src/main.rs"), Some(&95.0));
    }

    #[test]
    fn test_parse_empty_tdg_output() {
        let json = r#"{
            "files": []
        }"#;

        let result = PmatIntegration::parse_tdg_output(json).unwrap();

        assert_eq!(result.average_score, 0.0);
        assert_eq!(result.max_score, 0.0);
        assert_eq!(result.file_scores.len(), 0);
    }

    #[test]
    fn test_get_file_score() {
        let mut file_scores = HashMap::new();
        file_scores.insert("src/main.rs".to_string(), 95.0);
        file_scores.insert("src/lib.rs".to_string(), 88.0);

        let analysis = TdgAnalysis {
            file_scores,
            average_score: 91.5,
            max_score: 95.0,
        };

        assert_eq!(
            PmatIntegration::get_file_score(&analysis, "src/main.rs"),
            Some(95.0)
        );
        assert_eq!(
            PmatIntegration::get_file_score(&analysis, "nonexistent.rs"),
            None
        );
    }

    // Integration test requiring pmat to be installed
    #[test]
    #[ignore]
    fn test_analyze_tdg_integration() {
        // This test requires pmat to be installed
        let temp_dir = tempfile::TempDir::new().unwrap();

        // Create a simple Rust file
        std::fs::write(
            temp_dir.path().join("test.rs"),
            "fn main() { println!(\"Hello\"); }",
        )
        .unwrap();

        let result = PmatIntegration::analyze_tdg(temp_dir.path());

        // Should either succeed or fail gracefully if pmat not available
        match result {
            Ok(analysis) => {
                assert!(analysis.average_score >= 0.0);
                assert!(analysis.average_score <= 100.0);
            }
            Err(e) => {
                // Expected if pmat not installed
                assert!(e.to_string().contains("pmat"));
            }
        }
    }
}
