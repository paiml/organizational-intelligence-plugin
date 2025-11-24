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
        // Actual format: {"files": [{"file_path": "src/main.rs", "total": 95.0, "grade": "APLus"}]}

        #[derive(Deserialize)]
        struct PmatFile {
            file_path: String,
            total: f32,
            #[allow(dead_code)]
            #[serde(default)]
            grade: String,
        }

        #[derive(Deserialize)]
        struct PmatOutput {
            files: Vec<PmatFile>,
        }

        let parsed: PmatOutput = serde_json::from_str(json_output)
            .map_err(|e| anyhow!("Failed to parse pmat JSON: {}", e))?;

        let mut file_scores = HashMap::new();
        let mut total_score = 0.0_f32;
        let mut max_score = 0.0_f32;

        for file in &parsed.files {
            file_scores.insert(file.file_path.clone(), file.total);
            total_score += file.total;
            max_score = max_score.max(file.total);
        }

        let average_score = if parsed.files.is_empty() {
            0.0
        } else {
            total_score / parsed.files.len() as f32
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
                {"file_path": "src/main.rs", "total": 95.0, "grade": "APLus"},
                {"file_path": "src/lib.rs", "total": 88.0, "grade": "A"}
            ]
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

    #[test]
    fn test_parse_tdg_invalid_json() {
        let invalid_json = "not valid json";

        let result = PmatIntegration::parse_tdg_output(invalid_json);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("parse"));
    }

    #[test]
    fn test_parse_tdg_single_file() {
        let json = r#"{
            "files": [
                {"file_path": "src/single.rs", "total": 100.0, "grade": "APlusPlus"}
            ]
        }"#;

        let result = PmatIntegration::parse_tdg_output(json).unwrap();

        assert_eq!(result.average_score, 100.0);
        assert_eq!(result.max_score, 100.0);
        assert_eq!(result.file_scores.len(), 1);
    }

    #[test]
    fn test_parse_tdg_multiple_files() {
        let json = r#"{
            "files": [
                {"file_path": "file1.rs", "total": 90.0, "grade": "A"},
                {"file_path": "file2.rs", "total": 85.0, "grade": "B"},
                {"file_path": "file3.rs", "total": 95.0, "grade": "APLus"}
            ]
        }"#;

        let result = PmatIntegration::parse_tdg_output(json).unwrap();

        assert_eq!(result.average_score, 90.0);
        assert_eq!(result.max_score, 95.0);
        assert_eq!(result.file_scores.len(), 3);
    }

    #[test]
    fn test_parse_tdg_with_zero_scores() {
        let json = r#"{
            "files": [
                {"file_path": "bad1.rs", "total": 0.0, "grade": "F"},
                {"file_path": "bad2.rs", "total": 0.0, "grade": "F"}
            ]
        }"#;

        let result = PmatIntegration::parse_tdg_output(json).unwrap();

        assert_eq!(result.average_score, 0.0);
        assert_eq!(result.max_score, 0.0);
    }

    #[test]
    fn test_parse_tdg_without_grade_field() {
        let json = r#"{
            "files": [
                {"file_path": "src/main.rs", "total": 88.5}
            ]
        }"#;

        let result = PmatIntegration::parse_tdg_output(json).unwrap();

        assert_eq!(result.average_score, 88.5);
        assert_eq!(result.max_score, 88.5);
    }

    #[test]
    fn test_file_tdg_score_structure() {
        let score = FileTdgScore {
            path: "src/test.rs".to_string(),
            score: 92.5,
            grade: "A".to_string(),
        };

        assert_eq!(score.path, "src/test.rs");
        assert_eq!(score.score, 92.5);
        assert_eq!(score.grade, "A");
    }

    #[test]
    fn test_file_tdg_score_clone() {
        let original = FileTdgScore {
            path: "src/test.rs".to_string(),
            score: 92.5,
            grade: "A".to_string(),
        };

        let cloned = original.clone();

        assert_eq!(original.path, cloned.path);
        assert_eq!(original.score, cloned.score);
        assert_eq!(original.grade, cloned.grade);
    }

    #[test]
    fn test_file_tdg_score_debug() {
        let score = FileTdgScore {
            path: "src/test.rs".to_string(),
            score: 92.5,
            grade: "A".to_string(),
        };

        let debug_str = format!("{:?}", score);
        assert!(debug_str.contains("src/test.rs"));
        assert!(debug_str.contains("92.5"));
        assert!(debug_str.contains("A"));
    }

    #[test]
    fn test_tdg_analysis_clone() {
        let mut file_scores = HashMap::new();
        file_scores.insert("file.rs".to_string(), 85.0);

        let original = TdgAnalysis {
            file_scores: file_scores.clone(),
            average_score: 85.0,
            max_score: 85.0,
        };

        let cloned = original.clone();

        assert_eq!(original.average_score, cloned.average_score);
        assert_eq!(original.max_score, cloned.max_score);
        assert_eq!(original.file_scores.len(), cloned.file_scores.len());
    }

    #[test]
    fn test_tdg_analysis_debug() {
        let mut file_scores = HashMap::new();
        file_scores.insert("file.rs".to_string(), 85.0);

        let analysis = TdgAnalysis {
            file_scores,
            average_score: 85.0,
            max_score: 85.0,
        };

        let debug_str = format!("{:?}", analysis);
        assert!(debug_str.contains("85"));
    }

    #[test]
    fn test_get_file_score_nonexistent() {
        let analysis = TdgAnalysis {
            file_scores: HashMap::new(),
            average_score: 0.0,
            max_score: 0.0,
        };

        assert_eq!(
            PmatIntegration::get_file_score(&analysis, "missing.rs"),
            None
        );
    }

    #[test]
    fn test_get_file_score_empty_analysis() {
        let analysis = TdgAnalysis {
            file_scores: HashMap::new(),
            average_score: 0.0,
            max_score: 0.0,
        };

        assert_eq!(PmatIntegration::get_file_score(&analysis, "any.rs"), None);
    }

    #[test]
    fn test_parse_tdg_with_various_scores() {
        let json = r#"{
            "files": [
                {"file_path": "low.rs", "total": 10.5, "grade": "F"},
                {"file_path": "medium.rs", "total": 55.0, "grade": "C"},
                {"file_path": "high.rs", "total": 99.9, "grade": "APlusPlus"}
            ]
        }"#;

        let result = PmatIntegration::parse_tdg_output(json).unwrap();

        assert!(result.average_score > 50.0 && result.average_score < 60.0);
        assert_eq!(result.max_score, 99.9);
        assert_eq!(
            PmatIntegration::get_file_score(&result, "low.rs"),
            Some(10.5)
        );
    }

    #[test]
    fn test_file_tdg_score_serialization() {
        let score = FileTdgScore {
            path: "src/test.rs".to_string(),
            score: 92.5,
            grade: "A".to_string(),
        };

        let json = serde_json::to_string(&score).unwrap();
        let deserialized: FileTdgScore = serde_json::from_str(&json).unwrap();

        assert_eq!(score.path, deserialized.path);
        assert_eq!(score.score, deserialized.score);
        assert_eq!(score.grade, deserialized.grade);
    }

    #[test]
    fn test_parse_tdg_fractional_average() {
        let json = r#"{
            "files": [
                {"file_path": "file1.rs", "total": 33.3, "grade": "D"},
                {"file_path": "file2.rs", "total": 66.6, "grade": "B"},
                {"file_path": "file3.rs", "total": 99.9, "grade": "APlusPlus"}
            ]
        }"#;

        let result = PmatIntegration::parse_tdg_output(json).unwrap();

        // Average should be (33.3 + 66.6 + 99.9) / 3 = 66.6
        assert!((result.average_score - 66.6).abs() < 0.1);
    }

    #[test]
    fn test_parse_tdg_with_long_file_paths() {
        let long_path = "a/very/long/path/to/some/deeply/nested/directory/structure/file.rs";
        let json = format!(
            r#"{{
            "files": [
                {{"file_path": "{}", "total": 85.0, "grade": "A"}}
            ]
        }}"#,
            long_path
        );

        let result = PmatIntegration::parse_tdg_output(&json).unwrap();

        assert_eq!(
            PmatIntegration::get_file_score(&result, long_path),
            Some(85.0)
        );
    }

    #[test]
    fn test_is_pmat_available() {
        // This test will pass or fail depending on whether pmat is installed
        // But it exercises the code path
        let _available = PmatIntegration::is_pmat_available();
        // Just verify it returns without panicking
    }
}
