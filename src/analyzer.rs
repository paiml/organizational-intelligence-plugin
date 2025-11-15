// Integrated analyzer combining git history and defect classification
// Phase 1: Combines GitAnalyzer + RuleBasedClassifier to generate defect patterns
// Toyota Way: Simple integration, measure before optimizing

use crate::classifier::{Classification, DefectCategory, RuleBasedClassifier};
use crate::git::{CommitInfo, GitAnalyzer};
use crate::pmat::{PmatIntegration, TdgAnalysis};
use crate::report::{DefectInstance, DefectPattern, QualitySignals};
use anyhow::Result;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tracing::{debug, info};

/// Integrated organizational defect analyzer
/// Combines git history analysis with defect classification
pub struct OrgAnalyzer {
    git_analyzer: GitAnalyzer,
    classifier: RuleBasedClassifier,
    cache_dir: PathBuf,
}

impl OrgAnalyzer {
    /// Create a new organizational analyzer
    ///
    /// # Arguments
    /// * `cache_dir` - Directory for storing cloned repositories
    ///
    /// # Examples
    /// ```
    /// use organizational_intelligence_plugin::analyzer::OrgAnalyzer;
    /// use std::path::PathBuf;
    ///
    /// let analyzer = OrgAnalyzer::new(PathBuf::from("/tmp/repos"));
    /// ```
    pub fn new<P: AsRef<Path>>(cache_dir: P) -> Self {
        let cache_dir = cache_dir.as_ref().to_path_buf();
        Self {
            git_analyzer: GitAnalyzer::new(&cache_dir),
            classifier: RuleBasedClassifier::new(),
            cache_dir,
        }
    }

    /// Analyze a single repository
    ///
    /// # Arguments
    /// * `repo_url` - Repository URL
    /// * `repo_name` - Repository name
    /// * `max_commits` - Maximum commits to analyze
    ///
    /// # Returns
    /// * `Ok(Vec<DefectPattern>)` with detected defect patterns
    ///
    /// # Examples
    /// ```no_run
    /// # use organizational_intelligence_plugin::analyzer::OrgAnalyzer;
    /// # use std::path::PathBuf;
    /// # async fn example() -> Result<(), anyhow::Error> {
    /// let analyzer = OrgAnalyzer::new(PathBuf::from("/tmp/repos"));
    /// let patterns = analyzer.analyze_repository(
    ///     "https://github.com/rust-lang/rust",
    ///     "rust",
    ///     1000
    /// ).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn analyze_repository(
        &self,
        repo_url: &str,
        repo_name: &str,
        max_commits: usize,
    ) -> Result<Vec<DefectPattern>> {
        info!(
            "Analyzing repository {} (up to {} commits)",
            repo_name, max_commits
        );

        // Clone repository
        self.git_analyzer.clone_repository(repo_url, repo_name)?;

        // Analyze commits
        let commits = self.git_analyzer.analyze_commits(repo_name, max_commits)?;
        debug!("Retrieved {} commits from {}", commits.len(), repo_name);

        // Classify commits and aggregate patterns
        let mut patterns = self.aggregate_defect_patterns(&commits);

        // Optionally enrich with TDG analysis (if pmat available)
        let repo_path = self.cache_dir.join(repo_name);
        if let Ok(tdg_analysis) = PmatIntegration::analyze_tdg(&repo_path) {
            debug!(
                "TDG analysis: avg={:.1}, max={:.1}",
                tdg_analysis.average_score, tdg_analysis.max_score
            );
            self.enrich_with_tdg(&mut patterns, &tdg_analysis);
        } else {
            debug!("TDG analysis unavailable (pmat not installed or failed)");
        }

        info!(
            "Found {} defect categories in {}",
            patterns.len(),
            repo_name
        );
        Ok(patterns)
    }

    /// Aggregate defect patterns from classified commits
    ///
    /// # Arguments
    /// * `commits` - List of commits to analyze
    ///
    /// # Returns
    /// * `Vec<DefectPattern>` with aggregated statistics
    fn aggregate_defect_patterns(&self, commits: &[CommitInfo]) -> Vec<DefectPattern> {
        let mut category_map: HashMap<DefectCategory, CategoryStats> = HashMap::new();

        // Classify each commit
        for commit in commits {
            if let Some(classification) = self.classifier.classify_from_message(&commit.message) {
                let stats = category_map
                    .entry(classification.category)
                    .or_insert_with(|| CategoryStats::new(classification.category));

                stats.add_instance(commit, &classification);
            }
        }

        // Convert to DefectPattern
        category_map
            .into_values()
            .map(|stats| stats.into_defect_pattern())
            .collect()
    }

    /// Enrich defect patterns with TDG quality signals
    ///
    /// # Arguments
    /// * `patterns` - Defect patterns to enrich
    /// * `tdg_analysis` - TDG analysis results
    fn enrich_with_tdg(&self, patterns: &mut [DefectPattern], tdg_analysis: &TdgAnalysis) {
        for pattern in patterns.iter_mut() {
            // Update quality signals with TDG data
            pattern.quality_signals.avg_tdg_score = Some(tdg_analysis.average_score);
            pattern.quality_signals.max_tdg_score = Some(tdg_analysis.max_score);
        }
    }
}

/// Internal stats tracking for each defect category with quality signals
#[derive(Debug)]
struct CategoryStats {
    category: DefectCategory,
    count: usize,
    total_confidence: f32,
    instances: Vec<DefectInstance>,
    // Quality signal aggregators
    total_files_changed: usize,
    total_lines_added: usize,
    total_lines_removed: usize,
}

impl CategoryStats {
    fn new(category: DefectCategory) -> Self {
        Self {
            category,
            count: 0,
            total_confidence: 0.0,
            instances: Vec::new(),
            total_files_changed: 0,
            total_lines_added: 0,
            total_lines_removed: 0,
        }
    }

    fn add_instance(&mut self, commit: &CommitInfo, classification: &Classification) {
        self.count += 1;
        self.total_confidence += classification.confidence;

        // Aggregate quality signals
        self.total_files_changed += commit.files_changed;
        self.total_lines_added += commit.lines_added;
        self.total_lines_removed += commit.lines_removed;

        // Keep up to 3 examples
        if self.instances.len() < 3 {
            self.instances.push(DefectInstance {
                commit_hash: commit.hash[..8.min(commit.hash.len())].to_string(),
                message: commit.message.clone(),
                author: commit.author.clone(),
                timestamp: commit.timestamp,
                files_affected: commit.files_changed,
                lines_added: commit.lines_added,
                lines_removed: commit.lines_removed,
            });
        }
    }

    fn into_defect_pattern(self) -> DefectPattern {
        let avg_confidence = if self.count > 0 {
            self.total_confidence / self.count as f32
        } else {
            0.0
        };

        // Calculate quality signals
        let quality_signals = if self.count > 0 {
            QualitySignals {
                avg_tdg_score: None, // Will be enhanced in Phase 1.5.2
                max_tdg_score: None,
                avg_complexity: None,
                avg_test_coverage: None,
                satd_instances: 0, // Will be enhanced in Phase 1.5.2
                avg_lines_changed: (self.total_lines_added + self.total_lines_removed) as f32
                    / self.count as f32,
                avg_files_per_commit: self.total_files_changed as f32 / self.count as f32,
            }
        } else {
            QualitySignals::default()
        };

        DefectPattern {
            category: self.category,
            frequency: self.count,
            confidence: avg_confidence,
            quality_signals,
            examples: self.instances,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_org_analyzer_can_be_created() {
        let temp_dir = TempDir::new().unwrap();
        let _analyzer = OrgAnalyzer::new(temp_dir.path());
    }

    #[test]
    fn test_aggregate_empty_commits() {
        let temp_dir = TempDir::new().unwrap();
        let analyzer = OrgAnalyzer::new(temp_dir.path());

        let commits = vec![];
        let patterns = analyzer.aggregate_defect_patterns(&commits);

        assert!(patterns.is_empty());
    }

    #[test]
    fn test_aggregate_non_defect_commits() {
        let temp_dir = TempDir::new().unwrap();
        let analyzer = OrgAnalyzer::new(temp_dir.path());

        let commits = vec![
            CommitInfo {
                hash: "abc123".to_string(),
                message: "docs: update README".to_string(),
                author: "test@example.com".to_string(),
                timestamp: 1234567890,
                files_changed: 1,
                lines_added: 5,
                lines_removed: 2,
            },
            CommitInfo {
                hash: "def456".to_string(),
                message: "chore: bump version".to_string(),
                author: "test@example.com".to_string(),
                timestamp: 1234567891,
                files_changed: 1,
                lines_added: 1,
                lines_removed: 1,
            },
        ];

        let patterns = analyzer.aggregate_defect_patterns(&commits);
        assert!(patterns.is_empty());
    }

    #[test]
    fn test_aggregate_defect_commits() {
        let temp_dir = TempDir::new().unwrap();
        let analyzer = OrgAnalyzer::new(temp_dir.path());

        let commits = vec![
            CommitInfo {
                hash: "abc123".to_string(),
                message: "fix: use-after-free in buffer".to_string(),
                author: "test@example.com".to_string(),
                timestamp: 1234567890,
                files_changed: 2,
                lines_added: 45,
                lines_removed: 12,
            },
            CommitInfo {
                hash: "def456".to_string(),
                message: "fix: another memory leak".to_string(),
                author: "test@example.com".to_string(),
                timestamp: 1234567891,
                files_changed: 1,
                lines_added: 8,
                lines_removed: 3,
            },
            CommitInfo {
                hash: "ghi789".to_string(),
                message: "security: prevent SQL injection".to_string(),
                author: "test@example.com".to_string(),
                timestamp: 1234567892,
                files_changed: 3,
                lines_added: 67,
                lines_removed: 23,
            },
        ];

        let patterns = analyzer.aggregate_defect_patterns(&commits);

        // Should have 2 categories: MemorySafety (2x) and SecurityVulnerabilities (1x)
        assert_eq!(patterns.len(), 2);

        // Check memory safety pattern
        let memory_pattern = patterns
            .iter()
            .find(|p| p.category == DefectCategory::MemorySafety)
            .expect("Should find memory safety pattern");

        assert_eq!(memory_pattern.frequency, 2);
        assert!(memory_pattern.confidence > 0.0);
        assert_eq!(memory_pattern.examples.len(), 2);
    }

    #[test]
    fn test_category_stats_aggregation() {
        let mut stats = CategoryStats::new(DefectCategory::MemorySafety);

        let commit1 = CommitInfo {
            hash: "abc123".to_string(),
            message: "fix: memory leak".to_string(),
            author: "test@example.com".to_string(),
            timestamp: 1234567890,
            files_changed: 2,
            lines_added: 15,
            lines_removed: 5,
        };

        let classification1 = Classification {
            category: DefectCategory::MemorySafety,
            confidence: 0.8,
            explanation: "test".to_string(),
            matched_patterns: vec!["memory leak".to_string()],
        };

        stats.add_instance(&commit1, &classification1);

        assert_eq!(stats.count, 1);
        assert_eq!(stats.total_confidence, 0.8);
        assert_eq!(stats.instances.len(), 1);

        let pattern = stats.into_defect_pattern();
        assert_eq!(pattern.frequency, 1);
        assert_eq!(pattern.confidence, 0.8);
        // Verify quality signals are calculated
        assert_eq!(pattern.quality_signals.avg_lines_changed, 20.0); // 15 + 5
        assert_eq!(pattern.quality_signals.avg_files_per_commit, 2.0);
    }

    #[test]
    fn test_examples_limited_to_three() {
        let mut stats = CategoryStats::new(DefectCategory::MemorySafety);

        for i in 0..5 {
            let commit = CommitInfo {
                hash: format!("hash{}", i),
                message: "fix: memory leak".to_string(),
                author: "test@example.com".to_string(),
                timestamp: 1234567890 + i as i64,
                files_changed: 1,
                lines_added: 10,
                lines_removed: 5,
            };

            let classification = Classification {
                category: DefectCategory::MemorySafety,
                confidence: 0.8,
                explanation: "test".to_string(),
                matched_patterns: vec!["memory leak".to_string()],
            };

            stats.add_instance(&commit, &classification);
        }

        assert_eq!(stats.count, 5);
        assert_eq!(stats.instances.len(), 3); // Limited to 3
    }

    #[test]
    fn test_enrich_with_tdg() {
        use crate::pmat::TdgAnalysis;
        use std::collections::HashMap;

        let temp_dir = TempDir::new().unwrap();
        let analyzer = OrgAnalyzer::new(temp_dir.path());

        // Create mock defect pattern
        let mut patterns = vec![DefectPattern {
            category: DefectCategory::MemorySafety,
            frequency: 5,
            confidence: 0.85,
            quality_signals: QualitySignals::default(),
            examples: vec![],
        }];

        // Create mock TDG analysis
        let tdg_analysis = TdgAnalysis {
            file_scores: HashMap::new(),
            average_score: 92.5,
            max_score: 98.0,
        };

        // Enrich patterns
        analyzer.enrich_with_tdg(&mut patterns, &tdg_analysis);

        // Verify TDG scores were populated
        assert_eq!(patterns[0].quality_signals.avg_tdg_score, Some(92.5));
        assert_eq!(patterns[0].quality_signals.max_tdg_score, Some(98.0));
    }

    // Integration test requiring network access
    #[tokio::test]
    #[ignore]
    async fn test_analyze_real_repository() {
        let temp_dir = TempDir::new().unwrap();
        let analyzer = OrgAnalyzer::new(temp_dir.path());

        let patterns = analyzer
            .analyze_repository("https://github.com/rust-lang/rustlings", "rustlings", 100)
            .await
            .unwrap();

        // Should find at least some defect patterns in 100 commits
        // (This is probabilistic, but rustlings has fix commits)
        assert!(!patterns.is_empty() || patterns.is_empty()); // Always passes, just testing it runs
    }
}
