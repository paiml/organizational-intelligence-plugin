// Tarantula-style Spectrum-Based Fault Localization (SBFL)
// Toyota Way: Start with simplest formula, evolve based on evidence
// Phase 1: Classic Tarantula + Ochiai + DStar formulas
// Muda: Avoid waste by using lightweight SBFL before expensive MBFL
// Muri: Prevent overburden by presenting only top-N suspicious statements

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tracing::{debug, info};

/// Represents a code location for fault localization
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct StatementId {
    pub file: PathBuf,
    pub line: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub column: Option<usize>,
}

impl StatementId {
    pub fn new(file: impl Into<PathBuf>, line: usize) -> Self {
        Self {
            file: file.into(),
            line,
            column: None,
        }
    }

    pub fn with_column(mut self, column: usize) -> Self {
        self.column = Some(column);
        self
    }
}

/// Coverage information for a single statement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatementCoverage {
    pub id: StatementId,
    pub executed_by_passed: usize,
    pub executed_by_failed: usize,
}

impl StatementCoverage {
    pub fn new(id: StatementId, passed: usize, failed: usize) -> Self {
        Self {
            id,
            executed_by_passed: passed,
            executed_by_failed: failed,
        }
    }
}

/// Available fault localization formulas
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SbflFormula {
    /// Original Tarantula formula (Jones & Harrold, 2005)
    #[default]
    Tarantula,
    /// Ochiai formula - often outperforms Tarantula (Abreu et al., 2009)
    Ochiai,
    /// DStar with configurable exponent (Wong et al., 2014)
    DStar { exponent: u32 },
}

/// Individual suspiciousness ranking entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuspiciousnessRanking {
    pub rank: usize,
    pub statement: StatementId,
    pub suspiciousness: f32,
    pub scores: HashMap<String, f32>,
    pub explanation: String,
    pub failed_coverage: usize,
    pub passed_coverage: usize,
}

/// Result of fault localization analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaultLocalizationResult {
    pub rankings: Vec<SuspiciousnessRanking>,
    pub formula_used: SbflFormula,
    pub confidence: f32,
    pub total_passed_tests: usize,
    pub total_failed_tests: usize,
}

/// Classic Tarantula suspiciousness formula
///
/// Formula: (failed/totalFailed) / ((passed/totalPassed) + (failed/totalFailed))
///
/// Reference: Jones, J.A., Harrold, M.J. (2005). ASE '05
pub fn tarantula(failed: usize, passed: usize, total_failed: usize, total_passed: usize) -> f32 {
    let failed_ratio = if total_failed > 0 {
        failed as f32 / total_failed as f32
    } else {
        0.0
    };

    let passed_ratio = if total_passed > 0 {
        passed as f32 / total_passed as f32
    } else {
        0.0
    };

    let denominator = passed_ratio + failed_ratio;
    if denominator == 0.0 {
        0.0
    } else {
        failed_ratio / denominator
    }
}

/// Ochiai suspiciousness formula (from molecular biology)
///
/// Formula: failed / sqrt(totalFailed * (failed + passed))
///
/// Reference: Abreu et al. (2009). JSS 82(11)
pub fn ochiai(failed: usize, passed: usize, total_failed: usize) -> f32 {
    let denominator = ((total_failed * (failed + passed)) as f32).sqrt();
    if denominator == 0.0 {
        0.0
    } else {
        failed as f32 / denominator
    }
}

/// DStar suspiciousness formula with configurable exponent
///
/// Formula: failed^* / (passed + (totalFailed - failed))
///
/// Reference: Wong et al. (2014). IEEE TSE 40(1)
pub fn dstar(failed: usize, passed: usize, total_failed: usize, star: u32) -> f32 {
    let numerator = (failed as f32).powi(star as i32);
    let not_failed = total_failed.saturating_sub(failed);
    let denominator = passed as f32 + not_failed as f32;

    if denominator == 0.0 {
        if numerator > 0.0 {
            f32::MAX // Avoid infinity, use max finite value
        } else {
            0.0
        }
    } else {
        numerator / denominator
    }
}

/// Spectrum-Based Fault Localizer
///
/// Implements the core SBFL algorithms following Toyota Way principles:
/// - Start simple (Tarantula baseline)
/// - Measure and evolve (compare formulas)
/// - Eliminate waste (skip expensive analysis when simple works)
pub struct SbflLocalizer {
    formula: SbflFormula,
    top_n: usize,
    include_explanations: bool,
    min_confidence_threshold: f32,
}

impl Default for SbflLocalizer {
    fn default() -> Self {
        Self::new()
    }
}

impl SbflLocalizer {
    pub fn new() -> Self {
        Self {
            formula: SbflFormula::Tarantula,
            top_n: 10,
            include_explanations: true,
            min_confidence_threshold: 0.0,
        }
    }

    pub fn with_formula(mut self, formula: SbflFormula) -> Self {
        self.formula = formula;
        self
    }

    pub fn with_top_n(mut self, n: usize) -> Self {
        self.top_n = n;
        self
    }

    pub fn with_explanations(mut self, include: bool) -> Self {
        self.include_explanations = include;
        self
    }

    pub fn with_min_confidence(mut self, threshold: f32) -> Self {
        self.min_confidence_threshold = threshold;
        self
    }

    /// Localize faults using the configured SBFL formula
    ///
    /// # Arguments
    /// * `coverage` - Statement coverage data
    /// * `total_passed` - Total number of passing tests
    /// * `total_failed` - Total number of failing tests
    ///
    /// # Returns
    /// Ranked list of suspicious statements
    pub fn localize(
        &self,
        coverage: &[StatementCoverage],
        total_passed: usize,
        total_failed: usize,
    ) -> FaultLocalizationResult {
        info!(
            "Running {:?} fault localization on {} statements",
            self.formula,
            coverage.len()
        );

        // Calculate suspiciousness for each statement
        let mut scored: Vec<(StatementId, f32, usize, usize)> = coverage
            .iter()
            .map(|cov| {
                let score = self.calculate_score(
                    cov.executed_by_failed,
                    cov.executed_by_passed,
                    total_failed,
                    total_passed,
                );
                (
                    cov.id.clone(),
                    score,
                    cov.executed_by_failed,
                    cov.executed_by_passed,
                )
            })
            .collect();

        // Sort by suspiciousness (descending)
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top N
        let rankings: Vec<SuspiciousnessRanking> = scored
            .into_iter()
            .take(self.top_n)
            .enumerate()
            .filter(|(_, (_, score, _, _))| *score >= self.min_confidence_threshold)
            .map(|(rank, (stmt, score, failed, passed))| {
                let explanation = if self.include_explanations {
                    self.generate_explanation(failed, passed, total_failed, total_passed, score)
                } else {
                    String::new()
                };

                // Calculate all formula scores for comparison
                let mut scores = HashMap::new();
                scores.insert(
                    "tarantula".to_string(),
                    tarantula(failed, passed, total_failed, total_passed),
                );
                scores.insert("ochiai".to_string(), ochiai(failed, passed, total_failed));
                scores.insert("dstar2".to_string(), dstar(failed, passed, total_failed, 2));
                scores.insert("dstar3".to_string(), dstar(failed, passed, total_failed, 3));

                SuspiciousnessRanking {
                    rank: rank + 1,
                    statement: stmt,
                    suspiciousness: score,
                    scores,
                    explanation,
                    failed_coverage: failed,
                    passed_coverage: passed,
                }
            })
            .collect();

        // Calculate confidence based on test coverage density
        let confidence = self.calculate_confidence(coverage.len(), total_passed, total_failed);

        debug!(
            "Localized {} suspicious statements with confidence {}",
            rankings.len(),
            confidence
        );

        FaultLocalizationResult {
            rankings,
            formula_used: self.formula,
            confidence,
            total_passed_tests: total_passed,
            total_failed_tests: total_failed,
        }
    }

    fn calculate_score(
        &self,
        failed: usize,
        passed: usize,
        total_failed: usize,
        total_passed: usize,
    ) -> f32 {
        match self.formula {
            SbflFormula::Tarantula => tarantula(failed, passed, total_failed, total_passed),
            SbflFormula::Ochiai => ochiai(failed, passed, total_failed),
            SbflFormula::DStar { exponent } => dstar(failed, passed, total_failed, exponent),
        }
    }

    fn generate_explanation(
        &self,
        failed: usize,
        passed: usize,
        total_failed: usize,
        total_passed: usize,
        score: f32,
    ) -> String {
        let failed_pct = if total_failed > 0 {
            (failed as f32 / total_failed as f32 * 100.0) as u32
        } else {
            0
        };

        let passed_pct = if total_passed > 0 {
            (passed as f32 / total_passed as f32 * 100.0) as u32
        } else {
            0
        };

        format!(
            "Executed by {}% of failing tests ({}/{}) and {}% of passing tests ({}/{}). \
             Suspiciousness score: {:.3}",
            failed_pct, failed, total_failed, passed_pct, passed, total_passed, score
        )
    }

    fn calculate_confidence(
        &self,
        statement_count: usize,
        total_passed: usize,
        total_failed: usize,
    ) -> f32 {
        // Confidence based on:
        // 1. Number of failing tests (more = more signal)
        // 2. Ratio of failing to total tests (too few failing = noisy)
        // 3. Coverage density (more statements covered = more context)

        let total_tests = total_passed + total_failed;
        if total_tests == 0 || total_failed == 0 {
            return 0.0;
        }

        // Factor 1: Log scale for failing test count (diminishing returns)
        let fail_factor = (total_failed as f32).ln().min(3.0) / 3.0;

        // Factor 2: Failing ratio (sweet spot around 5-20%)
        let fail_ratio = total_failed as f32 / total_tests as f32;
        let ratio_factor = if fail_ratio < 0.01 {
            fail_ratio * 10.0 // Very few failures = low confidence
        } else if fail_ratio > 0.5 {
            1.0 - (fail_ratio - 0.5) // Too many failures = less localizing
        } else {
            1.0
        };

        // Factor 3: Statement coverage (more covered = more context)
        let coverage_factor = (statement_count as f32).ln().min(7.0) / 7.0;

        (fail_factor * ratio_factor * coverage_factor).min(1.0)
    }
}

/// LCOV coverage data parser for cargo-llvm-cov integration
#[derive(Debug, Default)]
pub struct LcovParser;

impl LcovParser {
    /// Parse LCOV format coverage file
    ///
    /// LCOV format:
    /// ```text
    /// SF:path/to/file.rs
    /// DA:line_number,execution_count
    /// ...
    /// end_of_record
    /// ```
    pub fn parse_file<P: AsRef<Path>>(path: P) -> Result<Vec<(StatementId, usize)>> {
        let content = std::fs::read_to_string(path.as_ref())
            .map_err(|e| anyhow!("Failed to read LCOV file: {}", e))?;
        Self::parse(&content)
    }

    pub fn parse(content: &str) -> Result<Vec<(StatementId, usize)>> {
        let mut results = Vec::new();
        let mut current_file: Option<PathBuf> = None;

        for line in content.lines() {
            let line = line.trim();

            if let Some(path) = line.strip_prefix("SF:") {
                current_file = Some(PathBuf::from(path));
            } else if let Some(da) = line.strip_prefix("DA:") {
                if let Some(ref file) = current_file {
                    let parts: Vec<&str> = da.split(',').collect();
                    if parts.len() >= 2 {
                        if let (Ok(line_num), Ok(count)) =
                            (parts[0].parse::<usize>(), parts[1].parse::<usize>())
                        {
                            results.push((StatementId::new(file.clone(), line_num), count));
                        }
                    }
                }
            } else if line == "end_of_record" {
                current_file = None;
            }
        }

        Ok(results)
    }

    /// Combine coverage from multiple test runs (passed and failed)
    ///
    /// # Arguments
    /// * `passed_coverage` - Coverage from passing tests
    /// * `failed_coverage` - Coverage from failing tests
    ///
    /// # Returns
    /// Combined statement coverage suitable for SBFL
    pub fn combine_coverage(
        passed_coverage: &[(StatementId, usize)],
        failed_coverage: &[(StatementId, usize)],
    ) -> Vec<StatementCoverage> {
        let mut coverage_map: HashMap<StatementId, (usize, usize)> = HashMap::new();

        // Count passed test coverage
        for (stmt, count) in passed_coverage {
            if *count > 0 {
                coverage_map.entry(stmt.clone()).or_insert((0, 0)).0 += 1;
            }
        }

        // Count failed test coverage
        for (stmt, count) in failed_coverage {
            if *count > 0 {
                coverage_map.entry(stmt.clone()).or_insert((0, 0)).1 += 1;
            }
        }

        coverage_map
            .into_iter()
            .map(|(id, (passed, failed))| StatementCoverage::new(id, passed, failed))
            .collect()
    }
}

// ============================================================================
// TarantulaIntegration - pmat-style integration for fault localization
// Toyota Way: Integrate with existing tools (cargo-llvm-cov, pmat TDG)
// ============================================================================

/// Report output format for fault localization results
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ReportFormat {
    #[default]
    Yaml,
    Json,
    Terminal,
}

/// Configuration for fault localization runs
#[derive(Debug, Clone)]
pub struct LocalizationConfig {
    pub formula: SbflFormula,
    pub top_n: usize,
    pub include_explanations: bool,
    pub min_confidence: f32,
}

impl Default for LocalizationConfig {
    fn default() -> Self {
        Self {
            formula: SbflFormula::Tarantula,
            top_n: 10,
            include_explanations: true,
            min_confidence: 0.0,
        }
    }
}

impl LocalizationConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_formula(mut self, formula: SbflFormula) -> Self {
        self.formula = formula;
        self
    }

    pub fn with_top_n(mut self, n: usize) -> Self {
        self.top_n = n;
        self
    }

    pub fn with_explanations(mut self, include: bool) -> Self {
        self.include_explanations = include;
        self
    }

    pub fn with_min_confidence(mut self, threshold: f32) -> Self {
        self.min_confidence = threshold;
        self
    }
}

/// Tarantula integration wrapper (pmat-style)
///
/// Provides high-level interface for fault localization that integrates
/// with cargo-llvm-cov for coverage and pmat for TDG enrichment.
///
/// # Toyota Way Principles
/// - **Genchi Genbutsu**: Uses actual coverage data, not estimates
/// - **Muda**: Avoids waste by reusing existing coverage tools
/// - **Jidoka**: Provides human-readable explanations
pub struct TarantulaIntegration;

impl TarantulaIntegration {
    /// Check if cargo-llvm-cov is available
    pub fn is_coverage_tool_available() -> bool {
        std::process::Command::new("cargo")
            .args(["llvm-cov", "--version"])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    /// Parse LCOV format output from cargo-llvm-cov
    pub fn parse_lcov_output(content: &str) -> Result<Vec<(StatementId, usize)>> {
        LcovParser::parse(content)
    }

    /// Run fault localization on coverage data
    ///
    /// # Arguments
    /// * `passed_coverage` - Coverage from passing tests
    /// * `failed_coverage` - Coverage from failing tests
    /// * `total_passed` - Number of passing tests
    /// * `total_failed` - Number of failing tests
    /// * `config` - Localization configuration
    pub fn run_localization(
        passed_coverage: &[(StatementId, usize)],
        failed_coverage: &[(StatementId, usize)],
        total_passed: usize,
        total_failed: usize,
        config: &LocalizationConfig,
    ) -> FaultLocalizationResult {
        info!(
            "Running fault localization: {} passed, {} failed tests",
            total_passed, total_failed
        );

        // Combine coverage data
        let combined = LcovParser::combine_coverage(passed_coverage, failed_coverage);

        // Run SBFL localization
        let localizer = SbflLocalizer::new()
            .with_formula(config.formula)
            .with_top_n(config.top_n)
            .with_explanations(config.include_explanations)
            .with_min_confidence(config.min_confidence);

        localizer.localize(&combined, total_passed, total_failed)
    }

    /// Generate report in specified format
    pub fn generate_report(
        result: &FaultLocalizationResult,
        format: ReportFormat,
    ) -> Result<String> {
        match format {
            ReportFormat::Yaml => {
                serde_yaml::to_string(result).map_err(|e| anyhow!("Failed to generate YAML: {}", e))
            }
            ReportFormat::Json => serde_json::to_string_pretty(result)
                .map_err(|e| anyhow!("Failed to generate JSON: {}", e)),
            ReportFormat::Terminal => Ok(Self::format_terminal_report(result)),
        }
    }

    /// Format report for terminal output
    fn format_terminal_report(result: &FaultLocalizationResult) -> String {
        let mut output = String::new();

        output.push_str("╔══════════════════════════════════════════════════════════════╗\n");
        output.push_str(&format!(
            "║           FAULT LOCALIZATION REPORT - {:?}              ║\n",
            result.formula_used
        ));
        output.push_str("╠══════════════════════════════════════════════════════════════╣\n");
        output.push_str(&format!(
            "║ Tests: {} passed, {} failed                              ║\n",
            result.total_passed_tests, result.total_failed_tests
        ));
        output.push_str(&format!(
            "║ Confidence: {:.2}                                          ║\n",
            result.confidence
        ));
        output.push_str("╠══════════════════════════════════════════════════════════════╣\n");
        output.push_str("║  TOP SUSPICIOUS STATEMENTS                                   ║\n");
        output.push_str("╠══════════════════════════════════════════════════════════════╣\n");

        for ranking in &result.rankings {
            let bar_len = (ranking.suspiciousness * 20.0) as usize;
            let bar: String = "█".repeat(bar_len) + &"░".repeat(20 - bar_len);

            output.push_str(&format!(
                "║  #{:<2} {}:{:<6}  {} {:.2}   ║\n",
                ranking.rank,
                ranking.statement.file.display(),
                ranking.statement.line,
                bar,
                ranking.suspiciousness
            ));
        }

        output.push_str("╚══════════════════════════════════════════════════════════════╝\n");

        output
    }

    /// Enrich fault localization results with TDG scores from pmat
    ///
    /// # Arguments
    /// * `result` - Mutable reference to localization result
    /// * `tdg_scores` - Map of file path to TDG score
    pub fn enrich_with_tdg(
        result: &mut FaultLocalizationResult,
        tdg_scores: &HashMap<String, f32>,
    ) {
        for ranking in &mut result.rankings {
            let file_path = ranking.statement.file.to_string_lossy().to_string();
            if let Some(&tdg) = tdg_scores.get(&file_path) {
                ranking.scores.insert("tdg".to_string(), tdg);
            }
        }
    }

    /// Run cargo-llvm-cov and collect coverage for a test run
    ///
    /// # Arguments
    /// * `repo_path` - Path to repository
    /// * `test_filter` - Optional test filter pattern
    ///
    /// # Returns
    /// LCOV format coverage data as string
    #[allow(dead_code)]
    pub fn collect_coverage<P: AsRef<Path>>(
        repo_path: P,
        test_filter: Option<&str>,
    ) -> Result<String> {
        let mut cmd = std::process::Command::new("cargo");
        cmd.current_dir(repo_path.as_ref())
            .args(["llvm-cov", "--lcov"]);

        if let Some(filter) = test_filter {
            cmd.args(["--", filter]);
        }

        let output = cmd
            .output()
            .map_err(|e| anyhow!("Failed to run cargo-llvm-cov: {}", e))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow!("cargo-llvm-cov failed: {}", stderr));
        }

        String::from_utf8(output.stdout)
            .map_err(|e| anyhow!("Invalid UTF-8 in coverage output: {}", e))
    }
}

// ============================================================================
// SZZ Algorithm - Bug-Introducing Commit Identification
// Toyota Way: Genchi Genbutsu - trace back to root cause using git history
// Reference: Śliwerski et al. (2005). "When do changes induce fixes?" MSR '05
// ============================================================================

/// Confidence level for SZZ tracing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SzzConfidence {
    /// Direct line trace via git blame
    High,
    /// Refactoring-aware trace (excluded cosmetic changes)
    Medium,
    /// Heuristic fallback (commit message patterns)
    Low,
}

/// Result of SZZ algorithm tracing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SzzResult {
    /// The commit that fixed the bug
    pub bug_fixing_commit: String,
    /// Commits that likely introduced the bug
    pub bug_introducing_commits: Vec<String>,
    /// Lines identified as faulty (file, line)
    pub faulty_lines: Vec<(String, usize)>,
    /// Confidence in the trace
    pub confidence: SzzConfidence,
    /// Commit message of the fix (for context)
    pub fix_message: String,
}

/// SZZ Algorithm implementation for bug-introducing commit identification
///
/// # Algorithm Steps
/// 1. Identify bug-fixing commits (from commit messages or issue links)
/// 2. Find lines modified in the fix
/// 3. Use git blame to trace back to introducing commits
/// 4. Filter out cosmetic changes (refactoring-aware)
pub struct SzzAnalyzer;

impl SzzAnalyzer {
    /// Identify bug-fixing commits from commit messages
    ///
    /// Looks for patterns like:
    /// - "fix:", "fixes:", "fixed:"
    /// - "bug:", "bugfix:"
    /// - Issue references: "#123", "JIRA-456"
    pub fn identify_bug_fixes(commits: &[(String, String)]) -> Vec<(String, String)> {
        let fix_patterns = [
            "fix:",
            "fixes:",
            "fixed:",
            "fix(",
            "bug:",
            "bugfix:",
            "hotfix:",
            "resolve:",
            "resolves:",
            "resolved:",
            "close:",
            "closes:",
            "closed:",
        ];

        commits
            .iter()
            .filter(|(_, msg)| {
                let lower = msg.to_lowercase();
                fix_patterns.iter().any(|p| lower.contains(p))
                    || lower.contains("#") && lower.chars().any(|c| c.is_ascii_digit())
            })
            .cloned()
            .collect()
    }

    /// Trace bug-introducing commits using simplified SZZ
    ///
    /// # Arguments
    /// * `fix_commit` - The bug-fixing commit hash
    /// * `fix_message` - Commit message
    /// * `changed_lines` - Lines changed in the fix (file, line, was_deleted)
    /// * `blame_data` - Git blame output (line -> (commit, author))
    pub fn trace_introducing_commits(
        fix_commit: &str,
        fix_message: &str,
        changed_lines: &[(String, usize, bool)],
        blame_data: &HashMap<(String, usize), (String, String)>,
    ) -> SzzResult {
        let mut introducing_commits: Vec<String> = Vec::new();
        let mut faulty_lines: Vec<(String, usize)> = Vec::new();

        // For each deleted/modified line in the fix, trace back
        for (file, line, was_deleted) in changed_lines {
            if *was_deleted {
                // Deleted lines are the key - they likely contained the bug
                if let Some((commit, _author)) = blame_data.get(&(file.clone(), *line)) {
                    if commit != fix_commit && !introducing_commits.contains(commit) {
                        introducing_commits.push(commit.clone());
                    }
                    faulty_lines.push((file.clone(), *line));
                }
            }
        }

        // Determine confidence
        let confidence = if !introducing_commits.is_empty() {
            SzzConfidence::High
        } else if !faulty_lines.is_empty() {
            SzzConfidence::Medium
        } else {
            SzzConfidence::Low
        };

        SzzResult {
            bug_fixing_commit: fix_commit.to_string(),
            bug_introducing_commits: introducing_commits,
            faulty_lines,
            confidence,
            fix_message: fix_message.to_string(),
        }
    }

    /// Filter out cosmetic changes (refactoring-aware SZZ)
    ///
    /// Excludes:
    /// - Whitespace-only changes
    /// - Comment-only changes
    /// - Import reordering
    pub fn filter_cosmetic_changes(
        changes: &[(String, usize, bool)],
        file_contents: &HashMap<String, Vec<String>>,
    ) -> Vec<(String, usize, bool)> {
        changes
            .iter()
            .filter(|(file, line, _)| {
                if let Some(lines) = file_contents.get(file) {
                    if let Some(content) = lines.get(line.saturating_sub(1)) {
                        let trimmed = content.trim();
                        // Keep if not cosmetic
                        !trimmed.is_empty()
                            && !trimmed.starts_with("//")
                            && !trimmed.starts_with("/*")
                            && !trimmed.starts_with("*")
                            && !trimmed.starts_with("use ")
                            && !trimmed.starts_with("import ")
                    } else {
                        true
                    }
                } else {
                    true
                }
            })
            .cloned()
            .collect()
    }

    /// Calculate suspiciousness for files based on SZZ results
    ///
    /// Combines SZZ bug-introduction data with historical defect frequency
    pub fn calculate_file_suspiciousness(szz_results: &[SzzResult]) -> HashMap<String, f32> {
        let mut file_bug_count: HashMap<String, usize> = HashMap::new();
        let total_bugs = szz_results.len();

        for result in szz_results {
            for (file, _line) in &result.faulty_lines {
                *file_bug_count.entry(file.clone()).or_insert(0) += 1;
            }
        }

        file_bug_count
            .into_iter()
            .map(|(file, count)| {
                let suspiciousness = if total_bugs > 0 {
                    count as f32 / total_bugs as f32
                } else {
                    0.0
                };
                (file, suspiciousness)
            })
            .collect()
    }
}

/// Combines SBFL with git history for enhanced fault localization
pub struct HybridFaultLocalizer;

impl HybridFaultLocalizer {
    /// Combine SBFL suspiciousness with SZZ historical data
    ///
    /// Formula: combined = α * sbfl_score + (1-α) * historical_score
    /// Where α is the weighting factor (default 0.7 for SBFL)
    pub fn combine_scores(
        sbfl_result: &FaultLocalizationResult,
        historical_suspiciousness: &HashMap<String, f32>,
        alpha: f32,
    ) -> FaultLocalizationResult {
        let mut combined_rankings: Vec<SuspiciousnessRanking> = sbfl_result
            .rankings
            .iter()
            .map(|r| {
                let file_path = r.statement.file.to_string_lossy().to_string();
                let historical = historical_suspiciousness
                    .get(&file_path)
                    .copied()
                    .unwrap_or(0.0);
                let combined = alpha * r.suspiciousness + (1.0 - alpha) * historical;

                let mut scores = r.scores.clone();
                scores.insert("historical".to_string(), historical);
                scores.insert("combined".to_string(), combined);

                SuspiciousnessRanking {
                    rank: 0, // Will be re-ranked
                    statement: r.statement.clone(),
                    suspiciousness: combined,
                    scores,
                    explanation: format!(
                        "{} Historical suspiciousness: {:.2}",
                        r.explanation, historical
                    ),
                    failed_coverage: r.failed_coverage,
                    passed_coverage: r.passed_coverage,
                }
            })
            .collect();

        // Re-rank by combined score
        combined_rankings.sort_by(|a, b| {
            b.suspiciousness
                .partial_cmp(&a.suspiciousness)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for (i, ranking) in combined_rankings.iter_mut().enumerate() {
            ranking.rank = i + 1;
        }

        FaultLocalizationResult {
            rankings: combined_rankings,
            formula_used: sbfl_result.formula_used,
            confidence: sbfl_result.confidence,
            total_passed_tests: sbfl_result.total_passed_tests,
            total_failed_tests: sbfl_result.total_failed_tests,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ============== Formula Unit Tests ==============

    #[test]
    fn test_tarantula_perfect_fault() {
        // Statement executed by all failing tests, no passing tests
        // Should have maximum suspiciousness
        let score = tarantula(10, 0, 10, 100);
        assert!((score - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_tarantula_perfect_clean() {
        // Statement executed by all passing tests, no failing tests
        // Should have minimum suspiciousness
        let score = tarantula(0, 100, 10, 100);
        assert!(score.abs() < 0.001);
    }

    #[test]
    fn test_tarantula_mixed() {
        // Statement executed by 50% of failing and 50% of passing
        let score = tarantula(5, 50, 10, 100);
        assert!(score > 0.0 && score < 1.0);
    }

    #[test]
    fn test_tarantula_no_tests() {
        // Edge case: no tests
        let score = tarantula(0, 0, 0, 0);
        assert!(score.abs() < 0.001);
    }

    #[test]
    fn test_ochiai_perfect_fault() {
        let score = ochiai(10, 0, 10);
        assert!((score - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_ochiai_no_execution() {
        let score = ochiai(0, 0, 10);
        assert!(score.abs() < 0.001);
    }

    #[test]
    fn test_ochiai_mixed() {
        let score = ochiai(5, 50, 10);
        assert!(score > 0.0 && score < 1.0);
    }

    #[test]
    fn test_dstar_perfect_fault() {
        let score = dstar(10, 0, 10, 2);
        // With star=2: 10^2 / (0 + 0) = infinity, but we cap at MAX
        assert!(score > 100.0);
    }

    #[test]
    fn test_dstar_mixed() {
        let score = dstar(5, 50, 10, 2);
        // 25 / (50 + 5) = 0.4545...
        assert!((score - 0.4545).abs() < 0.01);
    }

    #[test]
    fn test_dstar_exponent_effect() {
        let score2 = dstar(5, 10, 10, 2);
        let score3 = dstar(5, 10, 10, 3);
        // Higher exponent amplifies the signal
        assert!(score3 > score2);
    }

    // ============== Localizer Tests ==============

    #[test]
    fn test_localizer_basic() {
        let localizer = SbflLocalizer::new();

        let coverage = vec![
            StatementCoverage::new(StatementId::new("file.rs", 10), 0, 10), // All failing
            StatementCoverage::new(StatementId::new("file.rs", 20), 100, 0), // All passing
            StatementCoverage::new(StatementId::new("file.rs", 30), 50, 5), // Mixed
        ];

        let result = localizer.localize(&coverage, 100, 10);

        assert_eq!(result.rankings.len(), 3);
        assert_eq!(result.rankings[0].statement.line, 10); // Most suspicious first
        assert!(result.rankings[0].suspiciousness > result.rankings[1].suspiciousness);
    }

    #[test]
    fn test_localizer_top_n() {
        let localizer = SbflLocalizer::new().with_top_n(2);

        let coverage = vec![
            StatementCoverage::new(StatementId::new("file.rs", 10), 0, 10),
            StatementCoverage::new(StatementId::new("file.rs", 20), 50, 5),
            StatementCoverage::new(StatementId::new("file.rs", 30), 100, 0),
        ];

        let result = localizer.localize(&coverage, 100, 10);

        assert_eq!(result.rankings.len(), 2);
    }

    #[test]
    fn test_localizer_formula_selection() {
        let coverage = vec![StatementCoverage::new(
            StatementId::new("file.rs", 10),
            50,
            5,
        )];

        let tarantula_result = SbflLocalizer::new()
            .with_formula(SbflFormula::Tarantula)
            .localize(&coverage, 100, 10);

        let ochiai_result = SbflLocalizer::new()
            .with_formula(SbflFormula::Ochiai)
            .localize(&coverage, 100, 10);

        // Scores should differ between formulas
        assert_ne!(
            tarantula_result.rankings[0].suspiciousness,
            ochiai_result.rankings[0].suspiciousness
        );
    }

    #[test]
    fn test_localizer_includes_all_scores() {
        let localizer = SbflLocalizer::new();

        let coverage = vec![StatementCoverage::new(
            StatementId::new("file.rs", 10),
            50,
            5,
        )];

        let result = localizer.localize(&coverage, 100, 10);

        let scores = &result.rankings[0].scores;
        assert!(scores.contains_key("tarantula"));
        assert!(scores.contains_key("ochiai"));
        assert!(scores.contains_key("dstar2"));
        assert!(scores.contains_key("dstar3"));
    }

    #[test]
    fn test_localizer_explanation() {
        let localizer = SbflLocalizer::new().with_explanations(true);

        let coverage = vec![StatementCoverage::new(
            StatementId::new("file.rs", 10),
            10,
            5,
        )];

        let result = localizer.localize(&coverage, 100, 10);

        assert!(!result.rankings[0].explanation.is_empty());
        assert!(result.rankings[0].explanation.contains("50%")); // 5/10 = 50%
    }

    #[test]
    fn test_localizer_no_explanation() {
        let localizer = SbflLocalizer::new().with_explanations(false);

        let coverage = vec![StatementCoverage::new(
            StatementId::new("file.rs", 10),
            10,
            5,
        )];

        let result = localizer.localize(&coverage, 100, 10);

        assert!(result.rankings[0].explanation.is_empty());
    }

    #[test]
    fn test_localizer_confidence() {
        let localizer = SbflLocalizer::new();

        // More failing tests = higher confidence
        // Need multiple statements for coverage_factor to be non-zero
        let coverage: Vec<StatementCoverage> = (1..=100)
            .map(|i| StatementCoverage::new(StatementId::new("file.rs", i), 90, 10))
            .collect();

        let result_many_fail = localizer.localize(&coverage, 90, 10);
        let result_few_fail = localizer.localize(&coverage, 99, 1);

        assert!(result_many_fail.confidence > result_few_fail.confidence);
    }

    // ============== LCOV Parser Tests ==============

    #[test]
    fn test_lcov_parse_basic() {
        let lcov = r#"
SF:src/main.rs
DA:10,5
DA:20,0
DA:30,12
end_of_record
"#;

        let results = LcovParser::parse(lcov).unwrap();

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].0.line, 10);
        assert_eq!(results[0].1, 5);
    }

    #[test]
    fn test_lcov_parse_multiple_files() {
        let lcov = r#"
SF:src/a.rs
DA:10,5
end_of_record
SF:src/b.rs
DA:20,10
end_of_record
"#;

        let results = LcovParser::parse(lcov).unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0.file, PathBuf::from("src/a.rs"));
        assert_eq!(results[1].0.file, PathBuf::from("src/b.rs"));
    }

    #[test]
    fn test_lcov_combine_coverage() {
        let passed = vec![
            (StatementId::new("file.rs", 10), 5),
            (StatementId::new("file.rs", 20), 10),
        ];

        let failed = vec![
            (StatementId::new("file.rs", 10), 3),
            (StatementId::new("file.rs", 30), 1),
        ];

        let combined = LcovParser::combine_coverage(&passed, &failed);

        assert_eq!(combined.len(), 3);

        // Find the statement at line 10 (covered by both)
        let stmt_10 = combined.iter().find(|c| c.id.line == 10).unwrap();
        assert_eq!(stmt_10.executed_by_passed, 1); // At least one passed test
        assert_eq!(stmt_10.executed_by_failed, 1); // At least one failed test
    }

    // ============== Integration Tests ==============

    #[test]
    fn test_end_to_end_localization() {
        // Simulate a scenario where line 50 is the fault
        let coverage = vec![
            // Line 50: Executed by all failing tests, few passing
            StatementCoverage::new(StatementId::new("buggy.rs", 50), 5, 10),
            // Line 60: Common code - executed by all
            StatementCoverage::new(StatementId::new("buggy.rs", 60), 95, 10),
            // Line 70: Only passing tests
            StatementCoverage::new(StatementId::new("buggy.rs", 70), 90, 0),
        ];

        let result = SbflLocalizer::new()
            .with_formula(SbflFormula::Tarantula)
            .localize(&coverage, 100, 10);

        // Line 50 should be ranked first (most suspicious)
        assert_eq!(result.rankings[0].statement.line, 50);
    }

    #[test]
    fn test_statement_id_equality() {
        let id1 = StatementId::new("file.rs", 10);
        let id2 = StatementId::new("file.rs", 10);
        let id3 = StatementId::new("file.rs", 20);

        assert_eq!(id1, id2);
        assert_ne!(id1, id3);
    }

    #[test]
    fn test_statement_id_with_column() {
        let id = StatementId::new("file.rs", 10).with_column(5);

        assert_eq!(id.column, Some(5));
    }

    #[test]
    fn test_formula_default() {
        let formula = SbflFormula::default();
        assert_eq!(formula, SbflFormula::Tarantula);
    }

    #[test]
    fn test_localizer_default() {
        let localizer = SbflLocalizer::default();
        let coverage = vec![StatementCoverage::new(
            StatementId::new("file.rs", 10),
            50,
            5,
        )];

        let result = localizer.localize(&coverage, 100, 10);
        assert_eq!(result.formula_used, SbflFormula::Tarantula);
    }

    #[test]
    fn test_confidence_edge_cases() {
        let localizer = SbflLocalizer::new();

        // No tests
        let result = localizer.localize(&[], 0, 0);
        assert_eq!(result.confidence, 0.0);

        // No failing tests
        let coverage = vec![StatementCoverage::new(
            StatementId::new("file.rs", 10),
            100,
            0,
        )];
        let result = localizer.localize(&coverage, 100, 0);
        assert_eq!(result.confidence, 0.0);
    }

    #[test]
    fn test_min_confidence_threshold() {
        let localizer = SbflLocalizer::new().with_min_confidence(0.5);

        let coverage = vec![
            StatementCoverage::new(StatementId::new("file.rs", 10), 0, 10), // High score
            StatementCoverage::new(StatementId::new("file.rs", 20), 100, 1), // Low score
        ];

        let result = localizer.localize(&coverage, 100, 10);

        // Only high-score statement should be included
        assert!(result.rankings.iter().all(|r| r.suspiciousness >= 0.5));
    }

    #[test]
    fn test_serialization() {
        let result = FaultLocalizationResult {
            rankings: vec![SuspiciousnessRanking {
                rank: 1,
                statement: StatementId::new("file.rs", 10),
                suspiciousness: 0.95,
                scores: HashMap::new(),
                explanation: "Test".to_string(),
                failed_coverage: 10,
                passed_coverage: 5,
            }],
            formula_used: SbflFormula::Tarantula,
            confidence: 0.8,
            total_passed_tests: 100,
            total_failed_tests: 10,
        };

        let json = serde_json::to_string(&result).unwrap();
        let deserialized: FaultLocalizationResult = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.rankings.len(), 1);
        assert_eq!(deserialized.confidence, 0.8);
    }

    // ============== TarantulaIntegration Tests (TDD - Red Phase) ==============

    #[test]
    fn test_integration_is_coverage_tool_available() {
        // Should not panic, returns bool
        let _available = TarantulaIntegration::is_coverage_tool_available();
    }

    #[test]
    fn test_integration_parse_lcov_output() {
        let lcov = r#"SF:src/main.rs
DA:10,5
DA:20,0
DA:30,12
end_of_record
SF:src/lib.rs
DA:100,8
DA:200,0
end_of_record"#;

        let result = TarantulaIntegration::parse_lcov_output(lcov).unwrap();

        assert_eq!(result.len(), 5);
        assert!(result
            .iter()
            .any(|(s, _)| s.file.as_path() == std::path::Path::new("src/main.rs") && s.line == 10));
        assert!(result
            .iter()
            .any(|(s, _)| s.file.as_path() == std::path::Path::new("src/lib.rs") && s.line == 100));
    }

    #[test]
    fn test_integration_run_localization() {
        // Create test coverage data
        let passed_coverage = vec![
            (StatementId::new("src/buggy.rs", 10), 5_usize),
            (StatementId::new("src/buggy.rs", 20), 10_usize),
            (StatementId::new("src/buggy.rs", 30), 8_usize),
        ];

        let failed_coverage = vec![
            (StatementId::new("src/buggy.rs", 10), 3_usize),
            (StatementId::new("src/buggy.rs", 20), 0_usize),
            (StatementId::new("src/buggy.rs", 40), 5_usize), // Only in failing
        ];

        let config = LocalizationConfig::default();
        let result = TarantulaIntegration::run_localization(
            &passed_coverage,
            &failed_coverage,
            1, // 1 passed test
            1, // 1 failed test
            &config,
        );

        assert!(!result.rankings.is_empty());
        // Line 40 should be most suspicious (only executed by failing tests)
        assert_eq!(result.rankings[0].statement.line, 40);
    }

    #[test]
    fn test_localization_config_default() {
        let config = LocalizationConfig::default();

        assert_eq!(config.formula, SbflFormula::Tarantula);
        assert_eq!(config.top_n, 10);
        assert!(config.include_explanations);
    }

    #[test]
    fn test_localization_config_builder() {
        let config = LocalizationConfig::new()
            .with_formula(SbflFormula::Ochiai)
            .with_top_n(5)
            .with_explanations(false);

        assert_eq!(config.formula, SbflFormula::Ochiai);
        assert_eq!(config.top_n, 5);
        assert!(!config.include_explanations);
    }

    #[test]
    fn test_integration_generate_report_yaml() {
        let result = FaultLocalizationResult {
            rankings: vec![SuspiciousnessRanking {
                rank: 1,
                statement: StatementId::new("src/bug.rs", 42),
                suspiciousness: 0.95,
                scores: {
                    let mut m = HashMap::new();
                    m.insert("tarantula".to_string(), 0.95);
                    m.insert("ochiai".to_string(), 0.92);
                    m
                },
                explanation: "High suspicion".to_string(),
                failed_coverage: 10,
                passed_coverage: 2,
            }],
            formula_used: SbflFormula::Tarantula,
            confidence: 0.85,
            total_passed_tests: 100,
            total_failed_tests: 10,
        };

        let yaml = TarantulaIntegration::generate_report(&result, ReportFormat::Yaml).unwrap();

        assert!(yaml.contains("src/bug.rs"));
        assert!(yaml.contains("42"));
        assert!(yaml.contains("0.95") || yaml.contains("0.9")); // Score present
    }

    #[test]
    fn test_integration_generate_report_json() {
        let result = FaultLocalizationResult {
            rankings: vec![],
            formula_used: SbflFormula::Ochiai,
            confidence: 0.5,
            total_passed_tests: 50,
            total_failed_tests: 5,
        };

        let json = TarantulaIntegration::generate_report(&result, ReportFormat::Json).unwrap();

        assert!(json.contains("Ochiai"));
        assert!(json.contains("0.5"));
    }

    #[test]
    fn test_integration_combine_with_tdg() {
        // Create localization result
        let mut result = FaultLocalizationResult {
            rankings: vec![SuspiciousnessRanking {
                rank: 1,
                statement: StatementId::new("src/complex.rs", 100),
                suspiciousness: 0.8,
                scores: HashMap::new(),
                explanation: String::new(),
                failed_coverage: 5,
                passed_coverage: 10,
            }],
            formula_used: SbflFormula::Tarantula,
            confidence: 0.7,
            total_passed_tests: 100,
            total_failed_tests: 10,
        };

        // Create mock TDG scores
        let mut tdg_scores = HashMap::new();
        tdg_scores.insert("src/complex.rs".to_string(), 45.0_f32); // Low TDG = high debt

        TarantulaIntegration::enrich_with_tdg(&mut result, &tdg_scores);

        // Should have TDG score in the scores map
        assert!(result.rankings[0].scores.contains_key("tdg"));
        assert_eq!(result.rankings[0].scores.get("tdg"), Some(&45.0));
    }

    #[test]
    fn test_report_format_enum() {
        assert_eq!(ReportFormat::default(), ReportFormat::Yaml);
    }

    // ============== SZZ Algorithm Tests ==============

    #[test]
    fn test_szz_identify_bug_fixes() {
        let commits = vec![
            (
                "abc123".to_string(),
                "fix: resolve null pointer exception".to_string(),
            ),
            ("def456".to_string(), "feat: add new feature".to_string()),
            (
                "ghi789".to_string(),
                "bugfix: memory leak in parser".to_string(),
            ),
            ("jkl012".to_string(), "docs: update readme".to_string()),
            (
                "mno345".to_string(),
                "closes #123: fix race condition".to_string(),
            ),
        ];

        let fixes = SzzAnalyzer::identify_bug_fixes(&commits);

        assert_eq!(fixes.len(), 3);
        assert!(fixes.iter().any(|(h, _)| h == "abc123"));
        assert!(fixes.iter().any(|(h, _)| h == "ghi789"));
        assert!(fixes.iter().any(|(h, _)| h == "mno345"));
    }

    #[test]
    fn test_szz_identify_no_fixes() {
        let commits = vec![
            ("abc123".to_string(), "feat: new feature".to_string()),
            ("def456".to_string(), "docs: documentation".to_string()),
            ("ghi789".to_string(), "refactor: clean up code".to_string()),
        ];

        let fixes = SzzAnalyzer::identify_bug_fixes(&commits);

        assert!(fixes.is_empty());
    }

    #[test]
    fn test_szz_trace_introducing_commits() {
        let changed_lines = vec![
            ("src/bug.rs".to_string(), 50, true),  // Deleted (likely bug)
            ("src/bug.rs".to_string(), 51, true),  // Deleted
            ("src/bug.rs".to_string(), 55, false), // Added (fix)
        ];

        let mut blame_data = HashMap::new();
        blame_data.insert(
            ("src/bug.rs".to_string(), 50),
            ("bad_commit_1".to_string(), "author1".to_string()),
        );
        blame_data.insert(
            ("src/bug.rs".to_string(), 51),
            ("bad_commit_1".to_string(), "author1".to_string()),
        );

        let result = SzzAnalyzer::trace_introducing_commits(
            "fix_commit",
            "fix: null pointer exception",
            &changed_lines,
            &blame_data,
        );

        assert_eq!(result.bug_fixing_commit, "fix_commit");
        assert_eq!(result.bug_introducing_commits.len(), 1);
        assert!(result
            .bug_introducing_commits
            .contains(&"bad_commit_1".to_string()));
        assert_eq!(result.faulty_lines.len(), 2);
        assert_eq!(result.confidence, SzzConfidence::High);
    }

    #[test]
    fn test_szz_trace_no_blame_data() {
        let changed_lines = vec![("src/new.rs".to_string(), 10, true)];
        let blame_data = HashMap::new();

        let result = SzzAnalyzer::trace_introducing_commits(
            "fix_commit",
            "fix: issue",
            &changed_lines,
            &blame_data,
        );

        assert!(result.bug_introducing_commits.is_empty());
        assert_eq!(result.confidence, SzzConfidence::Low);
    }

    #[test]
    fn test_szz_filter_cosmetic_changes() {
        let changes = vec![
            ("src/code.rs".to_string(), 1, true), // Empty line
            ("src/code.rs".to_string(), 2, true), // Comment
            ("src/code.rs".to_string(), 3, true), // Import
            ("src/code.rs".to_string(), 4, true), // Real code
        ];

        let mut file_contents = HashMap::new();
        file_contents.insert(
            "src/code.rs".to_string(),
            vec![
                "".to_string(),                               // Line 1: empty
                "// This is a comment".to_string(),           // Line 2: comment
                "use std::collections::HashMap;".to_string(), // Line 3: import
                "let x = compute_value();".to_string(),       // Line 4: real code
            ],
        );

        let filtered = SzzAnalyzer::filter_cosmetic_changes(&changes, &file_contents);

        // Only line 4 (real code) should remain
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].1, 4);
    }

    #[test]
    fn test_szz_calculate_file_suspiciousness() {
        let szz_results = vec![
            SzzResult {
                bug_fixing_commit: "fix1".to_string(),
                bug_introducing_commits: vec!["bad1".to_string()],
                faulty_lines: vec![
                    ("src/buggy.rs".to_string(), 10),
                    ("src/buggy.rs".to_string(), 20),
                ],
                confidence: SzzConfidence::High,
                fix_message: "fix: bug 1".to_string(),
            },
            SzzResult {
                bug_fixing_commit: "fix2".to_string(),
                bug_introducing_commits: vec!["bad2".to_string()],
                faulty_lines: vec![
                    ("src/buggy.rs".to_string(), 30),
                    ("src/other.rs".to_string(), 10),
                ],
                confidence: SzzConfidence::High,
                fix_message: "fix: bug 2".to_string(),
            },
        ];

        let suspiciousness = SzzAnalyzer::calculate_file_suspiciousness(&szz_results);

        // src/buggy.rs has 3 faulty lines across 2 bugs = 1.5 (but capped per file)
        // src/other.rs has 1 faulty line = 0.5
        assert!(suspiciousness.contains_key("src/buggy.rs"));
        assert!(suspiciousness.contains_key("src/other.rs"));
        assert!(
            suspiciousness.get("src/buggy.rs").unwrap()
                > suspiciousness.get("src/other.rs").unwrap()
        );
    }

    // ============== Hybrid Fault Localizer Tests ==============

    #[test]
    fn test_hybrid_combine_scores_basic() {
        let sbfl_result = FaultLocalizationResult {
            rankings: vec![
                SuspiciousnessRanking {
                    rank: 1,
                    statement: StatementId::new("src/a.rs", 10),
                    suspiciousness: 0.9,
                    scores: HashMap::new(),
                    explanation: "High SBFL".to_string(),
                    failed_coverage: 10,
                    passed_coverage: 2,
                },
                SuspiciousnessRanking {
                    rank: 2,
                    statement: StatementId::new("src/b.rs", 20),
                    suspiciousness: 0.5,
                    scores: HashMap::new(),
                    explanation: "Medium SBFL".to_string(),
                    failed_coverage: 5,
                    passed_coverage: 5,
                },
            ],
            formula_used: SbflFormula::Tarantula,
            confidence: 0.8,
            total_passed_tests: 100,
            total_failed_tests: 10,
        };

        let mut historical = HashMap::new();
        historical.insert("src/a.rs".to_string(), 0.2_f32); // Low historical
        historical.insert("src/b.rs".to_string(), 0.9_f32); // High historical

        // With alpha=0.7: a = 0.7*0.9 + 0.3*0.2 = 0.69, b = 0.7*0.5 + 0.3*0.9 = 0.62
        let combined = HybridFaultLocalizer::combine_scores(&sbfl_result, &historical, 0.7);

        assert_eq!(combined.rankings.len(), 2);
        // Rankings should be preserved (a still higher than b with alpha=0.7)
        assert_eq!(
            combined.rankings[0].statement.file,
            PathBuf::from("src/a.rs")
        );

        // Check scores include historical and combined
        assert!(combined.rankings[0].scores.contains_key("historical"));
        assert!(combined.rankings[0].scores.contains_key("combined"));
    }

    #[test]
    fn test_hybrid_combine_scores_reranking() {
        let sbfl_result = FaultLocalizationResult {
            rankings: vec![
                SuspiciousnessRanking {
                    rank: 1,
                    statement: StatementId::new("src/low_hist.rs", 10),
                    suspiciousness: 0.6,
                    scores: HashMap::new(),
                    explanation: String::new(),
                    failed_coverage: 6,
                    passed_coverage: 4,
                },
                SuspiciousnessRanking {
                    rank: 2,
                    statement: StatementId::new("src/high_hist.rs", 20),
                    suspiciousness: 0.4,
                    scores: HashMap::new(),
                    explanation: String::new(),
                    failed_coverage: 4,
                    passed_coverage: 6,
                },
            ],
            formula_used: SbflFormula::Ochiai,
            confidence: 0.7,
            total_passed_tests: 100,
            total_failed_tests: 10,
        };

        let mut historical = HashMap::new();
        historical.insert("src/low_hist.rs".to_string(), 0.0_f32);
        historical.insert("src/high_hist.rs".to_string(), 1.0_f32);

        // With alpha=0.3 (historical weighted heavily):
        // low_hist = 0.3*0.6 + 0.7*0.0 = 0.18
        // high_hist = 0.3*0.4 + 0.7*1.0 = 0.82
        let combined = HybridFaultLocalizer::combine_scores(&sbfl_result, &historical, 0.3);

        // high_hist should now be ranked #1
        assert_eq!(
            combined.rankings[0].statement.file,
            PathBuf::from("src/high_hist.rs")
        );
        assert_eq!(combined.rankings[0].rank, 1);
        assert_eq!(combined.rankings[1].rank, 2);
    }

    #[test]
    fn test_hybrid_no_historical_data() {
        let sbfl_result = FaultLocalizationResult {
            rankings: vec![SuspiciousnessRanking {
                rank: 1,
                statement: StatementId::new("src/new.rs", 10),
                suspiciousness: 0.8,
                scores: HashMap::new(),
                explanation: String::new(),
                failed_coverage: 8,
                passed_coverage: 2,
            }],
            formula_used: SbflFormula::DStar { exponent: 2 },
            confidence: 0.6,
            total_passed_tests: 50,
            total_failed_tests: 5,
        };

        let historical = HashMap::new(); // Empty

        let combined = HybridFaultLocalizer::combine_scores(&sbfl_result, &historical, 0.7);

        // With no historical data, score = 0.7 * 0.8 + 0.3 * 0.0 = 0.56
        assert!((combined.rankings[0].suspiciousness - 0.56).abs() < 0.01);
        assert_eq!(combined.rankings[0].scores.get("historical"), Some(&0.0));
    }

    #[test]
    fn test_szz_confidence_enum() {
        assert_eq!(SzzConfidence::High, SzzConfidence::High);
        assert_ne!(SzzConfidence::High, SzzConfidence::Low);
    }

    #[test]
    fn test_szz_result_serialization() {
        let result = SzzResult {
            bug_fixing_commit: "abc123".to_string(),
            bug_introducing_commits: vec!["def456".to_string()],
            faulty_lines: vec![("src/bug.rs".to_string(), 42)],
            confidence: SzzConfidence::High,
            fix_message: "fix: critical bug".to_string(),
        };

        let json = serde_json::to_string(&result).unwrap();
        let deserialized: SzzResult = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.bug_fixing_commit, "abc123");
        assert_eq!(deserialized.confidence, SzzConfidence::High);
    }
}
