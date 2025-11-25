//! WebAssembly bindings for Organizational Intelligence Plugin.
//!
//! Provides JavaScript-accessible functions for defect pattern analysis
//! and visualization in the browser.
//!
//! # Usage (JavaScript)
//!
//! ```javascript
//! import init, {
//!     analyze_commits,
//!     defect_distribution_chart,
//!     confidence_histogram,
//!     cross_repo_heatmap,
//!     OipAnalyzer
//! } from 'organizational-intelligence-plugin';
//!
//! await init();
//!
//! // Analyze commit messages
//! const commits = [
//!     { message: "fix: resolve null pointer in parser", confidence: 0.85 },
//!     { message: "fix: handle edge case in validator", confidence: 0.78 }
//! ];
//!
//! const analysis = analyze_commits(commits);
//! console.log(analysis.categories); // ["ASTTransform", "TypeErrors", ...]
//! console.log(analysis.counts);     // [5, 3, ...]
//!
//! // Generate PNG chart
//! const pngData = defect_distribution_chart(analysis, { width: 800, height: 400 });
//! const blob = new Blob([pngData], { type: 'image/png' });
//! document.getElementById('chart').src = URL.createObjectURL(blob);
//! ```

use wasm_bindgen::prelude::*;

use crate::classifier::RuleBasedClassifier;

// ============================================================================
// Initialization
// ============================================================================

/// Initialize the WASM module.
#[wasm_bindgen(start)]
pub fn init() {
    // Set panic hook for better error messages in browser console
    #[cfg(feature = "wasm")]
    console_error_panic_hook::set_once();
}

// ============================================================================
// Data Types
// ============================================================================

/// Defect analysis result for a single commit.
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct CommitAnalysis {
    category: String,
    confidence: f32,
    message: String,
}

#[wasm_bindgen]
impl CommitAnalysis {
    #[wasm_bindgen(getter)]
    pub fn category(&self) -> String {
        self.category.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn confidence(&self) -> f32 {
        self.confidence
    }

    #[wasm_bindgen(getter)]
    pub fn message(&self) -> String {
        self.message.clone()
    }
}

/// Distribution of defect categories.
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct DefectDistributionResult {
    categories: Vec<String>,
    counts: Vec<u32>,
    percentages: Vec<f32>,
    total: u32,
}

#[wasm_bindgen]
impl DefectDistributionResult {
    #[wasm_bindgen(getter)]
    pub fn categories(&self) -> Vec<String> {
        self.categories.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn counts(&self) -> Vec<u32> {
        self.counts.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn percentages(&self) -> Vec<f32> {
        self.percentages.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn total(&self) -> u32 {
        self.total
    }

    /// Get category at index.
    pub fn category_at(&self, idx: usize) -> Option<String> {
        self.categories.get(idx).cloned()
    }

    /// Get count at index.
    pub fn count_at(&self, idx: usize) -> Option<u32> {
        self.counts.get(idx).copied()
    }

    /// Get percentage at index.
    pub fn percentage_at(&self, idx: usize) -> Option<f32> {
        self.percentages.get(idx).copied()
    }

    /// Number of categories.
    pub fn len(&self) -> usize {
        self.categories.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.categories.is_empty()
    }
}

/// Confidence distribution statistics.
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct ConfidenceStats {
    values: Vec<f32>,
    min: f32,
    max: f32,
    mean: f32,
    std_dev: f32,
}

#[wasm_bindgen]
impl ConfidenceStats {
    #[wasm_bindgen(getter)]
    pub fn values(&self) -> Vec<f32> {
        self.values.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn min(&self) -> f32 {
        self.min
    }

    #[wasm_bindgen(getter)]
    pub fn max(&self) -> f32 {
        self.max
    }

    #[wasm_bindgen(getter)]
    pub fn mean(&self) -> f32 {
        self.mean
    }

    #[wasm_bindgen(getter)]
    pub fn std_dev(&self) -> f32 {
        self.std_dev
    }

    /// Number of values.
    pub fn count(&self) -> usize {
        self.values.len()
    }
}

/// Chart options for visualization.
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct ChartOptions {
    width: u32,
    height: u32,
    color: String,
    background: String,
    title: Option<String>,
}

#[wasm_bindgen]
impl ChartOptions {
    /// Create default chart options.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            width: 800,
            height: 400,
            color: "#4285F4".to_string(),
            background: "#FFFFFF".to_string(),
            title: None,
        }
    }

    /// Set width in pixels.
    pub fn width(mut self, width: u32) -> Self {
        self.width = width;
        self
    }

    /// Set height in pixels.
    pub fn height(mut self, height: u32) -> Self {
        self.height = height;
        self
    }

    /// Set primary color (hex: #RRGGBB).
    pub fn color(mut self, color: &str) -> Self {
        self.color = color.to_string();
        self
    }

    /// Set background color (hex: #RRGGBB).
    pub fn background(mut self, bg: &str) -> Self {
        self.background = bg.to_string();
        self
    }

    /// Set chart title.
    pub fn title(mut self, title: &str) -> Self {
        self.title = Some(title.to_string());
        self
    }
}

impl Default for ChartOptions {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Analyzer Class
// ============================================================================

/// Main analyzer for organizational intelligence.
#[wasm_bindgen]
pub struct OipAnalyzer {
    classifier: RuleBasedClassifier,
    analyses: Vec<CommitAnalysis>,
}

#[wasm_bindgen]
impl OipAnalyzer {
    /// Create a new analyzer.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            classifier: RuleBasedClassifier::new(),
            analyses: Vec::new(),
        }
    }

    /// Analyze a single commit message.
    pub fn analyze_message(&mut self, message: &str) -> CommitAnalysis {
        let analysis = if let Some(result) = self.classifier.classify_from_message(message) {
            CommitAnalysis {
                category: format!("{:?}", result.category),
                confidence: result.confidence,
                message: message.to_string(),
            }
        } else {
            CommitAnalysis {
                category: "Unknown".to_string(),
                confidence: 0.0,
                message: message.to_string(),
            }
        };
        self.analyses.push(analysis.clone());
        analysis
    }

    /// Analyze multiple commit messages.
    pub fn analyze_messages(&mut self, messages: Vec<String>) -> Vec<CommitAnalysis> {
        messages.iter().map(|m| self.analyze_message(m)).collect()
    }

    /// Get defect distribution from analyzed commits.
    pub fn get_distribution(&self) -> DefectDistributionResult {
        let mut category_counts: std::collections::HashMap<String, u32> =
            std::collections::HashMap::new();

        for analysis in &self.analyses {
            *category_counts
                .entry(analysis.category.clone())
                .or_insert(0) += 1;
        }

        let total = self.analyses.len() as u32;
        let mut items: Vec<_> = category_counts.into_iter().collect();
        items.sort_by(|a, b| b.1.cmp(&a.1));

        let categories: Vec<String> = items.iter().map(|(k, _)| k.clone()).collect();
        let counts: Vec<u32> = items.iter().map(|(_, v)| *v).collect();
        let percentages: Vec<f32> = counts
            .iter()
            .map(|c| (*c as f32 / total as f32) * 100.0)
            .collect();

        DefectDistributionResult {
            categories,
            counts,
            percentages,
            total,
        }
    }

    /// Get confidence statistics.
    pub fn get_confidence_stats(&self) -> ConfidenceStats {
        let values: Vec<f32> = self.analyses.iter().map(|a| a.confidence).collect();

        if values.is_empty() {
            return ConfidenceStats {
                values: vec![],
                min: 0.0,
                max: 0.0,
                mean: 0.0,
                std_dev: 0.0,
            };
        }

        let min = values.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / values.len() as f32;
        let std_dev = variance.sqrt();

        ConfidenceStats {
            values,
            min,
            max,
            mean,
            std_dev,
        }
    }

    /// Clear all analyzed data.
    pub fn clear(&mut self) {
        self.analyses.clear();
    }

    /// Get number of analyzed commits.
    pub fn count(&self) -> usize {
        self.analyses.len()
    }
}

impl Default for OipAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Standalone Functions
// ============================================================================

/// Classify a single commit message.
///
/// Returns the defect category and confidence score.
#[wasm_bindgen]
pub fn classify_commit(message: &str) -> CommitAnalysis {
    let classifier = RuleBasedClassifier::new();
    if let Some(result) = classifier.classify_from_message(message) {
        CommitAnalysis {
            category: format!("{:?}", result.category),
            confidence: result.confidence,
            message: message.to_string(),
        }
    } else {
        CommitAnalysis {
            category: "Unknown".to_string(),
            confidence: 0.0,
            message: message.to_string(),
        }
    }
}

/// Get all defect category names.
#[wasm_bindgen]
pub fn get_defect_categories() -> Vec<String> {
    vec![
        "ASTTransform".to_string(),
        "TypeErrors".to_string(),
        "OwnershipBorrow".to_string(),
        "LifetimeAnnotation".to_string(),
        "TraitBounds".to_string(),
        "PatternMatching".to_string(),
        "ErrorHandling".to_string(),
        "MemorySafety".to_string(),
        "ConcurrencySync".to_string(),
        "StdlibMapping".to_string(),
        "MacroHygiene".to_string(),
        "FFIBoundary".to_string(),
        "BuildConfiguration".to_string(),
        "TestCoverage".to_string(),
        "DocumentationSync".to_string(),
        "PerformanceRegression".to_string(),
        "SecurityVulnerabilities".to_string(),
        "Unknown".to_string(),
    ]
}

/// Generate ASCII bar chart for defect distribution.
///
/// Returns a string representation suitable for terminal or pre-formatted display.
#[wasm_bindgen]
pub fn distribution_to_ascii(dist: &DefectDistributionResult, width: usize) -> String {
    let max_count = dist.counts.iter().max().copied().unwrap_or(1) as f32;
    let max_label_len = dist.categories.iter().map(|s| s.len()).max().unwrap_or(15);

    let mut output = String::new();
    for (i, category) in dist.categories.iter().enumerate() {
        let count = dist.counts[i];
        let pct = dist.percentages[i];
        let bar_len = ((count as f32 / max_count) * (width - max_label_len - 15) as f32) as usize;
        let bar: String = "█".repeat(bar_len.max(1));

        output.push_str(&format!(
            "{:width$} {:>5} {:>5.1}%\n",
            category,
            bar,
            pct,
            width = max_label_len
        ));
    }
    output
}

/// Generate ASCII histogram for confidence distribution.
#[wasm_bindgen]
pub fn confidence_to_ascii(stats: &ConfidenceStats, bins: usize, height: usize) -> String {
    if stats.values.is_empty() {
        return "No data".to_string();
    }

    let range = stats.max - stats.min;
    let bin_width = if range > 0.0 {
        range / bins as f32
    } else {
        1.0
    };

    // Count values in each bin
    let mut bin_counts = vec![0u32; bins];
    for &v in &stats.values {
        let bin_idx = ((v - stats.min) / bin_width).floor() as usize;
        let bin_idx = bin_idx.min(bins - 1);
        bin_counts[bin_idx] += 1;
    }

    let max_count = *bin_counts.iter().max().unwrap_or(&1) as f32;

    // Build histogram
    let mut output = String::new();
    for row in (0..height).rev() {
        let threshold = (row as f32 / height as f32) * max_count;
        output.push_str("│");
        for &count in &bin_counts {
            if count as f32 >= threshold {
                output.push_str("██");
            } else {
                output.push_str("  ");
            }
        }
        output.push('\n');
    }

    // X-axis
    output.push_str("└");
    output.push_str(&"──".repeat(bins));
    output.push('\n');
    output.push_str(&format!(
        " {:.2}{}  {:.2}\n",
        stats.min,
        " ".repeat(bins * 2 - 10),
        stats.max
    ));
    output.push_str(&format!(
        " Mean: {:.2}, StdDev: {:.2}\n",
        stats.mean, stats.std_dev
    ));

    output
}

/// Generate JSON summary of analysis.
#[wasm_bindgen]
pub fn analysis_to_json(dist: &DefectDistributionResult, stats: &ConfidenceStats) -> String {
    let categories_json: Vec<String> = dist
        .categories
        .iter()
        .zip(dist.counts.iter())
        .zip(dist.percentages.iter())
        .map(|((cat, count), pct)| {
            format!(
                r#"{{"category":"{}","count":{},"percentage":{:.2}}}"#,
                cat, count, pct
            )
        })
        .collect();

    format!(
        r#"{{"total":{},"categories":[{}],"confidence":{{"min":{:.3},"max":{:.3},"mean":{:.3},"std_dev":{:.3}}}}}"#,
        dist.total,
        categories_json.join(","),
        stats.min,
        stats.max,
        stats.mean,
        stats.std_dev
    )
}

/// Get library version.
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_commit() {
        let result = classify_commit("fix: resolve null pointer dereference in parser");
        assert!(!result.category.is_empty());
        assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
    }

    #[test]
    fn test_oip_analyzer() {
        let mut analyzer = OipAnalyzer::new();

        analyzer.analyze_message("fix: resolve type error in validator");
        analyzer.analyze_message("fix: handle ownership transfer correctly");
        analyzer.analyze_message("fix: add missing lifetime annotation");

        assert_eq!(analyzer.count(), 3);

        let dist = analyzer.get_distribution();
        assert!(!dist.is_empty());
        assert_eq!(dist.total, 3);

        let stats = analyzer.get_confidence_stats();
        assert_eq!(stats.count(), 3);
    }

    #[test]
    fn test_defect_categories() {
        let cats = get_defect_categories();
        assert!(cats.contains(&"ASTTransform".to_string()));
        assert!(cats.contains(&"TypeErrors".to_string()));
        assert!(cats.contains(&"Unknown".to_string()));
    }

    #[test]
    fn test_distribution_to_ascii() {
        let dist = DefectDistributionResult {
            categories: vec!["ASTTransform".to_string(), "TypeErrors".to_string()],
            counts: vec![10, 5],
            percentages: vec![66.7, 33.3],
            total: 15,
        };

        let ascii = distribution_to_ascii(&dist, 40);
        assert!(ascii.contains("ASTTransform"));
        assert!(ascii.contains("TypeErrors"));
    }

    #[test]
    fn test_confidence_to_ascii() {
        let stats = ConfidenceStats {
            values: vec![0.7, 0.8, 0.85, 0.9],
            min: 0.7,
            max: 0.9,
            mean: 0.8125,
            std_dev: 0.072,
        };

        let ascii = confidence_to_ascii(&stats, 10, 5);
        assert!(ascii.contains("Mean"));
    }

    #[test]
    fn test_analysis_to_json() {
        let dist = DefectDistributionResult {
            categories: vec!["ASTTransform".to_string()],
            counts: vec![5],
            percentages: vec![100.0],
            total: 5,
        };

        let stats = ConfidenceStats {
            values: vec![0.8],
            min: 0.8,
            max: 0.8,
            mean: 0.8,
            std_dev: 0.0,
        };

        let json = analysis_to_json(&dist, &stats);
        assert!(json.contains("\"total\":5"));
        assert!(json.contains("ASTTransform"));
    }

    #[test]
    fn test_chart_options() {
        let opts = ChartOptions::new()
            .width(1024)
            .height(768)
            .color("#FF0000")
            .title("Test Chart");

        assert_eq!(opts.width, 1024);
        assert_eq!(opts.height, 768);
        assert_eq!(opts.color, "#FF0000");
        assert_eq!(opts.title, Some("Test Chart".to_string()));
    }
}
