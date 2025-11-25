//! OIP WebAssembly - Standalone WASM package for browser-based defect analysis.
//!
//! This is a lightweight package containing only the classifier and visualization
//! components, without native dependencies (tokio, git2, wgpu).
//!
//! # Usage
//!
//! ```bash
//! cd wasm-pkg
//! wasm-pack build --target web
//! ```
//!
//! # JavaScript Example
//!
//! ```javascript
//! import init, { OipAnalyzer, classify_commit, get_defect_categories } from './oip_wasm.js';
//!
//! async function main() {
//!     await init();
//!
//!     // Single commit classification
//!     const result = classify_commit("fix: null pointer dereference in parser");
//!     console.log(result.category, result.confidence);
//!
//!     // Batch analysis with analyzer
//!     const analyzer = new OipAnalyzer();
//!     analyzer.analyze_message("fix: memory leak in allocator");
//!     analyzer.analyze_message("fix: race condition in mutex");
//!     analyzer.analyze_message("fix: sql injection vulnerability");
//!
//!     const dist = analyzer.get_distribution();
//!     console.log("Categories:", dist.categories);
//!     console.log("Counts:", dist.counts);
//!     console.log("Total:", dist.total);
//!
//!     const stats = analyzer.get_confidence_stats();
//!     console.log("Mean confidence:", stats.mean);
//! }
//!
//! main();
//! ```

use std::collections::HashMap;
use wasm_bindgen::prelude::*;

// ============================================================================
// Initialization
// ============================================================================

/// Initialize the WASM module.
#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
}

// ============================================================================
// Defect Categories
// ============================================================================

/// All recognized defect categories.
const DEFECT_CATEGORIES: &[(&str, &[&str], f32)] = &[
    (
        "ASTTransform",
        &[
            "ast",
            "transform",
            "node",
            "codegen",
            "syntax tree",
            "parse tree",
        ],
        0.85,
    ),
    (
        "TypeErrors",
        &[
            "type error",
            "type mismatch",
            "wrong type",
            "type annotation",
        ],
        0.85,
    ),
    (
        "OwnershipBorrow",
        &["ownership", "borrow", "move", "lifetime", "reference"],
        0.85,
    ),
    (
        "LifetimeAnnotation",
        &["lifetime", "'a", "'static", "lifetime bound"],
        0.80,
    ),
    (
        "TraitBounds",
        &["trait bound", "where clause", "impl trait", "dyn trait"],
        0.80,
    ),
    (
        "PatternMatching",
        &["pattern", "match arm", "destructure", "if let"],
        0.80,
    ),
    (
        "ErrorHandling",
        &["error handling", "unwrap", "expect", "result", "option"],
        0.80,
    ),
    (
        "MemorySafety",
        &[
            "null pointer",
            "memory leak",
            "buffer overflow",
            "use after free",
            "double free",
            "dangling",
        ],
        0.85,
    ),
    (
        "ConcurrencyBugs",
        &[
            "race condition",
            "deadlock",
            "mutex",
            "lock",
            "thread",
            "async",
            "concurrent",
        ],
        0.85,
    ),
    (
        "StdlibMapping",
        &["stdlib", "standard library", "mapping", "python to rust"],
        0.80,
    ),
    (
        "MacroHygiene",
        &["macro", "hygiene", "expansion", "proc macro"],
        0.80,
    ),
    (
        "FFIBoundary",
        &["ffi", "foreign function", "c binding", "extern", "unsafe"],
        0.80,
    ),
    (
        "BuildConfiguration",
        &["build", "cargo", "feature flag", "compile", "linker"],
        0.75,
    ),
    (
        "TestCoverage",
        &[
            "test",
            "coverage",
            "assertion",
            "unit test",
            "integration test",
        ],
        0.75,
    ),
    (
        "DocumentationSync",
        &["doc", "documentation", "rustdoc", "comment"],
        0.70,
    ),
    (
        "PerformanceRegression",
        &["performance", "slow", "optimize", "benchmark", "regression"],
        0.80,
    ),
    (
        "SecurityVulnerabilities",
        &[
            "security",
            "vulnerability",
            "cve",
            "exploit",
            "injection",
            "xss",
            "sql injection",
        ],
        0.90,
    ),
    (
        "OperatorPrecedence",
        &["precedence", "operator", "parentheses", "associativity"],
        0.80,
    ),
    (
        "TypeAnnotationGaps",
        &["type annotation", "inference", "generic type"],
        0.80,
    ),
    (
        "ComprehensionBugs",
        &["comprehension", "list comp", "dict comp", "generator"],
        0.80,
    ),
    (
        "IteratorChain",
        &["iterator", ".map(", ".filter(", ".fold(", "chain"],
        0.80,
    ),
];

// ============================================================================
// Data Types
// ============================================================================

/// Result of classifying a commit.
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
pub struct DefectDistribution {
    categories: Vec<String>,
    counts: Vec<u32>,
    percentages: Vec<f32>,
    total: u32,
}

#[wasm_bindgen]
impl DefectDistribution {
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

    /// Number of categories.
    pub fn len(&self) -> usize {
        self.categories.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.categories.is_empty()
    }

    /// Export as JSON string.
    pub fn to_json(&self) -> String {
        let items: Vec<String> = self
            .categories
            .iter()
            .zip(self.counts.iter())
            .zip(self.percentages.iter())
            .map(|((cat, count), pct)| {
                format!(
                    r#"{{"category":"{}","count":{},"percentage":{:.2}}}"#,
                    cat, count, pct
                )
            })
            .collect();
        format!(
            r#"{{"total":{},"distribution":[{}]}}"#,
            self.total,
            items.join(",")
        )
    }
}

/// Confidence statistics.
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct ConfidenceStats {
    min: f32,
    max: f32,
    mean: f32,
    std_dev: f32,
    count: usize,
}

#[wasm_bindgen]
impl ConfidenceStats {
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

    #[wasm_bindgen(getter)]
    pub fn count(&self) -> usize {
        self.count
    }

    /// Export as JSON string.
    pub fn to_json(&self) -> String {
        format!(
            r#"{{"min":{:.3},"max":{:.3},"mean":{:.3},"std_dev":{:.3},"count":{}}}"#,
            self.min, self.max, self.mean, self.std_dev, self.count
        )
    }
}

// ============================================================================
// Analyzer
// ============================================================================

/// Main analyzer for organizational intelligence.
#[wasm_bindgen]
pub struct OipAnalyzer {
    analyses: Vec<CommitAnalysis>,
}

#[wasm_bindgen]
impl OipAnalyzer {
    /// Create a new analyzer.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            analyses: Vec::new(),
        }
    }

    /// Analyze a single commit message.
    pub fn analyze_message(&mut self, message: &str) -> CommitAnalysis {
        let analysis = classify_message(message);
        self.analyses.push(analysis.clone());
        analysis
    }

    /// Analyze multiple commit messages.
    pub fn analyze_messages(&mut self, messages: Vec<String>) -> Vec<CommitAnalysis> {
        messages.iter().map(|m| self.analyze_message(m)).collect()
    }

    /// Get defect distribution from analyzed commits.
    pub fn get_distribution(&self) -> DefectDistribution {
        let mut counts: HashMap<String, u32> = HashMap::new();

        for analysis in &self.analyses {
            *counts.entry(analysis.category.clone()).or_insert(0) += 1;
        }

        let total = self.analyses.len() as u32;
        let mut items: Vec<_> = counts.into_iter().collect();
        items.sort_by(|a, b| b.1.cmp(&a.1));

        let categories: Vec<String> = items.iter().map(|(k, _)| k.clone()).collect();
        let count_vec: Vec<u32> = items.iter().map(|(_, v)| *v).collect();
        let percentages: Vec<f32> = count_vec
            .iter()
            .map(|c| {
                if total > 0 {
                    (*c as f32 / total as f32) * 100.0
                } else {
                    0.0
                }
            })
            .collect();

        DefectDistribution {
            categories,
            counts: count_vec,
            percentages,
            total,
        }
    }

    /// Get confidence statistics.
    pub fn get_confidence_stats(&self) -> ConfidenceStats {
        if self.analyses.is_empty() {
            return ConfidenceStats {
                min: 0.0,
                max: 0.0,
                mean: 0.0,
                std_dev: 0.0,
                count: 0,
            };
        }

        let values: Vec<f32> = self.analyses.iter().map(|a| a.confidence).collect();
        let min = values.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / values.len() as f32;

        ConfidenceStats {
            min,
            max,
            mean,
            std_dev: variance.sqrt(),
            count: values.len(),
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

    /// Export all analyses as JSON.
    pub fn to_json(&self) -> String {
        let dist = self.get_distribution();
        let stats = self.get_confidence_stats();
        format!(
            r#"{{"distribution":{},"confidence":{}}}"#,
            dist.to_json(),
            stats.to_json()
        )
    }
}

impl Default for OipAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Classification Functions
// ============================================================================

fn classify_message(message: &str) -> CommitAnalysis {
    let message_lower = message.to_lowercase();

    // Check if it's a defect-fix commit
    let is_fix = message_lower.starts_with("fix")
        || message_lower.contains("fix:")
        || message_lower.contains("bugfix")
        || message_lower.contains("hotfix")
        || message_lower.contains("patch:");

    if !is_fix {
        return CommitAnalysis {
            category: "NotDefect".to_string(),
            confidence: 0.0,
            message: message.to_string(),
        };
    }

    // Find best matching category
    let mut best_category = "Unknown";
    let mut best_confidence = 0.0f32;
    let mut match_count = 0;

    for (category, patterns, base_confidence) in DEFECT_CATEGORIES {
        let mut category_matches = 0;
        for pattern in *patterns {
            if message_lower.contains(pattern) {
                category_matches += 1;
            }
        }

        if category_matches > 0 {
            // Boost confidence for multiple pattern matches
            let confidence = (*base_confidence + 0.05 * (category_matches - 1) as f32).min(0.95);
            if confidence > best_confidence
                || (confidence == best_confidence && category_matches > match_count)
            {
                best_category = category;
                best_confidence = confidence;
                match_count = category_matches;
            }
        }
    }

    // Default confidence for fix commits without specific patterns
    if best_confidence == 0.0 && is_fix {
        best_category = "Unknown";
        best_confidence = 0.70;
    }

    CommitAnalysis {
        category: best_category.to_string(),
        confidence: best_confidence,
        message: message.to_string(),
    }
}

// ============================================================================
// Standalone Functions
// ============================================================================

/// Classify a single commit message.
#[wasm_bindgen]
pub fn classify_commit(message: &str) -> CommitAnalysis {
    classify_message(message)
}

/// Get all defect category names.
#[wasm_bindgen]
pub fn get_defect_categories() -> Vec<String> {
    DEFECT_CATEGORIES
        .iter()
        .map(|(name, _, _)| name.to_string())
        .collect()
}

/// Generate ASCII bar chart for distribution.
#[wasm_bindgen]
pub fn distribution_to_ascii(dist: &DefectDistribution, width: usize) -> String {
    if dist.is_empty() {
        return "No data".to_string();
    }

    let max_count = dist.counts.iter().max().copied().unwrap_or(1) as f32;
    let max_label_len = dist.categories.iter().map(|s| s.len()).max().unwrap_or(15);

    let mut output = String::new();
    for (i, category) in dist.categories.iter().enumerate() {
        let count = dist.counts[i];
        let pct = dist.percentages[i];
        let bar_width = width.saturating_sub(max_label_len + 15);
        let bar_len = ((count as f32 / max_count) * bar_width as f32) as usize;
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

/// Generate ASCII histogram for confidence values.
#[wasm_bindgen]
pub fn confidence_histogram(
    stats: &ConfidenceStats,
    values: Vec<f32>,
    bins: usize,
    height: usize,
) -> String {
    if values.is_empty() || bins == 0 || height == 0 {
        return "No data".to_string();
    }

    let range = stats.max - stats.min;
    let bin_width = if range > 0.0 {
        range / bins as f32
    } else {
        1.0
    };

    let mut bin_counts = vec![0u32; bins];
    for v in &values {
        let idx = ((*v - stats.min) / bin_width).floor() as usize;
        let idx = idx.min(bins - 1);
        bin_counts[idx] += 1;
    }

    let max_count = *bin_counts.iter().max().unwrap_or(&1) as f32;

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

    output.push_str("└");
    output.push_str(&"──".repeat(bins));
    output.push('\n');
    output.push_str(&format!(" {:.2}  ...  {:.2}\n", stats.min, stats.max));
    output.push_str(&format!(
        " Mean: {:.2}, StdDev: {:.2}\n",
        stats.mean, stats.std_dev
    ));

    output
}

/// Get library version.
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

// ============================================================================
// Semantic Clustering (powered by aprender)
// ============================================================================

use aprender::cluster::KMeans;
use aprender::primitives::{Matrix, Vector};
use aprender::text::similarity::pairwise_cosine_similarity;
use aprender::text::tokenize::WhitespaceTokenizer;
use aprender::text::vectorize::TfidfVectorizer;
use aprender::traits::UnsupervisedEstimator;

/// Semantic clustering result for visualization.
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct ClusterResult {
    labels: Vec<u32>,
    centroids_x: Vec<f32>,
    centroids_y: Vec<f32>,
    points_x: Vec<f32>,
    points_y: Vec<f32>,
    messages: Vec<String>,
    n_clusters: usize,
}

#[wasm_bindgen]
impl ClusterResult {
    #[wasm_bindgen(getter)]
    pub fn labels(&self) -> Vec<u32> {
        self.labels.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn centroids_x(&self) -> Vec<f32> {
        self.centroids_x.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn centroids_y(&self) -> Vec<f32> {
        self.centroids_y.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn points_x(&self) -> Vec<f32> {
        self.points_x.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn points_y(&self) -> Vec<f32> {
        self.points_y.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn messages(&self) -> Vec<String> {
        self.messages.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn n_clusters(&self) -> usize {
        self.n_clusters
    }

    pub fn to_json(&self) -> String {
        let points: Vec<String> = (0..self.points_x.len())
            .map(|i| {
                format!(
                    r#"{{"x":{:.4},"y":{:.4},"label":{},"message":"{}"}}"#,
                    self.points_x[i],
                    self.points_y[i],
                    self.labels[i],
                    self.messages[i].replace('"', "\\\"")
                )
            })
            .collect();
        format!(
            r#"{{"n_clusters":{},"points":[{}]}}"#,
            self.n_clusters,
            points.join(",")
        )
    }
}

/// Similarity matrix result.
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct SimilarityMatrix {
    data: Vec<f32>,
    size: usize,
    messages: Vec<String>,
}

#[wasm_bindgen]
impl SimilarityMatrix {
    #[wasm_bindgen(getter)]
    pub fn data(&self) -> Vec<f32> {
        self.data.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn size(&self) -> usize {
        self.size
    }

    #[wasm_bindgen(getter)]
    pub fn messages(&self) -> Vec<String> {
        self.messages.clone()
    }

    pub fn get(&self, i: usize, j: usize) -> f32 {
        if i < self.size && j < self.size {
            self.data[i * self.size + j]
        } else {
            0.0
        }
    }

    pub fn to_json(&self) -> String {
        format!(
            r#"{{"size":{},"data":[{}]}}"#,
            self.size,
            self.data
                .iter()
                .map(|v| format!("{:.4}", v))
                .collect::<Vec<_>>()
                .join(",")
        )
    }
}

/// Semantic analyzer using TF-IDF + K-Means clustering.
#[wasm_bindgen]
pub struct SemanticAnalyzer {
    messages: Vec<String>,
    tfidf_matrix: Option<Vec<Vec<f64>>>,
}

#[wasm_bindgen]
impl SemanticAnalyzer {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
            tfidf_matrix: None,
        }
    }

    /// Add commit messages for analysis.
    pub fn add_messages(&mut self, messages: Vec<String>) {
        self.messages.extend(messages);
        self.tfidf_matrix = None; // Invalidate cache
    }

    /// Add a single message.
    pub fn add_message(&mut self, message: String) {
        self.messages.push(message);
        self.tfidf_matrix = None;
    }

    /// Clear all messages.
    pub fn clear(&mut self) {
        self.messages.clear();
        self.tfidf_matrix = None;
    }

    /// Get message count.
    pub fn count(&self) -> usize {
        self.messages.len()
    }

    /// Compute TF-IDF vectors (lazy, cached).
    fn compute_tfidf(&mut self) -> Result<&Vec<Vec<f64>>, String> {
        if self.tfidf_matrix.is_some() {
            return Ok(self.tfidf_matrix.as_ref().unwrap());
        }

        if self.messages.is_empty() {
            return Err("No messages to analyze".to_string());
        }

        let mut vectorizer = TfidfVectorizer::new()
            .with_tokenizer(Box::new(WhitespaceTokenizer::new()))
            .with_lowercase(true);
        let matrix = vectorizer
            .fit_transform(&self.messages)
            .map_err(|e| format!("TF-IDF error: {:?}", e))?;

        // Convert Matrix to Vec<Vec<f64>>
        let rows = matrix.n_rows();
        let cols = matrix.n_cols();
        let mut result = Vec::with_capacity(rows);
        for i in 0..rows {
            let mut row = Vec::with_capacity(cols);
            for j in 0..cols {
                row.push(matrix.get(i, j));
            }
            result.push(row);
        }

        self.tfidf_matrix = Some(result);
        Ok(self.tfidf_matrix.as_ref().unwrap())
    }

    /// Compute pairwise similarity matrix.
    pub fn compute_similarity(&mut self) -> Result<SimilarityMatrix, String> {
        let tfidf = self.compute_tfidf()?;
        let n = tfidf.len();

        // Convert to Vector<f64> for aprender
        let vectors: Vec<Vector<f64>> = tfidf.iter().map(|row| Vector::from_slice(row)).collect();

        let sim_matrix = pairwise_cosine_similarity(&vectors)
            .map_err(|e| format!("Similarity error: {:?}", e))?;

        // Flatten to 1D
        let data: Vec<f32> = sim_matrix
            .iter()
            .flat_map(|row| row.iter().map(|&v| v as f32))
            .collect();

        Ok(SimilarityMatrix {
            data,
            size: n,
            messages: self.messages.clone(),
        })
    }

    /// Cluster messages using K-Means.
    pub fn cluster(&mut self, n_clusters: usize) -> Result<ClusterResult, String> {
        let tfidf = self.compute_tfidf()?;
        let n = tfidf.len();

        if n < n_clusters {
            return Err(format!(
                "Need at least {} messages for {} clusters",
                n_clusters, n_clusters
            ));
        }

        // Create Matrix<f32> for KMeans
        let n_features = tfidf[0].len();
        let flat_data: Vec<f32> = tfidf
            .iter()
            .flat_map(|row| row.iter().map(|&v| v as f32))
            .collect();

        let data_matrix = Matrix::from_vec(n, n_features, flat_data)
            .map_err(|e| format!("Matrix error: {}", e))?;
        let mut kmeans = KMeans::new(n_clusters).with_max_iter(100);
        kmeans
            .fit(&data_matrix)
            .map_err(|e| format!("KMeans fit error: {:?}", e))?;
        let labels = kmeans.predict(&data_matrix);

        // Project to 2D using simple PCA-like approach (first 2 dims of TF-IDF)
        let (points_x, points_y) = project_to_2d(tfidf);

        // Compute centroids in 2D
        let mut centroids_x = vec![0.0f32; n_clusters];
        let mut centroids_y = vec![0.0f32; n_clusters];
        let mut counts = vec![0usize; n_clusters];

        for (i, &label) in labels.iter().enumerate() {
            let l = label.max(0) as usize; // i32 -> usize safely
            centroids_x[l] += points_x[i];
            centroids_y[l] += points_y[i];
            counts[l] += 1;
        }

        for i in 0..n_clusters {
            if counts[i] > 0 {
                centroids_x[i] /= counts[i] as f32;
                centroids_y[i] /= counts[i] as f32;
            }
        }

        Ok(ClusterResult {
            labels: labels.iter().map(|&l| l.max(0) as u32).collect(),
            centroids_x,
            centroids_y,
            points_x,
            points_y,
            messages: self.messages.clone(),
            n_clusters,
        })
    }
}

impl Default for SemanticAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Project high-dimensional TF-IDF vectors to 2D for visualization.
/// Uses a simple variance-based projection (top 2 principal components approximation).
fn project_to_2d(tfidf: &[Vec<f64>]) -> (Vec<f32>, Vec<f32>) {
    let n = tfidf.len();
    if n == 0 || tfidf[0].is_empty() {
        return (vec![], vec![]);
    }

    let n_features = tfidf[0].len();

    // Find dimensions with highest variance
    let mut variances: Vec<(usize, f64)> = (0..n_features)
        .map(|j| {
            let col: Vec<f64> = tfidf.iter().map(|row| row[j]).collect();
            let mean = col.iter().sum::<f64>() / n as f64;
            let var = col.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;
            (j, var)
        })
        .collect();

    variances.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let dim1 = variances.first().map(|&(i, _)| i).unwrap_or(0);
    let dim2 = variances
        .get(1)
        .map(|&(i, _)| i)
        .unwrap_or(dim1.saturating_add(1).min(n_features - 1));

    let points_x: Vec<f32> = tfidf
        .iter()
        .map(|row| row.get(dim1).copied().unwrap_or(0.0) as f32)
        .collect();
    let points_y: Vec<f32> = tfidf
        .iter()
        .map(|row| row.get(dim2).copied().unwrap_or(0.0) as f32)
        .collect();

    // Normalize to [-1, 1] range
    let (min_x, max_x) = min_max(&points_x);
    let (min_y, max_y) = min_max(&points_y);

    let range_x = (max_x - min_x).max(0.001);
    let range_y = (max_y - min_y).max(0.001);

    let norm_x: Vec<f32> = points_x
        .iter()
        .map(|&x| 2.0 * (x - min_x) / range_x - 1.0)
        .collect();
    let norm_y: Vec<f32> = points_y
        .iter()
        .map(|&y| 2.0 * (y - min_y) / range_y - 1.0)
        .collect();

    (norm_x, norm_y)
}

fn min_max(v: &[f32]) -> (f32, f32) {
    let min = v.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    (min, max)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_fix_commit() {
        let result = classify_commit("fix: buffer overflow and memory leak");
        assert_eq!(result.category, "MemorySafety");
        assert!(result.confidence > 0.0);
    }

    #[test]
    fn test_classify_non_fix() {
        let result = classify_commit("feat: add new feature");
        assert_eq!(result.category, "NotDefect");
        assert_eq!(result.confidence, 0.0);
    }

    #[test]
    fn test_analyzer() {
        let mut analyzer = OipAnalyzer::new();
        analyzer.analyze_message("fix: memory leak");
        analyzer.analyze_message("fix: race condition");
        analyzer.analyze_message("fix: type error");

        assert_eq!(analyzer.count(), 3);

        let dist = analyzer.get_distribution();
        assert!(!dist.is_empty());
        assert_eq!(dist.total, 3);

        let stats = analyzer.get_confidence_stats();
        assert_eq!(stats.count, 3);
        assert!(stats.mean > 0.0);
    }

    #[test]
    fn test_defect_categories() {
        let cats = get_defect_categories();
        assert!(!cats.is_empty());
        assert!(cats.contains(&"MemorySafety".to_string()));
        assert!(cats.contains(&"SecurityVulnerabilities".to_string()));
    }

    #[test]
    fn test_json_export() {
        let mut analyzer = OipAnalyzer::new();
        analyzer.analyze_message("fix: sql injection");

        let json = analyzer.to_json();
        assert!(json.contains("distribution"));
        assert!(json.contains("confidence"));
    }

    #[test]
    fn test_semantic_analyzer() {
        let mut analyzer = SemanticAnalyzer::new();
        analyzer.add_message("fix: null pointer exception".to_string());
        analyzer.add_message("fix: memory leak in parser".to_string());
        analyzer.add_message("docs: update README".to_string());
        analyzer.add_message("docs: add API docs".to_string());

        assert_eq!(analyzer.count(), 4);

        // Test clustering
        let result = analyzer.cluster(2).expect("clustering should work");
        assert_eq!(result.n_clusters, 2);
        assert_eq!(result.labels.len(), 4);

        // Test similarity
        let sim = analyzer
            .compute_similarity()
            .expect("similarity should work");
        assert_eq!(sim.size, 4);
        assert_eq!(sim.data.len(), 16); // 4x4 matrix
    }
}
