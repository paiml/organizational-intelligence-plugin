//! Visualization module for OIP defect pattern analysis.
//!
//! Requires the `viz` feature flag: `cargo build --features viz`

#[cfg(feature = "viz")]
use trueno_viz::output::{TerminalEncoder, TerminalMode};
#[cfg(feature = "viz")]
use trueno_viz::plots::{BinStrategy, Heatmap, Histogram};

use crate::training::TrainingExample;

/// Defect distribution data for visualization
pub struct DefectDistribution {
    pub categories: Vec<String>,
    pub counts: Vec<u32>,
    pub percentages: Vec<f32>,
}

impl DefectDistribution {
    /// Create from training examples
    pub fn from_examples(examples: &[TrainingExample]) -> Self {
        let mut category_counts: std::collections::HashMap<String, u32> =
            std::collections::HashMap::new();

        for example in examples {
            let label_str = format!("{:?}", example.label);
            *category_counts.entry(label_str).or_insert(0) += 1;
        }

        let total = examples.len() as f32;
        let mut items: Vec<_> = category_counts.into_iter().collect();
        items.sort_by(|a, b| b.1.cmp(&a.1)); // Sort by count descending

        let categories: Vec<String> = items.iter().map(|(k, _)| k.clone()).collect();
        let counts: Vec<u32> = items.iter().map(|(_, v)| *v).collect();
        let percentages: Vec<f32> = counts.iter().map(|c| (*c as f32 / total) * 100.0).collect();

        Self {
            categories,
            counts,
            percentages,
        }
    }

    /// Render as ASCII bar chart (no trueno-viz dependency)
    pub fn to_ascii(&self, width: usize) -> String {
        let max_count = self.counts.iter().max().copied().unwrap_or(1) as f32;
        let max_label_len = self.categories.iter().map(|s| s.len()).max().unwrap_or(15);

        let mut output = String::new();
        for (i, category) in self.categories.iter().enumerate() {
            let count = self.counts[i];
            let pct = self.percentages[i];
            let bar_len =
                ((count as f32 / max_count) * (width - max_label_len - 15) as f32) as usize;
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
}

/// Confidence distribution for histogram
pub struct ConfidenceDistribution {
    pub values: Vec<f32>,
    pub min: f32,
    pub max: f32,
    pub mean: f32,
}

impl ConfidenceDistribution {
    /// Create from training examples
    pub fn from_examples(examples: &[TrainingExample]) -> Self {
        let values: Vec<f32> = examples.iter().map(|e| e.confidence).collect();
        let min = values.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mean = values.iter().sum::<f32>() / values.len() as f32;

        Self {
            values,
            min,
            max,
            mean,
        }
    }

    /// Render as ASCII histogram (no trueno-viz dependency)
    pub fn to_ascii(&self, bins: usize, height: usize) -> String {
        let range = self.max - self.min;
        let bin_width = range / bins as f32;

        // Count values in each bin
        let mut bin_counts = vec![0u32; bins];
        for &v in &self.values {
            let bin_idx = ((v - self.min) / bin_width).floor() as usize;
            let bin_idx = bin_idx.min(bins - 1);
            bin_counts[bin_idx] += 1;
        }

        let max_count = *bin_counts.iter().max().unwrap_or(&1) as f32;

        // Build histogram
        let mut output = String::new();
        for row in (0..height).rev() {
            let threshold = (row as f32 / height as f32) * max_count;
            output.push('│');
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
        output.push('└');
        output.push_str(&"──".repeat(bins));
        output.push('\n');
        output.push_str(&format!(
            " {:.2}{}  {:.2}\n",
            self.min,
            " ".repeat(bins * 2 - 10),
            self.max
        ));
        output.push_str(&format!(" Mean: {:.2}\n", self.mean));

        output
    }
}

/// Cross-repository defect matrix for heatmap
#[derive(Default)]
pub struct CrossRepoMatrix {
    pub repos: Vec<String>,
    pub categories: Vec<String>,
    pub matrix: Vec<Vec<f32>>, // [category][repo] = percentage
}

impl CrossRepoMatrix {
    /// Create from multiple repo analyses
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a repository's defect distribution
    pub fn add_repo(&mut self, repo_name: &str, dist: &DefectDistribution) {
        self.repos.push(repo_name.to_string());

        // Ensure all categories exist
        for cat in &dist.categories {
            if !self.categories.contains(cat) {
                self.categories.push(cat.clone());
                self.matrix.push(vec![0.0; self.repos.len() - 1]);
            }
        }

        // Add column for new repo
        for row in &mut self.matrix {
            row.push(0.0);
        }

        // Fill in percentages
        let repo_idx = self.repos.len() - 1;
        for (i, cat) in dist.categories.iter().enumerate() {
            if let Some(cat_idx) = self.categories.iter().position(|c| c == cat) {
                self.matrix[cat_idx][repo_idx] = dist.percentages[i];
            }
        }
    }

    /// Render as ASCII heatmap
    pub fn to_ascii(&self) -> String {
        let max_cat_len = self.categories.iter().map(|s| s.len()).max().unwrap_or(15);
        let col_width = 8;

        let mut output = String::new();

        // Header
        output.push_str(&" ".repeat(max_cat_len + 1));
        for repo in &self.repos {
            output.push_str(&format!(
                "{:>width$}",
                &repo[..repo.len().min(col_width)],
                width = col_width
            ));
        }
        output.push('\n');

        // Rows
        for (cat_idx, category) in self.categories.iter().enumerate() {
            output.push_str(&format!("{:width$} ", category, width = max_cat_len));
            for &pct in &self.matrix[cat_idx] {
                let block = match pct {
                    p if p >= 40.0 => "███████",
                    p if p >= 20.0 => "█████░░",
                    p if p >= 10.0 => "███░░░░",
                    p if p >= 5.0 => "██░░░░░",
                    p if p > 0.0 => "█░░░░░░",
                    _ => "░░░░░░░",
                };
                output.push_str(&format!(" {}", block));
            }
            output.push('\n');
        }

        output
    }
}

#[cfg(feature = "viz")]
/// Render defect distribution using trueno-viz
pub fn render_distribution_heatmap(matrix: &CrossRepoMatrix) -> Result<(), anyhow::Error> {
    // Flatten matrix data for heatmap
    let rows = matrix.categories.len();
    let cols = matrix.repos.len();
    let data: Vec<f32> = matrix.matrix.iter().flatten().copied().collect();

    let heatmap = Heatmap::new().data(&data, rows, cols).build()?;

    let fb = heatmap.to_framebuffer()?;
    TerminalEncoder::new()
        .mode(TerminalMode::UnicodeHalfBlock)
        .width(80)
        .print(&fb);

    Ok(())
}

#[cfg(feature = "viz")]
/// Render confidence histogram using trueno-viz
pub fn render_confidence_histogram(dist: &ConfidenceDistribution) -> Result<(), anyhow::Error> {
    let histogram = Histogram::new()
        .data(&dist.values)
        .bins(BinStrategy::Fixed(20))
        .build()?;

    let fb = histogram.to_framebuffer()?;
    TerminalEncoder::new()
        .mode(TerminalMode::UnicodeHalfBlock)
        .width(80)
        .print(&fb);

    Ok(())
}

/// Print defect summary report (ASCII, no dependencies)
pub fn print_summary_report(
    repo_name: &str,
    dist: &DefectDistribution,
    confidence: &ConfidenceDistribution,
) {
    println!();
    println!("Organizational Intelligence Report: {}", repo_name);
    println!("{}", "═".repeat(50));
    println!();
    println!(
        "Defect Distribution ({} commits analyzed)",
        dist.counts.iter().sum::<u32>()
    );
    println!("{}", "─".repeat(40));
    println!("{}", dist.to_ascii(50));
    println!();
    println!("Confidence Distribution");
    println!("{}", "─".repeat(40));
    println!("{}", confidence.to_ascii(20, 8));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::classifier::DefectCategory;

    fn make_example(label: DefectCategory, confidence: f32) -> TrainingExample {
        TrainingExample {
            message: "test fix".to_string(),
            label,
            confidence,
            commit_hash: "abc123".to_string(),
            author: "test".to_string(),
            timestamp: 0,
            lines_added: 10,
            lines_removed: 5,
            files_changed: 1,
        }
    }

    #[test]
    fn test_defect_distribution() {
        let examples = vec![
            make_example(DefectCategory::ASTTransform, 0.9),
            make_example(DefectCategory::ASTTransform, 0.85),
            make_example(DefectCategory::OwnershipBorrow, 0.8),
        ];

        let dist = DefectDistribution::from_examples(&examples);
        assert_eq!(dist.categories.len(), 2);
        assert_eq!(dist.counts[0], 2); // ASTTransform has 2

        let ascii = dist.to_ascii(40);
        assert!(ascii.contains("ASTTransform"));
    }

    #[test]
    fn test_confidence_distribution() {
        let examples = vec![
            make_example(DefectCategory::ASTTransform, 0.7),
            make_example(DefectCategory::ASTTransform, 0.9),
        ];

        let dist = ConfidenceDistribution::from_examples(&examples);
        assert!((dist.mean - 0.8).abs() < 0.01);
        assert!((dist.min - 0.7).abs() < 0.01);
        assert!((dist.max - 0.9).abs() < 0.01);
    }

    #[test]
    fn test_cross_repo_matrix() {
        let mut matrix = CrossRepoMatrix::new();

        let dist1 = DefectDistribution {
            categories: vec!["ASTTransform".to_string(), "Security".to_string()],
            counts: vec![50, 10],
            percentages: vec![50.0, 10.0],
        };

        let dist2 = DefectDistribution {
            categories: vec!["ASTTransform".to_string(), "Memory".to_string()],
            counts: vec![40, 20],
            percentages: vec![40.0, 20.0],
        };

        matrix.add_repo("depyler", &dist1);
        matrix.add_repo("bashrs", &dist2);

        assert_eq!(matrix.repos.len(), 2);
        assert_eq!(matrix.categories.len(), 3); // ASTTransform, Security, Memory

        let ascii = matrix.to_ascii();
        assert!(ascii.contains("depyler"));
        assert!(ascii.contains("bashrs"));
    }
}
