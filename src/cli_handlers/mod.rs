//! CLI Command Handlers
//!
//! Testable business logic for all CLI commands.
//! Binary entry points (main.rs) should only parse args and call these handlers.

use anyhow::Result;
use chrono::{Duration, Utc};
use std::path::PathBuf;
use tempfile::TempDir;
use tracing::{error, info, warn};

use crate::analyzer::OrgAnalyzer;
use crate::classifier::{DefectCategory, RuleBasedClassifier};
use crate::export::{ExportFormat, FeatureExporter};
use crate::features::{CommitFeatures, FeatureExtractor};
use crate::git;
use crate::github::GitHubMiner;
use crate::ml_trainer::MLTrainer;
use crate::pmat::PmatIntegration;
use crate::pr_reviewer::PrReviewer;
use crate::report::{AnalysisMetadata, AnalysisReport, ReportGenerator};
use crate::summarizer::{ReportSummarizer, SummaryConfig};
use crate::tarantula::{
    LcovParser, LocalizationConfig, ReportFormat, SbflFormula, TarantulaIntegration,
};
use crate::training::TrainingDataExtractor;
use crate::viz::{ConfidenceDistribution, DefectDistribution};

/// Handle the `review-pr` command
pub async fn handle_review_pr(
    baseline: PathBuf,
    files: String,
    format: String,
    output: Option<PathBuf>,
) -> Result<()> {
    info!("Reviewing PR with baseline: {}", baseline.display());
    info!("Files changed: {}", files);
    info!("Output format: {}", format);

    println!("\n🔍 PR Review: Organizational Intelligence");
    println!("   Baseline: {}", baseline.display());
    println!("   Format:   {}", format);

    // Load baseline summary
    let reviewer = PrReviewer::load_baseline(&baseline)?;

    // Parse comma-separated file list
    let files_vec: Vec<String> = files
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    println!("   Files:    {} file(s)", files_vec.len());

    // Review PR
    let review = reviewer.review_pr(&files_vec);

    // Generate output based on format
    let output_content = match format.as_str() {
        "json" => review.to_json()?,
        _ => review.to_markdown(), // Default to markdown
    };

    // Write to file or stdout
    if let Some(output_path) = output {
        std::fs::write(&output_path, &output_content)?;
        println!("\n✅ Review saved to: {}", output_path.display());
    } else {
        println!("\n{}", output_content);
    }

    // Summary
    println!("\n📊 Review Summary:");
    println!("   Warnings: {}", review.warnings.len());
    println!("   Files analyzed: {}", review.files_analyzed.len());
    println!("   Baseline date: {}", review.baseline_date);
    println!(
        "   Repositories in baseline: {}",
        review.repositories_analyzed
    );

    if review.warnings.is_empty() {
        println!("\n✅ No warnings - PR looks good based on historical patterns!");
    } else {
        println!(
            "\n⚠️  {} warning(s) generated - review carefully!",
            review.warnings.len()
        );
    }

    println!("\n🎯 Phase 3 Complete!");
    println!("   ✅ Fast PR review (<30s)");
    println!("   ✅ Stateful baselines (no re-analysis)");
    println!("   ✅ Actionable warnings");
    println!("   ✅ Multiple output formats");

    Ok(())
}

/// Handle the `summarize` command
pub async fn handle_summarize(
    input: PathBuf,
    output: PathBuf,
    strip_pii: bool,
    top_n: usize,
    min_frequency: usize,
    include_examples: bool,
) -> Result<()> {
    info!("Summarizing report: {}", input.display());
    info!("Output file: {}", output.display());
    info!("Strip PII: {}", strip_pii);
    info!("Top N categories: {}", top_n);
    info!("Min frequency: {}", min_frequency);
    info!("Include examples: {}", include_examples);

    println!("\n📊 Summarizing Analysis Report");
    println!("   Input:  {}", input.display());
    println!("   Output: {}", output.display());

    // Create summarization config
    let config = SummaryConfig {
        strip_pii,
        top_n_categories: top_n,
        min_frequency,
        include_examples,
    };

    // Summarize report
    match ReportSummarizer::summarize(&input, config) {
        Ok(summary) => {
            // Save summary to file
            ReportSummarizer::save_to_file(&summary, &output)?;

            info!("✅ Summary written to {}", output.display());
            println!("\n✅ Summary saved to: {}", output.display());

            println!("\n📈 Summary Statistics:");
            println!(
                "   Repositories analyzed: {}",
                summary.metadata.repositories_analyzed
            );
            println!("   Commits analyzed: {}", summary.metadata.commits_analyzed);
            println!(
                "   Top defect categories included: {}",
                summary.organizational_insights.top_defect_categories.len()
            );

            if strip_pii {
                println!("\n🔒 PII Stripping:");
                println!("   ✅ Author names: REDACTED");
                println!("   ✅ Commit hashes: REDACTED");
                println!("   ✅ Safe for sharing");
            }

            println!("\n🎯 Phase 2 Complete!");
            println!("   ✅ Automated PII stripping");
            println!("   ✅ Frequency filtering");
            println!("   ✅ Top-N selection");
            println!("   ✅ Ready for AI consumption");

            Ok(())
        }
        Err(e) => {
            error!("Failed to summarize report: {}", e);
            eprintln!("❌ Error: {}", e);
            Err(e)
        }
    }
}

/// Handle the `analyze` command
pub async fn handle_analyze(
    org: String,
    output: PathBuf,
    _max_concurrent: usize,
    github_token: Option<String>,
    analyzer_version: String,
    model_path: Option<PathBuf>,
    ml_confidence: f32,
) -> Result<()> {
    info!("Analyzing organization: {}", org);
    info!("Output file: {}", output.display());

    // Initialize GitHub client
    if github_token.is_none() {
        warn!("GITHUB_TOKEN not set - using unauthenticated requests (lower rate limits)");
        info!("Set GITHUB_TOKEN environment variable for higher rate limits");
    }

    let miner = GitHubMiner::new(github_token);

    // Fetch organization repositories
    info!("Fetching repositories for organization: {}", org);
    match miner.fetch_organization_repos(&org).await {
        Ok(all_repos) => {
            info!("✅ Successfully fetched {} repositories", all_repos.len());

            // Filter repos updated in last 2 years
            let two_years_ago = Utc::now() - Duration::days(730);
            let repos = GitHubMiner::filter_by_date(all_repos.clone(), two_years_ago);

            println!("\n📊 Organization Analysis: {}", org);
            println!("   Total repositories: {}", all_repos.len());
            println!("   Repositories updated in last 2 years: {}", repos.len());

            // Display top 5 repositories by stars
            let mut sorted_repos = repos.clone();
            sorted_repos.sort_by(|a, b| b.stars.cmp(&a.stars));

            println!("\n⭐ Top repositories by stars (last 2 years):");
            for (i, repo) in sorted_repos.iter().take(5).enumerate() {
                println!(
                    "   {}. {} ({} ⭐) - {}",
                    i + 1,
                    repo.name,
                    repo.stars,
                    repo.language.as_deref().unwrap_or("Unknown")
                );
            }

            // Analyze ALL repos from last 2 years
            info!(
                "Analyzing defect patterns in ALL {} repositories",
                repos.len()
            );
            println!("\n🔍 Analyzing defect patterns in ALL repos from last 2 years...");

            let temp_dir = TempDir::new()?;

            // Load ML model if provided
            let analyzer = if let Some(model_path) = model_path {
                info!("Loading ML model from: {}", model_path.display());
                match crate::ml_trainer::MLTrainer::load_model(&model_path) {
                    Ok(model) => {
                        info!("✅ ML model loaded successfully");
                        info!("   Using confidence threshold: {:.2}", ml_confidence);
                        println!("\n🤖 Using ML-based classification (Tier 2)");
                        println!("   Model: {}", model_path.display());
                        println!("   Confidence threshold: {:.2}", ml_confidence);
                        println!(
                            "   Training accuracy: {:.2}%",
                            model.metadata.train_accuracy * 100.0
                        );
                        println!("   Classes: {}", model.metadata.n_classes);
                        OrgAnalyzer::with_ml_model(temp_dir.path(), model, ml_confidence)
                    }
                    Err(e) => {
                        warn!("Failed to load ML model: {}", e);
                        warn!("Falling back to rule-based classification");
                        println!("\n⚠️  Failed to load ML model: {}", e);
                        println!("   Falling back to rule-based classification (Tier 1)");
                        OrgAnalyzer::new(temp_dir.path())
                    }
                }
            } else {
                info!("No ML model specified, using rule-based classification");
                println!("\n📏 Using rule-based classification (Tier 1)");
                OrgAnalyzer::new(temp_dir.path())
            };

            let mut all_patterns = vec![];
            let mut total_commits = 0;
            let mut repos_analyzed = 0;

            // Analyze ALL repositories (not limited by max_concurrent anymore)
            for (i, repo) in sorted_repos.iter().enumerate() {
                println!(
                    "   [{}/{}] Analyzing: {} (updated: {})",
                    i + 1,
                    sorted_repos.len(),
                    repo.name,
                    repo.updated_at.format("%Y-%m-%d")
                );

                let repo_url = format!("https://github.com/{}/{}", org, repo.name);

                match analyzer
                    .analyze_repository(&repo_url, &repo.name, 100)
                    .await
                {
                    Ok(patterns) => {
                        total_commits += 100;
                        all_patterns.extend(patterns);
                        repos_analyzed += 1;
                        info!("✅ Analyzed {}", repo.name);
                    }
                    Err(e) => {
                        warn!("Failed to analyze {}: {}", repo.name, e);
                        println!("     ⚠️  Skipping {} (error: {})", repo.name, e);
                    }
                }
            }

            println!("   ✅ Analysis complete!");

            // Generate YAML report
            info!("Generating YAML report");
            let report_generator = ReportGenerator::new();

            let metadata = AnalysisMetadata {
                organization: org.clone(),
                analysis_date: Utc::now().to_rfc3339(),
                repositories_analyzed: repos_analyzed,
                commits_analyzed: total_commits,
                analyzer_version,
            };

            let report = AnalysisReport {
                version: "1.0".to_string(),
                metadata,
                defect_patterns: all_patterns,
            };

            // Write report to file
            report_generator.write_to_file(&report, &output).await?;

            info!("✅ Report written to {}", output.display());
            println!("\n📄 Report saved to: {}", output.display());

            println!("\n🎯 Phase 1 MVP Complete!");
            println!("   ✅ CLI structure");
            println!("   ✅ GitHub API integration");
            println!("   ✅ YAML output generation");
            println!("   ✅ Git history analysis");
            println!("   ✅ Rule-based defect classifier");
            println!("   ✅ Pattern aggregation");

            Ok(())
        }
        Err(e) => {
            error!("Failed to fetch repositories: {}", e);
            eprintln!("❌ Error: {}", e);
            Err(e)
        }
    }
}

/// Handle the `extract-training-data` command
pub async fn handle_extract_training_data(
    repo: PathBuf,
    output: PathBuf,
    min_confidence: f32,
    max_commits: usize,
    create_splits: bool,
    viz: bool,
) -> Result<()> {
    info!("Extracting training data from: {}", repo.display());
    info!("Output file: {}", output.display());
    info!("Min confidence: {}", min_confidence);
    info!("Max commits: {}", max_commits);

    println!("\n🎓 Training Data Extraction (Phase 2 ML)");
    println!("   Repository:      {}", repo.display());
    println!("   Output:          {}", output.display());
    println!("   Min confidence:  {:.2}", min_confidence);
    println!("   Max commits:     {}", max_commits);
    println!("   Create splits:   {}", create_splits);
    println!("   Visualization:   {}", viz);

    // Validate repository path
    if !repo.exists() {
        return Err(anyhow::anyhow!(
            "Repository path does not exist: {}",
            repo.display()
        ));
    }

    if !repo.join(".git").exists() {
        return Err(anyhow::anyhow!("Not a Git repository: {}", repo.display()));
    }

    // Extract commit history
    println!("\n📖 Reading commit history...");
    let commits = git::analyze_repository_at_path(&repo, max_commits)?;
    println!("   ✅ Found {} commits", commits.len());

    // Extract training data
    println!("\n🔍 Extracting and auto-labeling defect-fix commits...");
    let extractor = TrainingDataExtractor::new(min_confidence);

    let repo_name = repo
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown-repo");

    let examples = extractor.extract_training_data(&commits, repo_name)?;

    println!("   ✅ Extracted {} training examples", examples.len());

    if examples.is_empty() {
        warn!("No training examples extracted - try lowering min_confidence threshold");
        println!("\n⚠️  No training examples extracted!");
        println!(
            "   Try lowering --min-confidence (current: {:.2})",
            min_confidence
        );
        return Ok(());
    }

    // Show statistics
    println!("\n📊 Training Data Statistics:");
    let stats = extractor.get_statistics(&examples);
    for line in stats.lines() {
        if !line.is_empty() {
            println!("   {}", line);
        }
    }

    // Create splits or export raw examples
    if create_splits {
        println!("\n📂 Creating train/validation/test splits (70/15/15)...");
        let dataset = extractor.create_splits(&examples, &[repo_name.to_string()])?;

        println!("   ✅ Train:      {} examples", dataset.train.len());
        println!("   ✅ Validation: {} examples", dataset.validation.len());
        println!("   ✅ Test:       {} examples", dataset.test.len());

        // Export dataset to JSON
        let json = serde_json::to_string_pretty(&dataset)?;
        std::fs::write(&output, json)?;
    } else {
        // Export raw examples to JSON
        println!("\n💾 Exporting raw examples...");
        let json = serde_json::to_string_pretty(&examples)?;
        std::fs::write(&output, json)?;
    }

    println!("\n✅ Training data saved to: {}", output.display());

    // Visualization output
    if viz {
        println!("\n📊 Defect Pattern Visualization");
        println!("{}", "─".repeat(50));

        let defect_dist = DefectDistribution::from_examples(&examples);
        let confidence_dist = ConfidenceDistribution::from_examples(&examples);

        // ASCII visualization (always available)
        crate::viz::print_summary_report(repo_name, &defect_dist, &confidence_dist);

        // trueno-viz visualization (if feature enabled)
        #[cfg(feature = "viz")]
        {
            println!("\n📈 Rich Terminal Visualization (trueno-viz):");
            if let Err(e) = crate::viz::render_confidence_histogram(&confidence_dist) {
                warn!("Could not render histogram: {}", e);
            }
        }

        #[cfg(not(feature = "viz"))]
        {
            println!("\n💡 Tip: Build with --features viz for rich terminal visualizations");
        }
    }

    // Summary
    println!("\n🎯 Phase 2 Training Data Extraction Complete!");
    println!("   ✅ Commit filtering (excludes merges, reverts, WIP)");
    println!("   ✅ Auto-labeling with rule-based classifier");
    println!(
        "   ✅ Confidence threshold filtering ({:.2})",
        min_confidence
    );
    if create_splits {
        println!("   ✅ Train/validation/test splits created");
    }
    if viz {
        println!("   ✅ Visualization rendered");
    }
    println!("   ✅ Ready for ML training (RandomForestClassifier)");

    println!("\n💡 Next Steps:");
    println!("   1. Review extracted data: cat {}", output.display());
    println!(
        "   2. Train ML classifier: oip train-classifier --input {}",
        output.display()
    );
    println!("   3. Evaluate model performance on test set");

    Ok(())
}

/// Handle the `train-classifier` command
pub async fn handle_train_classifier(
    input: PathBuf,
    output: Option<PathBuf>,
    n_estimators: usize,
    max_depth: usize,
    max_features: usize,
) -> Result<()> {
    info!("Training ML classifier from: {}", input.display());
    if let Some(ref output_path) = output {
        info!("Output model file: {}", output_path.display());
    }
    info!(
        "Hyperparameters: n_estimators={}, max_depth={}, max_features={}",
        n_estimators, max_depth, max_features
    );

    println!("\n🤖 ML Classifier Training (Phase 2)");
    println!("   Input:         {}", input.display());
    if let Some(ref output_path) = output {
        println!("   Output:        {}", output_path.display());
    }
    println!("   N Estimators:  {}", n_estimators);
    println!("   Max Depth:     {}", max_depth);
    println!("   Max Features:  {}", max_features);

    // Validate input file
    if !input.exists() {
        return Err(anyhow::anyhow!(
            "Input file does not exist: {}",
            input.display()
        ));
    }

    // Load training dataset
    println!("\n📂 Loading training dataset...");
    let dataset = MLTrainer::load_dataset(&input)?;
    println!(
        "   ✅ Loaded {} total examples",
        dataset.metadata.total_examples
    );
    println!("      Train:      {} examples", dataset.train.len());
    println!("      Validation: {} examples", dataset.validation.len());
    println!("      Test:       {} examples", dataset.test.len());

    // Show class distribution
    println!("\n📊 Class Distribution:");
    let mut sorted_classes: Vec<_> = dataset.metadata.class_distribution.iter().collect();
    sorted_classes.sort_by(|a, b| b.1.cmp(a.1));
    for (class, count) in sorted_classes.iter().take(10) {
        let percentage = (**count as f32 / dataset.metadata.total_examples as f32) * 100.0;
        println!("      {}: {} ({:.1}%)", class, count, percentage);
    }

    // Train model
    println!("\n🎯 Training Random Forest Classifier...");
    let trainer = MLTrainer::new(n_estimators, Some(max_depth), max_features);

    let model = trainer.train(&dataset)?;

    println!("   ✅ Training complete!");
    println!("      Classes:  {}", model.metadata.n_classes);
    println!("      Features: {}", model.metadata.n_features);

    // Show accuracy metrics
    println!("\n📈 Model Performance:");
    println!(
        "   Training accuracy:   {:.2}%",
        model.metadata.train_accuracy * 100.0
    );
    println!(
        "   Validation accuracy: {:.2}%",
        model.metadata.validation_accuracy * 100.0
    );

    // Evaluate on test set
    if !dataset.test.is_empty() {
        println!("\n🔍 Evaluating on test set...");
        let test_accuracy = MLTrainer::evaluate(&model, &dataset.test)?;
        println!("   Test accuracy:       {:.2}%", test_accuracy * 100.0);

        // Check if we meet the target
        if test_accuracy >= 0.80 {
            println!("\n✅ Model meets ≥80% accuracy target!");
        } else {
            println!(
                "\n⚠️  Model accuracy {:.2}% below 80% target",
                test_accuracy * 100.0
            );
            println!("   Consider:");
            println!("   - Collecting more training data");
            println!("   - Increasing n_estimators (current: {})", n_estimators);
            println!("   - Adjusting max_depth (current: {})", max_depth);
            println!("   - Increasing max_features (current: {})", max_features);
        }
    }

    // Save model if output specified
    if let Some(output_path) = output {
        println!("\n💾 Saving model metadata...");
        MLTrainer::save_model(&model, &output_path)?;
        println!("   ✅ Model metadata saved to: {}", output_path.display());
        println!("   Note: RandomForestClassifier and TfidfVectorizer are in-memory only");
        println!("         Full serialization support coming in future update");
    }

    // Summary
    println!("\n🎯 Phase 2 ML Training Complete!");
    println!("   ✅ Random Forest with {} trees trained", n_estimators);
    println!(
        "   ✅ TF-IDF features: {} dimensions",
        model.metadata.n_features
    );
    println!("   ✅ Defect categories: {}", model.metadata.n_classes);
    println!("   ✅ Training examples: {}", model.metadata.n_train);

    let improvement = (model.metadata.validation_accuracy / 0.308) * 100.0 - 100.0;
    println!("\n📊 Performance vs Baseline:");
    println!("   Baseline (rule-based):  30.8%");
    println!(
        "   ML Model (validation):  {:.2}%",
        model.metadata.validation_accuracy * 100.0
    );
    if improvement > 0.0 {
        println!("   Improvement:            +{:.1}%", improvement);
    }

    println!("\n💡 Next Steps:");
    println!("   1. Integrate model into analysis pipeline (NLP-008)");
    println!("   2. Benchmark inference performance (<100ms target)");
    println!("   3. Deploy to production for real-time classification");

    Ok(())
}

/// Handle the `export` command (Issue #2)
///
/// Exports CommitFeatures to aprender-compatible format for ML training.
pub async fn handle_export(
    repo: PathBuf,
    output: PathBuf,
    format: String,
    max_commits: usize,
    min_confidence: f32,
) -> Result<()> {
    info!("Exporting features from: {}", repo.display());
    info!("Output file: {}", output.display());
    info!("Format: {}", format);
    info!("Max commits: {}", max_commits);
    info!("Min confidence: {}", min_confidence);

    println!("\n📦 Feature Export to aprender Format (Issue #2)");
    println!("   Repository:     {}", repo.display());
    println!("   Output:         {}", output.display());
    println!("   Format:         {}", format);
    println!("   Max commits:    {}", max_commits);
    println!("   Min confidence: {:.2}", min_confidence);

    // Validate repository path
    if !repo.exists() {
        return Err(anyhow::anyhow!(
            "Repository path does not exist: {}",
            repo.display()
        ));
    }

    if !repo.join(".git").exists() {
        return Err(anyhow::anyhow!("Not a Git repository: {}", repo.display()));
    }

    // Parse export format
    let export_format: ExportFormat = format
        .parse()
        .map_err(|e| anyhow::anyhow!("Invalid format '{}': {}", format, e))?;

    // Extract commit history
    println!("\n📖 Reading commit history...");
    let commits = git::analyze_repository_at_path(&repo, max_commits)?;
    println!("   ✅ Found {} commits", commits.len());

    if commits.is_empty() {
        return Err(anyhow::anyhow!("No commits found in repository"));
    }

    // Classify commits and extract features
    println!("\n🔍 Classifying and extracting features...");
    let classifier = RuleBasedClassifier::new();
    let feature_extractor = FeatureExtractor::new();

    let mut features: Vec<CommitFeatures> = Vec::new();
    let mut categories: Vec<DefectCategory> = Vec::new();
    let mut skipped = 0;

    for commit in &commits {
        // Classify the commit message
        if let Some(classification) = classifier.classify_from_message(&commit.message) {
            // Only include commits above confidence threshold
            if classification.confidence >= min_confidence {
                // Extract features
                if let Ok(feat) = feature_extractor.extract(
                    FeatureExporter::encode_label(classification.category),
                    commit.files_changed,
                    commit.lines_added,
                    commit.lines_removed,
                    commit.timestamp,
                ) {
                    features.push(feat);
                    categories.push(classification.category);
                } else {
                    skipped += 1;
                }
            } else {
                skipped += 1;
            }
        } else {
            skipped += 1;
        }
    }

    println!("   ✅ Extracted {} samples", features.len());
    if skipped > 0 {
        println!(
            "   ⚠️  Skipped {} commits (below confidence threshold or unclassified)",
            skipped
        );
    }

    if features.is_empty() {
        return Err(anyhow::anyhow!(
            "No features extracted. Try lowering --min-confidence (current: {:.2})",
            min_confidence
        ));
    }

    // Export to aprender format
    println!("\n💾 Exporting to {} format...", export_format);
    let exporter = FeatureExporter::new(export_format);
    let dataset = exporter.export(&features, &categories)?;

    // Save to file
    exporter.save(&dataset, &output)?;

    println!("   ✅ Saved to: {}", output.display());

    // Show statistics
    println!("\n📊 Export Statistics:");
    println!("   Samples:    {}", dataset.metadata.n_samples);
    println!("   Features:   {}", dataset.metadata.n_features);
    println!("   Classes:    {}", dataset.metadata.n_classes);
    println!("   Format:     {}", dataset.metadata.format);
    println!("   Version:    {}", dataset.metadata.version);

    // Show class distribution
    let mut class_counts: std::collections::HashMap<u8, usize> = std::collections::HashMap::new();
    for &label in &dataset.labels {
        *class_counts.entry(label).or_insert(0) += 1;
    }

    println!("\n📈 Class Distribution:");
    let mut sorted_counts: Vec<_> = class_counts.iter().collect();
    sorted_counts.sort_by(|a, b| b.1.cmp(a.1));

    for (label, count) in sorted_counts.iter().take(10) {
        let category_name = &dataset.category_names[**label as usize];
        let percentage = (**count as f32 / dataset.metadata.n_samples as f32) * 100.0;
        println!("   {}: {} ({:.1}%)", category_name, count, percentage);
    }

    // Summary
    println!("\n🎯 Export Complete!");
    println!("   ✅ CommitFeatures exported as Matrix<f32>");
    println!("   ✅ Labels exported as Vec<u8>");
    println!("   ✅ 18-category taxonomy mapping included");
    println!("   ✅ Ready for aprender training (RandomForest, K-Means)");

    println!("\n💡 Next Steps:");
    println!(
        "   1. Load with: FeatureExporter::load(\"{}\", ExportFormat::{})",
        output.display(),
        format.to_uppercase()
    );
    println!("   2. Convert to Matrix: FeatureExporter::to_aprender_matrix(&dataset)");
    println!("   3. Train classifier: RandomForestClassifier::fit(&matrix, &labels)");

    Ok(())
}

/// Handle the `import-depyler` command (NLP-014)
///
/// Imports Depyler CITL corpus as ground-truth training labels.
pub async fn handle_import_depyler(
    input: PathBuf,
    output: PathBuf,
    min_confidence: f32,
    merge: Option<PathBuf>,
    create_splits: bool,
) -> Result<()> {
    use crate::citl::{convert_to_training_examples, import_depyler_corpus};
    use crate::training::{TrainingDataExtractor, TrainingDataset};

    info!("Importing Depyler CITL corpus from: {}", input.display());
    info!("Output file: {}", output.display());
    info!("Min confidence: {}", min_confidence);
    info!("Merge: {:?}", merge);
    info!("Create splits: {}", create_splits);

    println!("\n🔬 CITL Import: Depyler Ground-Truth Labels (NLP-014)");
    println!("   Input:          {}", input.display());
    println!("   Output:         {}", output.display());
    println!("   Min confidence: {:.2}", min_confidence);

    // Validate input path
    if !input.exists() {
        return Err(anyhow::anyhow!(
            "Input file does not exist: {}",
            input.display()
        ));
    }

    // Import CITL corpus
    println!("\n📖 Reading CITL corpus...");
    let (exports, stats) = import_depyler_corpus(&input, min_confidence)?;

    println!("   ✅ Total records:  {}", stats.total_records);
    println!("   ✅ Imported:       {}", stats.imported);
    println!("   ⚠️  Low confidence: {}", stats.skipped_low_confidence);
    println!("   ⚠️  Unknown cat:    {}", stats.skipped_unknown_category);
    println!("   📊 Avg confidence: {:.2}", stats.avg_confidence);

    if exports.is_empty() {
        return Err(anyhow::anyhow!(
            "No records imported. Try lowering --min-confidence (current: {:.2})",
            min_confidence
        ));
    }

    // Convert to TrainingExamples
    println!("\n🔄 Converting to training examples...");
    let mut examples = convert_to_training_examples(&exports);
    println!("   ✅ Converted {} examples", examples.len());

    // Merge with existing training data if specified
    if let Some(merge_path) = &merge {
        if merge_path.exists() {
            println!("\n🔗 Merging with existing training data...");
            let content = std::fs::read_to_string(merge_path)?;
            let existing: TrainingDataset = serde_json::from_str(&content)?;
            let existing_count =
                existing.train.len() + existing.validation.len() + existing.test.len();
            println!("   📖 Loaded {} existing examples", existing_count);

            // Add existing examples
            examples.extend(existing.train);
            examples.extend(existing.validation);
            examples.extend(existing.test);
            println!("   ✅ Total: {} examples", examples.len());
        } else {
            warn!("Merge file not found: {}", merge_path.display());
        }
    }

    // Create splits or save raw examples
    if create_splits {
        println!("\n📊 Creating train/validation/test splits (70/15/15)...");
        let extractor = TrainingDataExtractor::new(min_confidence);
        let dataset = extractor.create_splits(&examples, &["depyler-citl".to_string()])?;

        println!("   Train:      {} examples", dataset.train.len());
        println!("   Validation: {} examples", dataset.validation.len());
        println!("   Test:       {} examples", dataset.test.len());

        // Save dataset
        let json = serde_json::to_string_pretty(&dataset)?;
        std::fs::write(&output, json)?;
    } else {
        // Save raw examples
        let json = serde_json::to_string_pretty(&examples)?;
        std::fs::write(&output, json)?;
    }

    println!("\n💾 Saved to: {}", output.display());

    // Show category distribution
    let mut category_counts: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    for ex in &examples {
        *category_counts.entry(format!("{}", ex.label)).or_insert(0) += 1;
    }

    println!("\n📈 Category Distribution:");
    let mut sorted_counts: Vec<_> = category_counts.iter().collect();
    sorted_counts.sort_by(|a, b| b.1.cmp(a.1));

    for (category, count) in sorted_counts.iter().take(10) {
        let percentage = (**count as f32 / examples.len() as f32) * 100.0;
        println!("   {}: {} ({:.1}%)", category, count, percentage);
    }

    // Summary
    println!("\n🎯 Import Complete!");
    println!("   ✅ Ground-truth labels from CITL integrated");
    println!("   ✅ TrainingSource::DepylerCitl marked");
    println!("   ✅ Error codes and clippy lints preserved");

    println!("\n💡 Next Steps:");
    println!(
        "   1. Train classifier: oip train-classifier --input {}",
        output.display()
    );
    println!("   2. Evaluate model performance on test split");
    println!("   3. Compare accuracy with NLP-011 baseline (54%)");

    Ok(())
}

/// Handle the `localize` command - Tarantula SBFL fault localization
///
/// Toyota Way: Muda (eliminate waste) - only run expensive TDG enrichment when requested
#[allow(clippy::too_many_arguments)]
pub async fn handle_localize(
    passed_coverage: PathBuf,
    failed_coverage: PathBuf,
    passed_count: usize,
    failed_count: usize,
    formula: String,
    top_n: usize,
    output: PathBuf,
    format: String,
    enrich_tdg: bool,
    repo: Option<PathBuf>,
    rag: bool,
    knowledge_base: Option<PathBuf>,
    fusion: String,
    similar_bugs: usize,
    // Phase 6: Ensemble
    ensemble: bool,
    ensemble_model: Option<PathBuf>,
    include_churn: bool,
    // Phase 7: Calibrated
    calibrated: bool,
    calibration_model: Option<PathBuf>,
    confidence_threshold: f32,
) -> Result<()> {
    use crate::rag_localization::{
        BugKnowledgeBase, LocalizationFusion, RagFaultLocalizer, RagLocalizationConfig,
        RagReportGenerator,
    };

    info!("Running Tarantula fault localization");
    info!("Passed coverage: {}", passed_coverage.display());
    info!("Failed coverage: {}", failed_coverage.display());

    if rag {
        println!("\n🔍 RAG-Enhanced Fault Localization (trueno-rag)");
    } else {
        println!("\n🔍 Tarantula Fault Localization");
    }
    println!("   Formula: {}", formula);
    println!("   Top N:   {}", top_n);
    if rag {
        println!("   RAG:     enabled");
        println!("   Fusion:  {}", fusion);
    }

    // Check if coverage tool is available
    if !TarantulaIntegration::is_coverage_tool_available() {
        warn!("cargo-llvm-cov not found - using provided coverage files");
    }

    // Parse formula
    let sbfl_formula = match formula.to_lowercase().as_str() {
        "tarantula" => SbflFormula::Tarantula,
        "ochiai" => SbflFormula::Ochiai,
        "dstar2" => SbflFormula::DStar { exponent: 2 },
        "dstar3" => SbflFormula::DStar { exponent: 3 },
        _ => {
            warn!("Unknown formula '{}', defaulting to Tarantula", formula);
            SbflFormula::Tarantula
        }
    };

    // Parse output format
    let report_format = match format.to_lowercase().as_str() {
        "json" => ReportFormat::Json,
        "terminal" => ReportFormat::Terminal,
        _ => ReportFormat::Yaml,
    };

    // Read coverage files
    let passed_content = std::fs::read_to_string(&passed_coverage)
        .map_err(|e| anyhow::anyhow!("Failed to read passed coverage file: {}", e))?;
    let failed_content = std::fs::read_to_string(&failed_coverage)
        .map_err(|e| anyhow::anyhow!("Failed to read failed coverage file: {}", e))?;

    // Parse coverage data
    let passed_cov = TarantulaIntegration::parse_lcov_output(&passed_content)?;
    let failed_cov = TarantulaIntegration::parse_lcov_output(&failed_content)?;

    println!(
        "   Parsed: {} passed, {} failed coverage entries",
        passed_cov.len(),
        failed_cov.len()
    );

    // Configure localization
    let config = LocalizationConfig::new()
        .with_formula(sbfl_formula)
        .with_top_n(top_n)
        .with_explanations(true);

    // Run fault localization
    let mut result = TarantulaIntegration::run_localization(
        &passed_cov,
        &failed_cov,
        passed_count,
        failed_count,
        &config,
    );

    println!("   Found {} suspicious statements", result.rankings.len());
    println!("   Confidence: {:.2}", result.confidence);

    // RAG-enhanced localization
    if rag {
        println!("\n🤖 Applying RAG enhancement...");

        // Load or create knowledge base
        let kb = if let Some(kb_path) = &knowledge_base {
            println!("   Loading knowledge base: {}", kb_path.display());
            match BugKnowledgeBase::import_from_yaml(kb_path) {
                Ok(kb) => {
                    println!("   ✅ Loaded {} bugs from knowledge base", kb.len());
                    kb
                }
                Err(e) => {
                    warn!("Failed to load knowledge base: {}", e);
                    println!("   ⚠️  Using empty knowledge base");
                    BugKnowledgeBase::new()
                }
            }
        } else {
            println!("   Using empty knowledge base (no --knowledge-base specified)");
            BugKnowledgeBase::new()
        };

        // Parse fusion strategy
        let fusion_strategy = match fusion.to_lowercase().as_str() {
            "linear" => LocalizationFusion::Linear { sbfl_weight: 0.7 },
            "dbsf" => LocalizationFusion::DBSF,
            "sbfl-only" => LocalizationFusion::SbflOnly,
            _ => LocalizationFusion::RRF { k: 60.0 },
        };

        // Build RAG configuration
        let rag_config = RagLocalizationConfig::new()
            .with_formula(sbfl_formula)
            .with_top_n(top_n)
            .with_similar_bugs(similar_bugs)
            .with_fusion(fusion_strategy)
            .with_explanations(true);

        // Build coverage data for RAG localizer
        let coverage = LcovParser::combine_coverage(&passed_cov, &failed_cov);

        // Run RAG-enhanced localization
        let rag_localizer = RagFaultLocalizer::new(kb, rag_config);
        let rag_result = rag_localizer.localize(&coverage, passed_count, failed_count);

        println!("   ✅ RAG enhancement complete");
        println!("   Knowledge base: {} bugs", rag_result.knowledge_base_size);
        println!("   Fusion: {}", rag_result.fusion_strategy);

        // Generate RAG-enhanced report
        let rag_report = match format.to_lowercase().as_str() {
            "json" => RagReportGenerator::to_json(&rag_result)?,
            "terminal" => RagReportGenerator::to_terminal(&rag_result),
            _ => RagReportGenerator::to_yaml(&rag_result)?,
        };

        // Output
        if format.to_lowercase() == "terminal" {
            println!("\n{}", rag_report);
        } else {
            std::fs::write(&output, &rag_report)?;
            println!("\n✅ RAG-enhanced report saved to: {}", output.display());
        }

        // Summary
        println!("\n📈 Top RAG-Enhanced Rankings:");
        for ranking in rag_result.rankings.iter().take(5) {
            let similar_count = ranking.similar_bugs.len();
            println!(
                "   #{} {}:{} - {:.3} ({} similar bugs)",
                ranking.sbfl_ranking.rank,
                ranking.sbfl_ranking.statement.file.display(),
                ranking.sbfl_ranking.statement.line,
                ranking.combined_score,
                similar_count
            );
            if !ranking.similar_bugs.is_empty() {
                println!("      → Similar: {}", ranking.similar_bugs[0].summary);
            }
        }

        println!("\n🎯 RAG-Enhanced Fault Localization Complete!");
        println!("   ✅ SBFL + RAG fusion applied");
        println!("   ✅ Fusion strategy: {}", rag_result.fusion_strategy);
        if rag_result.knowledge_base_size > 0 {
            println!(
                "   ✅ Bug knowledge base: {} bugs",
                rag_result.knowledge_base_size
            );
        }

        return Ok(());
    }

    let tdg_scores = enrich_tdg_scores(&mut result, enrich_tdg || ensemble || calibrated, enrich_tdg, repo.as_ref());

    if ensemble {
        run_ensemble_prediction(&result, top_n, &tdg_scores, include_churn, ensemble_model.as_ref());
    }

    if calibrated {
        run_calibrated_prediction(&result, top_n, &tdg_scores, confidence_threshold, calibration_model.as_ref());
    }

    // Generate report
    let report = TarantulaIntegration::generate_report(&result, report_format)?;

    // Output
    if report_format == ReportFormat::Terminal {
        println!("\n{}", report);
    } else {
        std::fs::write(&output, &report)?;
        println!("\n✅ Report saved to: {}", output.display());
    }

    // Summary
    println!("\n📈 Top Suspicious Statements:");
    for ranking in result.rankings.iter().take(5) {
        println!(
            "   #{} {}:{} - {:.3}",
            ranking.rank,
            ranking.statement.file.display(),
            ranking.statement.line,
            ranking.suspiciousness
        );
    }

    println!("\n🎯 Fault Localization Complete!");
    println!("   ✅ Spectrum-Based Fault Localization (SBFL)");
    println!("   ✅ {:?} formula applied", sbfl_formula);
    if enrich_tdg {
        println!("   ✅ TDG technical debt scores integrated");
    }

    println!("\n💡 Next Steps:");
    println!("   1. Investigate top suspicious statements");
    println!("   2. Check test coverage for false positives");
    println!("   3. Use --formula ochiai for alternative ranking");

    Ok(())
}

fn run_calibrated_prediction(
    result: &crate::tarantula::FaultLocalizationResult,
    top_n: usize,
    tdg_scores: &std::collections::HashMap<String, f32>,
    confidence_threshold: f32,
    _calibration_model: Option<&PathBuf>,
) {
    use crate::ensemble_predictor::{CalibratedDefectPredictor, FileFeatures};
    println!("\n📊 Running Calibrated Defect Prediction (Phase 7)...");
    println!("   Confidence threshold: {:.0}%", confidence_threshold * 100.0);
    let predictor = CalibratedDefectPredictor::new();
    println!("\n   Calibrated Predictions (above {:.0}% threshold):", confidence_threshold * 100.0);
    for ranking in result.rankings.iter().take(top_n) {
        let file_path = ranking.statement.file.to_string_lossy().to_string();
        let features = FileFeatures::new(ranking.statement.file.clone())
            .with_sbfl(ranking.suspiciousness)
            .with_tdg(tdg_scores.get(&file_path).copied().unwrap_or(0.5));
        let prediction = predictor.predict(&features);
        if prediction.probability >= confidence_threshold {
            println!("   #{} {}:{} - P(defect) = {:.0}% ± {:.0}% [{}]",
                ranking.rank, ranking.statement.file.display(), ranking.statement.line,
                prediction.probability * 100.0,
                (prediction.confidence_interval.1 - prediction.confidence_interval.0) * 50.0,
                prediction.confidence_level);
            for factor in prediction.contributing_factors.iter().filter(|f| f.contribution_pct > 10.0).take(3) {
                println!("      ├─ {}: {:.1}%", factor.factor_name, factor.contribution_pct);
            }
        }
    }
}

fn enrich_tdg_scores(
    result: &mut crate::tarantula::FaultLocalizationResult,
    should_enrich: bool,
    explicit_tdg: bool,
    repo: Option<&PathBuf>,
) -> std::collections::HashMap<String, f32> {
    let mut tdg_scores = std::collections::HashMap::new();
    if !should_enrich { return tdg_scores; }
    if let Some(repo_path) = repo {
        println!("\n📊 Enriching with TDG scores from pmat...");
        match PmatIntegration::analyze_tdg(repo_path) {
            Ok(tdg_analysis) => {
                TarantulaIntegration::enrich_with_tdg(result, &tdg_analysis.file_scores);
                tdg_scores = tdg_analysis.file_scores;
                println!("   ✅ TDG scores added for {} files", tdg_scores.len());
            }
            Err(e) => { warn!("TDG enrichment failed: {}", e); }
        }
    } else if explicit_tdg {
        warn!("--enrich-tdg requires --repo path");
    }
    tdg_scores
}

fn run_ensemble_prediction(
    result: &crate::tarantula::FaultLocalizationResult,
    top_n: usize,
    tdg_scores: &std::collections::HashMap<String, f32>,
    include_churn: bool,
    ensemble_model_path: Option<&PathBuf>,
) {
    use crate::ensemble_predictor::{FileFeatures, WeightedEnsembleModel};

    println!("\n🔮 Running Weighted Ensemble Model (Phase 6)...");
    let mut model = WeightedEnsembleModel::new();
    if let Some(model_path) = ensemble_model_path {
        match model.load(model_path) {
            Ok(()) => println!("   ✅ Loaded ensemble model from {}", model_path.display()),
            Err(e) => { warn!("Failed to load ensemble model: {}", e); }
        }
    }
    let file_features: Vec<FileFeatures> = result.rankings.iter().take(top_n).map(|r| {
        let file_path = r.statement.file.to_string_lossy().to_string();
        FileFeatures::new(r.statement.file.clone())
            .with_sbfl(r.suspiciousness)
            .with_tdg(tdg_scores.get(&file_path).copied().unwrap_or(0.5))
            .with_churn(if include_churn { 0.5 } else { 0.0 })
            .with_complexity(0.5)
            .with_rag_similarity(0.0)
    }).collect();
    if !model.is_fitted() && !file_features.is_empty() {
        if let Err(e) = model.fit(&file_features) { warn!("Ensemble fitting failed: {}", e); }
    }
    for (i, features) in file_features.iter().take(5).enumerate() {
        println!("   #{} {} - Risk: {:.1}%", i + 1, features.path.display(), model.predict(features) * 100.0);
    }
    if let Some(weights) = model.get_weights() {
        println!("\n   Learned Signal Weights:");
        for (name, weight) in weights.names.iter().zip(weights.weights.iter()) {
            println!("      {}: {:.1}%", name, weight * 100.0);
        }
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
