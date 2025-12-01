// Example: Tarantula Fault Localization
// Demonstrates SBFL (Spectrum-Based Fault Localization) algorithms
//
// Usage:
//   cargo run --example fault_localization
//
// This example shows:
// - Tarantula, Ochiai, and DStar formulas
// - SZZ algorithm for bug-introducing commit detection
// - Hybrid fault localization combining SBFL with historical data
// - Weighted Ensemble Model (Phase 6)
// - Calibrated Defect Prediction (Phase 7)

use organizational_intelligence_plugin::ensemble_predictor::{
    CalibratedDefectPredictor, FileFeatures, WeightedEnsembleModel,
};
use organizational_intelligence_plugin::tarantula::{
    dstar, ochiai, tarantula, HybridFaultLocalizer, LocalizationConfig, ReportFormat, SbflFormula,
    SbflLocalizer, StatementCoverage, StatementId, SzzAnalyzer, TarantulaIntegration,
};
use std::collections::HashMap;
use std::path::PathBuf;

fn main() {
    // Initialize logging
    tracing_subscriber::fmt::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("🔍 Tarantula Fault Localization Demo\n");
    println!("Toyota Way: Genchi Genbutsu - Go and see the actual data\n");

    // Demo 1: Basic SBFL formulas
    demo_sbfl_formulas();

    // Demo 2: Full localization with LocalizationConfig
    demo_localizer();

    // Demo 3: SZZ algorithm for bug-introducing commits
    demo_szz_algorithm();

    // Demo 4: Hybrid fault localization
    demo_hybrid_localization();

    // Demo 5: Report generation
    demo_report_generation();

    // Demo 6: Weighted Ensemble Model (Phase 6)
    demo_weighted_ensemble();

    // Demo 7: Calibrated Defect Prediction (Phase 7)
    demo_calibrated_prediction();

    println!("\n🎯 Fault Localization Demo Complete!");
    println!("   Run 'oip localize --help' to use the CLI tool");
}

fn demo_sbfl_formulas() {
    println!("═══════════════════════════════════════════════════════════");
    println!("📊 SBFL Formula Comparison");
    println!("═══════════════════════════════════════════════════════════\n");

    // Simulated coverage data for a suspicious statement
    // Statement executed by 9/10 failing tests, 2/100 passing tests
    let failed = 9;
    let passed = 2;
    let total_failed = 10;
    let total_passed = 100;

    println!(
        "Scenario: Statement executed by {}/{} failing tests, {}/{} passing tests\n",
        failed, total_failed, passed, total_passed
    );

    // Tarantula formula
    let tarantula_score = tarantula(failed, passed, total_failed, total_passed);
    println!("Tarantula:  {:.4}", tarantula_score);
    println!("  Formula: (failed/totalFailed) / ((passed/totalPassed) + (failed/totalFailed))");

    // Ochiai formula
    let ochiai_score = ochiai(failed, passed, total_failed);
    println!("\nOchiai:     {:.4}", ochiai_score);
    println!("  Formula: failed / sqrt(totalFailed × (failed + passed))");

    // DStar formulas
    let dstar2_score = dstar(failed, passed, total_failed, 2);
    let dstar3_score = dstar(failed, passed, total_failed, 3);
    println!("\nDStar(2):   {:.4}", dstar2_score);
    println!("DStar(3):   {:.4}", dstar3_score);
    println!("  Formula: failed^* / (passed + (totalFailed - failed))");

    println!("\n✅ Higher scores = more suspicious\n");
}

fn demo_localizer() {
    println!("═══════════════════════════════════════════════════════════");
    println!("🔬 Full Localization with SbflLocalizer");
    println!("═══════════════════════════════════════════════════════════\n");

    // Create synthetic coverage data simulating a bug at line 50
    let coverage = vec![
        // Line 50: Bug location - executed by all failing tests, few passing
        StatementCoverage::new(StatementId::new("src/parser.rs", 50), 2, 10),
        // Line 55: Common code - executed by many tests
        StatementCoverage::new(StatementId::new("src/parser.rs", 55), 80, 8),
        // Line 60: Regular code
        StatementCoverage::new(StatementId::new("src/parser.rs", 60), 90, 5),
        // Line 100: Only passing tests
        StatementCoverage::new(StatementId::new("src/util.rs", 100), 95, 0),
    ];

    // Configure and run localization
    let config = LocalizationConfig::new()
        .with_formula(SbflFormula::Tarantula)
        .with_top_n(5)
        .with_explanations(true);

    let localizer = SbflLocalizer::new()
        .with_formula(config.formula)
        .with_top_n(config.top_n)
        .with_explanations(config.include_explanations);

    let result = localizer.localize(&coverage, 100, 10);

    println!("Localization Results (Tarantula):\n");
    for ranking in &result.rankings {
        println!(
            "  #{} {}:{} - {:.3}",
            ranking.rank,
            ranking.statement.file.display(),
            ranking.statement.line,
            ranking.suspiciousness
        );
        println!("      {}\n", ranking.explanation);
    }

    println!("Confidence: {:.2}", result.confidence);
    println!("Formula: {:?}\n", result.formula_used);
}

fn demo_szz_algorithm() {
    println!("═══════════════════════════════════════════════════════════");
    println!("🔬 SZZ Algorithm - Bug-Introducing Commit Detection");
    println!("═══════════════════════════════════════════════════════════\n");

    // Simulated commit history
    let commits = vec![
        (
            "abc123".to_string(),
            "fix: null pointer in parser".to_string(),
        ),
        (
            "def456".to_string(),
            "feat: add new parser feature".to_string(),
        ),
        (
            "ghi789".to_string(),
            "bugfix: race condition in handler".to_string(),
        ),
        ("jkl012".to_string(), "docs: update README".to_string()),
        (
            "mno345".to_string(),
            "closes #42: fix memory leak".to_string(),
        ),
    ];

    // Identify bug-fixing commits
    let fixes = SzzAnalyzer::identify_bug_fixes(&commits);

    println!("Bug-fixing commits identified: {}", fixes.len());
    for (hash, msg) in &fixes {
        let short_hash = if hash.len() >= 7 {
            &hash[..7]
        } else {
            hash.as_str()
        };
        println!("  - {}: \"{}\"", short_hash, msg);
    }

    // Simulate tracing a bug back to its introduction
    let changed_lines = vec![
        ("src/parser.rs".to_string(), 50, true), // Deleted buggy line
        ("src/parser.rs".to_string(), 51, true), // Deleted buggy line
        ("src/parser.rs".to_string(), 52, false), // Added fix
    ];

    let mut blame_data = HashMap::new();
    blame_data.insert(
        ("src/parser.rs".to_string(), 50),
        (
            "bad_commit_001".to_string(),
            "developer@example.com".to_string(),
        ),
    );
    blame_data.insert(
        ("src/parser.rs".to_string(), 51),
        (
            "bad_commit_001".to_string(),
            "developer@example.com".to_string(),
        ),
    );

    let szz_result = SzzAnalyzer::trace_introducing_commits(
        "abc123",
        "fix: null pointer in parser",
        &changed_lines,
        &blame_data,
    );

    println!("\nSZZ Trace for fix abc123:");
    println!(
        "  Bug-introducing commits: {:?}",
        szz_result.bug_introducing_commits
    );
    println!("  Faulty lines: {:?}", szz_result.faulty_lines);
    println!("  Confidence: {:?}", szz_result.confidence);

    // Calculate file suspiciousness from SZZ results
    let szz_results = vec![szz_result];
    let file_suspiciousness = SzzAnalyzer::calculate_file_suspiciousness(&szz_results);

    println!("\nHistorical file suspiciousness:");
    for (file, score) in &file_suspiciousness {
        println!("  {}: {:.2}", file, score);
    }
    println!();
}

fn demo_hybrid_localization() {
    println!("═══════════════════════════════════════════════════════════");
    println!("🎯 Hybrid Fault Localization (SBFL + Historical)");
    println!("═══════════════════════════════════════════════════════════\n");

    // SBFL results
    let coverage = vec![
        StatementCoverage::new(StatementId::new("src/parser.rs", 50), 5, 10),
        StatementCoverage::new(StatementId::new("src/handler.rs", 100), 20, 8),
    ];

    let sbfl_result = SbflLocalizer::new()
        .with_formula(SbflFormula::Tarantula)
        .localize(&coverage, 100, 10);

    // Historical suspiciousness from SZZ
    let mut historical = HashMap::new();
    historical.insert("src/parser.rs".to_string(), 0.8_f32); // High historical bugs
    historical.insert("src/handler.rs".to_string(), 0.3_f32); // Lower historical

    // Combine with alpha=0.7 (70% SBFL, 30% historical)
    let combined = HybridFaultLocalizer::combine_scores(&sbfl_result, &historical, 0.7);

    println!("Combined Rankings (α=0.7):\n");
    for ranking in &combined.rankings {
        let sbfl = ranking.scores.get("tarantula").unwrap_or(&0.0);
        let hist = ranking.scores.get("historical").unwrap_or(&0.0);
        println!(
            "  #{} {}:{}",
            ranking.rank,
            ranking.statement.file.display(),
            ranking.statement.line
        );
        println!(
            "      Combined: {:.3} (SBFL: {:.3}, Historical: {:.3})\n",
            ranking.suspiciousness, sbfl, hist
        );
    }
}

fn demo_report_generation() {
    println!("═══════════════════════════════════════════════════════════");
    println!("📄 Report Generation");
    println!("═══════════════════════════════════════════════════════════\n");

    let coverage = vec![StatementCoverage::new(
        StatementId::new("src/bug.rs", 42),
        5,
        10,
    )];

    let result = SbflLocalizer::new()
        .with_formula(SbflFormula::Ochiai)
        .with_top_n(3)
        .localize(&coverage, 50, 10);

    // Generate YAML report
    let yaml = TarantulaIntegration::generate_report(&result, ReportFormat::Yaml).unwrap();
    println!("YAML Report (truncated):");
    println!("{}\n", &yaml[..yaml.len().min(500)]);

    // Generate terminal report
    let terminal = TarantulaIntegration::generate_report(&result, ReportFormat::Terminal).unwrap();
    println!("Terminal Report:");
    println!("{}", terminal);
}

fn demo_weighted_ensemble() {
    println!("═══════════════════════════════════════════════════════════");
    println!("🔮 Weighted Ensemble Model (Phase 6)");
    println!("═══════════════════════════════════════════════════════════\n");

    // Create synthetic file features (simulating SBFL + TDG + Churn data)
    // All scores are normalized 0.0-1.0
    let features = vec![
        FileFeatures {
            path: PathBuf::from("src/parser.rs"),
            sbfl_score: 0.92, // High SBFL suspiciousness
            tdg_score: 0.65,
            churn_score: 0.83,      // Normalized: 25 commits / 30 max
            complexity_score: 0.75, // Normalized: 15 / 20 max
            rag_similarity: 0.0,
        },
        FileFeatures {
            path: PathBuf::from("src/handler.rs"),
            sbfl_score: 0.78,
            tdg_score: 0.45,
            churn_score: 0.27,      // 8 / 30
            complexity_score: 0.40, // 8 / 20
            rag_similarity: 0.0,
        },
        FileFeatures {
            path: PathBuf::from("src/util.rs"),
            sbfl_score: 0.35,      // Low SBFL suspiciousness
            tdg_score: 0.80,       // But high technical debt
            churn_score: 1.0,      // 42 / 30 (capped)
            complexity_score: 1.0, // 22 / 20 (capped)
            rag_similarity: 0.0,
        },
        FileFeatures {
            path: PathBuf::from("src/config.rs"),
            sbfl_score: 0.15,
            tdg_score: 0.20,
            churn_score: 0.10,      // 3 / 30
            complexity_score: 0.20, // 4 / 20
            rag_similarity: 0.0,
        },
    ];

    // Create ensemble model (uses default labeling functions)
    let mut ensemble = WeightedEnsembleModel::new();

    // Fit ensemble model using EM algorithm
    let _ = ensemble.fit(&features);

    // Display learned weights
    println!("Learned Signal Weights (via EM algorithm):\n");
    if let Some(weights) = ensemble.get_weights() {
        for (name, weight) in weights.names.iter().zip(weights.weights.iter()) {
            let bar = "█".repeat((weight * 20.0) as usize);
            let empty = "░".repeat(20 - (weight * 20.0) as usize);
            println!("  {:<12} {}{} {:.1}%", name, bar, empty, weight * 100.0);
        }
    }

    // Predict risk scores for each file
    println!("\nEnsemble Risk Predictions:\n");
    let mut predictions: Vec<_> = features.iter().map(|f| (f, ensemble.predict(f))).collect();
    predictions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    for (i, (file, risk)) in predictions.iter().enumerate() {
        let bar = "█".repeat((risk * 20.0) as usize);
        let empty = "░".repeat(20 - (risk * 20.0) as usize);
        println!(
            "  #{} {} {}{} Risk: {:.1}%",
            i + 1,
            file.path.display(),
            bar,
            empty,
            risk * 100.0
        );
    }
    println!();
}

fn demo_calibrated_prediction() {
    println!("═══════════════════════════════════════════════════════════");
    println!("📊 Calibrated Defect Prediction (Phase 7)");
    println!("═══════════════════════════════════════════════════════════\n");

    // Historical features for calibration training (normalized scores 0.0-1.0)
    let historical_features = vec![
        FileFeatures {
            path: PathBuf::from("train/file1.rs"),
            sbfl_score: 0.95,
            tdg_score: 0.70,
            churn_score: 1.0,       // 30 / 30
            complexity_score: 0.90, // 18 / 20
            rag_similarity: 0.0,
        },
        FileFeatures {
            path: PathBuf::from("train/file2.rs"),
            sbfl_score: 0.10,
            tdg_score: 0.20,
            churn_score: 0.17,      // 5 / 30
            complexity_score: 0.20, // 4 / 20
            rag_similarity: 0.0,
        },
        FileFeatures {
            path: PathBuf::from("train/file3.rs"),
            sbfl_score: 0.85,
            tdg_score: 0.55,
            churn_score: 0.50,      // 15 / 30
            complexity_score: 0.60, // 12 / 20
            rag_similarity: 0.0,
        },
        FileFeatures {
            path: PathBuf::from("train/file4.rs"),
            sbfl_score: 0.25,
            tdg_score: 0.30,
            churn_score: 0.27,      // 8 / 30
            complexity_score: 0.30, // 6 / 20
            rag_similarity: 0.0,
        },
        FileFeatures {
            path: PathBuf::from("train/file5.rs"),
            sbfl_score: 0.70,
            tdg_score: 0.60,
            churn_score: 0.73,      // 22 / 30
            complexity_score: 0.70, // 14 / 20
            rag_similarity: 0.0,
        },
    ];

    // Known defect labels for training
    let historical_labels = vec![true, false, true, false, true];

    // Test features for prediction
    let test_features = vec![
        FileFeatures {
            path: PathBuf::from("src/parser.rs"),
            sbfl_score: 0.92,
            tdg_score: 0.65,
            churn_score: 0.83,      // 25 / 30
            complexity_score: 0.75, // 15 / 20
            rag_similarity: 0.0,
        },
        FileFeatures {
            path: PathBuf::from("src/handler.rs"),
            sbfl_score: 0.55,
            tdg_score: 0.45,
            churn_score: 0.40,      // 12 / 30
            complexity_score: 0.45, // 9 / 20
            rag_similarity: 0.0,
        },
        FileFeatures {
            path: PathBuf::from("src/config.rs"),
            sbfl_score: 0.15,
            tdg_score: 0.20,
            churn_score: 0.10,      // 3 / 30
            complexity_score: 0.20, // 4 / 20
            rag_similarity: 0.0,
        },
    ];

    // Create and train calibrated predictor (uses default labeling functions)
    let mut predictor = CalibratedDefectPredictor::new();
    let _ = predictor.fit(&historical_features, &historical_labels);

    // Get calibrated predictions for each test file
    println!("Calibrated Predictions with Confidence Intervals:\n");
    let confidence_threshold = 0.5;

    for file in &test_features {
        let pred = predictor.predict(file);
        let (lo, hi) = pred.confidence_interval;
        let ci_width = (hi - lo) * 100.0;

        let level_str = match pred.confidence_level {
            organizational_intelligence_plugin::ensemble_predictor::ConfidenceLevel::High => {
                "[HIGH]  "
            }
            organizational_intelligence_plugin::ensemble_predictor::ConfidenceLevel::Medium => {
                "[MEDIUM]"
            }
            organizational_intelligence_plugin::ensemble_predictor::ConfidenceLevel::Low => {
                "[LOW]   "
            }
        };

        if pred.probability >= confidence_threshold {
            println!(
                "  {} {} - P(defect) = {:.0}% ± {:.0}%",
                level_str,
                file.path.display(),
                pred.probability * 100.0,
                ci_width / 2.0
            );

            // Show contributing factors
            for factor in &pred.contributing_factors {
                println!(
                    "      ├─ {}: {:.1}%",
                    factor.factor_name, factor.contribution_pct
                );
            }
            println!();
        } else {
            println!(
                "  {} {} - P(defect) = {:.0}% (below threshold)",
                level_str,
                file.path.display(),
                pred.probability * 100.0
            );
        }
    }

    println!(
        "\n  Confidence threshold: {:.0}%",
        confidence_threshold * 100.0
    );
    println!("  Files above threshold get full analysis\n");
}
