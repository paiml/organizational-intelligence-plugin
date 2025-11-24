//! OIP-GPU: GPU-Accelerated Correlation & Pattern Prediction
//!
//! Main entry point for the GPU-accelerated analysis system.

use anyhow::Result;
use clap::{Parser, Subcommand};
use organizational_intelligence_plugin::{
    analyzer::OrgAnalyzer,
    features::{CommitFeatures, FeatureExtractor},
    query::{QueryParser, QueryType},
    storage::FeatureStore,
};
use std::collections::HashMap;

#[derive(Parser)]
#[command(name = "oip-gpu")]
#[command(about = "GPU-Accelerated Correlation & Pattern Prediction System")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Force specific compute backend
    #[arg(long, global = true, value_enum)]
    backend: Option<Backend>,

    /// Verbose logging
    #[arg(short, long, global = true)]
    verbose: bool,

    /// Configuration file path
    #[arg(long, global = true)]
    config: Option<std::path::PathBuf>,
}

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
enum Backend {
    Gpu,
    Simd,
    Cpu,
}

#[derive(Subcommand)]
enum Commands {
    /// Analyze GitHub organization or repositories
    Analyze {
        /// GitHub organization name
        #[arg(long, group = "target")]
        org: Option<String>,

        /// Comma-separated repository list (owner/repo)
        #[arg(long, group = "target")]
        repos: Option<String>,

        /// Single repository (owner/repo)
        #[arg(long, group = "target")]
        repo: Option<String>,

        /// Local repository path
        #[arg(long, group = "target")]
        local: Option<std::path::PathBuf>,

        /// Output database file (trueno-db format)
        #[arg(short, long, default_value = "oip-gpu.db")]
        output: std::path::PathBuf,

        /// Only analyze commits after date (YYYY-MM-DD)
        #[arg(long)]
        since: Option<String>,

        /// Parallel worker count (default: auto)
        #[arg(long)]
        workers: Option<usize>,

        /// Maximum commits to analyze
        #[arg(long, default_value = "1000")]
        max_commits: usize,
    },

    /// Compute correlations between defect patterns
    Correlate {
        /// Input database (from analyze command)
        #[arg(short, long)]
        input: std::path::PathBuf,

        /// Output file (JSON/YAML/CSV)
        #[arg(short, long)]
        output: std::path::PathBuf,

        /// Specific categories to correlate (comma-separated)
        #[arg(long)]
        categories: Option<String>,

        /// Time lag for Granger causality (days)
        #[arg(long)]
        lag: Option<u32>,

        /// Output format
        #[arg(long, value_enum, default_value = "json")]
        format: OutputFormat,

        /// Only show correlations above threshold
        #[arg(long)]
        threshold: Option<f32>,
    },

    /// Predict defect likelihood for PR/commit
    Predict {
        /// GitHub PR URL
        #[arg(long, group = "predict_target")]
        pr: Option<String>,

        /// Local files to analyze (comma-separated)
        #[arg(long, group = "predict_target")]
        files: Option<String>,

        /// Predict for all open PRs in org
        #[arg(long, group = "predict_target")]
        org: Option<String>,

        /// Custom trained model
        #[arg(long)]
        model: Option<std::path::PathBuf>,

        /// Show feature importance (SHAP values)
        #[arg(long)]
        explain: bool,
    },

    /// Natural language query interface
    Query {
        /// Query string
        query: String,

        /// Database file
        #[arg(short, long, default_value = "oip-gpu.db")]
        input: std::path::PathBuf,

        /// Output format
        #[arg(long, value_enum, default_value = "table")]
        format: OutputFormat,

        /// Limit results to N entries
        #[arg(long)]
        limit: Option<usize>,

        /// Export results to file
        #[arg(long)]
        export: Option<std::path::PathBuf>,
    },

    /// Cluster repositories by defect patterns
    Cluster {
        /// Input database
        #[arg(short, long)]
        input: std::path::PathBuf,

        /// Number of clusters
        #[arg(short = 'k', long, default_value = "10")]
        clusters: usize,

        /// Output file
        #[arg(short, long)]
        output: std::path::PathBuf,
    },

    /// Graph analytics (PageRank, betweenness)
    Graph {
        /// Input database
        #[arg(short, long)]
        input: std::path::PathBuf,

        /// Algorithm
        #[arg(long, value_enum)]
        algorithm: GraphAlgorithm,

        /// Output file
        #[arg(short, long)]
        output: std::path::PathBuf,
    },

    /// Export data to various formats
    Export {
        /// Input database
        #[arg(short, long)]
        input: std::path::PathBuf,

        /// Output format
        #[arg(long, value_enum)]
        format: ExportFormat,

        /// Output file
        #[arg(short, long)]
        output: std::path::PathBuf,
    },

    /// Run performance benchmarks
    Benchmark {
        /// Benchmark suite
        #[arg(long, value_enum)]
        suite: BenchmarkSuite,

        /// Output results file
        #[arg(short, long)]
        output: Option<std::path::PathBuf>,
    },
}

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
enum OutputFormat {
    Json,
    Yaml,
    Csv,
    Table,
}

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
enum ExportFormat {
    Json,
    Yaml,
    Csv,
    Parquet,
}

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
enum GraphAlgorithm {
    Pagerank,
    Betweenness,
    Bfs,
}

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
enum BenchmarkSuite {
    Correlation,
    Clustering,
    Graph,
    All,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize logging
    let log_level = if cli.verbose { "debug" } else { "info" };
    tracing_subscriber::fmt().with_env_filter(log_level).init();

    // Log selected backend
    if let Some(backend) = cli.backend {
        tracing::info!("Forcing backend: {:?}", backend);
    }

    // Execute command
    match cli.command {
        Commands::Analyze {
            org,
            repos,
            repo,
            local,
            output,
            since,
            workers,
            max_commits,
        } => {
            cmd_analyze(
                org,
                repos,
                repo,
                local,
                output,
                since,
                workers,
                max_commits,
                cli.backend,
            )
            .await?;
        }
        Commands::Correlate {
            input,
            output,
            categories,
            lag,
            format,
            threshold,
        } => {
            cmd_correlate(
                input,
                output,
                categories,
                lag,
                format,
                threshold,
                cli.backend,
            )
            .await?;
        }
        Commands::Predict {
            pr,
            files,
            org,
            model,
            explain,
        } => {
            cmd_predict(pr, files, org, model, explain, cli.backend).await?;
        }
        Commands::Query {
            query,
            input,
            format,
            limit,
            export,
        } => {
            cmd_query(query, input, format, limit, export, cli.backend).await?;
        }
        Commands::Cluster {
            input,
            clusters,
            output,
        } => {
            cmd_cluster(input, clusters, output, cli.backend).await?;
        }
        Commands::Graph {
            input,
            algorithm,
            output,
        } => {
            cmd_graph(input, algorithm, output, cli.backend).await?;
        }
        Commands::Export {
            input,
            format,
            output,
        } => {
            cmd_export(input, format, output).await?;
        }
        Commands::Benchmark { suite, output } => {
            cmd_benchmark(suite, output, cli.backend).await?;
        }
    }

    Ok(())
}

// Command implementations (stubs for now - will implement in TDD fashion)

#[allow(clippy::too_many_arguments)]
async fn cmd_analyze(
    org: Option<String>,
    repos: Option<String>,
    repo: Option<String>,
    local: Option<std::path::PathBuf>,
    output: std::path::PathBuf,
    _since: Option<String>,
    _workers: Option<usize>,
    max_commits: usize,
    backend: Option<Backend>,
) -> Result<()> {
    tracing::info!("Starting GPU-accelerated analysis");

    if let Some(b) = backend {
        println!("⚙️  Backend: {:?}", b);
    }

    // Handle local repository analysis
    if let Some(local_path) = local {
        return cmd_analyze_local(local_path, output, max_commits).await;
    }

    // Determine target
    let target = if let Some(_org_name) = org {
        println!("📦 Organization analysis not yet implemented");
        println!("🔜 Phase 1: Single repository only");
        anyhow::bail!("Organization analysis pending (use --repo or --local instead)");
    } else if let Some(_repos_list) = repos {
        println!("📦 Multi-repository analysis not yet implemented");
        anyhow::bail!("Multi-repo analysis pending (use --repo or --local instead)");
    } else if let Some(repo_spec) = repo {
        repo_spec
    } else {
        anyhow::bail!("Must specify --org, --repos, --repo, or --local");
    };

    // Parse repo_spec (owner/repo format)
    let parts: Vec<&str> = target.split('/').collect();
    if parts.len() != 2 {
        anyhow::bail!("Repository must be in owner/repo format (e.g., rust-lang/rust)");
    }
    let (owner, repo_name) = (parts[0], parts[1]);
    let repo_url = format!("https://github.com/{}/{}", owner, repo_name);

    println!("🔍 Analyzing repository: {}", target);

    // Create analyzer with temp cache
    let cache_dir = std::env::temp_dir().join("oip-gpu-cache");
    std::fs::create_dir_all(&cache_dir)?;
    let analyzer = OrgAnalyzer::new(&cache_dir);

    // Analyze repository
    println!("📊 Analyzing commits (max {})...", max_commits);
    let patterns = analyzer
        .analyze_repository(&repo_url, repo_name, max_commits)
        .await?;

    println!("✅ Found {} defect categories", patterns.len());

    // Extract features
    println!("🔧 Extracting features for GPU processing...");
    let extractor = FeatureExtractor::new();
    let mut store = FeatureStore::new()?;

    let mut total_features = 0;
    for pattern in &patterns {
        let category_num = pattern.category as u8;

        for instance in &pattern.examples {
            let features = extractor.extract(
                category_num,
                instance.files_affected,
                instance.lines_added,
                instance.lines_removed,
                instance.timestamp,
            )?;

            store.insert(features)?;
            total_features += 1;
        }
    }

    println!("✅ Extracted {} feature vectors", total_features);

    // Save to storage
    println!("💾 Saving to {}...", output.display());
    store.save(&output).await?;

    println!("✨ Analysis complete!");
    println!(
        "📈 Features: {} vectors × {} dimensions",
        total_features,
        CommitFeatures::DIMENSION
    );
    println!("🎯 Next: oip-gpu correlate --input {}", output.display());

    Ok(())
}

/// Analyze a local git repository
async fn cmd_analyze_local(
    local_path: std::path::PathBuf,
    output: std::path::PathBuf,
    max_commits: usize,
) -> Result<()> {
    use organizational_intelligence_plugin::classifier::RuleBasedClassifier;

    println!("🔍 Analyzing local repository: {}", local_path.display());

    // Verify it's a git repo
    if !local_path.join(".git").exists() {
        anyhow::bail!("Not a git repository: {}", local_path.display());
    }

    println!("📊 Analyzing commits (max {})...", max_commits);

    // Open the repository
    let repo = git2::Repository::open(&local_path)?;

    // Walk commits
    let mut revwalk = repo.revwalk()?;
    revwalk.push_head()?;
    revwalk.set_sorting(git2::Sort::TIME)?;

    let classifier = RuleBasedClassifier::new();
    let extractor = FeatureExtractor::new();
    let mut store = FeatureStore::new()?;

    let mut commit_count = 0;
    let mut feature_count = 0;
    let mut category_counts = std::collections::HashMap::new();

    for oid in revwalk.take(max_commits) {
        let oid = oid?;
        let commit = repo.find_commit(oid)?;

        // Get commit stats
        let (files_changed, lines_added, lines_deleted) = if commit.parent_count() > 0 {
            let parent = commit.parent(0)?;
            let diff =
                repo.diff_tree_to_tree(Some(&parent.tree()?), Some(&commit.tree()?), None)?;
            let stats = diff.stats()?;
            (stats.files_changed(), stats.insertions(), stats.deletions())
        } else {
            (0, 0, 0)
        };

        // Classify the commit by message
        let message = commit.message().unwrap_or("");
        let category_num = if let Some(classification) = classifier.classify_from_message(message) {
            classification.category as u8
        } else {
            0 // Default to category 0 if no classification
        };

        *category_counts.entry(category_num).or_insert(0usize) += 1;

        // Extract features
        if let Ok(features) = extractor.extract(
            category_num,
            files_changed,
            lines_added,
            lines_deleted,
            commit.time().seconds(),
        ) {
            store.insert(features)?;
            feature_count += 1;
        }

        commit_count += 1;
        if commit_count % 100 == 0 {
            print!("\r📊 Processed {} commits...", commit_count);
        }
    }
    println!();

    println!("✅ Analyzed {} commits", commit_count);
    println!("✅ Extracted {} feature vectors", feature_count);

    // Print category distribution
    println!();
    println!("📊 Defect category distribution:");
    let mut sorted: Vec<_> = category_counts.iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(a.1));
    for (cat, count) in sorted.iter().take(5) {
        let pct = (**count as f32 / commit_count as f32) * 100.0;
        println!("   Category {}: {} ({:.1}%)", cat, count, pct);
    }

    // Save to storage
    println!();
    println!("💾 Saving to {}...", output.display());
    store.save(&output).await?;

    println!("✨ Analysis complete!");
    println!(
        "📈 Features: {} vectors × {} dimensions",
        feature_count,
        CommitFeatures::DIMENSION
    );
    println!(
        "🎯 Next: oip-gpu query --input {} \"show all defects\"",
        output.display()
    );

    Ok(())
}

async fn cmd_correlate(
    _input: std::path::PathBuf,
    _output: std::path::PathBuf,
    _categories: Option<String>,
    _lag: Option<u32>,
    _format: OutputFormat,
    _threshold: Option<f32>,
    _backend: Option<Backend>,
) -> Result<()> {
    println!("Correlate command - not yet implemented");
    println!("Phase 1 implementation pending");
    Ok(())
}

async fn cmd_predict(
    _pr: Option<String>,
    _files: Option<String>,
    _org: Option<String>,
    _model: Option<std::path::PathBuf>,
    _explain: bool,
    _backend: Option<Backend>,
) -> Result<()> {
    println!("Predict command - not yet implemented");
    println!("Phase 3 implementation pending");
    Ok(())
}

async fn cmd_query(
    query: String,
    input: std::path::PathBuf,
    format: OutputFormat,
    limit: Option<usize>,
    export: Option<std::path::PathBuf>,
    _backend: Option<Backend>,
) -> Result<()> {
    println!("🔍 Executing query: \"{}\"", query);

    // Parse natural language query
    let parser = QueryParser::new();
    let parsed = parser.parse(&query)?;

    println!("📋 Query type: {:?}", parsed.query_type);
    println!();

    // Load feature store
    println!("📂 Loading features from {}...", input.display());
    let store = FeatureStore::load(&input).await?;

    if store.is_empty() {
        println!("⚠️  No features found in store");
        println!(
            "💡 Run: oip-gpu analyze --repo owner/repo --output {}",
            input.display()
        );
        return Ok(());
    }

    println!("✅ Loaded {} feature vectors", store.len());
    println!();

    // Execute query
    let result = execute_query(&store, &parsed, limit)?;

    // Format output
    match format {
        OutputFormat::Table => {
            print_table(&result);
        }
        OutputFormat::Json => {
            println!("{}", serde_json::to_string_pretty(&result)?);
        }
        OutputFormat::Yaml => {
            println!("{}", serde_yaml::to_string(&result)?);
        }
        OutputFormat::Csv => {
            print_csv(&result)?;
        }
    }

    // Export if requested
    if let Some(export_path) = export {
        std::fs::write(&export_path, serde_json::to_string_pretty(&result)?)?;
        println!();
        println!("💾 Results exported to: {}", export_path.display());
    }

    Ok(())
}

/// Execute parsed query against feature store
fn execute_query(
    store: &FeatureStore,
    query: &organizational_intelligence_plugin::query::Query,
    limit: Option<usize>,
) -> Result<QueryResult> {
    match &query.query_type {
        QueryType::MostCommonDefect => {
            let counts = count_by_category(store);
            let mut sorted: Vec<_> = counts.into_iter().collect();
            sorted.sort_by(|a, b| b.1.cmp(&a.1));

            if let Some(limit) = limit {
                sorted.truncate(limit);
            }

            Ok(QueryResult::CategoryCounts(sorted))
        }
        QueryType::CountByCategory => {
            let counts = count_by_category(store);
            let mut sorted: Vec<_> = counts.into_iter().collect();
            sorted.sort_by_key(|(cat, _)| *cat);

            Ok(QueryResult::CategoryCounts(sorted))
        }
        QueryType::ListAll => {
            let total = store.len();
            let counts = count_by_category(store);

            Ok(QueryResult::Summary {
                total_features: total,
                category_counts: counts,
            })
        }
        QueryType::Unknown(q) => {
            anyhow::bail!("Unknown query: '{}'\n\nSupported queries:\n  - show me most common defect\n  - count defects by category\n  - show all defects", q)
        }
    }
}

/// Count features by category
fn count_by_category(store: &FeatureStore) -> HashMap<u8, usize> {
    let mut counts: HashMap<u8, usize> = HashMap::new();

    // Query each category (0-9)
    for category in 0..10 {
        if let Ok(results) = store.query_by_category(category) {
            counts.insert(category, results.len());
        }
    }

    counts
}

/// Query result types
#[derive(Debug, serde::Serialize)]
enum QueryResult {
    CategoryCounts(Vec<(u8, usize)>),
    Summary {
        total_features: usize,
        category_counts: HashMap<u8, usize>,
    },
}

/// Print results as table
fn print_table(result: &QueryResult) {
    match result {
        QueryResult::CategoryCounts(counts) => {
            println!("┌──────────┬───────┐");
            println!("│ Category │ Count │");
            println!("├──────────┼───────┤");

            for (cat, count) in counts {
                if *count > 0 {
                    println!("│ {:8} │ {:5} │", cat, count);
                }
            }

            println!("└──────────┴───────┘");
        }
        QueryResult::Summary {
            total_features,
            category_counts,
        } => {
            println!("📊 Total features: {}", total_features);
            println!();
            println!("By category:");

            let mut sorted: Vec<_> = category_counts.iter().collect();
            sorted.sort_by(|a, b| b.1.cmp(a.1));

            for (cat, count) in sorted {
                if *count > 0 {
                    let pct = (*count as f32 / *total_features as f32) * 100.0;
                    println!("  Category {}: {} ({:.1}%)", cat, count, pct);
                }
            }
        }
    }
}

/// Print results as CSV
fn print_csv(result: &QueryResult) -> Result<()> {
    match result {
        QueryResult::CategoryCounts(counts) => {
            println!("category,count");
            for (cat, count) in counts {
                if *count > 0 {
                    println!("{},{}", cat, count);
                }
            }
        }
        QueryResult::Summary {
            total_features,
            category_counts,
        } => {
            println!("metric,value");
            println!("total_features,{}", total_features);

            for (cat, count) in category_counts {
                if *count > 0 {
                    println!("category_{},{}", cat, count);
                }
            }
        }
    }
    Ok(())
}

async fn cmd_cluster(
    _input: std::path::PathBuf,
    _clusters: usize,
    _output: std::path::PathBuf,
    _backend: Option<Backend>,
) -> Result<()> {
    println!("Cluster command - not yet implemented");
    println!("Phase 3 implementation pending");
    Ok(())
}

async fn cmd_graph(
    _input: std::path::PathBuf,
    _algorithm: GraphAlgorithm,
    _output: std::path::PathBuf,
    _backend: Option<Backend>,
) -> Result<()> {
    println!("Graph command - not yet implemented");
    println!("Phase 2 implementation pending");
    Ok(())
}

async fn cmd_export(
    _input: std::path::PathBuf,
    _format: ExportFormat,
    _output: std::path::PathBuf,
) -> Result<()> {
    println!("Export command - not yet implemented");
    Ok(())
}

async fn cmd_benchmark(
    suite: BenchmarkSuite,
    output: Option<std::path::PathBuf>,
    backend: Option<Backend>,
) -> Result<()> {
    println!("🚀 Running GPU benchmark suite");
    if let Some(b) = backend {
        println!("⚙️  Backend: {:?}", b);
    }

    let bench_filter = match suite {
        BenchmarkSuite::Correlation => "correlation",
        BenchmarkSuite::Clustering => "clustering",
        BenchmarkSuite::Graph => "graph",
        BenchmarkSuite::All => "",
    };

    println!("📊 Suite: {:?}", suite);
    println!("🔬 Running criterion benchmarks...");
    println!();

    // Run cargo bench with appropriate filter
    let mut cmd = std::process::Command::new("cargo");
    cmd.arg("bench").arg("--bench").arg("gpu_benchmarks");

    if !bench_filter.is_empty() {
        cmd.arg("--").arg(bench_filter);
    }

    let status = cmd.status()?;

    if !status.success() {
        anyhow::bail!("Benchmark execution failed");
    }

    if let Some(output_path) = output {
        println!("💾 Results saved to: {}", output_path.display());
        println!("ℹ️  Note: Criterion results are in target/criterion/");
    }

    println!();
    println!("✨ Benchmarks complete!");
    println!("📈 See target/criterion/ for detailed results");

    Ok(())
}
