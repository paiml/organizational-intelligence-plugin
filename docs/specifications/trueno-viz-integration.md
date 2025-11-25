# OIP + trueno-viz Integration Specification

**Version**: 0.1.0
**Status**: SPECIFICATION
**Date**: 2025-11-25

## Executive Summary

Integrate trueno-viz visualization capabilities into Organizational Intelligence Plugin (OIP) to provide visual defect pattern analysis. Pure Rust, zero JavaScript, terminal + PNG output.

> **[1] Tufte, E. R. (1983). The Visual Display of Quantitative Information. Graphics Press.**
> Foundational work establishing principles for statistical graphics: data-ink ratio, chart-junk elimination, small multiples. OIP visualizations must maximize data-ink ratio - every pixel conveys defect pattern information.

---

## 1. Visualization Catalog

### 1.1 Defect Distribution Heatmap

```rust
use trueno_viz::plots::Heatmap;

/// Visualize defect categories across repositories
fn defect_heatmap(data: &OrgAnalysis) -> Framebuffer {
    let matrix = data.to_category_repo_matrix();  // 18 categories x N repos

    Heatmap::new()
        .data(&matrix)
        .x_labels(&data.repo_names)
        .y_labels(&DefectCategory::all_names())
        .colormap(Colormap::Viridis)
        .build()
        .to_framebuffer()
}
```

**Terminal Output**:
```
              depyler  bashrs  ruchy   decy
ASTTransform    ██████  █████   ████    ███████
OwnershipBorrow ████    ████    ████    ███
StdlibMapping   ██      █       █████   █
Security        █       ████    █       ░
MemorySafety    ░       ░       ░       ███
```

> **[2] Ware, C. (2012). Information Visualization: Perception for Design. Morgan Kaufmann.**
> Color encoding principles: sequential colormaps (Viridis) for magnitude, diverging for deviation from baseline. Heatmaps leverage preattentive processing for pattern detection in O(1) visual time.

> **[11] Liker, J. K. (2004). The Toyota Way: 14 Management Principles from the World's Greatest Manufacturer. McGraw-Hill.**
> **Principle 7: Use visual control so no problems are hidden.**
> The Defect Heatmap serves as an "Andon" board for the codebase, making "abnormalities" (high defect density clusters) instantly visible to the entire team, enabling immediate "swarming" to fix root causes.

> **[Review Response]**
> **Visual control quantified from Org Intelligence**:
> - depyler ASTTransform (50.7%) vs decy ASTTransform (62.3%) → 12% disparity visible in single glance
> - bashrs SecurityVulnerabilities (12.2%) as red hotspot → immediate attention signal
> - Cross-repo OwnershipBorrow consistency (15-20%) validates shared Rust learning curve
>
> Without heatmap: requires reading 4 separate reports. With heatmap: pattern recognition in <1 second (*Ware [2]*: preattentive processing).

### 1.2 Classifier Confusion Matrix

```rust
use trueno_viz::plots::ConfusionMatrix;

/// Visualize ML classifier performance across 18 defect categories
fn classifier_confusion(predictions: &[Category], actuals: &[Category]) -> Framebuffer {
    ConfusionMatrix::new()
        .predictions(predictions)
        .actuals(actuals)
        .labels(&DefectCategory::all_names())
        .normalize(true)  // Show percentages
        .annotate(true)   // Show values in cells
        .build()
        .to_framebuffer()
}
```

> **[3] Fawcett, T. (2006). An Introduction to ROC Analysis. Pattern Recognition Letters 27(8).**
> Confusion matrices reveal classifier bias patterns. For OIP's 18-category classifier, off-diagonal clusters indicate systematic misclassification (e.g., OwnershipBorrow confused with MemorySafety).

### 1.3 Confidence Histogram

```rust
use trueno_viz::plots::Histogram;

/// Distribution of classification confidence scores
fn confidence_distribution(examples: &[TrainingExample]) -> Framebuffer {
    let confidences: Vec<f32> = examples.iter().map(|e| e.confidence).collect();

    Histogram::new()
        .data(&confidences)
        .bins(20)
        .range(0.0..1.0)
        .x_label("Confidence")
        .y_label("Count")
        .build()
        .to_framebuffer()
}
```

**Terminal Output**:
```
Count
│
├─ 150 ┤                              ████
├─ 100 ┤                         ██████████
├─  50 ┤                    ████████████████
├─   0 ┼────────────────────────────────────
       0.0   0.25   0.5   0.75   1.0
                 Confidence
```

> **[4] Wilkinson, L. (2005). The Grammar of Graphics. Springer.**
> Histograms reveal distribution shape - OIP's bimodal confidence distribution (peaks at 0.75 and 0.95) indicates clear separation between ambiguous and confident classifications.

### 1.4 Defect Trend Lines

```rust
use trueno_viz::plots::Line;

/// Defect density over time (commits/week)
fn defect_trend(commits: &[CommitInfo]) -> Framebuffer {
    let weekly_counts = aggregate_by_week(commits);

    Line::new()
        .x(&weekly_counts.weeks)
        .y(&weekly_counts.defect_counts)
        .smooth(SmoothingMethod::MovingAverage { window: 4 })
        .x_label("Week")
        .y_label("Defects")
        .build()
        .to_framebuffer()
}
```

> **[5] Cleveland, W. S. (1993). Visualizing Data. Hobart Press.**
> Time series smoothing (moving average, LOESS) reveals trends obscured by noise. OIP applies 4-week moving average to defect counts, revealing macro patterns in development quality.

### 1.5 ROC/PR Curves

```rust
use trueno_viz::plots::RocPr;

/// Classifier performance curves for each defect category
fn roc_pr_analysis(model: &TrainedModel, test_set: &TestSet) -> Framebuffer {
    let predictions = model.predict_proba(test_set);

    RocPr::new()
        .predictions(&predictions)
        .labels(&test_set.labels)
        .show_auc(true)
        .show_ap(true)  // Average Precision
        .build()
        .to_framebuffer()
}
```

> **[6] Davis, J., & Goadrich, M. (2006). The Relationship Between Precision-Recall and ROC Curves. ICML 2006.**
> For imbalanced datasets (OIP: ASTTransform 50% vs PerformanceIssues 0.2%), PR curves are more informative than ROC. Area under PR curve (AP) better reflects classifier utility.

### 1.6 Repository Network Graph

```rust
use trueno_viz::plots::ForceGraph;

/// Visualize defect pattern similarity between repositories
fn repo_similarity_graph(analysis: &OrgAnalysis) -> Framebuffer {
    let similarity_matrix = compute_defect_similarity(analysis);
    let edges = similarity_to_edges(&similarity_matrix, threshold: 0.7);

    ForceGraph::new()
        .nodes(&analysis.repo_names)
        .edges(&edges)
        .node_size_by(|repo| analysis.defect_count(repo))
        .layout(ForceLayout::FruchtermanReingold)
        .build()
        .to_framebuffer()
}
```

> **[7] Fruchterman, T., & Reingold, E. (1991). Graph Drawing by Force-Directed Placement. Software: Practice and Experience 21(11).**
> Force-directed layouts reveal cluster structure. Repositories with similar defect profiles cluster together, identifying shared technical debt patterns or common code ancestry.

---

## 2. Terminal Output Modes

trueno-viz supports three terminal rendering modes:

| Mode | Characters | Colors | Use Case |
|------|------------|--------|----------|
| `Ascii` | `░▒▓█` | None | CI logs, SSH |
| `Unicode` | `▁▂▃▄▅▆▇█` | None | Modern terminals |
| `Ansi256` | Full block | 256 colors | Rich visualization |
| `TrueColor` | Full block | 24-bit RGB | Full fidelity |

```rust
use trueno_viz::output::{TerminalEncoder, TerminalMode};

let encoder = TerminalEncoder::new()
    .mode(TerminalMode::detect())  // Auto-detect capability
    .width(80)
    .height(24);

encoder.print(&framebuffer);
```

> **[8] Bertin, J. (1967). Semiology of Graphics. University of Wisconsin Press.**
> Visual variables (position, size, value, texture, color, orientation, shape) have different perceptual properties. ASCII mode uses value/texture; color modes add hue channel for categorical encoding.

---

## 3. Integration Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        OIP + trueno-viz                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │  OIP Core   │───▶│  Viz Layer  │───▶│   Output    │     │
│  │             │    │             │    │             │     │
│  │ - Analysis  │    │ - Heatmap   │    │ - Terminal  │     │
│  │ - Classify  │    │ - Confusion │    │ - PNG       │     │
│  │ - Training  │    │ - Histogram │    │ - SVG       │     │
│  │             │    │ - ROC/PR    │    │             │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│         │                  │                  │             │
│         ▼                  ▼                  ▼             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                    Data Flow                        │   │
│  │                                                     │   │
│  │  OrgAnalysis → CategoryMatrix → Framebuffer → PNG  │   │
│  │  TrainedModel → Predictions → ConfusionMatrix → ASCII │
│  │                                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.1 CLI Integration

```bash
# Generate defect heatmap
oip analyze paiml --viz heatmap --output defects.png

# Show classifier confusion matrix in terminal
oip train-classifier --input data.json --viz confusion

# Defect trend with ASCII output (CI-friendly)
oip analyze paiml --viz trend --format ascii
```

> **[12] Shneiderman, B. (1996). The Eyes Have It: A Task by Data Type Taxonomy for Information Visualizations. IEEE.**
> **"Overview first, zoom and filter, then details-on-demand."**
> The CLI design follows this mantra: `oip analyze` provides the high-level overview (heatmap), flags like `--viz` allow filtering, and specific reports provide the details-on-demand.

> **[Review Response]**
> **Information-seeking mantra mapped to OIP workflow**:
>
> | Mantra Phase | OIP Command | Output |
> |--------------|-------------|--------|
> | **Overview** | `oip analyze org --viz heatmap` | 18x4 defect matrix |
> | **Zoom** | `oip analyze repo --viz histogram` | Single repo distribution |
> | **Filter** | `oip analyze repo --category ASTTransform` | Category-specific commits |
> | **Details** | `oip summarize --commit abc123` | Individual commit analysis |
>
> Progressive disclosure: 1,296 commits → 4 repos → 18 categories → 1 commit. Each level reduces cognitive load by 10-100x.

### 3.2 Feature Flag

```toml
# Cargo.toml
[dependencies]
trueno-viz = { version = "0.1", optional = true }

[features]
default = []
viz = ["trueno-viz"]
```

> **[9] Few, S. (2012). Show Me the Numbers: Designing Tables and Graphs to Enlighten. Analytics Press.**
> Visualization should answer specific questions. OIP viz answers: "Where are defects concentrated?", "Is the classifier accurate?", "Are defects trending up/down?"

---

## 4. Example Outputs

### 4.1 README Demo (ASCII)

```
$ oip analyze paiml/depyler --viz summary

Organizational Intelligence Report: paiml/depyler
═══════════════════════════════════════════════════

Defect Distribution (489 commits analyzed)
──────────────────────────────────────────
ASTTransform     ████████████████████ 50.7%
OwnershipBorrow  ███████░░░░░░░░░░░░░ 18.6%
StdlibMapping    ███░░░░░░░░░░░░░░░░░  8.8%
Comprehension    ██░░░░░░░░░░░░░░░░░░  5.1%
TypeAnnotation   █░░░░░░░░░░░░░░░░░░░  3.7%
Other            ████░░░░░░░░░░░░░░░░ 13.1%

Confidence Distribution
───────────────────────
         ▁▂▃▅▇█▇▅▃▂▁
       0.6   0.8   1.0

Classifier Performance: AUC=0.89, AP=0.84
```

### 4.2 PNG Export

```rust
use trueno_viz::output::PngEncoder;

let fb = defect_heatmap(&analysis);
PngEncoder::new()
    .width(800)
    .height(600)
    .save(&fb, "defect_heatmap.png")?;
```

> **[10] Munzner, T. (2014). Visualization Analysis and Design. CRC Press.**
> Visualization design framework: What (data abstraction), Why (task abstraction), How (idiom). OIP viz: What=defect counts, Why=identify patterns, How=heatmap/histogram/ROC.

---

## 5. Implementation Plan

### Phase 1: Core Visualizations
- [ ] Add `trueno-viz` optional dependency
- [ ] Implement `DefectHeatmap` plot
- [ ] Implement `ConfidenceHistogram` plot
- [ ] Add `--viz` CLI flag

### Phase 2: ML Visualizations
- [ ] Implement `ConfusionMatrix` for classifier
- [ ] Implement `RocPrCurves` for multi-class
- [ ] Add `LossCurve` for training progress

### Phase 3: Advanced
- [ ] Repository similarity `ForceGraph`
- [ ] Temporal trend `Line` plots
- [ ] Interactive terminal mode (cursor navigation)

---

## 6. Quality Gates

| Metric | Threshold |
|--------|-----------|
| Test coverage (viz module) | 90%+ |
| PNG output matches reference | Pixel-exact |
| Terminal output deterministic | Hash-stable |
| Render time (heatmap 18x10) | <10ms |

---

## 7. References

1. Tufte, E. R. (1983). The Visual Display of Quantitative Information. Graphics Press.

2. Ware, C. (2012). Information Visualization: Perception for Design. Morgan Kaufmann. https://doi.org/10.1016/C2009-0-62432-6

3. Fawcett, T. (2006). An Introduction to ROC Analysis. Pattern Recognition Letters 27(8). https://doi.org/10.1016/j.patrec.2005.10.010

4. Wilkinson, L. (2005). The Grammar of Graphics. Springer. https://doi.org/10.1007/0-387-28695-0

5. Cleveland, W. S. (1993). Visualizing Data. Hobart Press.

6. Davis, J., & Goadrich, M. (2006). The Relationship Between Precision-Recall and ROC Curves. ICML 2006. https://doi.org/10.1145/1143844.1143874

7. Fruchterman, T., & Reingold, E. (1991). Graph Drawing by Force-Directed Placement. Software: Practice and Experience 21(11). https://doi.org/10.1002/spe.4380211102

8. Bertin, J. (1967). Semiology of Graphics. University of Wisconsin Press.

9. Few, S. (2012). Show Me the Numbers: Designing Tables and Graphs to Enlighten. Analytics Press.

10. Munzner, T. (2014). Visualization Analysis and Design. CRC Press. https://doi.org/10.1201/b17511

11. Liker, J. K. (2004). The Toyota Way: 14 Management Principles from the World's Greatest Manufacturer. McGraw-Hill.

12. Shneiderman, B. (1996). The Eyes Have It: A Task by Data Type Taxonomy for Information Visualizations. Proceedings of the IEEE Symposium on Visual Languages. https://doi.org/10.1109/VL.1996.545307

---

## Appendix A: Colormap Selection

| Data Type | Recommended Colormap | Rationale |
|-----------|---------------------|-----------|
| Sequential (counts) | Viridis | Perceptually uniform, colorblind-safe |
| Diverging (deviation) | RdBu | Clear positive/negative distinction |
| Categorical | Set1 | Maximum distinguishability |
| Binary (pass/fail) | RdYlGn | Intuitive red=bad, green=good |

## Appendix B: Terminal Capability Detection

```rust
fn detect_terminal_mode() -> TerminalMode {
    if std::env::var("COLORTERM").as_deref() == Ok("truecolor") {
        TerminalMode::TrueColor
    } else if std::env::var("TERM").map(|t| t.contains("256color")).unwrap_or(false) {
        TerminalMode::Ansi256
    } else if atty::is(atty::Stream::Stdout) {
        TerminalMode::Unicode
    } else {
        TerminalMode::Ascii  // CI/pipes
    }
}
```
