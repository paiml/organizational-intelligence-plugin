# Organizational Intelligence Plugin Guide

[Introduction](./introduction.md)

# Getting Started

- [Quick Start](./getting-started/quick-start.md)
- [Installation](./getting-started/installation.md)
- [Configuration](./getting-started/configuration.md)
- [First Analysis](./getting-started/first-analysis.md)

# Core Concepts

- [What is Defect Pattern Analysis?](./core-concepts/defect-pattern-analysis.md)
- [Organizational Intelligence](./core-concepts/organizational-intelligence.md)
- [18-Category Defect Taxonomy](./core-concepts/defect-taxonomy.md)
- [Three-Tier Classification](./core-concepts/three-tier-classification.md)

# Defect Categories

## General Defect Patterns

- [Memory Safety](./defect-categories/memory-safety.md)
- [Concurrency Bugs](./defect-categories/concurrency-bugs.md)
- [Type Errors](./defect-categories/type-errors.md)
- [Performance Issues](./defect-categories/performance-issues.md)
- [Security Vulnerabilities](./defect-categories/security-vulnerabilities.md)
- [Configuration Errors](./defect-categories/configuration-errors.md)
- [API Misuse](./defect-categories/api-misuse.md)
- [Integration Failures](./defect-categories/integration-failures.md)
- [Documentation Gaps](./defect-categories/documentation-gaps.md)
- [Testing Gaps](./defect-categories/testing-gaps.md)

## Transpiler-Specific Patterns

- [Operator Precedence](./defect-categories/operator-precedence.md)
- [Type Annotation Gaps](./defect-categories/type-annotation-gaps.md)
- [Stdlib Mapping](./defect-categories/stdlib-mapping.md)
- [AST Transform](./defect-categories/ast-transform.md)
- [Comprehension Bugs](./defect-categories/comprehension-bugs.md)
- [Iterator Chain](./defect-categories/iterator-chain.md)
- [Ownership/Borrow](./defect-categories/ownership-borrow.md)
- [Trait Bounds](./defect-categories/trait-bounds.md)

# CLI Usage

- [Analyze Command](./cli/analyze.md)
- [Extract Training Data](./cli/extract-training-data.md)
- [Train Classifier](./cli/train-classifier.md)
- [Summarize Reports](./cli/summarize.md)
- [Review Pull Requests](./cli/review-pr.md)

# ML Classification Pipeline

- [Overview](./ml-pipeline/overview.md)
- [TF-IDF Feature Extraction](./ml-pipeline/tfidf-features.md)
- [Random Forest Classifier](./ml-pipeline/random-forest.md)
- [Training Data Extraction](./ml-pipeline/training-data-extraction.md)
- [Model Training](./ml-pipeline/model-training.md)
- [Model Evaluation](./ml-pipeline/model-evaluation.md)
- [Hybrid Classification](./ml-pipeline/hybrid-classification.md)

# Three-Tier Architecture

- [Tier 1: Rule-Based (<10ms)](./architecture/tier1-rule-based.md)
- [Tier 2: TF-IDF + Random Forest (<100ms)](./architecture/tier2-tfidf-rf.md)
- [Tier 3: CodeBERT (Future, <1s)](./architecture/tier3-codebert.md)
- [Confidence-Based Routing](./architecture/confidence-routing.md)
- [Graceful Degradation](./architecture/graceful-degradation.md)

# GPU Acceleration

- [GPU vs CPU Performance](./gpu/gpu-vs-cpu.md)
- [Correlation Matrix Computation](./gpu/correlation-matrix.md)
- [Sliding Window Analysis](./gpu/sliding-window.md)
- [WebGPU Integration](./gpu/webgpu-integration.md)
- [Performance Benchmarks](./gpu/benchmarks.md)

# NLP Enhancement

- [Commit Message Processing](./nlp/commit-message-processing.md)
- [Tokenization and Stemming](./nlp/tokenization-stemming.md)
- [Stop Word Filtering](./nlp/stop-word-filtering.md)
- [N-Gram Extraction](./nlp/ngram-extraction.md)
- [Multi-Label Classification](./nlp/multi-label-classification.md)

# Real-World Examples

- [Example: Analyze Organization](./examples/analyze-organization.md)
- [Example: Classify Defects](./examples/classify-defects.md)
- [Example: Full Analysis Pipeline](./examples/full-analysis.md)
- [Example: Train Custom Model](./examples/train-custom-model.md)
- [Example: depyler Validation](./examples/depyler-validation.md)

# Validation Results

- [NLP-011: depyler Repository](./validation/nlp-011-depyler.md)
- [Performance Metrics](./validation/performance-metrics.md)
- [Accuracy Analysis](./validation/accuracy-analysis.md)
- [Class Imbalance](./validation/class-imbalance.md)
- [Overfitting Analysis](./validation/overfitting-analysis.md)

# Quality Gates

- [Pre-Commit Hooks](./quality-gates/pre-commit-hooks.md)
- [Code Formatting (rustfmt)](./quality-gates/code-formatting.md)
- [Linting (clippy)](./quality-gates/linting-clippy.md)
- [Test Coverage](./quality-gates/test-coverage.md)
- [Continuous Integration](./quality-gates/continuous-integration.md)

# EXTREME TDD Methodology

- [RED-GREEN-REFACTOR Cycle](./extreme-tdd/red-green-refactor.md)
- [Test-First Philosophy](./extreme-tdd/test-first-philosophy.md)
- [Property-Based Testing](./extreme-tdd/property-based-testing.md)
- [Mutation Testing](./extreme-tdd/mutation-testing.md)
- [Zero Tolerance Quality](./extreme-tdd/zero-tolerance.md)

# Toyota Way Principles

- [Kaizen (Continuous Improvement)](./toyota-way/kaizen.md)
- [Jidoka (Built-in Quality)](./toyota-way/jidoka.md)
- [Genchi Genbutsu (Go and See)](./toyota-way/genchi-genbutsu.md)
- [PDCA Cycle](./toyota-way/pdca-cycle.md)

# Sprint Development

- [Sprint v0.3.0: GPU Acceleration](./sprints/sprint-v030.md)
- [Sprint v0.4.0: NLP Enhancement](./sprints/sprint-v040.md)
- [Sprint v0.5.0: ML Integration](./sprints/sprint-v050.md)
- [Sprint Planning Process](./sprints/sprint-planning.md)
- [Sprint Retrospectives](./sprints/retrospectives.md)

# Advanced Topics

- [Class Imbalance Handling](./advanced/class-imbalance.md)
- [SMOTE Implementation](./advanced/smote.md)
- [Model Serialization](./advanced/model-serialization.md)
- [Active Learning](./advanced/active-learning.md)
- [Transfer Learning](./advanced/transfer-learning.md)

# Best Practices

- [Training Data Collection](./best-practices/training-data-collection.md)
- [Model Validation](./best-practices/model-validation.md)
- [Hyperparameter Tuning](./best-practices/hyperparameter-tuning.md)
- [Error Handling](./best-practices/error-handling.md)
- [API Design](./best-practices/api-design.md)

# Troubleshooting

- [Low Accuracy](./troubleshooting/low-accuracy.md)
- [Overfitting](./troubleshooting/overfitting.md)
- [Class Imbalance](./troubleshooting/class-imbalance.md)
- [Slow Inference](./troubleshooting/slow-inference.md)
- [Model Loading Errors](./troubleshooting/model-loading-errors.md)

# Appendix

- [Glossary](./appendix/glossary.md)
- [References](./appendix/references.md)
- [Further Reading](./appendix/further-reading.md)
- [Contributing](./appendix/contributing.md)
- [Changelog](./appendix/changelog.md)
