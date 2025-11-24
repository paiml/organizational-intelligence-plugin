CODE QUALITY REPORT
===================

Generated: 2025-11-24 17:07:00 UTC
Project: .
Total Defects: 31
Files Analyzed: 8

SEVERITY BREAKDOWN
------------------
high       9
medium     22

CATEGORY BREAKDOWN
------------------
Performance          31

TOP HOTSPOT FILES
-----------------
1. ./src/ml.rs (7 defects, score: 35.0)
2. ./src/sliding_window.rs (9 defects, score: 31.0)
3. ./src/summarizer.rs (6 defects, score: 18.0)
4. ./src/correlation.rs (4 defects, score: 12.0)
5. ./benches/gpu_benchmarks.rs (2 defects, score: 6.0)
6. ./src/gpu_main.rs (1 defects, score: 3.0)
7. ./src/perf.rs (1 defects, score: 3.0)
8. ./src/classifier.rs (1 defects, score: 3.0)

DEFECTS
-------
[Medium] Performance - ./benches/gpu_benchmarks.rs:15
  Function 'bench_pearson_correlation' has high time complexity: O(n²)
  Fix: Consider using a more efficient algorithm or data structure to reduce quadratic complexity

[Medium] Performance - ./benches/gpu_benchmarks.rs:35
  Function 'bench_feature_extraction' has high time complexity: O(n²)
  Fix: Consider using a more efficient algorithm or data structure to reduce quadratic complexity

[Medium] Performance - ./src/classifier.rs:253
  Function 'classify_from_message' has high time complexity: O(n²)
  Fix: Consider using a more efficient algorithm or data structure to reduce quadratic complexity

[Medium] Performance - ./src/correlation.rs:77
  Function 'new' has high time complexity: O(n²)
  Fix: Consider using a more efficient algorithm or data structure to reduce quadratic complexity

[Medium] Performance - ./src/correlation.rs:82
  Function 'with_capacity' has high time complexity: O(n²)
  Fix: Consider using a more efficient algorithm or data structure to reduce quadratic complexity

[Medium] Performance - ./src/correlation.rs:94
  Function 'correlate' has high time complexity: O(n²)
  Fix: Consider using a more efficient algorithm or data structure to reduce quadratic complexity

[Medium] Performance - ./src/correlation.rs:128
  Function 'correlation_matrix' has high time complexity: O(n²)
  Fix: Consider using a more efficient algorithm or data structure to reduce quadratic complexity

[Medium] Performance - ./src/gpu_main.rs:347
  Function 'cmd_analyze' has high time complexity: O(n²)
  Fix: Consider using a more efficient algorithm or data structure to reduce quadratic complexity

[Medium] Performance - ./src/perf.rs:299
  Function 'format_bytes' has high time complexity: O(n²)
  Fix: Consider using a more efficient algorithm or data structure to reduce quadratic complexity

[Medium] Performance - ./src/sliding_window.rs:32
  Function 'six_months_from' has high time complexity: O(n²)
  Fix: Consider using a more efficient algorithm or data structure to reduce quadratic complexity

[Medium] Performance - ./src/sliding_window.rs:40
  Function 'contains' has high time complexity: O(n²)
  Fix: Consider using a more efficient algorithm or data structure to reduce quadratic complexity

[Medium] Performance - ./src/sliding_window.rs:45
  Function 'duration' has high time complexity: O(n²)
  Fix: Consider using a more efficient algorithm or data structure to reduce quadratic complexity

[Medium] Performance - ./src/sliding_window.rs:66
  Function 'new_six_month' has high time complexity: O(n²)
  Fix: Consider using a more efficient algorithm or data structure to reduce quadratic complexity

[Medium] Performance - ./src/sliding_window.rs:74
  Function 'new' has high time complexity: O(n²)
  Fix: Consider using a more efficient algorithm or data structure to reduce quadratic complexity

[Medium] Performance - ./src/sliding_window.rs:82
  Function 'generate_windows' has high time complexity: O(n²)
  Fix: Consider using a more efficient algorithm or data structure to reduce quadratic complexity

[Medium] Performance - ./src/sliding_window.rs:100
  Function 'compute_window_correlation' has high time complexity: O(n²)
  Fix: Consider using a more efficient algorithm or data structure to reduce quadratic complexity

[Medium] Performance - ./src/summarizer.rs:74
  Function 'find_category' has high time complexity: O(n²)
  Fix: Consider using a more efficient algorithm or data structure to reduce quadratic complexity

[Medium] Performance - ./src/summarizer.rs:90
  Function 'from_file' has high time complexity: O(n²)
  Fix: Consider using a more efficient algorithm or data structure to reduce quadratic complexity

[Medium] Performance - ./src/summarizer.rs:119
  Function 'summarize' has high time complexity: O(n²)
  Fix: Consider using a more efficient algorithm or data structure to reduce quadratic complexity

[Medium] Performance - ./src/summarizer.rs:164
  Function 'strip_pii_from_patterns' has high time complexity: O(n²)
  Fix: Consider using a more efficient algorithm or data structure to reduce quadratic complexity

[Medium] Performance - ./src/summarizer.rs:189
  Function 'create_test_report' has high time complexity: O(n²)
  Fix: Consider using a more efficient algorithm or data structure to reduce quadratic complexity

[Medium] Performance - ./src/summarizer.rs:258
  Function 'test_pii_stripping_removes_sensitive_data' has high time complexity: O(n²)
  Fix: Consider using a more efficient algorithm or data structure to reduce quadratic complexity

[High] Performance - ./src/ml.rs:134
  Function 'params' has high time complexity: O(n³)
  Fix: Cubic complexity is rarely acceptable; consider algorithmic improvements

[High] Performance - ./src/ml.rs:139
  Function 'is_trained' has high time complexity: O(n³)
  Fix: Cubic complexity is rarely acceptable; consider algorithmic improvements

[High] Performance - ./src/ml.rs:143
  Function 'euclidean_distance' has high time complexity: O(n³)
  Fix: Cubic complexity is rarely acceptable; consider algorithmic improvements

[High] Performance - ./src/ml.rs:153
  Function 'default' has high time complexity: O(n³)
  Fix: Cubic complexity is rarely acceptable; consider algorithmic improvements

[High] Performance - ./src/ml.rs:170
  Function 'new' has high time complexity: O(n³)
  Fix: Cubic complexity is rarely acceptable; consider algorithmic improvements

[High] Performance - ./src/ml.rs:180
  Function 'with_k' has high time complexity: O(n³)
  Fix: Cubic complexity is rarely acceptable; consider algorithmic improvements

[High] Performance - ./src/ml.rs:190
  Function 'fit' has high time complexity: O(n³)
  Fix: Cubic complexity is rarely acceptable; consider algorithmic improvements

[High] Performance - ./src/sliding_window.rs:151
  Function 'compute_all_windows' has high time complexity: O(n³)
  Fix: Cubic complexity is rarely acceptable; consider algorithmic improvements

[High] Performance - ./src/sliding_window.rs:198
  Function 'detect_drift' has high time complexity: O(n³)
  Fix: Cubic complexity is rarely acceptable; consider algorithmic improvements

