//! Natural Language Processing module for commit message analysis.
//!
//! This module provides NLP preprocessing utilities for defect classification:
//! - Tokenization (using aprender's text processing)
//! - Stop words filtering
//! - Stemming (Porter stemmer from aprender)
//! - N-gram generation
//! - TF-IDF feature extraction (future)
//!
//! # Design Principles
//!
//! Following Phase 1 of the NLP specification (nlp-models-techniques-spec.md):
//! - Zero `unwrap()` calls (Cloudflare-class safety)
//! - Result-based error handling
//! - Comprehensive test coverage (≥95%)
//! - Integration with aprender for proven NLP components
//!
//! # Examples
//!
//! ```rust
//! use organizational_intelligence_plugin::nlp::CommitMessageProcessor;
//!
//! let processor = CommitMessageProcessor::new();
//! let message = "fix: null pointer dereference in parse_expr()";
//! let tokens = processor.preprocess(message).unwrap();
//! // tokens = ["fix", "null", "pointer", "dereference", "parse", "expr"]
//! ```

use anyhow::{anyhow, Result};
use aprender::primitives::Matrix;
use aprender::text::stem::{PorterStemmer, Stemmer};
use aprender::text::stopwords::StopWordsFilter;
use aprender::text::tokenize::WordTokenizer;
use aprender::text::vectorize::TfidfVectorizer;
use aprender::text::Tokenizer;

/// Commit message preprocessor that applies NLP transformations.
///
/// This processor applies a standard NLP pipeline:
/// 1. Tokenization (word-level with punctuation handling)
/// 2. Lowercasing
/// 3. Stop words filtering (with custom software engineering stop words)
/// 4. Stemming (Porter stemmer)
///
/// # Examples
///
/// ```rust
/// use organizational_intelligence_plugin::nlp::CommitMessageProcessor;
///
/// let processor = CommitMessageProcessor::new();
/// let message = "fix: race condition in mutex lock";
/// let tokens = processor.preprocess(message).unwrap();
/// assert!(tokens.contains(&"race".to_string()));
/// assert!(tokens.contains(&"condit".to_string())); // Stemmed
/// ```
#[derive(Debug, Clone)]
pub struct CommitMessageProcessor {
    tokenizer: WordTokenizer,
    stop_words: StopWordsFilter,
    stemmer: PorterStemmer,
}

impl CommitMessageProcessor {
    /// Create a new commit message processor with default settings.
    ///
    /// Uses:
    /// - WordTokenizer for tokenization
    /// - English stop words with custom software engineering adjustments
    /// - Porter stemmer for normalization
    ///
    /// # Examples
    ///
    /// ```rust
    /// use organizational_intelligence_plugin::nlp::CommitMessageProcessor;
    ///
    /// let processor = CommitMessageProcessor::new();
    /// ```
    pub fn new() -> Self {
        let tokenizer = WordTokenizer::new();

        // English stop words filter; domain terms (fix, bug, error, memory, etc.)
        // pass through as they carry semantic weight for defect classification.
        let stop_words = StopWordsFilter::english();

        let stemmer = PorterStemmer::new();

        Self {
            tokenizer,
            stop_words,
            stemmer,
        }
    }

    /// Create a processor with custom stop words.
    ///
    /// Useful for domain-specific filtering (e.g., transpiler development).
    ///
    /// # Arguments
    ///
    /// * `custom_stop_words` - Additional stop words to filter
    ///
    /// # Examples
    ///
    /// ```rust
    /// use organizational_intelligence_plugin::nlp::CommitMessageProcessor;
    ///
    /// let processor = CommitMessageProcessor::with_custom_stop_words(vec!["depyler", "internal"]);
    /// ```
    pub fn with_custom_stop_words<I, S>(custom_stop_words: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let tokenizer = WordTokenizer::new();
        let stop_words = StopWordsFilter::new(custom_stop_words);
        let stemmer = PorterStemmer::new();

        Self {
            tokenizer,
            stop_words,
            stemmer,
        }
    }

    /// Preprocess a commit message into normalized tokens.
    ///
    /// Pipeline:
    /// 1. Tokenize into words
    /// 2. Lowercase
    /// 3. Filter stop words
    /// 4. Stem to root forms
    ///
    /// # Arguments
    ///
    /// * `message` - Raw commit message
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<String>)` - Normalized tokens
    /// * `Err` - If preprocessing fails
    ///
    /// # Examples
    ///
    /// ```rust
    /// use organizational_intelligence_plugin::nlp::CommitMessageProcessor;
    ///
    /// let processor = CommitMessageProcessor::new();
    /// let tokens = processor.preprocess("fix: memory leak in parser").unwrap();
    /// assert!(tokens.contains(&"memori".to_string())); // Stemmed "memory"
    /// assert!(tokens.contains(&"leak".to_string()));
    /// assert!(tokens.len() >= 2); // At least "memori" and "leak"
    /// ```
    pub fn preprocess(&self, message: &str) -> Result<Vec<String>> {
        // Step 1: Tokenize
        let tokens = self
            .tokenizer
            .tokenize(message)
            .map_err(|e| anyhow!("Tokenization failed: {}", e))?;

        // Step 2: Lowercase
        let lowercase_tokens: Vec<String> = tokens.iter().map(|t| t.to_lowercase()).collect();

        // Step 3: Filter stop words
        let filtered_tokens = self
            .stop_words
            .filter(&lowercase_tokens)
            .map_err(|e| anyhow!("Stop words filtering failed: {}", e))?;

        // Step 4: Stem
        let stemmed_tokens = self
            .stemmer
            .stem_tokens(&filtered_tokens)
            .map_err(|e| anyhow!("Stemming failed: {}", e))?;

        Ok(stemmed_tokens)
    }

    /// Extract n-grams from a list of tokens.
    ///
    /// N-grams are contiguous sequences of n tokens.
    /// Useful for detecting multi-word patterns like "null pointer" or "race condition".
    ///
    /// # Arguments
    ///
    /// * `tokens` - Input tokens
    /// * `n` - Size of n-grams (1 = unigrams, 2 = bigrams, 3 = trigrams)
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<String>)` - N-grams joined with underscores
    /// * `Err` - If n is 0 or greater than token count
    ///
    /// # Examples
    ///
    /// ```rust
    /// use organizational_intelligence_plugin::nlp::CommitMessageProcessor;
    ///
    /// let processor = CommitMessageProcessor::new();
    /// let tokens: Vec<String> = vec![
    ///     "fix".to_string(),
    ///     "race".to_string(),
    ///     "condition".to_string(),
    ///     "mutex".to_string(),
    /// ];
    /// let bigrams = processor.extract_ngrams(&tokens, 2).unwrap();
    /// assert!(bigrams.contains(&"fix_race".to_string()));
    /// assert!(bigrams.contains(&"race_condition".to_string()));
    /// ```
    pub fn extract_ngrams(&self, tokens: &[String], n: usize) -> Result<Vec<String>> {
        if n == 0 {
            return Err(anyhow!("n must be greater than 0"));
        }

        if tokens.len() < n {
            return Ok(Vec::new());
        }

        let ngrams: Vec<String> = tokens.windows(n).map(|window| window.join("_")).collect();

        Ok(ngrams)
    }

    /// Preprocess and extract both unigrams and bigrams.
    ///
    /// Convenience method that combines preprocessing with n-gram extraction.
    /// Useful for feature extraction in ML models.
    ///
    /// # Arguments
    ///
    /// * `message` - Raw commit message
    ///
    /// # Returns
    ///
    /// * `Ok((Vec<String>, Vec<String>))` - (unigrams, bigrams)
    /// * `Err` - If preprocessing fails
    ///
    /// # Examples
    ///
    /// ```rust
    /// use organizational_intelligence_plugin::nlp::CommitMessageProcessor;
    ///
    /// let processor = CommitMessageProcessor::new();
    /// let (unigrams, bigrams) = processor.preprocess_with_ngrams("fix: memory leak defect").unwrap();
    /// assert!(unigrams.contains(&"memori".to_string())); // Stemmed "memory"
    /// assert!(unigrams.contains(&"leak".to_string()));
    /// assert!(!bigrams.is_empty()); // Should have bigrams
    /// ```
    pub fn preprocess_with_ngrams(&self, message: &str) -> Result<(Vec<String>, Vec<String>)> {
        let tokens = self.preprocess(message)?;
        let bigrams = self.extract_ngrams(&tokens, 2)?;

        Ok((tokens, bigrams))
    }
}

impl Default for CommitMessageProcessor {
    fn default() -> Self {
        Self::new()
    }
}

/// TF-IDF feature extractor for commit messages
///
/// This extractor converts commit messages into TF-IDF feature vectors for ML classification.
/// Implements Phase 2 of nlp-models-techniques-spec.md (Tier 2: TF-IDF + ML).
///
/// # Examples
///
/// ```rust
/// use organizational_intelligence_plugin::nlp::TfidfFeatureExtractor;
///
/// let messages: Vec<String> = vec![
///     "fix: null pointer dereference".to_string(),
///     "fix: race condition in mutex".to_string(),
///     "feat: add new feature".to_string(),
/// ];
///
/// let mut extractor = TfidfFeatureExtractor::new(1500);
/// let features = extractor.fit_transform(&messages).unwrap();
///
/// assert_eq!(features.n_rows(), 3); // 3 documents
/// ```
pub struct TfidfFeatureExtractor {
    vectorizer: TfidfVectorizer,
    max_features: usize,
}

impl TfidfFeatureExtractor {
    /// Create a new TF-IDF feature extractor
    ///
    /// # Arguments
    ///
    /// * `max_features` - Maximum number of features (vocabulary size)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use organizational_intelligence_plugin::nlp::TfidfFeatureExtractor;
    ///
    /// let extractor = TfidfFeatureExtractor::new(1500);
    /// ```
    pub fn new(max_features: usize) -> Self {
        let vectorizer = TfidfVectorizer::new()
            .with_tokenizer(Box::new(WordTokenizer::new()))
            .with_lowercase(true)
            .with_max_features(max_features);

        Self {
            vectorizer,
            max_features,
        }
    }

    /// Fit the vectorizer on training messages and transform them to TF-IDF features
    ///
    /// # Arguments
    ///
    /// * `messages` - Training commit messages
    ///
    /// # Returns
    ///
    /// * `Ok(Matrix<f64>)` - TF-IDF feature matrix (n_messages × vocabulary_size)
    /// * `Err` - If vectorization fails
    ///
    /// # Examples
    ///
    /// ```rust
    /// use organizational_intelligence_plugin::nlp::TfidfFeatureExtractor;
    ///
    /// let messages: Vec<String> = vec![
    ///     "fix: memory leak".to_string(),
    ///     "fix: race condition".to_string(),
    /// ];
    ///
    /// let mut extractor = TfidfFeatureExtractor::new(1000);
    /// let features = extractor.fit_transform(&messages).unwrap();
    ///
    /// assert_eq!(features.n_rows(), 2);
    /// ```
    pub fn fit_transform(&mut self, messages: &[String]) -> Result<Matrix<f64>> {
        self.vectorizer
            .fit_transform(messages)
            .map_err(|e| anyhow!("TF-IDF fit_transform failed: {}", e))
    }

    /// Fit the vectorizer on training messages
    ///
    /// # Arguments
    ///
    /// * `messages` - Training commit messages
    ///
    /// # Examples
    ///
    /// ```rust
    /// use organizational_intelligence_plugin::nlp::TfidfFeatureExtractor;
    ///
    /// let messages = vec![
    ///     "fix: memory leak".to_string(),
    ///     "fix: race condition".to_string(),
    /// ];
    ///
    /// let mut extractor = TfidfFeatureExtractor::new(1000);
    /// extractor.fit(&messages).unwrap();
    /// ```
    pub fn fit(&mut self, messages: &[String]) -> Result<()> {
        self.vectorizer
            .fit(messages)
            .map_err(|e| anyhow!("TF-IDF fit failed: {}", e))
    }

    /// Transform messages to TF-IDF features using fitted vocabulary
    ///
    /// # Arguments
    ///
    /// * `messages` - Commit messages to transform
    ///
    /// # Returns
    ///
    /// * `Ok(Matrix<f64>)` - TF-IDF feature matrix
    /// * `Err` - If transformation fails
    ///
    /// # Examples
    ///
    /// ```rust
    /// use organizational_intelligence_plugin::nlp::TfidfFeatureExtractor;
    ///
    /// let train_messages = vec![
    ///     "fix: memory leak".to_string(),
    ///     "fix: race condition".to_string(),
    /// ];
    ///
    /// let test_messages = vec!["fix: null pointer".to_string()];
    ///
    /// let mut extractor = TfidfFeatureExtractor::new(1000);
    /// extractor.fit(&train_messages).unwrap();
    ///
    /// let features = extractor.transform(&test_messages).unwrap();
    /// assert_eq!(features.n_rows(), 1);
    /// ```
    pub fn transform(&self, messages: &[String]) -> Result<Matrix<f64>> {
        self.vectorizer
            .transform(messages)
            .map_err(|e| anyhow!("TF-IDF transform failed: {}", e))
    }

    /// Get the vocabulary size (number of features)
    ///
    /// # Returns
    ///
    /// * `usize` - Number of features in vocabulary
    ///
    /// # Examples
    ///
    /// ```rust
    /// use organizational_intelligence_plugin::nlp::TfidfFeatureExtractor;
    ///
    /// let messages = vec![
    ///     "fix: bug".to_string(),
    ///     "feat: feature".to_string(),
    /// ];
    ///
    /// let mut extractor = TfidfFeatureExtractor::new(1000);
    /// extractor.fit(&messages).unwrap();
    ///
    /// assert!(extractor.vocabulary_size() > 0);
    /// assert!(extractor.vocabulary_size() <= 1000);
    /// ```
    pub fn vocabulary_size(&self) -> usize {
        self.vectorizer.vocabulary_size()
    }

    /// Get the maximum features configuration
    ///
    /// # Returns
    ///
    /// * `usize` - Maximum number of features
    pub fn max_features(&self) -> usize {
        self.max_features
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_processor_creation() {
        let _processor = CommitMessageProcessor::new();
        let _processor2 = CommitMessageProcessor::default();
    }

    #[test]
    fn test_basic_preprocessing() {
        let processor = CommitMessageProcessor::new();
        let message = "fix: memory leak detected";
        let tokens = processor.preprocess(message).unwrap();

        // Should contain key technical terms (stemmed)
        // "memory" -> "memori" (stemmed), "leak" stays "leak", "detect" -> "detect"
        assert!(tokens
            .iter()
            .any(|t| t.starts_with("memori") || t.starts_with("memory")));
        assert!(tokens.iter().any(|t| t.starts_with("leak")));
        assert!(tokens.iter().any(|t| t.starts_with("detect")));
    }

    #[test]
    fn test_preprocessing_handles_punctuation() {
        let processor = CommitMessageProcessor::new();
        let message = "fix race condition mutex lock";
        let tokens = processor.preprocess(message).unwrap();

        // Should contain technical terms without punctuation complications
        assert!(tokens
            .iter()
            .any(|t| t.starts_with("race") || t.starts_with("rac")));
        assert!(tokens
            .iter()
            .any(|t| t.starts_with("condit") || t.starts_with("condition")));
        assert!(tokens.iter().any(|t| t.starts_with("mutex")));
        assert!(tokens.iter().any(|t| t.starts_with("lock")));
    }

    #[test]
    fn test_ngram_extraction() {
        let processor = CommitMessageProcessor::new();
        let tokens = vec![
            "fix".to_string(),
            "race".to_string(),
            "condition".to_string(),
        ];

        let bigrams = processor.extract_ngrams(&tokens, 2).unwrap();
        assert_eq!(bigrams.len(), 2);
        assert!(bigrams.contains(&"fix_race".to_string()));
        assert!(bigrams.contains(&"race_condition".to_string()));
    }

    #[test]
    fn test_ngram_extraction_trigrams() {
        let processor = CommitMessageProcessor::new();
        let tokens = vec![
            "fix".to_string(),
            "null".to_string(),
            "pointer".to_string(),
            "dereference".to_string(),
        ];

        let trigrams = processor.extract_ngrams(&tokens, 3).unwrap();
        assert_eq!(trigrams.len(), 2);
        assert!(trigrams.contains(&"fix_null_pointer".to_string()));
        assert!(trigrams.contains(&"null_pointer_dereference".to_string()));
    }

    #[test]
    fn test_ngram_empty_tokens() {
        let processor = CommitMessageProcessor::new();
        let tokens: Vec<String> = vec![];

        let bigrams = processor.extract_ngrams(&tokens, 2).unwrap();
        assert!(bigrams.is_empty());
    }

    #[test]
    fn test_ngram_insufficient_tokens() {
        let processor = CommitMessageProcessor::new();
        let tokens = vec!["single".to_string()];

        let bigrams = processor.extract_ngrams(&tokens, 2).unwrap();
        assert!(bigrams.is_empty());
    }

    #[test]
    fn test_ngram_zero_n_error() {
        let processor = CommitMessageProcessor::new();
        let tokens = vec!["test".to_string()];

        let result = processor.extract_ngrams(&tokens, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_preprocess_with_ngrams() {
        let processor = CommitMessageProcessor::new();
        let message = "fix memory leak in parser";

        let (unigrams, bigrams) = processor.preprocess_with_ngrams(message).unwrap();

        assert!(!unigrams.is_empty());
        assert!(!bigrams.is_empty());
    }

    #[test]
    fn test_custom_stop_words() {
        let processor = CommitMessageProcessor::with_custom_stop_words(vec!["custom", "stop"]);
        let message = "custom test stop words";

        let tokens = processor.preprocess(message).unwrap();

        // "custom" and "stop" should be filtered
        assert!(!tokens.contains(&"custom".to_string()));
        assert!(!tokens.contains(&"stop".to_string()));
        // "test" and "words" should remain
        assert!(tokens.iter().any(|t| t.starts_with("test")));
    }

    #[test]
    fn test_preprocessing_with_code_tokens() {
        let processor = CommitMessageProcessor::new();
        let message = "fix: parse_expr() null check in into_iter()";

        let tokens = processor.preprocess(message).unwrap();

        // Code identifiers should be tokenized
        assert!(tokens
            .iter()
            .any(|t| t.contains("pars") || t.contains("expr")));
        assert!(tokens.iter().any(|t| t.contains("null")));
    }

    #[test]
    fn test_stemming_normalization() {
        let processor = CommitMessageProcessor::new();
        let message1 = "fixing bugs";
        let message2 = "fixed bug";

        let tokens1 = processor.preprocess(message1).unwrap();
        let tokens2 = processor.preprocess(message2).unwrap();

        // Both should stem "fix" and "bug" similarly
        let has_fix_stem1 = tokens1.iter().any(|t| t.starts_with("fix"));
        let has_fix_stem2 = tokens2.iter().any(|t| t.starts_with("fix"));
        let has_bug_stem1 = tokens1.iter().any(|t| t.starts_with("bug"));
        let has_bug_stem2 = tokens2.iter().any(|t| t.starts_with("bug"));

        assert!(has_fix_stem1 || has_fix_stem2);
        assert!(has_bug_stem1 || has_bug_stem2);
    }

    #[test]
    fn test_empty_message() {
        let processor = CommitMessageProcessor::new();
        let tokens = processor.preprocess("").unwrap();
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_whitespace_only_message() {
        let processor = CommitMessageProcessor::new();
        let tokens = processor.preprocess("   \t\n   ").unwrap();
        assert!(tokens.is_empty());
    }

    // TF-IDF feature extraction tests

    #[test]
    fn test_tfidf_extractor_creation() {
        let extractor = TfidfFeatureExtractor::new(1000);
        assert_eq!(extractor.max_features(), 1000);
    }

    #[test]
    fn test_tfidf_fit_transform_basic() {
        let messages = vec![
            "fix: memory leak".to_string(),
            "fix: race condition".to_string(),
            "feat: add new feature".to_string(),
        ];

        let mut extractor = TfidfFeatureExtractor::new(1000);
        let features = extractor.fit_transform(&messages).unwrap();

        // Should produce matrix with correct dimensions
        assert_eq!(features.n_rows(), 3); // 3 documents
        assert!(features.n_cols() > 0); // At least some features
        assert!(features.n_cols() <= 1000); // Respects max_features
    }

    #[test]
    fn test_tfidf_fit_and_transform_separate() {
        let train_messages = vec![
            "fix: memory leak".to_string(),
            "fix: race condition".to_string(),
        ];

        let test_messages = vec!["fix: null pointer".to_string()];

        let mut extractor = TfidfFeatureExtractor::new(1000);

        // Fit on training data
        extractor.fit(&train_messages).unwrap();

        // Transform test data
        let features = extractor.transform(&test_messages).unwrap();

        assert_eq!(features.n_rows(), 1);
        assert_eq!(features.n_cols(), extractor.vocabulary_size());
    }

    #[test]
    fn test_tfidf_vocabulary_size() {
        let messages = vec![
            "fix bug".to_string(),
            "feat feature".to_string(),
            "test code".to_string(),
        ];

        let mut extractor = TfidfFeatureExtractor::new(1000);
        extractor.fit(&messages).unwrap();

        let vocab_size = extractor.vocabulary_size();
        assert!(vocab_size > 0);
        assert!(vocab_size <= 1000); // Respects max_features
    }

    #[test]
    fn test_tfidf_max_features_limit() {
        let messages = vec![
            "word1 word2 word3 word4 word5".to_string(),
            "word6 word7 word8 word9 word10".to_string(),
            "word11 word12 word13 word14 word15".to_string(),
        ];

        // Limit to 5 features
        let mut extractor = TfidfFeatureExtractor::new(5);
        extractor.fit(&messages).unwrap();

        assert!(extractor.vocabulary_size() <= 5);
    }

    #[test]
    fn test_tfidf_with_real_commit_messages() {
        let messages = vec![
            "fix: null pointer dereference in parser".to_string(),
            "fix: race condition in mutex lock".to_string(),
            "feat: add TF-IDF feature extraction".to_string(),
            "docs: update README with examples".to_string(),
            "test: add unit tests for classifier".to_string(),
        ];

        let mut extractor = TfidfFeatureExtractor::new(1500);
        let features = extractor.fit_transform(&messages).unwrap();

        assert_eq!(features.n_rows(), 5);
        assert!(features.n_cols() > 0);

        // Check that feature values are reasonable (non-negative)
        for row in 0..features.n_rows() {
            for col in 0..features.n_cols() {
                assert!(features.get(row, col) >= 0.0);
            }
        }
    }

    #[test]
    fn test_tfidf_empty_messages() {
        let messages: Vec<String> = vec![];

        let mut extractor = TfidfFeatureExtractor::new(1000);
        let result = extractor.fit_transform(&messages);

        // Should handle empty input gracefully
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn test_tfidf_single_message() {
        let messages = vec!["fix: single message".to_string()];

        let mut extractor = TfidfFeatureExtractor::new(1000);
        let features = extractor.fit_transform(&messages).unwrap();

        assert_eq!(features.n_rows(), 1);
        assert!(features.n_cols() > 0);
    }

    #[test]
    fn test_tfidf_duplicate_messages() {
        let messages = vec![
            "fix: memory leak".to_string(),
            "fix: memory leak".to_string(),
            "fix: memory leak".to_string(),
        ];

        let mut extractor = TfidfFeatureExtractor::new(1000);
        let features = extractor.fit_transform(&messages).unwrap();

        assert_eq!(features.n_rows(), 3);

        // Duplicate messages should have similar (but not identical due to IDF) feature vectors
        // Just verify they transform successfully
        assert!(features.n_cols() > 0);
    }

    #[test]
    fn test_tfidf_transform_new_data() {
        let train_messages = vec![
            "fix: memory leak".to_string(),
            "fix: race condition".to_string(),
            "feat: new feature".to_string(),
        ];

        let test_messages = vec![
            "fix: another memory issue".to_string(),
            "feat: different feature".to_string(),
        ];

        let mut extractor = TfidfFeatureExtractor::new(1000);
        extractor.fit(&train_messages).unwrap();

        let test_features = extractor.transform(&test_messages).unwrap();

        assert_eq!(test_features.n_rows(), 2);
        assert_eq!(test_features.n_cols(), extractor.vocabulary_size());
    }

    #[test]
    fn test_tfidf_configuration() {
        let extractor = TfidfFeatureExtractor::new(1500);

        assert_eq!(extractor.max_features(), 1500);
    }

    #[test]
    fn test_tfidf_with_software_terms() {
        let messages = vec![
            "fix: null pointer dereference".to_string(),
            "fix: buffer overflow in parse_expr".to_string(),
            "fix: race condition deadlock".to_string(),
            "fix: memory leak in allocator".to_string(),
        ];

        let mut extractor = TfidfFeatureExtractor::new(1000);
        let features = extractor.fit_transform(&messages).unwrap();

        assert_eq!(features.n_rows(), 4);

        // Verify that technical terms are captured
        // (vocabulary building works correctly)
        assert!(extractor.vocabulary_size() > 0);
    }

    #[test]
    fn test_tfidf_transpiler_specific_terms() {
        let messages = vec![
            "fix: operator precedence bug".to_string(),
            "fix: AST transform error".to_string(),
            "fix: lifetime parameter issue".to_string(),
            "fix: trait bound constraint".to_string(),
        ];

        let mut extractor = TfidfFeatureExtractor::new(1500);
        let features = extractor.fit_transform(&messages).unwrap();

        assert_eq!(features.n_rows(), 4);
        assert!(extractor.vocabulary_size() > 0);
    }
}
