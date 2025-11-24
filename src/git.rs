// Git history analyzer
// Phase 1: Clone repositories and analyze commit history for defect patterns
// Toyota Way: Simple local cloning, can evolve to distributed if metrics show need

use anyhow::{anyhow, Result};
use git2::Repository;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use tracing::{debug, info};

/// Information about a single commit with quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommitInfo {
    pub hash: String,
    pub message: String,
    pub author: String,
    pub timestamp: i64,
    /// Number of files changed in this commit
    pub files_changed: usize,
    /// Lines added
    pub lines_added: usize,
    /// Lines removed
    pub lines_removed: usize,
}

/// Git repository analyzer
/// Clones and analyzes git repositories to extract commit history
pub struct GitAnalyzer {
    cache_dir: PathBuf,
}

impl GitAnalyzer {
    /// Create a new GitAnalyzer with specified cache directory
    ///
    /// # Arguments
    /// * `cache_dir` - Directory to store cloned repositories
    ///
    /// # Examples
    /// ```
    /// use organizational_intelligence_plugin::git::GitAnalyzer;
    /// use std::path::PathBuf;
    ///
    /// let analyzer = GitAnalyzer::new(PathBuf::from("/tmp/repos"));
    /// ```
    pub fn new<P: AsRef<Path>>(cache_dir: P) -> Self {
        let cache_dir = cache_dir.as_ref().to_path_buf();
        Self { cache_dir }
    }

    /// Clone a repository to the cache directory
    ///
    /// # Arguments
    /// * `repo_url` - Git repository URL (https)
    /// * `name` - Local name for the repository
    ///
    /// # Returns
    /// * `Ok(())` if successful
    /// * `Err` if clone fails
    ///
    /// # Examples
    /// ```no_run
    /// # use organizational_intelligence_plugin::git::GitAnalyzer;
    /// # use std::path::PathBuf;
    /// # async fn example() -> Result<(), anyhow::Error> {
    /// let analyzer = GitAnalyzer::new(PathBuf::from("/tmp/repos"));
    /// analyzer.clone_repository("https://github.com/rust-lang/rust", "rust")?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn clone_repository(&self, repo_url: &str, name: &str) -> Result<()> {
        let repo_path = self.cache_dir.join(name);

        // Skip if already cloned
        if repo_path.exists() {
            debug!("Repository {} already exists at {:?}", name, repo_path);
            return Ok(());
        }

        info!("Cloning repository {} from {}", name, repo_url);

        // Clone the repository
        Repository::clone(repo_url, &repo_path).map_err(|e| {
            anyhow!(
                "Failed to clone repository {} from {}: {}",
                name,
                repo_url,
                e
            )
        })?;

        info!("Successfully cloned {} to {:?}", name, repo_path);
        Ok(())
    }

    /// Analyze commits in a cloned repository
    ///
    /// # Arguments
    /// * `name` - Repository name (must be already cloned)
    /// * `limit` - Maximum number of commits to analyze
    ///
    /// # Returns
    /// * `Ok(Vec<CommitInfo>)` with commit information
    /// * `Err` if repository not found or analysis fails
    ///
    /// # Examples
    /// ```no_run
    /// # use organizational_intelligence_plugin::git::GitAnalyzer;
    /// # use std::path::PathBuf;
    /// # async fn example() -> Result<(), anyhow::Error> {
    /// let analyzer = GitAnalyzer::new(PathBuf::from("/tmp/repos"));
    /// analyzer.clone_repository("https://github.com/rust-lang/rust", "rust")?;
    /// let commits = analyzer.analyze_commits("rust", 100)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn analyze_commits(&self, name: &str, limit: usize) -> Result<Vec<CommitInfo>> {
        let repo_path = self.cache_dir.join(name);

        if !repo_path.exists() {
            return Err(anyhow!(
                "Repository {} not found at {:?}. Clone it first.",
                name,
                repo_path
            ));
        }

        debug!("Opening repository at {:?}", repo_path);
        let repo = Repository::open(&repo_path)
            .map_err(|e| anyhow!("Failed to open repository {}: {}", name, e))?;

        let mut revwalk = repo.revwalk()?;
        revwalk.push_head()?;

        let mut commits = Vec::new();

        for (i, oid) in revwalk.enumerate() {
            if i >= limit {
                break;
            }

            let oid = oid?;
            let commit = repo.find_commit(oid)?;

            let hash = commit.id().to_string();
            let message = commit.message().unwrap_or("").to_string();
            let author = commit.author().email().unwrap_or("unknown").to_string();
            let timestamp = commit.time().seconds();

            // Get diff stats
            let (files_changed, lines_added, lines_removed) = if commit.parent_count() > 0 {
                let parent = commit.parent(0)?;
                let diff =
                    repo.diff_tree_to_tree(Some(&parent.tree()?), Some(&commit.tree()?), None)?;
                let stats = diff.stats()?;
                (stats.files_changed(), stats.insertions(), stats.deletions())
            } else {
                // Initial commit - count all files as changed
                let tree = commit.tree()?;
                (tree.len(), 0, 0)
            };

            commits.push(CommitInfo {
                hash,
                message,
                author,
                timestamp,
                files_changed,
                lines_added,
                lines_removed,
            });
        }

        debug!("Analyzed {} commits from {}", commits.len(), name);
        Ok(commits)
    }
}

/// Analyze commits from an existing Git repository path
///
/// Unlike GitAnalyzer which requires cloning repos to a cache directory,
/// this function works with any existing Git repository.
///
/// # Arguments
/// * `repo_path` - Path to existing Git repository
/// * `limit` - Maximum number of commits to analyze
///
/// # Returns
/// * `Ok(Vec<CommitInfo>)` with commit information
/// * `Err` if repository not found or analysis fails
///
/// # Examples
/// ```no_run
/// use organizational_intelligence_plugin::git::analyze_repository_at_path;
/// use std::path::PathBuf;
///
/// let commits = analyze_repository_at_path(PathBuf::from("/path/to/repo"), 100).unwrap();
/// ```
pub fn analyze_repository_at_path<P: AsRef<Path>>(
    repo_path: P,
    limit: usize,
) -> Result<Vec<CommitInfo>> {
    let repo_path = repo_path.as_ref();

    if !repo_path.exists() {
        return Err(anyhow!("Repository path does not exist: {:?}", repo_path));
    }

    debug!("Opening repository at {:?}", repo_path);
    let repo = Repository::open(repo_path)
        .map_err(|e| anyhow!("Failed to open repository at {:?}: {}", repo_path, e))?;

    let mut revwalk = repo.revwalk()?;
    revwalk.push_head()?;

    let mut commits = Vec::new();

    for (i, oid) in revwalk.enumerate() {
        if i >= limit {
            break;
        }

        let oid = oid?;
        let commit = repo.find_commit(oid)?;

        let hash = commit.id().to_string();
        let message = commit.message().unwrap_or("").to_string();
        let author = commit.author().email().unwrap_or("unknown").to_string();
        let timestamp = commit.time().seconds();

        // Get diff stats
        let (files_changed, lines_added, lines_removed) = if commit.parent_count() > 0 {
            let parent = commit.parent(0)?;
            let diff =
                repo.diff_tree_to_tree(Some(&parent.tree()?), Some(&commit.tree()?), None)?;
            let stats = diff.stats()?;
            (stats.files_changed(), stats.insertions(), stats.deletions())
        } else {
            // Initial commit - count all files as changed
            let tree = commit.tree()?;
            (tree.len(), 0, 0)
        };

        commits.push(CommitInfo {
            hash,
            message,
            author,
            timestamp,
            files_changed,
            lines_added,
            lines_removed,
        });
    }

    debug!("Analyzed {} commits", commits.len());
    Ok(commits)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_git_analyzer_can_be_created() {
        let temp_dir = TempDir::new().unwrap();
        let _analyzer = GitAnalyzer::new(temp_dir.path());
    }

    #[test]
    fn test_commit_info_structure() {
        let commit = CommitInfo {
            hash: "abc123".to_string(),
            message: "fix: null pointer dereference".to_string(),
            author: "test@example.com".to_string(),
            timestamp: 1234567890,
            files_changed: 3,
            lines_added: 15,
            lines_removed: 8,
        };

        assert_eq!(commit.hash, "abc123");
        assert_eq!(commit.message, "fix: null pointer dereference");
        assert_eq!(commit.author, "test@example.com");
        assert_eq!(commit.timestamp, 1234567890);
        assert_eq!(commit.files_changed, 3);
        assert_eq!(commit.lines_added, 15);
        assert_eq!(commit.lines_removed, 8);
    }

    #[test]
    fn test_analyze_nonexistent_repo() {
        let temp_dir = TempDir::new().unwrap();
        let analyzer = GitAnalyzer::new(temp_dir.path());

        let result = analyzer.analyze_commits("nonexistent-repo", 10);

        assert!(result.is_err());
    }

    #[test]
    fn test_commit_info_serialization() {
        let commit = CommitInfo {
            hash: "test123".to_string(),
            message: "test commit".to_string(),
            author: "test@example.com".to_string(),
            timestamp: 1234567890,
            files_changed: 1,
            lines_added: 10,
            lines_removed: 5,
        };

        let json = serde_json::to_string(&commit).unwrap();
        let deserialized: CommitInfo = serde_json::from_str(&json).unwrap();

        assert_eq!(commit.hash, deserialized.hash);
        assert_eq!(commit.message, deserialized.message);
        assert_eq!(commit.author, deserialized.author);
    }

    #[test]
    fn test_analyze_local_repo_with_commits() {
        let temp_dir = TempDir::new().unwrap();
        let repo_path = temp_dir.path().join("test-repo");
        std::fs::create_dir(&repo_path).unwrap();

        // Initialize a git repository
        let repo = Repository::init(&repo_path).unwrap();

        // Create a test file
        let test_file = repo_path.join("test.txt");
        std::fs::write(&test_file, "Hello, world!").unwrap();

        // Configure git
        let mut config = repo.config().unwrap();
        config.set_str("user.name", "Test User").unwrap();
        config.set_str("user.email", "test@example.com").unwrap();

        // Add and commit the file
        let mut index = repo.index().unwrap();
        index.add_path(Path::new("test.txt")).unwrap();
        index.write().unwrap();

        let tree_id = index.write_tree().unwrap();
        let tree = repo.find_tree(tree_id).unwrap();
        let sig = repo.signature().unwrap();

        repo.commit(Some("HEAD"), &sig, &sig, "Initial commit", &tree, &[])
            .unwrap();

        // Now analyze the repo
        let analyzer = GitAnalyzer::new(temp_dir.path());
        let commits = analyzer.analyze_commits("test-repo", 10).unwrap();

        assert_eq!(commits.len(), 1);
        assert_eq!(commits[0].message, "Initial commit");
        assert!(commits[0].files_changed > 0);
    }

    #[test]
    fn test_analyze_local_repo_multiple_commits() {
        let temp_dir = TempDir::new().unwrap();
        let repo_path = temp_dir.path().join("multi-commit-repo");
        std::fs::create_dir(&repo_path).unwrap();

        let repo = Repository::init(&repo_path).unwrap();

        let mut config = repo.config().unwrap();
        config.set_str("user.name", "Test User").unwrap();
        config.set_str("user.email", "test@example.com").unwrap();

        // Helper to commit a file
        let commit_file = |name: &str, content: &str, message: &str| {
            let file_path = repo_path.join(name);
            std::fs::write(&file_path, content).unwrap();

            let mut index = repo.index().unwrap();
            index.add_path(Path::new(name)).unwrap();
            index.write().unwrap();

            let tree_id = index.write_tree().unwrap();
            let tree = repo.find_tree(tree_id).unwrap();
            let sig = repo.signature().unwrap();

            let parent = if let Ok(head) = repo.head() {
                let parent_commit = head.peel_to_commit().unwrap();
                vec![parent_commit]
            } else {
                vec![]
            };

            let parent_refs: Vec<_> = parent.iter().collect();

            repo.commit(Some("HEAD"), &sig, &sig, message, &tree, &parent_refs)
                .unwrap();
        };

        commit_file("file1.txt", "content 1", "Add file1");
        commit_file("file2.txt", "content 2", "Add file2");
        commit_file("file3.txt", "content 3", "Add file3");

        // Analyze the repo
        let analyzer = GitAnalyzer::new(temp_dir.path());
        let commits = analyzer.analyze_commits("multi-commit-repo", 10).unwrap();

        assert_eq!(commits.len(), 3);
        assert_eq!(commits[0].message, "Add file3"); // Most recent first
        assert_eq!(commits[1].message, "Add file2");
        assert_eq!(commits[2].message, "Add file1");
    }

    #[test]
    fn test_analyze_respects_commit_limit() {
        let temp_dir = TempDir::new().unwrap();
        let repo_path = temp_dir.path().join("limit-repo");
        std::fs::create_dir(&repo_path).unwrap();

        let repo = Repository::init(&repo_path).unwrap();

        let mut config = repo.config().unwrap();
        config.set_str("user.name", "Test").unwrap();
        config.set_str("user.email", "test@test.com").unwrap();

        // Create 5 commits
        for i in 0..5 {
            let file_path = repo_path.join(format!("file{}.txt", i));
            std::fs::write(&file_path, format!("content {}", i)).unwrap();

            let mut index = repo.index().unwrap();
            index
                .add_path(Path::new(&format!("file{}.txt", i)))
                .unwrap();
            index.write().unwrap();

            let tree_id = index.write_tree().unwrap();
            let tree = repo.find_tree(tree_id).unwrap();
            let sig = repo.signature().unwrap();

            let parent = if let Ok(head) = repo.head() {
                vec![head.peel_to_commit().unwrap()]
            } else {
                vec![]
            };
            let parent_refs: Vec<_> = parent.iter().collect();

            repo.commit(
                Some("HEAD"),
                &sig,
                &sig,
                &format!("Commit {}", i),
                &tree,
                &parent_refs,
            )
            .unwrap();
        }

        let analyzer = GitAnalyzer::new(temp_dir.path());

        // Test with limit of 2
        let commits = analyzer.analyze_commits("limit-repo", 2).unwrap();
        assert_eq!(commits.len(), 2);

        // Test with limit of 10 (more than available)
        let commits = analyzer.analyze_commits("limit-repo", 10).unwrap();
        assert_eq!(commits.len(), 5);
    }

    // Integration tests that require network access are marked as ignored
    // Run with: cargo test -- --ignored
    #[test]
    #[ignore]
    fn test_clone_small_repository() {
        let temp_dir = TempDir::new().unwrap();
        let analyzer = GitAnalyzer::new(temp_dir.path());

        // Use a very small test repository
        let result =
            analyzer.clone_repository("https://github.com/rust-lang/rustlings", "rustlings");

        assert!(result.is_ok());
    }

    #[test]
    #[ignore]
    fn test_analyze_commits_basic() {
        let temp_dir = TempDir::new().unwrap();
        let analyzer = GitAnalyzer::new(temp_dir.path());

        analyzer
            .clone_repository("https://github.com/rust-lang/rustlings", "rustlings")
            .unwrap();

        let commits = analyzer.analyze_commits("rustlings", 10).unwrap();

        assert!(!commits.is_empty());
        assert!(commits.len() <= 10);

        let first_commit = &commits[0];
        assert!(!first_commit.hash.is_empty());
        assert!(!first_commit.message.is_empty());
    }

    #[test]
    #[ignore]
    fn test_analyze_commits_respects_limit() {
        let temp_dir = TempDir::new().unwrap();
        let analyzer = GitAnalyzer::new(temp_dir.path());

        analyzer
            .clone_repository("https://github.com/rust-lang/rustlings", "rustlings")
            .unwrap();

        let commits_5 = analyzer.analyze_commits("rustlings", 5).unwrap();
        assert!(commits_5.len() <= 5);

        let commits_20 = analyzer.analyze_commits("rustlings", 20).unwrap();
        assert!(commits_20.len() <= 20);
    }

    #[test]
    #[ignore]
    fn test_analyzer_caches_cloned_repos() {
        let temp_dir = TempDir::new().unwrap();
        let analyzer = GitAnalyzer::new(temp_dir.path());

        // First clone
        analyzer
            .clone_repository("https://github.com/rust-lang/rustlings", "rustlings")
            .unwrap();

        // Second call should not re-clone
        let result =
            analyzer.clone_repository("https://github.com/rust-lang/rustlings", "rustlings");
        assert!(result.is_ok());

        // Verify we can still analyze
        let commits = analyzer.analyze_commits("rustlings", 5).unwrap();
        assert!(!commits.is_empty());
    }

    #[test]
    fn test_clone_repository_already_exists() {
        let temp_dir = TempDir::new().unwrap();
        let repo_path = temp_dir.path().join("existing-repo");
        std::fs::create_dir(&repo_path).unwrap();

        // Initialize a repo to simulate already exists
        Repository::init(&repo_path).unwrap();

        let analyzer = GitAnalyzer::new(temp_dir.path());

        // Should not fail when repo already exists
        let result = analyzer.clone_repository("https://example.com/repo.git", "existing-repo");
        assert!(result.is_ok());
    }

    #[test]
    fn test_clone_repository_invalid_url() {
        let temp_dir = TempDir::new().unwrap();
        let analyzer = GitAnalyzer::new(temp_dir.path());

        // Try to clone from invalid URL
        let result = analyzer.clone_repository("not-a-valid-url", "test-repo");
        assert!(result.is_err());
    }

    #[test]
    fn test_analyze_commits_with_zero_limit() {
        let temp_dir = TempDir::new().unwrap();
        let repo_path = temp_dir.path().join("zero-limit-repo");
        std::fs::create_dir(&repo_path).unwrap();

        let repo = Repository::init(&repo_path).unwrap();

        let mut config = repo.config().unwrap();
        config.set_str("user.name", "Test").unwrap();
        config.set_str("user.email", "test@test.com").unwrap();

        // Create one commit
        let file_path = repo_path.join("file.txt");
        std::fs::write(&file_path, "content").unwrap();

        let mut index = repo.index().unwrap();
        index.add_path(Path::new("file.txt")).unwrap();
        index.write().unwrap();

        let tree_id = index.write_tree().unwrap();
        let tree = repo.find_tree(tree_id).unwrap();
        let sig = repo.signature().unwrap();

        repo.commit(Some("HEAD"), &sig, &sig, "Test commit", &tree, &[])
            .unwrap();

        let analyzer = GitAnalyzer::new(temp_dir.path());

        // Analyze with limit=0 should return empty
        let commits = analyzer.analyze_commits("zero-limit-repo", 0).unwrap();
        assert_eq!(commits.len(), 0);
    }

    #[test]
    fn test_analyze_commits_with_modifications() {
        let temp_dir = TempDir::new().unwrap();
        let repo_path = temp_dir.path().join("modify-repo");
        std::fs::create_dir(&repo_path).unwrap();

        let repo = Repository::init(&repo_path).unwrap();

        let mut config = repo.config().unwrap();
        config.set_str("user.name", "Test User").unwrap();
        config.set_str("user.email", "test@example.com").unwrap();

        // Initial commit
        let file_path = repo_path.join("test.txt");
        std::fs::write(&file_path, "Line 1\nLine 2\nLine 3\n").unwrap();

        let mut index = repo.index().unwrap();
        index.add_path(Path::new("test.txt")).unwrap();
        index.write().unwrap();

        let tree_id = index.write_tree().unwrap();
        let tree = repo.find_tree(tree_id).unwrap();
        let sig = repo.signature().unwrap();

        repo.commit(Some("HEAD"), &sig, &sig, "Initial commit", &tree, &[])
            .unwrap();

        // Second commit with modifications
        std::fs::write(&file_path, "Line 1\nLine 2 modified\nLine 3\nLine 4\n").unwrap();

        let mut index = repo.index().unwrap();
        index.add_path(Path::new("test.txt")).unwrap();
        index.write().unwrap();

        let tree_id = index.write_tree().unwrap();
        let tree = repo.find_tree(tree_id).unwrap();

        let parent = repo.head().unwrap().peel_to_commit().unwrap();

        repo.commit(Some("HEAD"), &sig, &sig, "Modify file", &tree, &[&parent])
            .unwrap();

        let analyzer = GitAnalyzer::new(temp_dir.path());
        let commits = analyzer.analyze_commits("modify-repo", 10).unwrap();

        assert_eq!(commits.len(), 2);
        // Second commit should have lines added/removed
        assert!(commits[0].lines_added > 0 || commits[0].lines_removed > 0);
    }

    #[test]
    fn test_commit_info_clone() {
        let original = CommitInfo {
            hash: "abc123".to_string(),
            message: "test".to_string(),
            author: "test@example.com".to_string(),
            timestamp: 1234567890,
            files_changed: 1,
            lines_added: 10,
            lines_removed: 5,
        };

        let cloned = original.clone();

        assert_eq!(original.hash, cloned.hash);
        assert_eq!(original.message, cloned.message);
        assert_eq!(original.author, cloned.author);
        assert_eq!(original.timestamp, cloned.timestamp);
        assert_eq!(original.files_changed, cloned.files_changed);
        assert_eq!(original.lines_added, cloned.lines_added);
        assert_eq!(original.lines_removed, cloned.lines_removed);
    }

    #[test]
    fn test_commit_info_debug_format() {
        let commit = CommitInfo {
            hash: "abc123".to_string(),
            message: "test".to_string(),
            author: "test@example.com".to_string(),
            timestamp: 1234567890,
            files_changed: 1,
            lines_added: 10,
            lines_removed: 5,
        };

        let debug_str = format!("{:?}", commit);
        assert!(debug_str.contains("abc123"));
        assert!(debug_str.contains("test"));
    }

    #[test]
    fn test_analyzer_with_path_ref() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path();

        // Test that it accepts AsRef<Path>
        let _analyzer = GitAnalyzer::new(path);
        let _analyzer2 = GitAnalyzer::new(path.to_str().unwrap());
    }

    #[test]
    fn test_analyze_empty_repository() {
        let temp_dir = TempDir::new().unwrap();
        let repo_path = temp_dir.path().join("empty-repo");
        std::fs::create_dir(&repo_path).unwrap();

        // Initialize empty repository (no commits)
        Repository::init(&repo_path).unwrap();

        let analyzer = GitAnalyzer::new(temp_dir.path());

        // Analyzing empty repo should return error (no HEAD)
        let result = analyzer.analyze_commits("empty-repo", 10);
        assert!(result.is_err());
    }

    #[test]
    fn test_analyze_commits_with_large_diff() {
        let temp_dir = TempDir::new().unwrap();
        let repo_path = temp_dir.path().join("large-diff-repo");
        std::fs::create_dir(&repo_path).unwrap();

        let repo = Repository::init(&repo_path).unwrap();

        let mut config = repo.config().unwrap();
        config.set_str("user.name", "Test").unwrap();
        config.set_str("user.email", "test@test.com").unwrap();

        // Create a file with many lines
        let mut content = String::new();
        for i in 0..1000 {
            content.push_str(&format!("Line {}\n", i));
        }

        let file_path = repo_path.join("large.txt");
        std::fs::write(&file_path, &content).unwrap();

        let mut index = repo.index().unwrap();
        index.add_path(Path::new("large.txt")).unwrap();
        index.write().unwrap();

        let tree_id = index.write_tree().unwrap();
        let tree = repo.find_tree(tree_id).unwrap();
        let sig = repo.signature().unwrap();

        repo.commit(Some("HEAD"), &sig, &sig, "Large commit", &tree, &[])
            .unwrap();

        let analyzer = GitAnalyzer::new(temp_dir.path());
        let commits = analyzer.analyze_commits("large-diff-repo", 10).unwrap();

        assert_eq!(commits.len(), 1);
        assert_eq!(commits[0].files_changed, 1);
    }

    #[test]
    fn test_commit_info_with_empty_strings() {
        let commit = CommitInfo {
            hash: "".to_string(),
            message: "".to_string(),
            author: "".to_string(),
            timestamp: 0,
            files_changed: 0,
            lines_added: 0,
            lines_removed: 0,
        };

        assert_eq!(commit.hash, "");
        assert_eq!(commit.message, "");
        assert_eq!(commit.author, "");
    }

    #[test]
    fn test_multiple_files_in_single_commit() {
        let temp_dir = TempDir::new().unwrap();
        let repo_path = temp_dir.path().join("multi-file-repo");
        std::fs::create_dir(&repo_path).unwrap();

        let repo = Repository::init(&repo_path).unwrap();

        let mut config = repo.config().unwrap();
        config.set_str("user.name", "Test").unwrap();
        config.set_str("user.email", "test@test.com").unwrap();

        // Create multiple files in one commit
        for i in 0..5 {
            let file_path = repo_path.join(format!("file{}.txt", i));
            std::fs::write(&file_path, format!("content {}", i)).unwrap();
        }

        let mut index = repo.index().unwrap();
        for i in 0..5 {
            index
                .add_path(Path::new(&format!("file{}.txt", i)))
                .unwrap();
        }
        index.write().unwrap();

        let tree_id = index.write_tree().unwrap();
        let tree = repo.find_tree(tree_id).unwrap();
        let sig = repo.signature().unwrap();

        repo.commit(Some("HEAD"), &sig, &sig, "Add 5 files", &tree, &[])
            .unwrap();

        let analyzer = GitAnalyzer::new(temp_dir.path());
        let commits = analyzer.analyze_commits("multi-file-repo", 10).unwrap();

        assert_eq!(commits.len(), 1);
        assert_eq!(commits[0].files_changed, 5);
    }

    #[test]
    fn test_commit_with_deletions() {
        let temp_dir = TempDir::new().unwrap();
        let repo_path = temp_dir.path().join("deletion-repo");
        std::fs::create_dir(&repo_path).unwrap();

        let repo = Repository::init(&repo_path).unwrap();

        let mut config = repo.config().unwrap();
        config.set_str("user.name", "Test").unwrap();
        config.set_str("user.email", "test@test.com").unwrap();

        // First commit with content
        let file_path = repo_path.join("file.txt");
        std::fs::write(&file_path, "Line 1\nLine 2\nLine 3\nLine 4\nLine 5\n").unwrap();

        let mut index = repo.index().unwrap();
        index.add_path(Path::new("file.txt")).unwrap();
        index.write().unwrap();

        let tree_id = index.write_tree().unwrap();
        let tree = repo.find_tree(tree_id).unwrap();
        let sig = repo.signature().unwrap();

        repo.commit(Some("HEAD"), &sig, &sig, "Initial", &tree, &[])
            .unwrap();

        // Second commit with deletions
        std::fs::write(&file_path, "Line 1\nLine 5\n").unwrap();

        let mut index = repo.index().unwrap();
        index.add_path(Path::new("file.txt")).unwrap();
        index.write().unwrap();

        let tree_id = index.write_tree().unwrap();
        let tree = repo.find_tree(tree_id).unwrap();

        let parent = repo.head().unwrap().peel_to_commit().unwrap();

        repo.commit(Some("HEAD"), &sig, &sig, "Delete lines", &tree, &[&parent])
            .unwrap();

        let analyzer = GitAnalyzer::new(temp_dir.path());
        let commits = analyzer.analyze_commits("deletion-repo", 10).unwrap();

        assert_eq!(commits.len(), 2);
        // First commit (most recent) should have deletions
        assert!(commits[0].lines_removed > 0);
    }

    #[test]
    fn test_commit_info_deserialization() {
        let json = r#"{
            "hash": "abc123",
            "message": "test message",
            "author": "test@example.com",
            "timestamp": 1234567890,
            "files_changed": 3,
            "lines_added": 15,
            "lines_removed": 8
        }"#;

        let commit: CommitInfo = serde_json::from_str(json).unwrap();

        assert_eq!(commit.hash, "abc123");
        assert_eq!(commit.message, "test message");
        assert_eq!(commit.author, "test@example.com");
        assert_eq!(commit.timestamp, 1234567890);
        assert_eq!(commit.files_changed, 3);
        assert_eq!(commit.lines_added, 15);
        assert_eq!(commit.lines_removed, 8);
    }
}
