# Contributing to Organizational Intelligence Plugin

Thank you for considering contributing to OIP! This document provides guidelines and instructions for contributing.

## Code of Conduct

Be respectful, collaborative, and constructive. We follow the Toyota Way principle of "Respect for People."

## Getting Started

### Prerequisites

- Rust 1.70+
- Git
- pmat (for TDG integration)
- GitHub account

### Development Setup

```bash
# Clone the repository
git clone https://github.com/paiml/organizational-intelligence-plugin
cd organizational-intelligence-plugin

# Build and test
cargo build
make test-fast
make lint
```

## Development Workflow

We follow **EXTREME TDD** (Test-Driven Development):

1. **RED**: Write a failing test
2. **GREEN**: Write minimal code to make it pass
3. **REFACTOR**: Clean up the code while keeping tests green

### Before Starting Work

1. Check existing issues or create a new one
2. Comment on the issue to claim it
3. Create a feature branch from `main`

```bash
git checkout -b feature/your-feature-name
```

## Making Changes

### Code Style

- Follow Rust API guidelines
- Run `cargo fmt` before committing
- Ensure `cargo clippy` passes with zero warnings
- Document public APIs with doc comments and examples

### Testing Requirements

All new code must have:

- ✅ Unit tests (minimum 85% coverage)
- ✅ Integration tests for user-facing features
- ✅ Property-based tests for complex logic (when applicable)

```bash
# Run tests
make test-fast        # Quick unit tests
make test-all         # All tests
make coverage         # Coverage report (target: 85%+)
```

### Quality Gates

Before submitting a PR, ensure:

```bash
make lint             # ✅ No clippy warnings
make test-fast        # ✅ All tests pass
make coverage         # ✅ 85%+ coverage
```

## Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
type(scope): short description

Longer description if needed.

Fixes #123
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding tests
- `refactor`: Code refactoring
- `chore`: Maintenance tasks

**Examples**:
```
feat(summarizer): add PII stripping for commit hashes
fix(analyzer): handle empty repository gracefully
docs(readme): add installation instructions
test(cli): add integration tests for summarize command
```

## Pull Request Process

1. **Update Documentation**: Ensure README and docs reflect your changes
2. **Add Tests**: All new features must have tests
3. **Pass Quality Gates**: `make lint && make test-fast` must succeed
4. **Update CHANGELOG**: Add entry under "Unreleased"
5. **Submit PR**: Use the PR template, reference related issues

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] All tests pass locally

## Quality Gates
- [ ] make lint passes
- [ ] make test-fast passes
- [ ] Coverage ≥ 85% for new code

## Related Issues
Fixes #<issue_number>
```

## Code Review

All PRs require review before merging. Reviewers will check:

- Code quality and style
- Test coverage
- Documentation completeness
- Toyota Way principles adherence

## Areas for Contribution

### Good First Issues

Look for issues labeled `good-first-issue`:
- Documentation improvements
- Test coverage improvements
- Bug fixes

### Priority Areas

- Increase test coverage (current: 58.79%, target: 85%+)
- Add more defect classification patterns
- Improve CI/CD integration examples
- Phase 4: AI prompt integration

### Toyota Way Contributions

We especially value contributions that:

- **Eliminate Waste (Muda)**: Automate manual processes
- **Prevent Overburden (Muri)**: Optimize performance, reduce complexity
- **Build Quality In (Jidoka)**: Add validation, improve error handling
- **Continuous Improvement (Kaizen)**: Incremental improvements

## Getting Help

- 📖 Read the [README](README.md)
- 💬 Ask in [GitHub Discussions](https://github.com/paiml/organizational-intelligence-plugin/discussions)
- 🐛 Report bugs via [GitHub Issues](https://github.com/paiml/organizational-intelligence-plugin/issues)

## Recognition

Contributors are recognized in:
- Git commit history (Co-Authored-By)
- Release notes
- Project acknowledgments

Thank you for contributing! 🙏
