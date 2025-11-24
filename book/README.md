# Organizational Intelligence Plugin - Book

This directory contains the mdBook documentation for the Organizational Intelligence Plugin.

## Building the Book

### Prerequisites

```bash
cargo install mdbook
```

### Build

```bash
# Build the book
mdbook build

# Serve locally (auto-reload on changes)
mdbook serve --open

# Clean build artifacts
mdbook clean
```

## Structure

- `book.toml` - Book configuration
- `src/` - Markdown source files
  - `SUMMARY.md` - Table of contents
  - `introduction.md` - Book introduction
  - `getting-started/` - Installation and quick start guides
  - `core-concepts/` - Fundamental concepts
  - `defect-categories/` - All 18 defect category details
  - `cli/` - Command-line usage documentation
  - `ml-pipeline/` - Machine learning pipeline
  - `architecture/` - Three-tier architecture details
  - `gpu/` - GPU acceleration documentation
  - `nlp/` - NLP enhancement details
  - `examples/` - Real-world examples
  - `validation/` - Validation results and analysis
  - `quality-gates/` - Quality enforcement
  - `extreme-tdd/` - EXTREME TDD methodology
  - `toyota-way/` - Toyota Way principles
  - `sprints/` - Sprint development process
  - `advanced/` - Advanced topics
  - `best-practices/` - Best practices guide
  - `troubleshooting/` - Common issues and solutions
  - `appendix/` - Glossary, references, changelog

## Output

Built book is in `build/` directory (gitignored).

## Contributing

When adding new chapters:

1. Create the `.md` file in the appropriate directory
2. Add an entry in `src/SUMMARY.md`
3. Build and verify: `mdbook build`
4. Test locally: `mdbook serve`

## Style Guide

Follow the same style as the [aprender book](https://github.com/paiml/aprender/tree/main/book):

- Clear, concise language
- Code examples with explanations
- Real-world use cases
- Cross-references to related chapters
- "Next Steps" section at the end of chapters

## License

MIT License - see LICENSE file in repository root.
