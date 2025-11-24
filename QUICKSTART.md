# OIP-GPU Quick Start - Local Repository Analysis

## Installation

```bash
cargo build --release --bin oip-gpu
```

## Basic Usage

### 1. Analyze a Local Repository

```bash
./target/release/oip-gpu analyze \
    --local ../your-repo \
    --output analysis.db \
    --max-commits 1000
```

**Options:**
- `--local <PATH>` - Path to local git repository
- `--output <FILE>` - Output database file (default: oip-gpu.db)
- `--max-commits <N>` - Maximum commits to analyze (default: 1000)

### 2. Query the Results

**Show all defects:**
```bash
./target/release/oip-gpu query \
    --input analysis.db \
    "show all defects"
```

**Most common defect categories:**
```bash
./target/release/oip-gpu query \
    --input analysis.db \
    "most common defect"
```

**Count by category:**
```bash
./target/release/oip-gpu query \
    --input analysis.db \
    "count by category"
```

### 3. Output Formats

**Table (default):**
```bash
./target/release/oip-gpu query --input analysis.db "most common defect"
```

**JSON:**
```bash
./target/release/oip-gpu query \
    --input analysis.db \
    --format json \
    "count by category"
```

**CSV:**
```bash
./target/release/oip-gpu query \
    --input analysis.db \
    --format csv \
    "count by category"
```

**Export to file:**
```bash
./target/release/oip-gpu query \
    --input analysis.db \
    --format json \
    --export results.json \
    "count by category"
```

## Demo Script

Run the full demo on depyler repository:

```bash
./demo.sh
```

## Defect Categories

| Category | Description |
|----------|-------------|
| 0 | General/Uncategorized |
| 1 | Bug Fix |
| 2 | Feature Addition |
| 3 | Hotfix |
| 4 | Breaking Change |
| 5 | Performance |
| 6 | Security |
| 7 | Testing |
| 8 | Refactoring |
| 9 | Documentation |

## Example: Analyze depyler

```bash
# Analyze 500 commits
./target/release/oip-gpu analyze \
    --local ../depyler \
    --output depyler.db \
    --max-commits 500

# Query results
./target/release/oip-gpu query --input depyler.db "show all defects"
```

**Expected Output:**
```
📊 Total features: 500

By category:
  Category 0: 346 (69.2%)  # General
  Category 9: 54 (10.8%)   # Documentation
  Category 5: 40 (8.0%)    # Performance
  Category 6: 26 (5.2%)    # Security
  Category 8: 25 (5.0%)    # Refactoring
```

## Supported Queries

- `"show all defects"` - List all defects with distribution
- `"most common defect"` - Show defects sorted by frequency
- `"count by category"` - Count defects per category

## Troubleshooting

**Repository not found:**
```
Error: Not a git repository: /path/to/repo
```
→ Ensure the path contains a `.git` directory

**No features loaded:**
```
⚠️  No features found in store
```
→ Run analyze command first to generate the database

## Next Steps

- See `docs/USER_GUIDE.md` for complete documentation
- Run `./target/release/oip-gpu --help` for all commands
- View `docs/specifications/` for technical details
