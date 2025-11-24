# Installation

This guide covers all installation methods for OIP.

## System Requirements

- **Rust**: 1.75.0 or later
- **Git**: Any recent version
- **OS**: Linux, macOS, or Windows
- **Memory**: 4GB minimum (8GB recommended for ML training)
- **Disk**: 500MB for build artifacts

## Method 1: Build from Source (Recommended)

### 1. Install Rust

If you don't have Rust installed:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

Verify installation:

```bash
rustc --version  # Should show 1.75.0 or later
cargo --version
```

### 2. Clone and Build

```bash
# Clone the repository
git clone https://github.com/paiml/organizational-intelligence-plugin.git
cd organizational-intelligence-plugin

# Build release binary
cargo build --release

# Binary location
ls -lh target/release/oip
```

### 3. Optional: Install Globally

```bash
cargo install --path .

# Now you can run from anywhere
oip --version
```

## Method 2: Cargo Install (When Published)

Once published to crates.io:

```bash
cargo install organizational-intelligence-plugin

# Verify
oip --version
```

## GPU Support (Optional)

OIP includes GPU acceleration via WebGPU. It works out-of-the-box on most systems.

### Verify GPU Support

```bash
# Build GPU-enabled binary
cargo build --release --bin oip-gpu

# Check GPU availability
./target/release/oip-gpu --help
```

### Troubleshooting GPU

If GPU acceleration doesn't work:

1. **Update graphics drivers**
2. **Install Vulkan** (Linux):
   ```bash
   # Ubuntu/Debian
   sudo apt install mesa-vulkan-drivers vulkan-utils

   # Verify
   vulkaninfo | grep deviceName
   ```

3. **Fall back to CPU**: OIP works perfectly without GPU, just slower for correlation analysis.

## GitHub Token Setup

For analyzing GitHub organizations (not required for local repos):

### 1. Create Personal Access Token

1. Visit https://github.com/settings/tokens
2. Click "Generate new token (classic)"
3. Select scopes: `repo` (full control)
4. Click "Generate token"
5. **Copy the token** (you won't see it again!)

### 2. Set Environment Variable

```bash
# Temporary (current session)
export GITHUB_TOKEN=ghp_your_token_here

# Permanent (add to ~/.bashrc or ~/.zshrc)
echo 'export GITHUB_TOKEN=ghp_your_token_here' >> ~/.bashrc
source ~/.bashrc
```

### 3. Verify Token

```bash
# Test with a small organization
./target/release/oip analyze \
  --org rust-lang \
  --output test.yaml \
  --max-commits 10
```

## Development Setup (For Contributors)

If you plan to contribute to OIP:

```bash
# Clone with all submodules
git clone --recursive https://github.com/paiml/organizational-intelligence-plugin.git
cd organizational-intelligence-plugin

# Install development tools
rustup component add rustfmt clippy
cargo install cargo-mutants criterion

# Run tests
cargo test

# Run quality checks
cargo fmt --check
cargo clippy -- -D warnings

# Run benchmarks
cargo bench
```

## mdbook (For This Documentation)

To build this book locally:

```bash
# Install mdbook
cargo install mdbook

# Serve locally
cd book
mdbook serve --open

# Build static site
mdbook build
```

## Verification

After installation, verify everything works:

```bash
# Check version
./target/release/oip --version
# Output: organizational-intelligence-plugin v0.1.0

# List available commands
./target/release/oip --help

# Run test analysis on example repository
./target/release/oip analyze \
  --org dummy \
  --local ../some-repo \
  --output test.yaml
```

## Troubleshooting

### Error: "linker 'cc' not found"

**Solution** (Linux):
```bash
sudo apt install build-essential
```

**Solution** (macOS):
```bash
xcode-select --install
```

### Error: "aprender" dependency not found

**Solution**: Clone with aprender as sibling directory:
```bash
cd ..
git clone https://github.com/paiml/aprender.git
cd organizational-intelligence-plugin
cargo build --release
```

Or update `Cargo.toml` to use crates.io version (when published).

### Slow Compilation

**Solution**: Use faster linker:
```bash
# Install mold (Linux)
sudo apt install mold

# Or lld
sudo apt install lld

# Configure in ~/.cargo/config.toml
[target.x86_64-unknown-linux-gnu]
linker = "clang"
rustflags = ["-C", "link-arg=-fuse-ld=mold"]
```

## Next Steps

- **[Configuration](./configuration.md)** - Customize OIP settings
- **[First Analysis](./first-analysis.md)** - Run a comprehensive analysis
- **[CLI Usage](../cli/analyze.md)** - Master all commands

---

**✅ Installation complete! Ready to analyze defect patterns.**
