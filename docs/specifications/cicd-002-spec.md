---
title: Release Automation
issue: CICD-002
status: Complete
created: 2025-11-24
updated: 2025-11-24
---

# CICD-002: Release Automation

**Ticket ID**: CICD-002
**Status**: Complete

## Summary

Automated release workflow for building, packaging, and publishing binaries across all major platforms with automatic changelog generation and checksum verification.

## Requirements

### Functional Requirements
- [x] Multi-platform release builds (Linux, macOS, Windows)
- [x] Build both `oip` and `oip-gpu` binaries
- [x] Automatic changelog generation from git commits
- [x] SHA256 checksums for all release artifacts
- [x] GitHub Release creation with proper metadata
- [x] Prerelease detection for semver prereleases
- [x] Manual workflow dispatch support
- [x] crates.io publish preparation (dry-run)

### Non-Functional Requirements
- [x] Cross-platform compatible build process
- [x] Proper artifact naming convention
- [x] Caching for faster builds

## Workflow Design

### Triggers
- Push tags matching `v*` pattern
- Manual workflow dispatch with version input

### Jobs

#### 1. create-release
Creates GitHub release with:
- Auto-generated changelog from git log
- Installation instructions
- Prerelease detection

#### 2. build-release (matrix)
Builds for each platform:
- `x86_64-unknown-linux-gnu` (Linux)
- `x86_64-apple-darwin` (macOS)
- `x86_64-pc-windows-msvc` (Windows)

Artifacts:
- `oip-{platform}` - Standard CLI
- `oip-gpu-{platform}` - GPU-accelerated CLI
- `checksums-{platform}.txt` - SHA256 checksums

#### 3. publish-crate
Prepares crates.io publication (dry-run for non-prerelease)

## Release Artifacts

| Platform | Binary | GPU Binary |
|----------|--------|------------|
| Linux | oip-linux-amd64 | oip-gpu-linux-amd64 |
| macOS | oip-macos-amd64 | oip-gpu-macos-amd64 |
| Windows | oip-windows-amd64.exe | oip-gpu-windows-amd64.exe |

## Usage

### Tag Release
```bash
git tag v0.1.0
git push origin v0.1.0
```

### Manual Release
1. Go to Actions > Release
2. Click "Run workflow"
3. Enter version (e.g., v0.1.0)

## Success Criteria

- [x] All platforms build successfully
- [x] Both binaries included in release
- [x] Checksums generated and uploaded
- [x] Changelog auto-generated
- [x] Prerelease detection works
- [x] Manual dispatch works

## References

- `.github/workflows/release.yml`
