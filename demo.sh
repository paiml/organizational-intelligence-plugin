#!/bin/bash
# OIP-GPU Demo Script - Local Repository Analysis

set -e

echo "🚀 OIP-GPU Local Repository Analysis Demo"
echo "=========================================="
echo ""

# Build if needed
if [ ! -f "target/release/oip-gpu" ]; then
    echo "📦 Building oip-gpu..."
    cargo build --release --bin oip-gpu
    echo ""
fi

# Clean up previous demo files
rm -f demo-output.db

echo "📊 Step 1: Analyze Local Repository (depyler)"
echo "----------------------------------------------"
./target/release/oip-gpu analyze \
    --local ../depyler \
    --output demo-output.db \
    --max-commits 500
echo ""

echo "📋 Step 2: Query All Defects"
echo "-----------------------------"
./target/release/oip-gpu query \
    --input demo-output.db \
    "show all defects"
echo ""

echo "🔝 Step 3: Most Common Defect Categories"
echo "----------------------------------------"
./target/release/oip-gpu query \
    --input demo-output.db \
    "most common defect"
echo ""

echo "📊 Step 4: Count by Category (CSV)"
echo "-----------------------------------"
./target/release/oip-gpu query \
    --input demo-output.db \
    --format csv \
    "count by category"
echo ""

echo "💾 Step 5: Export to JSON"
echo "-------------------------"
./target/release/oip-gpu query \
    --input demo-output.db \
    --format json \
    --export demo-results.json \
    "count by category"
echo ""

echo "📈 Database Statistics"
echo "---------------------"
ls -lh demo-output.db
echo ""
echo "Feature count: $(jq '. | length' demo-output.db)"
echo ""

echo "✨ Demo Complete!"
echo ""
echo "Generated files:"
echo "  - demo-output.db (feature database)"
echo "  - demo-results.json (query results)"
echo ""
echo "Next steps:"
echo "  ./target/release/oip-gpu query --input demo-output.db \"YOUR QUERY\""
