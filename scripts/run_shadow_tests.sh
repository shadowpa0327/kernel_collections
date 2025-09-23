#!/bin/bash

# Script to run shadow attention tests
# Usage: ./run_shadow_tests.sh [quick|full]

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "Shadow Attention Test Runner"
echo "============================"
echo "Project root: $PROJECT_ROOT"
echo "Current directory: $(pwd)"
echo ""

# Check if xKV/shadow_attn.py exists
if [[ ! -f "xKV/shadow_attn.py" ]]; then
    echo "Error: xKV/shadow_attn.py not found!"
    echo "Make sure you're running this from the project root."
    exit 1
fi

# Determine which test to run
TEST_TYPE="${1:-quick}"

case "$TEST_TYPE" in
    "quick")
        echo "Running quick test (fewer parameters, faster execution)..."
        echo "Testing only 'full' mode with 3 batch sizes and 3 prefill lengths"
        python scripts/quick_shadow_test.py
        ;;
    "full")
        echo "Running full comparison sweep (all configurations, comprehensive)..."
        echo "Testing all three configurations:"
        echo "  - xkv:  rank_k=384, rank_v=576"
        echo "  - xkey: rank_k=256"
        echo "  - full: baseline full attention"
        echo "With 5 batch sizes and 6 prefill lengths each"
        python scripts/test_shadow_attn_sweep.py
        ;;
    "single")
        # Run a single test with the original command format
        echo "Running single test with your original parameters..."
        python xKV/shadow_attn.py --mode xkv --model_name local --rank_k 384 --rank_v 576 --batch_size 8 --prefill_len 65536
        ;;
    *)
        echo "Usage: $0 [quick|full|single]"
        echo ""
        echo "  quick  - Run quick test with 'full' mode only (3x3 = 9 runs)"
        echo "  full   - Run comprehensive comparison of all three modes (3x5x6 = 90 runs)" 
        echo "  single - Run your original single test command"
        echo ""
        echo "Configurations tested in full mode:"
        echo "  - xkv:  rank_k=384, rank_v=576"
        echo "  - xkey: rank_k=256"
        echo "  - full: baseline full attention"
        echo ""
        echo "All tests dump complete JSON results to console and files."
        exit 1
        ;;
esac

echo ""
echo "Test completed!" 