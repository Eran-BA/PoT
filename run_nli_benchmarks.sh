#!/bin/bash
# Run NLI Benchmarks with proper virtual environment

cd /Users/rnbnrzy/Desktop/PoT

# Activate virtual environment
source venv/bin/activate

# Install any missing dependencies
echo "Installing missing dependencies..."
pip install pyyaml datasets --quiet

echo ""
echo "========================================"
echo "Which benchmark would you like to run?"
echo "========================================"
echo "1. Quick synthetic test (3 min)"
echo "2. Full synthetic benchmark (30-60 min)"
echo "3. Real NLI quick test (30 min, 5K samples)"
echo "4. Real NLI full benchmark (2-3 hours)"
echo "5. Run both synthetic and real (quick)"
echo ""
read -p "Enter choice (1-5): " choice

case $choice in
    1)
        echo "Running quick synthetic test..."
        PYTHONPATH=$PWD python experiments/quick_nli_test.py
        ;;
    2)
        echo "Running full synthetic benchmark..."
        PYTHONPATH=$PWD python experiments/fair_ab_nli.py
        ;;
    3)
        echo "Running real NLI quick test..."
        PYTHONPATH=$PWD python experiments/real_nli_benchmark.py \
            --dataset snli --max_train_samples 5000 --max_steps 2000
        ;;
    4)
        echo "Running real NLI full benchmark..."
        PYTHONPATH=$PWD python experiments/real_nli_benchmark.py \
            --dataset snli --max_steps 20000
        ;;
    5)
        echo "Running both quick benchmarks..."
        echo ""
        echo "=== Quick Synthetic Test ==="
        PYTHONPATH=$PWD python experiments/quick_nli_test.py
        echo ""
        echo "=== Real NLI Quick Test ==="
        PYTHONPATH=$PWD python experiments/real_nli_benchmark.py \
            --dataset snli --max_train_samples 5000 --max_steps 2000
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "✅ Benchmark complete!"
echo "Check results in experiments/results/"

