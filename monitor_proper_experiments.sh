#!/bin/bash
# Monitor properly configured experiments

echo "================================================================================"
echo "EXPERIMENT MONITOR (Proper Config: 50 epochs, ~300 train mazes)"
echo "================================================================================"
echo ""

# Check running processes
echo "🔄 Running Python processes:"
ps aux | grep "run_12x12_8m_benchmark.py" | grep -v grep | wc -l | xargs echo "  Active experiments:"

echo ""
echo "--------------------------------------------------------------------------------"
echo "1️⃣  O(R) Memory (Standard - all iterations)"
echo "--------------------------------------------------------------------------------"
if [ -f experiments/results/comparison_OR_memory_proper.log ]; then
    tail -5 experiments/results/comparison_OR_memory_proper.log
    echo ""
    if [ -f experiments/results/comparison_OR_memory_proper/results.json ]; then
        echo "✅ COMPLETED!"
        cat experiments/results/comparison_OR_memory_proper/results.json
    fi
else
    echo "⏳ Not started yet"
fi

echo ""
echo "--------------------------------------------------------------------------------"
echo "2️⃣  O(1) Memory (Last iteration only)"
echo "--------------------------------------------------------------------------------"
if [ -f experiments/results/comparison_O1_memory_proper.log ]; then
    tail -5 experiments/results/comparison_O1_memory_proper.log
    echo ""
    if [ -f experiments/results/comparison_O1_memory_proper/results.json ]; then
        echo "✅ COMPLETED!"
        cat experiments/results/comparison_O1_memory_proper/results.json
    fi
else
    echo "⏳ Not started yet"
fi

echo ""
echo "--------------------------------------------------------------------------------"
echo "3️⃣  Sparse Supervision (every 3rd step)"
echo "--------------------------------------------------------------------------------"
if [ -f experiments/results/comparison_sparse_supervision_proper.log ]; then
    tail -5 experiments/results/comparison_sparse_supervision_proper.log
    echo ""
    if [ -f experiments/results/comparison_sparse_supervision_proper/results.json ]; then
        echo "✅ COMPLETED!"
        cat experiments/results/comparison_sparse_supervision_proper/results.json
    fi
else
    echo "⏳ Not started yet"
fi

echo ""
echo "================================================================================"

