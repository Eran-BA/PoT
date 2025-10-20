#!/bin/bash
# Monitor running experiments

echo "================================================================================"
echo "EXPERIMENT MONITOR"
echo "================================================================================"
echo ""

# Check running processes
echo "🔄 Running Python processes:"
ps aux | grep "run_12x12_8m_benchmark.py" | grep -v grep | wc -l | xargs echo "  Active experiments:"

echo ""
echo "--------------------------------------------------------------------------------"
echo "1️⃣  O(R) Memory (Standard - all iterations)"
echo "--------------------------------------------------------------------------------"
if [ -f experiments/results/comparison_OR_memory.log ]; then
    tail -5 experiments/results/comparison_OR_memory.log
    echo ""
    if [ -f experiments/results/comparison_OR_memory/results.json ]; then
        echo "✅ COMPLETED!"
        cat experiments/results/comparison_OR_memory/results.json
    fi
else
    echo "⏳ Not started yet"
fi

echo ""
echo "--------------------------------------------------------------------------------"
echo "2️⃣  O(1) Memory (Last iteration only)"
echo "--------------------------------------------------------------------------------"
if [ -f experiments/results/comparison_O1_memory.log ]; then
    tail -5 experiments/results/comparison_O1_memory.log
    echo ""
    if [ -f experiments/results/comparison_O1_memory/results.json ]; then
        echo "✅ COMPLETED!"
        cat experiments/results/comparison_O1_memory/results.json
    fi
else
    echo "⏳ Not started yet"
fi

echo ""
echo "--------------------------------------------------------------------------------"
echo "3️⃣  Sparse Supervision (every 3rd step)"
echo "--------------------------------------------------------------------------------"
if [ -f experiments/results/comparison_sparse_supervision.log ]; then
    tail -5 experiments/results/comparison_sparse_supervision.log
    echo ""
    if [ -f experiments/results/comparison_sparse_supervision/results.json ]; then
        echo "✅ COMPLETED!"
        cat experiments/results/comparison_sparse_supervision/results.json
    fi
else
    echo "⏳ Not started yet"
fi

echo ""
echo "================================================================================"

