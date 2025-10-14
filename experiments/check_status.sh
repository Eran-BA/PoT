#!/bin/bash
# Quick status check for maze benchmark
# Usage: ./check_status.sh

LOG_FILE="experiments/results/maze_local_investigation.log"

echo "🔍 Quick Status Check"
echo "═══════════════════════════════════════════════════════"
echo ""

# Check if running
if pgrep -f "maze_scaling_benchmark.py" > /dev/null; then
    echo "✅ Benchmark is RUNNING"
    PID=$(pgrep -f "maze_scaling_benchmark.py")
    echo "   PID: $PID"
else
    echo "⚠️  Benchmark is NOT running"
fi
echo ""

# Check if log exists
if [ ! -f "$LOG_FILE" ]; then
    echo "❌ No log file found yet"
    echo "   Waiting for: $LOG_FILE"
    exit 0
fi

# Current progress
echo "📍 Current Progress:"
CURRENT_SIZE=$(grep "MAZE SIZE:" "$LOG_FILE" | tail -1)
if [ -n "$CURRENT_SIZE" ]; then
    echo "   $CURRENT_SIZE"
fi

CURRENT_MODEL=$(grep "Training:" "$LOG_FILE" | tail -1)
if [ -n "$CURRENT_MODEL" ]; then
    echo "   $CURRENT_MODEL"
fi

LATEST_EPOCH=$(grep "Epoch.*Loss:" "$LOG_FILE" | tail -1)
if [ -n "$LATEST_EPOCH" ]; then
    echo "   $LATEST_EPOCH"
fi
echo ""

# Completed sections
COMPLETED=$(grep -c "Final Performance:" "$LOG_FILE")
echo "✅ Completed: $COMPLETED/9 model trainings"
echo ""

# Latest results
echo "📊 Latest Results:"
grep "Final Performance:" "$LOG_FILE" | tail -3 | while read line; do
    echo "   • $line"
done
echo ""

# Estimated time remaining
if [ $COMPLETED -gt 0 ]; then
    REMAINING=$((9 - COMPLETED))
    EST_MIN=$((REMAINING * 20))
    echo "⏱️  Estimated time remaining: ~$EST_MIN minutes"
fi

echo ""
echo "═══════════════════════════════════════════════════════"
echo "💡 Tips:"
echo "   • Full monitor: ./experiments/monitor_progress.sh"
echo "   • View log: tail -f $LOG_FILE"
echo "   • Stop: pkill -f maze_scaling_benchmark.py"

