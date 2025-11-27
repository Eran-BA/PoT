#!/bin/bash
# Monitor maze scaling benchmark progress
# Usage: ./monitor_progress.sh [log_file]

LOG_FILE="${1:-experiments/results/maze_local_investigation.log}"

echo "📊 Monitoring: $LOG_FILE"
echo "Press Ctrl+C to stop monitoring"
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

while true; do
    clear
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🔬 MAZE SCALING BENCHMARK - LIVE MONITOR"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "⏰ Last updated: $(date '+%H:%M:%S')"
    echo ""
    
    if [ ! -f "$LOG_FILE" ]; then
        echo -e "${RED}❌ Log file not found: $LOG_FILE${NC}"
        echo "Waiting for benchmark to start..."
        sleep 5
        continue
    fi
    
    # Current maze size
    CURRENT_SIZE=$(grep "MAZE SIZE:" "$LOG_FILE" | tail -1)
    if [ -n "$CURRENT_SIZE" ]; then
        echo -e "${BLUE}$CURRENT_SIZE${NC}"
        echo ""
    fi
    
    # Current model training
    CURRENT_MODEL=$(grep "Training:" "$LOG_FILE" | tail -1)
    if [ -n "$CURRENT_MODEL" ]; then
        echo -e "${YELLOW}$CURRENT_MODEL${NC}"
    fi
    
    # Latest epoch info
    LATEST_EPOCH=$(grep "Epoch.*Loss:" "$LOG_FILE" | tail -1)
    if [ -n "$LATEST_EPOCH" ]; then
        echo "  $LATEST_EPOCH"
    fi
    echo ""
    
    # Completed maze sizes
    echo -e "${GREEN}✅ Completed Sections:${NC}"
    grep "Final Performance:" "$LOG_FILE" | while read line; do
        echo "  • $line"
    done
    echo ""
    
    # Summary of results so far
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📈 RESULTS SUMMARY:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Extract accuracy and optimality for each model
    grep -E "Baseline|BERT|PoH-HRM" "$LOG_FILE" | grep -A2 "Training:" | grep -E "Accuracy|Optimality" | tail -9
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Press Ctrl+C to stop monitoring"
    
    # Check if process is still running
    if pgrep -f "maze_scaling_benchmark.py" > /dev/null; then
        echo -e "${GREEN}✓ Process is running${NC}"
    else
        echo -e "${YELLOW}⚠ Process not found - may have completed or stopped${NC}"
    fi
    
    sleep 300  # Update every 5 minutes
done

