# Local Maze Investigation Tools

This directory contains tools for investigating the PoH-HRM 16×16 maze optimality issue locally on your Mac.

## 🚀 Quick Start

### 1. Run the Main Investigation (16×16, 20×20, 30×30)

```bash
cd /Users/rnbnrzy/Desktop/PoT
./experiments/run_local_investigation.sh
```

**What it does:**
- Tests maze sizes: 16×16, 20×20, 30×30
- 200 training samples per size
- 50 test samples per size  
- 30 epochs per size
- Config: R=4, T=4, n_heads=4

**Expected runtime:** ~3-3.5 hours

**Output:**
- `experiments/results/maze_local_investigation.json` - Full results
- `experiments/results/maze_local_investigation.png` - Scaling plot
- `experiments/results/maze_local_investigation.log` - Live log

---

## 📊 Monitoring Tools

### Quick Status Check

```bash
./experiments/check_status.sh
```

Shows:
- ✅ Is benchmark running?
- 📍 Current maze size & model
- 📊 Latest results
- ⏱️ Estimated time remaining

**Use this frequently** to check progress without cluttering terminal.

### Live Monitor (Updates every 5 min)

```bash
./experiments/monitor_progress.sh
```

Shows:
- Current maze size and model
- Latest epoch metrics
- All completed results
- Live process status

**Press Ctrl+C to stop monitoring**

### View Raw Log

```bash
tail -f experiments/results/maze_local_investigation.log
```

---

## 🔬 Hyperparameter Investigation

Once the main run completes (or if 16×16 still shows low optimality), run these tests:

### Test Different R, T, n_heads for 16×16

```bash
./experiments/quick_16x16_tests.sh
```

**What it tests:**
1. **R variation**: R=2, 4, 6, 8 (keeps T=4, heads=4)
2. **T variation**: T=2, 4, 8, 16 (keeps R=4, heads=4)  
3. **n_heads variation**: heads=2, 4, 8 (keeps R=4, T=4)

**Total runtime:** ~4-5 hours (10 experiments × ~30 min each)

**Results:** `experiments/results/16x16_R*_T*_h*.json`

---

## 📈 Results Analysis

### Quick Results Check

```python
# In Python/iPython:
import json

# Load main results
with open('experiments/results/maze_local_investigation.json') as f:
    results = json.load(f)

# Check PoH optimality for each size
for i, size in enumerate(results['maze_sizes']):
    poh_opt = results['poh']['optimality'][i]
    baseline_opt = results['baseline']['optimality'][i]
    print(f"{size}×{size}: PoH {poh_opt:.1%}, Baseline {baseline_opt:.1%}")
```

### Compare Hyperparameter Sweeps

```python
# Check all 16×16 experiments
import glob
import json

for f in sorted(glob.glob('experiments/results/16x16_*.json')):
    with open(f) as file:
        r = json.load(file)
        poh_opt = r['poh']['optimality'][0]
        print(f"{f.split('/')[-1]}: {poh_opt:.1%}")
```

---

## 🎯 What We're Looking For

### Main Investigation Results:

**Good outcomes:**
```
16×16: PoH 70-80% opt (improved with harder mazes!)
20×20: PoH 75-85% opt
30×30: PoH 80-90% opt (HRM shining at scale)
```

**Bad outcomes (problem persists):**
```
16×16: PoH still ~48% opt
→ Run hyperparameter tests
```

### Hyperparameter Test Results:

**H1 (Insufficient R):** If R=6 or R=8 improves 16×16 → Need more iterations
**H2 (T mismatch):** If T=2 or T=8 improves 16×16 → Period needs tuning
**H3 (Head saturation):** If n_heads=8 improves 16×16 → Need more capacity
**H4 (No improvement):** If none help → Deeper investigation needed

---

## 🛑 Stop/Control

### Stop the benchmark:
```bash
pkill -f maze_scaling_benchmark.py
```

### Check if running:
```bash
ps aux | grep maze_scaling_benchmark
```

### Resume from checkpoint:
(Not currently supported - would need to add checkpoint saving)

---

## 📁 File Structure

```
experiments/
├── run_local_investigation.sh      # Main runner
├── check_status.sh                 # Quick status
├── monitor_progress.sh             # Live monitor
├── quick_16x16_tests.sh           # Hyperparameter sweep
├── MAZE_POH_16x16_INVESTIGATION.md # Investigation doc
└── results/
    ├── maze_local_investigation.json
    ├── maze_local_investigation.png
    ├── maze_local_investigation.log
    └── 16x16_R*_T*_h*.json         # Hyperparameter results
```

---

## 💡 Tips

1. **Start with main investigation** to see if harder mazes fix the issue
2. **Monitor with check_status.sh** every ~30 min
3. **Only run hyperparameter tests** if problem persists
4. **Compare with Colab results** (full dataset, 6 sizes)
5. **Document findings** in MAZE_POH_16x16_INVESTIGATION.md

---

## 🚨 Troubleshooting

### "python: command not found"
Use `python3` instead of `python` on Mac

### "Permission denied"
Run `chmod +x experiments/*.sh` to make scripts executable

### Process killed/crashed
Check log file: `tail experiments/results/maze_local_investigation.log`

### Out of memory
Reduce `--train` and `--test` sizes:
```bash
python3 experiments/maze_scaling_benchmark.py \
    --maze-sizes 16 --train 100 --test 30 ...
```

---

## 📞 Next Steps

After local investigation completes:

1. **Compare with Colab** full benchmark results
2. **Update MAZE_POH_16x16_INVESTIGATION.md** with findings
3. **If problem persists:** Run `quick_16x16_tests.sh`
4. **If solved:** Document solution and update benchmark defaults
5. **Share results:** Push findings to repo

Happy investigating! 🔬

