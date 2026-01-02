# Self-Modifying Parameter Optimizer

A Python evolution engine that modifies its own source code parameters and measures performance improvement.

## What It Does

1. **Modifies** parameters in `UNIFIED_RSI_EXTENDED.py`
2. **Tests** modifications via benchmark (5 generations, random seeds)
3. **Accepts** changes only if score improves (Δ > 0.001)
4. **Logs** all attempts in `.rsi_state/simple_rsi_log.jsonl`

## Verified Results

- **Performance gain**: 13.2% (score: 13.685 → 11.871)
- **Modified parameters**: `reuse_weight`, `transfer_weight`, `task_count`
- **Tests performed**: 27 modifications tested across 3 rounds
- **Evidence**: Backup file + detailed logs included

## Files

- `UNIFIED_RSI_EXTENDED.py` - Evolution engine (gets modified)
- `simple_rsi.py` - Parameter optimizer (does the modification)

## Usage

```bash
# Run parameter optimization
python simple_rsi.py --engine UNIFIED_RSI_EXTENDED.py --iterations 20

# Test the engine
python UNIFIED_RSI_EXTENDED.py evolve --fresh --generations 10 --seed 42

# View modification log
cat .rsi_state/simple_rsi_log.jsonl
```

## How It Works

1. **Backup**: Creates `UNIFIED_RSI_EXTENDED.py.simple_backup`
2. **Grid Search**: Tests parameters × multipliers (0.5, 0.8, 1.2, 1.5, 2.0)
3. **Benchmark**: Runs evolution for 5 generations with random seed
4. **Validate**: Passes `test_suite.py` before accepting
5. **Apply**: Writes best modification to source file

## Key Features

- **Regex-based modification**: Avoids AST unparsing issues
- **Random seeds**: Breaks determinism for fair comparison
- **Test suite validation**: Ensures modifications don't break functionality
- **Automatic rollback**: Restores backup if modification fails

## Evidence of Self-Modification

### Before (backup)
```python
class RewardModel:
    reuse_weight: float = 0.38883  # Original
```

### After (modified)
```python
class RewardModel:
    reuse_weight: float = 0.583245  # Modified by simple_rsi.py
```

## Requirements

- Python 3.8+
- Standard library only (no external dependencies)

## This is parameter optimization with source code modification.
The engine modifies numeric constants in its own file and measures if performance improves.

## License

MIT
