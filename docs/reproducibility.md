# Reproducibility Guide

## 1. Quick Verification (Test Suite)
Run all 85 unit and integration tests:
```bash
pytest tests/ -v
```
Expected result: **85 passed in ~20 seconds**.

## 2. Re-Running Classical PD Simulation
```bash
python examples/run_pd_simulation.py
```

## 3. Evaluating Trained RL Checkpoints
```bash
python examples/evaluate_trained_models.py
```

## 4. Re-Generating Figures & Visualizations
```bash
python examples/visualize_trajectories.py
```
