# Experimental Results & Comparative Analysis

## 1. Master Benchmark Performance (12 Scenarios)
| Controller | Validation Pass (S1–S7) | Held-Out Pass (H1–H5) | Overall Pass Rate | Mean Settling Time | Final Pointing Error | Tumbles | Energy ($	ext{N}^2	ext{s}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Classical PD** | **7/7 (100%)** | **5/5 (100%)** | **12/12 (100%)** | 12.8 s | **0.0000°** | **0** | **6.84** |
| **SAC Champion** | 6/7 (85.7%) | **3/5 (60.0%)** | **9/12 (75.0%)** | 11.2 s | 0.0872° | 1 (S7) | 16.90 |
| **TD3 Champion** | 6/7 (85.7%) | 2/5 (40.0%) | 8/12 (66.7%) | **9.4 s** | 0.0497° | **0** | 18.22 |
| **PPO Baseline** | 0/7 (0.0%) | 0/5 (0.0%) | 0/12 (0.0%) | DNC | 6.8500° | 1 (S7) | 0.15 |

## 2. Key Observations
1. **Classical PD:** Unmatched steady-state precision, zero overshoot, minimal energy.
2. **TD3 Champion:** Fastest initial slew ($9.4	ext{ s}$), zero tumbles across all 12 scenarios.
3. **SAC Champion:** Best generalization among RL agents (stabilized 3/5 held-out scenarios), smooth action commands.
4. **PPO Baseline:** Suffered from low control torque commands near the origin.
