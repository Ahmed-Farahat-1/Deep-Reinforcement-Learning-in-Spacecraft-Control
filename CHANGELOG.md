# Research Project Changelog

## [1.0.0] - 2026-11-23: Phase 5 Final Packaging & GitHub Release
- Built clean, reproducible, self-contained academic repository.
- Integrated full Ansys STK 12 scenario and 20 standardized attitude ephemeris files.
- Structured documentation across 12 comprehensive academic guides.
- Published verified 85/85 automated test suite results.
- Archived full comparative benchmark datasets (12 scenarios across Classical PD, PPO, TD3, SAC).

## [0.4.7] - 2026-11-22: Phase 4G Autonomous Off-Policy DRL Experiment
- Implemented and trained Twin Delayed DDPG (TD3) and Soft Actor-Critic (SAC).
- Conducted multi-seed training across Seeds 42, 123, and 2026 for 500,000 steps each (3,000,000 total cumulative steps).
- Selected champion models: TD3 Seed 123 @ 350k (Score +177.28) and SAC Seed 123 @ 250k (Score +178.78).
- Executed master 12-scenario comparative evaluation suite.

## [0.4.5] - 2026-11-21: Phase 4E PPO Anti-Symmetric Redesign & Training
- Formulated custom Anti-Symmetric Policy Network enforcing pi(-s) = -pi(s).
- Implemented Huber fine-pointing reward and reaction wheel quadratic penalty barrier.
- Verified mathematical odd-symmetry across positive and negative initial attitude errors.

## [0.4.3] - 2026-11-21: Phase 4C Root-Cause Investigation & Audit
- Diagnosed PPO pilot policy drift and under-actuation in fine-pointing deadbands.
- Identified angular momentum conservation feasibility constraints under continuous disturbance.

## [0.3.0] - 2026-11-21: Phase 3 Farama Gymnasium Environment
- Standardized custom Gymnasium environment with normalized observation space and continuous action space.
- Validated Runge-Kutta 4th-order numerical integration at dt = 0.01 s.

## [0.2.0] - 2026-11-21: Phase 2 Physics Engine & Classical PD Baseline
- Implemented single-axis rigid-body spacecraft rotational dynamics with internal reaction wheel.
- Tuned classical continuous PD controller (Kp = 5.0, Kd = 8.0, damping ratio = 0.89).
