# Project Overview

## 1. Title
**AI-Driven Spacecraft Attitude Stabilization Using Deep Reinforcement Learning: A Comparative Study of Classical PD, PPO, TD3, and SAC Controllers.**

## 2. Research Author & Institutional Context
- **Author:** Ahmed Farahat
- **Academic Position:** M.Sc. Student — Aircraft Overall Design
- **Affiliation:** School of Astronautics, Beihang University
- **Course:** Spacecraft Control
- **Course Lecturer:** Yang Chen

## 3. Executive Abstract
This repository contains the complete open-source research package investigating autonomous attitude stabilization of a single-axis spacecraft equipped with an internal reaction wheel momentum actuator. The dynamics are governed by non-linear rigid-body rotational equations integrated via 4th-order Runge-Kutta (RK4) integration with electromechanical actuator constraints: motor torque saturation ($\pm 0.055	ext{ Nm}$) and rotor speed saturation ($\pm 5500	ext{ RPM}$).

Four control architectures were comparatively evaluated across 12 operational scenarios:
1. **Classical PD:** Linear proportional-derivative state feedback ($K_p = 5.0, K_d = 8.0$).
2. **PPO (Proximal Policy Optimization):** On-policy actor-critic with anti-symmetric policy.
3. **TD3 (Twin Delayed DDPG):** Off-policy deterministic actor-critic with twin Q-critics.
4. **SAC (Soft Actor-Critic):** Off-policy maximum-entropy stochastic actor-critic.

## 4. Key Scientific Findings
- **Classical PD is the Benchmark Standard:** Achieved $100\%$ success (12/12 scenarios), machine-precision pointing, and minimal energy consumption ($6.84	ext{ N}^2\cdot	ext{s}$).
- **Off-Policy Superiority:** TD3 and SAC dramatically outperformed PPO, achieving rapid settling times ($9.4	ext{ s}$ and $11.2	ext{ s}$).
- **Conservation of Angular Momentum:** In Scenario 7 (initial wheel speed $\Omega_0 = 4200	ext{ RPM}$ with $+0.02	ext{ Nm}$ disturbance), internal reaction wheel desaturation is physically impossible without external thrusters, resulting in inevitable vehicle drift.
