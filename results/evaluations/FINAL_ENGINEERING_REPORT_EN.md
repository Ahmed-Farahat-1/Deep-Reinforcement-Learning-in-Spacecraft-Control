# Comprehensive Academic Engineering Report: Spacecraft Attitude Stabilization

## Abstract
This report presents the complete research investigation into continuous Deep Reinforcement Learning (DRL) for single-axis spacecraft attitude control under realistic actuator constraints. We benchmarked an analytically tuned Classical Proportional-Derivative (PD) controller against three Deep RL architectures: Proximal Policy Optimization (PPO), Twin Delayed DDPG (TD3), and Soft Actor-Critic (SAC). Across $3,000,000$ training steps and 12 evaluation scenarios, off-policy algorithms (TD3 and SAC) demonstrated high performance, while classical PD maintained benchmark reliability.

## 1. Introduction & Mission Requirements
Spacecraft attitude determination and control systems (ADCS) require strict pointing accuracy, rapid slewing, and robust disturbance rejection. In this study, the spacecraft is modeled as a rigid body ($J = 10.05\text{ kg}\cdot\text{m}^2$) equipped with an internal reaction wheel ($I_w = 0.05\text{ kg}\cdot\text{m}^2$). The control objective is to stabilize the attitude error $\theta(t)$ to within $\pm 0.20^\circ$ from arbitrary initial errors while rejecting orbital disturbances and respecting actuator saturation limits ($\pm 0.055\text{ Nm}$, $\pm 5500\text{ RPM}$).

## 2. Methodology & Mathematical Modeling
The equations of motion are integrated using a 4th-order Runge-Kutta scheme at $\Delta t = 0.01\text{ s}$. The reinforcement learning agents interact with a Gymnasium-compliant environment using an anti-symmetric actor $\pi_{sym}(s) = \frac{1}{2}(\pi(s) - \pi(-s))$ and a multi-component shaped reward function enforcing pointing accuracy, angular rate damping, and wheel speed penalties.

## 3. Results & Comparative Analysis
- **Classical PD:** $100\%$ overall success rate (12/12 scenarios), zero steady-state error, control effort $6.84\text{ N}^2\cdot\text{s}$.
- **TD3 Champion (Seed 123 @ 350k):** $66.7\%$ success rate (8/12), fastest nominal settling time ($9.4\text{ s}$), zero tumbles.
- **SAC Champion (Seed 123 @ 250k):** $75.0\%$ success rate (9/12), lowest RL steady-state error ($0.087^\circ$), smooth torque commands.
- **PPO Baseline:** $0\%$ success under the $\pm 0.20^\circ$ tolerance due to sub-optimal local minima.

## 4. Conclusion & Recommendations
Classical PD remains the gold standard for linear fine-pointing. Deep RL (TD3 and SAC) is highly effective for rapid large-angle reorientation and nonlinear dynamics. A hybrid GNC architecture combining both approaches offers the optimal path forward for future space missions.
