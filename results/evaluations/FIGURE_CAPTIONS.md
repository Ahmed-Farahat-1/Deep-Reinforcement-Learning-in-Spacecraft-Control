# Comprehensive Figure Captions & Data Provenance

- **Figure 1.1:** Closed-Loop GNC Architecture. Schematic diagram of spacecraft attitude control loop with reaction wheel and RK4 dynamics. Provenance: `scripts/generate_finals_figures.py`.
- **Figure 2.1:** Spacecraft Rigid Body Schematic. Cross-sectional layout of the spacecraft bus ($J = 10.05\text{ kg}\cdot\text{m}^2$) and reaction wheel ($I_w = 0.05\text{ kg}\cdot\text{m}^2$).
- **Figure 3.1:** Controller Architectures Comparison. High-level mathematical formulation comparison of Classical PD, PPO, TD3, and SAC.
- **Figure 4.1:** Multi-Seed Learning Curves. Training reward progression across 3 independent random seeds ($42, 123, 2026$) over $500,000$ environment steps. Source: TensorBoard logs in `finals/03_RAW_DATA/TRAINING_TELEMETRY/`.
- **Figure 5.1:** Nominal Attitude Tracking Responses. Closed-loop attitude error trajectories $\theta(t)$ across Scenarios S1 through S4 comparing all 4 controllers.
- **Figure 7.2:** Reaction Wheel Momentum Saturation (Scenario 7). Wheel speed trajectory hitting hard $5500\text{ RPM}$ saturation boundary.
- **Figure 9.1:** Held-Out Generalization Evaluation. Response across 5 extreme unseen scenarios (H1 to H5).
- **Figure 10.1:** Master 12-Scenario Success Matrix Heatmap. Comprehensive pass/fail matrix under $\pm 0.20^\circ$ pointing criterion.
