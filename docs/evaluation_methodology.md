# Evaluation Methodology & Benchmark Suite

## 1. Performance Metrics
- **Settling Time ($T_s$):** Time required for attitude error to enter and permanently remain within $\pm 0.20^\circ$.
- **Steady-State Error ($e_{ss}$):** Absolute attitude error at the final simulation timestep ($t = 30	ext{ s}$).
- **RMS Attitude Error:** Root-mean-square attitude error over the entire trajectory.
- **Control Effort:** Integral of squared torque $\int_0^T u^2 dt$ ($	ext{N}^2\cdot	ext{s}$).
- **Stabilization Pass:** Boolean criteria: $e_{ss} \le 0.20^\circ$ and $T_s < T_{sim}$.
- **Tumble Count:** Vehicle tumbling beyond $\pm 90^\circ$.

## 2. Benchmark Scenarios Suite (12 Scenarios)
- **Nominal Validation Suite (S1–S7):** Nominal angles ($\pm 10^\circ$, $+15^\circ$), constant disturbance ($0.01	ext{ Nm}$), sinusoidal disturbance ($0.05	ext{ Hz}$), torque pulse ($0.03	ext{ Nm}$), and momentum stress ($\Omega_0 = 4200	ext{ RPM}$).
- **Held-Out Generalization Suite (H1–H5):** Extreme angles ($\pm 17.5^\circ$, $-22.5^\circ$), mixed initial rates, high-frequency sinusoid ($0.25	ext{ Hz}$), and heavy constant disturbance ($-0.035	ext{ Nm}$).
