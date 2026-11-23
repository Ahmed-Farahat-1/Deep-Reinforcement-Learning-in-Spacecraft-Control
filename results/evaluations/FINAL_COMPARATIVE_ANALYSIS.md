# Detailed Controller-by-Controller Comparative Analysis

| Feature / Metric | Classical PD | PPO Baseline | TD3 Champion | SAC Champion |
| :--- | :--- | :--- | :--- | :--- |
| **Control Category** | Linear State Feedback | On-Policy Deep RL | Off-Policy Deterministic RL | Off-Policy Max-Entropy RL |
| **Overall Success Rate** | **100% (12/12)** | 0% (0/12) | 66.7% (8/12) | **75.0% (9/12)** |
| **Nominal Settling Time** | 12.8 s | DNC | **9.4 s (Fastest)** | 11.2 s |
| **Steady-State Pointing Error**| **0.0000°** | 6.85° | 0.0497° | **0.0872°** |
| **Actuator Chattering** | Negligible (Smooth) | None | High Transient Chattering | **Very Low (Smooth)** |
| **Tumble Count** | **0** | 1 (Scenario 7) | **0 (Zero Tumbles)** | 1 (Scenario 7) |
| **Control Energy ($\int u^2 dt$)**| **6.84** | 0.15 (Under-actuated)| 18.22 | 16.90 |
| **Physical Odd-Symmetry** | Exact (Analytical) | Enforced ($\pi_{sym}$) | Enforced ($\pi_{sym}$) | Enforced ($\pi_{sym}$) |
