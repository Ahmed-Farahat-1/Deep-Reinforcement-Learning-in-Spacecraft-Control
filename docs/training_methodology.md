# Training Methodology & Multi-Seed Protocol

## 1. Off-Policy Algorithms (TD3 & SAC)
- **Training Budget:** $500,000$ environment timesteps per seed.
- **Random Seeds:** $42$, $123$, $2026$ (totaling $1,500,000$ steps for TD3 and $1,500,000$ steps for SAC).
- **Replay Buffer:** $200,000$ transitions with uniform sampling.
- **Batch Size:** $256$.
- **Learning Rate:** $3 	imes 10^{-4}$ (Adam optimizer).
- **Target Network Polyak Factor:** $	au = 0.005$.
- **Policy Update Frequency:** Delayed $d=2$ in TD3; $d=1$ with auto-tuned entropy temperature $lpha$ in SAC.

## 2. Checkpoint Selection Gate
Checkpoints saved every $50,000$ steps were evaluated across an 8-scenario suite using a composite performance score:
$$	ext{Score} = 0.6 \cdot R_{norm} + 0.2 \cdot S_{norm} + 0.1 \cdot E_{norm} - 500 \cdot N_{tumble}$$
- **TD3 Champion:** Seed 123 @ 350,000 steps (Composite Score: $+177.28$).
- **SAC Champion:** Seed 123 @ 250,000 steps (Composite Score: $+178.78$).
