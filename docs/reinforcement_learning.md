# Reinforcement Learning Formulation

## 1. Markov Decision Process (MDP) Tuple
$$\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma angle$$

- **State Space ($\mathcal{S} \subset \mathbb{R}^4$):**
  - $	heta_{norm} = 	heta / 	heta_{\max} \in [-1, 1]$ ($	heta_{\max} = 30^\circ$)
  - $\dot{	heta}_{norm} = \dot{	heta} / \dot{	heta}_{\max} \in [-1, 1]$ ($\dot{	heta}_{\max} = 10^\circ/	ext{s}$)
  - $\Omega_{norm} = \Omega / \Omega_{\max} \in [-1, 1]$ ($\Omega_{\max} = 5500	ext{ RPM}$)
  - $e_{I,norm} = e_I / e_{I,\max} \in [-1, 1]$ (Anti-windup integrated error)
- **Action Space ($\mathcal{A} \subset [-1, 1]$):**
  - Normalized motor torque command: $u = a \cdot 	au_{\max}$.
- **Discount Factor ($\gamma$):**
  - $0.995$ (TD3 and SAC), $0.998$ (PPO).

## 2. Anti-Symmetric Neural Policy Architecture
Spacecraft rotational dynamics about an inertial axis satisfy odd symmetry: reversing the initial state must reverse the commanded torque:
$$\pi(-s) = -\pi(s)$$

We enforce this directly in the network forward pass:
$$\pi_{sym}(s) = rac{1}{2} \Big( \pi(s) - \pi(-s) \Big)$$
