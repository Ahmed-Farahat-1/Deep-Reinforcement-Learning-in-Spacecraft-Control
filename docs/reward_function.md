# Reward Function Design

The multi-objective reward function balances pointing precision, rate damping, actuator conservation, and wheel speed safety:
$$r_t = r_{pointing} + r_{rate} + r_{action} + r_{wheel} + r_{integral}$$

1. **Huber Fine-Pointing Reward:**
   $$r_{pointing} = 1.0 \cdot \exp(-(	heta / \sigma)^2) - 0.5 \cdot 	ext{Huber}(	heta, \delta=0.05	ext{ rad})$$
2. **Angular Rate Damping Penalty:**
   $$r_{rate} = -0.1 \cdot (\dot{	heta} / \dot{	heta}_{\max})^2$$
3. **Actuator Effort Penalty:**
   $$r_{action} = -0.01 \cdot a^2 - 0.05 \cdot (a_t - a_{t-1})^2$$
4. **Protective Wheel Speed Barrier:**
   $$r_{wheel} = -0.5 \left( rac{|\Omega| - 4500}{5500 - 4500} ight)^2 \quad orall |\Omega| > 4500	ext{ RPM}$$
