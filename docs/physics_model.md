# Physical Model & Spacecraft Dynamics

## 1. Spacecraft-Reaction Wheel System
The spacecraft is modeled as a rigid body with principal moment of inertia $J = 10.05	ext{ kg}\cdot	ext{m}^2$ rotating about its Z-axis (yaw). A reaction wheel with rotor moment of inertia $I_w = 0.05	ext{ kg}\cdot	ext{m}^2$ is aligned along the body Z-axis.

## 2. Governing Equations of Motion
Total angular momentum about the Z-axis in the inertial frame is:
$$H = J \dot{	heta} + I_w \Omega$$

Differentiating with respect to time:
$$\dot{H} = J \ddot{	heta} + I_w (\ddot{	heta} + \dot{\Omega}) = M_d$$

Since motor torque $u$ accelerates the wheel ($I_w (\ddot{	heta} + \dot{\Omega}) = u pprox I_w \dot{\Omega}$):
$$J \ddot{	heta} = M_d - u$$
$$\dot{\Omega} = rac{u}{I_w}$$

## 3. Physical Parameters Table
| Parameter | Symbol | Value | Unit | Description |
| :--- | :--- | :--- | :--- | :--- |
| Bus Inertia | $J$ | 10.05 | $	ext{kg}\cdot	ext{m}^2$ | Principal spacecraft moment of inertia |
| Wheel Inertia | $I_w$ | 0.05 | $	ext{kg}\cdot	ext{m}^2$ | Reaction wheel rotor inertia |
| Max Motor Torque | $	au_{\max}$ | 0.055 | $	ext{N}\cdot	ext{m}$ | Motor saturation limit |
| Max Wheel Speed | $\Omega_{\max}$ | 5500.0 | $	ext{RPM}$ | Hard mechanical speed limit ($pprox 575.96	ext{ rad/s}$) |
| Soft Wheel Limit | $\Omega_{soft}$ | 4500.0 | $	ext{RPM}$ | Penalty barrier during RL training |
| Integration Step | $\Delta t$ | 0.01 | $	ext{s}$ | Fixed RK4 integration timestep |
| Simulation Horizon | $T$ | 30.0 | $	ext{s}$ | Closed-loop scenario duration |
