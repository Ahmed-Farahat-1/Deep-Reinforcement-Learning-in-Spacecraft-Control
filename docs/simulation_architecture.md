# Simulation Architecture & Data Flow

## 1. Closed-Loop Execution Loop
The closed-loop architecture follows the standard aerospace Guidance, Navigation, and Control (GNC) structure:
1. **Sensors / Observation:** Spacecraft state $x = [	heta, \dot{	heta}, \Omega, e_I]$ is observed and normalized.
2. **Controller:** Computes commanded torque $u_{cmd}$ (Classical PD or Deep RL Actor).
3. **Actuator Non-Linearity:** Motor torque is clipped to $\pm 0.055	ext{ Nm}$. If wheel speed reaches $\pm 5500	ext{ RPM}$, accelerating torque is zeroed.
4. **Physics Engine (RK4):** Evaluates system state derivative $\dot{x} = f(x, u, M_d)$ four times per timestep.
5. **Telemetry Logger:** Records time, attitude error, angular rate, wheel speed, and torque.

## 2. Module Mapping
- `src/simulation/spacecraft.py`: Core `Spacecraft1D` rigid-body dynamics class.
- `src/simulation/disturbances.py`: Analytical disturbance torque generators.
- `src/envs/spacecraft_env.py`: Farama Gymnasium environment interface.
- `src/controllers/pd_controller.py`: Continuous classical PD control law.
- `src/agents/`: RL training, evaluation, and policy network implementations.
