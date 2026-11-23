# Data Dictionary & Schema Documentation

## 1. Trajectory Datasets (`data/trajectories/*.csv`)
Synchronized multi-controller time histories for Scenarios S1–S7 and H1–H5:
- `time_s`: Elapsed simulation time [s] ($0.0$ to $29.99	ext{ s}$, $\Delta t = 0.01	ext{ s}$).
- `pd_theta_deg`: Classical PD attitude error [deg].
- `pd_wheel_rpm`: Classical PD reaction wheel speed [RPM].
- `pd_u_applied`: Classical PD normalized torque command $[-1.0, 1.0]$.
- `ppo_theta_deg`, `ppo_wheel_rpm`, `ppo_u_applied`: PPO telemetry.
- `td3_theta_deg`, `td3_wheel_rpm`, `td3_u_applied`: TD3 champion telemetry.
- `sac_theta_deg`, `sac_wheel_rpm`, `sac_u_applied`: SAC champion telemetry.

## 2. Evaluation Summary Table (`data/evaluation/comparative_metrics_all_scenarios.csv`)
- `scenario_id`: Unique identifier (e.g., `scenario_1_pos10deg`).
- `pd_is_stabilized`: Boolean pass flag for Classical PD.
- `pd_final_error_deg`: Pointing error at final timestep.
- `td3_settling_time_s`: Settling time for TD3.
- `sac_settling_time_s`: Settling time for SAC.

## 3. STK Ephemeris (`STK/attitude_ephemeris/*.a`)
- ASCII STK Attitude format with Sequence 321 Euler Angles (Yaw, Pitch, Roll) sampled at $10	ext{ Hz}$.
