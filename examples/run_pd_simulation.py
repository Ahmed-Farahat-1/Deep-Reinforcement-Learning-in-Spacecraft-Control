"""
Example: Run Classical PD Spacecraft Attitude Stabilization Simulation.
Demonstrates closed-loop stabilization from +10 deg initial error under constant disturbance.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
from src.simulation.spacecraft import Spacecraft1D
from src.simulation.disturbances import ConstantDisturbance
from src.controllers.pd_controller import PDController
from src.evaluation.metrics import calculate_performance_metrics

def main():
    print("=" * 60)
    print("Running Classical PD Spacecraft Attitude Stabilization")
    print("=" * 60)
    
    # Spacecraft & Actuator parameters (J_body = 10.0 kg*m^2, I_wheel = 0.05 kg*m^2 -> J_total = 10.05 kg*m^2)
    spacecraft = Spacecraft1D(J_body=10.0, I_wheel=0.05, max_torque=0.055, max_speed=575.96)
    disturbance = ConstantDisturbance(torque_value=0.01)  # 0.01 Nm constant disturbance
    controller = PDController(Kp=5.0, Kd=8.0, max_torque=0.055)
    
    # Initial conditions
    theta_0 = np.radians(10.0)  # +10 degrees
    theta_dot_0 = 0.0
    omega_0 = 0.0
    spacecraft.reset(initial_state=np.array([theta_0, theta_dot_0, omega_0]))
    
    dt = 0.01
    sim_time = 30.0
    steps = int(sim_time / dt)
    
    times = []
    thetas = []
    theta_dots = []
    wheel_speeds = []
    torques = []
    
    for i in range(steps):
        t = i * dt
        state = spacecraft.state
        theta, theta_dot, omega = state[0], state[1], state[2]
        
        # PD control law: u = Kp * e + Kd * e_dot
        u, _ = controller.compute_torque(theta, theta_dot)
        Md = disturbance.torque(t)
        
        spacecraft.step(u, Md, dt)
        
        times.append(t)
        thetas.append(theta)
        theta_dots.append(theta_dot)
        wheel_speeds.append(omega)
        torques.append(u)
        
    metrics = calculate_performance_metrics(
        time=np.array(times),
        theta=np.array(thetas),
        theta_dot=np.array(theta_dots),
        Omega=np.array(wheel_speeds),
        torque=np.array(torques),
        scenario_id="example_pd_sim"
    )
    thetas_deg = np.degrees(np.array(thetas))
    rms_error_deg = np.sqrt(np.mean(thetas_deg**2))
    final_error_deg = metrics.get('final_error_from_target_deg', abs(thetas_deg[-1]))
    is_stabilized = final_error_deg <= 0.20 and metrics.get('settling_time_s') is not None

    print(f"Settling Time (s):     {metrics.get('settling_time_s', 'N/A')}")
    print(f"RMS Attitude Error:    {rms_error_deg:.4f} deg")
    print(f"Final Attitude Error:  {final_error_deg:.6f} deg")
    print(f"Control Effort:        {metrics.get('squared_control_torque_integral_N2m2s', 0.0):.4f} N^2*s")
    print(f"Max Wheel Speed:       {metrics.get('max_wheel_speed_rpm', 0.0):.1f} RPM")
    print(f"Stabilized (<= 0.20°): {is_stabilized}")
    print("=" * 60)

if __name__ == '__main__':
    main()
