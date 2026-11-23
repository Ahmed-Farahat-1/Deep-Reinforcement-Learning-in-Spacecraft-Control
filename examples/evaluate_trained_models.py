"""
Example: Evaluate Trained Champions (TD3 & SAC) against Classical PD Baseline.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from stable_baselines3 import TD3, SAC
from src.envs.spacecraft_env import SpacecraftAttitudeEnv
from src.controllers.pd_controller import PDController
from src.simulation.disturbances import ZeroDisturbance, ConstantDisturbance

def evaluate_controller(name, policy_fn, initial_theta_deg=10.0, dist_torque=0.0):
    env = SpacecraftAttitudeEnv(include_integral=True)
    obs, info = env.reset(options={"initial_state": [np.radians(initial_theta_deg), 0.0, 0.0]})
    if dist_torque != 0:
        env.disturbance = ConstantDisturbance(torque=dist_torque)
        
    done = False
    thetas = []
    while not done:
        action = policy_fn(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        thetas.append(info.get("theta_deg", 0.0))
        done = terminated or truncated
        
    thetas = np.array(thetas)
    final_err = abs(thetas[-1])
    rms_err = np.sqrt(np.mean(thetas**2))
    stabilized = final_err <= 0.20
    return {"Controller": name, "Initial_deg": initial_theta_deg, "Final_Error_deg": final_err, "RMS_Error_deg": rms_err, "Stabilized": stabilized}

def main():
    print("=" * 70)
    print("Evaluating TD3, SAC, and Classical PD on Nominal +10° Scenario")
    print("=" * 70)
    
    # 1. Classical PD
    pd_ctrl = PDController(Kp=5.0, Kd=8.0, max_torque=0.055)
    def pd_policy(obs):
        # obs: [theta_norm, theta_dot_norm, omega_norm, int_err_norm]
        theta_rad = obs[0] * np.radians(30.0)
        theta_dot = obs[1] * np.radians(10.0)
        u_nm, _ = pd_ctrl.compute_torque(theta_rad, theta_dot)
        return np.array([u_nm / 0.055], dtype=np.float32)
        
    res_pd = evaluate_controller("Classical PD", pd_policy)
    print(res_pd)
    
    # Check champion model availability
    td3_path = "models/td3/td3_final_champion_best.zip"
    if os.path.exists(td3_path):
        model_td3 = TD3.load(td3_path)
        res_td3 = evaluate_controller("TD3 Champion", lambda o: model_td3.predict(o, deterministic=True)[0])
        print(res_td3)
        
    sac_path = "models/sac/sac_final_champion_best.zip"
    if os.path.exists(sac_path):
        model_sac = SAC.load(sac_path)
        res_sac = evaluate_controller("SAC Champion", lambda o: model_sac.predict(o, deterministic=True)[0])
        print(res_sac)
        
    print("=" * 70)

if __name__ == '__main__':
    main()
