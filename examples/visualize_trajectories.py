"""
Example: Visualize Pre-computed Comparative Trajectories (PD vs PPO vs TD3 vs SAC).
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import pandas as pd

def main():
    traj_file = "data/trajectories/trajectory_scenario_1_pos10deg.csv"
    if not os.path.exists(traj_file):
        print(f"File not found: {traj_file}")
        return
        
    df = pd.read_csv(traj_file)
    print(f"Loaded {len(df)} simulation timesteps from {traj_file}")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # 1. Attitude Error
    ax1.plot(df['time_s'], df['pd_theta_deg'], label='Classical PD', color='#1f77b4', lw=2)
    ax1.plot(df['time_s'], df['td3_theta_deg'], label='TD3 Champion', color='#2ca02c', lw=2)
    ax1.plot(df['time_s'], df['sac_theta_deg'], label='SAC Champion', color='#9467bd', lw=2)
    ax1.plot(df['time_s'], df['ppo_theta_deg'], label='PPO', color='#d62728', linestyle=':', lw=1.5)
    ax1.axhline(0.2, color='red', linestyle='--', alpha=0.5, label='Tolerance (±0.2°)')
    ax1.axhline(-0.2, color='red', linestyle='--', alpha=0.5)
    ax1.set_ylabel('Attitude Error [deg]')
    ax1.set_title('Attitude Stabilization Response')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # 2. Wheel Speed
    ax2.plot(df['time_s'], df['pd_wheel_rpm'], label='PD', color='#1f77b4')
    ax2.plot(df['time_s'], df['td3_wheel_rpm'], label='TD3', color='#2ca02c')
    ax2.plot(df['time_s'], df['sac_wheel_rpm'], label='SAC', color='#9467bd')
    ax2.axhline(4500, color='orange', linestyle='--', label='Soft Limit (4500 RPM)')
    ax2.set_ylabel('Wheel Speed [RPM]')
    ax2.set_title('Reaction Wheel Rotor Speed')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # 3. Control Torque
    ax3.plot(df['time_s'], df['pd_u_applied'] * 0.055, label='PD', color='#1f77b4')
    ax3.plot(df['time_s'], df['td3_u_applied'] * 0.055, label='TD3', color='#2ca02c')
    ax3.plot(df['time_s'], df['sac_u_applied'] * 0.055, label='SAC', color='#9467bd')
    ax3.set_ylabel('Motor Torque [N·m]')
    ax3.set_xlabel('Time [s]')
    ax3.set_title('Applied Motor Torque')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # 4. Zoomed Fine Pointing
    mask = df['time_s'] >= 15.0
    sub = df[mask]
    ax4.plot(sub['time_s'], sub['pd_theta_deg'], label='PD', color='#1f77b4', lw=2)
    ax4.plot(sub['time_s'], sub['td3_theta_deg'], label='TD3', color='#2ca02c', lw=2)
    ax4.plot(sub['time_s'], sub['sac_theta_deg'], label='SAC', color='#9467bd', lw=2)
    ax4.axhline(0.2, color='red', linestyle='--')
    ax4.axhline(-0.2, color='red', linestyle='--')
    ax4.set_ylim(-0.4, 0.4)
    ax4.set_ylabel('Fine Error [deg]')
    ax4.set_xlabel('Time [s]')
    ax4.set_title('Steady-State Pointing Window (t >= 15 s)')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    out_png = "results/figures/example_scenario_1_trajectory.png"
    plt.savefig(out_png, dpi=200)
    print(f"Saved visualization plot to: {out_png}")

if __name__ == '__main__':
    main()
