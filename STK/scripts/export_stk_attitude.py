"""
Standalone Utility: Export Simulation Trajectories to AGI STK 12 Attitude File (.a)
Usage:
    python export_stk_attitude.py --input <trajectory_csv> --output <attitude_file.a> --controller <td3|sac|pd|ppo>
"""

import argparse
import pandas as pd
import numpy as np

def convert_csv_to_stk_a(csv_file, a_file, ctrl='td3', epoch="1 Jun 2026 12:00:00.000000000", sample_hz=10.0):
    df = pd.read_csv(csv_file)
    t = df['time_s'].values
    theta_col = f"{ctrl.lower()}_theta_deg"
    if theta_col not in df.columns:
        raise ValueError(f"Column '{theta_col}' not found in CSV. Available: {list(df.columns)}")
    yaw = df[theta_col].values
    
    # Downsample if needed
    step_size = max(1, int(1.0 / (sample_hz * (t[1] - t[0]))))
    t_sub = t[::step_size]
    yaw_sub = yaw[::step_size]
    
    with open(a_file, 'w') as f:
        f.write("stk.v.12.0\n")
        f.write("BEGIN Attitude\n")
        f.write(f"ScenarioEpoch           {epoch}\n")
        f.write(f"NumberOfAttitudePoints  {len(t_sub)}\n")
        f.write("Blocking                1\n")
        f.write("InterpolationOrder      1\n")
        f.write("CentralBody             Earth\n")
        f.write("CoordinateAxes          VNC\n")
        f.write("Sequence                321\n")
        f.write("AttitudeTimeEulerAngles\n")
        for ti, yi in zip(t_sub, yaw_sub):
            f.write(f"  {ti:10.4f}  {yi:12.6f}  {0.0:10.4f}  {0.0:10.4f}\n")
        f.write("END Attitude\n")
    print(f"Exported {len(t_sub)} attitude points to {a_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Export simulation CSV to STK .a")
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output .a file path")
    parser.add_argument("--controller", default="td3", help="Controller name (pd, ppo, td3, sac)")
    args = parser.parse_args()
    convert_csv_to_stk_a(args.input, args.output, args.controller)
