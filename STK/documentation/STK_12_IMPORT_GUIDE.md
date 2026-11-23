# Detailed STK 12 Step-by-Step Import & Verification Guide

### Step 1: Scenario Creation
- Name: `Spacecraft_Attitude_Control_Verification`
- Start Time: `1 Jun 2026 12:00:00.000 UTCG`
- Stop Time: `1 Jun 2026 12:01:40.000 UTCG` (100 seconds total simulation window)

### Step 2: Spacecraft Definition
- Insert Satellite: `ADCS_Sat_TD3`
- 3D Model: Assign standard spacecraft 3D model (e.g. `satellite.mdl` or `cubesat.mdl`).

### Step 3: Importing `.a` Attitude Files
1. Open Satellite Properties -> Attitude page.
2. Select **Attitude Type: File**.
3. File Format: `STK Attitude File (*.a)`.
4. Point to: `SPACECRAFT_TRAJECTORIES/spacecraft_TD3_scenario_1_pos10deg.a`.
5. Interpolation: Linear (InterpolationOrder 1).

### Step 4: Verification of Attitude Behavior
- At $t = 0$ s: Spacecraft yaw axis shows $+10.0^\circ$ offset relative to target VNC Nadir track.
- At $t = 9.4$ s: TD3 controller drives yaw to within $\pm 0.2^\circ$.
- Steady-state: Spacecraft remains stabilized with minimal residual drift.
