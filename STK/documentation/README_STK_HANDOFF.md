# STK 12 Visualization & Handoff Guide

## 1. Executive Summary
This directory contains all data, scripts, and configuration files necessary to import the spacecraft attitude stabilization trajectories into **Ansys Systems Tool Kit (STK 12)**.

The project models a **single-axis rigid-body spacecraft with a reaction wheel** undergoing attitude regulation about the principal Z-axis.

## 2. Directory Structure
- `ATTITUDE_TIME_HISTORIES/`: Subsampled (10 Hz) CSV attitude files containing `time_s`, `roll_deg` (0°), `pitch_deg` (0°), `yaw_deg` ($\theta(t)$), and `yaw_rate_deg_s`.
- `SPACECRAFT_TRAJECTORIES/`: Ready-to-import STK standard `.a` attitude ephemeris files (Sequence 3-2-1, VNC reference frame).
- `STK_EXPORT_SCRIPTS/`: Standalone Python script `export_stk_attitude.py` for exporting any new trajectory into STK format.
- `STK_SCENARIO_TEMPLATE/`: JSON configuration template for STK scenario setup.
- `ORBITAL_ELEMENTS/`: Illustrative orbital baseline parameters and documentation of physical assumptions.

## 3. Quick Start in STK 12
1. Launch STK 12 and create a new Scenario:
   - **Scenario Epoch:** `1 Jun 2026 12:00:00.000 UTCG`
   - **Scenario Duration:** `100 seconds` (or full orbit for visualization).
2. Insert a Satellite object (e.g. `Spacecraft_ADCS`).
3. Set the Satellite Orbit:
   - Propagator: **TwoBody** (or J2).
   - Semi-major Axis: **6878.14 km** (500 km altitude circular orbit).
   - Inclination: **45.0 deg**.
4. Configure Satellite Attitude:
   - Right-click Satellite -> **Properties** -> **Attitude**.
   - Basic Attitude Type: Change from *Standard* to **External / File**.
   - Browse and select one of the provided files in `SPACECRAFT_TRAJECTORIES/`, e.g.:
     `spacecraft_TD3_scenario_1_pos10deg.a`
   - Coordinate Frame: Select **VNC** (Velocity - Normal - Co-normal).
5. Click **Apply** and **OK**.
6. Open a 3D Graphics window and run Animation to visualize real-time attitude stabilization.
