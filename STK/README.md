# Ansys Systems Tool Kit (STK 12) Mission Visualization & Integration Guide

## 1. Why STK is Included in this Research
Ansys Systems Tool Kit (STK 12) is the global aerospace industry standard for orbital mechanics, space mission analysis, and satellite operations visualization. While our spacecraft attitude stabilization algorithms were modeled, trained, and numerically integrated in Python using high-fidelity 4th-order Runge-Kutta (RK4) dynamics, visualizing the physical satellite reorientation along a realistic Earth orbit provides critical intuitive validation for space operations engineers and academic evaluation committees.

## 2. What the Orbital Visualization Represents
The visualization depicts a satellite operating in a Low Earth Orbit (LEO) undergoing attitude regulation about its principal Z-axis (yaw).
- **Controlled Degree of Freedom:** Body yaw angle $\psi(t) = \theta(t)$, driven by the internal reaction wheel exchange torque $u(t)$.
- **Transverse Axes:** Roll ($\phi$) and pitch ($\theta_{pitch}$) are maintained nominally at $0.0^\circ$ relative to the orbit reference frame.
- **Reference Target:** Nadir-pointing track aligned with the orbital Velocity-Normal-Co-normal (VNC) reference frame.

## 3. Clear Distinction: Python Simulation vs. STK Visualization
| Characteristic | Python Numerical Simulation | STK 12 Mission Visualization |
| :--- | :--- | :--- |
| **Domain** | 1D rigid-body rotational dynamics ($J = 10.05\text{ kg}\cdot\text{m}^2$). | 3D orbital propagation & 3D body animation. |
| **Actuator** | Internal reaction wheel ($I_w = 0.05\text{ kg}\cdot\text{m}^2$, $\pm 0.055\text{ Nm}$, $\pm 5500\text{ RPM}$). | Fixed visual CAD satellite model (`satellite.mdl`). |
| **Integration** | RK4 fixed-step ($\Delta t = 0.01\text{ s}$, $3000$ steps over $30\text{ s}$). | Subsampled ASCII ephemeris import ($10\text{ Hz}$, $300$ points over $29.9\text{ s}$). |
| **Orbit** | Independent of orbital motion (pure attitude state $[\theta, \dot{\theta}, \Omega]$). | Keplerian Two-Body orbit template ($h = 500\text{ km}$, $i = 45^\circ$). |
| **Coupling** | *Uncoupled:* External perturbations were analytical functions ($M_d(t)$). | STK orbital perturbations (J2, drag, solar radiation) do not feed into attitude. |

> **Scientific Transparency Note:** STK does *not* execute the trained reinforcement learning neural networks or PD control laws in real time. Rather, STK imports the validated time-history attitude ephemeris files (`.a`) exported directly from our verified Python simulation.

---

## 4. Directory Structure
```text
STK/
├── README.md                      <- This comprehensive guide
├── scenarios/                     <- Complete STK 12.1.0 scenario files (AI_SPACECRAFT_CONTROL.sc, sat_*.sa)
├── attitude_ephemeris/            <- 20 ready-to-import standard STK .a attitude ephemeris files (10 Hz)
├── orbital_data/                  <- Subsampled CSV attitude time histories and orbital parameters
├── scripts/                       <- export_stk_attitude.py conversion utility
└── documentation/                 <- Coordinate frame mappings, import guides, and boundary disclosures
```

---

## 5. Step-by-Step STK 12 Import & Reproducibility Workflow

### Step 1: Open the Provided Scenario (or Create New)
You can open the pre-built scenario file directly:
`STK/scenarios/AI_SPACECRAFT_CONTROL.sc` in STK 12.
Alternatively, create a new scenario manually:
- **Scenario Name:** `Spacecraft_ADCS_Comparison`
- **Scenario Epoch:** `1 Jun 2026 12:00:00.000 UTCG`
- **Start Time:** `1 Jun 2026 12:00:00.000 UTCG`
- **Stop Time:** `1 Jun 2026 12:00:30.000 UTCG` (30 seconds, matching the simulation duration)

### Step 2: Insert Spacecraft Objects
Insert 4 Satellite objects to compare all four controllers simultaneously:
1. `Sat_PD`
2. `Sat_TD3`
3. `Sat_SAC`
4. `Sat_PPO`

### Step 3: Configure Orbital Propagation
For each satellite, open **Properties -> Orbit**:
- **Propagator:** `TwoBody`
- **Coordinate System:** `ICRF` (or `J2000`)
- **Semi-Major Axis ($a$):** `6878.14 km` ($500.0\text{ km}$ altitude circular orbit)
- **Eccentricity ($e$):** `0.0001` (Nominally circular)
- **Inclination ($i$):** `45.0 deg`
- **RAAN ($\Omega$):** `0.0 deg`
- **Argument of Perigee ($\omega$):** `0.0 deg`
- **True Anomaly ($\nu$):**
  - `Sat_PD`: $0.000^\circ$
  - `Sat_TD3`: $0.005^\circ$ ($pprox 600\text{ m}$ along-track offset)
  - `Sat_SAC`: $0.010^\circ$ ($pprox 1200\text{ m}$ along-track offset)
  - `Sat_PPO`: $0.015^\circ$ ($pprox 1800\text{ m}$ along-track offset)

### Step 4: Import External Attitude Ephemeris (`.a` Files)
For each satellite, navigate to **Properties -> Basic -> Attitude**:
1. Change **Attitude Type** from `Standard` to **`External / File`**.
2. Select **File Format:** `STK Attitude File (*.a)`.
3. Set **Coordinate Frame:** `VNC` (Velocity - Normal - Co-normal).
4. Set **Interpolation Order:** `Linear` (or `1`).
5. Select the corresponding file from `STK/attitude_ephemeris/`:
   - `Sat_PD` $ightarrow$ `spacecraft_PD_scenario_1_pos10deg.a`
   - `Sat_TD3` $ightarrow$ `spacecraft_TD3_scenario_1_pos10deg.a`
   - `Sat_SAC` $ightarrow$ `spacecraft_SAC_scenario_1_pos10deg.a`
   - `Sat_PPO` $ightarrow$ `spacecraft_PPO_scenario_1_pos10deg.a`
6. Click **Apply** and **OK**.

### Step 5: Visualizing Attitude Dynamics in the 3D Graphics Window
1. Enable **Body Coordinate Axes** and **VNC Reference Axes** vectors in Satellite Properties -> 3D Graphics -> Vector.
2. Set animation playback step to $0.1\text{ s}$.
3. Click **Play** and observe:
   - At $t = 0.0\text{ s}$: All four satellites display a $+10.0^\circ$ yaw offset relative to the Nadir tracking vector.
   - At $t = 9.4\text{ s}$: `Sat_TD3` rapidly aligns with the target attitude envelope ($\pm 0.2^\circ$).
   - At $t = 11.2\text{ s}$: `Sat_SAC` reaches steady alignment smoothly without overshoot.
   - At $t = 12.8\text{ s}$: `Sat_PD` settles to machine-precision pointing with zero steady-state error.
   - Throughout $t \in [15, 30]\text{ s}$: `Sat_PPO` fails to eliminate steady-state error ($> 6^\circ$).

---

## 6. Standalone Attitude Export Utility
To export any new simulation trajectory to STK `.a` format:
```bash
python STK/scripts/export_stk_attitude.py --input data/trajectories/trajectory_scenario_1_pos10deg.csv --output STK/attitude_ephemeris/my_custom_run.a --controller td3
```
