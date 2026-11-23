# Coordinate Frame Mapping & Kinematic Transformations

## 1. Frame Hierarchy
In STK 12, attitude is defined by rotating the Spacecraft Body Frame with respect to an Orbital Reference Frame.

### A. Spacecraft Body Frame (B)
- Origin: Spacecraft Center of Mass.
- $Z_B$: Reaction wheel rotor spin axis. This is the **single active control axis** in our study.
- $X_B$: Transverse axis 1 (roll axis, nominally locked to $0^\circ$).
- $Y_B$: Transverse axis 2 (pitch axis, nominally locked to $0^\circ$).

### B. Orbital Reference Frame: VNC (Velocity - Normal - Co-normal)
- $X_{VNC}$: Aligned with orbital velocity vector $\vec{v}$.
- $Y_{VNC}$: Orbit normal vector $\vec{h} = \vec{r} \times \vec{v}$.
- $Z_{VNC}$: Completes right-handed system (Nadir pointing).

## 2. Euler Angle Mapping (Sequence 3-2-1)
Rotation matrix from VNC to Body:
$R_{B/VNC} = R_1(\phi) R_2(\theta_{pitch}) R_3(\psi)$

In our single-axis study:
- $\phi = 0.0^\circ$ (Roll locked)
- $\theta_{pitch} = 0.0^\circ$ (Pitch locked)
- $\psi = \theta(t)$ (Yaw angle from Python RK4 simulation)
