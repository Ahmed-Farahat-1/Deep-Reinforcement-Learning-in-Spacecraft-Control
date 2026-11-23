# Explicit Documentation of STK Physical Scope & Missing Inputs

## 1. Single-Axis vs 3-Axis Attitude Dynamics
- **Modeled in Research:** 1D rotational dynamics about principal inertia axis $Z$ ($J = 10.05$ kg$\cdot$m$^2$).
- **Not Modeled / Assumptions:**
  - Roll and pitch angular rates are assumed uncoupled and strictly zero ($\dot{\phi} = \dot{\theta}_{pitch} = 0$).
  - Actuator gyroscopic cross-coupling ($ec{\omega} \times \mathbf{I}\vec{\omega} + \vec{\omega} \times \vec{h}_{rw}$) is absent because roll and pitch rates are identically zero.
  - Multi-axis reaction wheel assemblies (e.g. tetrahedral 4-wheel pyramids) are not modeled.

## 2. Orbital Propagation Coupling
- The Python simulation does **not** include an active orbital propagator.
- External gravity gradient torques, aerodynamic drag torques, and magnetic dipole moments were modeled as analytical disturbance profiles ($M_d(t)$ constant, sinusoidal, and pulse), rather than evaluated along an orbital ephemeris.
- The 500 km LEO circular orbit provided is an **illustrative baseline template** for STK visualization.
