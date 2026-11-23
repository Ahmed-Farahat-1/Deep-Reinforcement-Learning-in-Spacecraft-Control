# Explicit Documentation of Experimental Limitations & Boundary Assumptions

1. **Single-Axis Rotational Restriction:**
   The simulation assumes attitude rotation is strictly confined to the principal Z-axis. Multi-axis cross-coupling gyroscopic terms ($ec{\omega} \times \mathbf{J}\vec{\omega}$) are not present.
2. **Absence of External Desaturation Actuators:**
   The spacecraft is equipped solely with a reaction wheel. When the wheel reaches hard speed saturation ($5500\text{ RPM}$), internal momentum transfer is blocked. Without reaction control thrusters or magnetic torquers, desaturation is physically impossible.
3. **Idealized Sensing:**
   State observations assume zero sensor noise, zero latency, and zero bias. In real spaceflight, an extended Kalman filter (MEKF) with star tracker and gyro noise would be required.
4. **Orbital Decoupling:**
   Environmental disturbance torques were modeled parametrically rather than coupled to a live 6-DOF orbital propagation.
