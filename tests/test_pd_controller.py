"""
Unit and Validation Tests for Classical PD Spacecraft Attitude Controller.

Validates:
1. Positive attitude error produces corrective positive torque (causing negative acceleration).
2. Negative attitude error produces corrective negative torque (causing positive acceleration).
3. Closed-loop asymptotic stabilization of spacecraft attitude under zero disturbance.
4. Derivative gain Kd provides damping that eliminates/reduces oscillations.
5. Motor torque saturation limits are strictly respected.
6. Reaction wheel speed limits are strictly respected.
7. Correct steady-state behavior and disturbance rejection under external torques.
"""

import sys
from pathlib import Path
import pytest
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.simulation.spacecraft import Spacecraft1D
from src.simulation.disturbances import (
    ZeroDisturbance,
    ConstantDisturbance,
    PulseDisturbance,
)
from src.controllers.pd_controller import PDController
from src.evaluation.metrics import calculate_performance_metrics, compute_settling_time


class TestPDControlLawDirection:
    """Test Suite 1: Corrective torque direction of PD control law."""

    def test_positive_attitude_error_corrective_torque(self):
        """
        When theta > 0 and theta_dot >= 0:
        Controller must command u > 0, which accelerates the wheel positively (Omega_dot > 0)
        and exerts negative reaction torque on the body (theta_ddot = -u/J < 0),
        driving theta back towards 0.
        """
        controller = PDController(Kp=5.0, Kd=10.0, max_torque=1.0)
        sim = Spacecraft1D(J_body=10.0, I_wheel=0.05)

        theta = 0.1  # rad > 0
        theta_dot = 0.0
        u_cmd, info = controller.compute_torque(theta=theta, theta_dot=theta_dot)

        assert u_cmd > 0.0, "Commanded torque must be positive for positive attitude error"
        assert u_cmd == pytest.approx(controller.Kp * theta)

        # Apply to dynamics
        dstate, u_applied = sim.derivatives(0.0, [theta, theta_dot, 0.0], u_cmd=u_cmd, disturbance=0.0)
        theta_ddot = dstate[1]

        assert theta_ddot < 0.0, "Spacecraft angular acceleration must be negative to correct positive error"
        assert theta_ddot == pytest.approx(-u_applied / sim.J)

    def test_negative_attitude_error_corrective_torque(self):
        """
        When theta < 0 and theta_dot <= 0:
        Controller must command u < 0, exerting positive reaction torque on the body
        (theta_ddot = -u/J > 0), driving theta back towards 0.
        """
        controller = PDController(Kp=5.0, Kd=10.0, max_torque=1.0)
        sim = Spacecraft1D(J_body=10.0, I_wheel=0.05)

        theta = -0.15  # rad < 0
        theta_dot = 0.0
        u_cmd, info = controller.compute_torque(theta=theta, theta_dot=theta_dot)

        assert u_cmd < 0.0, "Commanded torque must be negative for negative attitude error"
        assert u_cmd == pytest.approx(controller.Kp * theta)

        dstate, u_applied = sim.derivatives(0.0, [theta, theta_dot, 0.0], u_cmd=u_cmd, disturbance=0.0)
        theta_ddot = dstate[1]

        assert theta_ddot > 0.0, "Spacecraft angular acceleration must be positive to correct negative error"
        assert theta_ddot == pytest.approx(-u_applied / sim.J)


class TestClosedLoopStabilization:
    """Test Suite 2: Asymptotic stabilization under zero disturbance."""

    def test_pd_stabilization_zero_disturbance(self):
        """
        Closed loop starting from theta_0 = +15 deg must settle to |theta| < 0.001 rad (~0.05 deg)
        and |theta_dot| < 0.0001 rad/s within 15 seconds.
        """
        sim = Spacecraft1D(J_body=10.0, I_wheel=0.05, dt=0.01)
        controller = PDController(Kp=5.0, Kd=10.0, max_torque=1.0)

        initial_state = [np.radians(15.0), 0.0, 0.0]
        sim.reset(initial_state=initial_state)

        steps = 2500  # 25.0 seconds
        time_hist = np.zeros(steps)
        theta_hist = np.zeros(steps)

        for i in range(steps):
            u_cmd, _ = controller.compute_torque(sim.theta, sim.theta_dot, sim.Omega)
            state, _ = sim.step(u_cmd=u_cmd, disturbance=0.0)
            time_hist[i] = sim.time
            theta_hist[i] = state[0]

        # Verify asymptotic stability
        final_theta = abs(sim.theta)
        final_theta_dot = abs(sim.theta_dot)

        assert final_theta < 1e-4, f"Final attitude error {final_theta} rad exceeds 1e-4 rad tolerance"
        assert final_theta_dot < 1e-4, f"Final angular rate {final_theta_dot} rad/s exceeds 1e-4 rad/s tolerance"

        # Verify settling time
        ts = compute_settling_time(time_hist, theta_hist, target=0.0)
        assert ts is not None, "System should settle within 25 seconds"
        assert ts < 10.0, f"Settling time {ts:.2f}s should be under 10 seconds"


class TestDampingEffect:
    """Test Suite 3: Damping effectiveness of derivative gain Kd."""

    def test_damping_reduces_oscillations(self):
        """
        Compare undamped controller (Kd = 0) vs. properly damped controller (Kd = 10.0):
        - Undamped (Kd = 0): J * theta_ddot + Kp * theta = 0 (pure harmonic oscillator, energy conserved, no decay).
        - Damped (Kd = 10.0): zeta ~ 0.707 (rapid decay, overshoot < 10%, asymptotically settles).
        """
        sim_undamped = Spacecraft1D(J_body=10.0, I_wheel=0.05, dt=0.01)
        sim_damped = Spacecraft1D(J_body=10.0, I_wheel=0.05, dt=0.01)

        ctrl_undamped = PDController(Kp=5.0, Kd=0.001, max_torque=5.0)  # minimal damping
        ctrl_damped = PDController(Kp=5.0, Kd=10.0, max_torque=5.0)

        initial_state = [np.radians(10.0), 0.0, 0.0]
        sim_undamped.reset(initial_state=initial_state)
        sim_damped.reset(initial_state=initial_state)

        steps = 2000  # 20.0 seconds
        theta_undamped = []
        theta_damped = []

        for _ in range(steps):
            u_un = ctrl_undamped(sim_undamped.theta, sim_undamped.theta_dot)
            state_un, _ = sim_undamped.step(u_cmd=u_un, disturbance=0.0)
            theta_undamped.append(state_un[0])

            u_d = ctrl_damped(sim_damped.theta, sim_damped.theta_dot)
            state_d, _ = sim_damped.step(u_cmd=u_d, disturbance=0.0)
            theta_damped.append(state_d[0])

        theta_un_arr = np.array(theta_undamped)
        theta_d_arr = np.array(theta_damped)

        # In the undamped case, oscillations persist throughout the 20 seconds
        oscillation_amplitude_end_undamped = np.max(np.abs(theta_un_arr[-200:]))
        oscillation_amplitude_end_damped = np.max(np.abs(theta_d_arr[-200:]))

        assert oscillation_amplitude_end_undamped > np.radians(5.0), "Undamped system must maintain oscillations"
        assert oscillation_amplitude_end_damped < np.radians(0.01), "Damped system must suppress oscillations"


class TestActuatorLimitsRespect:
    """Test Suite 4: Respect of motor torque and wheel speed limits."""

    def test_motor_torque_saturation_respected(self):
        """
        For very large attitude error (e.g. 90 deg), controller raw torque would be
        5.0 * 1.57 = 7.85 N*m. Controller and simulator must clamp command to max_torque = 1.0 N*m.
        """
        controller = PDController(Kp=5.0, Kd=10.0, max_torque=1.0)
        sim = Spacecraft1D(J_body=10.0, I_wheel=0.05, max_torque=1.0, dt=0.01)

        sim.reset(initial_state=[np.radians(90.0), 0.0, 0.0])

        for _ in range(100):
            u_cmd, info = controller.compute_torque(sim.theta, sim.theta_dot, sim.Omega)
            assert abs(u_cmd) <= 1.0 + 1e-12, f"Commanded torque {u_cmd} exceeded 1.0 N*m"
            assert info["is_torque_saturated"] is True

            _, step_info = sim.step(u_cmd=u_cmd, disturbance=0.0)
            assert abs(step_info["u_applied"]) <= 1.0 + 1e-12

    def test_wheel_speed_limit_respected(self):
        """
        When reaction wheel reaches max_speed, torque that would accelerate it further
        is zeroed, preventing overspeed beyond max_speed.
        """
        max_speed = 300.0  # rad/s
        controller = PDController(Kp=5.0, Kd=10.0, max_torque=1.0, max_wheel_speed=max_speed)
        sim = Spacecraft1D(J_body=10.0, I_wheel=0.05, max_speed=max_speed, dt=0.01)

        # Start with wheel already near limit and positive attitude error commanding u > 0
        sim.reset(initial_state=[np.radians(20.0), 0.0, max_speed])

        u_cmd, info = controller.compute_torque(sim.theta, sim.theta_dot, sim.Omega)
        assert u_cmd == 0.0, "Positive torque must be blocked at upper wheel speed limit"
        assert info["is_wheel_saturated"] is True

        state, step_info = sim.step(u_cmd=u_cmd, disturbance=0.0)
        assert state[2] <= max_speed + 1e-6, "Wheel speed must not exceed max_speed"


class TestDisturbanceRejection:
    """Test Suite 5: Controller behavior under external disturbances."""

    def test_constant_disturbance_theoretical_steady_state_error(self):
        """
        Under constant disturbance Md, a PD controller has a theoretical steady-state error:
        theta_ss = Md / Kp
        For Md = 0.02 N*m and Kp = 5.0 N*m/rad:
        theta_ss = 0.02 / 5.0 = 0.004 rad (~0.2292 deg).
        """
        sim = Spacecraft1D(J_body=10.0, I_wheel=0.05, dt=0.01)
        controller = PDController(Kp=5.0, Kd=10.0, max_torque=1.0)

        Md = 0.02
        dist = ConstantDisturbance(torque_value=Md)
        sim.reset(initial_state=[0.0, 0.0, 0.0])

        steps = 3000  # 30.0 seconds
        for _ in range(steps):
            u_cmd, _ = controller.compute_torque(sim.theta, sim.theta_dot, sim.Omega)
            sim.step(u_cmd=u_cmd, disturbance=dist)

        expected_theta_ss = Md / controller.Kp
        actual_theta_ss = sim.theta

        # Verify simulation matches analytical steady-state offset within tight tolerance
        assert abs(actual_theta_ss - expected_theta_ss) < 1e-4, (
            f"Expected steady-state error {expected_theta_ss:.6f} rad, got {actual_theta_ss:.6f} rad"
        )

    def test_pulse_disturbance_recovery(self):
        """
        A pulse disturbance between t=5s and t=7s perturbs attitude.
        The PD controller must recover and restore attitude to near-zero after pulse ends.
        """
        sim = Spacecraft1D(J_body=10.0, I_wheel=0.05, dt=0.01)
        controller = PDController(Kp=5.0, Kd=10.0, max_torque=1.0)
        dist = PulseDisturbance(magnitude=0.05, t_start=5.0, t_end=7.0)

        sim.reset(initial_state=[0.0, 0.0, 0.0])

        steps = 2500  # 25.0 seconds
        max_perturbation = 0.0

        for _ in range(steps):
            t = sim.time
            u_cmd, _ = controller.compute_torque(sim.theta, sim.theta_dot, sim.Omega)
            sim.step(u_cmd=u_cmd, disturbance=dist)

            if t >= 5.0:
                max_perturbation = max(max_perturbation, abs(sim.theta))

        # Perturbation should have been observed during pulse
        assert max_perturbation > np.radians(0.1), "Pulse disturbance should induce attitude deflection"

        # After pulse ends and system settles, attitude error must return to zero
        assert abs(sim.theta) < 1e-4, f"Failed to restabilize after pulse: final theta = {sim.theta:.6f} rad"
        assert abs(sim.theta_dot) < 1e-4, f"Failed to restabilize rate: final theta_dot = {sim.theta_dot:.6f} rad/s"
