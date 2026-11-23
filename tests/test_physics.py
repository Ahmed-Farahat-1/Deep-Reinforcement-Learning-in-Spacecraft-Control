"""
Unit and Validation Tests for Spacecraft Attitude Physics Dynamics.

Validates:
1. Angular momentum conservation under zero external disturbance (M_d = 0).
2. Angular momentum balance under constant and time-varying external disturbances (delta H = int M_d dt).
3. Action-reaction direction between the reaction wheel and spacecraft body.
4. Wheel speed limits (saturation prohibits overspeed but permits despin torque).
5. Motor torque limits (clamping of commanded torque to +/- u_max).
6. RK4 integration accuracy against closed-form analytical solutions.
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
    SinusoidalDisturbance,
    PulseDisturbance,
)


class TestAngularMomentumConservation:
    """Test Suite 1: Angular momentum conservation when external torque is zero."""

    def test_momentum_conservation_free_drift(self):
        """With M_d = 0 and u = 0, H(t) must remain strictly constant."""
        sim = Spacecraft1D(J_body=10.0, I_wheel=0.05, dt=0.01)
        sim.reset(initial_state=[0.5, 0.2, 50.0])

        H_initial = sim.total_angular_momentum()
        assert abs(H_initial) > 0.0, "Initial momentum should be non-zero for meaningful test"

        for _ in range(1000):  # 10.0 seconds simulation
            _, info = sim.step(u_cmd=0.0, disturbance=0.0)
            H_current = info["total_angular_momentum"]
            # Under free drift, numerical momentum drift must be negligible
            assert abs(H_current - H_initial) < 1e-12

    def test_momentum_conservation_with_active_internal_control(self):
        """
        With M_d = 0 and active internal torque u != 0, momentum is exchanged
        between body and wheel, but total system H(t) must remain conserved.
        """
        sim = Spacecraft1D(J_body=10.0, I_wheel=0.05, dt=0.01)
        sim.reset(initial_state=[0.1, 0.0, 0.0])

        H_initial = sim.total_angular_momentum()

        for step_idx in range(500):  # 5.0 seconds
            t = step_idx * 0.01
            # Apply time-varying internal torque
            u = 0.5 * np.sin(2.0 * t)
            _, info = sim.step(u_cmd=u, disturbance=0.0)

            H_current = info["total_angular_momentum"]
            # Total angular momentum must remain conserved despite internal torque
            assert abs(H_current - H_initial) < 1e-10


class TestAngularMomentumDisturbanceBalance:
    """Test Suite 2: Angular momentum balance under external disturbance torque."""

    def test_momentum_change_constant_disturbance(self):
        """
        Under constant external disturbance M_d, delta H must equal M_d * T:
        H(T) - H(0) = M_d * T
        """
        sim = Spacecraft1D(J_body=10.0, I_wheel=0.05, dt=0.01)
        sim.reset(initial_state=[0.0, 0.0, 0.0])

        H_0 = sim.total_angular_momentum()
        Md_const = 0.05  # N*m
        dist = ConstantDisturbance(torque_value=Md_const)

        steps = 500  # T = 5.0 s
        T = steps * 0.01

        for _ in range(steps):
            # Also apply arbitrary internal control torque to verify independence
            u_internal = 0.2
            sim.step(u_cmd=u_internal, disturbance=dist)

        H_final = sim.total_angular_momentum()
        delta_H_actual = H_final - H_0
        delta_H_expected = Md_const * T

        assert abs(delta_H_actual - delta_H_expected) < 1e-10

    def test_momentum_change_sinusoidal_disturbance(self):
        """
        Under sinusoidal disturbance M_d(t) = A * sin(omega * t):
        delta H = int_0^T M_d(t) dt = (A / omega) * (1 - cos(omega * T))
        """
        sim = Spacecraft1D(J_body=10.0, I_wheel=0.05, dt=0.005)
        sim.reset(initial_state=[0.0, 0.0, 0.0])

        A = 0.1  # N*m
        f = 0.2  # Hz
        omega = 2.0 * np.pi * f
        dist = SinusoidalDisturbance(amplitude=A, frequency=f)

        H_0 = sim.total_angular_momentum()
        steps = 1000  # T = 5.0 s
        T = steps * 0.005

        for _ in range(steps):
            sim.step(u_cmd=0.0, disturbance=dist)

        H_final = sim.total_angular_momentum()
        delta_H_actual = H_final - H_0
        delta_H_expected = (A / omega) * (1.0 - np.cos(omega * T))

        # High precision check for RK4 integration of sinusoidal disturbance
        assert abs(delta_H_actual - delta_H_expected) < 1e-6


class TestActionReactionDirection:
    """Test Suite 3: Correct action-reaction direction between wheel and spacecraft."""

    def test_positive_torque_reaction(self):
        """
        u > 0 applies positive torque on the wheel:
        - Wheel angular acceleration: Omega_dot = u / I > 0
        - Spacecraft angular acceleration: theta_ddot = -u / J < 0
        """
        sim = Spacecraft1D(J_body=10.0, I_wheel=0.05, dt=0.01)
        sim.reset(initial_state=[0.0, 0.0, 0.0])

        u = 0.5  # N*m > 0
        dstate, u_applied = sim.derivatives(0.0, sim.state, u_cmd=u, disturbance=0.0)

        theta_dot_deriv = dstate[0]
        theta_ddot = dstate[1]
        Omega_dot = dstate[2]

        assert u_applied == pytest.approx(0.5)
        assert theta_dot_deriv == pytest.approx(0.0)
        assert Omega_dot > 0.0, "Wheel must accelerate positively when u > 0"
        assert theta_ddot < 0.0, "Spacecraft must accelerate negatively (reaction) when u > 0"
        assert Omega_dot == pytest.approx(u / sim.I)
        assert theta_ddot == pytest.approx(-u / sim.J)

    def test_negative_torque_reaction(self):
        """
        u < 0 applies negative torque on the wheel:
        - Wheel angular acceleration: Omega_dot = u / I < 0
        - Spacecraft angular acceleration: theta_ddot = -u / J > 0
        """
        sim = Spacecraft1D(J_body=10.0, I_wheel=0.05, dt=0.01)
        sim.reset(initial_state=[0.0, 0.0, 0.0])

        u = -0.8  # N*m < 0
        dstate, u_applied = sim.derivatives(0.0, sim.state, u_cmd=u, disturbance=0.0)

        theta_ddot = dstate[1]
        Omega_dot = dstate[2]

        assert u_applied == pytest.approx(-0.8)
        assert Omega_dot < 0.0, "Wheel must accelerate negatively when u < 0"
        assert theta_ddot > 0.0, "Spacecraft must accelerate positively when u < 0"
        assert Omega_dot == pytest.approx(u / sim.I)
        assert theta_ddot == pytest.approx(-u / sim.J)


class TestWheelSpeedLimits:
    """Test Suite 4: Separate wheel speed limits and despin capability."""

    def test_upper_speed_limit_blocks_accelerating_torque(self):
        """At Omega >= Omega_max, positive torque u > 0 must be blocked (u_applied = 0)."""
        max_speed = 500.0  # rad/s
        sim = Spacecraft1D(J_body=10.0, I_wheel=0.05, max_speed=max_speed, dt=0.01)
        sim.reset(initial_state=[0.0, 0.0, max_speed])

        u_eff, is_torque_sat, is_wheel_sat = sim.compute_effective_torque(u_cmd=0.5, Omega=max_speed)
        assert u_eff == 0.0, "Accelerating torque at positive speed limit must be zeroed"
        assert is_wheel_sat is True

        dstate, u_applied = sim.derivatives(0.0, sim.state, u_cmd=0.5, disturbance=0.0)
        assert u_applied == 0.0
        assert dstate[1] == 0.0, "No reaction torque on spacecraft when wheel torque is blocked"
        assert dstate[2] == 0.0, "Wheel cannot accelerate beyond max speed"

    def test_upper_speed_limit_allows_despinning_torque(self):
        """At Omega >= Omega_max, negative torque u < 0 MUST be permitted to despin wheel."""
        max_speed = 500.0
        sim = Spacecraft1D(J_body=10.0, I_wheel=0.05, max_speed=max_speed, dt=0.01)
        sim.reset(initial_state=[0.0, 0.0, max_speed])

        u_eff, is_torque_sat, is_wheel_sat = sim.compute_effective_torque(u_cmd=-0.5, Omega=max_speed)
        assert u_eff == pytest.approx(-0.5), "Braking torque must be allowed at upper speed limit"
        assert is_wheel_sat is False

        next_state, info = sim.step(u_cmd=-0.5, disturbance=0.0)
        assert next_state[2] < max_speed, "Wheel speed must decrease after applying braking torque"
        assert info["u_applied"] == pytest.approx(-0.5)

    def test_lower_speed_limit_blocks_and_allows(self):
        """At Omega <= -Omega_max, u < 0 is blocked, but u > 0 is allowed to despin."""
        max_speed = 500.0
        sim = Spacecraft1D(J_body=10.0, I_wheel=0.05, max_speed=max_speed, dt=0.01)
        sim.reset(initial_state=[0.0, 0.0, -max_speed])

        # Block negative accelerating torque
        u_eff_neg, _, is_sat_neg = sim.compute_effective_torque(u_cmd=-0.5, Omega=-max_speed)
        assert u_eff_neg == 0.0
        assert is_sat_neg is True

        # Allow positive braking torque
        u_eff_pos, _, is_sat_pos = sim.compute_effective_torque(u_cmd=0.5, Omega=-max_speed)
        assert u_eff_pos == pytest.approx(0.5)
        assert is_sat_pos is False


class TestMotorTorqueLimits:
    """Test Suite 5: Motor torque limits."""

    def test_motor_torque_saturation(self):
        """Commanded torque exceeding max_torque must be clipped to +/- max_torque."""
        max_torque = 1.0  # N*m
        sim = Spacecraft1D(J_body=10.0, I_wheel=0.05, max_torque=max_torque, dt=0.01)
        sim.reset(initial_state=[0.0, 0.0, 0.0])

        # Test positive saturation
        u_eff_pos, is_sat_pos, _ = sim.compute_effective_torque(u_cmd=5.0, Omega=0.0)
        assert u_eff_pos == pytest.approx(1.0)
        assert is_sat_pos is True

        # Test negative saturation
        u_eff_neg, is_sat_neg, _ = sim.compute_effective_torque(u_cmd=-5.0, Omega=0.0)
        assert u_eff_neg == pytest.approx(-1.0)
        assert is_sat_neg is True

        # Test within limits (no saturation)
        u_eff_ok, is_sat_ok, _ = sim.compute_effective_torque(u_cmd=0.7, Omega=0.0)
        assert u_eff_ok == pytest.approx(0.7)
        assert is_sat_ok is False


class TestRK4NumericalIntegrationAccuracy:
    """Test Suite 6: RK4 numerical integration accuracy against analytical solutions."""

    def test_rk4_accuracy_constant_torque(self):
        """
        Under constant motor torque u_0 and M_d = 0:
        theta_ddot = -u_0 / J = const
        theta_dot(t) = theta_dot_0 - (u_0 / J) * t
        theta(t) = theta_0 + theta_dot_0 * t - 0.5 * (u_0 / J) * t^2
        Omega(t) = Omega_0 + (u_0 / I) * t

        RK4 is theoretically exact for polynomials of degree <= 3.
        Numerical solution should match analytical to floating-point precision.
        """
        sim = Spacecraft1D(J_body=10.0, I_wheel=0.05, dt=0.01)
        theta_0 = 0.5
        theta_dot_0 = -0.1
        Omega_0 = 20.0
        sim.reset(initial_state=[theta_0, theta_dot_0, Omega_0])

        u_0 = 0.4  # N*m within max_torque = 1.0
        J = sim.J
        I = sim.I

        steps = 500  # 5.0 seconds
        for step in range(1, steps + 1):
            t = step * 0.01
            state, _ = sim.step(u_cmd=u_0, disturbance=0.0)

            # Analytical formulas
            expected_theta_dot = theta_dot_0 - (u_0 / J) * t
            expected_theta = theta_0 + theta_dot_0 * t - 0.5 * (u_0 / J) * (t**2)
            expected_Omega = Omega_0 + (u_0 / I) * t

            assert abs(state[0] - expected_theta) < 1e-11, f"Theta mismatch at t={t}"
            assert abs(state[1] - expected_theta_dot) < 1e-11, f"Theta_dot mismatch at t={t}"
            assert abs(state[2] - expected_Omega) < 1e-11, f"Omega mismatch at t={t}"

    def test_rk4_accuracy_sinusoidal_disturbance(self):
        """
        Under sinusoidal disturbance M_d(t) = A * sin(omega * t) with u = 0:
        theta_ddot(t) = (A / J) * sin(omega * t)
        theta_dot(t) = (A / (J * omega)) * (1 - cos(omega * t))
        theta(t) = (A / (J * omega)) * t - (A / (J * omega^2)) * sin(omega * t)
        """
        dt = 0.005
        sim = Spacecraft1D(J_body=10.0, I_wheel=0.05, dt=dt)
        sim.reset(initial_state=[0.0, 0.0, 0.0])

        A = 0.05  # N*m
        f = 0.5   # Hz
        omega = 2.0 * np.pi * f
        dist = SinusoidalDisturbance(amplitude=A, frequency=f)
        J = sim.J

        steps = 1000  # 5.0 seconds
        for step in range(1, steps + 1):
            t = step * dt
            state, _ = sim.step(u_cmd=0.0, disturbance=dist)

            expected_theta_dot = (A / (J * omega)) * (1.0 - np.cos(omega * t))
            expected_theta = (A / (J * omega)) * t - (A / (J * (omega**2))) * np.sin(omega * t)

            # High-order RK4 accuracy check
            assert abs(state[1] - expected_theta_dot) < 1e-7, f"Theta_dot mismatch at t={t}"
            assert abs(state[0] - expected_theta) < 1e-6, f"Theta mismatch at t={t}"
