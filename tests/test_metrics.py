"""
Automated Unit Tests for Audited GNC Performance Metrics (Phase 2C).

Validates:
1. Settling time calculations around zero and around non-zero steady-state offsets.
2. Error decay time (90% to 10%) calculation.
3. Overshoot calculation relative to steady-state reference.
4. Periodic steady-state metrics (mean, RMS, peak amplitude, peak-to-peak).
5. Pulse disturbance recovery metrics (initial, absolute, and recovery times).
6. Rigorous separation of motor torque saturation from wheel speed saturation.
7. Squared control torque integral calculation.
8. Constant disturbance wheel momentum accumulation and time-to-saturation projection.
"""

import sys
from pathlib import Path
import pytest
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import (
    compute_settling_time,
    compute_decay_time,
    compute_overshoot,
    compute_periodic_metrics,
    compute_pulse_recovery_metrics,
    analyze_saturation,
    analyze_constant_disturbance_momentum,
    compute_squared_torque_integral,
)


class TestSettlingTimeAudited:
    """Test Suite for scenario-aware settling time."""

    def test_settling_time_around_zero(self):
        """Exponential decay towards zero: theta(t) = theta_0 * exp(-t / tau)."""
        time = np.linspace(0, 20, 2001)
        tau = 2.0
        theta_0 = 1.0
        theta = theta_0 * np.exp(-time / tau)

        # Theoretical 2% settling time: exp(-t / 2) = 0.02 => t = -2 * ln(0.02) ~ 7.824 s
        t_settle = compute_settling_time(time, theta, target=0.0, tolerance_pct=0.02)
        assert t_settle is not None
        assert t_settle == pytest.approx(-tau * np.log(0.02), abs=0.02)

    def test_settling_time_around_nonzero_steady_state(self):
        """Trajectory approaching non-zero steady state theta_ss = 0.5."""
        time = np.linspace(0, 20, 2001)
        theta_0 = 2.0
        theta_ss = 0.5
        tau = 2.0
        theta = theta_ss + (theta_0 - theta_ss) * np.exp(-time / tau)

        # Settling band: theta_ss +/- 0.02 * |theta_0 - theta_ss|
        t_settle = compute_settling_time(
            time, theta, target=0.0, steady_state_val=theta_ss, tolerance_pct=0.02
        )
        assert t_settle is not None
        assert t_settle == pytest.approx(-tau * np.log(0.02), abs=0.02)


class TestOvershootAndDecayTime:
    """Test Suite for overshoot and decay time."""

    def test_overshoot_relative_to_steady_state(self):
        """Overshoot must be measured past the steady-state equilibrium."""
        # Starts at 10, undershoots steady state (2.0) down to 1.5, then settles at 2.0
        time = np.linspace(0, 10, 1001)
        theta = 2.0 + 8.0 * np.exp(-time) * np.cos(time)

        os_dict = compute_overshoot(theta, steady_state_val=2.0)
        assert os_dict["overshoot_pct"] > 0.0
        # Peak undershoot past 2.0:
        min_th = np.min(theta)
        expected_os_pct = ((2.0 - min_th) / (10.0 - 2.0)) * 100.0
        assert os_dict["overshoot_pct"] == pytest.approx(expected_os_pct)

    def test_decay_time_calculation(self):
        """Decay time from 90% error to 10% error."""
        time = np.linspace(0, 20, 2001)
        tau = 2.0
        theta = 1.0 * np.exp(-time / tau)

        decay = compute_decay_time(time, theta, steady_state_val=0.0)
        # exp(-t10 / tau) = 0.9 => t10 = -tau * ln(0.9)
        # exp(-t90 / tau) = 0.1 => t90 = -tau * ln(0.1)
        # decay = t90 - t10 = tau * (ln(0.9) - ln(0.1)) = tau * ln(9) ~ 4.394 s
        expected_decay = tau * np.log(9.0)
        assert decay is not None
        assert decay == pytest.approx(expected_decay, abs=0.02)


class TestPeriodicMetrics:
    """Test Suite for periodic disturbance steady-state metrics."""

    def test_periodic_sine_wave(self):
        """Pure sine wave theta(t) = A * sin(2*pi*f*t) with f = 0.1 Hz."""
        f = 0.1
        T = 1.0 / f
        time = np.linspace(0, 30, 3001)
        A = 0.2
        theta = A * np.sin(2.0 * np.pi * f * time)

        p_metrics = compute_periodic_metrics(time, theta, period=T, num_cycles=1)
        assert p_metrics["periodic_mean_rad"] == pytest.approx(0.0, abs=1e-3)
        assert p_metrics["periodic_rms_rad"] == pytest.approx(A / np.sqrt(2.0), abs=1e-3)
        assert p_metrics["periodic_peak_amp_rad"] == pytest.approx(A, abs=1e-3)
        assert p_metrics["periodic_p2p_rad"] == pytest.approx(2.0 * A, abs=1e-3)


class TestPulseRecoveryMetrics:
    """Test Suite for pulse disturbance metrics."""

    def test_pulse_recovery(self):
        """Simulate initial decay, a perturbation pulse between 10s and 12s, and recovery."""
        time = np.linspace(0, 30, 3001)
        theta = np.zeros_like(time)

        # Initial transient settled by 8s
        theta[time < 10.0] = 1.0 * np.exp(-time[time < 10.0] / 1.5)

        # Pulse disturbance between 10s and 12s
        pulse_mask = (time >= 10.0) & (time <= 12.0)
        theta[pulse_mask] = 0.3 * np.sin(np.pi * (time[pulse_mask] - 10.0) / 2.0)

        # Recovery after 12s
        post_mask = time > 12.0
        theta[post_mask] = 0.3 * np.exp(-(time[post_mask] - 12.0) / 1.0)

        metrics = compute_pulse_recovery_metrics(time, theta, pulse_end_time=12.0, target=0.0, tolerance_pct=0.02)
        assert metrics["initial_settling_time_s"] < 10.0
        assert metrics["absolute_settling_time_s"] > 12.0
        assert metrics["recovery_time_s"] > 0.0
        assert metrics["pulse_max_deflection_deg"] > 0.0


class TestSaturationAnalysisSeparation:
    """Test Suite for distinct motor torque and wheel speed saturation tracking."""

    def test_torque_saturation_tracking(self):
        """Ensure torque saturation duration and events are counted independently."""
        time = np.linspace(0, 10, 1001)
        torque = np.zeros_like(time)
        # Saturated between 1s and 2s, and between 5s and 6s
        torque[(time >= 1.0) & (time <= 2.0)] = 1.0
        torque[(time >= 5.0) & (time <= 6.0)] = -1.0

        Omega = np.zeros_like(time)  # Wheel never saturated

        sat = analyze_saturation(time, torque, max_torque=1.0, Omega=Omega, max_speed=500.0)
        assert sat["torque_sat_events"] == 2
        assert sat["torque_sat_duration_s"] == pytest.approx(2.0, abs=0.05)
        assert sat["wheel_sat_events"] == 0
        assert sat["wheel_sat_duration_s"] == 0.0


class TestSquaredTorqueIntegral:
    """Test Suite for squared control torque integral."""

    def test_constant_torque_effort(self):
        """Constant torque u = 0.5 over 10s: integral = 0.5^2 * 10 = 2.5 N^2*m^2*s."""
        time = np.linspace(0, 10, 1001)
        torque = np.full_like(time, 0.5)

        effort = compute_squared_torque_integral(time, torque)
        assert effort == pytest.approx(2.5, abs=1e-3)


class TestMomentumAccumulationProjection:
    """Test Suite for momentum accumulation rate and saturation projection."""

    def test_momentum_accumulation_constant_disturbance(self):
        """Slope dOmega/dt = Md / I = 0.02 / 0.05 = 0.4 rad/s^2."""
        time = np.linspace(0, 30, 3001)
        I_wheel = 0.05
        Md = 0.02
        max_speed = 523.6
        Omega = (Md / I_wheel) * time  # linearly increasing

        res = analyze_constant_disturbance_momentum(time, Omega, Md=Md, I_wheel=I_wheel, max_speed=max_speed)
        assert res["theoretical_dOmega_dt_rad_s2"] == pytest.approx(0.4)
        assert res["empirical_dOmega_dt_rad_s2"] == pytest.approx(0.4, abs=1e-4)
        # Saturation from t=0 should be 523.6 / 0.4 = 1309.0 s
        assert res["time_to_saturation_s"] == pytest.approx(1309.0, abs=0.1)
