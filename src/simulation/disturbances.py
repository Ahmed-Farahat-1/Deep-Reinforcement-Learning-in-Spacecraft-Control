"""
Environmental Disturbance Torque Models for Spacecraft Attitude Simulation.

Provides configurable external disturbance torque models including:
- ZeroDisturbance: Nominal torque-free space environment.
- ConstantDisturbance: Aerodynamic drag / solar radiation pressure constant bias.
- SinusoidalDisturbance: Periodic disturbances (e.g., gravity gradient, magnetic dipole oscillation).
- PulseDisturbance: Transient external torques (e.g., micrometeoroid impact, thruster firing).
- CompositeDisturbance: Superposition of multiple disturbance sources.

All torques are expressed in Newton-meters (N*m) and time in seconds (s).
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import numpy as np


class BaseDisturbance(ABC):
    """Abstract base class for all spacecraft disturbance models."""

    @abstractmethod
    def torque(self, t: float) -> float:
        """
        Compute the external disturbance torque at time t.

        Parameters
        ----------
        t : float
            Simulation time in seconds (s).

        Returns
        -------
        float
            Disturbance torque in Newton-meters (N*m).
        """
        pass

    def __call__(self, t: float) -> float:
        """Allow disturbance instances to be called directly as functions."""
        return self.torque(t)


class ZeroDisturbance(BaseDisturbance):
    """Zero disturbance model (ideal torque-free environment)."""

    def torque(self, t: float) -> float:
        return 0.0

    def __repr__(self) -> str:
        return "ZeroDisturbance()"


class ConstantDisturbance(BaseDisturbance):
    """
    Constant external disturbance torque.

    Parameters
    ----------
    torque_value : float
        Constant disturbance torque magnitude in N*m.
    """

    def __init__(self, torque_value: float = 0.0):
        self.torque_value = float(torque_value)

    def torque(self, t: float) -> float:
        return self.torque_value

    def __repr__(self) -> str:
        return f"ConstantDisturbance(torque={self.torque_value:.4e} N*m)"


class SinusoidalDisturbance(BaseDisturbance):
    """
    Sinusoidal external disturbance torque:
    M_d(t) = amplitude * sin(2 * pi * frequency * t + phase) + bias

    Parameters
    ----------
    amplitude : float
        Disturbance amplitude in N*m.
    frequency : float
        Frequency in Hertz (Hz).
    phase : float, optional
        Phase angle in radians (default: 0.0).
    bias : float, optional
        Constant offset/bias in N*m (default: 0.0).
    """

    def __init__(
        self,
        amplitude: float = 0.01,
        frequency: float = 0.1,
        phase: float = 0.0,
        bias: float = 0.0,
    ):
        self.amplitude = float(amplitude)
        self.frequency = float(frequency)
        self.phase = float(phase)
        self.bias = float(bias)
        self.omega = 2.0 * np.pi * self.frequency

    def torque(self, t: float) -> float:
        return self.amplitude * np.sin(self.omega * t + self.phase) + self.bias

    def __repr__(self) -> str:
        return (
            f"SinusoidalDisturbance(A={self.amplitude:.4e} N*m, "
            f"f={self.frequency:.4e} Hz, phi={self.phase:.2f} rad, bias={self.bias:.4e} N*m)"
        )


class PulseDisturbance(BaseDisturbance):
    """
    Pulse (step-like transient) external disturbance torque:
    M_d(t) = magnitude for t_start <= t <= t_end, else 0.0

    Parameters
    ----------
    magnitude : float
        Pulse torque magnitude in N*m.
    t_start : float
        Start time of the pulse in seconds.
    t_end : float
        End time of the pulse in seconds.
    """

    def __init__(self, magnitude: float = 0.05, t_start: float = 10.0, t_end: float = 12.0):
        if t_end < t_start:
            raise ValueError(f"t_end ({t_end}) must be greater than or equal to t_start ({t_start}).")
        self.magnitude = float(magnitude)
        self.t_start = float(t_start)
        self.t_end = float(t_end)

    def torque(self, t: float) -> float:
        if self.t_start <= t <= self.t_end:
            return self.magnitude
        return 0.0

    def __repr__(self) -> str:
        return (
            f"PulseDisturbance(magnitude={self.magnitude:.4e} N*m, "
            f"t_start={self.t_start:.2f} s, t_end={self.t_end:.2f} s)"
        )


class CompositeDisturbance(BaseDisturbance):
    """
    Superposition of multiple disturbance models:
    M_d(t) = sum(d_i.torque(t)) for d_i in disturbances

    Parameters
    ----------
    disturbances : List[BaseDisturbance]
        List of disturbance instances to combine linearly.
    """

    def __init__(self, disturbances: Optional[List[BaseDisturbance]] = None):
        self.disturbances: List[BaseDisturbance] = list(disturbances) if disturbances else []

    def add(self, disturbance: BaseDisturbance) -> None:
        """Add a disturbance component to the composite model."""
        self.disturbances.append(disturbance)

    def torque(self, t: float) -> float:
        return sum(d.torque(t) for d in self.disturbances)

    def __repr__(self) -> str:
        return f"CompositeDisturbance({len(self.disturbances)} components)"


def load_disturbance_from_dict(config: Dict[str, Any]) -> BaseDisturbance:
    """
    Factory function to instantiate a disturbance model from a configuration dictionary.

    Parameters
    ----------
    config : Dict[str, Any]
        Dictionary matching the `disturbances` section of `physics_config.yaml`.

    Returns
    -------
    BaseDisturbance
        Instantiated disturbance model.
    """
    if not config.get("enabled", False):
        return ZeroDisturbance()

    components: List[BaseDisturbance] = []

    # Constant torque component
    const_val = float(config.get("constant_torque", 0.0))
    if abs(const_val) > 1e-12:
        components.append(ConstantDisturbance(torque_value=const_val))

    # Sinusoidal component
    sin_amp = float(config.get("sinusoidal_amplitude", 0.0))
    if abs(sin_amp) > 1e-12:
        sin_freq = float(config.get("sinusoidal_frequency", 0.1))
        components.append(SinusoidalDisturbance(amplitude=sin_amp, frequency=sin_freq))

    # Pulse component
    pulse_mag = float(config.get("pulse_magnitude", 0.0))
    if abs(pulse_mag) > 1e-12:
        t_range = config.get("pulse_time_range", [10.0, 12.0])
        components.append(
            PulseDisturbance(magnitude=pulse_mag, t_start=t_range[0], t_end=t_range[1])
        )

    if not components:
        return ZeroDisturbance()
    elif len(components) == 1:
        return components[0]
    else:
        return CompositeDisturbance(components)
