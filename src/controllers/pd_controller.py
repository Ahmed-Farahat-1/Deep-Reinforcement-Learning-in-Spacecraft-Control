"""
Classical Proportional-Derivative (PD) Spacecraft Attitude Controller.

Controls the single-axis attitude angle theta towards a desired target theta_target
using reaction wheel torque commands u_cmd:

    u_cmd = Kp * (theta - theta_target) + Kd * (theta_dot - theta_dot_target)

Under the spacecraft dynamics:
    J * theta_ddot = M_d - u_applied

For positive gains (Kp > 0, Kd > 0) and zero disturbance (M_d = 0):
    J * theta_ddot + Kd * theta_dot + Kp * theta = 0

This yields a standard second-order linear differential equation:
    theta_ddot + 2 * zeta * omega_n * theta_dot + omega_n^2 * theta = 0
where:
    omega_n = sqrt(Kp / J)           # Natural frequency (rad/s)
    zeta = Kd / (2 * sqrt(J * Kp))   # Damping ratio (dimensionless)

All parameters use strict SI units (rad, rad/s, N*m, N*m/rad, N*m*s/rad).
"""

from typing import Optional, Dict, Any, Tuple, Union
from pathlib import Path
import yaml
import numpy as np


class PDController:
    """
    Classical PD attitude stabilization controller with actuator saturation handling.

    Parameters
    ----------
    Kp : float, optional
        Proportional gain in N*m/rad, default: 5.0.
    Kd : float, optional
        Derivative gain in N*m*s/rad, default: 10.0.
    max_torque : float, optional
        Maximum motor control torque magnitude |u_max| in N*m, default: 1.0.
    max_wheel_speed : float, optional
        Maximum allowable reaction wheel speed |Omega_max| in rad/s, default: 523.6 (~5000 RPM).
    target_theta : float, optional
        Desired target attitude angle in radians, default: 0.0.
    target_theta_dot : float, optional
        Desired target angular velocity in radians per second, default: 0.0.
    """

    def __init__(
        self,
        Kp: float = 5.0,
        Kd: float = 10.0,
        max_torque: float = 1.0,
        max_wheel_speed: float = 523.6,
        target_theta: float = 0.0,
        target_theta_dot: float = 0.0,
    ):
        if Kp <= 0.0:
            raise ValueError(f"Proportional gain Kp must be strictly positive, got {Kp}")
        if Kd <= 0.0:
            raise ValueError(f"Derivative gain Kd must be strictly positive, got {Kd}")
        if max_torque <= 0.0:
            raise ValueError(f"max_torque must be strictly positive, got {max_torque}")
        if max_wheel_speed <= 0.0:
            raise ValueError(f"max_wheel_speed must be strictly positive, got {max_wheel_speed}")

        self.Kp: float = float(Kp)
        self.Kd: float = float(Kd)
        self.max_torque: float = float(max_torque)
        self.max_wheel_speed: float = float(max_wheel_speed)
        self.target_theta: float = float(target_theta)
        self.target_theta_dot: float = float(target_theta_dot)

    @classmethod
    def from_config(cls, config_path: Union[str, Path]) -> "PDController":
        """
        Instantiate a PDController from a YAML configuration file.

        Parameters
        ----------
        config_path : Union[str, Path]
            Path to the physics configuration YAML file.

        Returns
        -------
        PDController
            Configured controller instance.
        """
        path = Path(config_path)
        if not path.is_file():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        pd_cfg = cfg.get("pd_controller", {})
        rw_cfg = cfg.get("reaction_wheel", {})

        Kp = float(pd_cfg.get("Kp", 5.0))
        Kd = float(pd_cfg.get("Kd", 10.0))
        max_torque = float(pd_cfg.get("torque_limit", rw_cfg.get("max_torque", 1.0)))
        max_wheel_speed = float(rw_cfg.get("max_speed", 523.6))

        return cls(
            Kp=Kp,
            Kd=Kd,
            max_torque=max_torque,
            max_wheel_speed=max_wheel_speed,
        )

    def natural_frequency(self, J: float) -> float:
        """
        Calculate closed-loop natural frequency omega_n = sqrt(Kp / J).

        Parameters
        ----------
        J : float
            Total spacecraft moment of inertia in kg*m^2.

        Returns
        -------
        float
            Natural frequency in rad/s.
        """
        if J <= 0.0:
            raise ValueError(f"Moment of inertia J must be positive, got {J}")
        return float(np.sqrt(self.Kp / J))

    def damping_ratio(self, J: float) -> float:
        """
        Calculate closed-loop damping ratio zeta = Kd / (2 * sqrt(J * Kp)).

        Parameters
        ----------
        J : float
            Total spacecraft moment of inertia in kg*m^2.

        Returns
        -------
        float
            Dimensionless damping ratio zeta.
        """
        if J <= 0.0:
            raise ValueError(f"Moment of inertia J must be positive, got {J}")
        return float(self.Kd / (2.0 * np.sqrt(J * self.Kp)))

    def compute_torque(
        self,
        theta: float,
        theta_dot: float,
        Omega: Optional[float] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute control torque command given current attitude and angular velocity.

        Control Law:
            e_theta = theta - target_theta
            e_theta_dot = theta_dot - target_theta_dot
            u_raw = Kp * e_theta + Kd * e_theta_dot
            u_sat = clip(u_raw, -max_torque, +max_torque)

        Actuator Constraints:
            If Omega is provided and reaches speed limits:
            - Omega >= max_wheel_speed and u_sat > 0: u_cmd = 0 (prohibit further acceleration)
            - Omega <= -max_wheel_speed and u_sat < 0: u_cmd = 0
            - Otherwise: u_cmd = u_sat (allows despinning/braking torque)

        Parameters
        ----------
        theta : float
            Current spacecraft attitude angle in radians.
        theta_dot : float
            Current spacecraft angular rate in radians per second.
        Omega : Optional[float], optional
            Current reaction wheel relative speed in radians per second.

        Returns
        -------
        Tuple[float, Dict[str, Any]]
            u_cmd: Requested control torque in N*m.
            info: Diagnostic telemetry dictionary.
        """
        e_theta = float(theta - self.target_theta)
        e_theta_dot = float(theta_dot - self.target_theta_dot)

        # Raw linear PD control torque
        u_raw = float(self.Kp * e_theta + self.Kd * e_theta_dot)

        # 1. Apply motor torque saturation
        u_sat = float(np.clip(u_raw, -self.max_torque, self.max_torque))
        is_torque_saturated = abs(u_raw) > self.max_torque

        # 2. Check wheel speed limit if Omega is provided
        is_wheel_saturated = False
        u_cmd = u_sat
        if Omega is not None:
            if Omega >= self.max_wheel_speed:
                if u_sat > 0.0:
                    u_cmd = 0.0
                    is_wheel_saturated = True
            elif Omega <= -self.max_wheel_speed:
                if u_sat < 0.0:
                    u_cmd = 0.0
                    is_wheel_saturated = True

        info = {
            "e_theta": e_theta,
            "e_theta_dot": e_theta_dot,
            "u_raw": u_raw,
            "u_sat": u_sat,
            "u_cmd": u_cmd,
            "is_torque_saturated": is_torque_saturated,
            "is_wheel_saturated": is_wheel_saturated,
        }
        return u_cmd, info

    def __call__(
        self,
        theta: float,
        theta_dot: float,
        Omega: Optional[float] = None,
    ) -> float:
        """Allow controller instance to be called directly, returning u_cmd."""
        u_cmd, _ = self.compute_torque(theta, theta_dot, Omega)
        return u_cmd

    def __repr__(self) -> str:
        return (
            f"PDController(Kp={self.Kp:.2f} N*m/rad, Kd={self.Kd:.2f} N*m*s/rad, "
            f"max_torque={self.max_torque:.2f} N*m, max_speed={self.max_wheel_speed:.1f} rad/s)"
        )
