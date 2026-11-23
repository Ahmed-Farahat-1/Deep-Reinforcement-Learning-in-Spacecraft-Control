"""
Spacecraft Attitude Dynamics Simulator with Reaction Wheel.

Implements a single-axis attitude dynamics simulation governed by
angular momentum conservation and Newton-Euler formulation:

    J * theta_ddot + I * Omega_dot = M_d
    u = I * Omega_dot
    J * theta_ddot = M_d - u

Where:
    - J = J_body + I: Total effective system moment of inertia (kg*m^2).
    - I: Reaction wheel polar moment of inertia (kg*m^2).
    - theta: Spacecraft body attitude angle (rad).
    - theta_dot: Spacecraft body angular velocity (rad/s).
    - Omega: Reaction wheel angular velocity relative to spacecraft (rad/s).
    - omega_w = theta_dot + Omega: Absolute wheel angular velocity (rad/s).
    - M_d: External environmental disturbance torque (N*m).
    - u: Internal motor control torque applied to reaction wheel (N*m).

State Vector:
    x = [theta, theta_dot, Omega]^T

All calculations use strict SI units (rad, rad/s, kg*m^2, N*m, s).
"""

from typing import Optional, Dict, Any, Union, Tuple, Callable
from pathlib import Path
import yaml
import numpy as np

from src.simulation.disturbances import BaseDisturbance, ZeroDisturbance


class Spacecraft1D:
    """
    Single-axis spacecraft attitude simulator actuated by a reaction wheel.

    Parameters
    ----------
    J_body : float, optional
        Spacecraft body moment of inertia (kg*m^2), default: 10.0.
    I_wheel : float, optional
        Reaction wheel polar moment of inertia (kg*m^2), default: 0.05.
    max_torque : float, optional
        Maximum motor control torque magnitude |u_max| (N*m), default: 1.0.
    max_speed : float, optional
        Maximum relative wheel angular speed |Omega_max| (rad/s), default: 523.6 (~5000 RPM).
    dt : float, optional
        Simulation integration time step in seconds, default: 0.01.
    config_path : Optional[Union[str, Path]], optional
        Optional path to a YAML configuration file to override parameters.
    """

    def __init__(
        self,
        J_body: float = 10.0,
        I_wheel: float = 0.05,
        max_torque: float = 1.0,
        max_speed: float = 523.6,
        dt: float = 0.01,
        config_path: Optional[Union[str, Path]] = None,
    ):
        # Default physical parameters (illustrative simulation baseline)
        self.J_body: float = float(J_body)
        self.I_wheel: float = float(I_wheel)
        self.max_torque: float = float(max_torque)
        self.max_speed: float = float(max_speed)
        self.dt: float = float(dt)

        # Load from config file if provided
        if config_path is not None:
            self._load_config(config_path)

        # Total system inertia about rotation axis: J = J_body + I_wheel
        self.J: float = self.J_body + self.I_wheel
        self.I: float = self.I_wheel

        # Simulation state: x = [theta, theta_dot, Omega]
        self._state: np.ndarray = np.zeros(3, dtype=np.float64)
        self._time: float = 0.0

    def _load_config(self, config_path: Union[str, Path]) -> None:
        """Load physical parameters from a YAML configuration file."""
        path = Path(config_path)
        if not path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        if "spacecraft" in cfg:
            sc = cfg["spacecraft"]
            self.J_body = float(sc.get("J_body", self.J_body))
            self.I_wheel = float(sc.get("I_wheel", self.I_wheel))

        if "reaction_wheel" in cfg:
            rw = cfg["reaction_wheel"]
            self.max_torque = float(rw.get("max_torque", self.max_torque))
            self.max_speed = float(rw.get("max_speed", self.max_speed))

        if "simulation" in cfg:
            sim = cfg["simulation"]
            self.dt = float(sim.get("time_step", self.dt))

    def reset(
        self,
        initial_state: Optional[Union[np.ndarray, list, tuple]] = None,
        t0: float = 0.0,
    ) -> np.ndarray:
        """
        Reset the simulation to a specified initial state.

        Parameters
        ----------
        initial_state : Optional[Union[np.ndarray, list, tuple]]
            Initial state vector [theta_0, theta_dot_0, Omega_0].
            If None, initializes to [0.0, 0.0, 0.0].
        t0 : float, optional
            Initial simulation time (s), default: 0.0.

        Returns
        -------
        np.ndarray
            Initial state vector of shape (3,).
        """
        if initial_state is not None:
            arr = np.asarray(initial_state, dtype=np.float64)
            if arr.shape != (3,):
                raise ValueError(f"initial_state must have shape (3,), got {arr.shape}")
            self._state = arr.copy()
        else:
            self._state = np.zeros(3, dtype=np.float64)

        # Enforce initial wheel speed limit
        self._state[2] = np.clip(self._state[2], -self.max_speed, self.max_speed)
        self._time = float(t0)
        return self._state.copy()

    def compute_effective_torque(self, u_cmd: float, Omega: float) -> Tuple[float, bool, bool]:
        """
        Compute effective applied motor torque considering separate torque and speed limits.

        Saturation Logic:
        1. Motor torque limit: u_sat = clip(u_cmd, -max_torque, +max_torque).
        2. Upper speed limit: If Omega >= max_speed and u_sat > 0, prohibit further
           positive acceleration (u_applied = 0), but allow negative braking torque (u_sat < 0).
        3. Lower speed limit: If Omega <= -max_speed and u_sat < 0, prohibit further
           negative acceleration (u_applied = 0), but allow positive braking torque (u_sat > 0).

        Parameters
        ----------
        u_cmd : float
            Commanded motor torque (N*m).
        Omega : float
            Current relative wheel angular speed (rad/s).

        Returns
        -------
        Tuple[float, bool, bool]
            (u_applied, is_torque_saturated, is_wheel_saturated)
        """
        # 1. Motor torque limit
        u_sat = float(np.clip(u_cmd, -self.max_torque, self.max_torque))
        is_torque_saturated = abs(u_cmd) > self.max_torque

        # 2. Wheel speed limit checks
        is_wheel_saturated = False
        if Omega >= self.max_speed:
            if u_sat > 0.0:
                u_applied = 0.0
                is_wheel_saturated = True
            else:
                u_applied = u_sat
        elif Omega <= -self.max_speed:
            if u_sat < 0.0:
                u_applied = 0.0
                is_wheel_saturated = True
            else:
                u_applied = u_sat
        else:
            u_applied = u_sat

        return u_applied, is_torque_saturated, is_wheel_saturated

    def derivatives(
        self,
        t: float,
        state: np.ndarray,
        u_cmd: float,
        disturbance: Union[float, Callable[[float], float], BaseDisturbance] = 0.0,
    ) -> Tuple[np.ndarray, float]:
        """
        Compute continuous state derivatives: x_dot = [theta_dot, theta_ddot, Omega_dot].

        Parameters
        ----------
        t : float
            Current time in seconds.
        state : np.ndarray
            Current state vector [theta, theta_dot, Omega].
        u_cmd : float
            Commanded control torque (N*m).
        disturbance : Union[float, Callable[[float], float], BaseDisturbance]
            Disturbance torque or disturbance model at time t.

        Returns
        -------
        Tuple[np.ndarray, float]
            (state_derivative, u_applied)
        """
        theta, theta_dot, Omega = state[0], state[1], state[2]

        # Evaluate disturbance torque at time t
        if isinstance(disturbance, (BaseDisturbance, Callable)):
            Md = float(disturbance(t))
        else:
            Md = float(disturbance)

        # Apply actuator saturation logic
        u_applied, _, _ = self.compute_effective_torque(u_cmd, Omega)

        # State derivative equations:
        # theta_dot = x[1]
        # theta_ddot = (M_d - u_applied) / J
        # Omega_dot = u_applied / I
        theta_ddot = (Md - u_applied) / self.J
        Omega_dot = u_applied / self.I

        dstate = np.array([theta_dot, theta_ddot, Omega_dot], dtype=np.float64)
        return dstate, u_applied

    def rk4_step(
        self,
        state: np.ndarray,
        t: float,
        dt: float,
        u_cmd: float,
        disturbance: Union[float, Callable[[float], float], BaseDisturbance] = 0.0,
    ) -> Tuple[np.ndarray, float]:
        """
        Perform a single fourth-order Runge-Kutta (RK4) numerical integration step.

        Parameters
        ----------
        state : np.ndarray
            Current state vector.
        t : float
            Current time in seconds.
        dt : float
            Time step in seconds.
        u_cmd : float
            Commanded control torque in N*m.
        disturbance : Union[float, Callable[[float], float], BaseDisturbance]
            External disturbance torque.

        Returns
        -------
        Tuple[np.ndarray, float]
            (next_state, u_applied_at_start)
        """
        # k1 at t
        k1, u_applied = self.derivatives(t, state, u_cmd, disturbance)

        # k2 at t + dt/2
        state_k2 = state + 0.5 * dt * k1
        k2, _ = self.derivatives(t + 0.5 * dt, state_k2, u_cmd, disturbance)

        # k3 at t + dt/2
        state_k3 = state + 0.5 * dt * k2
        k3, _ = self.derivatives(t + 0.5 * dt, state_k3, u_cmd, disturbance)

        # k4 at t + dt
        state_k4 = state + dt * k3
        k4, _ = self.derivatives(t + dt, state_k4, u_cmd, disturbance)

        # State update
        next_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        # Enforce wheel speed limits to prevent numerical integration overshoot
        next_state[2] = np.clip(next_state[2], -self.max_speed, self.max_speed)

        return next_state, u_applied

    def step(
        self,
        u_cmd: float,
        disturbance: Union[float, Callable[[float], float], BaseDisturbance] = 0.0,
        dt: Optional[float] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Advance the simulation by one time step dt using RK4.

        Parameters
        ----------
        u_cmd : float
            Commanded motor torque in N*m.
        disturbance : Union[float, Callable[[float], float], BaseDisturbance]
            External disturbance torque in N*m.
        dt : Optional[float]
            Optional step size override. Defaults to self.dt.

        Returns
        -------
        Tuple[np.ndarray, Dict[str, Any]]
            next_state: Updated state vector [theta, theta_dot, Omega].
            info: Diagnostic telemetry dictionary.
        """
        step_dt = float(dt) if dt is not None else self.dt
        t_current = self._time
        state_current = self._state

        # Compute saturation flags before integration
        u_eff, is_torque_sat, is_wheel_sat = self.compute_effective_torque(
            u_cmd, state_current[2]
        )

        # Numerical integration via RK4
        next_state, u_applied = self.rk4_step(
            state_current, t_current, step_dt, u_cmd, disturbance
        )

        # Update internal state
        self._state = next_state
        self._time += step_dt

        # Evaluate disturbance torque for logging
        if isinstance(disturbance, (BaseDisturbance, Callable)):
            md_val = float(disturbance(t_current))
        else:
            md_val = float(disturbance)

        info = {
            "time": self._time,
            "u_cmd": float(u_cmd),
            "u_applied": float(u_applied),
            "disturbance_torque": md_val,
            "total_angular_momentum": self.total_angular_momentum(next_state),
            "absolute_wheel_speed": self.absolute_wheel_speed(next_state),
            "is_torque_saturated": is_torque_sat,
            "is_wheel_saturated": is_wheel_sat,
        }

        return self._state.copy(), info

    def total_angular_momentum(self, state: Optional[np.ndarray] = None) -> float:
        """
        Calculate total system angular momentum about rotation axis in inertial space:
        H = J * theta_dot + I * Omega

        Parameters
        ----------
        state : Optional[np.ndarray]
            State vector to evaluate. Defaults to current simulation state.

        Returns
        -------
        float
            Total angular momentum in kg*m^2/s (or N*m*s).
        """
        st = self._state if state is None else state
        theta_dot = st[1]
        Omega = st[2]
        return float(self.J * theta_dot + self.I * Omega)

    def absolute_wheel_speed(self, state: Optional[np.ndarray] = None) -> float:
        """
        Calculate absolute (inertial) angular velocity of the reaction wheel:
        omega_w = theta_dot + Omega

        Parameters
        ----------
        state : Optional[np.ndarray]
            State vector to evaluate. Defaults to current simulation state.

        Returns
        -------
        float
            Absolute wheel speed in rad/s.
        """
        st = self._state if state is None else state
        return float(st[1] + st[2])

    @property
    def state(self) -> np.ndarray:
        """Current state vector [theta, theta_dot, Omega]."""
        return self._state.copy()

    @property
    def theta(self) -> float:
        """Spacecraft attitude angle in radians (rad)."""
        return float(self._state[0])

    @property
    def theta_dot(self) -> float:
        """Spacecraft angular velocity in radians per second (rad/s)."""
        return float(self._state[1])

    @property
    def Omega(self) -> float:
        """Reaction wheel relative angular velocity in radians per second (rad/s)."""
        return float(self._state[2])

    @property
    def time(self) -> float:
        """Current simulation elapsed time in seconds (s)."""
        return self._time
