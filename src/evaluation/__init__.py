"""
Performance metrics, comparative benchmarking, and visualization routines.
"""

from src.evaluation.metrics import (
    compute_settling_time,
    compute_decay_time,
    compute_overshoot,
    compute_periodic_metrics,
    compute_pulse_recovery_metrics,
    analyze_saturation,
    analyze_constant_disturbance_momentum,
    compute_squared_torque_integral,
    calculate_comprehensive_metrics,
)
from src.evaluation.visualization import (
    plot_simulation_run,
    plot_comparative_scenarios,
)

__all__ = [
    "compute_settling_time",
    "compute_decay_time",
    "compute_overshoot",
    "compute_periodic_metrics",
    "compute_pulse_recovery_metrics",
    "analyze_saturation",
    "analyze_constant_disturbance_momentum",
    "compute_squared_torque_integral",
    "calculate_comprehensive_metrics",
    "plot_simulation_run",
    "plot_comparative_scenarios",
]
