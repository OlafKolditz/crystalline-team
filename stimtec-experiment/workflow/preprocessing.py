"""Preprocessing helpers for the STIMTEC DFN workflow."""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class FlowRateSchedule:
    borehole_length: float
    flow_times: np.ndarray
    flow_values: np.ndarray
    time_max: float = 3000.0


def _validate_flow_schedule(schedule: FlowRateSchedule) -> None:
    if schedule.flow_times.ndim != 1 or schedule.flow_values.ndim != 1:
        raise ValueError("FLOW_TIMES and FLOW_VALUES must be one-dimensional arrays.")
    if len(schedule.flow_times) != len(schedule.flow_values):
        raise ValueError("FLOW_TIMES and FLOW_VALUES must have the same length.")
    if len(schedule.flow_times) == 0:
        raise ValueError("FLOW_TIMES and FLOW_VALUES must not be empty.")
    if not np.all(np.diff(schedule.flow_times) > 0.0):
        raise ValueError("FLOW_TIMES must be strictly increasing.")
    if schedule.borehole_length <= 0.0:
        raise ValueError("BOREHOLE_LENGTH must be positive.")
    if schedule.time_max <= 0.0:
        raise ValueError("time_max must be positive.")


def _format_ogs_float(value: float) -> str:
    value = float(value)
    if value == 0.0:
        return "0.0"
    return f"{value:.15g}"


def flow_rate_total(time_values, schedule: FlowRateSchedule):
    _validate_flow_schedule(schedule)
    time_values = np.asarray(time_values, dtype=float)
    value_indices = np.searchsorted(schedule.flow_times, time_values, side="right")
    schedule_values = np.append(schedule.flow_values, 0.0)
    return schedule_values[value_indices]


def input_flux(time_values, schedule: FlowRateSchedule):
    return flow_rate_total(time_values, schedule) / schedule.borehole_length


def build_q_in_expression(schedule: FlowRateSchedule) -> str:
    """Build the OGS q_in function expression from the preprocessing schedule."""
    _validate_flow_schedule(schedule)

    expression = "0.0"
    for time_limit, flow_value in reversed(list(zip(schedule.flow_times, schedule.flow_values))):
        expression = (
            f"(t < {_format_ogs_float(time_limit)} ? "
            f"{_format_ogs_float(flow_value)} : {expression})"
        )

    return f"({expression} / {_format_ogs_float(schedule.borehole_length)})"


def apply_preprocessing_to_project(project, schedule: FlowRateSchedule) -> str:
    """Sync the preprocessing-derived input flux into an ogstools project."""
    expression_xpath = "./parameters/parameter[name='q_in']/expression"
    if project.tree is None or project.tree.getroot().find(expression_xpath) is None:
        raise KeyError("The project file does not define a q_in expression parameter.")

    q_in_expression = build_q_in_expression(schedule)
    project.replace_text(q_in_expression, xpath=expression_xpath)
    return q_in_expression


def plot_input_flux_data(schedule: FlowRateSchedule) -> None:
    time_values = np.linspace(0.0, schedule.time_max, 1000)
    total_flow = flow_rate_total(time_values, schedule)
    flux = input_flux(time_values, schedule)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

    axes[0].step(time_values, total_flow, where="post", linewidth=2.0)
    axes[0].set_title("Total Flow Rate")
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Flow rate [m^3/s]")

    axes[1].step(time_values, flux, where="post", linewidth=2.0)
    axes[1].set_title("Input Flux For OGS")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Flux [m^2/s]")

    for axis in axes:
        axis.grid(True, linestyle="--", alpha=0.35)
        axis.set_xlim(0.0, schedule.time_max)
        axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    fig.tight_layout()
    plt.show()
    plt.close(fig)


def run_preprocessing(schedule: FlowRateSchedule) -> None:
    """Run preprocessing plots for the workflow."""
    plot_input_flux_data(schedule)
