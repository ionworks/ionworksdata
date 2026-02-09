import pybamm
import pandas as pd
import numpy as np
import pathlib
import iwutil


def add_constant_current(model, parameter_values, sol=None):
    V_min = parameter_values["Lower voltage cut-off [V]"]
    V_max = parameter_values["Upper voltage cut-off [V]"]

    def charge(c_rate):
        return (
            pybamm.step.c_rate(-c_rate, termination=f"{V_max} V", period=6),
            pybamm.step.voltage(V_max, termination="C/50", period=6),
            pybamm.step.rest(3600),
        )

    def discharge(c_rate):
        return (
            pybamm.step.c_rate(c_rate, termination=f"{V_min} V", period=6),
            pybamm.step.rest(3600),
        )

    experiment = pybamm.Experiment(
        [
            charge(1 / 5),
            discharge(1 / 5),
            charge(1 / 5),
            discharge(1 / 2),
            charge(1 / 2),
            discharge(1),
            charge(1),
        ]
    )
    sim = pybamm.Simulation(
        model, parameter_values=parameter_values, experiment=experiment
    )
    if sol:
        sol = sim.solve(starting_solution=sol)
    else:
        sol = sim.solve(initial_soc=0.5)
    return sol


def add_pulse(model, parameter_values, experiment, sol):
    experiment, max_cycles = experiment
    sim = pybamm.Simulation(
        model, parameter_values=parameter_values, experiment=experiment
    )
    i = 0

    # Run until the voltage drops below the lower cut-off
    # during the second-last step
    while i == 0 or sol.cycles[-1].steps[-2].termination == "final time":
        sol = sim.solve(starting_solution=sol)
        i += 1
        if i > max_cycles:
            raise ValueError("Reached maximum number of cycles")

    return sol


def add_hppt(model, parameter_values, sol):
    # we want the C/3 discharge to increment SOC by 5%
    # C/3 discharge duration = 3h
    # 5% SOC increment = 0.05 * 3h = 0.15h
    discharge_duration = 0.15 * 3600
    # V_max = parameter_values["Upper voltage cut-off [V]"]
    V_min = parameter_values["Lower voltage cut-off [V]"]
    max_cycles = 25

    hppt_experiment = pybamm.Experiment(
        [
            (
                pybamm.step.c_rate(
                    1 / 2, duration=10, period=0.1, termination=f"{V_min} V"
                ),
                pybamm.step.rest(duration=300, period=5),
                pybamm.step.c_rate(
                    -1 / 2,
                    duration=10,
                    period=0.1,  # termination=f"{V_max} V"
                ),
                pybamm.step.rest(duration=300, period=5),
                pybamm.step.c_rate(
                    1, duration=10, period=0.1, termination=f"{V_min} V"
                ),
                pybamm.step.rest(duration=300, period=5),
                pybamm.step.c_rate(
                    -1,
                    duration=10,
                    period=0.1,  # termination=f"{V_max} V"
                ),
                pybamm.step.rest(duration=300, period=5),
                pybamm.step.c_rate(
                    1 / 3,
                    duration=discharge_duration,
                    period=discharge_duration / 100,
                    termination=f"{V_min} V",
                ),
                pybamm.step.rest(duration=3600, period=10),
            ),
        ]
    )
    sol = add_pulse(model, parameter_values, (hppt_experiment, max_cycles), sol)
    return sol


def add_gitt(model, parameter_values, sol):
    C_rate = -0.1
    step_s = 30 * 60
    step_h = step_s / 3600
    max_cycles = int(1 / abs(C_rate) / step_h * 1.5)
    V_max = parameter_values["Upper voltage cut-off [V]"]

    gitt_experiment = pybamm.Experiment(
        [
            (
                pybamm.step.c_rate(
                    C_rate,
                    duration=step_s,
                    period=step_s / 100,
                    termination=f"{V_max} V",
                ),
                pybamm.step.rest(3600, period=10),
            ),
        ]
    )
    sol = add_pulse(model, parameter_values, (gitt_experiment, max_cycles), sol)
    return sol


pybamm.set_logging_level("NOTICE")
model = pybamm.lithium_ion.SPM({"thermal": "lumped"})
model.variables["Temperature [degC]"] = model.variables[
    "X-averaged cell temperature [C]"
]
parameter_values = pybamm.ParameterValues("Chen2020")
sol = add_constant_current(model, parameter_values)
sol = add_hppt(model, parameter_values, sol)
sol = add_gitt(model, parameter_values, sol)
sol = add_constant_current(model, parameter_values, sol)
sol = add_hppt(model, parameter_values, sol)
sol = add_gitt(model, parameter_values, sol)

variables = ["Time [s]", "Current [A]", "Voltage [V]", "Temperature [degC]"]
df = pd.DataFrame(sol.get_data_dict(variables, cycles_and_steps=True))
# cut-off all max voltages at 4.2 V
df["Voltage [V]"] = np.minimum(df["Voltage [V]"], 4.2)

data_dir = pathlib.Path(__file__).resolve().parent / "data"
iwutil.save.csv(df, data_dir / "full_rpt" / "data.csv")
