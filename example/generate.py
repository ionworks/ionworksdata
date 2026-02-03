import pybamm
import pandas as pd
from scipy.integrate import cumulative_trapezoid
import pathlib
import iwutil
import ionworksdata as iwdata

data_dir = pathlib.Path(__file__).resolve().parent / "data"


def save_data_metadata(df, metadata, folder):
    iwutil.save.csv(df, folder / "data.csv")
    iwutil.save.json(metadata, folder / "metadata.json")


def generate_pulse_data(experiment_fn):
    model = pybamm.lithium_ion.SPM()
    parameter_values = pybamm.ParameterValues("Chen2020")
    experiment = experiment_fn(parameter_values)

    sim = pybamm.Simulation(
        model, parameter_values=parameter_values, experiment=experiment
    )
    sol = sim.solve()

    # Save data to csv
    variables = ["Time [s]", "Current [A]", "Voltage [V]"]
    df = pd.DataFrame(sol.get_data_dict(variables))
    df = df.rename(columns={"Cycle": "Cycle number", "Step": "Step number"})
    df = df.astype({"Cycle number": int, "Step number": int})
    t = df["Time [s]"].values
    current = df["Current [A]"].values
    df["Capacity [A.h]"] = cumulative_trapezoid(current, t / 3600, initial=0)

    try:
        iwdata.validate.pulse(df)
        validated = True
    except AssertionError:
        validated = False

    return df, validated


def generate_gitt_data():
    def experiment_fn(parameter_values):
        C_rate = 0.1
        pause_s = 10  # 10s
        step_s = 15 * 60  # 15m
        rest_s = 15 * 60  # 15m
        V_min = parameter_values["Lower voltage cut-off [V]"]

        gitt_experiment = pybamm.Experiment(
            [
                (
                    pybamm.step.rest(pause_s, period=1),
                    pybamm.step.c_rate(
                        C_rate,
                        duration=step_s,
                        period=step_s / 1000,
                        termination=f"{V_min} V",
                    ),
                    pybamm.step.rest(rest_s, period=rest_s / 100),
                ),
            ]
            * 5
        )
        return gitt_experiment

    df, validated = generate_pulse_data(experiment_fn)

    metadata = {
        "experiment": "gitt",
        "components": "full",
        "direction": "discharge",
        "validated": validated,
    }
    save_data_metadata(df, metadata, data_dir / "gitt")


def generate_hppt_data():
    def experiment_fn(parameter_values):
        # we want the C/3 discharge to increment SOC by 5%
        # C/3 discharge duration = 3h
        # 5% SOC increment = 0.05 * 3h = 0.15h
        discharge_duration = 0.15 * 3600
        V_min = parameter_values["Lower voltage cut-off [V]"]

        hppt_experiment = pybamm.Experiment(
            [
                (
                    pybamm.step.rest(duration=10, period=1),
                    pybamm.step.c_rate(1, duration=10, period=0.1),
                    pybamm.step.rest(duration=300, period=5),
                    pybamm.step.c_rate(-1, duration=10, period=0.1),
                    pybamm.step.rest(duration=300, period=5),
                    pybamm.step.c_rate(2, duration=10, period=0.1),
                    pybamm.step.rest(duration=300, period=5),
                    pybamm.step.c_rate(-2, duration=10, period=0.1),
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
            * 5
        )
        return hppt_experiment

    df, validated = generate_pulse_data(experiment_fn)

    metadata = {
        "experiment": "hppt",
        "components": "full",
        "direction": "discharge",
        "validated": validated,
    }

    save_data_metadata(df, metadata, data_dir / "hppt")


def generate_ocv_and_rate_capability_data():
    model = pybamm.lithium_ion.SPM({"thermal": "lumped"})
    model.variables["Temperature [degC]"] = model.variables[
        "X-averaged cell temperature [C]"
    ]

    parameter_values = pybamm.ParameterValues("Chen2020")
    V_min = parameter_values["Lower voltage cut-off [V]"]
    for C_rate in [1 / 50, 1 / 5, 1 / 2, 1, 2]:
        experiment = pybamm.Experiment(
            [
                pybamm.step.c_rate(
                    C_rate, termination=f"{V_min} V", period=6 * 1 / C_rate
                )
            ]
        )
        sim = pybamm.Simulation(
            model, parameter_values=parameter_values, experiment=experiment
        )
        sol = sim.solve(initial_soc=1)
        if C_rate == 1 / 50:
            df = pd.DataFrame(
                {
                    "Capacity [A.h]": sol["Discharge capacity [A.h]"].data,
                    "Voltage [V]": sol["Voltage [V]"].data,
                }
            )
            metadata = {
                "experiment": "ocv",
                "components": "full",
                "direction": "discharge",
            }
            try:
                iwdata.validate.ocv(df)
                metadata["validated"] = True
            except AssertionError:
                metadata["validated"] = False
            save_data_metadata(df, metadata, data_dir / "ocv")
        else:
            variables = ["Time [s]", "Current [A]", "Voltage [V]", "Temperature [degC]"]
            df = pd.DataFrame(sol.get_data_dict(variables))
            df["Capacity [A.h]"] = sol["Discharge capacity [A.h]"].data
            metadata = {
                "experiment": "constant current",
                "components": "full",
                "direction": "discharge",
                "C-rate": C_rate,
            }
            try:
                iwdata.validate.constant_current(df)
                metadata["validated"] = True
            except AssertionError:
                metadata["validated"] = False
            save_data_metadata(df, metadata, data_dir / f"{C_rate}C")


print("Generating OCV and rate capability data ...")
generate_ocv_and_rate_capability_data()
print("Generating GITT data ...")
generate_gitt_data()
print("Generating HPPT data ...")
generate_hppt_data()
print("Done.")
