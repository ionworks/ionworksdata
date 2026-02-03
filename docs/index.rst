########################
Ionworks Data Processing
########################

Ionworks Data Processing (`ionworksdata`) is a library for processing experimental data into a common format. It contains readers for a variety of file formats, including Maccor, Biologic, and more., as well as tools for loading processed data for use in other Ionworks software.

For any questions or feedback, please contact info@ionworks.com.

Quick Start
===========

Processing time series data
---------------------------

Time series data can be extracted from a cycler file using the ``read.time_series`` function, which takes the filename and optionally the reader name (e.g. ``csv``, ``biologic_mpt``, ``maccor``, ``neware``, ``repower``). If the reader is not specified, it will be automatically detected from the file. The function returns a Polars DataFrame.

.. code-block:: python

    # With explicit reader
    data = iwdata.read.time_series("path/to/file.mpt", "biologic_mpt")

    # With auto-detection (reader is optional)
    data = iwdata.read.time_series("path/to/file.mpt")

The function automatically adds several columns to the output:
- "Step count": Cumulative step count (always present)
- "Discharge capacity [A.h]": Discharge capacity in ampere-hours (always present)
- "Charge capacity [A.h]": Charge capacity in ampere-hours (always present)
- "Discharge energy [W.h]": Discharge energy in watt-hours (always present)
- "Charge energy [W.h]": Charge energy in watt-hours (always present)
- "Step from cycler": Step number from cycler file (if provided)
- "Cycle from cycler": Cycle number from cycler file (if provided)

For information on the expected and returned columns, see the reader documentation. Additional columns can be added by passing a dictionary to the ``extra_column_mappings`` argument.

.. code-block:: python

    data = iwdata.read.time_series(
        "path/to/file.mpt", "biologic_mpt", extra_column_mappings={"My new column": "Old column name"}
    )

Processing step data
--------------------
Given a processed time series data, the step summary data can be extracted as follows:

.. code-block:: python

    steps = iwdata.steps.summarize(data)

This function identifies distinct steps within battery cycling data by detecting changes in the "Step count" column (which must be present in the input data). For each identified step, it extracts and calculates relevant metrics (voltage, current, capacity, energy, etc.) and determines the step type.

The output always includes:
- "Cycle count": Cumulative cycle count (defaults to 0 if no cycle information is available)
- "Cycle from cycler": Cycle number from cycler file (only if provided in the input data)
- "Discharge capacity [A.h]": Discharge capacity for the step
- "Charge capacity [A.h]": Charge capacity for the step
- "Discharge energy [W.h]": Discharge energy for the step
- "Charge energy [W.h]": Charge energy for the step
- "Step from cycler": Step number from cycler file (only if provided in the input data)

Note: The ``step_column`` and ``cycle_column`` parameters have been removed. The function now always uses "Step count" for step identification and "Cycle from cycler" (if available) for cycle tracking.

Alternatively, the time series and step data can be extracted together using the ``read.time_series_and_steps`` function.

.. code-block:: python

    # With explicit reader
    data, steps = iwdata.read.time_series_and_steps("path/to/file.mpt", "biologic_mpt")

    # With auto-detection (reader is optional)
    data, steps = iwdata.read.time_series_and_steps("path/to/file.mpt")

Labeling steps
--------------

Steps can be labeled using the functions in the ``steps`` module. For example, the following code labels the steps as cycling and pulse (charge and discharge).

.. code-block:: python

    options = {"cell_metadata": {"Nominal cell capacity [A.h]": 5}}
    steps = iwdata.steps.label_cycling(steps, options)
    for direction in ["charge", "discharge"]:
        options["current direction"] = direction
        steps = iwdata.steps.label_pulse(steps, options)

Measurement details
-------------------
The function ``ionworksdata.read.measurement_details`` can be used to return a ``measurement_details`` dictionary, which has keys "measurement", "time_series", and "steps". You need to first create the measurement dictionary (which contains details about the test, such as its name), and then pass it to the function along with the reader name and filename, and any additional arguments. The function will return the updated measurement details dictionary, which includes information extracted from the file, such as the start time, and the cycler name. This function also automatically labels the steps with some sensible defaults (custom labels can be added by passing a list of dictionaries to the ``labels`` argument).

.. code-block:: python

    measurement_details = iwdata.read.measurement_details(
        "path/to/file.mpt",
        measurement,
        "biologic_mpt",
        options={"cell_metadata": {"Nominal cell capacity [A.h]": 5}},
    )
    measurement = measurement_details["measurement"]
    time_series = measurement_details["time_series"]
    steps = measurement_details["steps"]

API Documentation
=================

Get detailed information about the functions, modules, and objects included in `ionworksdata` in the :doc:`source/api/index`.

.. toctree::
    :hidden:
    :maxdepth: 2

    source/api/index
