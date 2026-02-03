.. _api_steps:

Step Analysis
=============

Functions for analyzing and labeling battery cycling steps. This module provides tools to summarize time series data into step-level information, label steps by type (cycling, pulse, EIS), and annotate time series with step labels.

.. currentmodule:: ionworksdata.steps

Core Functions
--------------

.. autofunction:: summarize

.. autofunction:: identify

.. autofunction:: annotate

.. autofunction:: validate

.. autofunction:: infer_type

.. autofunction:: set_cycle_capacity

.. autofunction:: set_cycle_energy

Labeling Functions
------------------

.. autofunction:: label_cycling

.. autofunction:: label_pulse

.. autofunction:: label_eis
