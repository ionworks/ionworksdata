.. _api_read:

Data Reading
============

Classes for reading data from various file formats. Currently the following cycler file formats are supported:

.. toctree::
    :maxdepth: 2

    biologic/index
    csv/index
    maccor/index
    neware/index
    repower/index


.. currentmodule:: ionworksdata.read

Base Reader
-----------

.. autoclass:: BaseReader
   :members:
   :undoc-members:
   :show-inheritance:

Functions
---------

.. autofunction:: time_series

.. autofunction:: time_series_and_steps

.. autofunction:: keep_required_columns

.. autofunction:: measurement_details

.. autofunction:: start_time