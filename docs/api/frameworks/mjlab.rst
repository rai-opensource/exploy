MjLab Framework
===============

This section contains documentation for the MjLab framework adapter, which provides tools for
exporting policies from MjLab to ONNX format using exploy.


Environment
-----------

The exportable environment adapter for MjLab.

.. automodule:: exploy.exporter.frameworks.mjlab.env
   :members:
   :undoc-members:
   :show-inheritance:


Actor
-----

Actor wrappers for RSL-RL v5 policy networks.

.. automodule:: exploy.exporter.frameworks.mjlab.actor
   :members:
   :undoc-members:
   :show-inheritance:


Inputs
------

Utilities for registering sensor and command inputs from MjLab managers.

.. automodule:: exploy.exporter.frameworks.mjlab.inputs
   :members:
   :undoc-members:
   :show-inheritance:


Outputs
-------

Utilities for registering action outputs from MjLab managers.

.. automodule:: exploy.exporter.frameworks.mjlab.outputs
   :members:
   :undoc-members:
   :show-inheritance:


Memory
------

Utilities for registering memory components from MjLab managers.

.. automodule:: exploy.exporter.frameworks.mjlab.memory
   :members:
   :undoc-members:
   :show-inheritance:


Data Sources
------------

Adaptor classes that mirror MjLab data interfaces while managing their own
tensor data for ONNX export.

.. automodule:: exploy.exporter.frameworks.mjlab.entity_data
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: exploy.exporter.frameworks.mjlab.raycaster_data
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: exploy.exporter.frameworks.mjlab.sensor_proxy
   :members:
   :undoc-members:
   :show-inheritance:


Utilities
---------

Helper functions for MjLab integration.

.. automodule:: exploy.exporter.frameworks.mjlab.utils
   :members:
   :undoc-members:
   :show-inheritance:
