Isaac Lab Framework
===================

This section contains documentation for the Isaac Lab framework adapter, which provides tools for
exporting policies from Isaac Lab to ONNX format using exploy.


Environment
-----------

The exportable environment adapter for Isaac Lab.

.. automodule:: exploy.exporter.frameworks.isaaclab.env
   :members:
   :undoc-members:
   :show-inheritance:


Actor
-----

Actor wrappers for RSL-RL policy networks.

.. automodule:: exploy.exporter.frameworks.isaaclab.actor
   :members:
   :undoc-members:
   :show-inheritance:


Inputs
------

Utilities for registering command inputs from Isaac Lab managers.

.. automodule:: exploy.exporter.frameworks.isaaclab.inputs
   :members:
   :undoc-members:
   :show-inheritance:


Outputs
-------

Utilities for registering action outputs from Isaac Lab managers.

.. automodule:: exploy.exporter.frameworks.isaaclab.outputs
   :members:
   :undoc-members:
   :show-inheritance:


Memory
------

Utilities for registering memory components parsed from Isaac Lab managers.

.. automodule:: exploy.exporter.frameworks.isaaclab.memory
   :members:
   :undoc-members:
   :show-inheritance:


Data Sources
------------

Adaptor classes that mirror Isaac Lab data interfaces while managing their own
tensor data for ONNX export.

.. automodule:: exploy.exporter.frameworks.isaaclab.articulation_data
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: exploy.exporter.frameworks.isaaclab.raycaster_data
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: exploy.exporter.frameworks.isaaclab.rigid_object_data
   :members:
   :undoc-members:
   :show-inheritance:


Utilities
---------

Helper functions for Isaac Lab integration.

.. automodule:: exploy.exporter.frameworks.isaaclab.utils
   :members:
   :undoc-members:
   :show-inheritance:
