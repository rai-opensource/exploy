Isaac Lab 3 Framework
=====================

This section contains documentation for the Isaac Lab 3 framework adapter, which provides tools for
exporting policies from Isaac Lab 3 to ONNX format using exploy. Isaac Lab 2 is supported by the
:doc:`isaaclab` adapter.

Isaac Lab 3 runs without Isaac Sim (kit-less), e.g. with the Newton physics backend, also on the CPU.
Compared to Isaac Lab 2, the adapter accounts for the following changes:

- **ProxyArray data.** Asset and sensor data return ``ProxyArray`` objects, read by observation terms
  through ``.torch``. During export, the data sources return a
  :class:`~exploy.exporter.frameworks.isaaclab3.proxy_array.TorchProxyArray`, whose ``.torch`` is the
  exploy-managed tensor that becomes an input of the ONNX graph.
- **Quaternions.** Isaac Lab 3 uses the (x, y, z, w) convention. The ONNX inputs keep the (w, x, y, z)
  convention of exploy's interface (shared with the other frameworks and exploy's C++ controller), and
  are converted within the graph. Exported policies are therefore deployed as with Isaac Lab 2.
- **Joint targets.** Isaac Lab 3 writes joint targets with warp kernels, which cannot be traced. During
  export, the asset of each action term is replaced by an
  :class:`~exploy.exporter.frameworks.isaaclab3.writes.ArticulationWriteRecorder`, which records the
  targets into torch buffers, from which the outputs are read.
- **RSL-RL 5.** Actors are RSL-RL ``MLPModel`` and ``RNNModel`` instances (e.g. ``runner.alg.actor``).
- **Play mode.** Tasks have no ``-Play`` variants anymore: resolve their configuration in play mode
  (e.g. ``resolve_task_config(task, agent_entry_point, play_mode=True)``) to disable observation noise.

See ``examples/exporter_scripts/isaaclab3/export_isaaclab3.py`` for a complete export.


Environment
-----------

.. automodule:: exploy.exporter.frameworks.isaaclab3.env
   :members:
   :undoc-members:
   :show-inheritance:


Actor
-----

.. automodule:: exploy.exporter.frameworks.isaaclab3.actor
   :members:
   :undoc-members:
   :show-inheritance:


Inputs
------

.. automodule:: exploy.exporter.frameworks.isaaclab3.inputs
   :members:
   :undoc-members:
   :show-inheritance:


Outputs
-------

.. automodule:: exploy.exporter.frameworks.isaaclab3.outputs
   :members:
   :undoc-members:
   :show-inheritance:


Memory
------

.. automodule:: exploy.exporter.frameworks.isaaclab3.memory
   :members:
   :undoc-members:
   :show-inheritance:


Data Sources
------------

Adaptor classes that mirror Isaac Lab 3 data interfaces while managing their own tensor data for ONNX
export.

.. automodule:: exploy.exporter.frameworks.isaaclab3.body_data
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: exploy.exporter.frameworks.isaaclab3.articulation_data
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: exploy.exporter.frameworks.isaaclab3.raycaster_data
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: exploy.exporter.frameworks.isaaclab3.proxy_array
   :members:
   :undoc-members:
   :show-inheritance:


Joint Target Recording
----------------------

.. automodule:: exploy.exporter.frameworks.isaaclab3.writes
   :members:
   :undoc-members:
   :show-inheritance:


Utilities
---------

.. automodule:: exploy.exporter.frameworks.isaaclab3.utils
   :members:
   :undoc-members:
   :show-inheritance:
