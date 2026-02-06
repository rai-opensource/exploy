Exporter Documentation
======================

Welcome to the Exporter documentation!

Exporter is a development environment for a C++ library with Python bindings,
integrated with ONNX Runtime and PyTorch. It provides tools for exporting
reinforcement learning policies and controllers to ONNX format for deployment.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   api/index

Getting Started
===============

Installation
------------

Install the environment using Pixi:

.. code-block:: bash

   pixi install

Available Environments
~~~~~~~~~~~~~~~~~~~~~~

* ``python`` - Python-only development
* ``cpp`` - C++-only development
* ``exporter`` - Python + Isaac Lab (model export)
* ``controller`` - Python + C++ (controller development)

Building the Project
--------------------

Configure and build the C++ library:

.. code-block:: bash

   pixi run -e controller configure
   pixi run -e controller build

Running Tests
-------------

.. code-block:: bash

   # Python tests
   pixi run -e python test

   # C++ tests
   pixi run -e cpp test

Building Documentation
----------------------

To build this documentation:

.. code-block:: bash

   pixi run docs

The built documentation will be in ``docs/_build/html/``.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
