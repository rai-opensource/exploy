Getting Started
===============

Prerequisites
-------------

* `Pixi <https://pixi.sh>`_ installed on your system
* For Isaac Lab features: NVIDIA GPU with CUDA support

Environment Setup
-----------------

1. Clone the repository:

.. code-block:: bash

   git clone <repository-url>
   cd export

2. Install dependencies:

.. code-block:: bash

   pixi install

3. Choose your environment based on your needs:

Python Development
~~~~~~~~~~~~~~~~~~

For Python-only development:

.. code-block:: bash

   pixi shell -e python

C++ Development
~~~~~~~~~~~~~~~

For C++ controller development:

.. code-block:: bash

   pixi shell -e cpp
   pixi run -e cpp configure
   pixi run -e cpp build

Exporter with Isaac Lab
~~~~~~~~~~~~~~~~~~~~~~~~

For exporting policies from Isaac Lab:

.. code-block:: bash

   pixi shell -e exporter
   pixi run -e exporter setup

Full Controller Development
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For both Python and C++ development:

.. code-block:: bash

   pixi shell -e controller
   pixi run -e controller setup
   pixi run -e controller build

Development Workflow
--------------------

Code Formatting
~~~~~~~~~~~~~~~

.. code-block:: bash

   # Format Python code
   pixi run -e python format

   # Format C++ code
   pixi run -e cpp format

Linting
~~~~~~~

.. code-block:: bash

   # Lint Python code
   pixi run -e python lint

   # Check formatting
   pixi run -e python check

Testing
~~~~~~~

.. code-block:: bash

   # Run Python tests
   pixi run -e python test

   # Run C++ tests
   pixi run -e cpp test

Building Documentation
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pixi run docs

   # Clean documentation build
   pixi run docs-clean

Next Steps
----------

* See :doc:`api/index` for detailed API documentation
* Explore examples in the ``examples/`` directory
