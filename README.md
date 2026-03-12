# Exploy

EXport and dePLOY Reinforcement Learning policies.

The core idea lies in a "self-contained" export approach:
Rather than exporting only the neural network policy, this tool captures the entire environment
logic, including observation generation and action processing, into a single ONNX file.
By tracing Torch operations from the simulation environment, the exporter embeds the computational
layers required to transform raw robot state interfaces into policy inputs and policy outputs into
executable commands.

By encapsulating the environment's computation graph within the model file itself,
this library minimizes operational effort and maximizes confidence that a policy will behave
identically in simulation and on physical hardware.

Authors: Dario Bellicoso, Annika Wollschläger

## Features

- **Environment Exporting**: Export RL environments and policies from
  simulation frameworks in a self-contained ONNX file.
- **C++ Controller with ONNX Runtime Integration**: Deploy trained policies using ONNX Runtime
  for real-time policy execution
- **Multi-Framework Support**: Built-in support for IsaacLab with extensible
  framework integration

## Project Structure

- `control/`: C++ controller library with ONNX Runtime integration
- `exploy/`: Python exporter package for policy and environment export
- `examples/`: Usage examples for supported frameworks
- `docs/`: Documentation source files

## Getting Started

### Prerequisites

- [Pixi](https://pixi.sh) installed on your system

### Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/bdaiinstitute/exploy.git
   cd exploy
   ```

2. **Initialize the environment and install dependencies**:

   ```bash
   pixi install
   ```

### Building the Project

1. **Configure and Build C++ Library**:

   ```bash
   pixi run -e controller configure
   pixi run -e controller build
   ```

2. **Run tests**:

   ```bash
   # Python tests
   pixi run -e core test

   # C++ tests
   pixi run -e controller test
   ```

### Tutorials

- [**Exporter Tutorial**](docs/tutorial/exporter/exporter_tutorial.md) — Step-by-step guide to
  exporting an RL environment and policy to a self-contained ONNX file using `exploy.exporter.core`.
- [**Controller Tutorial**](docs/tutorial/controller/controller_tutorial.md) — Step-by-step guide to
  deploying a trained policy on a robot using the C++ controller with ONNX Runtime integration.

## Versioning

This project uses semantic versioning (MAJOR.MINOR.PATCH). The current version is specified in `pixi.toml`.

Releases are published using GitHub Releases. Version tags follow the format `vX.Y.Z`.

## Limitations

- IsaacLab integration requires NVIDIA GPU with CUDA support
- C++ controller is designed for real-time systems but performance depends
  on hardware
- ONNX model execution time varies based on policy complexity

## Dependencies

All dependencies are managed through Pixi and specified in `pixi.toml` with version constraints.

### Core Dependencies

- **Python**: 3.11.x
- **C++ Build Tools**: CMake (3.24), Ninja, GCC/G++
- **C++ Libraries**: ONNX Runtime (>=1.15), Eigen (>=3.4), fmt (>=9.1), nlohmann_json (>=3.11)
- **Python Libraries**: PyTorch, onnxscript
- **Testing**: GoogleTest, pytest

See `pixi.toml` for complete dependency specifications with version ranges.

## Development

### Pre-commit Hooks

We use pre-commit hooks to ensure code quality. To set up pre-commit hooks:

```bash
# Install pre-commit hooks
pixi run -e python pre-commit install

# Run pre-commit on all files (optional)
pixi run -e python pre-commit run --all-files
```

### Available Tasks

Specified in `pixi.toml`:

**Python tasks** (use `-e core` environment):

- `pixi run -e core test`: Run Python tests with pytest
- `pixi run -e core lint-python`: Check Python code with ruff
- `pixi run -e core format-python`: Format Python code with ruff

**C++ tasks** (use `-e controller` environment):

- `pixi run -e controller configure`: Run CMake configuration
- `pixi run -e controller build`: Build the C++ library
- `pixi run -e controller test`: Run C++ tests with CTest
- `pixi run -e controller format-cpp`: Format C++ code with clang-format

## Maintenance and Support

This project is under **light maintenance**. No feature development is
guaranteed, but if you have bug reports and/or pull requests that fix bugs,
expect an RAI maintainer to respond within a few weeks.

## Contributing

We welcome your contributions!

To contribute:

- Report bugs and suggest improvements by opening an issue.
- Submit pull requests for code changes or documentation updates.
- Follow the project's code style and testing guidelines (see comments and existing code for reference).
- Ensure your changes pass all tests and pre-commit checks.

All contributions are reviewed by project maintainers before merging. Thank you for helping improve this project!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE)
file for details.

Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute

## Citation

If you use this work in your research or project, please consider citing it using the 'Cite this repository'
button in the sidebar, or using:

```bibtex
@misc{exploy2026,
  author = {Dario Bellicoso, Annika Wollschläger},
  title = {Exploy: EXport and dePLOY Reinforcement Learning policies},
  month = {March},
  year = {2026},
  url = {https://github.com/bdaiinstitute/exploy}
}
```
