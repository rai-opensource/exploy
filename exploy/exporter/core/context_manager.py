# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from typing import Any

import numpy as np
import torch

from exploy.exporter.core.components import Connection, Group, Input, Memory, Output


class ContextManager:
    """Manages all components (inputs, outputs, memory) for an environment export."""

    def __init__(
        self,
    ):
        self._components: list[Input | Output | Connection] = []
        self._groups: list[Group] = []
        self._modules: list[torch.nn.Module] = []

    def add_component(self, component: Input | Output | Connection) -> None:
        """Add a component (Input, Output, or Connection) to the context manager.

        Args:
            component: The component to add. Can be an Input, Output, or Connection.

        Raises:
            AssertionError: If a component with the same name already exists.
        """
        assert isinstance(component, (Input, Output, Connection)), (
            "Component must be an Input, Output, or Connection."
        )
        if isinstance(component, Input):
            self.assert_unique_name(component.input_name)
            self.assert_unique_id(component.id, component.input_name)
        elif isinstance(component, Output):
            self.assert_unique_name(component.output_name)

        self._components.append(component)

    def add_components(self, components: list[Input | Output | Connection]) -> None:
        """Add multiple components to the context manager.

        Args:
            components: A list of components to add. Each can be an Input, Output, or Connection.

        Raises:
            AssertionError: If any component has a name that already exists.
        """
        for component in components:
            self.add_component(component)

    def add_group(self, group: Group) -> None:
        """Add a group of components to the context manager.

        Recursively adds all items in the group, including nested groups.

        Args:
            group: The group to add containing inputs, outputs, or nested groups.

        Raises:
            AssertionError: If a group or component with the same name already exists.
        """
        self.assert_unique_name(group.name)

        for item in group.items:
            if isinstance(item, Group):
                self.add_group(item)
            else:
                self.add_component(item)
        self._groups.append(group)

    def get_input_components(self) -> list[Input | Memory]:
        """Get all input components including memory components.

        Returns:
            A list of all Input and Memory components.
        """
        return [comp for comp in self._components if isinstance(comp, Input | Memory)]

    def get_connection_components(self) -> list[Connection]:
        """Get all connection components.

        Returns:
            A list of all Connection components.
        """
        return [comp for comp in self._components if isinstance(comp, Connection)]

    def get_output_components(self) -> list[Output | Memory]:
        """Get all output components including memory components.

        Returns:
            A list of all Output and Memory components.
        """
        return [comp for comp in self._components if isinstance(comp, Output | Memory)]

    def get_memory_components(self) -> list[Memory]:
        """Get all memory components.

        Returns:
            A list of all Memory components.
        """
        return [comp for comp in self._components if isinstance(comp, Memory)]

    def get_component_by_name(self, name: str) -> Input | Output | None:
        """Get a component by its name.

        Args:
            name: The name of the component to find.

        Returns:
            The component with the given name, or None if not found.
        """
        for component in self._components:
            if isinstance(component, Output) and component.output_name == name:
                return component
            if isinstance(component, Input) and component.input_name == name:
                return component
        return None

    def read_inputs(self) -> None:
        """Read and update all input components from the environment.

        Calls the read() method on each input component to refresh their data.
        """
        for component in self.get_input_components():
            component.read()

    def write_connections(self) -> None:
        """Write all connection components to transfer data.

        Calls the write() method on each connection component to transfer data
        from getters to setters.
        """
        for component in self.get_connection_components():
            component.write()

    def get_inputs(self, to_numpy: bool = False) -> dict[str, torch.Tensor] | dict[str, np.ndarray]:
        """Get all input data as a dictionary.

        Args:
            to_numpy: If True, return data as numpy arrays. If False, return as torch tensors.

        Returns:
            Dictionary mapping input names to their data (as torch.Tensor or np.ndarray).
        """
        inputs = {}
        for component in self.get_input_components():
            inputs[component.input_name] = (
                component.input_data_numpy if to_numpy else component.input_data
            )
        return inputs

    def get_input_names(self) -> list[str]:
        """Get the names of all input components.

        Returns:
            A list of input component names.
        """
        return [component.input_name for component in self.get_input_components()]

    def get_outputs(
        self, to_numpy: bool = False
    ) -> dict[str, torch.Tensor] | dict[str, np.ndarray]:
        """Get all output data as a dictionary.

        Args:
            to_numpy: If True, return data as numpy arrays. If False, return as torch tensors.

        Returns:
            Dictionary mapping output names to their values (as torch.Tensor or np.ndarray).
        """
        outputs = {}
        for component in self.get_output_components():
            outputs[component.output_name] = component.value_numpy if to_numpy else component.value
        return outputs

    def get_output_names(self) -> list[str]:
        """Get the names of all output components.

        Returns:
            A list of output component names.
        """
        return [output.output_name for output in self.get_output_components()]

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            **{
                comp.input_name: comp.metadata
                for comp in self._components
                if isinstance(comp, Input) and comp.metadata is not None
            },
            **{
                comp.output_name: comp.metadata
                for comp in self._components
                if isinstance(comp, Output) and comp.metadata is not None
            },
            **{group.name: group.metadata for group in self._groups if group.metadata is not None},
        }

    def assert_unique_name(self, name: str):
        """Assert that the given name is unique across all components and groups.

        Args:
            name: The name to check for uniqueness.

        Raises:
            KeyError: If the name already exists in groups, inputs, or outputs.
        """
        if name in [g.name for g in self._groups]:
            raise KeyError(f"Name '{name}' already exists as a group name.")
        if name in [comp.input_name for comp in self.get_input_components()]:
            raise KeyError(f"Name '{name}' already exists as an input name.")
        if name in [comp.output_name for comp in self.get_output_components()]:
            raise KeyError(f"Name '{name}' already exists as an output name.")

    def assert_unique_id(self, id: int, name: str = None):
        """Assert that the given id is unique across all components.

        Args:
            id: The id to check for uniqueness.

        Raises:
            KeyError: If the id already exists in any component.
        """
        for comp in self._components:
            if isinstance(comp, Input) and comp.id == id:
                raise KeyError(
                    f"ID '{id}' of {name} already exists in component {comp.input_name}."
                )

    def add_module(self, module: torch.nn.Module) -> None:
        """Register a PyTorch module with this context manager.

        This helper allows adding pretrained models as components of the context manager
        so they can be treated as submodules of the exporter and properly included when
        exporting to ONNX.

        Args:
            module: The PyTorch module to register.
        """
        for added_module in self._modules:
            if module is added_module:
                # Module is already registered, do nothing.
                return
        self._modules.append(module)

    @property
    def modules(self) -> tuple[torch.nn.Module, ...]:
        """Get all registered modules.

        Returns:
            A tuple of all registered torch modules.
        """
        return tuple(self._modules)
