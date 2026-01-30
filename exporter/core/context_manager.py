from typing import Any

import numpy as np
import torch

from .components import Group, Input, Memory, Output


class ContextManager:
    """Manages all components (inputs, outputs, memory) for an environment export."""

    def __init__(
        self,
    ):
        self._components: list[Input | Output] = []
        self._groups: list[Group] = []

    def add_component(self, component: Input | Output) -> None:
        self._components.append(component)

    def add_group(self, group: Group) -> None:
        for item in group.items:
            if isinstance(item, Group):
                self.add_group(item)
            else:
                self.add_component(item)
        self._groups.append(group)

    def get_input_components(self) -> list[Input]:
        return [comp for comp in self._components if isinstance(comp, Input | Memory)]

    def get_output_components(self) -> list[Output]:
        return [comp for comp in self._components if isinstance(comp, Output | Memory)]

    def get_memory_components(self) -> list[Memory]:
        return [comp for comp in self._components if isinstance(comp, Memory)]

    def get_component_by_name(self, name: str) -> Input | Output | None:
        for component in self._components:
            if isinstance(component, Output) and component.output_name == name:
                return component
            if isinstance(component, Input) and component.input_name == name:
                return component
        return None

    def read_inputs(self) -> None:
        for component in self.get_input_components():
            component.read()

    def write_inputs_to_env(self) -> None:
        for component in self.get_input_components():
            print(f"set input for name: {component.input_name}")
            component.write()

    def get_inputs(self, to_numpy: bool = False) -> dict[str, torch.Tensor] | dict[str, np.ndarray]:
        inputs = {}
        for component in self.get_input_components():
            inputs[component.input_name] = (
                component.input_data_numpy if to_numpy else component.input_data
            )
        return inputs

    def get_input_names(self) -> list[str]:
        return [component.input_name for component in self.get_input_components()]

    def get_outputs(self, to_numpy: bool = False) -> dict[str, torch.Tensor]:
        outputs = {}
        for component in self.get_output_components():
            outputs[component.output_name] = component.value_numpy if to_numpy else component.value
        return outputs

    def get_output_names(self) -> list[str]:
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
