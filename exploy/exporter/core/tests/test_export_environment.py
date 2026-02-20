# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import pathlib
import tempfile

import torch

from exploy.exporter.core.context_manager import Input, Output
from exploy.exporter.core.evaluator import evaluate
from exploy.exporter.core.exportable_environment import ExportableEnvironment
from exploy.exporter.core.exporter import export_environment_as_onnx
from exploy.exporter.core.session_wrapper import SessionWrapper


class DataSource:
    """A simple data source that provides three tensors to compute observations for an environment."""

    def __init__(self):
        self.foo = torch.Tensor([[1.0, 2.0, 3.0, 4.0]])
        self.bar = torch.Tensor([[0.5, 0.6]])
        self.baz = torch.Tensor([[-7.0, -8.0]])


class Env(ExportableEnvironment):
    """A simple environment that concatenates three tensors from a data source to compute
    observations.
    """

    def __init__(self, data_source: DataSource):
        super().__init__()

        self._data_source = data_source
        self._decimation = 4
        self._actions = torch.zeros(1, 2)
        self._processed_action = torch.zeros_like(self._actions)
        self._output = torch.zeros(1, 3)

    @property
    def num_act(self) -> int:
        return self._actions.shape[-1]

    @property
    def num_obs(self) -> int:
        return self.compute_observations().shape[-1]

    @property
    def data_source(self):
        return self._data_source

    def compute_observations(self) -> torch.Tensor:
        return torch.cat(
            [
                self.data_source.foo,
                self.data_source.bar,
                self.data_source.baz,
            ],
            dim=-1,
        )

    def process_actions(self, actions: torch.Tensor):
        self._processed_action = 3 * actions

    def apply_actions(self):
        self._output = self._processed_action + 2

    def prepare_export(self):
        pass

    def empty_actor_observations(self) -> torch.Tensor:
        return torch.zeros_like(self.compute_observations())

    def empty_actions(self) -> torch.Tensor:
        return torch.zeros_like(self._actions)

    def metadata(self):
        return {"env_name": "Env", "version": "1.0"}

    @property
    def decimation(self) -> int:
        return self._decimation

    def register_evaluation_hooks(self, update, reset, evaluate_substep):
        pass

    def step(self, actions: torch.Tensor):
        return self.compute_observations(), False

    def get_observation_names(self):
        return ["obs1", "obs2", "obs3"]

    def observations_reset(self):
        return self.compute_observations()


class EnvWithTorchModule(Env):
    """A simple environment that uses a torch Module to compute observations. The module is added
    to the export context, so it should be included in the exported ONNX graph.
    """

    def __init__(
        self,
        data_source: DataSource,
    ):
        module_dim_in = data_source.foo.shape[-1]
        module_dim_out = module_dim_in
        self._module = torch.nn.Linear(module_dim_in, module_dim_out)
        super().__init__(data_source=data_source)

    @property
    def module(self) -> torch.nn.Module:
        return self._module

    def compute_observations(self) -> torch.Tensor:
        return torch.cat(
            [
                self._module(self.data_source.foo),
                self.data_source.bar,
                self.data_source.baz,
            ],
            dim=-1,
        )


class Actor(torch.nn.Module):
    def __init__(self, num_obs: int, num_act: int):
        super().__init__()
        self._net = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=num_obs,
                out_features=num_act,
            ),
            torch.nn.ELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._net(x)


def export_and_evaluate_env(
    env: Env,
    actor: torch.nn.Module,
    onnx_file_name: str,
    num_eval_steps: int,
) -> bool:
    """Helper function to export an environment and evaluate it using the exported ONNX graph."""
    env.context_manager().add_components(
        [
            Input(
                name="foo",
                get_from_env_cb=lambda: env.data_source.foo,
            ),
            Input(
                name="bar",
                get_from_env_cb=lambda: env.data_source.bar,
            ),
            Input(
                name="baz",
                get_from_env_cb=lambda: env.data_source.baz,
            ),
            Output(
                name="out",
                get_from_env_cb=lambda: env._output,
            ),
        ]
    )

    # Export to ONNX.
    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = pathlib.Path(tmpdir) / "exploy"
        export_environment_as_onnx(
            env=env,
            actor=actor,
            path=onnx_path,
            filename=onnx_file_name,
            verbose=False,
        )

        # Make a session wrapper.
        session_wrapper = SessionWrapper(
            onnx_folder=onnx_path,
            onnx_file_name=onnx_file_name,
            policy=actor,
            optimize=True,
        )

        # Evaluate.
        evaluate_steps = num_eval_steps
        with torch.inference_mode():
            export_ok, _ = evaluate(
                env=env,
                context_manager=env.context_manager(),
                session_wrapper=session_wrapper,
                num_steps=evaluate_steps,
                verbose=False,
                pause_on_failure=False,
            )

    return export_ok


class TestExportableEnvironment:
    def test_env(self):
        """Test exporting an environment."""
        data_source = DataSource()
        env = Env(data_source=data_source)
        actor = Actor(num_obs=env.num_obs, num_act=env.num_act).eval()

        export_ok = export_and_evaluate_env(
            env=env,
            actor=actor,
            onnx_file_name="test_export_env.onnx",
            num_eval_steps=20,
        )
        assert export_ok, "ONNX export validation failed"

    def test_env_with_module(self):
        """Test exporting an environment that uses a torch Module
        to compute observations."""
        data_source = DataSource()
        env = EnvWithTorchModule(data_source=data_source)
        actor = Actor(num_obs=env.num_obs, num_act=env.num_act)
        env.context_manager().add_module(env.module)

        export_ok = export_and_evaluate_env(
            env=env,
            actor=actor,
            onnx_file_name="test_export_env_with_module.onnx",
            num_eval_steps=20,
        )
        assert export_ok, "ONNX export validation failed"
