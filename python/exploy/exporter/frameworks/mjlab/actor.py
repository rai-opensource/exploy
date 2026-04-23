# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from __future__ import annotations

import torch
from rsl_rl.models import MLPModel, RNNModel
from tensordict import TensorDict

from exploy.exporter.core.actor import ExportableActor, add_actor_memory
from exploy.exporter.core.exportable_environment import ExportableEnvironment


class RslRlV5Actor(ExportableActor):
    """Wraps an RSL-RL v5 ``MLPModel`` (non-recurrent) for ONNX export.

    Converts the flat observation tensor into the ``TensorDict`` format that
    ``MLPModel.forward`` expects, using the model's own ``obs_groups`` list to
    know which keys are needed.
    """

    def __init__(self, model: MLPModel):
        super().__init__()
        self.model = model
        # obs_groups is the ordered list of TensorDict keys the model reads from.
        self._obs_groups: list[str] = model.obs_groups

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs_td = TensorDict(
            dict.fromkeys(self._obs_groups, obs),
            batch_size=obs.shape[:-1],
        )
        return self.model(obs_td)


class RslRlV5RecurrentActor(ExportableActor):
    """Wraps an RSL-RL v5 ``RNNModel`` (recurrent) for ONNX export.

    Same TensorDict wrapping as ``RslRlV5Actor`` but also exposes the RNN's
    hidden state via ``get_state()`` so that ``add_actor_memory`` can register
    it as an ONNX input/output.
    """

    def __init__(self, model: RNNModel):
        super().__init__()
        self.model = model
        self._obs_groups: list[str] = model.obs_groups

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs_td = TensorDict(
            dict.fromkeys(self._obs_groups, obs),
            batch_size=obs.shape[:-1],
        )
        return self.model(obs_td)

    def reset(self, dones: torch.Tensor) -> None:
        self.model.reset(dones)

    def get_state(self) -> tuple[torch.Tensor, ...] | None:
        hidden = self.model.get_hidden_state()
        if hidden is None:
            return None
        if isinstance(hidden, tuple):
            return hidden  # LSTM: (h, c)
        return (hidden,)  # GRU: h


def make_exportable_actor(
    env: ExportableEnvironment,
    actor_model: MLPModel | RNNModel,
    device: str,
) -> ExportableActor:
    """Create an exportable actor from an RSL-RL v5 ``MLPModel`` or ``RNNModel``.

    Args:
        env: The exportable environment that the actor will be used in.
        actor_model: The RSL-RL v5 actor model (``PPO.actor``).
        device: The device to place the actor on.
    """
    if type(actor_model) is MLPModel:
        actor = RslRlV5Actor(actor_model).to(device).eval()
    elif type(actor_model) is RNNModel:
        actor = RslRlV5RecurrentActor(actor_model).to(device).eval()

        # Call the actor once to initialize the hidden state.
        empty_obs = env.empty_actor_observations()
        actor(empty_obs)
    else:
        raise ValueError(f"Unsupported actor model type: {type(actor_model)}")

    # Add the actor inputs to the context.
    add_actor_memory(
        context_manager=env.context_manager(),
        get_hidden_states_func=actor.get_state,
    )

    return actor
