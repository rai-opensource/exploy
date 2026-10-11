# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from __future__ import annotations

import torch
from rsl_rl.models import MLPModel, RNNModel

from exploy.exporter.core.actor import ExportableActor, add_actor_memory
from exploy.exporter.core.exportable_environment import ExportableEnvironment


def _check_obs_groups(model: MLPModel):
    if list(model.obs_groups) != ["policy"]:
        raise ValueError(
            "Only actors observing the `policy` observation group can be exported, got observation groups:"
            f" {model.obs_groups}"
        )


def _deterministic_output(model: MLPModel, mlp_output: torch.Tensor) -> torch.Tensor:
    return (
        model.distribution.deterministic_output(mlp_output)
        if model.distribution is not None
        else mlp_output
    )


class RslRlActor(ExportableActor):
    """An actor that wraps an RSL-RL ``MLPModel``: normalizes the policy observations and outputs the
    deterministic actions."""

    def __init__(self, model: MLPModel):
        super().__init__()
        _check_obs_groups(model)
        self.model = model

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        latent = self.model.obs_normalizer(obs)
        return _deterministic_output(self.model, self.model.mlp(latent))


class RslRlRecurrentActor(ExportableActor):
    """An actor that wraps an RSL-RL ``RNNModel`` and exposes its hidden state, which is added as memory
    to the context."""

    def __init__(self, model: RNNModel):
        super().__init__()
        _check_obs_groups(model)
        self.model = model

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        latent = self.model.obs_normalizer(obs)
        latent = self.model.rnn(latent).squeeze(dim=0)
        return _deterministic_output(self.model, self.model.mlp(latent))

    def reset(self, dones: torch.Tensor):
        self.model.reset(dones)

    def get_state(self) -> tuple[torch.Tensor, ...] | None:
        hidden_state = self.model.get_hidden_state()
        if hidden_state is None or isinstance(hidden_state, tuple):
            return hidden_state
        # GRUs have a single hidden state tensor.
        return (hidden_state,)


def make_exportable_actor(
    env: ExportableEnvironment,
    actor: MLPModel,
    device: str,
) -> ExportableActor:
    """Create an exportable actor from an RSL-RL actor model (e.g. ``runner.alg.actor``).

    Args:
        env: The exportable environment that the actor will be used in.
        actor: The RSL-RL ``MLPModel`` or ``RNNModel`` to wrap.
        device: The device to put the actor on.
    """
    if type(actor) is MLPModel:
        exportable_actor = RslRlActor(actor).to(device).eval()
    elif type(actor) is RNNModel:
        exportable_actor = RslRlRecurrentActor(actor).to(device).eval()

        # Call the actor once to initialize its hidden state.
        exportable_actor(env.empty_actor_observations())
    else:
        raise ValueError(f"Unsupported actor type: {type(actor)}")

    # Add the actor inputs to the context.
    add_actor_memory(
        context_manager=env.context_manager(),
        get_hidden_states_func=exportable_actor.get_state,
    )

    return exportable_actor
