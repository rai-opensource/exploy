import torch
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent
from rsl_rl.modules.actor_critic_recurrent import Memory

from exploy.exporter.core.actor import ExportableActor, add_actor_memory
from exploy.exporter.core.exportable_environment import ExportableEnvironment


class RslRlRecurrentActor(ExportableActor):
    """An actor that wraps an RslRl ActorCriticRecurrent and adds its memory as inputs to the context."""

    def __init__(self, actor_critic: ActorCriticRecurrent):
        super().__init__()
        self.memory: Memory = actor_critic.memory_a
        self.actor = actor_critic.actor
        self.normalizer = actor_critic.actor_obs_normalizer

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        norm_obs = self.normalizer(obs)
        memory_out = self.memory(norm_obs).squeeze(dim=0)
        return self.actor(memory_out)

    def reset(self, dones: torch.Tensor):
        self.memory.reset(dones)

    def get_state(self) -> tuple[torch.Tensor, ...] | None:
        return self.memory.hidden_states


class RslRlActor(ExportableActor):
    """An actor that wraps an RslRl ActorCritic and normalizes its observations."""

    def __init__(self, actor_critic: ActorCritic):
        super().__init__()
        self.actor = actor_critic.actor
        self.normalizer = actor_critic.actor_obs_normalizer

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        norm_obs = self.normalizer(obs)
        return self.actor(norm_obs)


def make_exportable_actor(
    env: ExportableEnvironment,
    actor_critic: ActorCritic | ActorCriticRecurrent,
    device: str,
) -> ExportableActor:
    """Create an exportable actor from an RslRl ActorCritic or ActorCriticRecurrent.

    Args:
        env: The exportable environment that the actor will be used in.
        actor_critic: The RslRl ActorCritic or ActorCriticRecurrent to wrap.
        device: The device to put the actor on.
    """
    if type(actor_critic) is ActorCritic:
        actor = RslRlActor(actor_critic).to(device).eval()
    elif type(actor_critic) is ActorCriticRecurrent:
        actor = RslRlRecurrentActor(actor_critic).to(device).eval()

        # Call the actor once to initialize its hidden state.
        empty_obs = env.empty_actor_observations()
        actor(empty_obs)
    else:
        raise ValueError(f"Unsupported actor critic type: {type(actor_critic)}")

    # Add the actor inputs to the context.
    add_actor_memory(
        context_manager=env.context_manager(),
        get_hidden_states_func=actor.get_state,
    )

    return actor
