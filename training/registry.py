"""Every training environment, registered once (docs/ARCHITECTURE.md section 3).

The trainer, the CLI choices, `ops` and the tests all read this table; there is no second list. Importing it is
cheap: factories and randomisers are resolved only when a run starts (JAX is not needed to list environments).

    from training.registry import ENVS, make_env, randomizer
    env = make_env("terrain_push")
"""

import importlib
from dataclasses import dataclass, field


@dataclass(frozen=True)
class EnvSpec:
    name: str
    skill: str                      # the policy slot it trains for: gait, emote, righting, grasp, push
    factory: str                    # "module:callable"
    kwargs: dict = field(default_factory=dict)
    randomize: str = "training.robot_model:domain_randomize"
    covers: tuple = ()              # extra behaviours the policy is trained on (written to config.json)
    description: str = ""
    retired: bool = False           # kept so old runs stay reproducible; not offered for new experiments


def _resolve(path):
    mod, attr = path.split(":")
    return getattr(importlib.import_module(mod), attr)


_TERRAIN = "training.terrain_env:terrain_randomize"
_WALK_PUSH = dict(refs="walk_v2+push", push_vel=0.12, push_interval_s=2.5)

ENVS = {s.name: s for s in [
    EnvSpec("gait", "gait", "training.gait_env:CrabGaitEnv",
            description="residual walking on the scripted trot (decision 50)"),
    EnvSpec("gait2", "gait", "training.gait_env:CrabGaitEnv", dict(vel_alpha=0.06, vel_sigma=0.03),
            description="walking with a step-cycle velocity filter and a kinder tolerance", retired=True),
    EnvSpec("gait3", "gait", "training.gait_env:CrabGaitEnv", dict(push_vel=0.12, push_interval_s=2.5),
            description="gait-v1 rewards with bench-strength shoves", retired=True),
    EnvSpec("gait_v2", "gait", "training.gait_env:CrabGaitEnv", _WALK_PUSH, covers=("walk", "push"),
            description="48 walking commands plus crouched pushing references (decision 84)"),
    EnvSpec("terrain", "gait", "training.terrain_env:CrabTerrainEnv", randomize=_TERRAIN,
            description="hurdles, ramps and steps per robot (decision 58)"),
    EnvSpec("terrain_hard", "gait", "training.terrain_env:CrabTerrainEnv", dict(hard=1.5), randomize=_TERRAIN,
            description="terrain with half the robots on bars to 15 mm and ramps to 21 degrees"),
    EnvSpec("terrain_push", "gait", "training.terrain_env:CrabTerrainEnv", _WALK_PUSH, randomize=_TERRAIN,
            covers=("walk", "push"), description="one walker for hurdles, ramps, 48 commands and pushing"),
    EnvSpec("push2", "push", "training.push2_env:CrabPush2Env", randomize="training.push2_env:push2_randomize",
            description="learned push steering over the JAX gait (decision 67; failed, decision 88)"),
    EnvSpec("emote", "emote", "training.emote_env:CrabEmoteEnv", description="balance while emoting (4 ms physics)"),
    EnvSpec("emote2", "emote", "training.emote_env:CrabEmoteEnv", dict(sim_dt=0.002),
            description="emote balance in 2 ms physics, like the twin"),
    EnvSpec("righting", "righting", "training.righting_env:CrabRightingEnv",
            description="get up off its back with all 18 joints (decision 76)"),
    EnvSpec("righting_smooth", "righting", "training.righting_env:CrabRightingEnv",
            dict(reward_scales=dict(action_rate=-0.05)), description="get-up with 5x the action-rate cost"),
    EnvSpec("righting2", "righting", "training.righting_env:CrabRightingEnv",
            dict(reward_scales=dict(energy=0.0, action_rate=-0.005)), description="get-up without the energy cost"),
    EnvSpec("grasp", "grasp", "training.grasp_env:CrabGraspEnv", randomize="training.grasp_env:grasp_randomize",
            description="close the claw on a desk thing and lift it"),
]}


def spec(name):
    if name not in ENVS:
        raise ValueError(f"unknown env {name!r}; known: {', '.join(sorted(ENVS))}")
    return ENVS[name]


def _kwargs(s):
    kw = dict(s.kwargs)
    if kw.get("refs") == "walk_v2+push":
        from training.gait_env import GAIT_CMDS_V2, concat_references, push_references, walk_references
        kw["refs"] = concat_references([walk_references(GAIT_CMDS_V2), push_references()])
    if "hard" in kw:                 # terrain difficulty is a module setting in terrain_env
        import training.terrain_env as te
        te.HARD = kw.pop("hard")
    return kw


def make_env(name):
    s = spec(name)
    env = _resolve(s.factory)(**_kwargs(s))
    if s.covers:
        env.covers = list(s.covers)
    return env


def randomizer(name):
    return _resolve(spec(name).randomize)
