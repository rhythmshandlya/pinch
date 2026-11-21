"""Pinch's own MDP pieces for mjlab (needs torch + mjlab): the leg action, IMU observations with a mounting
error, the bucketed velocity command. Everything else comes from mjlab's velocity task."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

from training.mjlab_pinch import layout

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_ROBOT = SceneEntityCfg("robot")


# ------------------------------------------------------------------ action
class PinchLegAction(JointPositionAction):
    """Position offsets around the stand pose for the 12 leg servos; the six arm servos hold the stand pose
    (their targets are rewritten every step because mjlab zeroes targets on reset)."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        arm_ids, _ = self._entity.find_joints(layout.ARM_REGEX)
        self._arm_ids = torch.tensor(arm_ids, device=self.device, dtype=torch.long)

    def apply_actions(self) -> None:
        super().apply_actions()
        hold = self._entity.data.default_joint_pos[:, self._arm_ids]
        self._entity.set_joint_position_target(hold, joint_ids=self._arm_ids)


@dataclass(kw_only=True)
class PinchLegActionCfg(JointPositionActionCfg):
    def build(self, env) -> PinchLegAction:
        return PinchLegAction(self, env)


# ------------------------------------------------------------------ IMU with a mounting error
def _imu_quat(env: ManagerBasedRlEnv) -> torch.Tensor:
    q = getattr(env, "_pinch_imu_quat", None)
    if q is None:
        q = torch.zeros(env.num_envs, 4, device=env.device)
        q[:, 0] = 1.0
        env._pinch_imu_quat = q
    return q


def randomize_imu_misalignment(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None, max_angle_deg: float) -> None:
    """A fresh IMU mounting error per episode: a random axis, an angle up to max_angle_deg. Not accumulating:
    each reset overwrites the previous draw."""
    q = _imu_quat(env)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    n = len(env_ids)
    axis = torch.randn(n, 3, device=env.device)
    axis = axis / axis.norm(dim=1, keepdim=True).clamp(min=1e-6)
    ang = torch.rand(n, device=env.device) * math.radians(max_angle_deg)
    q[env_ids, 0] = torch.cos(ang / 2)
    q[env_ids, 1:] = axis * torch.sin(ang / 2).unsqueeze(1)


def imu_gyro(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
    w = env.scene[asset_cfg.name].data.root_link_ang_vel_b
    return quat_apply_inverse(_imu_quat(env), w)


def imu_gravity(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
    g = env.scene[asset_cfg.name].data.projected_gravity_b
    return quat_apply_inverse(_imu_quat(env), g)


def command_scaled(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """The command in units of its maximum (layout.CMD_SCALE), so the policy's input is about -1..1."""
    cmd = env.command_manager.get_command(command_name)
    scale = torch.tensor(layout.CMD_SCALE, device=cmd.device, dtype=cmd.dtype)
    return cmd * scale


# ------------------------------------------------------------------ command with explicit buckets
class PinchVelocityCommand(UniformVelocityCommand):
    """vx, vy, yaw rate drawn from layout.BUCKETS: zero command, turn on the spot, forward only, sideways only
    and mixed (vy sampled exactly as densely as vx). Walking commands never sit near zero by chance."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.bucket = torch.zeros(self.num_envs, device=self.device)

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        u = torch.rand(len(env_ids), 4, device=self.device)
        vx, vy, wz, idx = layout.bucket_commands(u, cmd_max=self.cfg.cmd_max)
        self.vel_command_b[env_ids] = torch.stack([vx, vy, wz], dim=-1)
        self.vel_command_w[env_ids] = self.vel_command_b[env_ids]
        self.bucket[env_ids] = idx
        self.is_standing_env[env_ids] = idx == 0
        self.is_heading_env[env_ids] = False
        self.is_world_env[env_ids] = False
        self.is_forward_env[env_ids] = False


@dataclass(kw_only=True)
class PinchVelocityCommandCfg(UniformVelocityCommandCfg):
    cmd_max: tuple[float, float, float] = layout.CMD_MAX

    def build(self, env) -> PinchVelocityCommand:
        return PinchVelocityCommand(self, env)
