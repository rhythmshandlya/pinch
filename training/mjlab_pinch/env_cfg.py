"""The Pinch velocity task for mjlab: configuration (needs torch + mjlab).

Follows pollen-robotics/microduck_rl's velocity recipe (shared observation layout, domain randomisation incl.
delay, encoder bias and IMU misalignment, explicit zero and turn-in-place command buckets, reward rules in its
AGENTS.md) on top of mujocolab/mjlab's velocity task. Sized for a 20 cm crab: 50 Hz control, commands up to
6 cm/s, centimetre-scale foot clearance. See docs/dev/MJLAB.md for the choices and what is still open.

make_env_cfg(actuators="position") uses today's servo numbers as plain position actuators; "bam" uses the
Rhoban BAM measured model (voltage/torque/friction), which at the moment is the STS3215 7.4 V file: the robot
body has to be re-sized for that servo before the BAM variant means anything physically (see the doc).
"""

from __future__ import annotations

import math

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, ObjRef, RingPatternCfg, TerrainHeightSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp as vmdp
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.utils.spec_config import CollisionCfg
from mjlab.viewer import ViewerConfig

from training.mjlab_pinch import layout, mdp, robot_spec

PHYSICS_DT = 0.005
DECIMATION = 4                                  # 4 x 5 ms = 20 ms: 50 Hz control
EPISODE_S = 20.0
ACTION_DELAY_STEPS = (0, 3)                     # control steps (60 ms at most), applied as physics-step lag
BAM_MOTOR = ("feetech_sts3215_7_4V", "m6")

LEGS = SceneEntityCfg("robot", joint_names=(layout.LEG_REGEX,))
FEET = SceneEntityCfg("robot", geom_names=robot_spec.FOOT_GEOMS)


def robot_cfg(actuators: str = "position") -> EntityCfg:
    delay = dict(delay_min_lag=ACTION_DELAY_STEPS[0] * DECIMATION, delay_max_lag=ACTION_DELAY_STEPS[1] * DECIMATION,
                 delay_update_period=DECIMATION * 25)        # a new lag about every half second, per env
    arms = BuiltinPositionActuatorCfg(target_names_expr=(layout.ARM_REGEX,), damping=0.0,
                                      **robot_spec.arm_servo_numbers())
    if actuators == "position":
        legs = BuiltinPositionActuatorCfg(target_names_expr=(layout.LEG_REGEX,), damping=0.0,
                                          **robot_spec.leg_servo_numbers(), **delay)
    elif actuators == "bam":
        from bam.mjlab import BamActuatorCfg
        legs = BamActuatorCfg(target_names_expr=(layout.LEG_REGEX,), motor_name=BAM_MOTOR[0], model=BAM_MOTOR[1],
                              vin_range=(6.6, 8.2),             # a 2S pack from nearly flat to full, per env
                              friction_scale_range=(0.8, 1.2), **delay)
    else:
        raise ValueError(f"actuators must be 'position' or 'bam', not {actuators!r}")
    feet = CollisionCfg(geom_names_expr=(f"^({'|'.join(robot_spec.FOOT_GEOMS)})$",), contype=0, conaffinity=1,
                        condim=3, priority=1, friction=(0.8, 0.02, 0.001))
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, robot_spec.START_HEIGHT),
                                             joint_pos=robot_spec.init_joint_pos(), joint_vel={".*": 0.0}),
        spec_fn=robot_spec.spec,
        collisions=(feet,),
        articulation=EntityArticulationInfoCfg(actuators=(legs, arms), soft_joint_pos_limit_factor=0.95),
    )


def make_env_cfg(play: bool = False, actuators: str = "position") -> ManagerBasedRlEnvCfg:
    bam = actuators == "bam"

    # ---------------------------------------------------------------- sensors
    foot_contact = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(mode="geom", pattern=robot_spec.FOOT_GEOMS, entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"), reduce="netforce", num_slots=1, track_air_time=True)
    foot_height = TerrainHeightSensorCfg(
        name="foot_height_scan",
        frame=tuple(ObjRef(type="site", name=s, entity="robot") for s in robot_spec.FOOT_SITES),
        pattern=RingPatternCfg.single_ring(radius=0.01, num_samples=4),
        ray_alignment="yaw", max_distance=0.5, exclude_parent_body=True, include_geom_groups=(0,))

    # ---------------------------------------------------------------- observations
    # What the robot can sense, with a short history. No arm feedback (those servos report nothing).
    hist = dict(history_length=layout.HISTORY, flatten_history_dim=True)
    imu_delay = dict(delay_min_lag=0, delay_max_lag=1, delay_update_period=64)     # 0-20 ms of sensor latency
    actor_terms = {
        "gyro": ObservationTermCfg(func=mdp.imu_gyro, noise=Unoise(n_min=-0.05, n_max=0.05),
                                   scale=layout.GYRO_SCALE, **imu_delay, **hist),
        "gravity": ObservationTermCfg(func=mdp.imu_gravity, noise=Unoise(n_min=-0.02, n_max=0.02),
                                      **imu_delay, **hist),
        "leg_pos": ObservationTermCfg(func=envs_mdp.joint_pos_rel, params={"biased": True, "asset_cfg": LEGS},
                                      noise=Unoise(n_min=-0.005, n_max=0.005), **hist),
        "leg_vel": ObservationTermCfg(func=envs_mdp.joint_vel_rel, params={"asset_cfg": LEGS},
                                      noise=Unoise(n_min=-1.0, n_max=1.0), scale=layout.JOINT_VEL_SCALE, **hist),
        "last_action": ObservationTermCfg(func=envs_mdp.last_action, **hist),
        "command": ObservationTermCfg(func=mdp.command_scaled, params={"command_name": "twist"}),
    }
    critic_terms = {
        **{k: ObservationTermCfg(func=v.func, params=v.params, scale=v.scale, **hist) if k != "command" else v
           for k, v in actor_terms.items()},
        # privileged: what the sim knows and the robot does not
        "leg_pos": ObservationTermCfg(func=envs_mdp.joint_pos_rel, params={"asset_cfg": LEGS}, **hist),
        "base_lin_vel": ObservationTermCfg(func=envs_mdp.base_lin_vel),
        "foot_contact": ObservationTermCfg(func=vmdp.foot_contact, params={"sensor_name": foot_contact.name}),
        "foot_height": ObservationTermCfg(func=vmdp.foot_height, params={"sensor_name": foot_height.name}),
    }
    observations = {
        "actor": ObservationGroupCfg(terms=actor_terms, concatenate_terms=True, enable_corruption=True),
        "critic": ObservationGroupCfg(terms=critic_terms, concatenate_terms=True, enable_corruption=False),
    }

    # ---------------------------------------------------------------- actions
    actions = {"joint_pos": mdp.PinchLegActionCfg(entity_name="robot", actuator_names=(layout.LEG_REGEX,),
                                                  scale=layout.ACTION_SCALE, use_default_offset=True)}

    # ---------------------------------------------------------------- commands
    commands = {"twist": mdp.PinchVelocityCommandCfg(
        entity_name="robot", resampling_time_range=(3.0, 8.0), heading_command=False, debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(lin_vel_x=(-layout.CMD_MAX[0], layout.CMD_MAX[0]),
                                                    lin_vel_y=(-layout.CMD_MAX[1], layout.CMD_MAX[1]),
                                                    ang_vel_z=(-layout.CMD_MAX[2], layout.CMD_MAX[2])))}

    # ---------------------------------------------------------------- events (domain randomisation)
    body_all = SceneEntityCfg("robot", body_names=(".*",))
    events = {
        "reset_base": EventTermCfg(func=envs_mdp.reset_root_state_uniform, mode="reset", params={
            "pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (0.0, 0.004), "roll": (-0.05, 0.05),
                           "pitch": (-0.05, 0.05), "yaw": (-math.pi, math.pi)}, "velocity_range": {}}),
        "reset_leg_joints": EventTermCfg(func=envs_mdp.reset_joints_by_offset, mode="reset", params={
            "position_range": (-0.05, 0.05), "velocity_range": (0.0, 0.0), "asset_cfg": LEGS}),
        "imu_misalignment": EventTermCfg(func=mdp.randomize_imu_misalignment, mode="reset",
                                         params={"max_angle_deg": 5.0}),
        # a shove is a velocity kick: the walking speed is 6 cm/s, so 4 cm/s is a firm push
        "push_robot": EventTermCfg(func=envs_mdp.push_by_setting_velocity, mode="interval", interval_range_s=(1.5, 4.0),
                                   params={"velocity_range": {"x": (-0.04, 0.04), "y": (-0.04, 0.04),
                                                              "roll": (-0.2, 0.2), "pitch": (-0.2, 0.2),
                                                              "yaw": (-0.3, 0.3)}}),
        "foot_friction": EventTermCfg(mode="startup", func=dr.geom_friction, params={
            "asset_cfg": FEET, "operation": "abs", "ranges": (0.3, 1.25), "shared_random": True}),
        "body_mass": EventTermCfg(mode="startup", func=dr.pseudo_inertia, params={
            "asset_cfg": body_all, "alpha_range": (-0.07, 0.07)}),           # about +-15 % mass with its inertia
        "body_com": EventTermCfg(mode="startup", func=dr.body_com_offset, params={
            "asset_cfg": SceneEntityCfg("robot", body_names=(robot_spec.BODY,)), "operation": "add",
            "ranges": {0: (-0.005, 0.005), 1: (-0.005, 0.005), 2: (-0.004, 0.004)}}),
        "encoder_bias": EventTermCfg(mode="startup", func=dr.encoder_bias, params={
            "asset_cfg": LEGS, "bias_range": (-0.025, 0.025)}),                # about +-1.4 degrees
    }
    if bam:
        from bam.mjlab import bam_init
        events["bam_init"] = EventTermCfg(func=bam_init, mode="startup")      # per-env friction fields
    else:
        events["servo_gains"] = EventTermCfg(mode="startup", func=dr.pd_gains, params={
            "asset_cfg": LEGS, "kp_range": (0.75, 1.25), "kd_range": (1.0, 1.0), "operation": "scale"})
        events["motor_strength"] = EventTermCfg(mode="startup", func=dr.effort_limits, params={
            "asset_cfg": LEGS, "effort_limit_range": (0.65, 1.15), "operation": "scale"})
        events["joint_armature"] = EventTermCfg(mode="startup", func=dr.joint_armature, params={
            "asset_cfg": LEGS, "ranges": (0.8, 1.2), "operation": "scale"})
        events["joint_damping"] = EventTermCfg(mode="startup", func=dr.joint_damping, params={
            "asset_cfg": LEGS, "ranges": (0.85, 1.15), "operation": "scale"})

    # ---------------------------------------------------------------- rewards
    # Sizing: the robot walks at <= 6 cm/s, so the stock Gaussians (std ~0.3-0.5 m/s) would be flat; tracking
    # stds are about half the command maximum. Penalty weights are scaled so each is of the same order as the
    # tracking terms at this speed (mm and cm quantities square to tiny numbers). Penalty functions return >= 0
    # and take a NEGATIVE weight; check every Episode_Reward/<penalty> is <= 0 in the log.
    stand = {r".*_coxa": 0.05, r".*_femur": 0.08, r".*_tibia": 0.12}
    walk = {r".*_coxa": 0.30, r".*_femur": 0.40, r".*_tibia": 0.55}
    rewards = {
        "track_linear_velocity": RewardTermCfg(func=vmdp.track_linear_velocity, weight=3.0,
                                               params={"command_name": "twist", "std": 0.04}),
        "track_angular_velocity": RewardTermCfg(func=vmdp.track_angular_velocity, weight=2.0,
                                                params={"command_name": "twist", "std": 0.3}),
        "upright": RewardTermCfg(func=vmdp.upright, weight=1.0, params={
            "std": math.sqrt(0.05), "asset_cfg": SceneEntityCfg("robot", body_names=(robot_spec.BODY,))}),
        "pose": RewardTermCfg(func=vmdp.variable_posture, weight=1.0, params={
            "asset_cfg": LEGS, "command_name": "twist", "std_standing": stand, "std_walking": walk,
            "std_running": walk, "walking_threshold": 0.01, "running_threshold": 100.0}),
        "body_ang_vel": RewardTermCfg(func=vmdp.body_angular_velocity_penalty, weight=-0.3, params={
            "asset_cfg": SceneEntityCfg("robot", body_names=(robot_spec.BODY,))}),
        "dof_pos_limits": RewardTermCfg(func=envs_mdp.joint_pos_limits, weight=-1.0,
                                        params={"asset_cfg": LEGS}),
        "action_rate_l2": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.05),
        "air_time": RewardTermCfg(func=vmdp.feet_air_time, weight=3.0, params={
            "sensor_name": foot_contact.name, "threshold_min": 0.08, "threshold_max": 0.30,
            "command_name": "twist", "command_threshold": 0.01}),
        "foot_clearance": RewardTermCfg(func=vmdp.feet_clearance, weight=-5.0, params={
            "target_height": 0.012, "height_sensor_name": foot_height.name, "command_name": "twist",
            "command_threshold": 0.01, "asset_cfg": SceneEntityCfg("robot", site_names=robot_spec.FOOT_SITES)}),
        "foot_swing_height": RewardTermCfg(func=vmdp.feet_swing_height, weight=-0.25, params={
            "sensor_name": foot_contact.name, "height_sensor_name": foot_height.name, "target_height": 0.012,
            "command_name": "twist", "command_threshold": 0.01}),
        "foot_slip": RewardTermCfg(func=vmdp.feet_slip, weight=-3.0, params={
            "sensor_name": foot_contact.name, "command_name": "twist", "command_threshold": 0.01,
            "asset_cfg": SceneEntityCfg("robot", site_names=robot_spec.FOOT_SITES)}),
        "fell": RewardTermCfg(func=envs_mdp.is_terminated, weight=-5.0),
    }

    # ---------------------------------------------------------------- terminations
    terminations = {
        "time_out": TerminationTermCfg(func=vmdp.time_out, time_out=True),
        "fell_over": TerminationTermCfg(func=vmdp.bad_orientation, params={"limit_angle": math.radians(60.0)}),
        # only the feet collide, so a collapsed body would sink through the desk: stop it
        "body_too_low": TerminationTermCfg(func=envs_mdp.root_height_below_minimum, params={"minimum_height": 0.028}),
        "nan_state": TerminationTermCfg(func=envs_mdp.nan_detection),
    }

    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(terrain=TerrainEntityCfg(terrain_type="plane"), entities={"robot": robot_cfg(actuators)},
                       sensors=(foot_contact, foot_height), num_envs=1, env_spacing=0.6, extent=0.5),
        observations=observations, actions=actions, commands=commands, events=events, rewards=rewards,
        terminations=terminations,
        viewer=ViewerConfig(origin_type=ViewerConfig.OriginType.ASSET_BODY, entity_name="robot",
                            body_name=robot_spec.BODY, distance=0.5, elevation=-20.0, azimuth=90.0),
        sim=SimulationCfg(nconmax=None, njmax=300, mujoco=MujocoCfg(timestep=PHYSICS_DT, iterations=10,
                                                                     ls_iterations=20)),
        decimation=DECIMATION, episode_length_s=EPISODE_S)
    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
    return cfg


def make_rl_cfg() -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(hidden_dims=(256, 128, 64), activation="elu", obs_normalization=True,
                            distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.6,
                                              "std_type": "scalar"}),
        critic=RslRlModelCfg(hidden_dims=(256, 128, 64), activation="elu", obs_normalization=True),
        algorithm=RslRlPpoAlgorithmCfg(value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2,
                                       entropy_coef=0.01, num_learning_epochs=5, num_mini_batches=4,
                                       learning_rate=1.0e-3, schedule="adaptive", gamma=0.99, lam=0.95,
                                       desired_kl=0.01, max_grad_norm=1.0),
        experiment_name="pinch_velocity", save_interval=50, num_steps_per_env=12, max_iterations=1500)
