"""Training ground for Pinch's walking: mjlab terrain families at a 15-20 cm robot's scale, with a curriculum.

    task "Pinch-Velocity-Rough" (training/mjlab_pinch/tasks.py): the velocity task on these terrains

Why these families and weights (docs/DECISIONS.md, phase 5): for small hobby-servo robots the actuator model
matters more than terrain variety (pollen-robotics/microduck_rl, which this task follows), so the terrain mix is
modest and indoor: flat ground with varied friction stays the biggest share; then the things a home floor or desk
actually has - rough floor, rug lips and thresholds (3-35 mm), gentle ramps (up to ~20 deg), undulating soft
ground, small things lying around. Sizes are scaled from the ANYmal-scale ranges of legged_gym (Rudin et al.
2021, arXiv 2109.11978) by Pinch's leg length (~1/8). Edges and drops are not here: walking off a desk is the
cliff reflex's and the mind's job (sim/worlds), not the gait's.

Curriculum: the game-inspired terrain curriculum of Rudin et al. 2021 - each family is a column, difficulty grows
down the rows; a robot that walked far enough goes one row harder, one that covered under half of what it was
commanded goes one easier, and one past the hardest row is sent to a random row. A recent comparison of 14
curricula on five quadrupeds with matched budgets (arXiv 2610.40364) found this success-driven curriculum in the
top group most often and uniform sampling never. mjlab's own terrain_levels_vel promotes after half a tile; at
Pinch's speeds (<= 6 cm/s, 20 s episodes: <= 1.2 m) that is the right size only if the tile is ~1.2 m, so the
tile is 1.2 m and the promotion threshold is min(0.8 x commanded distance, half a tile) (legged_gym's rule).

FAMILIES is plain data (no mjlab), so tests run without the trainer's environment.
"""

TILE = (1.2, 1.2)            # m: half a tile ~ what a Pinch walks in one 20 s episode at its top speed
ROWS = 10                    # difficulty levels; one column per family (mjlab's curriculum mode), and each
                             # family's proportion is its share of the robots
BORDER = 0.5
PROMOTE_FRACTION = 0.8

# name: (proportion, mjlab config class, its parameters; "_range" parameters grow with difficulty)
FAMILIES = {
    "flat":       (0.25, "BoxFlatTerrainCfg", {}),
    "rough":      (0.14, "HfRandomUniformTerrainCfg", dict(noise_range=(0.0, 0.008), noise_step=0.001,
                                                          horizontal_scale=0.02, vertical_scale=0.001,
                                                          scale_with_difficulty=True)),
    "slopes":     (0.10, "HfPyramidSlopedTerrainCfg", dict(slope_range=(0.0, 0.36), platform_width=0.3,
                                                          horizontal_scale=0.02, vertical_scale=0.001)),
    "valleys":    (0.05, "HfPyramidSlopedTerrainCfg", dict(slope_range=(0.0, 0.36), platform_width=0.3, inverted=True,
                                                          horizontal_scale=0.02, vertical_scale=0.001)),
    "thresholds": (0.12, "BoxPyramidStairsTerrainCfg", dict(step_height_range=(0.003, 0.035), step_width=0.15,
                                                           platform_width=0.3)),
    "rug_lips":   (0.12, "BoxRandomGridTerrainCfg", dict(grid_width=0.25, grid_height_range=(0.002, 0.015),
                                                        platform_width=0.3, border_width=0.05)),
    "soft_waves": (0.10, "HfWaveTerrainCfg", dict(amplitude_range=(0.0, 0.012), num_waves=4,
                                                  horizontal_scale=0.02, vertical_scale=0.001)),
    "clutter":    (0.12, "BoxRandomSpreadTerrainCfg", dict(num_boxes=25, box_width_range=(0.02, 0.08),
                                                          box_length_range=(0.02, 0.10), box_height_range=(0.003, 0.025),
                                                          platform_width=0.3, border_width=0.05)),
}


def generator_cfg(seed=0):
    """mjlab's TerrainGeneratorCfg for these families (needs mjlab)."""
    from mjlab import terrains as T
    subs = {}
    for name, (prop, cls, params) in FAMILIES.items():
        subs[name] = getattr(T, cls)(proportion=prop, size=TILE, **params)
    return T.TerrainGeneratorCfg(seed=seed, curriculum=True, size=TILE, border_width=BORDER, num_rows=ROWS,
                                 num_cols=len(FAMILIES), sub_terrains=subs, difficulty_range=(0.0, 1.0), color_scheme="height")


def terrain_levels_pinch(env, env_ids, command_name, asset_cfg=None):
    """Rudin et al.'s promotion rule: up a row after walking min(0.8 x commanded distance, half a tile), down a row
    after covering less than half of what was commanded; frozen on the first reset (mjlab's terrain_levels_vel)."""
    import torch
    from mjlab.managers.scene_entity_config import SceneEntityCfg
    asset = env.scene[(asset_cfg or SceneEntityCfg("robot")).name]
    terrain = env.scene.terrain
    gen = terrain.cfg.terrain_generator
    command = env.command_manager.get_command(command_name)
    walked = torch.norm(asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    commanded = torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s
    move_up = walked > torch.clamp(commanded * PROMOTE_FRACTION, max=gen.size[0] / 2)
    move_up &= commanded > 0.05                              # standing-still commands don't promote
    move_down = (walked < commanded * 0.5) & ~move_up
    if env.common_step_counter == 0:
        move_up, move_down = torch.zeros_like(move_up), torch.zeros_like(move_down)
    terrain.update_env_origins(env_ids, move_up, move_down)
    levels = terrain.terrain_levels.float()
    return {"mean": levels.mean()}


def make_rough_env_cfg(play=False, actuators="position", seed=0):
    """The Pinch velocity task (env_cfg.make_env_cfg) on the terrain families, with the curriculum."""
    from mjlab.managers.curriculum_manager import CurriculumTermCfg
    from mjlab.managers.termination_manager import TerminationTermCfg
    from mjlab.tasks.velocity import mdp as vmdp
    from mjlab.terrains import TerrainEntityCfg

    from training.mjlab_pinch.env_cfg import make_env_cfg
    cfg = make_env_cfg(play=play, actuators=actuators)
    cfg.scene.terrain = TerrainEntityCfg(terrain_type="generator", terrain_generator=generator_cfg(seed),
                                         max_init_terrain_level=2)
    cfg.curriculum = {"terrain_levels": CurriculumTermCfg(func=terrain_levels_pinch, params={"command_name": "twist"})}
    cfg.terminations["out_of_terrain_bounds"] = TerminationTermCfg(func=vmdp.out_of_terrain_bounds, time_out=True)
    if play:
        cfg.scene.terrain.max_init_terrain_level = None
        cfg.curriculum = {}
    return cfg
