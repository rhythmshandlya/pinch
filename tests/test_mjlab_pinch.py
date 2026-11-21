"""The mjlab port (training/mjlab_pinch): everything that can be checked without a GPU, torch or mjlab.

Robot spec and servo numbers against crab.xml, the observation layout and its history order, the command buckets,
the numpy policy round trip, and the sim-to-sim machinery on the CPU twin. The parts that need mjlab (the task
building, a smoke run) are exercised on the trainer by training/mjlab_pinch/check_env.py; the torch/ONNX export
round trip runs here too when torch is installed (it is on the trainer)."""

import re
import sys
import tempfile
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from training.mjlab_pinch import layout, robot_spec  # noqa: E402
from training.mjlab_pinch.np_policy import NpPolicy  # noqa: E402


def test_robot_spec_has_the_names_the_task_needs():
    m = robot_spec.compile_check()
    assert m.nu == 0 and m.nmesh == 0, "actuators come from the actuator configs, display meshes are dropped"
    joints = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(1, m.njnt)]
    legs = [j for j in joints if j in layout.LEG_JOINT_NAMES]
    assert legs == layout.LEG_JOINT_NAMES, "the model order must be the action order: " + str(legs)
    assert [j for j in joints if j in layout.ARM_JOINT_NAMES] == layout.ARM_JOINT_NAMES
    for n in robot_spec.FOOT_GEOMS:
        assert m.geom(n).type[0] == mujoco.mjtGeom.mjGEOM_SPHERE
    for n in (*robot_spec.FOOT_SITES, "imu"):
        m.site(n)
    m.body(robot_spec.BODY)
    assert re.compile(layout.LEG_REGEX).fullmatch("FL_coxa") and not re.compile(layout.LEG_REGEX).search("FL_coxa_backlash")
    assert all(re.compile(layout.ARM_REGEX).fullmatch(n) for n in layout.ARM_JOINT_NAMES)


def test_servo_numbers_are_the_models_own():
    """The plain position actuators use today's numbers: they must equal what sim/models/crab.xml was built with."""
    m = mujoco.MjModel.from_xml_path(str(robot_spec.MODEL))
    for names, nums in ((layout.LEG_JOINT_NAMES, robot_spec.leg_servo_numbers()),
                        (layout.ARM_JOINT_NAMES, robot_spec.arm_servo_numbers())):
        for n in names:
            a = m.actuator(n)
            assert abs(m.actuator_gainprm[a.id, 0] - nums["stiffness"]) < 1e-9, n
            assert abs(m.actuator_forcerange[a.id, 1] - nums["effort_limit"]) < 1e-9, n
            j = m.joint(n)
            assert abs(m.dof_armature[j.dofadr[0]] - nums["armature"]) < 1e-9, n
            assert abs(m.dof_damping[j.dofadr[0]] - nums["viscous_damping"]) < 1e-6, n
            assert abs(m.dof_frictionloss[j.dofadr[0]] - nums["frictionloss"]) < 1e-9, n


def test_stand_pose_is_the_projects():
    init = robot_spec.init_joint_pos()
    assert len(init) == 18 and all(k.startswith("^") and k.endswith("$") for k in init)
    m = mujoco.MjModel.from_xml_path(str(robot_spec.MODEL))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("stand").id)
    for n in layout.LEG_JOINT_NAMES + layout.ARM_JOINT_NAMES:
        assert abs(d.qpos[m.jnt_qposadr[m.joint(n).id]] - layout.STAND_POSE[n]) < 1e-9, n


def test_observation_layout_sizes_and_order():
    assert layout.OBS_FRAME == 42 and layout.OBS_SIZE == 42 * layout.HISTORY + 3 == 129
    assert sum(n for _, n in layout.OBS_LAYOUT) == layout.OBS_SIZE
    hist = layout.History()
    frames = []
    for t in range(4):
        fr = layout.frame(np.full(3, t + 1.0), np.full(3, 10.0 + t), np.full(12, 20.0 + t), np.full(12, 100.0 + t),
                          np.full(12, 40.0 + t))
        frames.append(fr)
        hist.push(fr)
    o = hist.obs((0.03, -0.06, 0.3))
    assert o.shape == (layout.OBS_SIZE,)
    # term-major, oldest first: the gyro term holds frames 1, 2, 3 (t = 1..3), scaled
    g = o[:9].reshape(3, 3)[:, 0]
    assert np.allclose(g, np.array([2.0, 3.0, 4.0]) * layout.GYRO_SCALE)
    gravity = o[9:18].reshape(3, 3)[:, 0]
    assert np.allclose(gravity, [11.0, 12.0, 13.0])
    vel = o[18 + 36:18 + 72].reshape(3, 12)[:, 0]
    assert np.allclose(vel, np.array([101.0, 102.0, 103.0]) * layout.JOINT_VEL_SCALE)
    assert np.allclose(o[-3:], np.array([0.03, -0.06, 0.3]) * layout.CMD_SCALE)
    assert np.allclose(o[-3:], [0.5, -1.0, 0.5])
    # backfill: the first frame fills every slot
    h2 = layout.History()
    h2.push(frames[0])
    o2 = h2.obs((0, 0, 0))
    assert np.allclose(o2[:9].reshape(3, 3), frames[0][:3])


def test_env_cfg_declares_the_layout():
    """mjlab cannot be imported here, so read the config's source: the actor terms are the layout's, in order,
    and history, action scale and the arm exclusion come from layout.py."""
    src = (ROOT / "training" / "mjlab_pinch" / "env_cfg.py").read_text(encoding="utf-8")
    block = src[src.index("actor_terms = {"):src.index("critic_terms")]
    names = re.findall(r'^\s{8}"(\w+)": ObservationTermCfg', block, re.M)
    assert names == [n for n, _ in layout.FRAME_TERMS] + ["command"], names
    assert "layout.HISTORY" in src and "layout.ACTION_SCALE" in src and "layout.LEG_REGEX" in src
    assert not re.search(r"shoulder|claw", block), "no arm feedback in the observation"
    assert "ACTION_DELAY_STEPS = (0, 3)" in src and "dr.encoder_bias" in src and "randomize_imu_misalignment" in src
    assert "DECIMATION = 4" in src and "PHYSICS_DT = 0.005" in src            # 50 Hz control


def test_command_buckets_cover_zero_turn_and_sideways():
    rng = np.random.default_rng(0)
    u = rng.random((40000, 4))
    vx, vy, wz, idx = layout.bucket_commands(u)
    frac = [float((idx == i).mean()) for i in range(5)]
    for got, want in zip(frac, layout.BUCKETS.values(), strict=True):
        assert abs(got - want) < 0.01, (frac, layout.BUCKETS)
    zero, turn, fwd, side = (idx == i for i in range(4))
    assert (vx[zero] == 0).all() and (vy[zero] == 0).all() and (wz[zero] == 0).all(), "an exact zero command"
    assert (vx[turn] == 0).all() and (vy[turn] == 0).all()
    assert (np.abs(wz[turn]) >= layout.TURN_MIN * layout.CMD_MAX[2] - 1e-9).all()
    assert (wz[turn] > 0).mean() > 0.4 and (wz[turn] < 0).mean() > 0.4, "both turning directions"
    assert (vy[fwd] == 0).all() and (wz[fwd] == 0).all() and (vx[fwd] != 0).all()
    assert (vx[side] == 0).all() and (np.abs(vy[side]) >= layout.MIN_LIN * layout.CMD_MAX[1] - 1e-9).all()
    assert abs(vx.max() - layout.CMD_MAX[0]) < 1e-3 and abs(vy.min() + layout.CMD_MAX[1]) < 1e-3
    # sideways as densely as forward: the same share of commands is a firm walk along each axis
    for thr in (0.01, 0.03, 0.05):
        assert abs((np.abs(vx) > thr).mean() - (np.abs(vy) > thr).mean()) < 0.01, thr
    assert abs(vx.mean()) < 1e-3 and abs(vy.mean()) < 1e-3 and abs(wz.mean()) < 0.01


def _random_state_dict(sizes, rng):
    sd = {"obs_normalizer._mean": rng.normal(0, 1, (1, sizes[0])).astype(np.float32),
          "obs_normalizer._std": rng.uniform(0.2, 2.0, (1, sizes[0])).astype(np.float32),
          "obs_normalizer.count": np.array(1000)}
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:], strict=True)):
        sd[f"mlp.{2 * i}.weight"] = (rng.normal(0, 1, (b, a)) / np.sqrt(a)).astype(np.float32)   # torch: (out, in)
        sd[f"mlp.{2 * i}.bias"] = rng.normal(0, 0.1, b).astype(np.float32)
    sd["distribution.std_param"] = np.full(sizes[-1], 0.5, np.float32)
    return sd


def _reference_forward(sd, obs, n_layers):
    """Written out independently of np_policy: normaliser, ELU hidden layers, linear output."""
    x = (obs - sd["obs_normalizer._mean"][0]) / (sd["obs_normalizer._std"][0] + 1e-2)
    for i in range(n_layers):
        x = sd[f"mlp.{2 * i}.weight"] @ x + sd[f"mlp.{2 * i}.bias"]
        if i < n_layers - 1:
            x = np.where(x > 0, x, np.exp(np.minimum(x, 0)) - 1)
    return x


def test_numpy_policy_round_trip_with_a_random_mlp():
    rng = np.random.default_rng(1)
    sizes = [layout.OBS_SIZE, 64, 32, 16, layout.N_LEG]
    sd = _random_state_dict(sizes, rng)
    pol = NpPolicy.from_state_dict(sd, meta={"task": "t", "obs_size": layout.OBS_SIZE})
    assert pol.obs_size == layout.OBS_SIZE and pol.action_size == layout.N_LEG
    with tempfile.TemporaryDirectory() as td:
        pol.save(Path(td) / "p.npz")
        back = NpPolicy.load(Path(td) / "p.npz")
    assert back.meta == {"task": "t", "obs_size": layout.OBS_SIZE}
    for _ in range(20):
        obs = (sd["obs_normalizer._mean"][0] + rng.normal(0, 1, layout.OBS_SIZE) * sd["obs_normalizer._std"][0]).astype(np.float32)
        want = _reference_forward(sd, obs, 4)
        assert np.abs(pol(obs) - want).max() < 1e-4
        assert np.abs(back(obs) - want).max() < 1e-4
    batch = rng.normal(0, 1, (7, layout.OBS_SIZE)).astype(np.float32)
    assert pol(batch).shape == (7, layout.N_LEG)


def test_torch_onnx_export_matches_numpy_when_torch_is_installed():
    try:
        import onnxruntime as ort
        import torch
    except ImportError:
        print("    (torch/onnxruntime not installed here: this check runs on the trainer)")
        return
    from training.mjlab_pinch import export
    rng = np.random.default_rng(2)
    sd = _random_state_dict([layout.OBS_SIZE, 48, 24, layout.N_LEG], rng)
    pol = export.build_torch_policy(sd)
    npol = NpPolicy.from_state_dict(sd)
    x = (npol.mean + rng.normal(0, 1, (32, layout.OBS_SIZE)) * npol.std).astype(np.float32)
    with torch.no_grad():
        a_torch = pol(torch.as_tensor(x)).numpy()
    assert np.abs(a_torch - npol(x)).max() < 1e-4
    with tempfile.TemporaryDirectory() as td:
        f = str(Path(td) / "p.onnx")
        torch.onnx.export(pol, (torch.zeros(1, layout.OBS_SIZE),), f, input_names=["obs"], output_names=["actions"],
                          opset_version=18, dynamo=False)
        sess = ort.InferenceSession(f, providers=["CPUExecutionProvider"])
        a_onnx = np.concatenate([sess.run(None, {"obs": x[i:i + 1]})[0] for i in range(len(x))])
    assert np.abs(a_onnx - a_torch).max() < 1e-4


def test_sim_replay_runs_on_the_twin_and_a_zero_policy_stands():
    """A policy that always outputs zero holds the stand pose: the replay machinery (observation, targets, falls)
    works on the CPU twin and the stand pose is stable for 3 s."""
    from training.mjlab_pinch import sim_replay
    zero = NpPolicy(np.zeros(layout.OBS_SIZE), np.ones(layout.OBS_SIZE), [(np.zeros((layout.OBS_SIZE, layout.N_LEG)),
                                                                          np.zeros(layout.N_LEG))])
    twin = sim_replay.Twin()
    r = sim_replay.rollout(twin, zero, (0.0, 0.0, 0.0), seconds=3.0, seed=0)
    assert not r["fell"] and r["moved"] < 0.01 and r["height"] > 0.035, r
