"""Export an RSL-RL checkpoint of a Pinch policy: ONNX with the observation normaliser baked in, plus plain numpy.

    python -m training.mjlab_pinch.export logs/rsl_rl/pinch_velocity/<run>/model_N.pt out_dir

Writes out_dir/policy.onnx (input "obs" 129, output "actions" 12; normaliser inside), out_dir/policy.npz
(np_policy.NpPolicy format) and out_dir/export.json (what it is and how well the three forward passes agree).
Fails unless torch, onnxruntime and numpy agree to 1e-4 on random observations drawn around the normaliser's
statistics. Needs torch, onnx and onnxruntime (the trainer's environment).
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TOL = 1e-4


def build_torch_policy(sd):
    """The deterministic actor as a torch module with the normaliser inside (what the ONNX file contains)."""
    import torch
    from torch import nn

    idx = sorted({int(k.split(".")[1]) for k in sd if k.startswith("mlp.") and k.endswith(".weight")})
    mods = []
    for n, i in enumerate(idx):
        w, b = sd[f"mlp.{i}.weight"], sd[f"mlp.{i}.bias"]
        lin = nn.Linear(w.shape[1], w.shape[0])
        lin.weight.data = torch.as_tensor(np.asarray(w, np.float32))
        lin.bias.data = torch.as_tensor(np.asarray(b, np.float32))
        mods.append(lin)
        if n < len(idx) - 1:
            mods.append(nn.ELU())

    class Policy(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("mean", torch.as_tensor(np.asarray(sd["obs_normalizer._mean"], np.float32).reshape(1, -1)))
            self.register_buffer("std", torch.as_tensor(np.asarray(sd["obs_normalizer._std"], np.float32).reshape(1, -1)))
            self.mlp = nn.Sequential(*mods)

        def forward(self, obs):
            return self.mlp((obs - self.mean) / (self.std + 1e-2))

    return Policy().eval()


def export(checkpoint, out_dir, task="Pinch-Velocity-Flat"):
    import onnxruntime as ort
    import torch

    from training.mjlab_pinch import layout
    from training.mjlab_pinch.np_policy import NpPolicy

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    sd = {k: v.detach().cpu().numpy() for k, v in ckpt["actor_state_dict"].items()}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = dict(task=task, checkpoint=Path(checkpoint).name, iteration=int(ckpt.get("iter", -1)),
                obs_size=layout.OBS_SIZE, obs_layout=layout.OBS_LAYOUT, history=layout.HISTORY,
                action_names=layout.LEG_JOINT_NAMES, action_scale=layout.ACTION_SCALE,
                default_leg_pos=layout.DEFAULT_LEG_POS.tolist(), ctrl_hz=layout.CTRL_HZ,
                cmd_max=layout.CMD_MAX, cmd_scale=layout.CMD_SCALE)
    npol = NpPolicy.from_state_dict(sd, meta=meta)
    assert npol.obs_size == layout.OBS_SIZE, (npol.obs_size, layout.OBS_SIZE)
    npol.save(out / "policy.npz")

    pol = build_torch_policy(sd)
    torch.onnx.export(pol, (torch.zeros(1, layout.OBS_SIZE),), str(out / "policy.onnx"), input_names=["obs"],
                      output_names=["actions"], opset_version=18, dynamo=False, dynamic_axes={})
    sess = ort.InferenceSession(str(out / "policy.onnx"), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    x = (npol.mean + rng.normal(0, 1, (256, layout.OBS_SIZE)) * npol.std).astype(np.float32)
    with torch.no_grad():
        a_torch = pol(torch.as_tensor(x)).numpy()
    a_onnx = np.concatenate([sess.run(None, {"obs": x[i:i + 1]})[0] for i in range(len(x))])
    a_np = npol(x)
    diff = dict(torch_vs_numpy=float(np.abs(a_torch - a_np).max()), torch_vs_onnx=float(np.abs(a_torch - a_onnx).max()))
    (out / "export.json").write_text(json.dumps(dict(meta=meta, max_abs_diff=diff), indent=1))
    print("exported", checkpoint, "->", out, diff)
    assert max(diff.values()) < TOL, diff
    return diff


if __name__ == "__main__":
    export(sys.argv[1], sys.argv[2], *sys.argv[3:4])
