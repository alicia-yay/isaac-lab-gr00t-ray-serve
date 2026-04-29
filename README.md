# GR00T humanoid robot, served by Ray Serve on Anyscale

End-to-end demo of NVIDIA's GR00T-N1.7-3B vision-language-action model controlling a Unitree G1 humanoid in NVIDIA Isaac Lab simulation, served at scale via Ray Serve on Anyscale.

## What this shows

Three things, end to end:

1. The GR00T-N1.7-3B foundation model deployed behind an HTTP endpoint with Ray Serve
2. The policy responding to observations with correctly-shaped action chunks at sub-second latency
3. A full Isaac Lab rollout rendered as a GIF, displayed inline in the notebook

## Architecture

![Architecture diagram](architecture.svg)

The notebook on the head node deploys the GR00T-N1.7-3B policy to a GPU worker via Ray Serve, then launches a separate Ray task on a different GPU worker that runs Isaac Lab and queries the policy over HTTP. The simulator's rendered frames return as a GIF that displays inline.

## Demo rollout

A zero-shot rollout from the GR00T-N1.7-3B base model. The robot is in Isaac Lab's pick-place scene, moving under real policy control.

![GR00T zero-shot rollout](g1_groot_n17_zeroshot.gif)

The motion is exploratory rather than task-completing because:

1. The rollout feeds zeroed joint state to the policy. The policy was trained to read its own arm and hand state. Wiring Isaac Lab's real `robot.data.joint_pos` into the GR00T schema makes the actions far more directed.
2. NVIDIA published `nvidia/GR00T-N1.6-G1-PnPAppleToPlate`, a checkpoint specifically post-trained on this exact pick-place task. That checkpoint produces a clean grasp out of the box and is available as Path B in this repo.

Both improvements land on the same Ray Serve infrastructure shown above. The model swap is a one-line change.

## Why this is interesting

Anyscale is the platform for running modern AI workloads at scale. Robotics foundation models like GR00T combine three hard things at once:

- **Heavy GPU inference** for a 3B parameter VLA (vision-language-action) model
- **Realtime physics simulation** with NVIDIA Isaac Lab and Isaac Sim
- **Multi-machine orchestration** with the policy and simulator on separate GPUs, communicating over HTTP

Ray Serve handles GR00T inference scaling. Ray tasks fan out the Isaac Lab simulators. Anyscale runs the cluster. The same primitives that scale LLM inference scale cleanly to robotics.

## Repo layout

```
groot_demo.ipynb                 The notebook. Open this and run top to bottom.

path_a_ray_serve/                Ray Serve HTTP architecture, GR00T-N1.7-3B base
  policy_server.py               Ray Serve deployment, FastAPI ingress
  sim_worker.py                  Isaac Lab subprocess that queries policy via HTTP
  g1_env.py                      Isaac Lab G1 wrapper, obs and action translation
  run_demo.py                    Orchestrator for parallel rollouts
  single_shot.py                 Self-contained end-to-end test

path_b_file_bridge/              File-bridge architecture, NVIDIA G1 fine-tune
  n16_inference_server.py        Loads GR00T-N1.6-G1-PnPAppleToPlate
  sim_runner_n16.py              Isaac Lab runner, file-bridged to inference
  orchestrate_n16.sh             Launches both processes on one GPU worker

demos/                           Pre-rendered demo GIFs
  g1_groot_n17_zeroshot.gif      N1.7-3B base, REAL_G1 embodiment
  g1_groot_n16_g1pnp.gif         N1.6 G1 fine-tune, UNITREE_G1 embodiment
  g1_groot_n16_polished.gif      Same as above, post-processed
  g1_groot_comparison.gif        Side-by-side N1.7 vs N1.6

docker/
  Dockerfile                     Cluster image: Isaac Sim 5.1 + Isaac Lab + GR00T

tools/                           Helpers (token distribution, polish scripts, smoke tests)
```

## Running the notebook

### Cluster requirements

- Anyscale workspace using the cluster image in `docker/Dockerfile`
- At least 2 GPU workers (A10G or better)
- A Hugging Face token with access to `nvidia/Cosmos-Reason2-2B` (gated; accept terms at https://huggingface.co/nvidia/Cosmos-Reason2-2B)

### Steps

1. Start the cluster with the custom image
2. Open `groot_demo.ipynb` in Anyscale's JupyterLab
3. Set `HF_TOKEN` in the Step 0 cell
4. Run cells top to bottom

Total runtime on a warm cluster: roughly 3 minutes.

## Path B: G1 pick-place fine-tune

Path B uses NVIDIA's published G1 pick-place fine-tune `nvidia/GR00T-N1.6-G1-PnPAppleToPlate` instead of the N1.7-3B base model. It runs in a dedicated `groot-n16` conda env because N1.6's pinned dependencies conflict with Isaac Sim 5.1's torch pin. The two processes communicate via pickle files in `/tmp/bridge/`.

```bash
bash path_b_file_bridge/orchestrate_n16.sh
```

## Required runtime patches

Working with current pip versions against pinned model expectations needs four patches. All are baked into `docker/Dockerfile`:

1. **VideoInput shim**: in transformers 4.54+, `VideoInput` was moved from `transformers.image_utils` to `transformers.video_utils`. GR00T's Eagle dynamic processor still imports from the old location.
2. **flash_attention_2 force**: Qwen3 VLM asserts `_attn_implementation == "flash_attention_2"` but `AutoModel.from_config` does not propagate the `attn_implementation` kwarg through. Patched via `_BaseAutoModelClass.from_config`.
3. **HF_TOKEN propagation**: Cosmos-Reason2-2B is gated. Worker subprocesses receive `HF_TOKEN` via Ray's `runtime_env={"env_vars": {"HF_TOKEN": ...}}`.
4. **Pinocchio pre-import**: NVIDIA IsaacLab issue #4090. Pinocchio's C++ `std::vector<std::string>` binding gets corrupted after Isaac Lab loads a robot URDF. Workaround: `import pinocchio` before `AppLauncher`.

## Embodiment and obs/action schemas

### N1.7 with `EmbodimentTag.REAL_G1`

Obs (nested dict):

- `video.ego_view`: `(B, 2, H, W, 3) uint8` (2-frame stack)
- `state`: 7 keys including `left_wrist_eef_9d` (3 pos + 6-element flattened rotation matrix), arms, hands, waist
- `language.annotation.human.task_description`: `[[str]]`

Action: 40-step chunk with 9 keys (left and right wrist, arms, hands, waist, base height, navigate command).

### N1.6 with `EmbodimentTag.UNITREE_G1`

Obs (nested dict):

- `video.ego_view`: `(B, 1, H, W, 3) uint8` (single frame)
- `state`: 7 full-body keys (`left_leg`, `right_leg`, `waist`, `left_arm`, `right_arm`, `left_hand`, `right_hand`)
- `language`: same as above

Action: 30-step chunk with 7 keys (upper body and waist, plus base_height_command, navigate_command).

### Isaac Lab task action mapping

`Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-v0` accepts actions of shape `(1, 28)`. The policy output packs as `left_arm(7) + right_arm(7) + left_hand(7) + right_hand(7) = 28`.

## Joint index mapping (Isaac Lab 43-DOF G1 to N1.6 schema)

```
left_leg [6]:   [0, 3, 6, 9, 13, 17]    hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
right_leg [6]:  [1, 4, 7, 10, 14, 18]
waist [3]:      [2, 5, 8]                yaw, roll, pitch
left_arm [7]:   [11, 15, 19, 21, 23, 25, 27]   shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw
right_arm [7]:  [12, 16, 20, 22, 24, 26, 28]
left_hand [7]:  [29, 35, 30, 36, 31, 37, 41]   index 0+1, middle 0+1, thumb 0+1+2
right_hand [7]: [32, 38, 33, 39, 34, 40, 42]
```

## Going further

### Scale the policy server horizontally

```python
deployment = GR00TPolicyServer.options(num_replicas=4).bind(...)
```

Ray Serve schedules each replica on its own GPU. Sim workers load-balance across them automatically.

### Run many sim rollouts in parallel

```python
results = ray.get([run_sim_rollout.remote(POLICY_URL) for _ in range(100)])
```

Each rollout grabs a GPU worker, queries the shared policy fleet, and saves its own GIF.

## Acknowledgments

- NVIDIA Isaac Lab team for the [pinocchio #4090 workaround](https://github.com/isaac-sim/IsaacLab/issues/4090)
- NVIDIA Isaac-GR00T `n1.6-release` and `main` branches
- Anyscale for the cluster
