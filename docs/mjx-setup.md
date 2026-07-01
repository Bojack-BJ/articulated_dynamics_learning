# MuJoCo MJX Setup

This branch is the starting point for migrating dynamics identification from black-box MuJoCo rollouts to differentiable MuJoCo MJX rollouts.

## Why MJX Here

The current `identify-dynamics` command uses C MuJoCo rollouts plus finite-difference gradients. That is a practical baseline, but it is still black-box optimization around the simulator.

MJX-JAX is the relevant next step when you want:

- end-to-end gradients through rollout
- vectorized batches of system-ID problems
- direct integration with JAX-based learned dynamics models

It is not automatically a universal upgrade. The official MuJoCo docs note that:

- MJX is distributed as `mujoco-mjx` and depends on `mujoco`
- MJX-JAX is the differentiable implementation
- MJX-Warp does **not** support autodiff
- MJX-JAX can be a poor fit for single-scene simulation compared with C MuJoCo

Sources:

- MuJoCo MJX docs: https://mujoco.readthedocs.io/en/latest/mjx.html
- JAX installation docs: https://docs.jax.dev/en/latest/installation.html

## Install

Install the new optional dependency group:

```bash
python -m pip install -e ".[simulation,mjx]"
```

That installs:

- `mujoco`
- `mujoco-mjx`
- `jax`

On Apple Silicon, the JAX docs currently describe Apple GPU support as experimental. CPU installation on macOS is straightforward with:

```bash
pip install --upgrade jax
```

If you specifically want Apple GPU execution, follow the current official JAX installation page rather than hardcoding an outdated plugin recipe here. In this project, the practical path is:

```bash
python -m pip install jax-metal
PYTHONPATH=src python -m rgbd_urdf_mvp probe-mjx --jax-platform metal --enable-pjrt-compatibility --no-rollout-test
```

The repository defaults MJX commands to `--jax-platform cpu` even when `jax-metal` is installed. This avoids accidental crashes in headless or sandboxed environments where the Apple GPU is not visible.

## Probe

Run:

```bash
PYTHONPATH=src python -m rgbd_urdf_mvp probe-mjx
```

This checks:

- `jax` import
- visible JAX devices
- `mujoco.mjx` import
- a tiny `mjx.step` smoke test

If you only want import/backend information:

```bash
PYTHONPATH=src python -m rgbd_urdf_mvp probe-mjx --no-rollout-test
PYTHONPATH=src python -m rgbd_urdf_mvp probe-mjx --jax-platform metal --enable-pjrt-compatibility --no-rollout-test
```

## Intended Next Step

After the probe is green, the repository now includes a parallel `identify-dynamics-mjx` path:

```text
articulation_artifact + MJCF + q(t)
  -> mjx.put_model
  -> differentiable rollout
  -> optimize mass / damping / friction with JAX grad
```

Run it with:

```bash
PYTHONPATH=src python -m rgbd_urdf_mvp identify-dynamics-mjx \
  outputs/recordings/$OBJECT_ID/episode.json \
  outputs/recordings/$OBJECT_ID/pointcloud_4d_partseg/inferred_articulation/articulation_artifact.json \
  outputs/recordings/$OBJECT_ID/pointcloud_4d_partseg/inferred_articulation/urdf/$OBJECT_ID.mjcf.xml \
  --jax-platform metal \
  --enable-pjrt-compatibility
```

The current implementation uses:

- `mjx.put_model` / `mjx.put_data`
- differentiable rollout through `mjx.step`
- JAX forward-mode autodiff via `jacfwd`
- Adam-style first-order updates on mass / damping / friction parameters

On Metal, the project passes `impl="jax"` and the active JAX device explicitly into `mjx.put_model` / `mjx.put_data`. This works around MuJoCo MJX's current auto-device resolver, which does not yet recognize the `METAL` backend.

It still lives alongside the MuJoCo finite-difference optimizer because:

- C MuJoCo is still the easier baseline for single-scene debugging
- the MJX path currently assumes a shared timestamp schedule
- contact-rich replay and control replay are not implemented yet
- some `jax-metal` / `jax` combinations still fail during device upload with `default_memory_space is not supported`; when that happens, fall back to `--jax-platform cpu` or use a dedicated Metal environment with a known-good pinned stack
