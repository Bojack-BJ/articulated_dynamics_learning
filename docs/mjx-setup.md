# MuJoCo MJX Setup

This branch is the starting point for migrating dynamics identification from
black-box MuJoCo rollouts to differentiable MuJoCo MJX rollouts.

## Why MJX Here

The current `identify-dynamics` command uses C MuJoCo rollouts plus
finite-difference gradients. That is a practical baseline, but it is still
black-box optimization around the simulator.

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

On Apple Silicon, the JAX docs currently describe Apple GPU support as
experimental. CPU installation on macOS is straightforward with:

```bash
pip install --upgrade jax
```

If you specifically want Apple GPU execution, follow the current official JAX
installation page rather than hardcoding an outdated plugin recipe here.

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
```

## Intended Next Step

After the probe is green, the next implementation target should be a parallel
`identify-dynamics-mjx` path:

```text
articulation_artifact + MJCF + q(t)
  -> mjx.put_model
  -> differentiable rollout
  -> optimize mass / damping / friction with JAX grad
```

That path should live alongside the current MuJoCo finite-difference optimizer
until feature parity and numerical behavior are good enough to replace it.
