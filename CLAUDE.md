# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

AWEbox is a Python toolbox for modelling and optimal control of rigid-wing, lift-/drag-mode
multiple-kite Airborne Wind Energy (AWE) systems. Given a system architecture and a set of options,
it symbolically builds system dynamics (via CasADi), formulates a direct-collocation/multiple-shooting
optimal control problem, solves it with IPOPT via a homotopy continuation strategy, and provides
post-processing, visualization, and quality-check tooling. It also supports offline MPC simulation
and "SAM" (Stroboscopic Averaging Method) trajectories for long-horizon, multi-cycle problems.

## Setup

```bash
pip3 install -e .          # or: python3 setup.py install
```

Depends heavily on CasADi (pinned to `casadi==3.6.4`, see `requirements.txt`/`setup.py`) which
interfaces the IPOPT NLP solver. HSL linear solvers (e.g. `ma27`, `ma57`) are optional but strongly
recommended for performance — see `INSTALLATION.md`. Many example/experiment scripts set
`options['solver.linear_solver'] = 'ma27'`, which requires HSL to be installed and visible to CasADi.

## Common commands

Run an example from the repo root (not from inside `examples/`):

```bash
python3 examples/ampyx_ap2_trajectory.py
```

Run the test suite (from `test/`, as per CI):

```bash
cd test/
python3 -m pytest units      # fast unit tests — this is what CI runs
python3 -m pytest reg        # regression tests (slower, solves real trajectories)
python3 -m pytest int        # integration tests (save/load/serialization/visualization)
python3 -m pytest trials     # trial-level tests
python3 -m pytest units/test_model.py::test_architecture   # single test
```

Lint (per CONTRIBUTING.md):

```bash
pylint -d w -d c -d r awebox
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
```

Generate Sphinx docs:

```bash
cd docs/
sphinx-apidoc -f -o source ../awebox
make html
```
Do not commit the auto-generated `*.rst` files or `docs/build/`.

## Architecture

The public API is re-exported from `awebox/__init__.py` (imported as `import awebox as awe`):
`awe.Trial`, `awe.Sweep`, `awe.Options`, `awe.Visualization`, and kite parameter presets under
`awe.ampyx_data` / `awe.megawes_data` / `awe.boeing747_data` / `awe.bubbledancer_data`.

### The core workflow

1. Build an `options` dict (dotted-path keys like `'user_options.system_model.architecture'`,
   `'nlp.n_k'`, `'solver.linear_solver'`) — defaults live in `awebox/opts/default.py`, with
   `awebox/opts/options.py` handling seeding/merging (`Options.fill_in_seed`).
2. `trial = awe.Trial(options, 'trial_name')` — a `Trial` (`awebox/trial.py`) can also be
   constructed from a previously-saved `.dict` file or an in-memory seed dict (for warmstarting/
   reloading), see `Trial.__init__`.
3. `trial.build()` constructs, in order: the system `Architecture` (`awebox/mdl/architecture.py`,
   tree-structured parent-map of kite nodes), the `Model` (`awebox/mdl/model.py` — dynamics,
   aerodynamics under `awebox/mdl/aero/`, DAE formulation), the `Formulation`
   (`awebox/ocp/formulation.py`), and the `NLP` (`awebox/ocp/nlp.py` — discretization via
   collocation or multiple shooting, `awebox/ocp/discretization.py`).
4. `trial.optimize()` runs `awebox/opti/optimization.py`, which drives IPOPT through a homotopy
   continuation ("hippo") over relaxed sub-problems until the final problem is solved.
   Health-checking of ill-conditioned sub-problems is available via
   `options['solver.health_check.when']`.
5. Results are accessed through `trial.optimization` (raw solution, e.g. `V_final_si`),
   `trial.visualization.plot_dict` (post-processed/interpolated trajectories, power/performance
   metrics), and `trial.quality` (`awebox/quality.py`, `awebox/quality_funcs.py` — feasibility/
   sanity checks). `trial.plot(...)` renders standard plot sets (`'states'`, `'constraints'`,
   `'animation'`, etc.) via `awebox/viz/`.

`Sweep` (`awebox/sweep.py`, `awebox/sweep_funcs.py`) runs a grid of `Trial`s over varying option
values — used for scaling/tuning studies and comparison plots (`sweep.plot('comp_stats', ...)`).

`awebox/pmpc.py` / `awebox/sim.py` provide the periodic-MPC and closed-loop simulation machinery
built on top of a solved `Trial`.

### SAM (Stroboscopic Averaging Method)

For long-horizon multi-cycle trajectories, `options['nlp.SAM.use'] = True` switches on averaging:
only `d` "micro-integration" cycles are explicitly discretized while `N` full cycles are
represented via an averaged macro-integration, with regularization terms
(`options['nlp.SAM.Regularization.*']`) coupling them. This produces two parallel result sets: the
averaged/SAM solution (`trial.visualization.plot_dict_SAM`) and the reconstructed full trajectory
(`trial.visualization.plot_dict`). SAM-specific helpers live in
`awebox/tools/sam_functionalities.py` and `awebox/tools/struct_operations.py`
(e.g. `calculate_SAM_regions`); `Trial` automatically uses `viz.visualization.VisualizationSAM`
instead of `Visualization` when SAM is enabled.

### Directory map

- `awebox/mdl/` — system architecture, dynamics, aerodynamics (`aero/`), atmosphere, wind models.
- `awebox/ocp/` — optimal control problem formulation, discretization, constraints, variable
  structs/bounds.
- `awebox/opti/` — optimization driver, homotopy scheduling, initial-guess generation
  (`initialization_dir/`).
- `awebox/opts/` — options system: defaults, kite parameter data sets (`kite_data/`).
- `awebox/tools/` — cross-cutting utilities (struct/vector operations, caching, save/load,
  SAM helpers).
- `awebox/viz/` — plotting and animation.
- `examples/` — runnable scripts demonstrating typical usage, including
  `examples/paper_benchmarks/` and `examples/De_Schutter_2023_paper_benchmarks/` (reproduce
  published results) and `examples/paper_benchmarks/reference_options.py`
  (shared option presets used across examples, e.g. `set_reference_options`, `set_dual_kite_options`).
- `test/units`, `test/reg`, `test/int`, `test/trials` — see Common commands above.

## Background papers

Two papers in the repo root explain the theory behind this toolbox; read them for the "why" behind
the architecture above.

- **`De Schutter et al. - 2023 - AWEbox ...pdf`** (Energies 16, 1900) — the core AWEbox paper.
  Defines the tree-structured multi-aircraft system (tether-endpoint nodes; aircraft vs. layer
  nodes; parent map — see `awebox/mdl/architecture.py`), the non-minimal-coordinates 6DOF
  Lagrangian dynamics with Baumgarte-stabilized consistency conditions (index-1 DAE), and the
  periodic power-optimal OCP transcribed via Radau collocation. Its main contribution is two
  interior-point homotopy strategies used to build a feasible initial guess from a cheap analytic
  circular-flight guess: **CIPH** (classic fixed-step continuation) and **PIPH** (penalty-based —
  homotopy parameters become penalized decision variables so IPOPT's own line search picks the
  step size), plus **SIPH** (sweep interior-point homotopy) for warm-starting parametric sweeps
  (e.g., power curves over wind speed). Case studies show PIPH/CIPH converge far more reliably —
  and to the same local optimum — than solving directly from the raw circular guess. Maps to
  `awebox/opti/optimization.py` (homotopy stages), `awebox/opti/initialization_dir/` (circular
  initial guess), and `options['solver.homotopy.*']`.
- **`Harzer et al. - 2025 - ... Stroboscopic Averaging Method.pdf`** (IEEE Control Systems
  Letters 9, 703) — the SAM paper, and the focus of this branch. A pumping trajectory's reel-out
  phase consists of many near-identical crosswind loops; SAM discretizes only `d << N` full
  "micro-integration" cycles (high-accuracy Radau IRK) and represents the slow drift between loops
  with a macro-integration (Gauss–Legendre collocation) of an averaged/"stroboscopic" state
  trajectory, coupled via central-difference averaging conditions. Three regularization terms
  (on micro-cycle duration similarity, cycle-to-cycle state change, and macro-trajectory smoothness)
  keep the approximation accurate — see `options['nlp.SAM.Regularization.*']`. The full N-loop
  trajectory is then reconstructed from the `d` micro-cycles plus the averaged macro-trajectory, and
  validated via closed-loop MPC tracking simulation (the system is open-loop unstable, so a forward
  simulation alone can't verify it). Headline result: N=30-loop trajectories solved with SAM at
  d=5 match the accuracy/cost of solving the full N=5 problem directly, at a fraction of the NLP
  size. This is the theory behind `options['nlp.SAM.*']`, `awebox/tools/sam_functionalities.py`,
  `calculate_SAM_regions` in `awebox/tools/struct_operations.py`, and the distinction between
  `trial.visualization.plot_dict_SAM` (averaged/SAM solution) and `plot_dict` (reconstructed full
  trajectory) mentioned above.

## Notes specific to this fork/branch

This is a research fork (`develop_SAM_experiments`) used for AWE trajectory-optimization
experiments (LCSS/SAM paper work); expect example scripts in `examples/` to contain exploratory,
commented-out option overrides — treat them as scratch/experiment scripts rather than stable APIs.
`FAQ.md` documents common non-obvious workflows (warmstarting from a saved `.dict`, tuning solver
scaling via a `Sweep` when the power sub-problem fails to converge, running the health-checker, and
reading out solved quantities like `theta`, `zeta`, `avg_power`) — check it before re-deriving these.
