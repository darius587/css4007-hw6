# HW6 Implementation — Reproduction Guide

**Course:** CSS-4007 Artificial Intelligence  
**Assignment:** HW6 — Grid World

## 1. Project Overview

This project has three parts:

- **Part A:** Handwritten Grid World calculations.
- **Part B:** FrozenLake simulations using Gymnasium.
- **Part C:** Value Iteration solver and custom Grid World simulations.

All Python programs were tested on my computer.

## 2. Tested Environment

- Operating System: Windows
- Python Version: 3.13.15
- Gymnasium Version: 1.2.2

Both `frozenlake.py` and `gridworld.py` ran successfully.

## 3. Installation and Setup

### Windows PowerShell

Create a virtual environment:

```powershell
py -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, use:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

Then activate the environment again.

## 4. Run Part B — FrozenLake

Run:

```powershell
python frozenlake.py
```

The program uses `FrozenLake-v1` with the standard 4×4 map.

It compares two policies:

- Random policy
- Fixed policy

Each policy is tested with `is_slippery=False` and `is_slippery=True`.

There are three batches with seeds 11, 22, and 33. Each batch contains 50 episodes.

**Total: 600 episodes and 12 batch summaries.**

Results are saved in:

- `outputs/part_b_episodes.csv`
- `outputs/part_b_summary.csv`
- `outputs/part_b_versions.txt`
- `outputs/part_b_goal_trace.txt`
- `outputs/part_b_hole_trace.txt`

The results table is automatically added to `SUBMISSION.md`.

## 5. Run Part C — Grid World Solver

Run:

```powershell
python gridworld.py
```

The program uses Value Iteration with:

- Discount factor: 0.9
- Convergence tolerance: 0.000001
- Maximum iterations: 10,000

The solver calculates two maps:

**Original map:**

```text
.M.G
.#IH
S...
```

**Extended map:**

```text
.M.G
.#IH
SD.W
```

The extended map includes sand (`D`, reward −2) and shallow water (`W`, reward −5).

### Solver Results

| Map | Iterations | Converged |
|---|---:|---|
| Original | 12 | Yes |
| Extended | 11 | Yes |

Each map is simulated for three batches of 50 episodes.

**Total: 300 episodes.**

Results are saved in:

- `outputs/part_c_episodes.csv`
- `outputs/part_c_summary.csv`
- `outputs/original_values_policy.txt`
- `outputs/extended_values_policy.txt`
- `outputs/part_c_solver_report.txt`
- `outputs/rule_checks.txt`

## 6. Part A — Handwritten Calculations

Part A was completed by hand before adding the Python programs.

The photos are saved in:

- [Page 1 — Definitions and V1](part_a/1_paper.jpg)
- [Page 2 — V1 Map and V2](part_a/2_paper.jpg)
- [Page 3 — Ice and Best Actions](part_a/3_paper.jpg)

The handwritten work includes V1, V2, Value Iteration calculations, and best-action arrows.

## 7. Git Commits

The homework is organized into separate commits:

1. **Part A:** Handwritten calculations and photos.
2. **Part B:** FrozenLake simulations and results.
3. **Part C:** Grid World solver and results.
4. **Final Report:** `SUBMISSION.md`, `README_RUN.md`, and `requirements.txt`.

Part A was committed first:

`d780b1e — Add papers in Part A`

The project is maintained in my public GitHub fork.

No pull request to the starter repository is required.

## 8. Reproduce All Results

To reproduce the experiments, activate the virtual environment and run:

```powershell
python frozenlake.py
python gridworld.py
```

The programs generate the episode data, batch summaries, value maps, policies, and rule checks in the `outputs/` directory.

For the full results and explanations, see [SUBMISSION.md](SUBMISSION.md).

## 9. AI Tools

OpenAI ChatGPT (GPT-6) was used to help understand Value Iteration, prepare and review Python code, check calculations, and organize the documentation.

The handwritten calculations were completed on paper, and the Python experiments were run locally.