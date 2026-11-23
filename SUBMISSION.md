# HW6 submission

## AI-use declaration

I used OpenAI ChatGPT (GPT-6) to understand value iteration, help prepare and review Python code, check calculations, design rule checks, and organize this report.

I wrote Part A by hand and ran the Python programs on my computer to collect the simulation results. I reviewed the outputs before submission.

## Setup and reproduction

I used **Python 3.13.15** and **Gymnasium 1.2.2** on Windows.

Installation and run commands (PowerShell):

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python frozenlake.py
python gridworld.py
```

See [README_RUN.md](README_RUN.md) for more details and [the installed dependency versions](outputs/part_b_versions.txt). The Gymnasium version was checked with `python -m pip show gymnasium`.

## Part A — Paper calculations

### Handwritten calculations

I completed Part A by hand. My calculations are shown in three photos:

- [Page 1 — Definitions and V1](part_a/1_paper.jpg)
- [Page 2 — V1 map and V2 calculations](part_a/2_paper.jpg)
- [Page 3 — Ice calculations, best actions and explanation](part_a/3_paper.jpg)

**MDP:** There are 11 valid states (9 nonterminal, 2 terminal) and four actions: up, right, down, left. Moves are deterministic except on ice (0.8 intended direction, 0.1 each perpendicular direction). Invalid moves leave the agent in place and pay the reward of the tile where it stays. I maximize expected discounted rewards with gamma = 0.9, starting from V0 = 0.

**V1:**

```text
 -1  -1  10  G(0)
 -1   #  -1  H(0)
 -1  -1  -1   -1
```

**V2 calculations (U, R, D, L):**

- (1,2): U = -3.9, R = 8, D = -3.9, L = -1.9; best = R (→).
- (1,3): U = 8, R = 10, D = -1.9, L = -3.9; best = R (→).
- (2,3), ice:
  - U = 0.8(8) + 0.1(-1.9) + 0.1(-10) = 5.21
  - R = 0.8(-10) + 0.1(8) + 0.1(-1.9) = -7.39
  - D = 0.8(-1.9) + 0.1(-1.9) + 0.1(-10) = -2.71
  - L = 0.8(-1.9) + 0.1(8) + 0.1(-1.9) = -0.91
  - Best = U (↑).

No actions tie for best at these three cells. Two rounds do not prove an infinite-horizon optimal policy because the effects of more distant rewards have not fully reached every state.

## Part B — FrozenLake

### Fixed policy, written before the experiments

I designed a fixed policy from the 4×4 map. Gymnasium action codes: 0 = left, 1 = down, 2 = right, 3 = up. In the nonslippery map, the route right/right/down/down/down/right avoids the holes and reaches the goal. Every nonterminal state has a defined action:

| State (row,col) | Index | Action |
|---|---:|---|
| (0,0) | 0 | R (2) |
| (0,1) | 1 | R (2) |
| (0,2) | 2 | D (1) |
| (0,3) | 3 | L (0) |
| (1,0) | 4 | U (3) |
| (1,2) | 6 | D (1) |
| (2,0) | 8 | R (2) |
| (2,1) | 9 | R (2) |
| (2,2) | 10 | D (1) |
| (3,1) | 13 | R (2) |
| (3,2) | 14 | R (2) |

The fixed policy is unchanged across experiments. The random policy chooses each of the four actions with probability 0.25. For each condition I used three batches of 50 episodes (batch seeds 11, 22, 33). Episode `i` uses seed `1000 * batch_seed + i` for `i = 0..49`; the random action generator also uses the episode seed. Each episode has a 100-step limit.

### Results

I ran **600 episodes** in Gymnasium 1.2.2, covering four conditions and three batches of 50 episodes each. The detailed data, summaries and dependency versions are saved here:

- [All Part B episodes](outputs/part_b_episodes.csv)
- [Part B batch summaries](outputs/part_b_summary.csv)
- [Python and library versions](outputs/part_b_versions.txt)

<!-- PART_B_RESULTS_START -->
| Slippery? | Policy | Batch seed | Goals /50 | Holes /50 | Time limits /50 | Success % | Mean steps |
|---|---|---:|---:|---:|---:|---:|---:|
| False | random | 11 | 0 | 50 | 0 | 0.0% | 8.14 |
| False | random | 22 | 1 | 49 | 0 | 2.0% | 7.56 |
| False | random | 33 | 0 | 50 | 0 | 0.0% | 6.88 |
| False | fixed | 11 | 50 | 0 | 0 | 100.0% | 6.00 |
| False | fixed | 22 | 50 | 0 | 0 | 100.0% | 6.00 |
| False | fixed | 33 | 50 | 0 | 0 | 100.0% | 6.00 |
| True | random | 11 | 0 | 50 | 0 | 0.0% | 8.06 |
| True | random | 22 | 0 | 50 | 0 | 0.0% | 7.60 |
| True | random | 33 | 0 | 50 | 0 | 0.0% | 7.98 |
| True | fixed | 11 | 0 | 50 | 0 | 0.0% | 4.84 |
| True | fixed | 22 | 1 | 49 | 0 | 2.0% | 5.04 |
| True | fixed | 33 | 0 | 50 | 0 | 0.0% | 5.60 |
<!-- PART_B_RESULTS_END -->

### Example traces

- [Successful episode (goal)](outputs/part_b_goal_trace.txt)
- [Failed episode (hole)](outputs/part_b_hole_trace.txt)

These two files are written by `frozenlake.py` when the corresponding outcomes are observed, as they were in my run. The code also uses text rendering for episode traces.

### Explanation

Without slipping, the fixed policy reached the goal in all 150 episodes, while the random policy reached it once. With slipping, the fixed policy reached the goal only once, and the random policy never reached it in these batches.

Slipping makes movement uncertain and can send the agent into a hole. The three batches were broadly consistent, but there were small differences in success counts and mean steps. One successful episode is not enough to estimate a reliable success rate.

These runs are simulations, not training. The fixed policy does not change, and the random policy does not learn from results.

## Part C — Custom solver

### Compare code with paper

The program reproduces the handwritten Part A V1 map. The requested V2 values are **(1,2) = 8.0**, **(1,3) = 10.0**, and **(2,3) = 5.21**. No discrepancies were found. An invalid slip on ice must count as staying on ice with the ice tile reward, not as a lost probability branch.

### Values, policies and convergence

The solver uses synchronous Value Iteration with gamma = 0.9, tolerance `1e-6`, and at most 10,000 iterations. Ties are resolved in **U, R, D, L** order. Its value and policy maps are saved here:

- [Original value and policy map](outputs/original_values_policy.txt)
- [Extended value and policy map](outputs/extended_values_policy.txt)
- [Convergence report](outputs/part_c_solver_report.txt)

**Original map:** 12 iterations, last full-update maximum change **2.16529310482e-7**; converged.

**Extended map:** 11 iterations, last full-update maximum change **2.40588123201e-7**; converged.

Original final values:

```text
 4.200   8.000  10.000   0.000 G
 2.780       #   5.824   0.000 H
 1.536   2.818   4.242   2.818
```

Original policy:

```text
→  →  →  G
↑  #  ↑  H
→  →  ↑  ←
```

Extended final values:

```text
 4.200   8.000  10.000   0.000 G
 2.780       #   5.824   0.000 H
 1.502   2.818   4.242   2.818
```

Extended policy:

```text
→  →  →  G
↑  #  ↑  H
↑  →  ↑  ←
```

### Simulation results

I simulated each computed policy in three batches of 50 episodes, with seeds 11, 22, and 33 and a 100-step limit. The return uses `r0 + 0.9*r1 + 0.9^2*r2 + ...`.

- [All Part C episodes](outputs/part_c_episodes.csv)
- [Part C batch summaries](outputs/part_c_summary.csv)

| Map | Batch seed | Goals /50 | Holes /50 | Time limits /50 | Mean steps | Mean discounted return |
|---|---:|---:|---:|---:|---:|---:|
| original | 11 | 42 | 8 | 0 | 4.98 | 0.8913 |
| original | 22 | 46 | 4 | 0 | 5.04 | 1.9148 |
| original | 33 | 48 | 2 | 0 | 5.04 | 2.4921 |
| extended | 11 | 50 | 0 | 0 | 5.00 | 1.5020 |
| extended | 22 | 50 | 0 | 0 | 5.00 | 1.5020 |
| extended | 33 | 50 | 0 | 0 | 5.00 | 1.5020 |

### Explain the results

On the original map, the start state's value is about **1.536** and its best action is **right**. This path approaches ice, where choosing **up** has an 0.8 chance of moving upward and an 0.1 chance of slipping right into the hole. The solver accounts for these probabilities.

The extended map adds sand `D` with entry reward **-2** and shallow water `W` with entry reward **-5**. Sand makes the rightward route from the start less attractive. The best first action from `S` changes to **up**, and its value decreases from about **1.536** to **1.502**. The other displayed values and best actions stay the same. The policy avoids water on its route from the start.

The original-map simulations sometimes end in a hole because of slips on ice; their returns vary between batches. The extended-map policy follows a deterministic five-step route in these runs and avoids the ice risk, so all 150 episodes reach the goal.

**Infinite-horizon values versus simulations:** `V(S)` is the expected discounted return from the transition model under the optimal policy. Each batch mean is an estimate from only 50 episodes, with a 100-step cap, so it can differ because of sampling variation and the finite horizon. The cap did not affect the reported episodes, since they ended earlier.

### Rule checks

See [rule checks](outputs/rule_checks.txt). They test boundaries and walls (staying put with the resulting tile's reward), ice probabilities summing to one (0.8 + 0.1 + 0.1), rewards for entering mud (**-3**), sand (**-2**), water (**-5**), goal (**+10**) and hole (**-10**), and no additional rewards after reaching a terminal tile.
