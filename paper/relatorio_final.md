# An Adaptive Density-Based Framework for Switching Between Classical and Reinforcement Learning Planners in Mobile Robot Navigation

**Yan Santos Leite** · Escola de Engenharia Elétrica, Mecânica e de Computação — UFG
**Orientador:** Prof. Aldo André Diaz Salazar · Instituto de Informática — UFG
**Projeto FAPEG PI08078-2024** · Relatório Final de IC PIBIC 2025–2026
**Repositório:** github.com/santtyan/adaptive-planner-switching

---

## Abstract

<!-- Write LAST (after results, Week 5). Target: 250 words EN + resumo PT. -->
<!-- Cover: problem → contribution → method → results → code availability. -->
*[To be written after results are finalized.]*

---

## 1. Introduction

### 1.1 Motivation

Autonomous mobile robot navigation requires path planners capable of handling highly heterogeneous environments: open corridors where global geometric planners excel and dense obstacle regions where reactive learned policies outperform rigid graph searches. Deploying a single fixed planner across all conditions leads to suboptimal behavior — classical planners become brittle under high obstacle density, while RL policies waste time in open spaces where analytical solutions are optimal.

The question of *which planner to use, when*, has received comparatively little attention as a first-class optimization objective. Most hybrid navigation systems use ad-hoc heuristics or fixed switching policies tuned by hand for a specific environment, limiting generalization and reproducibility.

### 1.2 Problem Statement

Given a mobile robot navigating from start $s$ to goal $g$ in environment $\mathcal{E}$ represented as an occupancy grid, we define the *local obstacle density* at pose $p$ with window $w$:

$$\rho(p, w) = \frac{|\{c \in W(p,w) : \text{occ}(c) \geq \theta\}|}{|W(p,w)|}$$

where $W(p,w)$ is the set of grid cells within a square window of side $w$ centered at $p$, $\text{occ}(c) \in \{0, \ldots, 100, -1\}$ is the cell occupancy value, and $\theta$ is an occupancy threshold (default 65, consistent with Nav2 defaults). Unknown cells ($\text{occ}(c) = -1$) are treated conservatively as occupied.

The **adaptive planner selection problem** is: learn or design a policy $\pi(\rho)$ that maps local density to a planner choice $\pi \in \{\pi_\text{classical}, \pi_\text{RL}\}$ such that navigation performance — measured by success rate, time-to-goal, and path quality — is maximized over a distribution of start-goal pairs and environment configurations.

### 1.3 Contributions

This work makes the following contributions:

1. **Density-based adaptive switching criterion** ($\rho$-criterion): a threshold-based policy with hysteresis that selects between a classical A\* planner (Nav2 `SmacPlanner2D`) and a learned RL policy (PPO or SAC) based solely on local obstacle density, without requiring global environment knowledge.

2. **Theoretical regret analysis**: we derive regret bounds showing the $\rho$-criterion achieves within 2.2% of an oracle selector that has perfect knowledge of environment difficulty, under a Monte Carlo calibrated simulation model.

3. **ROS2 implementation**: a complete, open-source ROS2 Humble implementation integrating Nav2, Stable-Baselines3, and Gazebo, enabling reproducible evaluation on TurtleBot3 Waffle in simulation.

4. **Empirical validation**: a systematic benchmark across 3 map types × 6 planner conditions × 30 trials, with statistical hypothesis testing (Wilcoxon signed-rank, Holm-Bonferroni correction).

### 1.4 Paper Organization

Section 2 surveys related work on classical planning, RL-based navigation, and hybrid approaches. Section 3 formalizes the problem. Section 4 details the adaptive framework architecture. Section 5 describes the ROS2 implementation. Section 6 presents experimental setup and hypotheses. Section 7 reports results. Section 8 discusses findings and limitations. Section 9 concludes.

---

## 2. Related Work

### 2.1 Classical Path Planning

Classical path planning algorithms provide completeness and optimality guarantees under deterministic environments. Dijkstra's algorithm [CITE Cormen 2022] finds single-source shortest paths in $O(V \log V + E)$ using a priority queue. A\* [CITE Hart 1968] augments Dijkstra with an admissible heuristic, directing search toward the goal and reducing average-case complexity. For all-pairs shortest paths, Floyd-Warshall [CITE Cormen 2022] runs in $O(V^3)$ and is practical for dense graphs; Johnson's algorithm [CITE Cormen 2022] achieves $O(VE \log V)$ via Bellman-Ford reweighting followed by per-source Dijkstra, handling sparse graphs with potential negative weights.

In the Nav2 ecosystem for ROS2, `SmacPlanner2D` implements A\* on a 2-D costmap grid and is the canonical global planner for differential-drive robots. `ThetaStar` extends A\* with line-of-sight shortcuts (any-angle A\*), producing smoother paths at the cost of slightly higher computation [CITE Daniel 2010].

Survey [CITE Huang 2022] covers 80+ planning algorithms across 10 categories, noting that classical planners dominate in structured, sparse environments but degrade in dynamic and highly cluttered settings.

### 2.2 Learning-Based Navigation

Deep reinforcement learning has enabled end-to-end navigation policies that generalize across environments. PPO [CITE Schulman 2017] is a policy gradient method with a clipped surrogate objective, widely used for continuous control due to its stability and simplicity. SAC [CITE Haarnoja 2018] adds maximum-entropy regularization, improving sample efficiency and robustness to hyperparameter choices. Both have been applied to mobile robot navigation with Gymnasium wrappers over Gazebo [CITE Raffin 2021].

[CITE SHA 2024] provides a recent survey of RL-based urban navigation, showing that RL planners outperform classical approaches in dense dynamic scenarios but struggle with long-horizon sparse-reward settings. DDPG [CITE Lillicrap 2015], an earlier continuous-control algorithm, has been largely superseded by SAC in the literature.

### 2.3 Hybrid and Switching Approaches

[CITE Zhao 2022] proposes combining RRT\* with a GAN-based predictor to identify narrow passages, pre-populating the sampling space. While effective, it requires a trained adversarial model and does not generalize to arbitrary environments.

[CITE He 2025] introduces a multi-objective switching framework that selects planners based on safety, comfort, and efficiency scores. The selection criterion relies on learned embeddings, making it difficult to interpret or certify.

The present work differs in that the switching criterion is a *single, interpretable scalar* ($\rho$) computed directly from the costmap with no learned components in the selector itself. This makes the selection policy certifiable, lightweight (< 1 ms), and easy to tune with a single threshold.

---

## 3. Problem Formulation

### 3.1 Notation

Let $\mathcal{E} = (\mathcal{G}, \mathcal{O})$ denote the environment, where $\mathcal{G} = [0, W] \times [0, H]$ is the navigable space and $\mathcal{O} \subset \mathcal{G}$ is the obstacle set. The robot state at time $t$ is $x_t = (p_t, \theta_t)$ where $p_t \in \mathcal{G}$ is its 2-D position and $\theta_t$ its heading. The costmap $\mathcal{C}: \mathcal{G} \to \{0, \ldots, 100, -1\}$ is an occupancy grid with resolution $r$ m/cell.

A **planner** $\pi: (\mathcal{C}, x, g) \to \mathcal{A}$ maps the current costmap, robot state, and goal $g$ to an action $a \in \mathcal{A}$. For the classical planner, $\mathcal{A}$ is a sequence of waypoints; for the RL planner, $\mathcal{A} = \mathbb{R}^2$ (linear and angular velocity commands).

### 3.2 Density as Context Variable

The local obstacle density at pose $p$ with window half-side $w$ is:

$$\rho(p, w, \mathcal{C}) = \frac{1}{|W|} \sum_{c \in W(p,w)} \mathbb{1}[\text{occ}(c) \geq \theta \;\lor\; \text{occ}(c) = -1]$$

where $W(p,w) = \{c \in \mathcal{C} : \|c - p\|_\infty \leq w/2r\}$ is the set of cells within the square window. Default parameters: $w = 2.0$ m, $r = 0.05$ m/cell, $\theta = 65$.

### 3.3 Selection Policy

The **adaptive selection policy** $\pi^*$ is:

$$\text{mode}_{t+1} = \begin{cases}
\pi_\text{classical} & \text{if } \rho_t \leq \tau - h \text{ and } \Delta t \geq d \\
\pi_\text{RL}        & \text{if } \rho_t \geq \tau + h \text{ and } \Delta t \geq d \\
\text{mode}_t        & \text{otherwise (hysteresis)}
\end{cases}$$

where $\tau = 0.30$ is the density threshold, $h = 0.05$ is the hysteresis half-band, and $d = 1.5$ s is the minimum dwell time before a transition is permitted. This prevents rapid oscillation when $\rho$ is near the threshold.

### 3.4 Optimality Criterion

Let $J(\pi, \xi)$ be the navigation cost (negative reward) of policy $\pi$ on trajectory $\xi$, and $\pi^{**}(\xi)$ be the oracle selector with full trajectory knowledge. We define the **regret** of $\pi^*$ as:

$$R(\pi^*, \mathcal{D}) = \mathbb{E}_{\xi \sim \mathcal{D}} \left[ J(\pi^*(\xi), \xi) - J(\pi^{**}(\xi), \xi) \right]$$

Phase 1 results (Section 7.1) show $R(\pi^*, \mathcal{D}) \leq 2.2\%$ of oracle cost under Monte Carlo calibrated simulation.

---

## 4. Adaptive Switching Framework

<!-- Sections 4.1–4.5: write in parallel with implementation (Weeks 2–3). -->

### 4.1 Local Density Estimator

*[Describe `density_estimator.py`: QoS TRANSIENT_LOCAL, window computation, unknown-as-occupied, 2 Hz publication.]*

### 4.2 Mode Transition State Machine

*[Describe FSM: IDLE → NAV2_ACTIVE → RL_ACTIVE → TRANSITIONING. Hysteresis 0.30±0.05, dwell 1.5s. Cancel goal handshake.]*

### 4.3 Classical Planner Backend

*[Nav2 SmacPlanner2D (A* 2D, diff-drive). Justify over RRT* (no official Nav2 Humble plugin). ThetaStar as future work.]*

### 4.4 RL Planner Backend

*[PPO and SAC via SB3. 27-dim observation (obs_utils.py). Action space v∈[-0.22,0.22], ω∈[-2.84,2.84].]*

### 4.5 Theoretical Analysis

*[Regret bounds from theoretical_analysis.py. 2.2% vs oracle, 1.7% optimality gap.]*

---

## 5. Implementation

<!-- Write alongside coding, Weeks 1–3. -->

### 5.1 ROS2 Architecture Overview

*[Diagram F1: nodes, topics, QoS annotations.]*

### 5.2 Density Estimator Node

*[QoS TRANSIENT_LOCAL, /global_costmap/costmap + /amcl_pose → /adaptive_planner/local_density at 2 Hz.]*

### 5.3 RL Controller Node

*[Load SB3 .zip, obs_utils.py canonical observation, /cmd_vel_rl publisher.]*

### 5.4 Adaptive Switcher Node

*[FSM, cancel_goal_async().result(), twist_mux lock, goal hand-off.]*

### 5.5 Twist Multiplexer

*[twist_mux config: cmd_vel_rl priority > cmd_vel_nav2.]*

---

## 6. Experimental Setup

### 6.1 Phase 1 — Analytical Validation

Phase 1 experiments used Monte Carlo calibrated simulation in a Python abstract environment (`validation_abstract/`) to validate the $\rho$ criterion and establish theoretical regret bounds. The planners in this phase are **calibrated mocks**: `MockRRTStarPlanner` produces stochastic straight-line paths, and `PPOPlannerMockScientific` uses density-conditioned success rates. These mocks are *not* OMPL or Stable-Baselines3 implementations; their purpose is to stress-test the switching criterion across a large statistical sample (N > 3,000 trials) rather than to measure absolute planner performance.

Phase 1 results should be interpreted as: *the $\rho = 0.30$ threshold is near-optimal under the calibrated model*. The definitive performance comparison between planners is provided by Phase 2.

### 6.2 Phase 2 — Empirical Validation in ROS2/Gazebo

*[TurtleBot3 Waffle, ROS2 Humble, Gazebo Classic. 3 maps: turtlebot3_world (open), turtlebot3_house (mixed), dense_custom.]*

### 6.3 Evaluation Metrics

| Metric | Symbol | Definition |
|---|---|---|
| Success rate | SR | trials reaching goal without collision / total trials |
| Time to goal | TTG | wall-clock seconds from start to goal arrival |
| Path length | PL | total odometry distance (m) |
| Min clearance | MC | minimum obstacle distance along path (m) |
| Mode switches | NS | number of Nav2↔RL transitions per trial |
| Normalized SR | SR* | SR / SR(Nav2 pure, same map+seed) |
| Planning latency | LAT | mean density→cmd_vel pipeline latency (ms) |

### 6.4 Hypotheses

**H1:** In maps with mean local density $\bar{\rho} \geq 0.30$, the Adaptive(Nav2+SAC) policy achieves higher success rate than Nav2 pure (Wilcoxon signed-rank, $p < 0.05$).

**H2:** In maps with mean local density $\bar{\rho} \leq 0.20$, the Adaptive policy achieves time-to-goal $\leq$ Nav2 pure (Wilcoxon signed-rank, $p < 0.05$).

**H3** (exploratory): The number of mode switches per trial correlates positively with the variance of $\rho$ along the trajectory (Spearman $r > 0.5$).

### 6.5 Statistical Tests

All pairwise comparisons use the Wilcoxon signed-rank test (non-parametric; distributions are expected non-normal due to timeout truncation). Multiple comparisons are corrected with the Holm-Bonferroni method. Effect sizes are reported as Cliff's delta ($\delta$): small $|\delta| < 0.147$, medium $< 0.33$, large $\geq 0.33$.

---

## 7. Results

### 7.1 Phase 1 — Analytical Validation

*[Summary from results_abstract/: 85.3% success rate, 2.2% regret vs oracle, 1.7% optimality gap.]*

*[Disclosure: results from calibrated mock planners — see Section 6.1.]*

### 7.2 Training Curves

*[F4: PPO vs SAC ep_rew_mean × steps. Wall-clock time reported.]*

### 7.3 Empirical Results in ROS2/Gazebo

*[Table: 6 conditions × 3 maps × metrics. Generated from results_ros2/master.csv.]*

### 7.4 Hypothesis Testing

*[H1, H2, H3 confirmed/rejected with p-values and Cliff's delta.]*

### 7.5 Mode Transition Analysis

*[F7: timeline ρ + active mode on mixed map.]*

---

## 8. Discussion

### 8.1 Why Adaptive Outperforms

*[Interpretation: RL gains in dense; classical efficiency in open.]*

### 8.2 Sim-to-Real Gap

*[Limitations of Gazebo-only evaluation. TB3 physical as future work.]*

### 8.3 Threats to Validity

*[Domain randomization, ρ threshold not validated in Gazebo (limitation), mock planners in Phase 1.]*

### 8.4 Generalization

*[Discussion: other robots, other simulators, dynamic obstacles (SIGAA gap).]*

---

## 9. Conclusion and Future Work

### 9.1 Summary

*[Main contributions + one-sentence results.]*

### 9.2 Future Work

- DDPG comparison (cut for deadline; SAC supersedes in literature)
- Nav2 C++ controller plugin (vs twist_mux bypass)
- Multi-agent CBS integration in ROS2 (extension chapter)
- Physical TurtleBot3 validation (sim-to-real)
- Dynamic obstacles (addresses SIGAA objective not fully covered)
- Ablation of $\rho$ threshold in Gazebo (fixed at 0.30; sensitivity study deferred)

---

## Acknowledgments

UFG, FAPEG (PIBIC bolsa PI08078-2024), Prof. Aldo André Diaz Salazar, CERISE, AKCIT/CEIA.

---

## References

<!-- IEEE style. Fill in BibTeX keys from paper/refs.bib. -->

[1] T. H. Cormen, C. E. Leiserson, R. L. Rivest, and C. Stein, *Introduction to Algorithms*, 4th ed. MIT Press, 2022.

[2] P. E. Hart, N. J. Nilsson, and B. Raphael, "A formal basis for the heuristic determination of minimum cost paths," *IEEE Trans. Syst. Sci. Cybern.*, vol. 4, no. 2, pp. 100–107, 1968.

[3] S. M. LaValle, *Planning Algorithms*. Cambridge Univ. Press, 2006.

[4] Y. Huang *et al.*, "Survey of motion planning for autonomous vehicles," *IEEE Trans. Intell. Transp. Syst.*, 2022.

[5] X. Zhao *et al.*, "GAN-assisted RRT* for robot path planning," *IEEE Trans. Intell. Transp. Syst.*, 2022.

[6] *[SHA 2024 — RL-based urban navigation survey]*.

[7] J. Schulman *et al.*, "Proximal policy optimization algorithms," arXiv:1707.06347, 2017.

[8] T. Haarnoja *et al.*, "Soft actor-critic: off-policy maximum entropy deep RL with a stochastic actor," *ICML*, 2018.

[9] T. P. Lillicrap *et al.*, "Continuous control with deep reinforcement learning," *ICLR*, 2016.

[10] S. Macenski *et al.*, "The Marathon 2: a navigation system," *IROS*, 2020.

[11] A. Raffin *et al.*, "Stable-Baselines3: reliable RL implementations," *J. Mach. Learn. Res.*, 2021.

[12] G. Sharon *et al.*, "Conflict-based search for optimal multi-agent pathfinding," *Artif. Intell.*, 2015.

[13] *[He 2025 — multi-optimization switching]*.

[14] G. Brockman *et al.*, "OpenAI Gym," arXiv:1606.01540, 2016.

[15] N. Koenig and A. Howard, "Design and use paradigms for Gazebo," *IROS*, 2004.

---

## Appendices

### A. Hyperparameters

| Parameter | PPO | SAC |
|---|---|---|
| Learning rate | 3e-4 | 3e-4 |
| Batch size | 64 | 256 |
| n_steps | 2048 | — |
| Buffer size | — | 1,000,000 |
| Entropy coef | 0.01 | auto |
| Total steps | 500,000 | 500,000 |
| Seed (train) | 42 | 42 |

FSM thresholds: $\tau = 0.30$, $h = 0.05$, $d = 1.5$ s.

### B. Observation Encoding

27-dimensional observation vector (defined in `ros2_ws/src/adaptive_planner_ros/adaptive_planner_ros/obs_utils.py`):

- Indices 0–23: 24 downsampled lidar rays. 360 raw rays bucketed into 24 bins (average pooling). Normalized: $o_i = \min(r_i, r_\text{max}) / r_\text{max}$, $r_\text{max} = 3.5$ m.
- Index 24: $r_\text{norm} = \min(\|p - g\|_2, d_\text{max}) / d_\text{max}$, $d_\text{max} = 6.0$ m.
- Index 25: $\sin(\alpha)$ where $\alpha$ is the bearing to goal in robot frame.
- Index 26: $\cos(\alpha)$.

### C. Reproducibility

```bash
# Build
cd ros2_ws && colcon build --symlink-install
source install/setup.bash

# Smoke test
python3 -m pytest src/adaptive_planner_ros/test/ -v

# Launch full demo
ros2 launch adaptive_planner_ros demo.launch.py \
  rl_model:=models/sac_42_500k.zip map:=dense_custom

# Benchmark (N=30)
python3 src/adaptive_planner_ros/benchmark/run_benchmark.py \
  --trials 30 --maps open dense mixed --conditions all
```

Package versions: `stable-baselines3==2.3.0`, `gymnasium==0.29.1`, `numpy==1.24.4`.
Seeds: training seed 42; benchmark seeds 1..30.

### D. Statistical Tables

*[Full per-condition per-map tables generated from results_ros2/master.csv.]*

### E. Ablation — PPO Abstract vs PPO Gazebo

*[Compare 6 legacy .zip models (trained in SimpleEnvironment, 5-dim obs) with new PPO trained in Gazebo (27-dim obs). Measures sim-to-sim transfer gap at zero additional training cost.]*
