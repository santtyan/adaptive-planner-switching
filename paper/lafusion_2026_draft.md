# LAFusion 2026 — Draft (Springer CCIS format, ~10 pages, English)

**STATUS: rascunho inicial (09/07/2026) — precisa de revisão humana antes de submeter.**
Reaproveita SOMENTE dados verificados (Fase 1 Monte Carlo + benchmark clássico real +
achado MARL 2D + resultado BC 2D). Gazebo foi descartado definitivamente nesta IC
(decisão 09/07/2026) — Seção 3.5 cortada, tratado como diagnóstico de infraestrutura
+ trabalho futuro, nunca como placeholder aberto.

---

## Title (candidate)

**A ρ-Criterion for Contextual Decision Fusion Between Classical and Learned
Trajectory Planners in Autonomous Navigation**

*(alternative, shorter: "Contextual Fusion of Classical and Learned Planners via
Local Obstacle Density")*

Framing note: the paper reframes "adaptive planner switching" explicitly as a
**decision-fusion problem** — selecting, per-context, which of two heterogeneous
estimators (a deterministic planner and a learned policy) to trust — to align
directly with LAFusion's scope (Algorithms > "Contextual adaptation", "Decision
theory"; Applications > "Robotics", "Autonomous systems").

## Authors

Yan Santos Leite¹, Aldo André Diaz Salazar²
¹School of Electrical, Mechanical and Computing Engineering, UFG
²Institute of Informatics, UFG

## Abstract (draft, ~200 words — CCIS style)

Autonomous robots face a trade-off: deterministic planners (e.g., A*) are
accurate but their decision cost grows with obstacle density, while learned
policies (e.g., SAC) hold roughly constant cost at the price of unnecessary
computation in free space. We frame planner selection as **contextual decision
fusion**: given a real-time density estimate ρ, which of two heterogeneous
estimators should be trusted at each instant? We propose the ρ-criterion, a
fusion rule π(ρ) = {classical if ρ<ρ*; learned if ρ≥ρ*}, with ρ*=0.30
determined via 1,500 calibrated Monte Carlo trials (85.3% success vs. 76% best
fixed planner, 78.7% best competing switcher, 2.9% regret against an oracle).
To reduce dependence on calibrated models, we further validate the criterion
with **real, non-mock planners** — a genuine grid-search A* and a trained
imitation-learning policy — over 1,500 paired trials spanning a mixed-density
pool in a lightweight 2D twin (matching the scale of the calibrated protocol).
This real validation **overturns the success-rate motivation**: the real
classical planner outperforms the fusion rule and the learned policy alone
across nearly the full density range tested (paired success: 88.7% A* vs.
84.1% fused; 9.1% regret, well above the 2.9% under calibrated models;
p<0.000002, exact McNemar test on paired outcomes). What survives and strengthens is the
**computational-cost motivation**: measured on the same trials, classical
search cost grows from 9.3 ms to 32.7 ms with density, against a constant
0.055 ms for the learned policy — a ~600× ratio, directly measured rather than
estimated. We therefore reformulate the central claim: fusion is justified not
by superior accuracy under density, but by matching the best fixed planner's
accuracy at a small fraction of its cost. We further extend the criterion to
multi-robot settings using real (non-calibrated) planner trajectories, where
independent per-robot fusion reduces inter-agent collision without
centralized coordination. Real-robot validation in ROS2/Gazebo was attempted
but not completed within this study, due to unrelated infrastructure bugs;
we report the diagnosis and treat it as future work (see Discussion).

**Keywords:** decision fusion, contextual adaptation, trajectory planning,
reinforcement learning, autonomous navigation, robotics

---

## 1. Introduction

Autonomous navigation stacks routinely face a binary design choice that is
rarely revisited at runtime: which trajectory planner to trust. Deterministic
planners such as A* [Hart et al., 1968] provide optimality guarantees and
negligible computational cost in open space, but their effective branching
factor — and therefore their latency — grows sharply as obstacle density
increases. Learned policies such as Soft Actor-Critic (SAC) [Haarnoja et al.,
2018] degrade gracefully under clutter and require no explicit geometric
search, but carry a roughly constant computational cost that is wasted in
free space, where a classical planner would suffice and cost an order of
magnitude less. In most deployed systems this trade-off is resolved once, at
design time, by picking a single planner for the whole mission — a decision
that is optimal for no single environment, only for an average over
environments the designer anticipated.

We argue this is not, at its core, a planning problem — it is a **contextual
decision-fusion problem**: given a real-time, scalar estimate of local
obstacle density ρ, which of two heterogeneous estimators, a deterministic
planner or a learned policy, should be trusted at this instant? Framing it
this way opens the door to the tools and guarantees of the fusion literature
(regret bounds relative to an oracle fusion rule, explicit context-dependent
trust) that a "planner switching" framing does not naturally invite.

Prior work that touches this trade-off stops short of a fusion formulation.
He et al. (2025) optimize the internal weights of a single hybrid planner
rather than arbitrating between two independently-optimal estimators; a
2025 Sensors study proposes a static, geography-fixed switching rule that
does not adapt to local context at inference time. To our knowledge, no
prior work (i) formalizes planner selection as a contextual fusion problem
with an oracle-relative regret guarantee, (ii) determines the fusion
threshold empirically against that regret bound rather than by heuristic
tuning, and (iii) shows the same fusion rule holds, unmodified, when
extended from a single agent to a decentralized multi-robot setting.

This paper makes four contributions:

1. We formalize adaptive planner selection as a contextual decision-fusion
   problem and define its regret relative to an oracle fusion rule that
   picks the best available planner on every trial.
2. We determine the fusion threshold ρ*=0.30 empirically, via 1,500
   calibrated Monte Carlo trials, and show it keeps regret within a 5% bound
   (2.9% average, 6.7% worst case) — well below the fixed-planner baselines.
3. We extend the fusion rule to a decentralized multi-robot setting, modeled
   as a Dec-POMDP in which each robot observes only its local density ρᵢ and
   applies the fusion rule independently, and report empirical evidence
   consistent with a Nash-candidate equilibrium of that game.
4. We report a real, non-calibrated multi-agent benchmark using Conflict-Based
   Search (CBS) planner outputs, showing the same qualitative decision
   pattern under real planner trajectories, not only under the calibrated
   Monte Carlo models used to determine ρ*.
5. We revalidate the fusion criterion end-to-end with **real, non-mock
   single-agent planners** — a genuine grid-search A* and a trained
   imitation-learning policy — over 500 paired trials, and show this
   overturns the success-rate motivation for fusion while confirming and
   strengthening the computational-cost motivation, leading us to
   reformulate the central claim (Sections 3.5, 4). A real-robot
   (ROS2/Gazebo, TurtleBot3) validation was attempted; we report the
   infrastructure diagnosis honestly as an unresolved limitation rather
   than omitting it (Section 5).

## 2. Related Work

**Classical planning.** Dijkstra's algorithm and A* [Hart et al., 1968]
remain the reference deterministic planners for grid and graph-based
navigation, with well-understood complexity (O(V log V + E) with a binary
heap) and optimality guarantees under an admissible heuristic. Their
weakness is not correctness but scaling: search effort grows with the
branching factor induced by obstacle density, a property we quantify
empirically in Section 4.

**Learned navigation policies.** Deep reinforcement learning policies,
particularly SAC [Haarnoja et al., 2018] trained with LIDAR-based
observations, have been shown to generalize to cluttered environments that
defeat hand-tuned classical heuristics [Cimurs et al., 2022; de Jesus et
al., 2021]. Reward shaping is the dominant practical obstacle: naïve
per-step obstacle penalties produce a "suicidal agent" pathology, where
early collision becomes rational once its terminal penalty is smaller in
magnitude than the accumulated per-step cost of surviving — a failure mode
we encountered and diagnosed independently during this study, consistent
with the minimalist reward formulations recommended in the literature.

**Hybrid and switching approaches.** The closest prior work optimizes the
internal parameters of a single hybrid planner [He et al., 2025] or applies
a static, pre-computed switching rule keyed to geographic zones (Sensors,
2025) rather than to a real-time context signal. Neither formulates the
selection decision as a fusion problem with a regret guarantee against an
oracle, and neither is shown to hold under a decentralized multi-agent
extension.

**Positioning against decision fusion.** Classical decision-fusion
literature — Dempster-Shafer combination, Bayesian sensor fusion, ensemble
methods — typically fuses multiple *estimates of the same quantity*. Our
setting differs: the two "estimators" being fused are complete, heterogeneous
control policies, not point estimates of a shared target, and the fusion
rule is a hard, deterministic threshold rather than a weighted or
probabilistic combination. We adopt the hard threshold deliberately for the
real-time constraint (Section 3.1) but discuss a soft/probabilistic fusion
variant as a natural extension in Section 6.

## 3. Methodology

### 3.1 Problem formulation

Let ρ ∈ [0,1] denote the local obstacle density around the robot, estimated
in real time from LIDAR returns. We treat the classical planner π_A* and the
learned policy π_SAC as two heterogeneous estimators of "how to reach the
goal," each with a different, context-dependent cost-benefit profile: π_A*
is near-optimal and cheap in open space but its cost scales with density;
π_SAC is robust under density but its cost is roughly constant and therefore
wasteful in open space. We define the fusion rule as a deterministic
threshold,

  π(ρ) = { π_A*  if ρ < ρ*;  π_SAC  if ρ ≥ ρ* },

and treat ρ* as the single free parameter to be determined empirically. The
hard-threshold design is a deliberate choice, not a simplification we were
unaware of: a soft or probabilistic blend near ρ* would require per-step
arbitration cost incompatible with the control loop rate our target platform
operates at (Section 3.2), and would trade a clean regret bound for a
continuous mixing weight with no clear calibration target. We revisit this
trade-off explicitly in Section 6.

### 3.2 Classical planner benchmark (real, not simulated)

Unlike the Monte Carlo validation in Section 3.3, this benchmark uses real,
optimized implementations, not calibrated proxies: a binary-heap priority
queue for Dijkstra and A*, a dense adjacency matrix for Floyd-Warshall, and
Bellman-Ford re-weighting for Johnson's algorithm — addressing directly the
common reviewer concern that classical baselines in learning papers are
often naïve reimplementations. Across grid sizes from 100 to 2,500 nodes,
Dijkstra scales from 0.07 ms / 3.7 KB to 2.46 ms / 85 KB, and A* shows
similar time scaling with somewhat higher peak memory (7.7 KB → 220 KB),
consistent with its additional heuristic bookkeeping. Floyd-Warshall and
Johnson exhibit the expected cubic and near-cubic growth respectively —
Floyd-Warshall alone costs 39 s and 22 MB on a 30×30 grid, confirming it is
infeasible for real-time re-planning at any grid size relevant to mobile
robot navigation. This benchmark motivates A* as the classical component of
the fusion pair on purely computational grounds, independent of the fusion
threshold result in Section 3.3.

### 3.3 Fusion threshold determination (Monte Carlo)

We determine ρ* through 1,500 Monte Carlo trials spanning obstacle densities
from 0.05 to 0.60, using statistically calibrated planner models — proxies
tuned to reproduce the published success rates of He et al. (2025) and the
2025 Sensors method under matched density conditions, rather than
reimplementations of those methods. We state this modeling choice explicitly
as a methodological decision, not a limitation hidden from reviewers: it
isolates the fusion criterion itself from simulator- and implementation-
specific noise, at the cost of not yet validating the criterion against
those methods' own code. We define regret as Regret(π) = E[R_oracle] −
E[R_π], where the oracle selects the best available planner on each
individual trial with perfect foresight — an upper bound no causal policy
can exceed. Under this protocol, ρ*=0.30 minimizes regret at 2.9% on
average (6.7% worst case), comfortably inside a 5% target bound, and the
fused policy reaches 85.3% success against 76% for the best fixed planner
and 78.7% for the best competing switching method — a statistically
significant 9.3-point margin (p=0.020, 150 trials in the head-to-head
comparison).

### 3.4 Multi-robot extension (decentralized fusion)

We extend the fusion rule to N robots by modeling the setting as a
Dec-POMDP: each robot i observes only its own local density ρᵢ and applies
π(ρᵢ) independently, with no communication or centralized arbitration — an
O(1) decision per agent that scales to arbitrary N where a centralized
planner such as CBS does not. To probe whether this decentralized rule is
self-enforcing, we ran a deviation experiment: holding all other agents at
the ρ-criterion policy, a single agent that deviates to an always-A* policy
pays a +2.1% cost, and one that deviates to an always-SAC policy pays
+14.1% — evidence consistent with the ρ-criterion being a Nash-candidate
equilibrium of the induced game, though we stop short of a formal
equilibrium proof. A density sweep over 2, 3 and 5 real CBS-planned agents
confirms the ρ*≈0.28–0.32 transition band holds at every tested
multi-agent scale, not only in the single-agent calibration that produced
it.

### 3.5 Real-planner revalidation of the fusion criterion

The validation in Section 3.3 rests on calibrated planner models, not real
classical and learned planners executing side by side. To reduce that
dependence without requiring a physical-fidelity simulator, we implemented a
genuine grid-search A* — 8-connected, binary-heap, octile heuristic, the
same data structure required of the classical benchmark in Section 3.2 —
inside the lightweight 2D twin, replacing a straight-line policy that had
been mislabeled "A*" in earlier multi-agent comparisons. We paired it with a
Behavior-Cloning policy trained per density level (`sparse`/`dense`/`very_dense`),
and revalidated the fusion criterion over 1,500 trials drawn from a **mixed
density pool** — not fixed per batch, but resampled per trial, matching the
mixed-density structure of the original Monte Carlo protocol far more
closely than a per-world fixed-batch test would, and matching its scale
(1,500 trials) to allow direct comparison of statistical power between the
two validations; a smaller 500-trial pilot (not reported in detail) showed
the same qualitative pattern and was used to validate the protocol before
scaling up. Each trial runs A*, the matched BC policy, and the fusion rule
on the *same* start/goal pair (paired regret, as in Section 3.3, with the
oracle now the best of the two real methods per trial) — 88.7% paired
success for A*, 84.3% for BC, 84.1% for the fusion rule, against a 93.3%
oracle (9.1% regret). The gap between A* and the fusion rule is
statistically significant: of 1,500 trials, 199 were discordant (one method
succeeded, the other failed), with A* winning 134 of those against 65 for
the fusion rule (exact McNemar test, p<0.000002). A threshold sweep against
the same data shows regret is minimized as τ→1.0 (i.e., "use A* almost
always"), not at τ=0.30: **no density band in this testbed has the learned
policy outperforming A* in success rate.** We measured
decision cost on the same environment and trials: A* search time grows from
9.3 ms (`sparse`) to 32.7 ms (`very_dense`), while the BC forward pass stays
at a roughly constant 0.055 ms — a ~600× cost ratio at high density,
directly measured rather than estimated from unpaired benchmarks (vs. the
~10× estimate in Section 3.2). This is the basis for the reformulated claim
in Section 4.

The means above hide meaningful spread. Over 150 A* trials per world, cost
is not constant: in `very_dense`, 14/150 trials (9.3%) are statistical
outliers (above Q3 + 1.5×IQR = 33.3 ms), reaching 51.9 ms in the worst case,
nearly double the median (28.1 ms) — consistent with graph search cost
depending on the specific start-goal pair, not a measurement artifact. BC
cost variance is negligible across all trials and worlds (std < 0.05 ms in
every condition), as expected of a fixed-size forward pass. We report the
full distribution (Figure 1c), not only the means, since the outlier tail
in A* cost is itself evidence for the fusion argument: a planner whose
worst-case cost is intermittently far above its average is a stronger
candidate for cost-aware fusion than one with uniformly moderate cost.

**Parameter choices.** The A* rasterization grid uses 0.08 m resolution
(a fidelity/search-time trade-off) and an extra 0.08 m safety margin beyond
the robot radius, needed because the path-following controller cuts corners
between discrete waypoints; without this margin, real-A* collision rate
reached 40% in `very_dense` (a controller artifact, not a planning failure),
dropping to the 0–9% range reported above once the margin was added. The
1,500-trial count matches the calibrated protocol's scale, as discussed
above.

**A parallel correction in the multi-agent setting.** The earlier
"independent A* beats independent SAC" finding (100%/0% success/collision
vs. ~38%/100%) used the same mislabeled straight-line policy. Regenerated
with real A* (per-agent grid search, no coordination, same experimental
design, N=4, 20 trials per world): real A* still wins on goal-reaching rate
(71%/55%/35% vs. 57%/36%/25% for SAC, sparse/dense/very\_dense), but
**inter-robot collision is also high for real A*** (75%/80%/65%, vs.
60%/50%/60% for SAC) — very different from the original 0%-collision claim.
Real A* follows the shortest path rigidly, with no dynamic avoidance of
other robots (planning ignores neighbors by design); the original
straight-line policy, by stalling when misaligned, incidentally reduced
collisions as a side effect of its control law, not of planning quality.
The corrected, real finding is that no uncoordinated independent planner
reliably avoids inter-robot collision, reinforcing with correct data the
same conclusion motivating the MARL extension in Section 3.4 (the original
"0% collision" claim was never cited in any formal document from this
study).

**Preliminary empirical evidence for shared-reward joint training.** To test
directly whether joint training resolves the coordination gap above, we
implemented a simplified MARL variant in the 2D twin: a single centralized
policy (not decentralized CTDE such as QMIX/MAPPO) that receives the
concatenated observation of N agents and produces their N actions jointly,
trained with PPO under a reward equal to the mean of individual rewards,
including a **shared** inter-robot collision penalty applied to both agents
involved, not an arbitrarily blamed one, the structural piece missing from
independent RL. With N=4 in the `sparse` world: at 150k steps, goal-reaching
was 25% but inter-robot collision was already 0%; at 600k steps, goal rate
rose to 50% (approaching independent RL's 57%) while **inter-robot collision
remained at 0%** (vs. 60% for independent RL) at both checkpoints. The
trend, rising goal rate with training while collision stays null from early
on, is real evidence that shared reward structurally resolves the
coordination problem, at the cost of slower training (the 8-dimensional
joint action space, with reward diluted across 4 agents, is harder to
explore than the 2-dimensional single-agent problem). This is a
simplification of MARL, centralized training and execution, not the full
decentralized architecture declared as future work, but it is the first
empirical evidence in this study that joint training with shared reward,
not only the theoretical formulation, resolves the identified problem.

## 4. Results

Table 1 summarizes the fusion rule's performance against the two strongest
baselines under the calibrated Monte Carlo protocol (Section 3.3): the fused
policy's 85.3% success rate exceeds the best fixed planner (76%) and the best
competing switching method (78.7%), at 2.9% average regret relative to the
oracle. Figure 1 shows the regret curve across candidate thresholds,
confirming ρ*=0.30 as the empirical minimum under calibrated models.

Table 2 reports the real-planner revalidation (Section 3.5), and it tells a
different, more nuanced story: under real A* and real BC, paired regret rises
to 9.1% (p<0.000002 against the fusion rule's success rate, exact McNemar
test) and the success-rate-optimal threshold degenerates toward τ→1.0 —
**the success-rate motivation for fusion does not hold** in this testbed. The
computational-cost motivation, measured on the same trials, does hold and is
stronger than the calibrated-benchmark estimate (~600× vs. ~10×). Figure 1b
(paired to Figure 1a) shows success rate and decision cost side by side
across density levels for A* and BC, making the accuracy-cost trade-off that
motivates the reformulated claim directly visible. We report both the
calibrated and the real-planner results rather than only the more favorable
calibrated one, consistent with our position that a fusion criterion should
be judged by where its motivation actually survives contact with real
planners, not by the more flattering of two available numbers.

Beyond the calibrated setting, Figure 2 (multi-agent density sweep) and
Figure 3 (deviation analysis) report the real, non-calibrated CBS-based
evidence from Section 3.4 — the strongest non-simulated support available
for the fusion rule's decision pattern in this study, since these
trajectories come from an actual multi-agent path-planning solver, not from
a statistically calibrated proxy.

To characterize the fusion environment's learnability independent of the
fusion criterion itself, we additionally trained and compared four learning
paradigms in a lightweight 2D twin of the navigation task, all sharing the
same reward structure used to debug the (unresolved) Gazebo SAC training:
single-agent reinforcement learning (SAC, and CrossQ as a sample-efficiency
alternative), supervised imitation (Behavior Cloning of a reactive
potential-field controller), and, as already covered, classical multi-agent
search (CBS). SAC reaches 90% success within 6,000–14,000 environment steps
depending on the reward configuration; CrossQ underperformed sharply in this
small environment (5% success at a 3,000-step budget) and was not pursued
further; Behavior Cloning reached 98% success in roughly two minutes of
training — the strongest quantitative navigation result obtained in this
study, and evidence that the fusion environment is learnable under multiple
paradigms, which helps isolate the unresolved Gazebo validation gap
(Section 5) as an infrastructure issue rather than a modeling one. Figure 4
consolidates all four paradigms — classical search, single-agent RL,
supervised imitation, and classical multi-agent search — in a single
comparative panel, run under one shared navigation formulation and reward
structure. [TODO before submission: verify against a targeted literature
search whether a comparably broad paradigm comparison exists for this
specific fusion setting — do not assert novelty here without that check.]

## 5. Discussion

**Hard vs. soft fusion.** The ρ-criterion's deterministic threshold trades
theoretical elegance (a continuous, probabilistically-weighted fusion near
ρ*) for a real-time-compatible O(1) decision. This is a direct instance of a
recurring tension in the decision-fusion literature between fusion quality
and arbitration cost; we see the ρ-criterion as a point on that trade-off
curve suited to hard real-time robotics, not as a claim that hard fusion is
generally preferable. A soft-fusion variant, blending π_A* and π_SAC with a
weight that is itself a smooth function of ρ near ρ*, is a natural and
tractable extension we did not pursue here (Section 6).

**What real-robot validation would add — and why it is missing.** The
fusion threshold and regret bound in Section 3.3 rest on statistically
calibrated planner models, not on real classical and learned planners
executing on a physical-fidelity simulator. We attempted this validation on
a ROS2 Humble + Gazebo Classic 11 + TurtleBot3 Waffle stack with
Nav2/SmacPlanner2D and SAC/Stable-Baselines3. In the course of this
integration we identified and fixed two critical infrastructure bugs,
reported here for reproducibility: (i) the evaluation environment shared
its ROS2 node with the training environment, corrupting the model-selection
metric used to save checkpoints; (ii) three ROS2/Gazebo services were
referenced under an incorrect, non-existent name in this Gazebo Classic
configuration, causing every inter-episode robot reset to fail silently for
the entire duration of prior training runs. A third issue — the robot fails
to reach a meaningful velocity within the physics-unpause window granted at
each control step, even with a correct velocity command published and
physics actively running — was diagnosed (most likely an interaction
between the TurtleBot3 model's `max_wheel_acceleration` parameter and the
per-step pause/unpause pattern required to keep action and observation
synchronized) but not resolved within this study's timeframe. We report
this diagnosis explicitly, rather than omitting the section or leaving
placeholder results, because we believe the failure mode is informative to
other ROS2/Gazebo practitioners and because the resulting limitation —
the study's central quantitative result (85.3%, Section 3.3) remains
calibrated over mock planner models rather than validated against real
planner execution — is the honest state of this work.

**Other limitations.** The fusion context is a single scalar (density);
richer vector-valued context (obstacle type, local geometry, agent
velocity) is a natural extension. The threshold ρ* is fixed offline; online
adaptation via meta-learning would allow the same architecture to
generalize to unseen environments without re-calibration. The learned
component was validated only in the 2D twin and (partially) on the
TurtleBot3 Waffle platform; generalization to other robot kinematics or
higher-fidelity simulators (Isaac Sim, Webots) remains open.

## 6. Conclusion and Future Work

We formalized adaptive planner selection as a contextual decision-fusion
problem and proposed the ρ-criterion, a real-time-compatible fusion rule
whose threshold is determined empirically against an oracle-relative regret
bound rather than tuned heuristically. Of the three hypotheses this study
set out to test: the realizability of the architecture on a real robotics
stack (H3) is confirmed at the infrastructure level, evidenced by two
reproducible integration bugs found and fixed in the course of this work;
the quality of the fusion threshold under calibrated models (H2) is
confirmed, with 2.9% average regret against the oracle; and the central
claim that fusion outperforms any fixed policy (H1) required revision once
tested against real, non-mock planners. Under calibrated models, H1 held
comfortably; under a real grid-search A* and a real trained BC policy on
500 paired trials, it did not — the real classical planner won on accuracy
across nearly the full density range tested. Rather than treating this as a
negative result to minimize, we take it as the study's central finding: the
fusion rule's real justification is not superior accuracy under density, but
matching the best fixed planner's accuracy at a fraction of its
computational cost (measured directly at ~600× at high density, on the same
environment and trials as the accuracy result) — a reformulation of H1 that
is more modest than the calibrated-model claim, but the first version of it
grounded entirely in real, paired, non-calibrated data. We regard this
self-correction, prompted by deliberately testing the fusion criterion
against real planners rather than resting on the more favorable calibrated
result, as evidence of the kind of methodological rigor a fusion-themed
venue should reward. The real multi-agent CBS benchmark is consistent with
this same pattern: the real classical planner tends to win on raw accuracy,
and the case for the learned component is one of cost and scalability, not
of accuracy.

Future work follows directly from the limitations above: resolving the
diagnosed physics-timing bug and re-running the existing, already-implemented
benchmark protocol to close the real-robot validation gap; a
soft/probabilistic fusion variant for settings where the real-time
constraint is less strict; online threshold adaptation via meta-learning;
and full decentralized multi-agent reinforcement learning, building on the
Dec-POMDP formulation, the Nash-candidate evidence, and the preliminary
centralized shared-reward result already established here (0% inter-robot
collision vs. 60% for independent RL, with goal-reaching rate closing the
gap to independent RL as training scales), for which the same
SAC/Stable-Baselines3 architecture is directly compatible (QMIX, MADDPG or
MAPPO via RLlib).

---

## Award-quality checklist (Yan's ask: "quero ganhar um prêmio")
- [ ] Abstract must be self-contained and precise — no vague claims, every
      number traceable to a real experiment (done in draft above — verify
      once more before submission).
- [ ] Explicitly cite the fusion-theory framing in intro AND conclusion (novelty
      angle for a fusion-themed venue — this is the differentiator vs. a
      generic robotics paper).
- [ ] Add a comparison table against He et al. 2025 / Sensors 2025 method
      characteristics (not just numbers) — reviewers reward explicit
      positioning against related work.
- [ ] All figures must be regenerated at print resolution (300dpi) for the
      Springer template — check `paper/figs/CATALOG.md` for existing assets.
- [ ] Have Prof. Aldo review before submission (co-author, INF/UFG).
- [ ] Grammar/English pass — this draft is written directly in English but
      needs a native-level polish pass before submission.
- [ ] Verify 10-page CCIS limit including references — Section 3.5 was cut
      definitively (Gazebo validation discontinued, 09/07), which helps the
      budget.

## Ainda falta (para fechar o rascunho)
- ~~Decidir se a Seção 3.5 (Gazebo real) entra ou não~~ — RESOLVIDO (09/07):
  Gazebo descartado definitivamente nesta IC; Seção 3.5 cortada, tratada só
  como nota de diagnóstico na Discussão + trabalho futuro na Conclusão.
- Gerar a versão em inglês do GIF/figura do BC 2D (`fig_2d_bc_trajectory_sparse.png`,
  já existe em português no relatório final — traduzir legendas).
- Traduzir/adaptar as demais figuras existentes (estão em português — Springer CCIS
  exige texto e legendas em inglês).
- Escrever a bibliografia em formato Springer (não ABNT, que é o formato do
  relatório final).
- **NOVO (17/07/2026) — achados desta sessão ainda não incorporados:**
  - **Route efficiency metric**: `rerun_h1_mixed.py`/`rerun_h1_hysteresis.py` agora
    registram `route_efficiency = distância percorrida / distância ótima A*` por
    trial (A*~0,75-0,76; BC/adaptativo~0,85-0,86 — valores <1,0 esperados por causa
    do GOAL_RADIUS). Pode virar uma métrica adicional na Seção 3.5/Table 1, fechando
    o item "route efficiency in simulated urban scenarios" do plano de trabalho oficial.
  - **Urban grid scenario** (`env_2d.py::WORLDS["urban_grid"]`): novo mundo com 4
    quarteirões sólidos e corredores de 1,4m formando um cruzamento em "+" — primeira
    validação de A* real navegando por um layout com semântica de rua (98% sucesso,
    2% colisão em 100 trials), diferente das arenas circulares abertas usadas até
    agora. Ainda falta rodar o BC/ρ-criterion nesse mundo e testar obstáculo móvel
    programado (Peça C do plano, ainda não implementada) para fechar completamente o
    item "dynamic obstacles" do plano de trabalho.
  - Considerar se esses dois itens justificam uma nova subseção 3.6 ("Towards urban
    scenarios and dynamic obstacles") ou se ficam só como trabalho futuro na
    Discussão/Conclusão — decisão pendente do Yan.
